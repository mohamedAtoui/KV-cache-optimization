"""Simulated INT8/INT4 quantization for KV-cache benchmark.

Applies round-trip quant/dequant noise to K and V projections via
forward hooks, so PPL reflects actual compression error.

Keys are quantized **after** RoPE (matching real KV caches) by replacing
self_attn.forward.  Values (no RoPE) still use a v_proj forward hook.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from streaming_attention.hybrid_attention import _apply_rotary_pos_emb
from streaming_attention.stratigraphic import (
    ZONE_FP16, ZONE_INT8, ZONE_INT4, ZONE_EVICT, ZONE_TQ3, ZONE_TQ4,
)
from streaming_attention.turboquant import (
    simulate_turboquant, turboquant_quantize, turboquant_attention_scores,
    TurboQuantConfig, TurboQuantState,
)


def simulate_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-token INT8 quant → dequant round-trip."""
    scale = tensor.abs().amax(dim=-1, keepdim=True) / 127
    scale = scale.clamp(min=1e-8)
    return ((tensor / scale).round().clamp(-128, 127)) * scale


def simulate_int4(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-token INT4 quant → dequant round-trip."""
    scale = tensor.abs().amax(dim=-1, keepdim=True) / 7
    scale = scale.clamp(min=1e-8)
    return ((tensor / scale).round().clamp(-8, 7)) * scale


def simulate_int8_per_channel(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-channel INT8 quant (KIVI-style, for keys).

    One scale per head_dim channel shared across tokens (dim=-2).
    """
    scale = tensor.abs().amax(dim=-2, keepdim=True) / 127
    scale = scale.clamp(min=1e-8)
    return ((tensor / scale).round().clamp(-128, 127)) * scale


def simulate_int4_per_channel(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-channel INT4 quant (KIVI-style, for keys).

    One scale per head_dim channel shared across tokens (dim=-2).
    """
    scale = tensor.abs().amax(dim=-2, keepdim=True) / 7
    scale = scale.clamp(min=1e-8)
    return ((tensor / scale).round().clamp(-8, 7)) * scale


def make_quant_hook(zone_mask: torch.Tensor, num_kv_heads: int, head_dim: int,
                    per_channel: bool = False,
                    tq_state: TurboQuantState | None = None,
                    tq3_config: TurboQuantConfig | None = None,
                    tq4_config: TurboQuantConfig | None = None):
    """Create a forward hook that applies simulated quantization.

    Args:
        zone_mask: [num_kv_heads, seq_len] zone assignments for one layer.
        num_kv_heads: Number of KV heads.
        head_dim: Dimension per head.
        per_channel: If True, use per-channel quant (KIVI-style, for keys).
                     If False, use per-token quant (for values).
        tq_state: TurboQuant state (rotation/QJL matrices + codebooks).
        tq3_config: TurboQuant config for 3-bit zones.
        tq4_config: TurboQuant config for 4-bit zones.

    Returns:
        Hook function for register_forward_hook on k_proj or v_proj.
    """
    int8_fn = simulate_int8_per_channel if per_channel else simulate_int8
    int4_fn = simulate_int4_per_channel if per_channel else simulate_int4

    def hook(module, input, output):
        # output shape: [B, T, num_kv_heads * head_dim]
        B, T, _ = output.shape
        out = output.view(B, T, num_kv_heads, head_dim)

        zones = zone_mask[:, :T]  # [num_kv_heads, T]
        int8_mask = (zones == ZONE_INT8)  # [num_kv_heads, T]
        int4_mask = (zones == ZONE_INT4)

        for h in range(num_kv_heads):
            if int8_mask[h].any():
                out[:, int8_mask[h], h, :] = int8_fn(out[:, int8_mask[h], h, :])
            if int4_mask[h].any():
                out[:, int4_mask[h], h, :] = int4_fn(out[:, int4_mask[h], h, :])

        # TurboQuant zones
        if tq_state is not None:
            tq3_mask = (zones == ZONE_TQ3)
            tq4_mask = (zones == ZONE_TQ4)
            for h in range(num_kv_heads):
                if tq3_mask[h].any() and tq3_config is not None:
                    out[:, tq3_mask[h], h, :] = simulate_turboquant(
                        out[:, tq3_mask[h], h, :], tq_state, tq3_config,
                    )
                if tq4_mask[h].any() and tq4_config is not None:
                    out[:, tq4_mask[h], h, :] = simulate_turboquant(
                        out[:, tq4_mask[h], h, :], tq_state, tq4_config,
                    )

        return out.view(B, T, -1)

    return hook


def make_post_rope_key_quant_forward(
    attn, zone_mask: torch.Tensor, num_kv_heads: int, num_q_heads: int, head_dim: int,
    tq_state: TurboQuantState | None = None,
    tq3_config: TurboQuantConfig | None = None,
    tq4_config: TurboQuantConfig | None = None,
):
    """Return a replacement ``self_attn.forward`` that quantises keys **after** RoPE.

    The wrapper re-implements the standard Llama attention forward path
    (project → reshape → RoPE → SDPA → o_proj) with a quantisation step
    inserted between RoPE and the attention computation.

    ``zone_mask`` shape: ``[num_kv_heads, seq_len]`` with per-position zone
    assignments (FP16 / INT8 / INT4 / EVICT).
    """
    q_per_kv = num_q_heads // num_kv_heads
    original_forward = attn._original_forward_for_quant

    def _quant_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, hidden_size = hidden_states.shape

        # --- Project Q, K, V ---
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Reshape: [B, T, num_heads * head_dim] -> [B, num_heads, T, head_dim]
        q = q.view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # --- Apply RoPE ---
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = attn.rotary_emb(v, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # --- Quantise keys (post-RoPE) per zone_mask ---
        zones = zone_mask[:, :q_len]  # [num_kv_heads, T]
        int8_mask = (zones == ZONE_INT8)
        int4_mask = (zones == ZONE_INT4)

        for h in range(num_kv_heads):
            if int8_mask[h].any():
                k[:, h, int8_mask[h], :] = simulate_int8_per_channel(
                    k[:, h, int8_mask[h], :]
                )
            if int4_mask[h].any():
                k[:, h, int4_mask[h], :] = simulate_int4_per_channel(
                    k[:, h, int4_mask[h], :]
                )

        # --- TurboQuant key quantisation (post-RoPE) ---
        if tq_state is not None:
            tq3_mask = (zones == ZONE_TQ3)
            tq4_mask = (zones == ZONE_TQ4)
            for h in range(num_kv_heads):
                if tq3_mask[h].any() and tq3_config is not None:
                    k[:, h, tq3_mask[h], :] = simulate_turboquant(
                        k[:, h, tq3_mask[h], :], tq_state, tq3_config,
                    )
                if tq4_mask[h].any() and tq4_config is not None:
                    k[:, h, tq4_mask[h], :] = simulate_turboquant(
                        k[:, h, tq4_mask[h], :], tq_state, tq4_config,
                    )

        # --- GQA expansion ---
        k = k.repeat_interleave(q_per_kv, dim=1)  # [B, num_q_heads, T, D]
        v = v.repeat_interleave(q_per_kv, dim=1)

        # --- Scaled dot-product attention ---
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask (causal + eviction, provided by runner)
        if attention_mask is not None and attention_mask.dim() == 4:
            mask_slice = attention_mask
            if mask_slice.shape[-1] > q_len:
                mask_slice = mask_slice[..., :q_len]
            if mask_slice.shape[-2] > q_len:
                mask_slice = mask_slice[..., :q_len, :]
            attn_weights = attn_weights + mask_slice
        elif q_len > 1:
            causal = torch.triu(
                torch.full((q_len, q_len), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)

        # [B, num_q_heads, T, D] -> [B, T, hidden_size]
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
        output = attn.o_proj(output)

        if output_attentions:
            return output, attn_weights
        return output, None

    return _quant_forward


def install_quant_hooks(model, zone_masks, model_config):
    """Install quantization on k_proj (post-RoPE) and v_proj for all layers with zones.

    Keys are quantised **after** RoPE by replacing ``self_attn.forward``.
    Values are quantised via a ``v_proj`` forward hook (no RoPE on values).

    Args:
        model: HuggingFace causal LM.
        zone_masks: {layer_idx: Tensor[num_kv_heads, seq_len]} zone assignments.
        model_config: Model config for head counts.

    Returns:
        (hooks, patched_attns): hook handles and attention modules whose
        forward was replaced.  Caller must ``.remove()`` hooks and restore
        ``attn.forward`` from ``attn._original_forward_for_quant``.
    """
    num_kv_heads = getattr(
        model_config, "num_key_value_heads", model_config.num_attention_heads
    )
    num_q_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // num_q_heads

    # Check for TurboQuant state stored on model by strategy.setup()
    tq_state = getattr(model, "_tq_state", None)
    tq3_config = None
    tq4_config = None
    if tq_state is not None:
        tq3_config = TurboQuantConfig(head_dim=head_dim, bits_stage1=2, qjl_enabled=True)
        tq4_config = TurboQuantConfig(head_dim=head_dim, bits_stage1=3, qjl_enabled=True)

    hooks = []
    patched_attns = []
    for layer_idx, zone_mask in zone_masks.items():
        attn = model.model.layers[layer_idx].self_attn

        # V hook stays on v_proj (no RoPE on values)
        v_hook = make_quant_hook(
            zone_mask, num_kv_heads, head_dim, per_channel=False,
            tq_state=tq_state, tq3_config=tq3_config, tq4_config=tq4_config,
        )
        hooks.append(attn.v_proj.register_forward_hook(v_hook))

        # K: replace self_attn.forward for post-RoPE quant
        attn._original_forward_for_quant = attn.forward
        attn.forward = make_post_rope_key_quant_forward(
            attn, zone_mask, num_kv_heads, num_q_heads, head_dim,
            tq_state=tq_state, tq3_config=tq3_config, tq4_config=tq4_config,
        )
        patched_attns.append(attn)

    return hooks, patched_attns
