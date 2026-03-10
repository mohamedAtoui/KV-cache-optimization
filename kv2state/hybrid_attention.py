"""Hybrid attention: monkey-patch Llama attention to use recurrent state for streaming heads.

For each layer, streaming heads (identified by HeadClassification) use a fixed-size
recurrent state matrix instead of growing KV cache. Retrieval heads keep standard KV cache.
"""

import logging
import math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv2state.head_classifier import HeadClassification
from kv2state.state_attention import DecayedLinearState, StateCache

logger = logging.getLogger(__name__)


@dataclass
class KV2StateConfig:
    """Configuration for KV2State hybrid attention."""
    decay_init: float = 0.99
    learnable_decay: bool = False  # False for zero-shot, True for calibration
    chunk_size: int = 64  # Chunk size for parallel prefill of state heads
    skip_feature_map: bool = False  # True for calibration: pass raw Q/K, only apply decay


def patch_model_for_kv2state(
    model: nn.Module,
    head_classification: HeadClassification,
    config: Optional[KV2StateConfig] = None,
) -> tuple[nn.Module, StateCache]:
    """Monkey-patch a Llama model to use KV2State hybrid attention.

    Replaces the forward method of each LlamaAttention layer to:
    - Use standard softmax attention + KV cache for retrieval heads
    - Use decayed linear state for streaming heads

    Args:
        model: HuggingFace LlamaForCausalLM (or compatible).
        head_classification: Binary mask from head_classifier.
        config: KV2State configuration. Uses defaults if None.

    Returns:
        model: The patched model (modified in-place).
        state_cache: StateCache object to manage recurrent states.
    """
    if config is None:
        config = KV2StateConfig()

    state_cache = StateCache()
    mask = head_classification.mask  # [num_layers, num_kv_heads]
    num_layers = mask.shape[0]
    num_kv_heads = mask.shape[1]

    model_config = model.config
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_q_heads = model_config.num_attention_heads
    num_kv_heads_model = getattr(model_config, "num_key_value_heads", num_q_heads)
    q_per_kv = num_q_heads // num_kv_heads_model

    assert num_kv_heads == num_kv_heads_model, (
        f"Head classification has {num_kv_heads} KV heads but model has {num_kv_heads_model}"
    )

    # Create per-head state modules for streaming heads
    state_modules = {}
    for layer_idx in range(num_layers):
        for head_idx in range(num_kv_heads):
            if not mask[layer_idx, head_idx]:  # Streaming head
                state_mod = DecayedLinearState(
                    head_dim=head_dim,
                    decay_init=config.decay_init,
                    learnable_decay=config.learnable_decay,
                )
                # Move to same device/dtype as model
                layer = model.model.layers[layer_idx]
                device = next(layer.parameters()).device
                dtype = next(layer.parameters()).dtype
                state_mod = state_mod.to(device=device, dtype=dtype)
                state_modules[(layer_idx, head_idx)] = state_mod

    logger.info(f"Created {len(state_modules)} state modules for streaming heads")

    # Patch each attention layer
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        layer_mask = mask[layer_idx]  # [num_kv_heads], True=retrieval
        layer_state_modules = {
            h: state_modules[(layer_idx, h)]
            for h in range(num_kv_heads)
            if (layer_idx, h) in state_modules
        }

        if not layer_state_modules:
            # All heads are retrieval — no patching needed
            continue

        _patch_attention_layer(
            attn=attn,
            layer_idx=layer_idx,
            layer_mask=layer_mask,
            state_modules=layer_state_modules,
            state_cache=state_cache,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            q_per_kv=q_per_kv,
            chunk_size=config.chunk_size,
            skip_feature_map=config.skip_feature_map,
        )

    # Store state modules on model so they're included in parameters()
    model._kv2state_modules = nn.ModuleDict({
        f"layer{l}_head{h}": mod for (l, h), mod in state_modules.items()
    })
    model._kv2state_cache = state_cache

    streaming_heads = head_classification.get_streaming_heads()
    logger.info(
        f"Patched {len(streaming_heads)} streaming heads across {num_layers} layers. "
        f"State memory per token: {len(streaming_heads) * head_dim * head_dim * 2 / 1024:.1f} KB "
        f"(vs KV cache: O(seq_len) per head)"
    )

    return model, state_cache


def _patch_attention_layer(
    attn: nn.Module,
    layer_idx: int,
    layer_mask: torch.Tensor,
    state_modules: dict[int, DecayedLinearState],
    state_cache: StateCache,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    q_per_kv: int,
    chunk_size: int,
    skip_feature_map: bool = False,
):
    """Replace a single LlamaAttention layer's forward with hybrid KV/state attention."""

    attn._original_forward = attn.forward
    original_forward = attn.forward

    def hybrid_forward(
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

        # Project Q, K, V using original weights
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Reshape: [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        q = q.view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = attn.rotary_emb(v, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Split heads into retrieval and streaming groups
        retrieval_kv_indices = []
        streaming_kv_indices = []
        for h in range(num_kv_heads):
            if layer_mask[h]:
                retrieval_kv_indices.append(h)
            else:
                streaming_kv_indices.append(h)

        # Initialize output tensor
        output = torch.zeros(bsz, num_q_heads, q_len, head_dim, dtype=q.dtype, device=q.device)

        # --- Handle retrieval heads (standard softmax attention) ---
        if retrieval_kv_indices:
            ret_k = k[:, retrieval_kv_indices]  # [B, n_ret, T, D]
            ret_v = v[:, retrieval_kv_indices]

            # Expand KV for GQA: repeat each KV head for its query group
            ret_k_expanded = ret_k.repeat_interleave(q_per_kv, dim=1)
            ret_v_expanded = ret_v.repeat_interleave(q_per_kv, dim=1)

            # Gather corresponding query heads
            ret_q_indices = []
            for kv_idx in retrieval_kv_indices:
                for qi in range(q_per_kv):
                    ret_q_indices.append(kv_idx * q_per_kv + qi)
            ret_q = q[:, ret_q_indices]  # [B, n_ret*q_per_kv, T_q, D]

            # Scaled dot-product attention
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.matmul(ret_q, ret_k_expanded.transpose(-2, -1)) * scale

            # Use transformers' pre-computed causal mask if available,
            # otherwise build our own
            kv_len = ret_k.shape[2]
            if attention_mask is not None and attention_mask.dim() == 4:
                # attention_mask from transformers already includes causal mask
                # Shape: [B, 1, T_q, T_kv] — broadcast over heads
                mask_slice = attention_mask
                if mask_slice.shape[-1] > kv_len:
                    mask_slice = mask_slice[..., :kv_len]
                if mask_slice.shape[-2] > q_len:
                    mask_slice = mask_slice[..., :q_len, :]
                attn_weights = attn_weights + mask_slice
            elif q_len > 1:
                # Fallback: manual causal mask for prefill
                causal = torch.triu(
                    torch.full((q_len, kv_len), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=kv_len - q_len + 1,
                )
                attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            ret_output = torch.matmul(attn_weights, ret_v_expanded)

            # Place into output tensor
            for i, q_idx in enumerate(ret_q_indices):
                output[:, q_idx] = ret_output[:, i]

        # --- Handle streaming heads (recurrent state) ---
        if streaming_kv_indices:
            _map = (lambda x: x) if skip_feature_map else _feature_map

            for kv_idx in streaming_kv_indices:
                state_mod = state_modules[kv_idx]
                # Apply feature map to K for non-negative linear attention
                # (skipped during calibration where raw Q/K are used with decay only)
                head_k = _map(k[:, kv_idx])  # [B, T, D]
                head_v = v[:, kv_idx]  # [B, T, D]

                # Get corresponding query heads for this KV head
                q_start = kv_idx * q_per_kv

                if q_len == 1:
                    # Decoding: single token, use recurrent update
                    state, z = state_cache.get(layer_idx, kv_idx)
                    if state is None:
                        state, z = state_mod.initial_state(bsz, q.dtype, q.device)

                    head_k_sq = head_k.squeeze(1)  # [B, D]
                    head_v_sq = head_v.squeeze(1)  # [B, D]

                    # Compute output for each query head in the group
                    for qi in range(q_per_kv):
                        head_q = _map(q[:, q_start + qi, 0:1]).squeeze(1)  # [B, D]
                        out, new_state, new_z = state_mod.recurrent_forward(
                            head_q, head_k_sq, head_v_sq, state, z
                        )
                        output[:, q_start + qi, 0] = out

                    # Update state (shared across query heads in group)
                    state_cache.set(layer_idx, kv_idx, new_state, new_z)

                else:
                    # Prefill: compute state once, then apply each Q head
                    # Use parallel_forward with first Q head to get state trajectory
                    first_q = _map(q[:, q_start])  # [B, T, D]
                    first_out, final_state, final_z = state_mod.parallel_forward(
                        first_q, head_k, head_v, chunk_size=chunk_size,
                    )
                    output[:, q_start] = first_out

                    # For remaining Q heads in group, reuse the same K,V state
                    for qi in range(1, q_per_kv):
                        head_q = _map(q[:, q_start + qi])  # [B, T, D]
                        qi_out, _, _ = state_mod.parallel_forward(
                            head_q, head_k, head_v, chunk_size=chunk_size,
                        )
                        output[:, q_start + qi] = qi_out

                    state_cache.set(layer_idx, kv_idx, final_state, final_z)

        # Transpose back and project: [B, num_heads, T, D] -> [B, T, hidden_size]
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
        output = attn.o_proj(output)

        # Return format expected by LlamaDecoderLayer
        # Newer transformers (4.46+): (output, attn_weights) — 2 values
        if output_attentions and retrieval_kv_indices:
            return output, attn_weights
        return output, None

    # Replace forward method
    attn.forward = hybrid_forward
    attn._kv2state_patched = True


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys."""
    # Handle different cos/sin shapes from different transformers versions
    if cos.dim() == 2:
        # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # [B, seq_len, head_dim] -> [B, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    # dim == 4: already [B, 1, seq_len, head_dim], no change needed

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map for non-negative linear attention (Katharopoulos et al., 2020).

    Maps arbitrary signed inputs to strictly positive values, ensuring
    Q·K products are non-negative for stable state accumulation.
    """
    return F.elu(x) + 1.0
