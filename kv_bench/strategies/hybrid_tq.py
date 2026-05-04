"""Hybrid strategy: StreamingAttention for streaming heads + TurboQuant for retrieval heads.

Combines: streaming heads → fixed-size recurrent state (StreamingAttention),
retrieval heads → TurboQuant quantized KV cache (instead of INT8/INT4).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy
from streaming_attention.stratigraphic import ZONE_FP16, ZONE_TQ3, ZONE_TQ4
from streaming_attention.turboquant import TurboQuantConfig, init_turboquant_state

logger = logging.getLogger(__name__)


class HybridTQStrategy(KVCacheStrategy):
    """StreamingAttention (streaming heads) + TurboQuant (retrieval heads)."""

    def __init__(
        self,
        pattern_dir: Optional[str] = None,
        threshold: float = 0.5,
        decay_init: float = 0.99,
        bits_stage1: int = 2,
        qjl: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        self.pattern_dir = pattern_dir
        self.threshold = threshold
        self.decay_init = decay_init
        self.bits_stage1 = bits_stage1
        self.qjl = qjl
        self.checkpoint_path = checkpoint_path
        effective = bits_stage1 + (1 if qjl else 0)
        self.name = f"Hybrid (Streaming + TQ{effective})"
        self._state_cache = None
        self._head_classification = None
        self._model_config = None

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        from streaming_attention.head_classifier import load_duo_attention_patterns
        from streaming_attention.hybrid_attention import (
            patch_model_for_streaming_attention, StreamingAttentionConfig,
        )

        if self.pattern_dir is None:
            raise ValueError("HybridTQStrategy requires pattern_dir")

        self._model_config = model_config

        self._head_classification = load_duo_attention_patterns(
            self.pattern_dir, threshold=self.threshold,
        )

        config = StreamingAttentionConfig(decay_init=self.decay_init)
        model, self._state_cache = patch_model_for_streaming_attention(
            model, self._head_classification, config,
        )

        if self.checkpoint_path:
            if hasattr(model, "_streaming_attention_modules"):
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                model._streaming_attention_modules.load_state_dict(state_dict, strict=False)

        # Create TurboQuant state for retrieval head quantization
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        tq_config = TurboQuantConfig(
            head_dim=head_dim,
            bits_stage1=self.bits_stage1,
            qjl_enabled=self.qjl,
        )
        model._tq_state = init_turboquant_state(
            tq_config, device_config.device, device_config.dtype,
        )

        return model

    def reset(self):
        if self._state_cache is not None:
            self._state_cache.reset()

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        # No importance scoring needed — we quantize all retrieval heads uniformly
        pass

    def get_keep_mask(self, seq_len, device):
        # Keep all tokens (no eviction)
        return torch.ones(seq_len, dtype=torch.bool, device=device)

    def get_zone_masks(self, seq_len, device):
        """Assign TQ zones to retrieval heads, FP16 to streaming heads.

        Streaming heads are already handled by the patched forward (recurrent state),
        so FP16 zone means the quant hooks are a no-op for those heads.
        """
        if self._model_config is None or self._head_classification is None:
            return None

        num_layers = self._model_config.num_hidden_layers
        num_kv_heads = getattr(
            self._model_config, "num_key_value_heads",
            self._model_config.num_attention_heads,
        )
        zone_id = ZONE_TQ3 if (self.bits_stage1 == 2 and self.qjl) else ZONE_TQ4

        # head_classification.mask: [num_layers, num_kv_heads]
        # True = retrieval head, False = streaming head
        retrieval_mask = self._head_classification.mask  # [L, H]

        zone_masks = {}
        for l in range(num_layers):
            zones = torch.full(
                (num_kv_heads, seq_len), ZONE_FP16,
                dtype=torch.long, device=device,
            )
            for h in range(num_kv_heads):
                if retrieval_mask[l, h]:
                    zones[h, :] = zone_id
            zone_masks[l] = zones

        return zone_masks

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        if self._head_classification is None:
            n_streaming = num_layers * num_kv_heads // 2
            n_retrieval = num_layers * num_kv_heads - n_streaming
        else:
            mask = self._head_classification.mask
            n_retrieval = mask.sum().item()
            n_streaming = mask.numel() - n_retrieval

        # Streaming heads: fixed state (D*D + D) per head, bf16
        streaming_bytes = int(n_streaming * (head_dim * head_dim + head_dim) * 2)

        # Retrieval heads: TurboQuant compressed KV cache
        effective_bits = self.bits_stage1 + (1 if self.qjl else 0)
        # K + V per retrieval head
        retrieval_kv_bytes = n_retrieval * seq_len * head_dim * (effective_bits / 8) * 2
        # Per-vector scale (1 float32 per vector, K+V), always stored
        scale_bytes = n_retrieval * seq_len * 4 * 2
        # Residual norms (1 float32 per vector, K+V) if QJL
        norm_bytes = 0
        if self.qjl:
            norm_bytes = n_retrieval * seq_len * 4 * 2

        return int(streaming_bytes + retrieval_kv_bytes + scale_bytes + norm_bytes)

    def teardown(self, model: nn.Module) -> nn.Module:
        # Restore streaming attention forwards
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "_original_forward"):
                attn.forward = attn._original_forward
                del attn._original_forward
                if hasattr(attn, "_streaming_attention_patched"):
                    del attn._streaming_attention_patched

        if hasattr(model, "_streaming_attention_modules"):
            del model._streaming_attention_modules
        if hasattr(model, "_streaming_attention_cache"):
            del model._streaming_attention_cache
        if hasattr(model, "_tq_state"):
            del model._tq_state

        self._state_cache = None
        self._model_config = None
        return model
