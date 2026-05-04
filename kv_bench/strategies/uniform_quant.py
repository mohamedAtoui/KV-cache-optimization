"""Uniform quantization strategy: quantize all KV entries to INT8 or INT4.

Simple baseline that applies symmetric quantization to all KV cache
entries uniformly, without importance-based tiering. Uses KIVI-style
per-channel quant for keys and per-token quant for values.
"""

import logging

import torch
import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy
from streaming_attention.stratigraphic import ZONE_INT8, ZONE_INT4

logger = logging.getLogger(__name__)


class UniformQuantStrategy(KVCacheStrategy):
    """Uniform quantization of all KV cache entries.

    Applies actual quant/dequant noise via forward hooks on k_proj/v_proj
    so that PPL reflects real quantization error.
    """

    def __init__(self, bits: int = 8, group_size: int = 128):
        """
        Args:
            bits: Quantization bitwidth (4 or 8).
            group_size: Number of elements per quantization group.
        """
        self.bits = bits
        self.group_size = group_size
        self.name = f"INT{bits}-all"
        self._model_config = None

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        self._model_config = model_config
        return model

    def needs_attention_weights(self) -> bool:
        # Need Pass 1 so we enter the two-pass branch in the runner,
        # which triggers get_keep_mask/get_zone_masks and installs quant hooks.
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        # We don't need the weights, but must accept the call.
        pass

    def get_keep_mask(self, seq_len, device):
        # Keep all tokens — no eviction, just quant.
        return torch.ones(seq_len, dtype=torch.bool, device=device)

    def get_zone_masks(self, seq_len, device):
        """Return uniform zone masks: all positions get the same quant level."""
        if self._model_config is None:
            return None
        num_layers = self._model_config.num_hidden_layers
        num_kv_heads = getattr(
            self._model_config, "num_key_value_heads",
            self._model_config.num_attention_heads,
        )
        zone_id = ZONE_INT8 if self.bits == 8 else ZONE_INT4
        zone_masks = {}
        for l in range(num_layers):
            zone_masks[l] = torch.full(
                (num_kv_heads, seq_len), zone_id,
                dtype=torch.long, device=device,
            )
        return zone_masks

    def reset(self):
        pass

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        bytes_per_element = self.bits / 8
        # K + V entries
        kv_bytes = num_layers * num_kv_heads * seq_len * head_dim * bytes_per_element * 2
        # Scale factors: one per group per row
        num_groups = (head_dim + self.group_size - 1) // self.group_size
        scale_bytes = num_layers * num_kv_heads * seq_len * num_groups * 4 * 2  # fp32 scales, K+V
        return int(kv_bytes + scale_bytes)

    def teardown(self, model: nn.Module) -> nn.Module:
        self._model_config = None
        return model
