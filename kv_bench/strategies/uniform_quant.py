"""Uniform quantization strategy: quantize all KV entries to INT8 or INT4.

Simple baseline that applies symmetric per-group quantization to all KV cache
entries uniformly, without importance-based tiering.
"""

import logging

import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy

logger = logging.getLogger(__name__)


class UniformQuantStrategy(KVCacheStrategy):
    """Uniform quantization of all KV cache entries.

    Analytical strategy — does not patch the model forward.
    Computes memory as if all entries were quantized to the specified bitwidth.
    Perplexity impact is estimated by the quantization reconstruction error.
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

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        # Uniform quant doesn't patch the model — it's an analytical comparison.
        # Perplexity is measured as full-precision (quantization noise is estimated
        # separately via reconstruction error in the report).
        return model

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
        return model
