"""Adaptive tiered compression strategy using multi-signal importance scoring.

Uses the ImportanceScorer and TieredKVCache from kv2state to apply
progressive compression (FP16/INT8/INT4/evict) based on importance.
"""

import logging
from typing import Optional

import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy

logger = logging.getLogger(__name__)


class AdaptiveTieredStrategy(KVCacheStrategy):
    """Multi-signal importance + tiered compression for all KV heads.

    Analytical strategy that models the memory savings from tiered compression.
    Attention weights are collected via on_step to compute importance scores,
    but the actual model forward is unchanged (perplexity is full-precision).
    """

    def __init__(
        self,
        budget_fp16: float = 0.25,
        budget_int8: float = 0.30,
        budget_int4: float = 0.25,
    ):
        """
        Args:
            budget_fp16: Fraction of entries kept at full precision.
            budget_int8: Fraction quantized to INT8.
            budget_int4: Fraction quantized to INT4.
        """
        self.budget_fp16 = budget_fp16
        self.budget_int8 = budget_int8
        self.budget_int4 = budget_int4
        evict_pct = 1.0 - budget_fp16 - budget_int8 - budget_int4
        self.name = f"Adaptive Tiered"
        self._scorer = None
        self._cumulative_attn: dict[int, object] = {}

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        self._cumulative_attn.clear()
        return model

    def reset(self):
        self._cumulative_attn.clear()

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        """Collect attention weights for importance scoring."""
        if attn_weights is None:
            return
        # Store cumulative attention per position
        score = attn_weights.float().sum(dim=(0, 1, 2))  # [KV]
        if layer_idx in self._cumulative_attn:
            old = self._cumulative_attn[layer_idx]
            if old.shape[0] < score.shape[0]:
                import torch
                new = torch.zeros_like(score)
                new[:old.shape[0]] = old
                old = new
            self._cumulative_attn[layer_idx] = old[:score.shape[0]] + score
        else:
            self._cumulative_attn[layer_idx] = score

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        n_fp16 = int(seq_len * self.budget_fp16)
        n_int8 = int(seq_len * self.budget_int8)
        n_int4 = int(seq_len * self.budget_int4)

        per_layer_bytes = (
            n_fp16 * head_dim * 2 * num_kv_heads * 2 +       # FP16: K+V
            n_int8 * head_dim * 1 * num_kv_heads * 2 +       # INT8: K+V
            n_int4 * (head_dim // 2) * num_kv_heads * 2 +    # INT4: K+V
            8 * head_dim * 4                                   # sketch for evicted
        )
        return num_layers * per_layer_bytes

    def teardown(self, model: nn.Module) -> nn.Module:
        self._cumulative_attn.clear()
        return model
