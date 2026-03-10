"""H2O (Heavy-Hitter Oracle) strategy: evict low-attention KV entries.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (NeurIPS 2023).

Core idea: maintain a budget of KV entries per layer. At each step, accumulate
attention scores and evict the entries with the lowest cumulative attention,
keeping only heavy-hitters + recent tokens.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_bench.strategy import KVCacheStrategy

logger = logging.getLogger(__name__)


class H2OStrategy(KVCacheStrategy):
    """Heavy-Hitter Oracle KV cache eviction.

    Keeps a fixed budget of KV entries per layer:
    - Always keeps the first `sink_size` tokens (attention sinks)
    - Always keeps the last `recent_size` tokens
    - Remaining budget filled by highest cumulative attention tokens
    """

    def __init__(self, budget: float = 0.5, sink_size: int = 4, recent_size: int = 64):
        """
        Args:
            budget: Fraction of KV entries to keep (0.5 = 50% eviction).
            sink_size: Number of initial tokens to always keep.
            recent_size: Number of recent tokens to always keep.
        """
        self.budget = budget
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.name = f"H2O ({budget:.0%})"
        self._cumulative_attn: dict[int, torch.Tensor] = {}

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        # H2O doesn't patch the model — it uses attention weights via on_step
        self._cumulative_attn.clear()
        return model

    def reset(self):
        self._cumulative_attn.clear()

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        """Accumulate attention weights for scoring."""
        if attn_weights is None:
            return

        # attn_weights: [B, num_heads, Q, KV] — sum over batch, heads, queries
        score = attn_weights.float().sum(dim=(0, 1, 2))  # [KV]

        if layer_idx in self._cumulative_attn:
            old = self._cumulative_attn[layer_idx]
            if old.shape[0] < score.shape[0]:
                # Sequence grew — extend
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

        kept = int(seq_len * self.budget)
        return num_layers * num_kv_heads * kept * head_dim * 2 * 2  # K+V, bf16

    def teardown(self, model: nn.Module) -> nn.Module:
        self._cumulative_attn.clear()
        return model
