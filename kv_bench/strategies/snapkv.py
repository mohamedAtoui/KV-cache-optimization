"""SnapKV strategy: observation-window-based KV cache selection.

Reference: Li et al., "SnapKV: LLM Knows What You Are Looking For Before
Generation" (2024).

Core idea: use attention patterns from a small observation window at the end
of the prefill to select which KV entries to keep for the full sequence.
Entries that receive high attention in the observation window are likely to
remain important during generation.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy

logger = logging.getLogger(__name__)


class SnapKVStrategy(KVCacheStrategy):
    """SnapKV observation-window KV selection.

    During prefill, uses the last `obs_window` tokens' attention patterns
    to score all previous KV entries. Keeps top `budget` fraction plus
    recent tokens and attention sinks.
    """

    def __init__(
        self,
        budget: float = 0.5,
        obs_window: int = 64,
        sink_size: int = 4,
        kernel_size: int = 5,
    ):
        """
        Args:
            budget: Fraction of KV entries to keep.
            obs_window: Number of tokens at end of prefill to use as observation.
            sink_size: Number of initial tokens to always keep.
            kernel_size: Pooling kernel for smoothing attention scores.
        """
        self.budget = budget
        self.obs_window = obs_window
        self.sink_size = sink_size
        self.kernel_size = kernel_size
        self.name = f"SnapKV ({budget:.0%})"
        self._obs_attn: dict[int, torch.Tensor] = {}

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        self._obs_attn.clear()
        return model

    def reset(self):
        self._obs_attn.clear()

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        """Collect attention from observation window."""
        if attn_weights is None:
            return

        # attn_weights: [B, num_heads, Q, KV]
        q_len = attn_weights.shape[2]
        kv_len = attn_weights.shape[3]

        # Use last obs_window query positions as observation
        obs_start = max(0, q_len - self.obs_window)
        obs_attn = attn_weights[:, :, obs_start:, :].float()

        # Average over batch and observation queries, max over heads
        # → per-position importance score
        score = obs_attn.mean(dim=(0, 2)).max(dim=0).values  # [KV]

        # Optional: smooth with average pooling to capture clusters
        if self.kernel_size > 1 and score.shape[0] > self.kernel_size:
            pad = self.kernel_size // 2
            score_padded = torch.nn.functional.pad(
                score.unsqueeze(0).unsqueeze(0), (pad, pad), mode="replicate"
            )
            score = torch.nn.functional.avg_pool1d(
                score_padded, self.kernel_size, stride=1
            ).squeeze()

        self._obs_attn[layer_idx] = score

    def get_keep_mask(self, seq_len, device):
        if not self._obs_attn:
            return None
        # Pad or truncate scores to match seq_len
        raw = torch.stack(list(self._obs_attn.values())).mean(dim=0)
        if raw.shape[0] < seq_len:
            scores = torch.zeros(seq_len, device=raw.device, dtype=raw.dtype)
            scores[:raw.shape[0]] = raw
        else:
            scores = raw[:seq_len]
        budget = int(seq_len * self.budget)
        keep = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep[:self.sink_size] = True
        keep[max(0, seq_len - self.obs_window):] = True
        remaining = budget - keep.sum().item()
        if remaining > 0:
            scores_masked = scores.clone()
            scores_masked[keep] = -float('inf')
            topk = scores_masked.topk(min(remaining, (~keep).sum().item())).indices
            keep[topk] = True
        return keep

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        kept = int(seq_len * self.budget)
        return num_layers * num_kv_heads * kept * head_dim * 2 * 2

    def teardown(self, model: nn.Module) -> nn.Module:
        self._obs_attn.clear()
        return model
