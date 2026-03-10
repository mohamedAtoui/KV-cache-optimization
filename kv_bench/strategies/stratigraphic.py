"""Stratigraphic KV-cache strategy for benchmarking.

Per-head zone assignment with monotonic downgrade, stylolite anchors,
and inverse layer budget (compress early layers more, preserve late layers).
"""

from __future__ import annotations

import torch
from torch import nn

from kv_bench.strategy import KVCacheStrategy
from kv2state.stratigraphic import (
    StratigraphicConfig,
    HeadZoneAssigner,
    AnchorDetector,
    ZONE_FP16,
    ZONE_INT8,
    ZONE_INT4,
    ZONE_EVICT,
)


class StratigraphicStrategy(KVCacheStrategy):
    """Stratigraphic KV-cache: per-head zone assignment with monotonic downgrade.

    Analytical strategy (no model patching). Collects attention weights to
    compute per-head importance scores, then assigns compression zones.
    """

    def __init__(self, config: StratigraphicConfig | None = None):
        self.config = config or StratigraphicConfig()
        self.name = "Stratigraphic"
        self._zone_assigner = HeadZoneAssigner(self.config)
        self._anchor_detector = AnchorDetector(self.config)
        self._cumulative_attn: dict[int, torch.Tensor] = {}
        self._num_layers: int | None = None
        self._num_kv_heads: int | None = None
        self._num_q_heads: int | None = None

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        """Initialize internal state (no model patching)."""
        self._cumulative_attn.clear()
        self._zone_assigner.clear()
        self._num_layers = model_config.num_hidden_layers
        self._num_kv_heads = getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        )
        self._num_q_heads = model_config.num_attention_heads
        return model

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        """Accumulate per-head attention scores preserving head identity.

        attn_weights shape: [B, num_q_heads, Q, KV]
        For GQA (Llama-3.1-8B): num_q_heads=32, num_kv_heads=8, group_size=4
        We reshape to [B, num_kv_heads, group_size, Q, KV], average within
        groups, then sum over batch and queries to get [num_kv_heads, KV].
        """
        if attn_weights is None:
            return

        num_q = self._num_q_heads or attn_weights.shape[1]
        num_kv = self._num_kv_heads or num_q
        group_size = num_q // num_kv

        # [B, num_q, Q, KV] → [B, num_kv, group_size, Q, KV]
        w = attn_weights.float()
        B, _, Q, KV = w.shape
        w = w.view(B, num_kv, group_size, Q, KV)

        # Average within group, sum over batch and queries → [num_kv, KV]
        score = w.mean(dim=2).sum(dim=(0, 2))  # [num_kv, KV]

        if layer_idx in self._cumulative_attn:
            old = self._cumulative_attn[layer_idx]
            if old.shape[-1] < KV:
                new = torch.zeros(num_kv, KV, device=score.device)
                new[:, : old.shape[-1]] = old
                old = new
            self._cumulative_attn[layer_idx] = old[:, :KV] + score
        else:
            self._cumulative_attn[layer_idx] = score

    def memory_bytes(self, seq_len: int, model_config) -> int:
        """Compute analytical memory with per-layer-varying compression.

        Each layer gets a different FP16 fraction based on depth:
        fp16_frac(l) = zone_surface * [(1 - lambda_) + lambda_ * l / L]
        """
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        )
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        cfg = self.config

        total_bytes = 0
        for l in range(num_layers):
            fp16_frac = cfg.zone_surface * (
                (1 - cfg.lambda_) + cfg.lambda_ * l / max(num_layers - 1, 1)
            )
            # Ensure at least anchor_budget fraction at FP16
            fp16_frac = max(fp16_frac, cfg.anchor_budget)

            n_fp16 = int(seq_len * fp16_frac)
            n_int8 = int(seq_len * cfg.zone_shallow)
            n_int4 = int(seq_len * cfg.zone_deep)
            # Remaining tokens are evicted (only sketch stored)

            layer_bytes = num_kv_heads * (
                n_fp16 * head_dim * 2 * 2  # K+V at FP16 (2 bytes each)
                + n_int8 * head_dim * 1 * 2  # K+V at INT8 (1 byte each)
                + n_int4 * (head_dim // 2) * 2  # K+V at INT4 (0.5 bytes each)
                + cfg.sketch_rank * head_dim * 4  # SVD sketch per head (float32)
            )
            total_bytes += layer_bytes

        return total_bytes

    def reset(self):
        """Clear cumulative attention and zone history between windows."""
        self._cumulative_attn.clear()
        self._zone_assigner.clear()

    def teardown(self, model: nn.Module) -> nn.Module:
        """Return model unchanged (no patching was done)."""
        self._cumulative_attn.clear()
        self._zone_assigner.clear()
        return model
