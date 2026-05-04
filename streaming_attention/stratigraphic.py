"""Stratigraphic KV-Cache: Per-head zone assignment with monotonic downgrade.

Inspired by geological stratigraphy — tokens are "deposited" in compression zones
and can only move to deeper compression (FP16 → INT8 → INT4 → evict), never back.
This prevents re-compression error compounding ("diagenetic overprinting").

Key innovations:
- Per-head zone assignment (each KV head gets its own compression profile)
- Monotonic downgrade-only constraint
- Stylolite anchors: high-attention tokens pinned at FP16
- Inverse layer budget: early layers compress more, late layers preserve more
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# Zone constants
ZONE_FP16 = 0
ZONE_INT8 = 1
ZONE_INT4 = 2
ZONE_EVICT = 3
ZONE_TQ4 = 4    # TurboQuant 4-bit (stage1=3 + QJL=1)
ZONE_TQ3 = 5    # TurboQuant 3-bit (stage1=2 + QJL=1)


@dataclass
class StratigraphicConfig:
    """Configuration for stratigraphic KV-cache compression."""

    # Layer budget: fp16_frac(l) = zone_surface * [(1 - lambda_) + lambda_ * l / L]
    zone_surface: float = 0.20  # max FP16 fraction (at deepest layer)
    lambda_: float = 0.6  # layer-scaling factor (0 = uniform, 1 = full gradient)

    # Zone fractions (must sum to 1.0 — no eviction, compression via quant only)
    # Memory: 0.20×1.0 + 0.40×0.5 + 0.40×0.25 = 0.50 → 2.0x compression
    zone_shallow: float = 0.40  # INT8 fraction
    zone_deep: float = 0.40  # INT4 fraction

    # Anchor detection
    anchor_budget: float = 0.05  # max fraction of tokens as anchors
    anchor_attn_percentile: float = 0.99  # attention threshold for anchors

    # Sink + recent protection (matches H2O/SnapKV)
    sink_size: int = 4  # initial tokens always kept (attention sinks)
    recent_size: int = 64  # recent tokens always kept (local context)

    # Compression params
    sketch_rank: int = 8  # rank of SVD sketch for evicted tokens
    int8_group_size: int = 128
    int4_group_size: int = 64

    # Diagnostic flags (for ablation experiments)
    eviction_only: bool = False  # If True, skip quant hooks (eviction-only mode)
    quant_only: bool = False  # If True, keep all tokens (quant-only mode)


class AnchorDetector:
    """Detect anchor tokens that should be pinned at FP16.

    Tokens above 99th-percentile cumulative attention (across heads)
    are marked as anchors, capped at anchor_budget fraction.
    """

    def __init__(self, config: StratigraphicConfig):
        self.config = config

    def detect_anchors(
        self,
        cumulative_attn: Tensor,
    ) -> Tensor:
        """Detect anchor positions.

        Args:
            cumulative_attn: [H, seq_len] cumulative attention per head.

        Returns:
            [seq_len] bool mask of anchor positions.
        """
        seq_len = cumulative_attn.shape[-1]
        device = cumulative_attn.device

        # High-attention tokens (mean across heads)
        mean_attn = cumulative_attn.float().mean(dim=0)  # [seq_len]
        if seq_len > 0:
            threshold = torch.quantile(mean_attn, self.config.anchor_attn_percentile)
            anchors = mean_attn >= threshold
        else:
            anchors = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Cap at anchor budget
        max_anchors = max(1, int(self.config.anchor_budget * seq_len))
        if anchors.sum() > max_anchors:
            # Keep highest-attention anchors within budget
            scores = mean_attn.clone()
            scores[~anchors] = -float("inf")
            _, top_idx = scores.topk(max_anchors)
            anchors = torch.zeros(seq_len, dtype=torch.bool, device=device)
            anchors[top_idx] = True

        return anchors


class HeadZoneAssigner:
    """Assign compression zones per-head with monotonic downgrade enforcement.

    Each KV head gets its own zone assignment based on per-head attention scores.
    Tokens can only move to deeper compression zones over time, never back.
    """

    def __init__(self, config: StratigraphicConfig):
        self.config = config
        # Zone history: (layer_idx, head_idx) → Tensor[seq_len] of zone assignments
        self._zone_history: dict[tuple[int, int], Tensor] = {}

    def assign_zones(
        self,
        layer_idx: int,
        per_head_scores: Tensor,
        anchors: Tensor,
        num_layers: int,
    ) -> Tensor:
        """Assign compression zones for each head at a given layer.

        Args:
            layer_idx: Current layer index.
            per_head_scores: [H, seq_len] importance scores per head.
            anchors: [seq_len] bool mask of anchor positions.
            num_layers: Total number of layers.

        Returns:
            [H, seq_len] zone assignments (0=FP16, 1=INT8, 2=INT4, 3=evict).
        """
        H, seq_len = per_head_scores.shape
        device = per_head_scores.device
        cfg = self.config

        # Layer-adjusted FP16 fraction: more compression in early layers
        fp16_frac = cfg.zone_surface * (
            (1 - cfg.lambda_) + cfg.lambda_ * layer_idx / max(num_layers - 1, 1)
        )

        zones = torch.full((H, seq_len), ZONE_EVICT, dtype=torch.long, device=device)

        for h in range(H):
            scores = per_head_scores[h]  # [seq_len]

            # Sort by score descending
            sorted_idx = scores.argsort(descending=True)

            # Compute budget counts
            n_fp16 = max(int(seq_len * fp16_frac), int(seq_len * cfg.anchor_budget))
            n_int8 = int(seq_len * cfg.zone_shallow)
            n_int4 = int(seq_len * cfg.zone_deep)

            # Assign zones by rank
            if n_fp16 > 0:
                zones[h, sorted_idx[:n_fp16]] = ZONE_FP16
            if n_int8 > 0:
                zones[h, sorted_idx[n_fp16 : n_fp16 + n_int8]] = ZONE_INT8
            if n_int4 > 0:
                zones[h, sorted_idx[n_fp16 + n_int8 : n_fp16 + n_int8 + n_int4]] = (
                    ZONE_INT4
                )

            # Override: anchors pinned to FP16
            zones[h, anchors] = ZONE_FP16

            # Monotonic enforcement: tokens can only move to deeper compression.
            # In the kv_bench sliding-window eval, this is inert because
            # zone_history is cleared between windows and assign_zones is
            # called once per window. Active in real-time generation scenarios.
            key = (layer_idx, h)
            if key in self._zone_history:
                old = self._zone_history[key]
                if old.shape[0] < seq_len:
                    # Sequence grew — extend old with ZONE_FP16 for new positions
                    extended = torch.full(
                        (seq_len,), ZONE_FP16, dtype=torch.long, device=device
                    )
                    extended[: old.shape[0]] = old
                    old = extended
                zones[h] = torch.maximum(old[:seq_len], zones[h])

            self._zone_history[key] = zones[h].clone()

        return zones

    def clear(self):
        """Reset all zone history."""
        self._zone_history.clear()
