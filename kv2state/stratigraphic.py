"""Stratigraphic KV-Cache: Per-head zone assignment with monotonic downgrade.

Inspired by geological stratigraphy — tokens are "deposited" in compression zones
and can only move to deeper compression (FP16 → INT8 → INT4 → evict), never back.
This prevents re-compression error compounding ("diagenetic overprinting").

Key innovations:
- Per-head zone assignment (each KV head gets its own compression profile)
- Monotonic downgrade-only constraint
- Stylolite anchors: high-attention + topic-shift tokens pinned at FP16
- Inverse layer budget: early layers compress more, late layers preserve more
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


# Zone constants
ZONE_FP16 = 0
ZONE_INT8 = 1
ZONE_INT4 = 2
ZONE_EVICT = 3


@dataclass
class StratigraphicConfig:
    """Configuration for stratigraphic KV-cache compression."""

    # Layer budget: fp16_frac(l) = zone_surface * [(1 - lambda_) + lambda_ * l / L]
    zone_surface: float = 0.30  # max FP16 fraction (at deepest layer)
    lambda_: float = 0.6  # layer-scaling factor (0 = uniform, 1 = full gradient)

    # Zone fractions (of remaining tokens after FP16 allocation)
    zone_shallow: float = 0.30  # INT8 fraction
    zone_deep: float = 0.25  # INT4 fraction
    # Remainder is evicted

    # Anchor detection
    anchor_budget: float = 0.05  # max fraction of tokens as anchors
    anchor_attn_percentile: float = 0.99  # attention threshold for anchors

    # Topic-shift detection
    topic_shift_window: int = 32  # sliding window for cosine distance
    topic_shift_threshold: float = 0.3  # cosine distance threshold for topic shift

    # Compression params
    sketch_rank: int = 8  # rank of SVD sketch for evicted tokens
    int8_group_size: int = 128
    int4_group_size: int = 64


class AnchorDetector:
    """Detect anchor tokens that should be pinned at FP16.

    Two signals:
    1. Tokens above 99th-percentile cumulative attention (across heads)
    2. Topic-shift boundaries (high cosine distance in key states)
    """

    def __init__(self, config: StratigraphicConfig):
        self.config = config

    def detect_anchors(
        self,
        cumulative_attn: Tensor,
        key_states: Tensor | None = None,
    ) -> Tensor:
        """Detect anchor positions.

        Args:
            cumulative_attn: [H, seq_len] cumulative attention per head.
            key_states: Optional [B, H, T, D] key states for topic detection.

        Returns:
            [seq_len] bool mask of anchor positions.
        """
        seq_len = cumulative_attn.shape[-1]
        device = cumulative_attn.device

        # Signal 1: high-attention tokens (mean across heads)
        mean_attn = cumulative_attn.float().mean(dim=0)  # [seq_len]
        if seq_len > 0:
            threshold = torch.quantile(mean_attn, self.config.anchor_attn_percentile)
            attn_anchors = mean_attn >= threshold
        else:
            attn_anchors = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Signal 2: topic-shift boundaries via sliding-window cosine distance
        if key_states is not None and seq_len > self.config.topic_shift_window:
            shift_anchors = self._detect_topic_shifts(key_states)
        else:
            shift_anchors = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # Combine with OR
        anchors = attn_anchors | shift_anchors

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

    def _detect_topic_shifts(self, key_states: Tensor) -> Tensor:
        """Detect topic boundaries via sliding-window cosine distance.

        Args:
            key_states: [B, H, T, D] key states.

        Returns:
            [T] bool mask of topic-shift positions.
        """
        # Average over batch and heads → [T, D]
        keys = key_states.float().mean(dim=(0, 1))
        T = keys.shape[0]
        w = self.config.topic_shift_window

        # Compute mean key in sliding windows
        # Left window: [t-w, t), right window: [t, t+w)
        shifts = torch.zeros(T, dtype=torch.bool, device=keys.device)
        if T < 2 * w:
            return shifts

        # Efficient: compute cumulative sum for windowed means
        cumsum = torch.cumsum(keys, dim=0)  # [T, D]

        for t in range(w, T - w):
            left_mean = (cumsum[t] - cumsum[t - w]) / w  # [D]
            right_mean = (cumsum[t + w] - cumsum[t]) / w  # [D]
            cos_sim = torch.nn.functional.cosine_similarity(
                left_mean.unsqueeze(0), right_mean.unsqueeze(0)
            )
            if (1.0 - cos_sim.item()) > self.config.topic_shift_threshold:
                shifts[t] = True

        return shifts


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

            # Monotonic enforcement: can only deepen (increase zone number)
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
