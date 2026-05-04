"""Stratigraphic KV-cache strategy for benchmarking.

Per-head zone assignment with monotonic downgrade, stylolite anchors,
and inverse layer budget (compress early layers more, preserve late layers).
"""

from __future__ import annotations

import torch
from torch import nn

from kv_bench.strategy import KVCacheStrategy
from streaming_attention.stratigraphic import (
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
        self._cached_zone_masks: dict[int, torch.Tensor] | None = None
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

        Scores are normalized by the number of attending queries per position
        to correct for causal attention bias (early positions get more queries).
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

        # Normalize by number of attending queries to remove causal bias.
        # Key position j is attended to by queries j..KV-1, so (KV - j) queries.
        num_queries = torch.arange(KV, 0, -1, device=score.device, dtype=score.dtype)
        score = score / num_queries.unsqueeze(0)  # [num_kv, KV]

        if layer_idx in self._cumulative_attn:
            old = self._cumulative_attn[layer_idx]
            if old.shape[-1] < KV:
                new = torch.zeros(num_kv, KV, device=score.device)
                new[:, : old.shape[-1]] = old
                old = new
            self._cumulative_attn[layer_idx] = old[:, :KV] + score
        else:
            self._cumulative_attn[layer_idx] = score

    def get_zone_masks(self, seq_len, device):
        """Return per-layer, per-head zone assignments (cached per window).

        Returns:
            {layer_idx: Tensor[num_kv_heads, seq_len]} with zone IDs
            (ZONE_FP16=0, ZONE_INT8=1, ZONE_INT4=2, ZONE_EVICT=3).
            Returns None if eviction_only mode (skip quant hooks).
        """
        if self.config.eviction_only:
            return None

        if self._cached_zone_masks is not None:
            return self._cached_zone_masks

        if not self._cumulative_attn:
            return None
        cfg = self.config
        num_layers = self._num_layers or 1
        num_kv = self._num_kv_heads or 1

        # Detect anchors from averaged cumulative attention
        all_scores = []
        for scores in self._cumulative_attn.values():
            all_scores.append(scores[:, :seq_len] if scores.dim() > 1 else scores[:seq_len])
        avg_attn = torch.stack(all_scores).mean(dim=0)  # [num_kv, seq_len] or [seq_len]
        if avg_attn.dim() == 1:
            avg_attn = avg_attn.unsqueeze(0)
        anchors = self._anchor_detector.detect_anchors(avg_attn)

        # Force sink + recent as anchors
        sink = min(cfg.sink_size, seq_len)
        recent = min(cfg.recent_size, seq_len)
        anchors[:sink] = True
        anchors[max(0, seq_len - recent):] = True

        zone_masks = {}
        for l in range(num_layers):
            if l in self._cumulative_attn:
                per_head = self._cumulative_attn[l][:, :seq_len]
            else:
                per_head = torch.zeros(num_kv, seq_len, device=device)

            zones = self._zone_assigner.assign_zones(
                layer_idx=l,
                per_head_scores=per_head,
                anchors=anchors,
                num_layers=num_layers,
            )
            zone_masks[l] = zones

        self._cached_zone_masks = zone_masks
        return zone_masks

    def get_keep_mask(self, seq_len, device):
        """Return keep mask for token eviction.

        Default (quant_only=True or default): all-True mask — no eviction,
        compression via quant zones only. Still returns a mask so Pass 2
        runs and quant hooks are applied.

        With quant_only=False (eviction_only mode): uses averaged eviction
        scoring for ablation experiments.
        """
        if not self._cumulative_attn:
            return None

        # Default behavior: keep everything, let quant do the compression
        if not self.config.eviction_only:
            return torch.ones(seq_len, dtype=torch.bool, device=device)

        # Eviction-only mode (diagnostic): averaged eviction scoring
        cfg = self.config
        num_layers = self._num_layers or 1

        keep_fracs = []
        for l in range(num_layers):
            fp16_frac = cfg.zone_surface * (
                (1 - cfg.lambda_) + cfg.lambda_ * l / max(num_layers - 1, 1)
            )
            fp16_frac = max(fp16_frac, cfg.anchor_budget)
            keep_fracs.append(min(fp16_frac + cfg.zone_shallow + cfg.zone_deep, 1.0))
        keep_count = int(seq_len * sum(keep_fracs) / len(keep_fracs))

        all_scores = []
        for scores in self._cumulative_attn.values():
            all_scores.append(scores[:, :seq_len].mean(dim=0) if scores.dim() > 1 else scores[:seq_len])
        avg_scores = torch.stack(all_scores).mean(dim=0)

        keep = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep[:min(cfg.sink_size, seq_len)] = True
        keep[max(0, seq_len - min(cfg.recent_size, seq_len)):] = True

        remaining = keep_count - keep.sum().item()
        if remaining > 0:
            masked = avg_scores.clone()
            masked[keep] = -float('inf')
            topk = masked.topk(min(remaining, (~keep).sum().item())).indices
            keep[topk] = True
        return keep

    def memory_bytes(self, seq_len: int, model_config) -> int:
        """Compute analytical memory with per-layer-varying compression.

        Each layer gets a different FP16 fraction based on depth:
        fp16_frac(l) = zone_surface * [(1 - lambda_) + lambda_ * l / L]

        All tokens are kept (no eviction). Zone counts sum to seq_len.
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
            fp16_frac = max(fp16_frac, cfg.anchor_budget)

            n_fp16 = max(
                int(seq_len * fp16_frac),
                cfg.sink_size + cfg.recent_size,
            )
            n_int8 = int(seq_len * cfg.zone_shallow)
            n_int4 = max(0, seq_len - n_fp16 - n_int8)

            layer_bytes = num_kv_heads * (
                n_fp16 * head_dim * 2 * 2  # K+V at FP16 (2 bytes each)
                + n_int8 * head_dim * 1 * 2  # K+V at INT8 (1 byte each)
                + n_int4 * (head_dim // 2) * 2  # K+V at INT4 (0.5 bytes each)
            )
            total_bytes += layer_bytes

        return total_bytes

    def reset(self):
        """Clear cumulative attention and zone history between windows."""
        self._cumulative_attn.clear()
        self._cached_zone_masks = None
        self._zone_assigner.clear()

    def teardown(self, model: nn.Module) -> nn.Module:
        """Return model unchanged (no patching was done)."""
        self._cumulative_attn.clear()
        self._cached_zone_masks = None
        self._zone_assigner.clear()
        return model
