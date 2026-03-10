"""Adaptive Tiered KV Cache for retrieval heads.

Extends the existing KV2State hybrid attention by adding multi-tier compression
to retrieval heads (which previously kept full FP16 KV cache). Works alongside
streaming heads' recurrent state.

Architecture (combining best ideas from all 10 analogies):
- Tier 0: FP16 (full precision) — high-importance tokens
- Tier 1: INT8 symmetric quantization — medium importance
- Tier 2: INT4 symmetric quantization — low importance
- Tier 3: Evicted + residual sketch (for potential recovery)

Key innovations:
- Multi-signal importance scoring (all 10 analogies agree)
- Layer-wise pyramid budget (PyramidKV, Triage #6, Geology #9)
- Redundancy penalization (GCKV #8)
- Error-aware attention weighting (Anastylosis #5)
- Adaptive threshold with hysteresis (OmniForage #3, MVT #4)
- Hockey-stick degradation monitoring (Triage #6, Geology #9)
"""

import math
import logging
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv2state.importance import ImportanceScorer, ImportanceConfig

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveCacheConfig:
    """Configuration for adaptive tiered KV cache."""
    # Budget allocation (fraction of KV entries per tier)
    budget_fp16: float = 0.25    # top 25% at full precision
    budget_int8: float = 0.30    # next 30% at INT8
    budget_int4: float = 0.25    # next 25% at INT4
    # remaining 20% evicted with sketch

    # Compression settings
    int8_group_size: int = 128   # quantization group size
    int4_group_size: int = 64

    # Error monitoring (Anastylosis #5: error-aware attention)
    error_aware_attention: bool = True
    error_penalty_lambda: float = 0.1  # penalize logits by λ·ε_j

    # Capacity trigger (Ripple-Cache #2: consolidate at 85% HBM)
    consolidation_threshold: float = 0.85  # fraction of max_seq_len

    # Re-scoring cadence (Triage #6: adaptive frequency)
    rescore_every: int = 32      # re-evaluate tiers every N tokens

    # Sketch for evicted tokens (for potential recovery)
    sketch_rank: int = 8         # low-rank sketch dimension

    # Importance scoring
    importance_config: ImportanceConfig = None

    def __post_init__(self):
        if self.importance_config is None:
            self.importance_config = ImportanceConfig()


class TieredKVCache:
    """Adaptive tiered KV cache with multi-signal importance scoring.

    Manages KV cache for retrieval heads with progressive compression.
    Maintains per-entry reconstruction error estimates for error-aware attention.
    """

    def __init__(
        self,
        config: AdaptiveCacheConfig,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Importance scorer
        self.scorer = ImportanceScorer(
            config=config.importance_config,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

        # Per-layer KV storage
        # key_cache[layer]: [num_kv_heads, seq_len, head_dim]
        # value_cache[layer]: same
        self.key_cache: dict[int, torch.Tensor] = {}
        self.value_cache: dict[int, torch.Tensor] = {}

        # Tier assignments: [seq_len] per layer
        self.tiers: dict[int, torch.Tensor] = {}

        # Per-entry reconstruction error estimates
        self.recon_error: dict[int, torch.Tensor] = {}

        # Quantized storage (compressed representations)
        # For INT8/INT4, store quantized values + scales
        self.quantized_keys: dict[int, dict] = {}    # layer -> {tier -> (data, scale, zero)}
        self.quantized_values: dict[int, dict] = {}

        # Eviction sketches (low-rank approximation for tier 3)
        self.sketches: dict[int, tuple] = {}  # layer -> (U, S, V) truncated SVD

        # Step counter for re-scoring cadence
        self.step = 0

        # Error monitoring: track quality degradation
        self.error_history: list[float] = []

    def reset(self):
        """Clear all cache state for a new sequence."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.tiers.clear()
        self.recon_error.clear()
        self.quantized_keys.clear()
        self.quantized_values.clear()
        self.sketches.clear()
        self.scorer.reset()
        self.step = 0
        self.error_history.clear()

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ):
        """Append new KV entries and optionally recompress.

        Args:
            layer_idx: Layer index.
            key_states: [batch, num_kv_heads, new_len, head_dim]
            value_states: Same shape.
            attn_weights: Optional [batch, num_kv_heads, q_len, kv_len] for scoring.
            token_ids: Optional [batch, kv_len] for structural detection.
        """
        B, H, T_new, D = key_states.shape

        # Append to raw cache (we store uncompressed first, compress on cadence)
        if layer_idx in self.key_cache:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )
        else:
            self.key_cache[layer_idx] = key_states.clone()
            self.value_cache[layer_idx] = value_states.clone()

        # Update importance scorer
        if attn_weights is not None:
            self.scorer.update(layer_idx, attn_weights, self.key_cache[layer_idx],
                               token_ids, self.step)

        self.step += 1

        # Check if we should recompress
        current_len = self.key_cache[layer_idx].shape[2]
        should_compress = (
            self.step % self.config.rescore_every == 0
            and current_len > 64  # don't bother for short sequences
        )

        if should_compress:
            self._compress_layer(layer_idx)

    def _compress_layer(self, layer_idx: int):
        """Recompress a layer's KV cache based on current importance scores.

        Applies tiered quantization:
        - Tier 0: Keep FP16
        - Tier 1: Quantize to INT8
        - Tier 2: Quantize to INT4
        - Tier 3: Evict (store low-rank sketch)
        """
        keys = self.key_cache[layer_idx]      # [B, H, T, D]
        values = self.value_cache[layer_idx]
        kv_len = keys.shape[2]

        # Get importance scores and assign tiers
        scores = self.scorer.get_scores(layer_idx, keys)
        tiers = self.scorer.assign_tiers(
            layer_idx, scores,
            budget_fp16=self.config.budget_fp16,
            budget_int8=self.config.budget_int8,
            budget_int4=self.config.budget_int4,
        )
        self.tiers[layer_idx] = tiers

        # Initialize reconstruction error
        self.recon_error[layer_idx] = torch.zeros(kv_len, device=self.device, dtype=torch.float32)

        # Tier 1: INT8 quantization
        int8_mask = tiers == 1
        if int8_mask.any():
            int8_indices = int8_mask.nonzero(as_tuple=True)[0]
            k_int8, k_scale8, k_zp8 = _symmetric_quantize(
                keys[:, :, int8_indices], bits=8, group_size=self.config.int8_group_size
            )
            v_int8, v_scale8, v_zp8 = _symmetric_quantize(
                values[:, :, int8_indices], bits=8, group_size=self.config.int8_group_size
            )

            # Compute reconstruction error
            k_recon = _symmetric_dequantize(k_int8, k_scale8, k_zp8)
            v_recon = _symmetric_dequantize(v_int8, v_scale8, v_zp8)
            k_err = (keys[:, :, int8_indices].float() - k_recon.float()).norm(dim=-1).mean(dim=(0, 1))
            v_err = (values[:, :, int8_indices].float() - v_recon.float()).norm(dim=-1).mean(dim=(0, 1))
            self.recon_error[layer_idx][int8_indices] = (k_err + v_err) / 2

            # Store compressed (replace FP16 entries)
            keys[:, :, int8_indices] = k_recon.to(keys.dtype)
            values[:, :, int8_indices] = v_recon.to(values.dtype)

        # Tier 2: INT4 quantization
        int4_mask = tiers == 2
        if int4_mask.any():
            int4_indices = int4_mask.nonzero(as_tuple=True)[0]
            k_int4, k_scale4, k_zp4 = _symmetric_quantize(
                keys[:, :, int4_indices], bits=4, group_size=self.config.int4_group_size
            )
            v_int4, v_scale4, v_zp4 = _symmetric_quantize(
                values[:, :, int4_indices], bits=4, group_size=self.config.int4_group_size
            )

            k_recon = _symmetric_dequantize(k_int4, k_scale4, k_zp4)
            v_recon = _symmetric_dequantize(v_int4, v_scale4, v_zp4)
            k_err = (keys[:, :, int4_indices].float() - k_recon.float()).norm(dim=-1).mean(dim=(0, 1))
            v_err = (values[:, :, int4_indices].float() - v_recon.float()).norm(dim=-1).mean(dim=(0, 1))
            self.recon_error[layer_idx][int4_indices] = (k_err + v_err) / 2

            keys[:, :, int4_indices] = k_recon.to(keys.dtype)
            values[:, :, int4_indices] = v_recon.to(values.dtype)

        # Tier 3: Eviction with sketch
        evict_mask = tiers == 3
        if evict_mask.any():
            evict_indices = evict_mask.nonzero(as_tuple=True)[0]
            n_evict = len(evict_indices)

            if n_evict > self.config.sketch_rank:
                # Store low-rank sketch via truncated SVD
                evicted_k = keys[:, :, evict_indices].float().mean(dim=0)  # [H, n_evict, D]
                # Reshape to 2D for SVD: [H*n_evict, D]
                evicted_flat = evicted_k.reshape(-1, self.head_dim)
                rank = min(self.config.sketch_rank, *evicted_flat.shape)
                U, S, Vh = torch.linalg.svd(evicted_flat, full_matrices=False)
                self.sketches[layer_idx] = (
                    U[:, :rank].to(self.dtype),
                    S[:rank].to(self.dtype),
                    Vh[:rank].to(self.dtype),
                    evict_indices,
                )

            # Set evicted entries to zero (they're "gone" from the active cache)
            keys[:, :, evict_indices] = 0
            values[:, :, evict_indices] = 0
            self.recon_error[layer_idx][evict_indices] = float("inf")

        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values

    def get_effective_kv(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get the effective KV cache for attention computation.

        Returns:
            keys: [B, H, T, D] — mixed-precision reconstructed keys
            values: Same shape
            error_weights: Optional [T] — per-position error for error-aware attention
        """
        keys = self.key_cache.get(layer_idx)
        values = self.value_cache.get(layer_idx)

        if keys is None:
            return None, None, None

        error_weights = None
        if self.config.error_aware_attention and layer_idx in self.recon_error:
            error_weights = self.recon_error[layer_idx]

        return keys, values, error_weights

    @property
    def memory_bytes(self) -> int:
        """Estimate total memory usage across all layers."""
        total = 0
        for layer_idx in self.key_cache:
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
            kv_len = k.shape[2]

            if layer_idx in self.tiers:
                tiers = self.tiers[layer_idx]
                # FP16 entries
                n_fp16 = (tiers == 0).sum().item()
                # INT8 entries (1 byte per element + scales)
                n_int8 = (tiers == 1).sum().item()
                # INT4 entries (0.5 bytes per element + scales)
                n_int4 = (tiers == 2).sum().item()
                # Evicted: sketch only
                n_evict = (tiers == 3).sum().item()

                bytes_per_head = self.head_dim
                n_heads = self.num_kv_heads
                # K + V for each tier
                fp16_bytes = n_fp16 * bytes_per_head * 2 * n_heads * 2  # 2 bytes per element
                int8_bytes = n_int8 * bytes_per_head * 1 * n_heads * 2 + n_int8 * 4  # + scales
                int4_bytes = n_int4 * (bytes_per_head // 2) * n_heads * 2 + n_int4 * 4
                sketch_bytes = self.config.sketch_rank * self.head_dim * 4 if n_evict > 0 else 0

                total += fp16_bytes + int8_bytes + int4_bytes + sketch_bytes
            else:
                # No compression yet — full FP16
                total += k.nelement() * k.element_size() + v.nelement() * v.element_size()

        return total

    @property
    def compression_ratio(self) -> float:
        """Current compression ratio vs full FP16."""
        if not self.key_cache:
            return 1.0
        full_bytes = 0
        for k in self.key_cache.values():
            full_bytes += k.nelement() * 2 * 2  # K + V, 2 bytes each
        if full_bytes == 0:
            return 1.0
        return full_bytes / max(self.memory_bytes, 1)

    def get_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        stats = {
            "total_entries": 0,
            "tier_counts": {0: 0, 1: 0, 2: 0, 3: 0},
            "compression_ratio": self.compression_ratio,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "mean_recon_error": {},
        }
        for layer_idx in self.tiers:
            tiers = self.tiers[layer_idx]
            stats["total_entries"] += len(tiers)
            for t in range(4):
                stats["tier_counts"][t] += (tiers == t).sum().item()

            if layer_idx in self.recon_error:
                finite_err = self.recon_error[layer_idx]
                finite_mask = finite_err.isfinite()
                if finite_mask.any():
                    stats["mean_recon_error"][layer_idx] = finite_err[finite_mask].mean().item()

        return stats


# ============================================================
# Quantization utilities
# ============================================================

def _symmetric_quantize(
    tensor: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-group quantization.

    Args:
        tensor: [..., D] tensor to quantize.
        bits: Number of bits (4 or 8).
        group_size: Number of elements per quantization group.

    Returns:
        quantized: Same shape, dtype=int8 (INT4 packed in INT8 for storage).
        scale: Per-group scale factors.
        zero_point: Always 0 for symmetric (kept for API consistency).
    """
    shape = tensor.shape
    D = shape[-1]
    flat = tensor.reshape(-1, D).float()

    # Pad to multiple of group_size
    pad = (group_size - D % group_size) % group_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))

    # Reshape into groups
    grouped = flat.reshape(-1, group_size)

    # Compute scale: max absolute value per group
    qmax = 2 ** (bits - 1) - 1
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax

    # Quantize
    quantized = (grouped / scale).round().clamp(-qmax, qmax).to(torch.int8)

    # Reshape back
    quantized = quantized.reshape(flat.shape[0], -1)[:, :D].reshape(shape)
    num_groups = (D + group_size - 1) // group_size
    scale = scale.reshape(flat.shape[0], num_groups)

    return quantized, scale, torch.zeros_like(scale)


def _symmetric_dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize symmetric per-group quantized tensor.

    Returns floating-point reconstruction.
    """
    shape = quantized.shape
    D = shape[-1]
    flat_q = quantized.reshape(-1, D).float()

    # Expand scale to match
    group_size = D // max(scale.shape[-1], 1)
    if group_size == 0:
        group_size = D

    scale_expanded = scale.repeat_interleave(group_size, dim=-1)
    if scale_expanded.shape[-1] < D:
        # Pad with per-row last scale value, not global mean
        pad_len = D - scale_expanded.shape[-1]
        last_scale = scale[:, -1:].expand(-1, pad_len)
        scale_expanded = torch.cat([scale_expanded, last_scale], dim=-1)
    scale_expanded = scale_expanded[:, :D]

    # Dequantize
    flat_q = flat_q.reshape(-1, D)
    result = flat_q * scale_expanded[:flat_q.shape[0]]

    return result.reshape(shape)


# ============================================================
# Error-aware attention modifier
# ============================================================

def apply_error_aware_attention(
    attn_logits: torch.Tensor,
    recon_error: torch.Tensor,
    penalty_lambda: float = 0.1,
) -> torch.Tensor:
    """Modify attention logits by penalizing entries with high reconstruction error.

    Anastylosis analogy (#5): ã_ij = QK^T/√d - λ·ε_j

    This prevents the model from confidently attending to poorly-reconstructed
    entries, reducing the "false stone" failure mode.

    Args:
        attn_logits: [B, H, Q, KV] pre-softmax attention scores.
        recon_error: [KV] per-position reconstruction error.
        penalty_lambda: Scaling factor for error penalty.

    Returns:
        Modified attention logits.
    """
    # Normalize error to [0, 1] range
    finite_mask = recon_error.isfinite()
    if not finite_mask.any():
        return attn_logits

    err = recon_error.clone()
    err[~finite_mask] = err[finite_mask].max() * 2  # evicted tokens get large penalty

    err_max = err.max()
    if err_max > 0:
        err = err / err_max

    # Apply penalty: [1, 1, 1, KV] broadcast
    penalty = penalty_lambda * err.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return attn_logits - penalty.to(attn_logits.dtype)
