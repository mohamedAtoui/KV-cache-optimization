"""Multi-signal importance scoring for KV cache entries.

Implements composite importance scoring inspired by:
- Neuroscience (#1, #2): congruence/schema-deviation scoring
- Ecology (#3, #4): profitability = importance / cost, adaptive threshold
- Immunology (#7, #8): cross-head entropy, redundancy penalization
- Triage (#6): multi-signal composite with structural salience

Core signals:
1. Cumulative attention weight (H2O heavy-hitter)
2. Exponential recency decay
3. Cross-head attention variance (uncertainty signal)
4. Semantic distinctiveness (1 - max cosine similarity to neighbors)
5. Structural salience (BOS, separators, high-entropy tokens)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImportanceConfig:
    """Configuration for importance scoring."""
    # Signal weights (sum to 1.0 internally)
    w_attention: float = 0.35      # cumulative attention weight
    w_recency: float = 0.20        # exponential recency
    w_cross_head_var: float = 0.15 # inter-head attention variance
    w_distinctiveness: float = 0.20 # semantic uniqueness
    w_structural: float = 0.10     # structural salience (BOS, separators)

    # Recency decay
    recency_halflife: int = 256    # tokens until importance halves

    # Redundancy penalization (GCKV analogy #8)
    redundancy_threshold: float = 0.92  # cosine sim above this = redundant
    redundancy_penalty: float = 0.5     # multiply score by this for redundant pairs

    # Adaptive threshold (OmniForage analogy #3)
    ema_alpha: float = 0.01        # EMA smoothing for threshold λ*
    hysteresis_band: float = 0.05  # prevent tier thrashing

    # Structural tokens (token IDs to pin)
    pin_token_ids: list = field(default_factory=list)  # BOS, EOS, separators
    pin_first_n: int = 4           # always keep first N tokens (attention sinks)


class ImportanceScorer:
    """Computes multi-signal composite importance for KV cache entries.

    Maintains running statistics per layer. Call `update()` at each decoding
    step with the attention weights from that step.

    Usage:
        scorer = ImportanceScorer(config, num_layers=32, num_kv_heads=8, device='cuda')
        # At each step:
        scorer.update(layer_idx, attn_weights, key_states, token_ids, step)
        # To get eviction candidates:
        scores = scorer.get_scores(layer_idx)
        # Adaptive threshold:
        threshold = scorer.get_threshold(layer_idx)
    """

    def __init__(
        self,
        config: ImportanceConfig,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int = 8192,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Normalize weights
        total_w = (config.w_attention + config.w_recency + config.w_cross_head_var
                   + config.w_distinctiveness + config.w_structural)
        self.w = {
            "attn": config.w_attention / total_w,
            "recency": config.w_recency / total_w,
            "cross_head": config.w_cross_head_var / total_w,
            "distinct": config.w_distinctiveness / total_w,
            "structural": config.w_structural / total_w,
        }

        # Per-layer running accumulators
        # cumulative_attn[layer]: [num_kv_heads, current_seq_len]
        self.cumulative_attn: dict[int, torch.Tensor] = {}
        # cross_head_var[layer]: [current_seq_len] — variance of attention across heads
        self.cross_head_var: dict[int, torch.Tensor] = {}
        # token_positions[layer]: tracks which positions are occupied
        self.seq_lens: dict[int, int] = {}
        # structural_mask[layer]: [current_seq_len] — 1.0 for pinned tokens
        self.structural_mask: dict[int, torch.Tensor] = {}
        # adaptive threshold per layer
        self.threshold: dict[int, float] = {l: 0.0 for l in range(num_layers)}
        # tier assignments: 0=FP16, 1=INT8, 2=INT4, 3=evicted
        self.tiers: dict[int, torch.Tensor] = {}

        # Recency decay factor per step
        self.recency_gamma = 0.5 ** (1.0 / config.recency_halflife)

    def reset(self):
        """Clear all state for a new sequence."""
        self.cumulative_attn.clear()
        self.cross_head_var.clear()
        self.seq_lens.clear()
        self.structural_mask.clear()
        self.tiers.clear()
        self.threshold = {l: 0.0 for l in range(self.num_layers)}

    def _ensure_layer(self, layer_idx: int, seq_len: int):
        """Initialize or extend accumulators for a layer."""
        if layer_idx not in self.cumulative_attn or self.seq_lens.get(layer_idx, 0) < seq_len:
            old_len = self.seq_lens.get(layer_idx, 0)
            new_attn = torch.zeros(self.num_kv_heads, seq_len, device=self.device, dtype=self.dtype)
            new_var = torch.zeros(seq_len, device=self.device, dtype=self.dtype)
            new_struct = torch.zeros(seq_len, device=self.device, dtype=self.dtype)

            if old_len > 0 and layer_idx in self.cumulative_attn:
                new_attn[:, :old_len] = self.cumulative_attn[layer_idx][:, :old_len]
                new_var[:old_len] = self.cross_head_var[layer_idx][:old_len]
                new_struct[:old_len] = self.structural_mask[layer_idx][:old_len]

            self.cumulative_attn[layer_idx] = new_attn
            self.cross_head_var[layer_idx] = new_var
            self.structural_mask[layer_idx] = new_struct
            self.seq_lens[layer_idx] = seq_len

    def update(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        step: int = 0,
    ):
        """Update importance scores with new attention weights.

        Args:
            layer_idx: Which layer these weights are from.
            attn_weights: [batch, num_kv_heads, q_len, kv_len] attention weights
                          (after softmax, before value multiplication).
                          For GQA, average over query groups first.
            key_states: Optional [batch, num_kv_heads, kv_len, head_dim] for
                        distinctiveness computation.
            token_ids: Optional [batch, kv_len] token IDs for structural detection.
            step: Current decoding step (for recency).
        """
        # Average over batch and query positions
        # attn_weights: [B, H, Q, KV] -> [H, KV]
        attn_avg = attn_weights.float().mean(dim=(0, 2))  # [H, KV]
        kv_len = attn_avg.shape[-1]

        self._ensure_layer(layer_idx, kv_len)

        # 1. Update cumulative attention (EMA-style)
        self.cumulative_attn[layer_idx][:, :kv_len] *= self.recency_gamma
        self.cumulative_attn[layer_idx][:, :kv_len] += attn_avg

        # 2. Update cross-head variance
        # Variance of attention across heads for each position
        head_var = attn_avg.var(dim=0)  # [KV]
        # EMA update
        alpha = self.config.ema_alpha
        self.cross_head_var[layer_idx][:kv_len] = (
            (1 - alpha) * self.cross_head_var[layer_idx][:kv_len] + alpha * head_var
        )

        # 3. Mark structural tokens
        if token_ids is not None:
            struct = self.structural_mask[layer_idx]
            # Pin first N tokens (attention sinks)
            struct[:min(self.config.pin_first_n, kv_len)] = 1.0
            # Pin specific token IDs
            for tid in self.config.pin_token_ids:
                mask = (token_ids[0, :kv_len] == tid)
                struct[:kv_len][mask] = 1.0
            self.structural_mask[layer_idx] = struct

    def get_scores(
        self,
        layer_idx: int,
        key_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute composite importance scores for a layer.

        Args:
            layer_idx: Layer index.
            key_states: Optional [batch, num_kv_heads, kv_len, head_dim] for
                        distinctiveness. If None, distinctiveness score is uniform.

        Returns:
            scores: [kv_len] composite importance scores in [0, 1].
        """
        if layer_idx not in self.cumulative_attn:
            return torch.ones(1, device=self.device, dtype=self.dtype)

        kv_len = self.seq_lens[layer_idx]

        # Signal 1: Cumulative attention (max across heads, then normalize)
        attn_scores = self.cumulative_attn[layer_idx][:, :kv_len].max(dim=0).values
        attn_scores = _safe_normalize(attn_scores)

        # Signal 2: Recency (exponential decay from end)
        positions = torch.arange(kv_len, device=self.device, dtype=self.dtype)
        recency_scores = self.recency_gamma ** (kv_len - 1 - positions)

        # Signal 3: Cross-head variance (high variance = uncertain = keep)
        var_scores = _safe_normalize(self.cross_head_var[layer_idx][:kv_len])

        # Signal 4: Semantic distinctiveness
        if key_states is not None:
            distinct_scores = self._compute_distinctiveness(key_states, kv_len)
        else:
            distinct_scores = torch.ones(kv_len, device=self.device, dtype=self.dtype) * 0.5

        # Signal 5: Structural salience
        struct_scores = self.structural_mask[layer_idx][:kv_len]

        # Composite score
        scores = (
            self.w["attn"] * attn_scores
            + self.w["recency"] * recency_scores
            + self.w["cross_head"] * var_scores
            + self.w["distinct"] * distinct_scores
            + self.w["structural"] * struct_scores
        )

        # Apply redundancy penalty (GCKV analogy #8)
        if key_states is not None:
            scores = self._apply_redundancy_penalty(scores, key_states, kv_len)

        # Pinned tokens get maximum score (never evicted)
        pinned = struct_scores > 0.5
        scores[pinned] = scores.max() + 1.0

        return scores

    def _compute_distinctiveness(
        self,
        key_states: torch.Tensor,
        kv_len: int,
    ) -> torch.Tensor:
        """Semantic distinctiveness: 1 - max cosine similarity to neighbors.

        Low distinctiveness = redundant (safe to compress).
        High distinctiveness = unique (keep at high fidelity).
        """
        # Average across batch and heads: [kv_len, head_dim]
        keys = key_states[:, :, :kv_len].float().mean(dim=(0, 1))

        if kv_len <= 1:
            return torch.ones(kv_len, device=self.device, dtype=self.dtype)

        # Normalize for cosine similarity
        keys_norm = F.normalize(keys, dim=-1)

        # Compute pairwise cosine similarity in windows (avoid O(n^2) for long seqs)
        window = min(64, kv_len)
        max_sim = torch.zeros(kv_len, device=self.device, dtype=self.dtype)

        for i in range(0, kv_len, window // 2):
            end = min(i + window, kv_len)
            chunk = keys_norm[i:end]
            sim = torch.mm(chunk, chunk.t())
            # Zero out self-similarity
            sim.fill_diagonal_(0.0)
            max_sim[i:end] = torch.maximum(max_sim[i:end], sim.max(dim=-1).values)

        # Distinctiveness = 1 - max_similarity
        return 1.0 - max_sim

    def _apply_redundancy_penalty(
        self,
        scores: torch.Tensor,
        key_states: torch.Tensor,
        kv_len: int,
    ) -> torch.Tensor:
        """Penalize redundant high-scoring entries (GCKV insight).

        If two entries are very similar (cosine > threshold), the lower-scored
        one gets penalized. Prevents cache clustering where two near-identical
        entries both survive while unique entries get evicted.
        """
        if kv_len <= 1:
            return scores

        keys = key_states[:, :, :kv_len].float().mean(dim=(0, 1))
        keys_norm = F.normalize(keys, dim=-1)

        # Only check top-scoring entries for redundancy (efficiency)
        top_k = min(256, kv_len)
        top_indices = scores.topk(top_k).indices

        top_keys = keys_norm[top_indices]
        sim = torch.mm(top_keys, top_keys.t())
        sim.fill_diagonal_(0.0)

        # Find redundant pairs
        redundant_mask = sim > self.config.redundancy_threshold
        if redundant_mask.any():
            # For each redundant pair, penalize the lower-scored entry
            top_scores = scores[top_indices]
            for i in range(top_k):
                partners = redundant_mask[i].nonzero(as_tuple=True)[0]
                for j in partners:
                    if top_scores[i] < top_scores[j]:
                        # i is the weaker partner — penalize
                        orig_idx = top_indices[i]
                        scores[orig_idx] *= self.config.redundancy_penalty
                        break  # only penalize once

        return scores

    def get_threshold(self, layer_idx: int) -> float:
        """Get adaptive eviction threshold λ*(t) for a layer.

        Uses EMA of median score with hysteresis band to prevent thrashing.
        Analogous to Charnov's marginal value theorem (#3, #4).
        """
        scores = self.get_scores(layer_idx)
        median_score = scores.median().item()

        old_threshold = self.threshold[layer_idx]
        new_threshold = (1 - self.config.ema_alpha) * old_threshold + self.config.ema_alpha * median_score

        # Hysteresis: only update if change exceeds band
        if abs(new_threshold - old_threshold) > self.config.hysteresis_band:
            self.threshold[layer_idx] = new_threshold

        return self.threshold[layer_idx]

    def assign_tiers(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        budget_fp16: float = 0.25,
        budget_int8: float = 0.25,
        budget_int4: float = 0.25,
    ) -> torch.Tensor:
        """Assign compression tiers based on scores.

        Tier 0: FP16 (full precision) — top budget_fp16 fraction
        Tier 1: INT8 — next budget_int8 fraction
        Tier 2: INT4 — next budget_int4 fraction
        Tier 3: Evicted (with sketch for potential recovery)

        Respects layer-wise pyramid allocation: early layers get more FP16 budget.

        Returns:
            tiers: [kv_len] tensor with values in {0, 1, 2, 3}
        """
        kv_len = scores.shape[0]

        # Layer-wise pyramid: early layers get tighter compression
        # (PyramidKV/Triage analogy: later layers are more sensitive)
        layer_factor = 1.0 + 0.5 * (layer_idx / max(self.num_layers - 1, 1))
        adj_fp16 = min(budget_fp16 * layer_factor, 0.8)
        adj_int8 = budget_int8
        adj_int4 = budget_int4

        # Sort by score (descending)
        sorted_indices = scores.argsort(descending=True)
        tiers = torch.full((kv_len,), 3, device=self.device, dtype=torch.long)  # default: evicted

        n_fp16 = max(int(kv_len * adj_fp16), self.config.pin_first_n)
        n_int8 = int(kv_len * adj_int8)
        n_int4 = int(kv_len * adj_int4)

        tiers[sorted_indices[:n_fp16]] = 0
        tiers[sorted_indices[n_fp16:n_fp16 + n_int8]] = 1
        tiers[sorted_indices[n_fp16 + n_int8:n_fp16 + n_int8 + n_int4]] = 2

        self.tiers[layer_idx] = tiers
        return tiers


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize to [0, 1] range, handling constant tensors."""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return torch.ones_like(x) * 0.5
    return (x - xmin) / (xmax - xmin)
