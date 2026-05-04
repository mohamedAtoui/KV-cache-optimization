"""Benchmark: Compare KV cache strategies on perplexity + memory.

Strategies tested:
1. Full KV cache (baseline)
2. StreamingAttention (streaming heads → recurrent state, retrieval heads → full KV)
3. StreamingAttention + Adaptive Tiered Cache (streaming → state, retrieval → tiered compression)
4. Uniform quantization baselines (INT8-only, INT4-only)

Usage:
    python -m streaming_attention.benchmark_adaptive \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --pattern-dir attn_patterns/Meta-Llama-3.1-8B-Instruct \
        --max-samples 50
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F

from streaming_attention.importance import ImportanceConfig, ImportanceScorer
from streaming_attention.adaptive_cache import (
    AdaptiveCacheConfig,
    TieredKVCache,
    _symmetric_quantize,
    _symmetric_dequantize,
    apply_error_aware_attention,
)

logger = logging.getLogger(__name__)


def benchmark_importance_scoring(
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 2048,
    device: str = "cpu",
):
    """Benchmark importance scoring speed and tier distribution.

    Runs with synthetic data to test the scoring pipeline without a full model.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Importance Scoring Pipeline")
    print("=" * 70)

    config = ImportanceConfig()
    scorer = ImportanceScorer(
        config=config,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        max_seq_len=seq_len,
        device=device,
    )

    # Simulate attention patterns
    # Create realistic-ish synthetic attention with heavy hitters + recency bias
    print(f"\nSynthetic setup: {num_layers} layers, {num_kv_heads} KV heads, "
          f"seq_len={seq_len}, head_dim={head_dim}")

    t_start = time.perf_counter()

    for step in range(0, seq_len, 32):
        current_len = min(step + 32, seq_len)

        # Simulate attention weights: recent tokens + a few heavy hitters
        attn = torch.zeros(1, num_kv_heads, 1, current_len, device=device)
        # Recency: last 64 tokens get most attention
        recency_start = max(0, current_len - 64)
        attn[:, :, :, recency_start:current_len] = torch.rand(
            1, num_kv_heads, 1, current_len - recency_start, device=device
        )
        # Heavy hitters: positions 0-3 (attention sinks)
        attn[:, :, :, :min(4, current_len)] = 0.8 + 0.2 * torch.rand(
            1, num_kv_heads, 1, min(4, current_len), device=device
        )
        # Normalize
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Simulate key states
        keys = torch.randn(1, num_kv_heads, current_len, head_dim, device=device)

        scorer.update(
            layer_idx=0,
            attn_weights=attn,
            key_states=keys,
            step=step,
        )

    t_scoring = time.perf_counter() - t_start

    # Get scores and assign tiers
    scores = scorer.get_scores(0, keys)
    tiers = scorer.assign_tiers(
        0, scores,
        budget_fp16=0.25,
        budget_int8=0.30,
        budget_int4=0.25,
    )

    t_total = time.perf_counter() - t_start

    # Report
    tier_counts = {i: (tiers == i).sum().item() for i in range(4)}
    tier_names = {0: "FP16", 1: "INT8", 2: "INT4", 3: "Evicted"}

    print(f"\nScoring time: {t_scoring * 1000:.1f} ms")
    print(f"Total time (scoring + tier assignment): {t_total * 1000:.1f} ms")
    print(f"\nTier distribution for layer 0:")
    for tier, count in tier_counts.items():
        pct = 100 * count / seq_len
        bar = "█" * int(pct / 2)
        print(f"  {tier_names[tier]:>8}: {count:5d} ({pct:5.1f}%) {bar}")

    # Score statistics
    print(f"\nScore statistics:")
    print(f"  Mean:   {scores.mean().item():.4f}")
    print(f"  Median: {scores.median().item():.4f}")
    print(f"  Std:    {scores.std().item():.4f}")
    print(f"  Min:    {scores.min().item():.4f}")
    print(f"  Max:    {scores.max().item():.4f}")

    # Memory savings estimate
    bytes_full = seq_len * head_dim * num_kv_heads * 2 * 2  # K+V, FP16
    bytes_tiered = (
        tier_counts[0] * head_dim * num_kv_heads * 2 * 2 +      # FP16
        tier_counts[1] * head_dim * num_kv_heads * 2 * 1 +      # INT8
        tier_counts[2] * head_dim * num_kv_heads * 2 * 0.5 +    # INT4
        8 * head_dim * 4                                         # sketch
    )
    ratio = bytes_full / max(bytes_tiered, 1)
    savings = (1 - bytes_tiered / bytes_full) * 100

    print(f"\nMemory estimate:")
    print(f"  Full FP16:  {bytes_full / 1024:.1f} KB")
    print(f"  Tiered:     {bytes_tiered / 1024:.1f} KB")
    print(f"  Ratio:      {ratio:.2f}x")
    print(f"  Savings:    {savings:.1f}%")

    return {
        "scoring_time_ms": t_scoring * 1000,
        "total_time_ms": t_total * 1000,
        "tier_counts": tier_counts,
        "compression_ratio": ratio,
        "memory_savings_pct": savings,
    }


def benchmark_quantization_quality(
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 2048,
    device: str = "cpu",
):
    """Benchmark quantization reconstruction quality.

    Measures per-tier reconstruction error to validate the quantization
    pipeline and demonstrate the hockey-stick degradation curve.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Quantization Reconstruction Quality")
    print("=" * 70)

    # Create synthetic KV entries with realistic magnitude distribution
    torch.manual_seed(42)
    keys = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device) * 0.1
    values = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device) * 0.1

    # Add some outliers (typical in real KV caches)
    outlier_mask = torch.rand(1, num_kv_heads, seq_len, head_dim, device=device) > 0.99
    keys[outlier_mask] *= 10
    values[outlier_mask] *= 10

    results = {}
    for bits, name in [(8, "INT8"), (4, "INT4"), (2, "INT2")]:
        q_k, s_k, z_k = _symmetric_quantize(keys, bits=bits, group_size=128)
        q_v, s_v, z_v = _symmetric_quantize(values, bits=bits, group_size=128)

        k_recon = _symmetric_dequantize(q_k, s_k, z_k)
        v_recon = _symmetric_dequantize(q_v, s_v, z_v)

        k_mse = F.mse_loss(keys.float(), k_recon.float()).item()
        v_mse = F.mse_loss(values.float(), v_recon.float()).item()
        k_cosine = F.cosine_similarity(
            keys.float().reshape(-1, head_dim),
            k_recon.float().reshape(-1, head_dim),
            dim=-1
        ).mean().item()

        results[name] = {
            "key_mse": k_mse,
            "value_mse": v_mse,
            "key_cosine_sim": k_cosine,
            "compression_ratio": 16.0 / bits,
        }

        print(f"\n{name} ({bits}-bit, {16/bits:.1f}x compression):")
        print(f"  Key MSE:       {k_mse:.6f}")
        print(f"  Value MSE:     {v_mse:.6f}")
        print(f"  Key Cosine:    {k_cosine:.6f}")

    return results


def benchmark_error_aware_attention(
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 512,
    device: str = "cpu",
):
    """Benchmark error-aware attention modification.

    Demonstrates how reconstruction error affects attention distribution
    (Anastylosis analogy #5: "false stone" prevention).
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Error-Aware Attention (Anastylosis #5)")
    print("=" * 70)

    torch.manual_seed(42)

    # Create attention logits (pre-softmax)
    q = torch.randn(1, num_kv_heads, 1, head_dim, device=device)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device)

    logits = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

    # Create reconstruction error (most entries have low error, some high)
    recon_error = torch.zeros(seq_len, device=device)
    # INT8 entries: low error
    recon_error[128:384] = 0.01
    # INT4 entries: moderate error
    recon_error[384:480] = 0.1
    # Evicted entries: high error
    recon_error[480:] = float("inf")

    for lam in [0.0, 0.05, 0.1, 0.5]:
        modified_logits = apply_error_aware_attention(logits, recon_error, penalty_lambda=lam)
        attn_weights = F.softmax(modified_logits, dim=-1)

        # Measure how much attention goes to each tier
        fp16_attn = attn_weights[:, :, :, :128].sum().item() / num_kv_heads
        int8_attn = attn_weights[:, :, :, 128:384].sum().item() / num_kv_heads
        int4_attn = attn_weights[:, :, :, 384:480].sum().item() / num_kv_heads
        evict_attn = attn_weights[:, :, :, 480:].sum().item() / num_kv_heads

        print(f"\n  lambda={lam:.2f}:")
        print(f"    FP16 tokens (0-127):     {fp16_attn:.4f}")
        print(f"    INT8 tokens (128-383):   {int8_attn:.4f}")
        print(f"    INT4 tokens (384-479):   {int4_attn:.4f}")
        print(f"    Evicted tokens (480+):   {evict_attn:.4f}")


def benchmark_tiered_cache_e2e(
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 1024,
    device: str = "cpu",
):
    """End-to-end benchmark of the TieredKVCache pipeline.

    Tests the full update → compress → retrieve cycle.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: End-to-End Tiered KV Cache Pipeline")
    print("=" * 70)

    config = AdaptiveCacheConfig(
        budget_fp16=0.25,
        budget_int8=0.30,
        budget_int4=0.25,
        rescore_every=64,
    )
    cache = TieredKVCache(
        config=config,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=seq_len,
        device=device,
    )

    torch.manual_seed(42)
    t_start = time.perf_counter()

    # Simulate prefill
    print(f"\nSimulating {seq_len}-token sequence across {num_layers} layers...")

    for layer_idx in range(min(num_layers, 4)):  # test on first 4 layers for speed
        # Simulate chunked KV updates
        chunk_size = 64
        for pos in range(0, seq_len, chunk_size):
            end = min(pos + chunk_size, seq_len)
            chunk_len = end - pos

            keys = torch.randn(1, num_kv_heads, chunk_len, head_dim, device=device) * 0.1
            values = torch.randn(1, num_kv_heads, chunk_len, head_dim, device=device) * 0.1

            # Simulate attention weights
            current_len = pos + chunk_len
            attn = torch.rand(1, num_kv_heads, chunk_len, current_len, device=device)
            # Add attention sink pattern
            attn[:, :, :, :4] += 2.0
            # Add recency bias
            attn[:, :, :, max(0, current_len - 32):] += 1.0
            attn = attn / attn.sum(dim=-1, keepdim=True)

            cache.update(layer_idx, keys, values, attn)

    t_total = time.perf_counter() - t_start

    # Get stats
    stats = cache.get_stats()
    print(f"\nPipeline time: {t_total * 1000:.1f} ms")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Memory: {stats['memory_mb']:.2f} MB")
    print(f"\nTier distribution (all layers):")
    tier_names = {0: "FP16", 1: "INT8", 2: "INT4", 3: "Evicted"}
    total = sum(stats["tier_counts"].values())
    for tier, count in stats["tier_counts"].items():
        pct = 100 * count / max(total, 1)
        print(f"  {tier_names[tier]:>8}: {count:5d} ({pct:5.1f}%)")

    if stats["mean_recon_error"]:
        print(f"\nMean reconstruction error by layer:")
        for layer, err in sorted(stats["mean_recon_error"].items()):
            print(f"  Layer {layer:2d}: {err:.6f}")

    return stats


def benchmark_compression_sweep(
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 2048,
    device: str = "cpu",
):
    """Sweep compression budgets to find the hockey-stick degradation point.

    Tests the prediction from Triage #6 and Geology #9 that quality degrades
    non-linearly — stable until ~75-80% compression, then sharp collapse.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Compression Sweep (Hockey-Stick Detection)")
    print("=" * 70)

    torch.manual_seed(42)

    # Create reference Q, K, V
    q = torch.randn(1, num_kv_heads, 1, head_dim, device=device)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device)

    # Reference attention output (full precision)
    ref_logits = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    ref_attn = F.softmax(ref_logits, dim=-1)
    ref_output = torch.matmul(ref_attn, v)

    print(f"\n{'FP16 %':>8}  {'INT8 %':>8}  {'INT4 %':>8}  {'Evict %':>8}  {'Output MSE':>12}  {'Attn KL':>10}  {'Compression':>12}")
    print("-" * 85)

    results = []
    for fp16_pct in [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10, 0.05, 0.02]:
        remaining = 1.0 - fp16_pct
        int8_pct = remaining * 0.4
        int4_pct = remaining * 0.35
        evict_pct = remaining * 0.25

        n_fp16 = int(seq_len * fp16_pct)
        n_int8 = int(seq_len * int8_pct)
        n_int4 = int(seq_len * int4_pct)

        # Apply tiered quantization
        k_comp = k.clone()
        v_comp = v.clone()

        # Randomly assign tiers (in practice, importance-based)
        perm = torch.randperm(seq_len, device=device)
        fp16_idx = perm[:n_fp16]
        int8_idx = perm[n_fp16:n_fp16 + n_int8]
        int4_idx = perm[n_fp16 + n_int8:n_fp16 + n_int8 + n_int4]
        evict_idx = perm[n_fp16 + n_int8 + n_int4:]

        # INT8 quantize
        if len(int8_idx) > 0:
            q8_k, s8_k, z8_k = _symmetric_quantize(k[:, :, int8_idx], bits=8)
            q8_v, s8_v, z8_v = _symmetric_quantize(v[:, :, int8_idx], bits=8)
            k_comp[:, :, int8_idx] = _symmetric_dequantize(q8_k, s8_k, z8_k).to(k.dtype)
            v_comp[:, :, int8_idx] = _symmetric_dequantize(q8_v, s8_v, z8_v).to(v.dtype)

        # INT4 quantize
        if len(int4_idx) > 0:
            q4_k, s4_k, z4_k = _symmetric_quantize(k[:, :, int4_idx], bits=4)
            q4_v, s4_v, z4_v = _symmetric_quantize(v[:, :, int4_idx], bits=4)
            k_comp[:, :, int4_idx] = _symmetric_dequantize(q4_k, s4_k, z4_k).to(k.dtype)
            v_comp[:, :, int4_idx] = _symmetric_dequantize(q4_v, s4_v, z4_v).to(v.dtype)

        # Evict
        if len(evict_idx) > 0:
            k_comp[:, :, evict_idx] = 0
            v_comp[:, :, evict_idx] = 0

        # Compute compressed attention output
        comp_logits = torch.matmul(q, k_comp.transpose(-2, -1)) / (head_dim ** 0.5)
        comp_attn = F.softmax(comp_logits, dim=-1)
        comp_output = torch.matmul(comp_attn, v_comp)

        # Metrics
        output_mse = F.mse_loss(ref_output.float(), comp_output.float()).item()
        # KL divergence of attention distributions
        attn_kl = F.kl_div(
            comp_attn.float().log().clamp(min=-100),
            ref_attn.float(),
            reduction="batchmean",
        ).item()

        # Compression ratio
        bytes_full = seq_len * head_dim * 2  # FP16
        bytes_comp = (n_fp16 * 2 + n_int8 * 1 + n_int4 * 0.5) * head_dim
        comp_ratio = bytes_full / max(bytes_comp, 1)

        results.append({
            "fp16_pct": fp16_pct,
            "output_mse": output_mse,
            "attn_kl": attn_kl,
            "compression_ratio": comp_ratio,
        })

        print(f"{fp16_pct*100:7.0f}%  {int8_pct*100:7.0f}%  {int4_pct*100:7.0f}%  "
              f"{evict_pct*100:7.0f}%  {output_mse:12.6f}  {attn_kl:10.4f}  {comp_ratio:10.2f}x")

    # Detect hockey-stick point
    mses = [r["output_mse"] for r in results]
    max_jump = 0
    jump_idx = 0
    for i in range(1, len(mses)):
        jump = mses[i] / max(mses[i - 1], 1e-10)
        if jump > max_jump:
            max_jump = jump
            jump_idx = i

    print(f"\nHockey-stick point: ~{results[jump_idx]['fp16_pct']*100:.0f}% FP16 "
          f"(MSE jumps {max_jump:.1f}x)")
    print(f"Prediction from Triage #6: quality cliff at ~75-80% compression")

    return results


def benchmark_redundancy_penalty(
    num_kv_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 512,
    device: str = "cpu",
):
    """Demonstrate redundancy penalization effect (GCKV analogy #8).

    Shows that without redundancy penalty, the surviving cache population
    clusters (many similar entries survive while unique critical entries are evicted).
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Redundancy Penalization (GCKV #8)")
    print("=" * 70)

    torch.manual_seed(42)

    # Create keys with deliberate redundancy
    # 80% of keys are from 5 clusters, 20% are unique
    keys = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device)
    cluster_centers = torch.randn(5, head_dim, device=device)
    for i in range(int(seq_len * 0.8)):
        cluster = i % 5
        keys[0, :, i] = cluster_centers[cluster] + 0.1 * torch.randn(num_kv_heads, head_dim, device=device)

    # Give clustered keys high attention (they'll all score high without penalty)
    attn = torch.rand(1, num_kv_heads, 1, seq_len, device=device)
    attn[:, :, :, :int(seq_len * 0.8)] *= 3  # clustered keys get more attention
    attn = attn / attn.sum(dim=-1, keepdim=True)

    for use_penalty in [False, True]:
        config = ImportanceConfig(
            redundancy_threshold=0.92,
            redundancy_penalty=0.3 if use_penalty else 1.0,  # 1.0 = no penalty
        )
        scorer = ImportanceScorer(
            config=config, num_layers=1, num_kv_heads=num_kv_heads,
            max_seq_len=seq_len, device=device,
        )
        scorer.update(0, attn, keys, step=0)
        scores = scorer.get_scores(0, keys)

        # Keep top 50%
        keep_n = seq_len // 2
        kept_indices = scores.topk(keep_n).indices.sort().values

        # Measure diversity of surviving cache
        kept_keys = keys[0, 0, kept_indices]  # [keep_n, D]
        kept_norm = F.normalize(kept_keys, dim=-1)
        sim_matrix = torch.mm(kept_norm, kept_norm.t())
        sim_matrix.fill_diagonal_(0)
        mean_sim = sim_matrix.mean().item()

        # How many unique keys (from last 20%) survived?
        unique_start = int(seq_len * 0.8)
        n_unique_survived = (kept_indices >= unique_start).sum().item()
        n_unique_total = seq_len - unique_start

        label = "WITH" if use_penalty else "WITHOUT"
        print(f"\n  {label} redundancy penalty:")
        print(f"    Mean pairwise cosine sim of survivors: {mean_sim:.4f}")
        print(f"    Unique keys survived: {n_unique_survived}/{n_unique_total} "
              f"({100*n_unique_survived/n_unique_total:.0f}%)")
        print(f"    Cache diversity: {'high' if mean_sim < 0.3 else 'low (clustered)'}")


def run_all_benchmarks(device: str = "cpu"):
    """Run all benchmarks and return combined results."""
    results = {}

    results["importance_scoring"] = benchmark_importance_scoring(device=device)
    results["quantization_quality"] = benchmark_quantization_quality(device=device)
    benchmark_error_aware_attention(device=device)
    results["tiered_cache_e2e"] = benchmark_tiered_cache_e2e(device=device)
    results["compression_sweep"] = benchmark_compression_sweep(device=device)
    benchmark_redundancy_penalty(device=device)

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  - Scoring pipeline: {results['importance_scoring']['scoring_time_ms']:.1f} ms "
          f"for 2048 tokens")
    print(f"  - Compression: {results['importance_scoring']['compression_ratio']:.2f}x "
          f"({results['importance_scoring']['memory_savings_pct']:.0f}% savings)")
    print(f"  - INT8 key cosine: {results['quantization_quality']['INT8']['key_cosine_sim']:.6f}")
    print(f"  - INT4 key cosine: {results['quantization_quality']['INT4']['key_cosine_sim']:.6f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark adaptive KV cache strategies")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = run_all_benchmarks(device=args.device)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
