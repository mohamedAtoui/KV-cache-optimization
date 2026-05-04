---
title: "KV-Cache Optimization: kv\_bench & kv2state Technical Guide"
author: "Mohamed Atoui"
date: "March 2025"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.15
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{float}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{KV-Cache Optimization Guide}
  - \fancyhead[R]{\thepage}
  - \fancyfoot[C]{}
---

\newpage

# Introduction

This document explains the two main packages in the KV-cache optimization project:

- **kv\_bench** — A benchmarking framework that compares KV-cache compression strategies by measuring perplexity (PPL) on WikiText-2.
- **kv2state** — A library that converts streaming attention heads' KV cache into fixed-size recurrent state, with supporting modules for head classification, calibration, and stratigraphic compression.

The target model is **Llama-3.1-8B-Instruct** (GQA: 32 layers, 32 query heads, 8 KV heads, head\_dim=128). Benchmarks run on **Llama-3.2-1B-Instruct** (16 layers, 32 query heads, 8 KV heads, head\_dim=64) for fast iteration.

## Why KV-Cache Compression Matters

During autoregressive generation, transformers store **Key** and **Value** tensors for every previous token in every layer. For Llama-3.1-8B with 8192 tokens:

$$\text{KV cache} = 2 \times 32 \times 8 \times 8192 \times 128 \times 2\ \text{bytes} = 1\ \text{GB}$$

This grows linearly with sequence length and dominates GPU memory at long contexts. Every strategy in this project aims to reduce this memory while preserving generation quality.

**Key principle:** The model weights are never modified. All strategies operate on the KV cache (runtime activations), not on learned parameters. No retraining is needed.

\newpage

# Part I: kv\_bench — Benchmarking Framework

## Architecture Overview

```
CLI (__main__.py)
  |
  v
BenchmarkRunner (runner.py)
  |-- _load_model()          Load HF model + tokenizer once
  |-- _load_dataset()        Tokenize WikiText-2 once
  |-- for each strategy:
        |-- strategy.setup()
        |-- _evaluate_perplexity()   Sliding window, two-pass
        |-- _measure_memory()
        |-- strategy.teardown()
  |
  v
report.py  -->  Table / JSON / Markdown
```

## The Strategy Interface

Every KV-cache strategy implements `KVCacheStrategy` (defined in `strategy.py`):

| Method | Purpose |
|--------|---------|
| `setup(model, config, device)` | Patch the model if needed |
| `teardown(model)` | Restore original model |
| `memory_bytes(seq_len, config)` | Analytical memory estimate |
| `needs_attention_weights()` | Does strategy need attention scores? |
| `on_step(layer_idx, attn_weights)` | Receive attention weights (pass 1) |
| `get_keep_mask(seq_len, device)` | Which tokens to keep (eviction mask) |
| `get_zone_masks(seq_len, device)` | Per-head quantization zone assignments |
| `reset()` | Clear state between eval windows |

## Two-Pass Evaluation

Strategies that need attention weights (H2O, SnapKV, Stratigraphic, etc.) trigger a two-pass evaluation per sliding window:

**Pass 1 — Observe:**

- Run the model with `output_attentions=True` and `use_cache=False`.
- Feed attention weights to `strategy.on_step()` for each layer.
- Strategy computes importance scores and returns a `keep_mask` (eviction) and/or `zone_masks` (quantization).

**Pass 2 — Evaluate with compression:**

- Free Pass 1 outputs to avoid OOM.
- Build a 4D attention mask: causal mask + eviction (block columns of evicted tokens).
- Install quantization hooks (post-RoPE for keys, v\_proj hook for values).
- Run the model again with the modified mask — this forward pass reflects the actual compression error.
- Compute loss on non-overlapping tokens only.
- Clean up: remove hooks, restore original attention forwards.

**Why two passes?** We need attention weights to decide *what* to compress (pass 1), then we need to measure *how much* that compression hurts (pass 2). The baseline strategy skips both passes — it just runs the model normally.

## Quantization Simulation (`quant_sim.py`)

### Why Post-RoPE Matters

Real KV caches store keys **after** Rotary Position Embeddings (RoPE). Pre-RoPE keys are smooth linear projections that INT8 (127 levels) can represent losslessly. Post-RoPE keys contain high-frequency rotary components that are harder to quantize.

If you quantize pre-RoPE keys, INT8 shows +0.00 PPL — the benchmark would be lying.

### How It Works

**Keys (post-RoPE):** We replace `self_attn.forward` with a wrapper that:

1. Projects Q, K, V via the original linear layers
2. Reshapes to `[B, num_heads, T, head_dim]`
3. Applies RoPE (using `_apply_rotary_pos_emb` from `kv2state`)
4. **Quantizes K per zone\_mask** — per-channel (KIVI-style, one scale per head\_dim channel)
5. Expands KV for GQA (`repeat_interleave`)
6. Runs scaled dot-product attention with the causal + eviction mask
7. Projects output via `o_proj`

**Values (no RoPE):** A forward hook on `v_proj` applies per-token quantization. Values never receive RoPE, so the hook location is already correct.

### Quantization Functions

| Function | Granularity | Levels | Used for |
|----------|-------------|--------|----------|
| `simulate_int8_per_channel` | Per head\_dim channel (dim=-2) | 127 | Keys |
| `simulate_int4_per_channel` | Per head\_dim channel (dim=-2) | 7 | Keys |
| `simulate_int8` | Per token (dim=-1) | 127 | Values |
| `simulate_int4` | Per token (dim=-1) | 7 | Values |

All use symmetric quantization: `scale = max(|x|) / levels`, `x_q = round(x / scale) * scale`.

\newpage

## Strategies Compared

### 1. FullKV Baseline

- **What:** No compression. Standard FP16/BF16 KV cache.
- **Memory:** `2 × num_layers × num_kv_heads × seq_len × head_dim × 2 bytes`
- **Source:** Standard transformer inference.
- **Purpose:** Reference perplexity that all other strategies are measured against.

### 2. H2O (Heavy-Hitter Oracle)

- **What:** Eviction-based. Keeps a budget (default 50%) of KV entries.
- **How:** Accumulates attention scores across all layers. Keeps:
  - Attention sinks (first 4 tokens) — always high-attention initial tokens
  - Recent tokens (last 64) — current context window
  - Heavy hitters — remaining budget filled by highest cumulative attention
- **Evicts:** Everything else — those columns are masked with $-\infty$ in the attention matrix.
- **Source:** Zhang et al., *"H2O: Heavy-Hitter Oracle"*, NeurIPS 2023.

### 3. SnapKV

- **What:** Eviction-based, smarter scoring than H2O.
- **How:** Uses attention patterns from only the last 64 tokens (observation window) to score all KV positions. Smooths scores with average pooling (kernel=5) to capture clusters of important tokens, not just isolated peaks.
- **Key insight:** Recent query attention patterns predict which KV entries will be needed during generation.
- **Source:** Li et al., *"SnapKV: LLM Knows What You Are Looking For Before Generation"*, 2024.

### 4. INT8-all / INT4-all (Uniform Quantization)

- **What:** Quantization-only, no eviction. Every token in every head gets the same bit-width.
- **How:** Returns `get_keep_mask() = all True` (keep everything) and `get_zone_masks()` with uniform ZONE\_INT8 or ZONE\_INT4 across all positions.
- **Key detail:** Uses KIVI-style per-channel quantization for keys (one scale per head\_dim channel shared across tokens) and per-token quantization for values.
- **Source:** Based on KIVI (Liu et al., ICML 2024).

### 5. Adaptive Tiered

- **What:** Eviction + analytical memory modeling.
- **How:** Scores tokens by cumulative attention (like H2O), keeps top 80% across three tiers (25% FP16 + 30% INT8 + 25% INT4), evicts the bottom 20%.
- **Limitation:** Only evicts in practice — the quantization tiers are used for memory estimation but don't inject actual quant noise. This means its PPL reflects eviction quality only.
- **Status:** Superseded by Stratigraphic.

### 6. Stratigraphic (Ours)

- **What:** Combined eviction + mixed-precision quantization with per-head, per-layer, per-position granularity.
- **How:** See Section 2.4 for full details.
- **Source:** Original contribution.

\newpage

## The Stratigraphic Strategy — Detailed

### Geological Metaphor

The name comes from stratigraphy — the study of rock layers. Each token position in the KV cache is assigned to a "geological zone" based on its importance:

| Zone | Name | Precision | Meaning |
|------|------|-----------|---------|
| 0 | Surface | FP16 | Recent/critical tokens — full precision |
| 1 | Shallow | INT8 | Moderately important — 8-bit quantization |
| 2 | Deep | INT4 | Low importance — 4-bit quantization |
| 3 | Bedrock | Evicted | Irrelevant — removed from attention entirely |

### Five Key Components

**1. Per-head, per-token zone assignment:**
Each KV head at each token position gets its own zone. Zone mask shape: `[num_kv_heads, seq_len]`. This allows head 3 to keep position 42 at FP16 while head 7 compresses it to INT4.

**2. Monotonic downgrade:**
Tokens can only move to deeper compression over time (FP16 → INT8 → INT4 → EVICT), never back up. The `HeadZoneAssigner` tracks zone history per (layer, head) pair.

**3. Inverse layer budget:**
Early layers compress more aggressively; late layers preserve more at FP16. The formula:

$$\text{fp16\_frac}(l) = \text{zone\_surface} \times \left[(1 - \lambda) + \lambda \cdot \frac{l}{L-1}\right]$$

With $\lambda = 0.6$ and zone\_surface = 0.25: layer 0 gets 10% FP16, layer 31 gets 25% FP16. This reflects the finding that later layers are more sensitive to cache quality.

**4. Anchor detection (Stylolites):**
The `AnchorDetector` identifies tokens with cumulative attention $\geq$ 99th percentile. These "stylolite" positions are pinned at FP16 regardless of the zone budget. Additionally, sink tokens (first 4) and recent tokens (last 64) are always anchored.

**5. Combined eviction + mixed-precision quantization:**
Unlike H2O/SnapKV (eviction-only) or INT8/INT4 (quant-only), Stratigraphic applies both. The least important tokens are evicted entirely, while the rest get progressively compressed based on importance.

### Attention Scoring

Per-head attention scores are computed with GQA awareness:

1. Attention weights `[B, num_q_heads, Q, KV]` are reshaped to `[B, num_kv_heads, group_size, Q, KV]`
2. Averaged within each GQA group, summed over batch and queries → `[num_kv_heads, KV]`
3. **Causal bias correction:** Normalized by the number of attending queries per position (position $j$ is attended by $KV - j$ queries)

### Memory Model

```
Total bytes = Σ over layers:
  num_kv_heads × (
    n_fp16 × head_dim × 2 × 2       (K+V at 2 bytes each)
  + n_int8  × head_dim × 1 × 2       (K+V at 1 byte each)
  + n_int4  × (head_dim/2) × 2       (K+V at 0.5 bytes each)
  )
```

Where `n_fp16`, `n_int8`, `n_int4` vary per layer according to the inverse layer budget.

\newpage

## Benchmark Results (Llama-3.2-1B, WikiText-2, A100)

| Strategy | PPL | $\Delta$PPL | Mem MB | Compression | Prefill ms |
|----------|-----|-------------|--------|-------------|------------|
| FullKV (baseline) | 11.15 | — | 128.0 | 1.0x | 64.5 |
| **Stratigraphic** | **11.18** | **+0.04** | **58.2** | **2.2x** | 1529.2 |
| INT8-all | 11.15 | +0.00 | 68.0 | 1.9x | 833.9 |
| INT4-all | 13.37 | +2.22 | 36.0 | 3.6x | 749.2 |
| SnapKV (50%) | 17.94 | +6.79 | 64.0 | 2.0x | 448.3 |
| H2O (50%) | 107.72 | +96.57 | 64.0 | 2.0x | 459.4 |
| Adaptive Tiered | 30.49 | +19.35 | 59.2 | 2.2x | 1026.9 |

### Key Takeaways

- **Stratigraphic dominates the quality-compression frontier:** +0.04 PPL at 2.2x compression. No other strategy achieves comparable quality at this compression level.
- **INT8 is essentially lossless** on this 1B model — 127 per-channel levels capture post-RoPE keys with negligible error.
- **INT4 shows real degradation** (+2.22) — 7 levels cannot represent the rotary frequency patterns accurately.
- **H2O fails catastrophically** at 50% budget — evicting half the cache destroys too much context.
- **SnapKV is much better than H2O** (+6.79 vs +96.57) thanks to smarter observation-window scoring, but still far worse than Stratigraphic.
- **Prefill latency** is higher for two-pass strategies. This is an evaluation cost, not an inference cost — real deployment would use a single pass with online zone assignment.

\newpage

# Part II: kv2state — KV Cache to Recurrent State

## Core Idea

Some attention heads are "streaming" — they attend diffusely across all positions with high entropy. These heads don't need a full KV cache; their behavior can be approximated by a fixed-size recurrent state:

$$S_t = \lambda \cdot S_{t-1} + v_t \cdot k_t^\top$$

where $S_t \in \mathbb{R}^{d \times d}$ is the state matrix, $\lambda \in [0, 1]$ is a learned decay factor, and $k_t, v_t$ are the key and value at position $t$.

**Output:** For a query $q_t$:

$$o_t = \frac{S_t \cdot q_t}{z_t \cdot q_t}, \quad z_t = \lambda \cdot z_{t-1} + k_t$$

This is **linear attention with exponential decay** — constant memory per head ($d^2 + d$ values) regardless of sequence length, versus $O(T \cdot d)$ for standard KV cache.

## Head Classification (`head_classifier.py`)

Not all heads can be converted. The system classifies each KV head as:

- **Retrieval heads** (low entropy, peaked attention): Keep full KV cache — these heads look up specific tokens.
- **Streaming heads** (high entropy, diffuse attention): Convert to recurrent state — these heads aggregate broad context.

### Classification Methods

**DuoAttention patterns** (preferred): Load pre-computed gate values from TSV files (shape `[num_layers, num_kv_heads]`). Gate $\geq 0.5$ = retrieval, $< 0.5$ = streaming. These patterns come from the MIT HAN Lab DuoAttention project.

**Entropy-based** (fallback): Run 50 batches through the model, compute per-head attention entropy, use Otsu thresholding to find the bimodal split.

The result is a `HeadClassification` object with a boolean mask `[num_layers, num_kv_heads]`.

## Hybrid Attention (`hybrid_attention.py`)

The model is monkey-patched so each attention layer splits its heads:

```
Input hidden_states
    |
    v
Q, K, V projections (unchanged)
    |
    v
Apply RoPE (unchanged)
    |
    +---> Retrieval heads: standard softmax attention + KV cache
    |         |
    |         v
    |     GQA expansion -> QK^T -> causal mask -> softmax -> AV
    |
    +---> Streaming heads: recurrent state attention
              |
              v
          Feature map (ELU+1) on K
              |
              v
          State update: S_t = lambda * S_{t-1} + v_t * k_t^T
              |
              v
          Output: o_t = S_t * q_t / z_t * q_t
    |
    v
Concatenate all head outputs -> o_proj
```

### Feature Map

Streaming heads use the ELU+1 feature map: $\phi(x) = \text{ELU}(x) + 1$. This maps arbitrary signed inputs to strictly positive values, ensuring $Q \cdot K$ products are non-negative for stable state accumulation. This is from Katharopoulos et al. (2020), "Transformers are RNNs."

### Parallel Prefill

For prefill (multiple tokens at once), `DecayedLinearState.parallel_forward` uses chunk-wise computation:

- **Intra-chunk:** Causal linear attention within each chunk of size 64. Attention weight $A[i,j] = \lambda^{i-j} \cdot (q_i \cdot k_j)$ for $j \leq i$.
- **Inter-chunk:** State from previous chunks contributes $\lambda^{i+1} \cdot S \cdot q_i$ to position $i$.
- This avoids materializing the full $T \times T$ attention matrix.

### Memory Savings

For Llama-3.1-8B with 50% streaming heads:

- **Standard KV cache per streaming head:** $T \times d \times 2$ values (K+V), grows with sequence length
- **Recurrent state per streaming head:** $d^2 + d$ values, **constant** regardless of sequence length

At $T = 8192$, $d = 128$: KV cache = 2M values vs. state = 16.5K values per head — **121x reduction per streaming head**.

## State Attention (`state_attention.py`)

### DecayedLinearState

The core recurrent module. Key properties:

- **Decay $\lambda$:** Controls how quickly history fades. Stored as `log_decay` parameter with sigmoid activation → $\lambda \in [0, 1]$.
- **Learnable:** During calibration (Stage 1), $\lambda$ is optimized per head. During inference, it's frozen.
- **Initial state:** Zero matrix and zero normalizer.

### StateCache

Manages per-head states for the full model:

- `states[(layer, head)]` → state tensor $[B, d, d]$
- `zs[(layer, head)]` → normalizer tensor $[B, d]$
- `reset()` clears all states between evaluation windows

## Calibration (`calibration.py`)

Zero-shot conversion fails catastrophically (PPL explodes to >1000). Two-stage calibration is required:

### Stage 1: Decay Alignment

- **Objective:** MSE between teacher (standard softmax) and student (state attention) outputs, per streaming head.
- **Trainable parameters:** Only `log_decay` per streaming head (all other params frozen).
- **Optimizer:** Adam, lr=0.02 with linear warmup + cosine decay.
- **Data:** SlimPajama-6B streaming dataset.

### Stage 2: LoRA Fine-Tuning

- **Objective:** Standard next-token prediction loss (cross-entropy).
- **Trainable parameters:** LoRA adapters on `q_proj` and `v_proj` (rank=16, alpha=32). Calibrated decay is frozen.
- **Optimizer:** AdamW, lr=2e-4 with weight decay 0.01.
- **Data:** Same SlimPajama-6B.

After calibration, the model uses hybrid attention during inference without further training.

## Importance Scoring (`importance.py`)

The `ImportanceScorer` computes per-token importance using 5 independent signals:

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Cumulative attention | 35% | How much total attention a position has received |
| Recency | 20% | Exponential decay from sequence end (halflife=256) |
| Cross-head variance | 15% | How differently heads attend to this position |
| Distinctiveness | 20% | 1 − max cosine similarity to neighboring keys |
| Structural salience | 10% | BOS tokens, separators, high-entropy positions |

Additionally applies a **redundancy penalty**: if two high-scoring tokens are very similar (cosine > 0.92), the weaker one gets penalized by 0.5x.

## Adaptive Cache (`adaptive_cache.py`)

The `TieredKVCache` manages retrieval heads' KV cache with progressive compression:

- **FP16 tier** (25%): Highest importance — full precision
- **INT8 tier** (30%): Moderate importance — 8-bit per-group quantization
- **INT4 tier** (25%): Lower importance — 4-bit per-group quantization
- **Evicted** (20%): Lowest importance — replaced by low-rank SVD sketch (rank=8)

Re-scoring happens every 32 tokens. Error-aware attention modifies logits: $\tilde{a}_{ij} = QK^T - \lambda \cdot \epsilon_j$, where $\epsilon_j$ is the reconstruction error for position $j$.

\newpage

# Part III: Novelty Analysis

## Literature Landscape

### Eviction-Only Systems

| System | Per-head | Layer budget | Year |
|--------|----------|-------------|------|
| H2O | No | No | NeurIPS 2023 |
| SnapKV | No | No | 2024 |
| FastGen | Yes | No | ICLR 2024 |
| PyramidKV | No | Yes | 2024 |

These only remove tokens — no quantization. Quality degrades quickly past 50% eviction.

### Quantization-Only Systems

| System | Per-head | Layer budget | Year |
|--------|----------|-------------|------|
| KIVI | No | No | ICML 2024 |
| KVQuant | No | No | NeurIPS 2024 |
| PM-KVQ | No | Yes | 2025 |

These only compress — no eviction. Cannot reclaim memory from truly irrelevant tokens.

### Combined Eviction + Quantization

| System | Per-head | Monotonic | Layer budget | Anchors | Zones |
|--------|----------|-----------|-------------|---------|-------|
| LeanKV | Yes | Yes | No | No | 3 |
| ARKV | No | No | Yes | No | 3 |
| MiniKV | No | No | Yes | No | 2 |
| **Stratigraphic** | **Yes** | **Yes** | **Yes** | **Yes** | **4** |

**Closest competitor: LeanKV** (Dec 2024) — has per-head zones with monotonic downgrade + combined eviction/quant. But lacks inverse layer budget, anchor detection, and has only 3 precision levels.

## What Makes Stratigraphic Novel

No published system combines all five components:

1. **Per-head, per-token zone assignment** — each head decides independently
2. **Monotonic downgrade** — zones only go deeper, preventing oscillation
3. **Inverse layer budget** — early layers compress more, late layers preserve
4. **Anchor detection** — attention sinks pinned at FP16
5. **4-zone mixed precision** — FP16 / INT8 / INT4 / EVICT in one framework

\newpage

# Part IV: How Everything Connects

## Package Dependency Map

```
kv2state/
  stratigraphic.py    -- Zone constants, config, assigner, anchor detector
  head_classifier.py  -- DuoAttention pattern loading, entropy classification
  hybrid_attention.py -- Model patching, RoPE utilities
  state_attention.py  -- Recurrent state (DecayedLinearState, StateCache)
  importance.py       -- Multi-signal importance scoring
  adaptive_cache.py   -- Tiered KV cache for retrieval heads
  calibration.py      -- Two-stage calibration (decay + LoRA)
  eval_perplexity.py  -- Sliding-window perplexity evaluation

kv_bench/
  __main__.py         -- CLI entry point, strategy registry
  runner.py           -- BenchmarkRunner: two-pass eval orchestration
  quant_sim.py        -- Post-RoPE key quant + v_proj hooks
  strategy.py         -- KVCacheStrategy ABC + StrategyResult
  device_config.py    -- GPU auto-detection
  report.py           -- JSON / Markdown / table output
  strategies/
    baseline.py         imports: strategy
    h2o.py              imports: strategy
    snapkv.py           imports: strategy
    uniform_quant.py    imports: strategy, kv2state.stratigraphic (zones)
    adaptive_tiered.py  imports: strategy
    stratigraphic.py    imports: strategy, kv2state.stratigraphic
    kv2state.py         imports: strategy, kv2state.*
    hybrid.py           imports: strategy, kv2state.*
```

## Running the Benchmark

```bash
python -m kv_bench \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategies baseline stratigraphic int8 int4 h2o snapkv \
  --max-samples 100 \
  --output results.json \
  --markdown results.md \
  -v
```

The CLI resolves strategy names via `STRATEGY_REGISTRY`, auto-detects the GPU, loads the model once, and runs each strategy sequentially with the two-pass evaluation loop.

## Device Configuration

The `auto_detect()` function configures parameters based on GPU VRAM:

| GPU | VRAM | batch\_size | max\_seq\_len | stride |
|-----|------|------------|--------------|--------|
| A100-80GB / H100 | $\geq$70 GB | 4 | 8192 | 2048 |
| A100-40GB | $\geq$30 GB | 2 | 4096 | 1024 |
| L4 | $<$30 GB | 1 | 2048 | 512 |
| CPU | — | 1 | 256 | 128 |

\newpage

# Appendix A: Quantization Details

## Per-Channel vs Per-Token

**Per-channel** (used for keys): One scale factor per head\_dim channel, shared across all token positions. Reduces along dim=-2 (tokens).

$$\text{scale}_c = \frac{\max_t |K_{t,c}|}{127}, \quad K_{t,c}^q = \text{round}\left(\frac{K_{t,c}}{\text{scale}_c}\right) \times \text{scale}_c$$

**Per-token** (used for values): One scale factor per token, shared across all channels. Reduces along dim=-1 (channels).

$$\text{scale}_t = \frac{\max_c |V_{t,c}|}{127}, \quad V_{t,c}^q = \text{round}\left(\frac{V_{t,c}}{\text{scale}_t}\right) \times \text{scale}_t$$

## Why Per-Channel for Keys?

After RoPE, different head\_dim channels correspond to different rotary frequencies. Low-frequency channels have large magnitudes; high-frequency channels have small magnitudes. Per-token quantization would use a single scale dominated by the large channels, wasting precision on small ones. Per-channel gives each frequency its own scale — this is the key insight from KIVI (Liu et al., 2024).

# Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **KV cache** | Stored Key and Value tensors from previous tokens, used for autoregressive attention |
| **GQA** | Grouped-Query Attention — multiple query heads share one KV head |
| **RoPE** | Rotary Position Embedding — encodes position via rotation of Q/K vectors |
| **PPL** | Perplexity — $e^{\text{avg cross-entropy loss}}$; lower is better |
| **$\Delta$PPL** | Perplexity increase relative to baseline |
| **KIVI** | Per-channel key, per-token value quantization scheme (Liu et al., ICML 2024) |
| **Attention sinks** | First few tokens that receive disproportionately high attention |
| **Stylolite** | (Geological term) pressure dissolution surface; here, anchored high-attention positions |
| **Zone mask** | `[num_kv_heads, seq_len]` tensor assigning each position to FP16/INT8/INT4/EVICT |
| **DuoAttention** | MIT HAN Lab method for classifying heads as retrieval vs streaming |
| **ELU+1** | Feature map $\phi(x) = \text{ELU}(x) + 1$ for non-negative linear attention |
