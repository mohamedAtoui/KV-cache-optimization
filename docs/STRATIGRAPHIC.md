# Stratigraphic KV-Cache Compression

A KV-cache compression strategy inspired by geological stratigraphy. Tokens are
treated as sedimentary deposits that undergo progressive compression over time —
moving from full precision to quantized to evicted, but never back.

The contribution is not any single technique (most components exist individually in
prior work) but the **unified framework** that integrates them under a coherent
geological design principle, plus **semantic topic-shift detection** for anchor
pinning, which is novel to KV-cache management.

## Motivation

Existing KV-cache methods fall into three camps:

1. **Eviction-only** (H2O, SnapKV, StreamingLLM) — binary keep/drop decisions. A
   token is either at full precision or completely lost.
2. **Quantization-only** (KIVI, KVQuant) — all tokens quantized uniformly. No
   selective attention to token importance.
3. **Mixed-precision** (MiKV, ZipCache, PM-KVQ) — important tokens get higher
   precision. Closest to our approach, but each addresses only a subset of the
   design space (see Related Work below).

No existing method integrates multi-tier quantization, per-layer budgets, per-head
zone assignment, monotonic constraints, and semantic anchor detection into a single
coherent framework. Stratigraphic KV-Cache fills this gap.

## The geological metaphor

The KV cache is modelled as a geological column. Tokens are "sediment" deposited over
the sequence. As context grows, older tokens are buried under pressure — pushed into
deeper compression zones, just as rock undergoes diagenesis (compaction and mineral
transformation with depth).

```
Surface   ░░░░░░░░░░░░░░░░  FP16  — fresh, highest-attention tokens
Shallow   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  INT8  — compacted, still useful
Deep      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  INT4  — heavily compressed
Bedrock   ████████████████  Evict — only a low-rank SVD sketch survives
```

This metaphor is not just aesthetic — it provides a natural design principle: geological
processes are irreversible (rock doesn't un-compact), which directly motivates the
monotonic downgrade constraint.

## Compression zones

Every token in every KV attention head is assigned to exactly one zone:

| Zone | Storage | Bytes per element (K+V) | Geological analogy |
|------|---------|------------------------|--------------------|
| Surface (FP16) | Full precision | 4 bytes | Fresh sediment |
| Shallow (INT8) | 8-bit quantized | 2 bytes | Compacted sandstone |
| Deep (INT4) | 4-bit quantized | 1 byte | Deep shale |
| Bedrock (evict) | SVD sketch only | ~0 (fixed rank) | Dissolved — only fossil imprint remains |

Evicted tokens are not completely lost. A low-rank SVD sketch (default rank 8) preserves
a compressed fingerprint that could be used for approximate retrieval.

## Design components

### 1. Multi-tier compression zones

Tokens are assigned to FP16 / INT8 / INT4 / evict tiers based on cumulative attention
importance scores. This is a **known technique** — PM-KVQ (2025) uses the same
FP16 -> INT8 -> INT4 precision ladder, and MiKV (2024) introduced importance-based
mixed-precision as an alternative to binary eviction.

**What we add:** We combine this with per-layer budgets, per-head assignment, monotonic
enforcement, and anchor pinning (components 2-5 below), which PM-KVQ and MiKV do not.

### 2. Inverse layer budget

Not all transformer layers are equally compressible. The Stratigraphic strategy assigns
a **different FP16 budget to each layer**:

```
fp16_frac(l) = zone_surface * [(1 - lambda) + lambda * l / (L - 1)]
```

Where:
- `zone_surface = 0.30` — maximum FP16 fraction (at the last layer)
- `lambda = 0.6` — gradient steepness (0 = uniform, 1 = full gradient)
- `l` — layer index, `L` — total layers

**Example for a 24-layer model (Qwen2.5-0.5B):**

| Layer | FP16 fraction | INT8 | INT4 | Evicted |
|-------|--------------|------|------|---------|
| 0 (early) | 12% | 30% | 25% | 33% |
| 12 (middle) | 21% | 30% | 25% | 24% |
| 23 (late) | 30% | 30% | 25% | 15% |

Per-layer budgets are a **known technique** — PyramidKV (2024) and SqueezeAttention
(2024) pioneered this. However, our direction differs: PyramidKV allocates MORE cache
to early layers (where attention is dispersed) and LESS to late layers. We do the
opposite — compressing early layers more and preserving late layers — based on the
observation that late-layer attention captures fine-grained semantics critical for
generation quality.

**What we add:** The specific inverse-gradient heuristic combined with the multi-tier
zones (components 1 and 2 together).

### 3. Monotonic downgrade constraint

A token can only move to a **deeper** (more compressed) zone, never back:

```
FP16 -> INT8 -> INT4 -> Evict     allowed
INT8 -> FP16                       forbidden
INT4 -> INT8                       forbidden
```

**Why this matters:** If a token is quantized to INT8 and later promoted back to FP16,
the quantization error is permanently baked in — the FP16 values are reconstructed from
lossy INT8 data, not the original. Repeated round-trips compound these errors
("diagenetic overprinting" in geological terms).

The constraint is enforced per-head, per-layer via a zone history:

```python
zones[head] = torch.maximum(previous_zones[head], new_zones[head])
```

This is a **known mechanism** — PM-KVQ (2025) implements the same progressive
downgrade (FP16 -> INT8 -> INT4 -> INT2 as memory fills), and all eviction methods are
trivially monotonic (keep -> evict is one-way). KVReviver (2025) was specifically
designed to recover from this irreversibility, acknowledging that monotonic compression
is the default behaviour.

**What we add:** We frame the monotonic constraint as a deliberate design principle
(diagenesis) rather than an incidental property, and combine it with per-head zone
tracking (not just a global precision level).

### 4. Stylolite anchors (attention + topic-shift detection)

Named after stylolites — geological pressure-dissolution surfaces that form at
boundaries between rock types. Anchor tokens are **pinned at FP16 permanently**,
regardless of their attention score ranking.

Two signals identify anchors:

**Signal 1: High-attention tokens (known technique)**

Tokens above the 99th percentile of cumulative attention. This is standard —
StreamingLLM pins sink tokens, ZipCache and KITTY give high-attention tokens higher
precision, and virtually all eviction methods use attention scores for importance
ranking.

**Signal 2: Semantic topic-shift boundaries (novel)**

Detected via sliding-window cosine distance on key states:

```
... talking about cats | talking about dogs ...
                       ^
              cosine distance > 0.3 -> anchor pinned at FP16
```

When the mean key vector in a left window diverges significantly from the right window,
that position marks a topic boundary. These boundaries are critical for the model to
distinguish between different semantic regions.

The closest prior work is SepLLM (2024), which retains separator tokens (punctuation)
as segment boundaries. However, SepLLM uses **syntactic** signals (comma, period,
newline) while our topic-shift detection uses **semantic** signals (cosine distance in
the key embedding space). A period does not always mark a topic shift, and topic shifts
do not always coincide with punctuation.

**What we add:** Semantic topic-shift detection as a KV-cache anchor signal. To our
knowledge, no prior KV-cache compression method uses embedding-space cosine distance
for topic boundary detection.

Anchors are capped at `anchor_budget = 5%` of tokens. When more candidates are
detected than the budget allows, the highest-attention anchors are prioritised.

### 5. Per-head zone assignment

Each KV attention head gets its own zone map. This is a **known technique** —
DuoAttention (2024) classifies heads as retrieval vs streaming, Ada-KV (2024) optimises
per-head budgets, and HeadKV (2024) assigns compression budgets proportional to head
importance.

**What we add:** Combining per-head assignment with multi-tier zones (not just binary
full/compressed) and monotonic enforcement per-head. Each head independently tracks its
zone history and can have a completely different FP16/INT8/INT4/evict distribution.

### 6. SVD sketch for evicted tokens

Evicted tokens retain a low-rank SVD sketch (default rank 8) rather than being
completely lost. This is **partially novel** — GEAR (2024) uses SVD for quantization
residual compensation, PALU (2024) applies SVD to projection matrices, and KVReviver
(2025) stores compressed sketches of evicted tokens for later reconstruction. MiKV
(2024) avoids the problem entirely by keeping evicted tokens at very low precision
instead of dropping them.

**What we add:** Integrating the SVD sketch as the "bedrock" tier in a multi-zone
system, where it serves as a minimal information-preserving fallback below INT4.

## What is novel vs what exists

| Component | Exists in prior work? | Our contribution |
|-----------|----------------------|------------------|
| Multi-tier FP16/INT8/INT4/evict | Yes (PM-KVQ, MiKV, ThinKV) | Integration with components below |
| Per-layer varying budget | Yes (PyramidKV, SqueezeAttention, LAVa) | Inverse direction + combined with multi-tier zones |
| Monotonic downgrade | Yes (PM-KVQ, implicit in all eviction) | Explicit per-head enforcement + geological framing |
| Attention-based anchor pinning | Yes (StreamingLLM, ZipCache, KITTY) | Combined with topic-shift signal |
| **Topic-shift detection for KV cache** | **No** | **Novel: cosine-distance boundary detection in key space** |
| Per-head zone assignment | Yes (DuoAttention, Ada-KV, HeadKV) | Combined with multi-tier zones + monotonic tracking |
| SVD sketch for evicted tokens | Partial (KVReviver, GEAR, PALU) | As the bedrock tier in a multi-zone system |
| **Unified framework (all above)** | **No single paper combines all** | **Novel integration under geological design principle** |

## Configuration

All parameters are controlled via `StratigraphicConfig`:

```python
@dataclass
class StratigraphicConfig:
    # Layer budget
    zone_surface: float = 0.30    # max FP16 fraction (at last layer)
    lambda_: float = 0.6          # layer gradient steepness

    # Zone fractions
    zone_shallow: float = 0.30    # INT8 fraction (all layers)
    zone_deep: float = 0.25       # INT4 fraction (all layers)
    # Remainder = evicted

    # Anchor detection
    anchor_budget: float = 0.05   # max fraction of anchor tokens
    anchor_attn_percentile: float = 0.99
    topic_shift_window: int = 32
    topic_shift_threshold: float = 0.3

    # Evicted token sketch
    sketch_rank: int = 8          # SVD rank for evicted tokens
    int8_group_size: int = 128
    int4_group_size: int = 64
```

## Adaptive Tiered (ablation variant)

The Adaptive Tiered strategy is a simplified version without the geological innovations:

- **Flat budgets:** 25% FP16, 30% INT8, 25% INT4, 20% evicted — same for every layer
- **No monotonic constraint** — tokens can be re-assigned freely
- **No anchor detection** — no special treatment for high-attention or topic-shift tokens
- **No per-head zones** — global importance scoring across all heads

Comparing Stratigraphic vs Adaptive Tiered isolates the value of the inverse layer
budget, monotonic constraint, and anchor mechanism.

## Implementation

- **`streaming_attention/stratigraphic.py`** — Core logic: `StratigraphicConfig`, `HeadZoneAssigner`
  (per-head zone assignment with monotonic enforcement), `AnchorDetector` (attention +
  topic-shift anchor detection)
- **`kv_bench/strategies/stratigraphic.py`** — Benchmark integration: collects attention
  weights via `on_step()`, computes eviction mask via `get_keep_mask()`, reports analytical
  memory via `memory_bytes()`

## Related work

### Eviction-only methods

- **H2O** (Zhang et al., NeurIPS 2023) — Heavy-Hitter Oracle. Keeps sink tokens + recent
  tokens + highest cumulative attention entries within a fixed budget. Binary keep/evict,
  uniform across layers. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)

- **SnapKV** (Li et al., 2024) — Uses attention patterns from an observation window at
  the end of prefill to select which KV entries to keep. Per-head scoring but binary
  keep/evict. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)

- **StreamingLLM** (Xiao et al., 2023) — Keeps attention sink tokens + a fixed-size
  rolling window. Pioneered the concept of pinning structurally important tokens.
  [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

- **ScissorHands** (Liu et al., 2023) — Fixed-budget eviction by attention score ranking.
  [arXiv:2305.17118](https://arxiv.org/abs/2305.17118)

### Mixed-precision and multi-tier methods

- **MiKV** (Yang et al., 2024) — "No Token Left Behind". Important tokens in FP16,
  less-important in INT4/INT2 instead of evicting. First to frame mixed-precision as
  an alternative to eviction. [arXiv:2402.18096](https://arxiv.org/abs/2402.18096)

- **PM-KVQ** (2025) — Progressive Mixed-precision KV Cache Quantization. FP16 -> INT8 ->
  INT4 -> INT2 progressive downgrade as memory fills. Block-wise budget allocation.
  Closest overall to our multi-tier + monotonic approach, but without per-head zones,
  anchor detection, or topic-shift signals.
  [arXiv:2505.18610](https://arxiv.org/abs/2505.18610)

- **ZipCache** (He et al., NeurIPS 2024) — Salient tokens at 4-bit, non-salient at
  2-bit. Two-tier saliency-based quantization.

- **ThinKV** (2025) — Assigns thought tokens to 8-bit, 4-bit, and 2-bit based on
  thought-type classification. [arXiv:2510.01290](https://arxiv.org/abs/2510.01290)

- **KITTY** (2025) — INT2 for most channels, INT4 for accuracy-sensitive, FP16 for
  sink/local tokens. Three-tier. [arXiv:2511.18643](https://arxiv.org/abs/2511.18643)

### Per-layer budget methods

- **PyramidKV** (Cai et al., 2024) — More cache to early layers (dispersed attention),
  less to late layers (concentrated attention). Opposite direction to our inverse budget.
  [arXiv:2406.02069](https://arxiv.org/abs/2406.02069)

- **SqueezeAttention** (2024) — First to explicitly optimise layer-wise KV budgets using
  cosine similarity to measure layer importance.
  [arXiv:2404.04793](https://arxiv.org/abs/2404.04793)

- **LAVa** (2025) — Dynamic per-layer budget allocation minimising information loss in
  residual streams. [arXiv:2509.09754](https://arxiv.org/abs/2509.09754)

- **EvolKV** (EMNLP 2025) — Evolutionary search for optimal per-layer budgets.
  [arXiv:2509.08315](https://arxiv.org/abs/2509.08315)

- **KVTuner** (ICML 2025) — Sensitivity-aware layer-wise mixed-precision quantization.
  [arXiv:2502.04420](https://arxiv.org/abs/2502.04420)

### Per-head compression methods

- **DuoAttention** (MIT, ICLR 2025) — Classifies heads as retrieval vs streaming.
  Streaming heads get compressed cache, retrieval heads get full cache.
  [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)

- **Ada-KV** (NeurIPS 2025) — Head-wise adaptive budget allocation using L1 eviction
  loss. [arXiv:2407.11550](https://arxiv.org/abs/2407.11550)

- **HeadKV** (ICLR 2025) — Budgets proportional to head importance, distinguishing
  retrieval and reasoning heads.
  [arXiv:2410.19258](https://arxiv.org/abs/2410.19258)

- **KV-Compress** (2024) — Variable compression rates per attention head with
  paged-attention compatibility.
  [arXiv:2410.00161](https://arxiv.org/abs/2410.00161)

### SVD and sketch-based methods

- **GEAR** (NeurIPS 2024) — Low-rank SVD of quantization residuals + sparse FP16
  correction. [arXiv:2403.05527](https://arxiv.org/abs/2403.05527)

- **KVReviver** (2025) — Sketch-based reconstruction of evicted tokens. First to enable
  reversible KV cache compression.
  [arXiv:2512.17917](https://arxiv.org/abs/2512.17917)

- **PALU** (ICLR 2025) — SVD-based low-rank decomposition of KV projection matrices.
  [arXiv:2407.21118](https://arxiv.org/abs/2407.21118)

### Segment and boundary detection

- **SepLLM** (ICML 2025) — Retains separator tokens (punctuation) as segment boundaries.
  Syntactic boundary detection, not semantic topic-shift.
  [arXiv:2412.12094](https://arxiv.org/abs/2412.12094)

- **Keyformer** (Adnan et al., 2024) — Notes that eviction distorts softmax distribution
  due to probability mass redistribution.
  [arXiv:2403.09054](https://arxiv.org/abs/2403.09054)
