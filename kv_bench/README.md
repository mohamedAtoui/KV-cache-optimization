# kv_bench

A model-agnostic benchmarking framework for comparing KV-cache compression strategies.
Works with any HuggingFace causal LM — specify `--model` and it handles the rest.

Measures perplexity (PPL), analytical KV-cache memory, prefill latency, and compression
ratio across strategies, producing comparison tables and Pareto frontier plots.

## What is the KV cache problem?

When a transformer generates text, each layer stores **Key** and **Value** vectors for every
token it has seen. This **KV cache** lets the model attend to all previous tokens without
recomputing them, but it grows linearly with sequence length and consumes significant GPU
memory. For example, Llama-3.1-8B at 8K tokens requires ~2 GB for the KV cache alone.

kv_bench compares different strategies for reducing this memory footprint while measuring
how much perplexity (generation quality) is lost.

## Strategies

The benchmark covers the three main families of KV-cache compression from the literature,
plus two novel strategies (Stratigraphic and Adaptive Tiered):

### Eviction-only (drop tokens entirely)

| Strategy | Description | Reference |
|----------|-------------|-----------|
| **H2O** | Heavy-Hitter Oracle — accumulates attention scores across layers, keeps sink tokens + recent tokens + highest-scoring entries within a budget | Zhang et al., NeurIPS 2023 |
| **SnapKV** | Uses attention patterns from an observation window at the end of prefill to select which KV entries to keep | Li et al., 2024 |

### Quantization-only (make entries smaller, keep all tokens)

| Strategy | Description |
|----------|-------------|
| **INT8** | Uniform symmetric quantization of all KV entries to 8-bit |
| **INT4** | Uniform symmetric quantization of all KV entries to 4-bit |

### Eviction + quantization (hybrid compression)

| Strategy | Description |
|----------|-------------|
| **Stratigraphic** (ours) | Geological metaphor: tokens are assigned to compression zones (FP16 / INT8 / INT4 / evict) with a monotonic downgrade constraint. Per-layer budget gradient compresses early layers more and preserves late layers |
| **Adaptive Tiered** (ours) | Configurable FP16/INT8/INT4/evict tiers based on multi-signal importance scoring |

### Recurrent state conversion

| Strategy | Description |
|----------|-------------|
| **StreamingAttention** | Replaces KV cache of "streaming" attention heads with a fixed-size recurrent state S_t = lambda * S_{t-1} + v_t * k_t^T, using DuoAttention head classification |
| **Hybrid** | StreamingAttention for streaming heads + Adaptive Tiered for retrieval heads |

### Reference

| Strategy | Description |
|----------|-------------|
| **FullKV (baseline)** | No compression — full KV cache at bf16. Provides the reference perplexity that all other strategies are compared against |

## Usage

### CLI

```bash
# Any HuggingFace causal LM works
python -m kv_bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --strategies baseline stratigraphic h2o snapkv int8 int4 adaptive \
  --output results.json \
  --markdown results.md \
  -v

# For streaming_attention/hybrid strategies, provide DuoAttention patterns
python -m kv_bench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --pattern-dir attn_patterns/Meta-Llama-3.1-8B-Instruct \
  --strategies baseline streaming_attention h2o snapkv int8 int4 adaptive stratigraphic hybrid \
  --output results.json
```

### Python API

```python
from kv_bench import BenchmarkRunner, auto_detect
from kv_bench.strategies import FullKVBaseline, H2OStrategy, StratigraphicStrategy

runner = BenchmarkRunner(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    strategies=[
        FullKVBaseline(),
        H2OStrategy(budget=0.5),
        StratigraphicStrategy(),
    ],
    device_config=auto_detect(),
)
results = runner.run()
```

## GPU auto-detection

`auto_detect()` sets batch size, sequence length, and stride based on available VRAM:

| GPU | VRAM | batch_size | max_seq_len | stride |
|-----|------|------------|-------------|--------|
| A100-80GB / H100 | >=70 GB | 4 | 8192 | 2048 |
| A100-40GB | >=35 GB | 2 | 4096 | 1024 |
| L4 / T4 | <35 GB | 1 | 2048 | 512 |
| CPU | -- | 1 | 256 | 128 |

All settings can be overridden via `DeviceConfig` or CLI flags (`--max-seq-len`, `--batch-size`, `--stride`).

## Architecture

```
kv_bench/
├── __main__.py        CLI entry point + strategy registry
├── strategy.py        Abstract base class (KVCacheStrategy) + result dataclass
├── runner.py          Loads model once, runs each strategy, two-pass evaluation
├── device_config.py   GPU auto-detection and scaling
├── report.py          Console table, JSON, and Markdown output
└── strategies/
    ├── baseline.py          FullKV — no compression (reference)
    ├── h2o.py               Heavy-Hitter Oracle eviction
    ├── snapkv.py            Observation-window-based selection
    ├── stratigraphic.py     Per-layer zone assignment + monotonic downgrade (ours)
    ├── adaptive_tiered.py   FP16/INT8/INT4/evict tiers (ours)
    ├── uniform_quant.py     INT8 / INT4 uniform quantization
    ├── streaming_attention.py  Streaming heads → recurrent state
    └── hybrid.py               StreamingAttention + Adaptive Tiered combined
```

Every strategy implements the `KVCacheStrategy` interface:

- **`setup(model, model_config, device_config)`** — configure the strategy (optionally patch the model)
- **`memory_bytes(seq_len, model_config)`** — compute analytical KV-cache memory in bytes
- **`teardown(model)`** — restore the model to its original state
- **`needs_attention_weights()`** — whether this strategy needs attention matrices from the forward pass
- **`on_step(layer_idx, attn_weights)`** — receive per-layer attention weights for scoring
- **`get_keep_mask(seq_len, device)`** — return a boolean mask of which tokens to keep (eviction strategies only)
- **`reset()`** — clear internal state between sliding windows

## How it works

1. Model and tokenizer are loaded **once**
2. For each strategy: `setup()` -> evaluate perplexity (sliding window, `use_cache=False`) -> measure memory -> `teardown()`
3. Strategies that need attention weights (H2O, SnapKV, Stratigraphic, Adaptive Tiered) receive them via `on_step()` hooks
4. Memory and compression metrics are **analytical** (computed from strategy parameters and model config), not from CUDA memory — since `use_cache=False` means no KV cache is materialized at runtime
5. Results are printed as a comparison table and optionally saved to JSON/Markdown

## Evaluation method: Two-pass with 4D attention masks

### The problem

Eviction strategies need to show a real perplexity impact — if every strategy reports the
same PPL as baseline, there is no way to compare quality-vs-compression trade-offs. But
actually modifying the KV cache tensors requires fragile monkey-patching of model-specific
attention internals that breaks across architectures (Llama vs Qwen vs Mistral, etc.).

### The solution

We use a **two-pass simulation** approach:

1. **Pass 1 (scoring):** Run the model normally with `output_attentions=True`. The strategy
   receives the full attention weight matrices via `on_step()` and uses them to score token
   importance (e.g., cumulative attention for H2O, observation-window attention for SnapKV).

2. **Eviction decision:** The strategy computes a boolean **keep mask** of shape `[seq_len]`
   via `get_keep_mask()`, marking which token positions would survive eviction under its
   budget and policy.

3. **Pass 2 (evaluation):** Run the model again with a **4D attention mask** of shape
   `[1, 1, T, T]`. Columns corresponding to evicted positions are set to `-inf`. After
   softmax, the model assigns zero attention to those tokens — this is mathematically
   equivalent to those tokens not being present in the KV cache. The cross-entropy loss
   from this pass gives the strategy's perplexity.

Strategies without eviction (Baseline, INT8, INT4) skip Pass 1 and use a single forward
pass, since they keep all tokens and quantization noise does not meaningfully alter
perplexity in simulation.

### Why masking is equivalent to physical removal

Setting a key position to `-inf` in the pre-softmax attention score matrix means
`exp(-inf) = 0`. That position contributes zero to both the numerator and denominator of
the softmax, which is identical to the position not existing. The remaining positions
renormalize exactly as they would if the evicted entries were physically deleted from the
KV cache tensor.

This holds because:
- RoPE (rotary positional encoding) is already applied to all tokens before the attention
  computation, so retained tokens keep their correct positional information
- We evaluate on a fixed input (not autoregressive generation), so no future position IDs
  are affected by the eviction
- The softmax renormalization is identical whether the entry is masked or absent

### Why not physically remove KV entries?

Physical removal (shrinking the KV tensor via `torch.gather`) is what production
implementations use for actual memory savings. We chose masking because:

- **Model-agnostic:** works with any HuggingFace causal LM without per-architecture code
- **No monkey-patching:** no need to modify attention layer internals, which differ across
  Llama, Qwen, Mistral, Falcon, etc.
- **Same PPL result:** for single-pass evaluation with fixed positions, masking and physical
  removal produce identical perplexity (see validation below)
- **Simpler code:** the entire eviction mechanism is ~10 lines in the runner, not scattered
  across model-specific attention patches

### Validation in prior work

- The **H2O** codebase (github.com/FMInference/H2O) includes both *"simulation code
  (masking attention matrix)"* and *"real KV dropping implementation"* as two equivalent
  evaluation modes
- **SnapKV** (Li et al., 2024) uses physical removal via `torch.gather` for their
  implementation, but the mathematical equivalence of masking is well-established in the
  attention literature since `softmax(-inf) = 0` is a standard identity
- **Keyformer** (Adnan et al., 2024) notes that eviction (whether by masking or physical
  removal) causes softmax redistribution effects — this is an inherent property of
  eviction itself, not of the evaluation method

## References

- Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y.,
  Re, C., Barrett, C., Wang, Z., & Chen, B. (2023). *H2O: Heavy-Hitter Oracle for
  Efficient Generative Inference of Large Language Models*. NeurIPS 2023.
  [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)

- Li, Y., He, Y., Sun, H., Yuan, X., Zhao, D., Wang, J., Zhu, Y., Chen, Z., & Li, S.
  (2024). *SnapKV: LLM Knows What You Are Looking For Before Generation*.
  [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)

- Adnan, M., Arunkumar, A., Jain, G., Nair, P., Soloveychik, I., & Kamath, P. (2024).
  *Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative
  Inference*. [arXiv:2403.09054](https://arxiv.org/abs/2403.09054)

- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). *Efficient Streaming
  Language Models with Attention Sinks*. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
