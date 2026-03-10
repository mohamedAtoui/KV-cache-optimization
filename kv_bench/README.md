# kv_bench

Benchmarking framework for comparing KV-cache compression strategies on real Llama-3.1-8B inference. Measures perplexity, analytical memory, latency, and compression ratio across strategies.

## Strategies

| Strategy | Description |
|----------|-------------|
| **FullKV (baseline)** | No compression — reference perplexity |
| **KV2State** | Streaming heads → fixed-size recurrent state via `kv2state` |
| **H2O** | Heavy-Hitter Oracle — evict low-attention entries |
| **SnapKV** | Observation-window-based KV selection |
| **INT8 / INT4** | Uniform symmetric quantization of all entries |
| **Adaptive Tiered** | Multi-signal importance scoring + FP16/INT8/INT4/evict tiers |
| **Hybrid** | KV2State (streaming heads) + Adaptive Tiered (retrieval heads) |

## Usage

### CLI

```bash
python -m kv_bench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --pattern-dir attn_patterns/Meta-Llama-3.1-8B-Instruct \
  --strategies baseline kv2state h2o snapkv int8 int4 adaptive hybrid \
  --output results.json
```

### Python API

```python
from kv_bench import BenchmarkRunner, auto_detect
from kv_bench.strategies import FullKVBaseline, KV2StateStrategy, H2OStrategy

runner = BenchmarkRunner(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    strategies=[
        FullKVBaseline(),
        KV2StateStrategy(pattern_dir="attn_patterns/Meta-Llama-3.1-8B-Instruct"),
        H2OStrategy(budget=0.5),
    ],
    device_config=auto_detect(),
)
results = runner.run()
```

## GPU auto-detection

`auto_detect()` sets batch size, sequence length, and stride based on available VRAM:

| GPU | VRAM | batch_size | max_seq_len |
|-----|------|------------|-------------|
| A100-80GB / H100 | ≥70 GB | 4 | 8192 |
| A100-40GB | ≥35 GB | 2 | 4096 |
| L4 / T4 | <35 GB | 1 | 2048 |
| CPU | — | 1 | 256 |

All settings can be overridden via `DeviceConfig` or CLI flags (`--max-seq-len`, `--batch-size`, `--stride`).

## How it works

1. Model and tokenizer are loaded **once**
2. For each strategy: `setup()` → evaluate perplexity (sliding window, `use_cache=False`) → measure memory → `teardown()`
3. Strategies that need attention weights (H2O, SnapKV, Adaptive) receive them via `on_step()` hooks
4. Memory and compression metrics are **analytical** (computed from strategy parameters), not from CUDA memory — since `use_cache=False` doesn't grow the KV cache at runtime
5. Results are printed as a comparison table and optionally saved to JSON/markdown
