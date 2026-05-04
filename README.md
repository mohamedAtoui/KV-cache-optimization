# KV Cache Optimization

Research project exploring KV cache compression strategies for large language models. Starting from a comparison of attention mechanisms (MHA, MQA, GQA, MLA) on toy models, this project develops and benchmarks production-scale techniques — including streaming recurrent state conversion and multi-tier adaptive compression — targeting Llama-3.1-8B-Instruct.

## Project Structure

```
KV-cache-optimization/
├── AttentionHeads/           Toy 16M-param models comparing MHA/MQA/GQA/MLA
│   ├── mha/                  Multi-Head Attention (baseline)
│   ├── mqa/                  Multi-Query Attention
│   ├── gqa/                  Grouped Query Attention
│   ├── mla/                  Multi-Head Latent Attention (DeepSeek-V2 style)
│   ├── notebooks/            Training and evaluation notebooks
│   └── results/              Comparison plots and analysis
│
├── streaming_attention/      Per-head KV cache → recurrent state conversion
│   ├── head_classifier.py    DuoAttention pattern loading + head classification
│   ├── state_attention.py    Decayed linear state: S_t = λ·S_{t-1} + v_t·k_tᵀ
│   ├── hybrid_attention.py   Monkey-patches Llama for hybrid KV/state attention
│   ├── calibration.py        Two-stage tuning: per-head MSE + LoRA fine-tuning
│   ├── importance.py         Multi-signal token importance scoring
│   ├── adaptive_cache.py     Tiered KV cache (FP16/INT8/INT4/evict)
│   ├── stratigraphic.py      Per-head zone assignment with geological metaphor
│   └── notebooks/            Experiments on H100 (zero-shot, calibration, benchmarks)
│
├── kv_bench/                 Benchmarking framework for KV cache strategies
│   ├── strategies/           8 strategy implementations (see kv_bench/README.md)
│   ├── runner.py             Model loading + strategy execution
│   └── report.py             Console/JSON/Markdown output
│
└── docs/                     Research diary and design documents
```

## Packages

### AttentionHeads

Foundational comparison of four attention architectures on TinyStories and SimpleStories datasets. Each variant is a ~16M parameter decoder-only transformer. See [AttentionHeads/README.md](AttentionHeads/README.md) for training instructions and results.

### streaming_attention

Production-scale KV cache optimization for Llama-3.1-8B-Instruct. Streaming heads (identified by DuoAttention) are converted to fixed-size recurrent state matrices, while retrieval heads keep full or tiered KV cache. Includes a two-stage calibration pipeline (decay alignment + LoRA) and multi-tier adaptive compression.

### kv_bench

Unified benchmarking framework comparing 8 KV cache strategies: FullKV baseline, H2O, SnapKV, INT8/INT4 uniform quantization, Stratigraphic compression, StreamingAttention, and Hybrid. See [kv_bench/README.md](kv_bench/README.md) for usage and strategy descriptions.

## Quick Start

```bash
pip install -e .                    # Install streaming_attention package
pip install -e ".[fla]"             # + Flash Linear Attention kernels
pip install -e ".[train]"           # + PEFT/Accelerate for calibration

# Run benchmarks
python -m kv_bench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --pattern-dir attn_patterns/Meta-Llama-3.1-8B-Instruct \
  --strategies baseline streaming_attention h2o snapkv int8 int4 \
  --output results.json -v
```

## Key Results

| Strategy | Perplexity | Memory vs Baseline |
|----------|------------|-------------------|
| FullKV (baseline) | 7.96 | 1.00x |
| H2O (50%) | 8.31 | 0.50x |
| SnapKV (50%) | 8.15 | 0.50x |
| INT8 | 7.97 | 0.50x |
| INT4 | 8.42 | 0.25x |
| Stratigraphic | 8.05 | 0.38x |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) — MQA
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
- [DuoAttention](https://arxiv.org/abs/2410.10819) (Xiao et al., 2024) — Head classification
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) (Zhang et al., 2023)
- [SnapKV](https://arxiv.org/abs/2404.14469) (Li et al., 2024)
