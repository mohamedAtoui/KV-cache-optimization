# KV Cache Optimization

**Final-year research project — making large language models cheaper to run by compressing the memory they use during inference.**

When a transformer generates text, it stores Key/Value (KV) vectors for every token it has already seen. This **KV cache** can dominate GPU memory: Llama-3.1-8B at an 8K context uses about 2 GB just for the cache. This project designs, implements, and benchmarks methods to shrink that footprint with minimal quality loss.

---

## Headline results

Llama-3.2-1B on WikiText-2, measured by [`kv_bench`](kv_bench/) (perplexity; lower = better):

| Strategy                         | Perplexity | Δ vs baseline | KV memory | Compression |
| -------------------------------- | ---------: | ------------: | --------: | ----------: |
| FullKV (baseline)                |      11.15 |             — |    128 MB |       1.00× |
| INT8 uniform quantization        |      11.15 |         +0.00 |     68 MB |       1.88× |
| **Stratigraphic (this project)** |  **11.18** |     **+0.04** | **58 MB** |   **2.20×** |
| Adaptive Tiered (this project)   |      30.49 |        +19.35 |     59 MB |       2.17× |
| INT4 uniform quantization        |      13.37 |         +2.22 |     36 MB |       3.56× |
| SnapKV (50%)                     |      17.94 |         +6.79 |     64 MB |       2.00× |
| H2O (50%)                        |     107.72 |        +96.57 |     64 MB |       2.00× |

The **Stratigraphic** strategy I designed cuts KV memory by ~55% with essentially no quality loss (+0.04 PPL), beating every published baseline on the quality-vs-compression Pareto frontier in this setting.

---

## What's in this repo

```
KV-cache-optimization/
├── AttentionHeads/         Foundational study: MHA vs MQA vs GQA vs MLA on toy models
├── streaming_attention/    Production-scale techniques for Llama-3.1-8B
├── kv_bench/               Model-agnostic benchmarking framework
├── tests/                  Test suite (67 tests, runs in CI)
└── docs/                   Research diary and design notes
```

### `AttentionHeads/` — comparing attention mechanisms from scratch

Four ~16M-parameter decoder-only transformers (MHA, MQA, GQA, MLA) trained from scratch on TinyStories and SimpleStories, then compared on perplexity, attention entropy, KV-cache size, FLOPs, inference speed, and generation diversity. Reproducible training pipeline; figures live in `AttentionHeads/results/`.

### `streaming_attention/` — Llama-3.1-8B-scale optimization

- **State conversion** — `state_attention.py` replaces the KV cache of "streaming" attention heads with a fixed-size recurrent state `S_t = λ·S_{t-1} + v_t·k_tᵀ`, using head classification from [DuoAttention](https://arxiv.org/abs/2410.10819).
- **Two-stage calibration** — `calibration.py` aligns per-head decay constants by MSE, then fine-tunes with LoRA via PEFT.
- **Adaptive tiered cache** — `adaptive_cache.py` mixes FP16 / INT8 / INT4 / evicted entries based on a multi-signal importance score.
- **Stratigraphic compression** — `stratigraphic.py`, the headline contribution: tokens are assigned to compression "zones" (FP16 surface → INT8 shallow → INT4 deep → evicted bedrock) with a monotonic downgrade constraint and a per-layer budget gradient that preserves late layers.
- **TurboQuant** — `turboquant.py` adds learned codebook quantization with cached Lloyd's-algorithm codebooks.

### `kv_bench/` — benchmarking framework

Model-agnostic CLI and Python API for comparing eight KV-cache strategies on any HuggingFace causal LM. Auto-detects GPU and scales batch size, sequence length, and stride accordingly. Outputs comparison tables (console / JSON / Markdown) and Pareto-frontier plots. Strategies covered: FullKV, H2O, SnapKV, INT8, INT4, Adaptive Tiered, Stratigraphic, StreamingAttention, Hybrid. See [`kv_bench/README.md`](kv_bench/README.md) for full details.

---

## Tech stack

**Languages & libraries** — Python · PyTorch · HuggingFace Transformers · PEFT (LoRA) · Accelerate · Flash Linear Attention · NumPy · Matplotlib · Jupyter

**Tooling** — `uv` workspace · pytest (67 tests) · GitLab CI · Modal (serverless GPU benchmarking on H100) · LaTeX (final report)

**Models targeted** — Llama-3.1-8B-Instruct · Llama-3.2-1B-Instruct · Qwen2.5-0.5B-Instruct (any HF causal LM via `kv_bench`)

---

## Quick start

```bash
git clone https://github.com/mohamedAtoui/KV-cache-optimization.git
cd KV-cache-optimization
uv sync                                # install workspace + dependencies

# Reproduce the headline benchmark
uv run python -m kv_bench \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --strategies baseline stratigraphic int8 int4 h2o snapkv adaptive \
    --max-samples 100 --max-seq-len 1024 --stride 512 \
    --output results.json --markdown results.md
```

A full Modal-on-H100 reproduction lives in [`streaming_attention/notebooks/03-kv-bench-modal.ipynb`](streaming_attention/notebooks/03-kv-bench-modal.ipynb) — set `HF_TOKEN` in the environment before running.

---

## Highlights

- Designed, implemented, and benchmarked an **original KV-cache compression strategy** (Stratigraphic) that improves on published baselines (H2O, SnapKV) at matched compression.
- Built a **model-agnostic benchmarking framework** with GPU auto-scaling, used in this repo and reusable for any HF causal LM.
- Reproduced and integrated three published methods (H2O, SnapKV, DuoAttention) as part of the comparison.
- Trained four attention variants (MHA / MQA / GQA / MLA) **from scratch** at 16M parameters with full reproducible pipeline.
- Set up the engineering side properly: `uv` workspace, 67-test suite, GitLab CI, serverless GPU runs via Modal, LaTeX final report.

---

## References

- Vaswani et al., 2017 — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, 2019 — [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- Ainslie et al., 2023 — [GQA: Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- DeepSeek-AI, 2024 — [Multi-head Latent Attention](https://arxiv.org/abs/2405.04434)
- Zhang et al., NeurIPS 2023 — [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)
- Li et al., 2024 — [SnapKV](https://arxiv.org/abs/2404.14469)
- Xiao et al., 2024 — [DuoAttention](https://arxiv.org/abs/2410.10819)

---

*Mohamed Atoui · Royal Holloway, University of London · 2025*
