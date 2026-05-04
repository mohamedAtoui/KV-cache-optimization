# AttentionHeads: Comparing Attention Mechanisms in Language Models

A clean PyTorch implementation of a **GPT-style decoder-only transformer** comparing **4 attention mechanisms** (MHA, MQA, GQA, MLA) on two story-generation datasets. All models use **Rotary Position Embeddings (RoPE)** and share the same training infrastructure for fair comparison.

## Attention Mechanisms

| Mechanism | Description | KV-Cache/Token/Layer | Relative Cache |
|-----------|-------------|---------------------|----------------|
| **MHA** (Multi-Head Attention) | Standard: h query heads, h KV heads | 512 values | 1.0x |
| **MQA** (Multi-Query Attention) | All heads share single K,V | 64 values | 0.125x |
| **GQA-4** (Grouped Query Attention) | Groups of 2 query heads share K,V | 256 values | 0.5x |
| **MLA** (Multi-Head Latent Attention) | Low-rank KV compression + decoupled RoPE | 144 values | 0.28x |

## Datasets

- **TinyStories**: 2.1M children's stories (Eldan & Li, 2023) - [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)
- **SimpleStories**: 2.1M parameterized synthetic stories (Finke et al., 2025) - [HuggingFace](https://huggingface.co/datasets/SimpleStories/SimpleStories)

## Features

- **4 Attention Variants**: MHA, MQA, GQA, MLA with shared infrastructure
- **Rotary Position Embeddings (RoPE)**: All models use RoPE instead of learned positional embeddings
- **Multi-Head Latent Attention (MLA)**: DeepSeek-V2 style KV compression with decoupled RoPE
- **2 Datasets**: TinyStories + SimpleStories for cross-dataset comparison
- **~16M Parameter Models**: Iso-parameter comparison at small scale
- **BFloat16 Mixed Precision**: Optimized for L4/A100 GPUs
- **Complete Pipeline**: Data loading, training, evaluation, generation, comparison
- **Colab-Ready Notebooks**: Train and compare all models on Google Colab

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd AttentionHeads
pip install -r requirements.txt
pip install -e .
```

### Training

Open training notebooks in Google Colab:

1. **TinyStories**: Train each model individually
   - `notebooks/train_gptneo_tinystories.ipynb` (MHA)
   - `notebooks/train_gptneo_mqa_tinystories.ipynb` (MQA)
   - `notebooks/train_gptneo_gqa4_tinystories.ipynb` (GQA)
   - `notebooks/train_gptneo_mla_tinystories.ipynb` (MLA)

2. **SimpleStories**: Train all 4 models sequentially
   - `notebooks/train_all_simplestories.ipynb`

3. **Compare**: Unified comparison across all models and datasets
   - `notebooks/compare_all_attention_mechanisms.ipynb`

Each model trains for ~6,000 steps (~45-60 min on L4 GPU).

### Text Generation

```python
from AttentionHeads.mla import create_gptneo_model
from transformers import GPT2Tokenizer
import torch

config = {
    'vocab_size': 50257, 'hidden_size': 256, 'num_layers': 4,
    'num_heads': 8, 'd_c': 128, 'd_rope': 16,
    'intermediate_size': 1024, 'max_position_embeddings': 256
}
model = create_gptneo_model(config)

# Load trained weights
checkpoint = torch.load('best_model_mla.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
input_ids = tokenizer.encode("Once upon a time", return_tensors='pt')
output = model.generate(input_ids, max_length=100, temperature=0.8)
print(tokenizer.decode(output[0]))
```

## Model Architecture

### GPTNeo Decoder-Only Transformer

```
Input Token IDs -> Token Embeddings (+ RoPE inside attention)
                 |
                Dropout (0.2)
                 |
         +----------------------+
         |  GPTNeo Block (x4)   |
         |  - Pre-LayerNorm     |
         |  - Self-Attention    |  <-- MHA / MQA / GQA / MLA
         |  - Residual + Drop   |
         |  - Pre-LayerNorm     |
         |  - Feed-Forward      |
         |  - Residual + Drop   |
         +----------------------+
                 |
            Final LayerNorm
                 |
          Language Model Head
                 |
            Output Logits
```

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **hidden_size** | 256 | Model dimension (d_model) |
| **num_layers** | 4 | Number of transformer blocks |
| **num_heads** | 8 | Attention heads |
| **intermediate_size** | 1024 | Feed-forward hidden dimension |
| **vocab_size** | 50,257 | GPT-2 tokenizer vocabulary |
| **max_seq_length** | 256 | Maximum sequence length |
| **dropout** | 0.2 | Dropout probability |
| **position_encoding** | RoPE | Rotary Position Embeddings |
| **Total Parameters** | ~16M | All variants within ~3% |

## Project Structure

```
AttentionHeads/
├── mha/                          # Multi-Head Attention
│   ├── __init__.py
│   ├── attention.py             # MHA implementation (+ RoPE support)
│   ├── transformer.py           # GPTNeo with MHA
│   ├── rope.py                  # Shared RoPE utility (used by all variants)
│   ├── layers.py                # LayerNorm, FFN, residuals (shared)
│   ├── train.py                 # Training loop (shared by all variants)
│   ├── data_loader.py           # Dataset loading (TinyStories + SimpleStories)
│   ├── utils.py                 # Metrics, logging, checkpointing (shared)
│   ├── config.json              # MHA config (TinyStories)
│   └── config_simplestories.json
├── mqa/                          # Multi-Query Attention
│   ├── __init__.py
│   ├── attention.py             # MQA implementation (+ RoPE)
│   ├── transformer.py           # GPTNeo with MQA
│   ├── config.json              # MQA config (TinyStories)
│   └── config_simplestories.json
├── gqa/                          # Grouped Query Attention
│   ├── __init__.py
│   ├── attention.py             # GQA implementation (+ RoPE)
│   ├── transformer.py           # GPTNeo with GQA
│   ├── config_kv4.json          # GQA config (4 KV heads, TinyStories)
│   └── config_simplestories.json
├── mla/                          # Multi-Head Latent Attention (NEW)
│   ├── __init__.py
│   ├── attention.py             # MLA implementation (DeepSeek-V2 style)
│   ├── transformer.py           # GPTNeo with MLA
│   ├── config.json              # MLA config (TinyStories)
│   └── config_simplestories.json
├── notebooks/
│   ├── train_gptneo_tinystories.ipynb          # MHA training
│   ├── train_gptneo_mqa_tinystories.ipynb      # MQA training
│   ├── train_gptneo_gqa4_tinystories.ipynb     # GQA training
│   ├── train_gptneo_mla_tinystories.ipynb      # MLA training (NEW)
│   ├── train_all_simplestories.ipynb           # All 4 on SimpleStories (NEW)
│   ├── compare_all_attention_mechanisms.ipynb   # Unified comparison (NEW)
│   ├── compare_mha_mqa_models.ipynb            # MHA vs MQA comparison
│   ├── evaluate_attention_comparison.ipynb      # MHA/MQA/GQA evaluation
│   └── analyse_tinystories_embeddings_pca.ipynb # Embedding analysis
├── checkpoints/                  # Model checkpoints
├── logs/                         # TensorBoard logs
├── ARCHITECTURE.md
├── TRAINING_RESULTS.md
├── README.md                     # This file
├── requirements.txt
└── setup.py
```

## Key Hyperparameters

- **Learning Rate**: 5e-5 with cosine decay
- **Warmup Steps**: 600 (10% of total training)
- **Gradient Clipping**: 0.5
- **Effective Batch Size**: 256 (64 x 4 gradient accumulation)
- **Training Steps**: 6,000 per model
- **Training Samples**: 30K (train), 5K (validation)

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with BFloat16 support)
- Transformers (for GPT-2 tokenizer)
- Datasets (for HuggingFace datasets)
- See `requirements.txt` for full list

## References

### Attention Mechanisms
- **MHA**: Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **MQA**: Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- **GQA**: Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **MLA**: DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

### Positional Encodings
- **RoPE**: Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

### Datasets
- **TinyStories**: Eldan, R. & Li, Y. (2023). "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" [arXiv:2305.07759](https://arxiv.org/abs/2305.07759) - [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)
- **SimpleStories**: Finke, L., et al. (2025). "Parameterized Synthetic Text Generation with SimpleStories." [arXiv:2504.09184](https://arxiv.org/abs/2504.09184) - [HuggingFace](https://huggingface.co/datasets/SimpleStories/SimpleStories)

### Implementation Resources
- [MLA technical deep-dive](https://planetbanatt.net/articles/mla.html)
- [MLA implementation walkthrough (shreyansh26)](https://shreyansh26.github.io/post/2025-11-08_multihead-latent-attention/)
- [LLMs-from-scratch MLA chapter](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/05_mla/README.md)
- [MiniGPT MLA implementation](https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
