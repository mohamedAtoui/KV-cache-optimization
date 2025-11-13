# AttentionHeads: GPTNeo Language Model

A clean PyTorch implementation of a **GPT-style decoder-only transformer** for language modeling, inspired by GPT-2 and GPTNeo architectures. Trained on the TinyStories dataset with optimized hyperparameters for small-scale language modeling.

## Training Results

**Best Performance** (see `TRAINING_RESULTS.md` for details):
- Validation Loss: 3.6065
- Validation Perplexity: 36.84
- Training Time: ~45-60 minutes on L4 GPU
- Dataset: TinyStories (30K training, 5K validation samples)

## Features

- **GPTNeo Decoder-Only Architecture**: Causal language modeling like GPT-2
- **Multi-Head Self-Attention**: 8-head attention mechanism with causal masking
- **Optimized Training**: BFloat16 mixed precision for L4 GPUs
- **Text Generation**: Greedy decoding, temperature sampling, top-k, nucleus sampling
- **Complete Training Pipeline**: Data loading, checkpointing, TensorBoard logging
- **Pre-configured for TinyStories**: Ready-to-use setup for story generation
- **Modular & Clean Code**: Easy to understand, extend, and modify
- **Well-Documented**: Comprehensive architecture docs and training results

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AttentionHeads

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Training

Open the training notebook in Google Colab:

1. Upload `notebooks/train_gptneo_tinystories.ipynb` to Colab
2. Select L4 GPU runtime (Runtime → Change runtime type → L4 GPU)
3. Run all cells to train the model
4. Training takes ~45-60 minutes for 6,000 steps

Or train locally:

```python
from mha import GPTNeo, TinyStoriesDataLoader
from mha.train import train_model
import json

# Load configuration
with open('mha/config.json', 'r') as f:
    config = json.load(f)

# Create model
model = GPTNeo(**config['model'])

# Load data
train_loader, val_loader = TinyStoriesDataLoader.get_dataloaders(
    **config['data']
)

# Train
train_model(model, train_loader, val_loader, **config['training'])
```

### Text Generation

```python
from mha import GPTNeo
from transformers import GPT2Tokenizer
import torch

# Load model
model = GPTNeo.from_pretrained('checkpoints/mha/best_model.pt')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate story
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=100, temperature=0.8)

story = tokenizer.decode(output[0])
print(story)
```

## Model Architecture

### GPTNeo Decoder-Only Transformer

```
Input Token IDs → Token Embeddings + Positional Embeddings
                 ↓
                Dropout (0.2)
                 ↓
         ┌──────────────────────┐
         │  GPTNeo Block (×4)   │
         │  - Pre-LayerNorm     │
         │  - Self-Attention    │
         │  - Residual          │
         │  - Pre-LayerNorm     │
         │  - Feed-Forward      │
         │  - Residual          │
         └──────────────────────┘
                 ↓
            Final LayerNorm
                 ↓
          Language Model Head
                 ↓
            Output Logits
```

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **hidden_size** | 256 | Model dimension (d_model) |
| **num_layers** | 4 | Number of transformer blocks |
| **num_heads** | 8 | Multi-head attention heads |
| **intermediate_size** | 1024 | Feed-forward hidden dimension |
| **vocab_size** | 50,257 | GPT-2 tokenizer vocabulary |
| **max_seq_length** | 256 | Maximum sequence length |
| **dropout** | 0.2 | Dropout probability |
| **Total Parameters** | ~16M | 3.2M non-embedding + 12.9M embedding |

### Key Features

- **Pre-normalization**: LayerNorm before attention/FFN (training stability)
- **Causal masking**: Prevents attending to future tokens
- **Learned positional embeddings**: Like GPT-2, not sinusoidal
- **GELU activation**: Smoother gradients than ReLU
- **Weight tying**: LM head shares weights with token embeddings

## Project Structure

```
AttentionHeads/
├── mha/                          # Main package
│   ├── __init__.py              # Package initialization
│   ├── transformer.py           # GPTNeo model implementation
│   ├── attention.py             # Multi-head self-attention
│   ├── layers.py                # LayerNorm, FFN, residuals
│   ├── train.py                 # Training loop with BFloat16
│   ├── data_loader.py           # TinyStories dataset
│   ├── utils.py                 # Metrics, logging, checkpointing
│   ├── config.json              # Model & training configuration
│   └── README.md                # Package documentation
├── notebooks/
│   └── train_gptneo_tinystories.ipynb   # Training notebook (Colab)
├── checkpoints/mha/             # Model checkpoints (auto-saved)
├── logs/mha/                    # TensorBoard logs
├── models/                      # Custom model architectures
├── data_processed/              # Preprocessed datasets
├── ARCHITECTURE.md              # Detailed architecture docs
├── TRAINING_RESULTS.md          # Training results & parameters
├── QUICKSTART.md                # Quick setup guide
├── README.md                    # This file
├── requirements.txt             # Dependencies
└── setup.py                     # Package installation
```

## Documentation

- **QUICKSTART.md**: Quick setup and training guide
- **ARCHITECTURE.md**: Detailed architecture explanation
- **TRAINING_RESULTS.md**: Complete training results and parameter analysis
- **mha/README.md**: Package-specific documentation

## Key Hyperparameters

**Critical parameters for successful training** (see `TRAINING_RESULTS.md` for full analysis):

- **Learning Rate**: 5e-5 (20x lower than initial failed attempts)
- **Warmup Steps**: 600 (10% of total training)
- **Gradient Clipping**: 0.5 (tight control for stability)
- **Effective Batch Size**: 256 (via gradient accumulation)
- **Model Layers**: 4 (reduced from 8 for faster, stable training)

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with BFloat16 support)
- Transformers (for GPT-2 tokenizer)
- Datasets (for TinyStories)
- See `requirements.txt` for full list

## License

MIT License - see LICENSE file for details.

## References

- **TinyStories Dataset**: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **GPT-2 Paper**: "Language Models are Unsupervised Multitask Learners"
- **Attention Paper**: "Attention Is All You Need" (Vaswani et al., 2017)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
