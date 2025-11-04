# Multi-Head Attention (MHA) Transformer

Complete implementation of a Transformer Encoder-Decoder with Multi-Head Attention for language modeling on WikiText dataset.

## Project Structure

```
mha/
├── __init__.py                    # Package initialization
├── config.json                    # Model and training configuration
├── layers.py                      # Core layers (LayerNorm, FFN, Residual)
├── positional_encoding.py         # Sinusoidal and learned positional encodings
├── attention.py                   # Multi-Head Attention implementation
├── transformer.py                 # Full encoder-decoder architecture
├── data_loader.py                 # WikiText data loading utilities
├── utils.py                       # Metrics, logging, visualization
├── train.py                       # Training script
└── README.md                      # This file
```

## Features

### Architecture Components
- ✅ **Multi-Head Attention** with scaled dot-product
- ✅ **Positional Encoding** (sinusoidal and learned)
- ✅ **Layer Normalization** and **Residual Connections**
- ✅ **Feed-Forward Networks** (GELU/ReLU activation)
- ✅ **Causal Masking** for autoregressive modeling
- ✅ **Padding Mask** support

### Training Features
- ✅ Label smoothing
- ✅ Gradient clipping
- ✅ Learning rate warmup
- ✅ TensorBoard logging
- ✅ Checkpoint management
- ✅ Perplexity tracking

## Quick Start

### 1. Local Testing

Test individual components:

```bash
# Test layers
cd mha
python layers.py

# Test positional encoding
python positional_encoding.py

# Test attention
python attention.py

# Test transformer
python transformer.py

# Test data loader
python data_loader.py
```

### 2. Local Training

Train locally:

```bash
cd mha
python train.py --config config.json
```

### 3. Colab Training

1. Open `notebooks/train_mha_colab.ipynb` in Google Colab
2. Follow the notebook instructions
3. Checkpoints will be saved to your Google Drive

## Configuration

Edit `config.json` to modify hyperparameters:

```json
{
  "model_config": {
    "d_model": 512,           // Model dimension
    "num_heads": 8,           // Number of attention heads
    "num_encoder_layers": 6,  // Encoder depth
    "num_decoder_layers": 6,  // Decoder depth
    "d_ff": 2048,            // FFN hidden dimension
    "dropout": 0.1           // Dropout rate
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 20,
    "warmup_steps": 4000
  }
}
```

## Model Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────┐
│ ENCODER (×6)    │
│ - Self-Attention│
│ - Feed-Forward  │
└─────────────────┘
    ↓
┌─────────────────┐
│ DECODER (×6)    │
│ - Self-Attention│
│ - Cross-Attn    │
│ - Feed-Forward  │
└─────────────────┘
    ↓
Linear + Softmax
    ↓
Output Logits
```

## Metrics

- **Loss**: Cross-entropy with label smoothing
- **Perplexity**: exp(loss) - lower is better
- **Learning Rate**: Warmup + inverse sqrt schedule

## Checkpoints

Checkpoints are saved to:
- Local: `../checkpoints/mha/`
- Colab: `/content/drive/MyDrive/LLM-Journey/checkpoints/mha/`

Best model: `best_model.pt`

## TensorBoard

View training progress:

```bash
tensorboard --logdir ../logs/mha
```

## Testing

All modules include unit tests in their `__main__` blocks. Run:

```bash
python <module_name>.py
```

## Future Extensions

When implementing other attention mechanisms (MQA, GQA, MLA):
1. Copy the `mha/` folder
2. Rename to `mqa/`, `gqa/`, etc.
3. Modify only the attention mechanism in `attention.py`
4. Keep all other components the same for fair comparison

## Requirements

See `../requirements.txt` for dependencies.

## Author

Your Name - FYP: Comparison of Transformer Attention Mechanisms
