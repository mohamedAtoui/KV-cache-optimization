# LLM-Journey: Transformer Implementation

A comprehensive PyTorch implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017), closely following **Harvard NLP's Annotated Transformer**.

## 📚 Based on Harvard NLP's Annotated Transformer

This implementation is based on the excellent [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) by Harvard NLP, which provides a line-by-line implementation of the original Transformer paper with detailed explanations.

### Citation

**Original Paper:**
```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

**Harvard NLP's Annotated Transformer:**
- Website: https://nlp.seas.harvard.edu/annotated-transformer/
- GitHub: https://github.com/harvardnlp/annotated-transformer

## 🎯 Features

- **Complete Transformer Architecture**: Full encoder-decoder implementation
- **Multi-Head Attention (MHA)**: Standard transformer attention mechanism
- **Harvard NLP Style**: Follows the annotated transformer's clean, educational structure
- **Multiple Generation Strategies**: Greedy, temperature, top-k, nucleus sampling
- **Training Infrastructure**: Learning rate scheduling, label smoothing, checkpointing
- **WikiText-2 Training**: Pre-configured for language modeling
- **Modular Design**: Easy to understand, extend, and modify
- **Backward Compatible**: Supports both Harvard NLP style and legacy API

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/attaimen/LLM-Journey.git
cd LLM-Journey

# Install in editable mode
pip install -e .
```

### Usage (Harvard NLP Style - Recommended)

```python
from mha import make_model

# Create a standard Transformer (Harvard NLP way)
model = make_model(
    src_vocab=10000,  # Source vocabulary size
    tgt_vocab=10000,  # Target vocabulary size
    N=6,              # Number of layers
    d_model=512,      # Model dimension
    d_ff=2048,        # Feed-forward dimension
    h=8,              # Number of attention heads
    dropout=0.1       # Dropout probability
)

# The model is ready to use!
# It includes proper Xavier initialization
```

### Usage (Legacy Style - Still Supported)

```python
from mha import Transformer

model = Transformer(
    vocab_size=50257,      # GPT-2 vocabulary
    d_model=512,           # Model dimension
    num_heads=8,           # Attention heads
    num_encoder_layers=6,  # Encoder layers
    num_decoder_layers=6,  # Decoder layers
    d_ff=2048,             # FFN dimension
    max_seq_length=512,    # Max sequence length
    dropout=0.1            # Dropout
)
```

### Text Generation

```python
from mha.inference import TextGenerator
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create generator
generator = TextGenerator(model, tokenizer, device='cuda')

# Generate text
text = generator.generate_greedy(
    prompt="The transformer architecture",
    max_length=50
)
print(text)

# Or use other sampling methods
text = generator.generate_with_temperature(prompt, temperature=0.8)
text = generator.generate_top_k(prompt, k=50)
text = generator.generate_nucleus(prompt, p=0.9)
```

## 📖 Implementation Details

### Architecture Components

| Component | Harvard NLP Class | Description |
|-----------|------------------|-------------|
| **Model** | `EncoderDecoder` | Main seq2seq wrapper |
| **Encoder** | `Encoder` | Stack of N encoder layers |
| **Decoder** | `Decoder` | Stack of N decoder layers |
| **Attention** | `MultiHeadedAttention` | Scaled dot-product attention with h heads |
| **FFN** | `PositionwiseFeedForward` | Two-layer feed-forward network |
| **Embeddings** | `Embeddings` | Token embeddings with √d_model scaling |
| **Positional Encoding** | `PositionalEncoding` | Sinusoidal position encodings |
| **Generator** | `Generator` | Linear projection + log softmax |

### Key Functions

- **`make_model()`**: Factory function to create a complete transformer
- **`attention()`**: Core scaled dot-product attention function
- **`rate()`**: Learning rate schedule with warmup
- **`greedy_decode()`**: Autoregressive text generation
- **`subsequent_mask()`**: Causal mask for decoder
- **`clones()`**: Deep copy N modules

### Training Utilities

- **`Batch`**: Batch processing with automatic masking
- **`rate()`**: Learning rate warmup schedule from the paper
- **`LabelSmoothing`**: Smoothed cross-entropy loss
- **`MetricsTracker`**: Loss and perplexity tracking
- **`CheckpointManager`**: Model checkpointing

## 🔬 Comparison: Original vs This Implementation

| Aspect | Harvard NLP Annotated Transformer | This Implementation |
|--------|-----------------------------------|---------------------|
| **Base Structure** | Single file tutorial | Modular package structure |
| **API Style** | `make_model()` factory | Both `make_model()` and `Transformer` class |
| **Architecture** | Exact paper implementation | Same + backward compatible |
| **Training** | Basic examples | Full WikiText-2 pipeline |
| **Inference** | Greedy decode | Multiple sampling strategies |
| **Organization** | Educational (single file) | Production (modular) |
| **Extras** | - | TensorBoard, checkpointing, visualization |

## 📂 Project Structure

```
LLM-Journey/
├── mha/                           # Main package
│   ├── __init__.py               # Package exports
│   ├── attention.py              # Multi-head attention
│   ├── layers.py                 # LayerNorm, FFN, residual connections
│   ├── positional_encoding.py   # Sinusoidal & learned PE
│   ├── transformer.py            # Full architecture
│   ├── utils.py                  # Training utilities
│   ├── inference.py              # Text generation
│   └── data_loader.py            # WikiText data loading
├── notebooks/
│   └── train_mha_colab.ipynb    # Training notebook for Colab
├── README.md                     # This file
└── setup.py                      # Package setup
```

## 🎓 Training on WikiText-2

We provide a complete Colab notebook for training:

1. Open `notebooks/train_mha_colab.ipynb` in Google Colab
2. Run all cells to train on WikiText-2
3. Model checkpoints are saved to Google Drive
4. Includes inference examples and attention visualization

## 🤝 Acknowledgments

This implementation is heavily inspired by and based on:

1. **Harvard NLP's Annotated Transformer** ([link](https://nlp.seas.harvard.edu/annotated-transformer/))
   - Provided the clean, educational implementation structure
   - Excellent line-by-line explanations of the paper
   - Reference implementation for `make_model()`, `attention()`, and other core functions

2. **"Attention Is All You Need"** by Vaswani et al. (2017)
   - Original Transformer paper
   - Introduced the architecture we implement here

3. **Community**: Thanks to the PyTorch and Hugging Face communities for tools and resources

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [Original Paper](https://arxiv.org/abs/1706.03762)
- [Harvard NLP Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **Star this repo if you find it useful!**

Built with ❤️ following Harvard NLP's excellent educational materials.
