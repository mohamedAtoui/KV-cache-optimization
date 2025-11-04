# MHA Transformer Architecture Documentation

**Complete Guide to the `mha` Package and Training Pipeline**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Core Modules (File by File)](#core-modules-file-by-file)
4. [Training Pipeline](#training-pipeline)
5. [Data Flow](#data-flow)
6. [Usage Examples](#usage-examples)
7. [Key Concepts](#key-concepts)

---

## Overview

This repository implements the Transformer architecture from **"Attention Is All You Need"** (Vaswani et al., 2017) following **Harvard NLP's Annotated Transformer** patterns. The implementation is modular, educational, and production-ready.

### Key Features

- ✅ **Harvard NLP Patterns**: Uses `make_model()`, `Batch` class, `rate()` scheduler
- ✅ **Modular Design**: Each component in separate files for clarity
- ✅ **Backward Compatible**: Supports both Harvard NLP and legacy API
- ✅ **Complete Training**: Full WikiText-2 training pipeline with monitoring
- ✅ **Multiple Generation Methods**: Greedy, temperature, top-k, nucleus sampling
- ✅ **Educational**: Extensive documentation and comments

---

## Package Structure

```
LLM-Journey/
├── mha/                              # Main package
│   ├── __init__.py                   # Package exports and version info
│   ├── attention.py                  # Multi-head attention mechanism
│   ├── layers.py                     # Layer components (FFN, LayerNorm, etc.)
│   ├── positional_encoding.py       # Position encodings (sinusoidal/learned)
│   ├── transformer.py                # Complete transformer architecture
│   ├── utils.py                      # Training utilities (Batch, rate, etc.)
│   ├── inference.py                  # Text generation and decoding
│   ├── data_loader.py                # WikiText-2 data loading
│   ├── train.py                      # Training script
│   └── config.json                   # Training configuration
│
├── notebooks/
│   ├── train_transformer_v2.ipynb    # Main training notebook (Colab)
│   └── train_mha_colab.ipynb         # Original training notebook
│
├── data_processed/                   # Preprocessed WikiText-2 dataset
├── checkpoints/                      # Model checkpoints
├── logs/                             # Training logs
│
├── setup.py                          # Package installation
├── requirements.txt                  # Dependencies
├── README.md                         # Project overview
└── LICENSE                           # MIT License
```

---

## Core Modules (File by File)

### 1. `mha/__init__.py`

**Purpose**: Package entry point that exports all public APIs

**Key Exports**:
```python
# Harvard NLP (RECOMMENDED)
from mha import make_model, Batch, rate, greedy_decode

# Legacy API (backward compatible)
from mha import Transformer, MultiHeadAttention
```

**What it does**:
- Defines package version (`__version__ = "1.0.0"`)
- Imports and re-exports all components
- Defines `__all__` for `from mha import *`
- Provides both Harvard NLP and legacy APIs

---

### 2. `mha/attention.py`

**Purpose**: Multi-head attention mechanism (core of transformer)

**Key Components**:

#### `attention(query, key, value, mask=None, dropout=None)`
*Harvard NLP's core attention function*

```python
def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        query: (batch, h, seq_len, d_k) - Query vectors
        key: (batch, h, seq_len, d_k) - Key vectors
        value: (batch, h, seq_len, d_k) - Value vectors
        mask: Optional mask to prevent attention to certain positions
        dropout: Optional dropout layer

    Returns:
        - Attention output: (batch, h, seq_len, d_k)
        - Attention weights: (batch, h, seq_len, seq_len)
    """
```

**How it works**:
1. Compute attention scores: `scores = Q @ K^T / sqrt(d_k)`
2. Apply mask (if provided): `scores = scores.masked_fill(mask == 0, -1e9)`
3. Compute attention probabilities: `p_attn = softmax(scores, dim=-1)`
4. Apply dropout (optional)
5. Apply attention to values: `output = p_attn @ V`

#### `MultiHeadedAttention`
*Harvard NLP's multi-head attention class*

```python
class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention allows the model to attend to information from
    different representation subspaces at different positions.

    Args:
        h: Number of attention heads
        d_model: Model dimension
        dropout: Dropout probability
    """
```

**Architecture**:
```
Input (d_model)
    ↓
Split into h heads (each d_k = d_model / h)
    ↓
Linear projections: Q, K, V (3 x Linear(d_model, d_model))
    ↓
Reshape to (batch, h, seq_len, d_k)
    ↓
Apply attention() function
    ↓
Concatenate heads
    ↓
Linear projection (Linear(d_model, d_model))
    ↓
Output (d_model)
```

#### `subsequent_mask(size)`
*Creates causal mask for decoder (prevents looking at future tokens)*

```python
def subsequent_mask(size):
    """
    Mask to hide subsequent positions (for decoder self-attention)

    Returns lower triangular matrix:
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
```

#### Utility Functions
```python
# Mask creation helpers
create_causal_mask(seq_len)           # For decoder self-attention
create_padding_mask(seq, pad_token)   # Hide padding tokens
create_combined_mask(seq, pad_token)  # Both causal + padding
```

**File Size**: ~350 lines
**Dependencies**: `torch`, `torch.nn`, `torch.nn.functional`, `math`

---

### 3. `mha/layers.py`

**Purpose**: Building blocks for transformer layers

**Key Components**:

#### `clones(module, N)`
*Harvard NLP's function to create N identical layers*

```python
def clones(module, N):
    """
    Produce N identical layers using deep copy

    Example:
        # Create 6 identical encoder layers
        layers = clones(encoder_layer, 6)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

#### `LayerNorm`
*Layer normalization (normalizes across features)*

```python
class LayerNorm(nn.Module):
    """
    Layer normalization: normalizes inputs across features

    mean = (1/d) * Σ x_i
    var = (1/d) * Σ (x_i - mean)^2
    output = (x - mean) / sqrt(var + eps) * gamma + beta
    """
```

#### `SublayerConnection`
*Harvard NLP's residual connection + layer norm pattern*

```python
class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer norm

    output = LayerNorm(x + Sublayer(x))

    Note: Harvard NLP applies LayerNorm AFTER addition (post-norm)
    """
```

**Why this matters**: Every transformer layer uses this pattern:
```python
# In encoder/decoder:
x = sublayer_connection(x, lambda x: self_attention(x))
x = sublayer_connection(x, lambda x: feed_forward(x))
```

#### `PositionwiseFeedForward`
*Harvard NLP's 2-layer FFN with ReLU*

```python
class PositionwiseFeedForward(nn.Module):
    """
    Feed-Forward Network (FFN)

    FFN(x) = ReLU(xW1 + b1)W2 + b2

    Args:
        d_model: Input/output dimension (e.g., 512)
        d_ff: Hidden dimension (e.g., 2048)
        dropout: Dropout probability

    Architecture:
        Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)
    """
```

**Typical dimensions**:
- Input: `(batch, seq_len, 512)`
- Hidden: `(batch, seq_len, 2048)`
- Output: `(batch, seq_len, 512)`

#### Legacy Components (Backward Compatibility)
```python
FeedForward              # Alias for PositionwiseFeedForward
ResidualConnection       # Simpler residual connection
DropoutLayer             # Standalone dropout layer
```

**File Size**: ~400 lines
**Dependencies**: `torch`, `torch.nn`, `copy`

---

### 4. `mha/positional_encoding.py`

**Purpose**: Add position information to token embeddings

**Why needed?** Transformers have no inherent notion of sequence order (unlike RNNs). Positional encodings inject position information.

**Key Components**:

#### `PositionalEncoding`
*Harvard NLP's sinusoidal positional encoding*

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the original paper

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where:
        pos = position in sequence (0 to max_len-1)
        i = dimension index (0 to d_model/2-1)

    Properties:
        - Fixed (not learned)
        - Allows model to attend to relative positions
        - PE(pos+k) can be represented as linear function of PE(pos)
    """
```

**Example values** (d_model=512, pos=0-2):
```
Position 0: [0.000, 1.000, 0.000, 1.000, ...]
Position 1: [0.841, 0.540, 0.010, 0.999, ...]
Position 2: [0.909, -0.416, 0.020, 0.998, ...]
```

#### `LearnedPositionalEncoding`
*Alternative: learnable position embeddings*

```python
class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings (like BERT, GPT)

    Each position has a learnable embedding vector
    """
```

#### `PositionalEncodingFactory`
*Factory to create different types of positional encodings*

```python
@staticmethod
def create(encoding_type='sinusoidal', d_model=512, max_len=5000, dropout=0.1):
    """
    Create positional encoding

    Args:
        encoding_type: 'sinusoidal' or 'learned'
    """
```

**File Size**: ~250 lines
**Dependencies**: `torch`, `torch.nn`, `math`

---

### 5. `mha/transformer.py`

**Purpose**: Complete transformer architecture (encoder + decoder)

**This is the main file that ties everything together!**

**Key Components**:

#### `EncoderDecoder`
*Harvard NLP's main seq2seq model wrapper*

```python
class EncoderDecoder(nn.Module):
    """
    Standard Encoder-Decoder architecture

    Components:
        - encoder: Stack of N encoder layers
        - decoder: Stack of N decoder layers
        - src_embed: Source embeddings + positional encoding
        - tgt_embed: Target embeddings + positional encoding
        - generator: Final linear + softmax projection
    """

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: Source sequence (batch, src_len)
            tgt: Target sequence (batch, tgt_len)
            src_mask: Source mask (batch, 1, src_len)
            tgt_mask: Target mask (batch, tgt_len, tgt_len)

        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        return self.decode(
            self.encode(src, src_mask),
            src_mask,
            tgt,
            tgt_mask
        )
```

#### `Encoder`
*Stack of N encoder layers*

```python
class Encoder(nn.Module):
    """
    N-layer encoder

    Each layer:
        1. Multi-head self-attention
        2. Feed-forward network
        (Each with residual connection + layer norm)
    """
```

#### `EncoderLayer`
*Single encoder layer*

```python
class EncoderLayer(nn.Module):
    """
    Encoder layer: self_attn + feed_forward

    Architecture:
        x = LayerNorm(x + MultiHeadAttention(x, x, x))
        x = LayerNorm(x + FeedForward(x))
    """
```

#### `Decoder`
*Stack of N decoder layers*

```python
class Decoder(nn.Module):
    """
    N-layer decoder

    Each layer:
        1. Masked multi-head self-attention (on target)
        2. Multi-head cross-attention (encoder-decoder attention)
        3. Feed-forward network
        (Each with residual connection + layer norm)
    """
```

#### `DecoderLayer`
*Single decoder layer*

```python
class DecoderLayer(nn.Module):
    """
    Decoder layer: self_attn + src_attn + feed_forward

    Architecture:
        # Self-attention (with causal mask)
        x = LayerNorm(x + MultiHeadAttention(x, x, x, mask=tgt_mask))

        # Cross-attention (attend to encoder output)
        x = LayerNorm(x + MultiHeadAttention(x, memory, memory, mask=src_mask))

        # Feed-forward
        x = LayerNorm(x + FeedForward(x))
    """
```

#### `Embeddings`
*Token embeddings with √d_model scaling*

```python
class Embeddings(nn.Module):
    """
    Token embeddings with scaling

    embedding = Embedding(vocab_size, d_model) * sqrt(d_model)

    Why scale? To balance magnitude with positional encodings
    """
```

#### `Generator`
*Final projection to vocabulary*

```python
class Generator(nn.Module):
    """
    Linear projection + log softmax

    output = log_softmax(Linear(d_model -> vocab_size))

    Returns log probabilities for NLLLoss
    """
```

#### `make_model()` ⭐
*Harvard NLP's factory function - THE RECOMMENDED WAY*

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048,
               h=8, dropout=0.1):
    """
    Construct a model from hyperparameters (Harvard NLP way)

    Args:
        src_vocab: Source vocabulary size
        tgt_vocab: Target vocabulary size
        N: Number of encoder/decoder layers
        d_model: Model dimension
        d_ff: Feed-forward dimension
        h: Number of attention heads
        dropout: Dropout probability

    Returns:
        EncoderDecoder model with Xavier initialization

    Example:
        model = make_model(10000, 10000, N=6)
    """
```

**What it does**:
1. Creates all components (attention, FFN, positional encoding)
2. Uses `clones()` to create N identical layers
3. Assembles them into EncoderDecoder
4. Applies Xavier uniform initialization
5. Returns ready-to-train model

#### `Transformer` (Legacy Class)
*Backward-compatible API*

```python
class Transformer(nn.Module):
    """
    Legacy transformer class (still supported)

    Use make_model() instead for new code!
    """
```

**Complete Architecture Diagram**:
```
Input Tokens (src)                Target Tokens (tgt)
        ↓                                  ↓
    Embedding                          Embedding
        ↓                                  ↓
Positional Encoding              Positional Encoding
        ↓                                  ↓
    ┌─────────┐                       ┌─────────┐
    │ Encoder │                       │ Decoder │
    │  Layer  │ ×N                    │  Layer  │ ×N
    └─────────┘                       └─────────┘
        │                                  │
        │         ┌──────────────────────►│
        │         │   Cross-Attention      │
        └─────────┘                        ↓
                                      Generator
                                           ↓
                                   Log Probabilities
```

**File Size**: ~600 lines
**Dependencies**: All other mha modules

---

### 6. `mha/utils.py`

**Purpose**: Training utilities and helpers

**Key Components**:

#### `rate(step, model_size, factor, warmup)`
*Harvard NLP's learning rate schedule*

```python
def rate(step, model_size, factor, warmup):
    """
    Learning rate schedule from the paper

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Behavior:
        - Linear warmup for first 'warmup' steps
        - Then inverse square root decay

    Example:
        step 1:    very small LR
        step 4000: peak LR (if warmup=4000)
        step 8000: LR decreases

    Args:
        step: Current training step (starts at 1)
        model_size: d_model (e.g., 512)
        factor: Scaling factor (usually 1.0)
        warmup: Warmup steps (e.g., 4000)
    """
```

**Used with PyTorch scheduler**:
```python
optimizer = Adam(model.parameters(), lr=1.0)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step + 1, 512, 1.0, 4000))
```

#### `Batch` ⭐
*Harvard NLP's batch object with automatic masking*

```python
class Batch:
    """
    Batch object for training (THE HARVARD NLP WAY)

    Automatically creates:
        - src: Source sequences
        - src_mask: Mask for source (hides padding)
        - tgt: Target input (excludes last token)
        - tgt_y: Target labels (excludes first token)
        - tgt_mask: Combined mask (hides padding + future)
        - ntokens: Number of valid tokens (excluding padding)

    Usage:
        batch = Batch(src=input_ids, tgt=input_ids, pad=pad_token_id)

        output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = criterion(output.reshape(-1, vocab), batch.tgt_y.reshape(-1))
        loss = loss / batch.ntokens  # Normalize
    """
```

**Why use Batch class?**
- ✅ Automatic mask creation (no manual masking needed)
- ✅ Proper src/tgt splitting for teacher forcing
- ✅ Accurate token counting (excludes padding)
- ✅ Cleaner training code
- ✅ Fewer bugs (no manual tensor manipulation)

#### `MetricsTracker`
*Track training/validation metrics*

```python
class MetricsTracker:
    """
    Track loss and compute perplexity

    Methods:
        update(loss, num_tokens)  # Add batch results
        get_average_loss()        # Get epoch average
        get_perplexity()          # Calculate exp(avg_loss)
    """
```

#### `LabelSmoothing`
*Smoothed cross-entropy loss*

```python
class LabelSmoothing(nn.Module):
    """
    Label smoothing for better generalization

    Instead of one-hot: [0, 0, 1, 0, 0]
    Use smoothed:       [ε/V, ε/V, 1-ε, ε/V, ε/V]

    where ε = smoothing (e.g., 0.1), V = vocab_size

    Benefits:
        - Prevents overconfidence
        - Better calibrated probabilities
        - Slight improvement in perplexity
    """
```

#### `CheckpointManager`
*Save/load model checkpoints*

```python
class CheckpointManager:
    """
    Manage model checkpoints

    Methods:
        save_checkpoint(model, optimizer, epoch, loss)
        load_checkpoint(path)
        get_best_checkpoint()
    """
```

#### `Logger`
*Training logger with TensorBoard support*

```python
class Logger:
    """
    Log training metrics

    Supports:
        - Console logging
        - File logging
        - TensorBoard logging
    """
```

#### `AttentionVisualizer`
*Visualize attention weights*

```python
class AttentionVisualizer:
    """
    Visualize attention patterns

    Creates heatmaps showing which tokens attend to which
    """
```

**File Size**: ~550 lines
**Dependencies**: `torch`, `numpy`, `matplotlib`

---

### 7. `mha/inference.py`

**Purpose**: Text generation and decoding strategies

**Key Components**:

#### `greedy_decode()`
*Harvard NLP's greedy decoding function*

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Greedy decoding for autoregressive generation

    Process:
        1. Encode source sequence once
        2. Start with <BOS> token
        3. For each position:
           - Decode current sequence
           - Take argmax (greedy choice)
           - Append to sequence
        4. Stop at max_len or <EOS>

    Args:
        model: EncoderDecoder model
        src: Source sequence (batch=1, src_len)
        src_mask: Source mask
        max_len: Maximum tokens to generate
        start_symbol: Start token ID (usually <BOS>)

    Returns:
        Generated sequence (1, generated_len)
    """
```

#### `TextGenerator`
*Class with multiple generation strategies*

```python
class TextGenerator:
    """
    Text generation with multiple decoding methods

    Methods:
        generate_greedy(prompt, max_length)
            - Always pick most likely token
            - Deterministic
            - Fast but can be repetitive

        generate_with_temperature(prompt, max_length, temperature)
            - Sample with temperature scaling
            - temperature < 1.0: more conservative
            - temperature > 1.0: more random
            - temperature = 1.0: normal sampling

        generate_top_k(prompt, max_length, k)
            - Sample from k most likely tokens
            - Filters out unlikely tokens
            - k=50 is common choice

        generate_nucleus(prompt, max_length, p)
            - Sample from smallest set with cumulative prob > p
            - Dynamic vocabulary size
            - p=0.9 is common choice
    """
```

**Generation comparison**:
```python
generator = TextGenerator(model, tokenizer, device='cuda')

# Greedy (deterministic)
text = generator.generate_greedy("The transformer", max_length=50)
# Output: "The transformer is a model that uses attention..."

# Temperature (controlled randomness)
text = generator.generate_with_temperature("The transformer", temperature=0.8)
# Output: "The transformer architecture was introduced..."

# Top-k (sample from top k)
text = generator.generate_top_k("The transformer", k=50)
# Output: "The transformer model has become..."

# Nucleus/top-p (dynamic sampling)
text = generator.generate_nucleus("The transformer", p=0.9)
# Output: "The transformer represents a breakthrough..."
```

**File Size**: ~320 lines
**Dependencies**: `torch`, `torch.nn.functional`

---

### 8. `mha/data_loader.py`

**Purpose**: Load and process WikiText-2 dataset

**Key Components**:

#### `WikiTextDataModule`
*Data loading and preprocessing*

```python
class WikiTextDataModule:
    """
    WikiText-2 data loading

    Methods:
        prepare_data()      # Download and preprocess
        setup()             # Create train/val/test splits
        get_dataloaders()   # Create PyTorch DataLoaders

    Preprocessing:
        1. Download WikiText-2 from HuggingFace
        2. Tokenize with GPT-2 tokenizer
        3. Chunk into fixed-length sequences
        4. Add padding/attention masks
        5. Save to disk
    """
```

#### `load_config()`
*Load training configuration from JSON*

```python
def load_config(config_path='config.json'):
    """
    Load training hyperparameters

    Returns dict with:
        - Model params (d_model, num_heads, etc.)
        - Training params (batch_size, epochs, etc.)
        - Data paths
    """
```

**File Size**: ~400 lines
**Dependencies**: `datasets`, `transformers`, `torch`

---

### 9. `mha/train.py`

**Purpose**: Standalone training script

**Can be run from command line:**
```bash
python -m mha.train --config config.json
```

**What it does**:
1. Load configuration
2. Setup data loaders
3. Create model with `make_model()`
4. Setup optimizer and scheduler with `rate()`
5. Train loop using `Batch` class
6. Validate each epoch
7. Save checkpoints
8. Log metrics

**File Size**: ~300 lines
**Dependencies**: All mha modules

---

### 10. `mha/config.json`

**Purpose**: Training configuration file

```json
{
  "model": {
    "src_vocab": 50257,
    "tgt_vocab": 50257,
    "N": 6,
    "d_model": 512,
    "d_ff": 2048,
    "h": 8,
    "dropout": 0.1,
    "max_seq_length": 512
  },
  "training": {
    "batch_size": 8,
    "num_epochs": 20,
    "learning_rate": 1.0,
    "warmup_steps": 4000,
    "gradient_clip": 1.0
  },
  "data": {
    "train_path": "./data_processed/wikitext2_processed/train",
    "val_path": "./data_processed/wikitext2_processed/validation"
  }
}
```

---

## Training Pipeline

### Complete Training Flow (Harvard NLP Method)

```
1. DATA PREPARATION
   ├── Download WikiText-2 (via datasets library)
   ├── Tokenize with GPT-2 tokenizer
   ├── Chunk into sequences of 512 tokens
   ├── Add padding and attention masks
   └── Save to disk (data_processed/)

2. MODEL CREATION
   └── model = make_model(
           src_vocab=50257,
           tgt_vocab=50257,
           N=6,              # 6 encoder + 6 decoder layers
           d_model=512,      # Model dimension
           d_ff=2048,        # FFN hidden dimension
           h=8,              # 8 attention heads
           dropout=0.1
       )

3. OPTIMIZER SETUP
   ├── optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
   └── scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step+1, 512, 1.0, 4000))

4. LOSS FUNCTION
   └── criterion = nn.NLLLoss(ignore_index=pad_token_id, reduction='sum')

5. TRAINING LOOP (Per Epoch)
   ├── For each batch:
   │   ├── Create Batch object (automatic masking)
   │   │   batch = Batch(src=input_ids, tgt=input_ids, pad=pad_token_id)
   │   │
   │   ├── Forward pass
   │   │   output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
   │   │   log_probs = model.generator(output)
   │   │
   │   ├── Compute loss
   │   │   loss_sum = criterion(log_probs.reshape(-1, vocab), batch.tgt_y.reshape(-1))
   │   │   loss = loss_sum / batch.ntokens  # Normalize by token count
   │   │
   │   ├── Backward pass
   │   │   optimizer.zero_grad()
   │   │   loss.backward()
   │   │   clip_grad_norm_(model.parameters(), 1.0)
   │   │
   │   └── Update weights
   │       optimizer.step()
   │       scheduler.step()  # Learning rate schedule
   │
   ├── Validation
   │   └── Same forward pass without gradients
   │
   ├── Generate samples (monitor progress)
   │   └── generator.generate_with_temperature("The Roman Empire", temperature=0.8)
   │
   ├── Save checkpoint if best
   │   └── torch.save({'model': model.state_dict(), ...}, 'best_model.pt')
   │
   └── Early stopping check
       └── Stop if val loss doesn't improve for 3 epochs

6. FINAL EVALUATION
   ├── Load best checkpoint
   ├── Generate text samples
   ├── Calculate perplexity
   └── Compute quality metrics
```

### Training Files

#### **Primary Training Method** (RECOMMENDED)
**`notebooks/train_transformer_v2.ipynb`**

**Purpose**: Complete Colab training notebook with Harvard NLP patterns

**Features**:
- ✅ Uses `Batch` class (proper Harvard NLP way)
- ✅ Extended training (20 epochs)
- ✅ Generation monitoring after each epoch
- ✅ Quality metrics (repetition, diversity)
- ✅ Early stopping
- ✅ LR schedule visualization
- ✅ Module reload (fixes Colab caching)
- ✅ Experimental mode (data subsetting)

**Structure**:
```
31 cells organized into sections:

1. Setup & Installation (4 cells)
   - GPU check
   - Google Drive mount
   - Clone repo & install
   - Import modules

2. Configuration (2 cells)
   - Hyperparameters
   - Data subset options for experiments

3. Data Loading (3 cells)
   - Load WikiText-2
   - Create DataLoaders
   - Show dataset info

4. Model Creation (1 cell)
   - make_model() with parameter counting

5. Visualization (1 cell)
   - LR schedule plot

6. Training Setup (2 cells)
   - Optimizer & scheduler
   - Loss function

7. Training Utilities (4 cells)
   - EarlyStopping class
   - train_epoch() function (uses Batch class!)
   - validate() function (uses Batch class!)
   - evaluate_generation() function

8. Main Training Loop (1 cell)
   - 20 epochs
   - Checkpoint saving
   - Progress monitoring

9. Visualization (1 cell)
   - 4-panel plot (loss, perplexity, grad norm, time)

10. Module Reload (1 cell)
    - Fix Python caching issues

11. Final Evaluation (5 cells)
    - Load best model
    - Comprehensive generation test
    - Quality metrics
    - Summary report
    - Conclusion
```

**Usage**:
```python
# 1. Open in Google Colab
# 2. Upload data_processed.zip
# 3. Run all cells
# 4. For experiments, modify config:
config['train_subset_size'] = 362  # 10% of data for 40min training
config['val_subset_size'] = 22
```

#### **Alternative Training Method**
**`mha/train.py`**

**Purpose**: Standalone command-line training script

**Usage**:
```bash
python -m mha.train --config config.json
```

**Features**:
- Command-line interface
- Uses config.json
- No notebook needed
- Good for remote servers

---

## Data Flow

### Forward Pass (Training)

```python
# Step-by-step data flow through the model

# 1. Input preparation
input_ids = torch.tensor([[1, 45, 234, 567, 2]])  # (batch=1, seq_len=5)

# 2. Create Batch (Harvard NLP way)
batch = Batch(src=input_ids, tgt=input_ids, pad=2)
# batch.src:      [1, 45, 234, 567, 2]      # Full sequence
# batch.tgt:      [1, 45, 234, 567]         # Excludes last token
# batch.tgt_y:    [45, 234, 567, 2]         # Excludes first token
# batch.src_mask: [1, 1, 1, 1, 1]           # All visible
# batch.tgt_mask: [[1, 0, 0, 0],            # Causal mask
#                  [1, 1, 0, 0],
#                  [1, 1, 1, 0],
#                  [1, 1, 1, 1]]

# 3. Embedding + Positional Encoding (Encoder side)
src_embedded = model.src_embed(batch.src)
# Shape: (1, 5, 512)
# Each token -> 512-dim vector + position encoding

# 4. Encoder
memory = model.encoder(src_embedded, batch.src_mask)
# Shape: (1, 5, 512)
# Process:
#   For each of 6 layers:
#     - Self-attention (all positions can attend to all)
#     - Feed-forward
# Output: Encoded representation of source

# 5. Embedding + Positional Encoding (Decoder side)
tgt_embedded = model.tgt_embed(batch.tgt)
# Shape: (1, 4, 512)

# 6. Decoder
decoder_output = model.decoder(tgt_embedded, memory, batch.src_mask, batch.tgt_mask)
# Shape: (1, 4, 512)
# Process:
#   For each of 6 layers:
#     - Masked self-attention (can't see future)
#     - Cross-attention (attend to encoder output)
#     - Feed-forward

# 7. Generator (project to vocabulary)
log_probs = model.generator(decoder_output)
# Shape: (1, 4, 50257)
# Each position -> probability distribution over vocabulary

# 8. Loss calculation
loss = criterion(
    log_probs.reshape(-1, 50257),  # (4, 50257)
    batch.tgt_y.reshape(-1)         # (4,)
)
# Measures how well predicted distribution matches actual next tokens

# 9. Normalize by number of tokens
loss = loss / batch.ntokens
```

### Attention Flow (Inside a Layer)

```python
# Multi-head attention breakdown

# Input: (batch=1, seq_len=5, d_model=512)
x = torch.randn(1, 5, 512)

# 1. Linear projections
Q = linear_q(x)  # Query:  (1, 5, 512)
K = linear_k(x)  # Key:    (1, 5, 512)
V = linear_v(x)  # Value:  (1, 5, 512)

# 2. Split into h=8 heads
Q = Q.view(1, 5, 8, 64).transpose(1, 2)  # (1, 8, 5, 64)
K = K.view(1, 5, 8, 64).transpose(1, 2)  # (1, 8, 5, 64)
V = V.view(1, 5, 8, 64).transpose(1, 2)  # (1, 8, 5, 64)

# 3. Attention scores
scores = Q @ K.transpose(-2, -1) / sqrt(64)  # (1, 8, 5, 5)
# Each position computes similarity with every other position

# 4. Apply mask (if decoder)
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

# 5. Attention probabilities
attn_probs = softmax(scores, dim=-1)  # (1, 8, 5, 5)
# Each row sums to 1

# 6. Apply attention to values
output = attn_probs @ V  # (1, 8, 5, 64)

# 7. Concatenate heads
output = output.transpose(1, 2).contiguous().view(1, 5, 512)

# 8. Final projection
output = linear_out(output)  # (1, 5, 512)
```

### Generation Flow (Inference)

```python
# Autoregressive text generation

prompt = "The transformer"
input_ids = tokenizer.encode(prompt)  # [464, 47385]

# Iteratively generate tokens
for i in range(max_length):
    # 1. Forward pass
    output = model(input_ids, input_ids, src_mask, tgt_mask)

    # 2. Get logits for last position
    logits = model.generator(output[:, -1, :])  # (1, 50257)

    # 3. Sample next token (various strategies)
    if greedy:
        next_token = argmax(logits)
    elif temperature:
        next_token = sample(softmax(logits / temp))
    elif top_k:
        next_token = sample_from_topk(logits, k)
    elif nucleus:
        next_token = sample_nucleus(logits, p)

    # 4. Append to sequence
    input_ids = torch.cat([input_ids, next_token], dim=1)

    # 5. Stop if EOS
    if next_token == eos_token_id:
        break

# Decode to text
generated_text = tokenizer.decode(input_ids)
```

---

## Usage Examples

### Example 1: Create Model (Harvard NLP Way)

```python
from mha import make_model

# Create standard transformer
model = make_model(
    src_vocab=50257,    # GPT-2 vocabulary size
    tgt_vocab=50257,
    N=6,                # 6 layers
    d_model=512,        # Model dimension
    d_ff=2048,          # FFN dimension
    h=8,                # 8 attention heads
    dropout=0.1
)

# Model is ready to use with proper Xavier initialization!
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Parameters: 44,418,048
```

### Example 2: Training Loop (Harvard NLP Way)

```python
from mha import make_model, Batch, rate
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Setup
model = make_model(50257, 50257, N=6)
optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step+1, 512, 1.0, 4000))
criterion = nn.NLLLoss(ignore_index=pad_token_id, reduction='sum')

# Training loop
model.train()
for epoch in range(num_epochs):
    for input_ids in dataloader:
        # Harvard NLP pattern: Use Batch class
        batch = Batch(src=input_ids, tgt=input_ids, pad=pad_token_id)

        # Forward
        output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        log_probs = model.generator(output)

        # Loss
        loss = criterion(log_probs.reshape(-1, vocab_size), batch.tgt_y.reshape(-1))
        loss = loss / batch.ntokens  # Normalize

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update
        optimizer.step()
        scheduler.step()
```

### Example 3: Text Generation

```python
from mha import make_model
from mha.inference import TextGenerator
from transformers import GPT2Tokenizer

# Load model
model = make_model(50257, 50257, N=6)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
model.eval()

# Create generator
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = TextGenerator(model, tokenizer, device='cuda')

# Generate with different methods
prompt = "The transformer architecture"

# Greedy (deterministic)
text = generator.generate_greedy(prompt, max_length=50)

# Temperature sampling (controlled randomness)
text = generator.generate_with_temperature(prompt, temperature=0.8, max_length=50)

# Top-k sampling
text = generator.generate_top_k(prompt, k=50, max_length=50)

# Nucleus sampling (top-p)
text = generator.generate_nucleus(prompt, p=0.9, max_length=50)

print(text)
```

### Example 4: Attention Visualization

```python
from mha.utils import AttentionVisualizer

# Extract attention weights during forward pass
model.eval()
with torch.no_grad():
    output = model(src, tgt, src_mask, tgt_mask)

    # Get attention from specific layer
    attn_weights = model.decoder.layers[0].self_attn.attn  # (batch, h, seq, seq)

# Visualize
visualizer = AttentionVisualizer()
visualizer.plot_attention(
    attn_weights[0, 0],  # First head
    src_tokens,
    tgt_tokens,
    title="Decoder Self-Attention (Layer 1, Head 1)"
)
```

### Example 5: Using Label Smoothing

```python
from mha.utils import LabelSmoothing

# Replace NLLLoss with LabelSmoothing
criterion = LabelSmoothing(
    vocab_size=50257,
    padding_idx=pad_token_id,
    smoothing=0.1  # Distribute 10% probability mass
)

# Use same as NLLLoss
loss = criterion(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
```

---

## Key Concepts

### 1. Why Harvard NLP Patterns?

**The `Batch` class example:**

**❌ Manual approach** (error-prone):
```python
src = input_ids[:, :-1]
tgt_input = input_ids[:, :-1]
tgt_output = labels[:, 1:]
src_mask = (src != pad).unsqueeze(-2)
tgt_mask = create_combined_mask(tgt_input, pad, causal=True)
num_tokens = (tgt_output != pad).sum().item()
```

**✅ Harvard NLP way** (clean, automatic):
```python
batch = Batch(src=input_ids, tgt=input_ids, pad=pad_token_id)
# Everything created automatically!
```

Benefits:
- Less code
- Fewer bugs
- Automatic masking
- Clear intent
- Educational

### 2. Teacher Forcing

During training, we use **teacher forcing**:

```python
# Instead of feeding model's predictions back as input
# We feed the ACTUAL target tokens (even when model is wrong)

Input:  [<BOS>, "The", "cat", "sat"]
Target: ["The", "cat", "sat", "on"]

# Decoder sees: [<BOS>, "The", "cat", "sat"]
# Predicts:     ["The", "cat", "sat", "on"]
# Even if it predicted "dog" instead of "cat", it still sees "cat" next
```

Why?
- Faster convergence
- More stable training
- Prevents error accumulation

### 3. Masking Types

**Padding mask** (hide padding tokens):
```python
# Sequence: ["The", "cat", "<PAD>", "<PAD>"]
# Mask:     [1, 1, 0, 0]
```

**Causal mask** (decoder can't see future):
```python
# For sequence of length 4:
[[1, 0, 0, 0],   # Token 0 can only see token 0
 [1, 1, 0, 0],   # Token 1 can see tokens 0-1
 [1, 1, 1, 0],   # Token 2 can see tokens 0-2
 [1, 1, 1, 1]]   # Token 3 can see tokens 0-3
```

**Combined mask** (both):
```python
# Apply both padding and causal mask
combined = padding_mask & causal_mask
```

### 4. Learning Rate Warmup

Why warmup?
- Adam with high initial LR can be unstable
- Gradients are large early in training
- Warmup gradually increases LR

```python
# rate() function behavior:
# Steps 0-4000: Linear increase (warmup)
# Steps 4000+:  Inverse sqrt decay

lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```

### 5. Perplexity

**Definition**: `perplexity = exp(cross_entropy_loss)`

**Interpretation**:
- Perplexity of 100 = model is "confused" between ~100 choices per token
- Lower is better
- Good models: 30-60 (on test set)
- Our WikiText-2 training: 180-220 (reasonable for 20 epochs)

### 6. Model Size vs Performance

Typical configurations:

| Size | N | d_model | d_ff | h | Params | Use Case |
|------|---|---------|------|---|--------|----------|
| **Tiny** | 4 | 256 | 1024 | 4 | ~11M | Debugging |
| **Small** | 6 | 512 | 2048 | 8 | ~44M | Our implementation |
| **Base** | 6 | 768 | 3072 | 12 | ~110M | BERT base |
| **Large** | 12 | 768 | 3072 | 12 | ~210M | BERT large |
| **XL** | 24 | 1024 | 4096 | 16 | ~340M | GPT-2 large |

### 7. Gradient Clipping

Why needed?
- Prevents exploding gradients
- Common in transformers
- Clips gradient norm to max value

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 8. Xavier Initialization

`make_model()` applies Xavier uniform initialization:

```python
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

Why?
- Maintains variance through layers
- Prevents vanishing/exploding activations
- Recommended in original paper

---

## Summary

### Package Philosophy

This implementation prioritizes:
1. **Education**: Clear, well-documented code following Harvard NLP
2. **Modularity**: Each component in separate file
3. **Correctness**: Faithful to original paper
4. **Usability**: Clean APIs and helpful utilities

### Harvard NLP Patterns Used

| Pattern | File | Purpose |
|---------|------|---------|
| `make_model()` | transformer.py | Factory function for model creation |
| `Batch` class | utils.py | Automatic masking and token counting |
| `rate()` | utils.py | Learning rate schedule from paper |
| `attention()` | attention.py | Core attention computation |
| `clones()` | layers.py | Create N identical layers |
| `greedy_decode()` | inference.py | Autoregressive generation |
| `subsequent_mask()` | attention.py | Causal mask for decoder |

### Training Pipeline Summary

```
Data → Tokenize → Batch → Model → Loss → Backprop → Update
                     ↓
              (Automatic masking)
```

### File Dependency Graph

```
transformer.py (main architecture)
    ├── attention.py (multi-head attention)
    ├── layers.py (FFN, LayerNorm, etc.)
    ├── positional_encoding.py (position info)
    └── utils.py (Batch, rate, etc.)

inference.py (text generation)
    └── Uses EncoderDecoder from transformer.py

train.py (training script)
    ├── Uses make_model() from transformer.py
    ├── Uses Batch, rate from utils.py
    └── Uses WikiTextDataModule from data_loader.py

notebooks/train_transformer_v2.ipynb (training notebook)
    └── Same dependencies as train.py
```

---

## Resources

- **Harvard NLP Annotated Transformer**: https://nlp.seas.harvard.edu/annotated-transformer/
- **Original Paper**: "Attention is All You Need" (Vaswani et al., 2017)
- **Repository**: https://github.com/mohamedAtoui/LLM-Journey
- **Illustrated Transformer**: http://jalammar.github.io/illustrated-transformer/

---

**This documentation provides a complete understanding of the `mha` package architecture, training pipeline, and Harvard NLP patterns used throughout the implementation.**
