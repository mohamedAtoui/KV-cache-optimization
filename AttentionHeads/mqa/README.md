# Multi-Query Attention (MQA) Package

GPTNeo decoder-only transformer with Multi-Query Attention for faster inference and reduced memory usage.

## What is Multi-Query Attention?

Multi-Query Attention (MQA) is a variant of multi-head attention where **keys and values are shared across all attention heads**, while queries still have multiple heads.

Based on: **"Fast Transformer Decoding: One Write-head is All You Need"** (Shazeer 2019)
Paper: https://arxiv.org/abs/1911.02150

### Key Difference from MHA

| Component | Multi-Head Attention (MHA) | Multi-Query Attention (MQA) |
|-----------|---------------------------|------------------------------|
| **Query (Q)** | h separate heads | h separate heads ✓ |
| **Key (K)** | h separate heads | **1 shared across all heads** |
| **Value (V)** | h separate heads | **1 shared across all heads** |
| **Parameters** | More | **44% fewer** |
| **Inference Speed** | Slower | **Faster** |
| **KV Cache Size** | Larger | **Smaller** |

### Architecture Diagram

```
MHA:  Q₁ K₁ V₁  Q₂ K₂ V₂  ...  Qₕ Kₕ Vₕ  →  Concat  →  Output
      ↓  ↓  ↓   ↓  ↓  ↓        ↓  ↓  ↓
      Attention  Attention      Attention

MQA:  Q₁  Q₂  ...  Qₕ  →  All use same K, V  →  Concat  →  Output
       ↓   ↓        ↓
       └───┴────────┘
             ↓
        Shared K, V
```

## Benefits

### 1. Faster Inference
- **Less memory bandwidth**: K and V are loaded once, used by all Q heads
- **Smaller KV cache**: Critical for autoregressive generation
- **Better throughput**: Especially noticeable during text generation

### 2. Fewer Parameters
With default config (hidden_size=256, num_heads=8):
- **MHA attention params**: ~262K per block
- **MQA attention params**: ~147K per block
- **Reduction**: 44% fewer parameters

### 3. Minor Quality Trade-off
- Paper reports "only minor quality degradation"
- Good for resource-constrained deployments
- Trade inference speed for slight accuracy loss

## Usage

### Installation

Same as MHA package:
```bash
cd AttentionHeads
pip install -r requirements.txt
pip install -e .
```

### Training

```python
from mqa import GPTNeoForCausalLM, create_gptneo_model
from mha import GPTNeoTrainer  # Training code is shared
import json

# Load MQA configuration
with open('mqa/config.json', 'r') as f:
    config = json.load(f)

# Create MQA model
model = create_gptneo_model(config['model'])

# Train (same as MHA)
trainer = GPTNeoTrainer(config)
trainer.train()
```

### Inference

```python
from mqa import GPTNeoForCausalLM
from transformers import GPT2Tokenizer
import torch

# Load model
model = GPTNeoForCausalLM(
    vocab_size=50257,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    intermediate_size=1024,
    max_position_embeddings=256,
    dropout=0.2
)

# Load trained weights
model.load_state_dict(torch.load('checkpoints/mqa/best_model.pt'))

# Generate
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

story = tokenizer.decode(output[0])
print(story)
```

## Configuration

MQA uses the same successful hyperparameters as MHA:

```json
{
  "model": {
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 8,
    "intermediate_size": 1024,
    "dropout": 0.2
  },
  "training": {
    "learning_rate": 5e-5,      // Critical for stability
    "warmup_steps": 600,          // 10% of training
    "gradient_clip": 0.5,         // Tight control
    "effective_batch_size": 256,  // Via gradient accumulation
    "use_bf16": true              // L4 GPU optimization
  }
}
```

See `config.json` for full configuration.

## Comparison: MHA vs MQA

| Metric | MHA | MQA | Difference |
|--------|-----|-----|------------|
| **Attention Parameters/Block** | ~262K | ~147K | **-44%** |
| **Total Model Parameters** | ~16M | ~12M | **-25%** |
| **Training Time** | ~45-60 min | ~40-55 min | **Slightly faster** |
| **Inference Speed** | Baseline | **1.2-1.5× faster** | **Better** |
| **KV Cache Size** | Baseline | **8× smaller** (h=8) | **Much better** |
| **Model Quality** | Baseline | **Minor degradation** | **Acceptable** |

## Package Structure

```
mqa/
├── __init__.py          # Package exports, imports from mha
├── attention.py         # MultiQueryAttention implementation
├── transformer.py       # GPTNeo with MQA
├── config.json          # MQA configuration
└── README.md            # This file
```

**All other modules imported from `mha/`**:
- `layers.py` - LayerNorm, FFN (same)
- `train.py` - Training loop (same)
- `data_loader.py` - TinyStories data (same)
- `utils.py` - Metrics, logging (same)

## When to Use MQA

### Good For:
- **Inference-heavy applications**: Chatbots, text generation services
- **Resource-constrained environments**: Mobile, edge devices
- **Large batch sizes**: Serving multiple users
- **Long sequences**: Benefits scale with sequence length

### Maybe Not For:
- **Training-only scenarios**: Benefits mainly during inference
- **Maximum quality requirements**: MHA might be slightly better
- **Small models**: Overhead of shared K/V less significant

## Implementation Details

### Parameter Calculation

**MQA Attention Block:**
```
Q projection: d_model × d_model = 256 × 256 = 65,536
K projection: d_model × d_k = 256 × 32 = 8,192   (shared!)
V projection: d_model × d_k = 256 × 32 = 8,192   (shared!)
Output:       d_model × d_model = 256 × 256 = 65,536
Total:        147,456 params
```

**vs MHA:**
```
Q, K, V projections: 3 × (d_model × d_model) = 196,608
Output:              d_model × d_model = 65,536
Total:               262,144 params
```

### Code Architecture

The implementation reuses everything from `mha/` except:
1. **attention.py**: New `MultiQueryAttention` class
2. **transformer.py**: Uses MQA instead of MHA

This minimal approach:
- ✅ No code duplication
- ✅ Easy to maintain
- ✅ Direct comparison possible
- ✅ Shared training infrastructure

## Expected Results

Using the same successful hyperparameters as MHA:

| Metric | Expected Value |
|--------|----------------|
| **Validation Loss** | 3.6-4.0 (similar to MHA) |
| **Validation Perplexity** | 36-55 (minor degradation acceptable) |
| **Training Time** | 40-55 minutes on L4 |
| **GPU Memory** | 7-9 GB (slightly less than MHA) |
| **Inference Speed** | 1.2-1.5× faster than MHA |

## References

1. **Shazeer, N. (2019)**. Fast Transformer Decoding: One Write-head is All You Need. arXiv:1911.02150.
2. **Eldan, R., & Li, Y. (2023)**. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.

## License

MIT License - see LICENSE file for details.

## Author

Attaimen (wmis066@live.rhul.ac.uk)
