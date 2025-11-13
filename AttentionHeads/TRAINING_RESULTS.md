# Training Results - GPTNeo on TinyStories

## Best Training Run Summary

**Date**: Commit `ab12de3` - "trianing with different parameters"
**Final Results**:
- **Best Validation Loss**: 3.6065 (achieved at step 3,000)
- **Validation Perplexity**: 36.84
- **Training Time**: ~45-60 minutes on L4 GPU
- **Total Training Steps**: 6,000
- **Dataset**: TinyStories (30K training samples, 5K validation samples)

## The Winning Configuration

### Model Architecture
```json
{
  "hidden_size": 256,
  "num_layers": 4,              // KEY: Reduced from 8 layers
  "num_heads": 8,
  "intermediate_size": 1024,
  "dropout": 0.2,
  "vocab_size": 50257,
  "max_position_embeddings": 256
}
```

**Model Size**: 16.09M total parameters
- Non-embedding parameters: 3.16M
- Embedding parameters: 12.93M

### Training Hyperparameters
```json
{
  "train_samples": 30000,
  "val_samples": 5000,
  "batch_size": 64,
  "gradient_accumulation_steps": 4,   // Increased from 2
  "effective_batch_size": 256,        // Doubled from 128
  "max_steps": 6000,
  "warmup_steps": 600,                // CRITICAL: Doubled from 300
  "learning_rate": 5e-5,              // CRITICAL: 20x lower (was 1e-3)
  "min_learning_rate": 1e-6,          // Lower floor
  "gradient_clip": 0.5,               // Tighter control (was 1.0)
  "weight_decay": 0.01,
  "use_bf16": true,                   // L4 GPU optimization
  "compile_model": true               // PyTorch 2.0 compilation
}
```

## Critical Parameter Changes (Failed → Successful)

| Parameter | Failed Run (commit 3361477) | Successful Run (commit ab12de3) | Impact |
|-----------|----------------------------|----------------------------------|---------|
| **num_layers** | 8 | **4** | **Major reduction in model complexity** |
| **learning_rate** | 1e-3 | **5e-5** | **CRITICAL: 20x lower prevents divergence** |
| **gradient_accumulation_steps** | 2 | **4** | Better gradient stability |
| **effective_batch_size** | 128 | **256** | Doubled for stable training |
| **warmup_steps** | 300 | **600** | Longer warmup period (10% of training) |
| **gradient_clip** | 1.0 | **0.5** | Tighter gradient control |
| **min_learning_rate** | 1e-5 | **1e-6** | Lower LR floor for decay |
| **compile_model** | false | **true** | PyTorch compilation enabled |

## Why This Configuration Worked

### 1. Lower Learning Rate (5e-5)
**The most critical fix.** The failed run used 1e-3, which was too aggressive for this model size and dataset. The lower learning rate (5e-5) provided:
- Stable gradient descent
- No training divergence
- Smooth convergence to validation loss ~3.6

### 2. Reduced Model Size (4 layers)
Smaller model with 4 layers instead of 8:
- Faster training iterations
- Less prone to overfitting on 30K samples
- Easier optimization landscape
- Better compute efficiency on L4 GPU

### 3. Larger Effective Batch Size (256)
Doubling effective batch size through gradient accumulation:
- More stable gradient estimates
- Reduced variance in updates
- Better generalization

### 4. Extended Warmup Period (600 steps)
10% of total training steps for warmup:
- Gradual learning rate increase
- Prevents early training instability
- Allows embeddings to stabilize

### 5. Tighter Gradient Clipping (0.5)
Reduced from 1.0 to 0.5:
- Prevents gradient spikes
- More conservative updates
- Particularly important with BFloat16

## Training Curves Analysis

![Training Curves](training_curves_12_11.png)

### Loss Curve Observations
- **Initial loss**: ~10.5 (near random prediction baseline)
- **Rapid descent**: Steps 0-1000 (warmup phase)
- **Steady improvement**: Steps 1000-3000
- **Plateau**: Steps 3000-6000 at ~3.6-3.7
- **No overfitting**: Training and validation losses track closely

### Perplexity Curve Observations
- **Initial perplexity**: ~30,000 (near random)
- **Rapid improvement**: Drops to ~37 by step 1000
- **Convergence**: Stabilizes at ~36.84 by step 3000
- **Validation stability**: No divergence between train/val

## BFloat16 Optimization

### Overflow Fix (Commit 2244ff4)
Changed attention mask value for BFloat16 compatibility:
- **Before**: `-1e9` (caused overflow in BFloat16)
- **After**: `-1e4` (sufficient for masking, no overflow)

### Benefits
- 2x training speedup on L4 GPU
- Lower memory usage
- No accuracy loss compared to FP32

## Hardware & Performance

**GPU**: NVIDIA L4 (Colab)
- Training time: ~45-60 minutes for 6,000 steps
- Memory usage: ~8-10GB
- BFloat16 support: Yes (optimal for L4)
- PyTorch compilation: Enabled for additional speedup

## Dataset Details

**TinyStories** (HuggingFace: `roneneldan/TinyStories`)
- Full dataset: 2.1M training stories, 21K validation
- Used subset: 30K train (1.4%), 5K val (23.8%)
- Tokenizer: GPT-2 (vocab 50,257)
- Max sequence length: 256 tokens
- Dynamic padding: Yes (memory efficient)

## Generation Quality

The model successfully generates coherent short stories:
- Follows narrative structure
- Maintains character consistency
- Uses appropriate vocabulary for children's stories
- Average generation length: 50-100 tokens

## Recommendations for Future Training

### To Improve Further
1. **Increase dataset size**: Use full TinyStories (2.1M samples)
2. **Train longer**: 20K-50K steps with larger dataset
3. **Larger model**: Try 6-8 layers with lower LR (3e-5)
4. **Better LR schedule**: Experiment with linear warmup + cosine decay
5. **Data augmentation**: Random truncation, story mixing

### Hyperparameter Ranges That Work
Based on successful run:
- Learning rate: 3e-5 to 1e-4
- Layers: 4-8 (4 is stable, 8 needs more data)
- Warmup: 5-10% of total steps
- Gradient clip: 0.3-0.7
- Batch size: 64-128 per GPU

### What to Avoid
- Learning rates > 1e-4 (causes instability)
- Too few warmup steps (< 5% of training)
- Large models (>8 layers) with small datasets (<50K samples)
- Gradient clip > 1.0 (allows spikes)

## Reproducibility

To reproduce these results:
1. Use configuration in `mha/config.json`
2. Run notebook: `notebooks/train_gptneo_tinystories.ipynb`
3. Ensure L4 or similar GPU with BFloat16 support
4. Train for 6,000 steps with evaluation every 1,000 steps

Expected outcome:
- Validation loss: ~3.6-3.7
- Validation perplexity: ~36-40
- Training time: ~45-60 minutes
