# Grouped Query Attention (GQA)

Implementation based on ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023).

## Overview

GQA is an interpolation between Multi-Head Attention (MHA) and Multi-Query Attention (MQA):

| Attention Type | Query Heads | KV Heads | Description |
|---------------|-------------|----------|-------------|
| MHA           | h           | h        | Each query head has its own K,V |
| MQA           | h           | 1        | All query heads share single K,V |
| GQA           | h           | g        | Groups of query heads share K,V (1 < g < h) |

## Key Benefits

- **Better quality than MQA**: More KV heads means more expressiveness
- **Smaller KV cache than MHA**: Fewer KV heads reduces memory bandwidth
- **Configurable trade-off**: Choose `num_kv_heads` based on quality vs efficiency needs

## Configuration Options

With `num_heads=8`:

| Config | num_kv_heads | Query heads per KV | Character |
|--------|--------------|-------------------|-----------|
| GQA-2  | 2            | 4                 | More efficient, closer to MQA |
| GQA-4  | 4            | 2                 | Better quality, closer to MHA |

## Files

- `attention.py` - `GroupedQueryAttention` class
- `transformer.py` - GPTNeo architecture with GQA
- `config_kv2.json` - Configuration with num_kv_heads=2
- `config_kv4.json` - Configuration with num_kv_heads=4

## Usage

```python
from AttentionHeads.gqa import GroupedQueryAttention, GPTNeoForCausalLM

# Create GQA attention layer
attn = GroupedQueryAttention(h=8, d_model=512, num_kv_heads=2)

# Create full model
model = GPTNeoForCausalLM(
    vocab_size=50257,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    num_kv_heads=2,  # GQA configuration
    intermediate_size=1024
)
```

## Parameter Comparison (d_model=256, h=8)

| Attention | Attention Params | vs MHA |
|-----------|-----------------|--------|
| MHA       | 262,144         | -      |
| GQA-4     | 196,608         | -25%   |
| GQA-2     | 163,840         | -37.5% |
| MQA       | 147,456         | -43.75%|

## References

```bibtex
@article{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebron, Federico and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023}
}
```
