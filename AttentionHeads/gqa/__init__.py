"""
Grouped Query Attention (GQA) Module

Implementation based on "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

GQA is an interpolation between MHA and MQA:
- MHA: h query heads, h KV heads
- MQA: h query heads, 1 KV head
- GQA: h query heads, g KV heads (1 < g < h)

Example usage:
    from AttentionHeads.gqa import GroupedQueryAttention, GPTNeoForCausalLM

    # Create GQA attention layer
    attn = GroupedQueryAttention(h=8, d_model=512, num_kv_heads=2)

    # Create full model with GQA
    model = GPTNeoForCausalLM(vocab_size=50257, num_heads=8, num_kv_heads=2)
"""

from .attention import GroupedQueryAttention, attention
from .transformer import (
    GPTNeoBlock,
    GPTNeoModel,
    GPTNeoForCausalLM,
    GPTNeo,
    create_gptneo_model
)

__all__ = [
    'GroupedQueryAttention',
    'attention',
    'GPTNeoBlock',
    'GPTNeoModel',
    'GPTNeoForCausalLM',
    'GPTNeo',
    'create_gptneo_model'
]
