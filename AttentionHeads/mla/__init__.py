"""
Multi-Head Latent Attention (MLA) Module

Implementation based on "DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model" (DeepSeek-AI, 2024)
https://arxiv.org/abs/2405.04434

MLA compresses key-value representations into a low-rank latent space,
reducing KV-cache memory while maintaining quality through:
- Low-rank KV compression (d_model -> d_c -> h*d_head)
- Decoupled RoPE applied to separate rope-specific projections
- Standard multi-head queries

Example usage:
    from AttentionHeads.mla import MultiHeadLatentAttention, GPTNeoForCausalLM

    # Create MLA attention layer
    attn = MultiHeadLatentAttention(h=8, d_model=256, d_c=128, d_rope=16)

    # Create full model with MLA
    model = GPTNeoForCausalLM(vocab_size=50257, num_heads=8, d_c=128, d_rope=16)

Reference:
    DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434.
"""

from .attention import MultiHeadLatentAttention, attention
from .transformer import (
    GPTNeoBlock,
    GPTNeoModel,
    GPTNeoForCausalLM,
    GPTNeo,
    create_gptneo_model
)

__all__ = [
    'MultiHeadLatentAttention',
    'attention',
    'GPTNeoBlock',
    'GPTNeoModel',
    'GPTNeoForCausalLM',
    'GPTNeo',
    'create_gptneo_model'
]
