"""
Rotary Position Embedding (RoPE) Implementation

Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
https://arxiv.org/abs/2104.09864

RoPE encodes position information directly into the attention computation by rotating
query and key vectors. This allows relative position information to naturally emerge
from the dot product of rotated vectors.

Shared utility imported by all attention variants (MHA, MQA, GQA, MLA).

Reference:
    Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
    RoFormer: Enhanced Transformer with Rotary Position Embedding.
    arXiv preprint arXiv:2104.09864.
"""

import torch
import torch.nn as nn


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    Precompute cosine and sine frequencies for rotary embeddings.

    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for the frequency computation (default: 10000.0)

    Returns:
        cos: (max_seq_len, dim) cosine frequencies
        sin: (max_seq_len, dim) sine frequencies
    """
    # Compute frequency bands: theta_i = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Compute position * frequency: (max_seq_len, dim/2)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)

    # Duplicate to full dimension: (max_seq_len, dim)
    cos = freqs.cos().repeat(1, 2)
    sin = freqs.sin().repeat(1, 2)

    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary embeddings to input tensor.

    Rotates pairs of dimensions: for each pair (x_2i, x_{2i+1}),
    applies rotation by angle theta_i * position.

    Args:
        x: Input tensor of shape (..., seq_len, dim)
        cos: Cosine frequencies (seq_len, dim) or broadcastable
        sin: Sine frequencies (seq_len, dim) or broadcastable

    Returns:
        Rotated tensor of same shape as x
    """
    # Split into pairs and rotate
    # x_rotated = [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)

    return x * cos + x_rotated * sin


if __name__ == "__main__":
    print("Testing Rotary Position Embedding (RoPE)...")
    print("=" * 70)

    # Test 1: precompute_freqs_cis
    print("\n1. Testing precompute_freqs_cis...")
    dim = 32
    max_seq_len = 256
    cos, sin = precompute_freqs_cis(dim, max_seq_len)
    print(f"   dim={dim}, max_seq_len={max_seq_len}")
    print(f"   cos shape: {cos.shape}")
    print(f"   sin shape: {sin.shape}")
    assert cos.shape == (max_seq_len, dim), f"Expected ({max_seq_len}, {dim}), got {cos.shape}"
    assert sin.shape == (max_seq_len, dim), f"Expected ({max_seq_len}, {dim}), got {sin.shape}"
    print("   Frequencies precomputed correctly")

    # Test 2: apply_rotary_emb
    print("\n2. Testing apply_rotary_emb...")
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 32

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    cos_slice = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin_slice = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x_rotated = apply_rotary_emb(x, cos_slice, sin_slice)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_rotated.shape}")
    assert x_rotated.shape == x.shape, "Shape mismatch after rotation"
    print("   Rotation applied correctly")

    # Test 3: Verify rotation preserves norm
    print("\n3. Testing norm preservation...")
    input_norm = torch.norm(x, dim=-1)
    output_norm = torch.norm(x_rotated, dim=-1)
    norm_diff = (input_norm - output_norm).abs().max().item()
    print(f"   Max norm difference: {norm_diff:.8f}")
    assert norm_diff < 1e-5, "RoPE should preserve vector norms"
    print("   Norm preservation verified")

    # Test 4: Verify relative position property
    print("\n4. Testing relative position encoding property...")
    # For two positions p1 and p2, the dot product of rotated vectors
    # should depend only on (p2 - p1), not on absolute positions
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Rotate at positions 0 and 5
    cos_0 = cos[0:1].unsqueeze(0).unsqueeze(0)
    sin_0 = sin[0:1].unsqueeze(0).unsqueeze(0)
    cos_5 = cos[5:6].unsqueeze(0).unsqueeze(0)
    sin_5 = sin[5:6].unsqueeze(0).unsqueeze(0)
    dot_0_5 = (apply_rotary_emb(q, cos_0, sin_0) * apply_rotary_emb(k, cos_5, sin_5)).sum()

    # Rotate at positions 10 and 15 (same relative distance of 5)
    cos_10 = cos[10:11].unsqueeze(0).unsqueeze(0)
    sin_10 = sin[10:11].unsqueeze(0).unsqueeze(0)
    cos_15 = cos[15:16].unsqueeze(0).unsqueeze(0)
    sin_15 = sin[15:16].unsqueeze(0).unsqueeze(0)
    dot_10_15 = (apply_rotary_emb(q, cos_10, sin_10) * apply_rotary_emb(k, cos_15, sin_15)).sum()

    diff = abs(dot_0_5.item() - dot_10_15.item())
    print(f"   dot(q@pos0, k@pos5) = {dot_0_5.item():.6f}")
    print(f"   dot(q@pos10, k@pos15) = {dot_10_15.item():.6f}")
    print(f"   Difference: {diff:.8f}")
    assert diff < 1e-4, "Same relative distance should give same dot product"
    print("   Relative position property verified")

    print("\n" + "=" * 70)
    print("All RoPE tests passed!")
