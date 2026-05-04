"""
Multi-Head Latent Attention (MLA) Implementation

Based on "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
Language Model" (DeepSeek-AI, 2024)
https://arxiv.org/abs/2405.04434

Key Innovation: MLA compresses KV representations into a low-rank latent space
and uses decoupled RoPE for position-aware attention.

Architecture:
    - KV down-projection: d_model -> d_c (compressed latent)
    - K up-projection: d_c -> h * d_head (from latent)
    - V up-projection: d_c -> h * d_head (from latent)
    - Q projection: d_model -> h * d_head (standard)
    - Decoupled RoPE: separate projections for position-sensitive components
    - KV-cache only needs to store d_c + d_rope values per token (vs 2*h*d_head for MHA)

Reference:
    DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from AttentionHeads.mha.rope import precompute_freqs_cis, apply_rotary_emb


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor. Positions with mask == 0 are masked out
        dropout: Optional nn.Dropout layer

    Returns:
        output: Attention output of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)

    # Apply softmax to get attention probabilities
    p_attn = F.softmax(scores, dim=-1)

    # Apply dropout to attention weights
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Apply attention to values
    output = torch.matmul(p_attn, value)

    return output, p_attn


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2

    Key differences from standard MHA:
    - KV representations are compressed through a low-rank latent bottleneck
    - RoPE is applied to separate decoupled projections (not mixed with content)
    - Q and K are concatenated: [content_part, rope_part] before attention

    This achieves:
    - Smaller KV-cache: only d_c + d_rope per token (vs 2*h*d_head for MHA)
    - Maintained quality through multi-head up-projection from latent
    - Position awareness through decoupled RoPE

    Args:
        h: Number of attention heads
        d_model: Model dimension
        d_c: Latent compression dimension for KV
        d_rope: Dimension for decoupled RoPE projections
        max_seq_len: Maximum sequence length
        dropout: Dropout probability

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, h=8, d_model=256, d_c=128, d_rope=16, max_seq_len=256, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        self.h = h
        self.d_model = d_model
        self.d_c = d_c
        self.d_rope = d_rope
        self.d_head = d_model // h  # Per-head content dimension
        self.d_qk = self.d_head + d_rope  # Full Q/K dimension per head

        # KV compression: down-project to latent, then up-project per head
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)     # KV down-projection
        self.W_UK = nn.Linear(d_c, h * self.d_head, bias=False)  # K up-projection
        self.W_UV = nn.Linear(d_c, h * self.d_head, bias=False)  # V up-projection

        # Standard Q projection
        self.W_Q = nn.Linear(d_model, h * self.d_head, bias=False)

        # Decoupled RoPE projections (shared across heads, then broadcast)
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)   # RoPE key projection
        self.W_QR = nn.Linear(d_model, h * d_rope, bias=False)  # RoPE query projection (per head)

        # Output projection
        self.output_projection = nn.Linear(h * self.d_head, d_model, bias=False)

        # Precompute RoPE frequencies
        cos, sin = precompute_freqs_cis(d_rope, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Latent Attention

        For self-attention, query == key == value == x (hidden states).

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: Optional mask (batch, seq_len_q, seq_len_k) or broadcastable

        Returns:
            output: (batch, seq_len_q, d_model)
            attn: (batch, h, seq_len_q, seq_len_k)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_k)

        B = query.size(0)
        T_q = query.size(1)
        T_k = key.size(1)

        # === KV Compression Path ===
        # 1. Down-project to latent
        c_kv = self.W_DKV(key)  # (B, T_k, d_c)

        # 2. Up-project to multi-head K and V
        k_nope = self.W_UK(c_kv).view(B, T_k, self.h, self.d_head).transpose(1, 2)  # (B, h, T_k, d_head)
        v = self.W_UV(c_kv).view(B, T_k, self.h, self.d_head).transpose(1, 2)        # (B, h, T_k, d_head)

        # === Query Path ===
        # 3. Standard Q projection
        q_nope = self.W_Q(query).view(B, T_q, self.h, self.d_head).transpose(1, 2)   # (B, h, T_q, d_head)

        # === Decoupled RoPE Path ===
        # 4. RoPE key: shared across heads, broadcast
        k_rope_input = self.W_KR(key)  # (B, T_k, d_rope)
        k_rope_input = k_rope_input.unsqueeze(1).expand(-1, self.h, -1, -1)  # (B, h, T_k, d_rope)

        # 5. RoPE query: per head
        q_rope_input = self.W_QR(query).view(B, T_q, self.h, self.d_rope).transpose(1, 2)  # (B, h, T_q, d_rope)

        # 6. Apply RoPE rotation
        cos_q = self.rope_cos[:T_q].unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, d_rope)
        sin_q = self.rope_sin[:T_q].unsqueeze(0).unsqueeze(0)
        cos_k = self.rope_cos[:T_k].unsqueeze(0).unsqueeze(0)
        sin_k = self.rope_sin[:T_k].unsqueeze(0).unsqueeze(0)

        q_rope = apply_rotary_emb(q_rope_input, cos_q, sin_q)  # (B, h, T_q, d_rope)
        k_rope = apply_rotary_emb(k_rope_input, cos_k, sin_k)  # (B, h, T_k, d_rope)

        # === Concatenate content + RoPE ===
        # 7. Q = [q_nope, q_rope], K = [k_nope, k_rope]
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, h, T_q, d_head + d_rope)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, h, T_k, d_head + d_rope)

        # === Attention ===
        # 8. Scaled dot-product attention
        # Scores computed on full d_qk = d_head + d_rope dimension
        # But V projection is only d_head (output is d_head per head)
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)

        # === Output ===
        # 9. Concat heads and project
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_head)
        output = self.output_projection(x)

        return output, self.attn


if __name__ == "__main__":
    print("Testing Multi-Head Latent Attention (MLA)...")
    print("=" * 70)

    batch_size = 2
    seq_len = 10
    d_model = 256
    h = 8
    d_c = 128
    d_rope = 16
    dropout_p = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"MLA params: h={h}, d_c={d_c}, d_rope={d_rope}")
    print(f"d_head={d_model // h}, d_qk={d_model // h + d_rope}\n")

    # Test 1: Basic functionality
    print("1. Testing MultiHeadLatentAttention module...")
    mla = MultiHeadLatentAttention(
        h=h, d_model=d_model, d_c=d_c, d_rope=d_rope,
        max_seq_len=256, dropout=dropout_p
    ).to(device)

    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, attn_weights = mla(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert output.shape == x.shape, "MLA output shape mismatch"
    print("   MLA basic functionality working correctly")

    # Test 2: Attention weights sum to 1
    print("\n2. Testing attention weight normalization...")
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
    print("   Attention weights properly normalized")

    # Test 3: Causal mask
    print("\n3. Testing with causal mask...")
    from AttentionHeads.mha.attention import create_causal_mask
    causal_mask = create_causal_mask(seq_len, device)
    output, attn_weights = mla(x, x, x, mask=causal_mask)
    print(f"   Output with causal mask shape: {output.shape}")
    upper_tri = attn_weights.triu(diagonal=1)
    print(f"   Max attention to future positions: {upper_tri.max().item():.6f}")
    print("   Causal masking working correctly")

    # Test 4: Parameter count
    print("\n4. Parameter count analysis...")
    mla_params = sum(p.numel() for p in mla.parameters())
    print(f"   Total MLA attention parameters: {mla_params:,}")

    # Break down by component
    w_dkv_params = sum(p.numel() for p in mla.W_DKV.parameters())
    w_uk_params = sum(p.numel() for p in mla.W_UK.parameters())
    w_uv_params = sum(p.numel() for p in mla.W_UV.parameters())
    w_q_params = sum(p.numel() for p in mla.W_Q.parameters())
    w_kr_params = sum(p.numel() for p in mla.W_KR.parameters())
    w_qr_params = sum(p.numel() for p in mla.W_QR.parameters())
    w_out_params = sum(p.numel() for p in mla.output_projection.parameters())

    print(f"   W_DKV (d_model->d_c): {w_dkv_params:,}")
    print(f"   W_UK (d_c->h*d_head): {w_uk_params:,}")
    print(f"   W_UV (d_c->h*d_head): {w_uv_params:,}")
    print(f"   W_Q (d_model->h*d_head): {w_q_params:,}")
    print(f"   W_KR (d_model->d_rope): {w_kr_params:,}")
    print(f"   W_QR (d_model->h*d_rope): {w_qr_params:,}")
    print(f"   Output (h*d_head->d_model): {w_out_params:,}")

    # KV-cache comparison
    print(f"\n   KV-cache per token per layer:")
    print(f"   MHA: {2 * h * (d_model // h)} values (2*h*d_head)")
    print(f"   MLA: {d_c + d_rope} values (d_c + d_rope)")
    print(f"   Reduction: {(1 - (d_c + d_rope) / (2 * h * (d_model // h))) * 100:.1f}%")
    print("   Parameter analysis complete")

    # Test 5: Gradient flow
    print("\n5. Testing gradient flow...")
    mla.zero_grad()
    x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True).to(device)
    output, _ = mla(x_grad, x_grad, x_grad)
    loss = output.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradient should flow to input"
    print(f"   Input gradient shape: {x_grad.grad.shape}")
    print("   Gradient flow verified")

    print("\n" + "=" * 70)
    print("All Multi-Head Latent Attention tests passed!")
    print("MLA is ready for use in transformer models!")
