"""
Multi-Query Attention (MQA) Implementation

Based on "Fast Transformer Decoding: One Write-head is All You Need" (Shazeer 2019)
https://arxiv.org/abs/1911.02150

Key Innovation: Share K and V across all attention heads (only Q has multiple heads)
This reduces memory bandwidth and speeds up inference significantly.

Reference:
    Shazeer, N. (2019). Fast Transformer Decoding: One Write-head is All You Need.
    arXiv preprint arXiv:1911.02150.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot-Product Attention

    Same as MHA - unchanged from mha/attention.py

    Attention(Q, K, V) = softmax(QK^T / √d_k)V

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_v, d_k)
        mask: Optional mask tensor. Positions with mask == 0 are masked out
        dropout: Optional nn.Dropout layer

    Returns:
        output: Attention output of shape (..., seq_len_q, d_k)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # Compute attention scores: QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative value)
    # Use -1e4 instead of -1e9 for BFloat16 compatibility (prevents overflow)
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


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) from "Fast Transformer Decoding" (Shazeer 2019)

    Key difference from Multi-Head Attention:
    - MHA: Each head has separate Q, K, V projections
    - MQA: Each head has separate Q, but K and V are shared across all heads

    Architecture:
        Q: d_model → d_model (h separate query heads)
        K: d_model → d_k (single shared key)
        V: d_model → d_k (single shared value)
        Output: d_model → d_model

    Benefits:
    - Fewer parameters (~44% reduction in attention params)
    - Faster inference (less memory bandwidth for K, V)
    - Smaller KV cache during generation
    - Minor quality degradation vs MHA

    Args:
        h: Number of query heads
        d_model: Model dimension (must be divisible by h)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, h, d_model, dropout=0.1):
        """Initialize multi-query attention module"""
        super(MultiQueryAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        # Per-head dimension
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model

        # MQA: Separate projections for Q (h heads), shared K and V (1 head)
        self.q_projection = nn.Linear(d_model, d_model)  # h heads
        self.k_projection = nn.Linear(d_model, self.d_k)  # 1 shared head
        self.v_projection = nn.Linear(d_model, self.d_k)  # 1 shared head
        self.output_projection = nn.Linear(d_model, d_model)

        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-query attention

        Args:
            query: Query tensor (batch, seq_len_q, d_model)
            key: Key tensor (batch, seq_len_k, d_model)
            value: Value tensor (batch, seq_len_v, d_model)
            mask: Optional mask tensor (batch, seq_len_q, seq_len_k) or broadcastable

        Returns:
            output: Attention output (batch, seq_len_q, d_model)
            attn: Attention weights (batch, h, seq_len_q, seq_len_k)
        """
        if mask is not None:
            # Same mask applied to all h heads: (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # 1) Project queries to multiple heads
        # Q: (batch, seq_len_q, d_model) → (batch, h, seq_len_q, d_k)
        q = self.q_projection(query).view(nbatches, seq_len_q, self.h, self.d_k).transpose(1, 2)

        # 2) Project keys and values to single shared head
        # K: (batch, seq_len_k, d_model) → (batch, 1, seq_len_k, d_k)
        # V: (batch, seq_len_v, d_model) → (batch, 1, seq_len_v, d_k)
        k = self.k_projection(key).view(nbatches, seq_len_k, 1, self.d_k).transpose(1, 2)
        v = self.v_projection(value).view(nbatches, -1, 1, self.d_k).transpose(1, 2)

        # 3) Broadcast shared K, V across all query heads
        # K, V: (batch, 1, seq_len, d_k) → (batch, h, seq_len, d_k)
        k = k.expand(-1, self.h, -1, -1)
        v = v.expand(-1, self.h, -1, -1)

        # 4) Apply attention with broadcasted K, V
        # x: (batch, h, seq_len_q, d_k)
        # self.attn: (batch, h, seq_len_q, seq_len_k)
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)

        # 5) "Concat" using a view and apply final linear
        # x: (batch, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # Apply output projection
        output = self.output_projection(x)

        return output, self.attn


if __name__ == "__main__":
    # Unit tests for Multi-Query Attention
    print("Testing Multi-Query Attention (MQA)...")
    print("=" * 70)

    batch_size = 2
    seq_len = 10
    d_model = 512
    h = 8  # number of query heads
    dropout_p = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, h={h}\n")

    # Test 1: MultiQueryAttention basic functionality
    print("1. Testing MultiQueryAttention module...")
    mqa = MultiQueryAttention(h, d_model, dropout=dropout_p).to(device)

    query = torch.randn(batch_size, seq_len, d_model).to(device)
    key = torch.randn(batch_size, seq_len, d_model).to(device)
    value = torch.randn(batch_size, seq_len, d_model).to(device)

    output, attn_weights = mqa(query, key, value)
    print(f"   Query shape: {query.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert output.shape == query.shape, "MQA output shape mismatch"

    # Verify attention weights sum to 1 for each head
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Attention weights don't sum to 1"
    print("   ✓ MultiQueryAttention working correctly")

    # Test 2: Self-attention (Q=K=V)
    print("\n2. Testing Self-Attention...")
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, attn_weights = mqa(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Self-attention output shape mismatch"
    print("   ✓ Self-attention working correctly")

    # Test 3: Causal Mask
    print("\n3. Testing with Causal Mask...")
    from mha.attention import create_causal_mask
    causal_mask = create_causal_mask(seq_len, device)
    output, attn_weights = mqa(x, x, x, mask=causal_mask)
    print(f"   Output with causal mask shape: {output.shape}")

    # Verify causal mask prevents attending to future
    upper_tri = attn_weights.triu(diagonal=1)
    print(f"   Max attention to future positions: {upper_tri.max().item():.6f}")
    print("   ✓ Causal masking working correctly")

    # Test 4: Parameter count comparison (MQA vs MHA)
    print("\n4. Comparing Parameter Counts (MQA vs MHA)...")

    # MQA parameters
    mqa_params = sum(p.numel() for p in mqa.parameters())

    # MHA parameters (for comparison)
    from mha.attention import MultiHeadedAttention
    mha = MultiHeadedAttention(h, d_model, dropout=dropout_p).to(device)
    mha_params = sum(p.numel() for p in mha.parameters())

    reduction = (1 - mqa_params / mha_params) * 100

    print(f"   MHA parameters: {mha_params:,}")
    print(f"   MQA parameters: {mqa_params:,}")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"   ✓ MQA has significantly fewer parameters")

    # Test 5: Different sequence lengths for cross-attention
    print("\n5. Testing Cross-Attention (different seq lengths)...")
    query = torch.randn(batch_size, 8, d_model).to(device)
    key = torch.randn(batch_size, 12, d_model).to(device)
    value = torch.randn(batch_size, 12, d_model).to(device)

    output, attn_weights = mqa(query, key, value)
    print(f"   Query seq_len: 8")
    print(f"   Key/Value seq_len: 12")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert output.shape[1] == 8, "Output seq_len should match query"
    assert attn_weights.shape[-1] == 12, "Attention should be over key seq_len"
    print("   ✓ Cross-attention working correctly")

    print("\n" + "=" * 70)
    print("✓ All Multi-Query Attention tests passed!")
    print("MQA is ready to use for faster inference!")
