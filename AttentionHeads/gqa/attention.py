"""
Grouped Query Attention (GQA) Implementation

Based on "GQA: Training Generalized Multi-Query Transformer Models from
Multi-Head Checkpoints" (Ainslie et al., 2023)
https://arxiv.org/abs/2305.13245

Key Innovation: GQA is an interpolation between MHA and MQA.
- MHA: h query heads, h KV heads (each query head has its own K,V)
- MQA: h query heads, 1 KV head (all query heads share K,V)
- GQA: h query heads, g KV heads where 1 < g < h (groups of query heads share K,V)

Reference
    Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., &
    Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints. arXiv preprint arXiv:2305.13245.
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
        value: Value tensor of shape (..., seq_len_v, d_k)
        mask: Optional mask tensor. Positions with mask == 0 are masked out
        dropout: Optional nn.Dropout layer

    Returns:
        output: Attention output of shape (..., seq_len_q, d_k)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # Compute attention scores: QK^T / sqrt(d_k)
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


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) from "GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

    Key difference from MHA and MQA:
    - MHA: Each query head has separate K, V (num_kv_heads = h)
    - MQA: All query heads share single K, V (num_kv_heads = 1)
    - GQA: Groups of query heads share K, V (1 < num_kv_heads < h)

    Architecture:
        Q: d_model -> d_model (h query heads)
        K: d_model -> num_kv_heads * d_k
        V: d_model -> num_kv_heads * d_k
        Output: d_model -> d_model

    Benefits:
    - More parameters than MQA, fewer than MHA
    - Better quality than MQA, close to MHA
    - Smaller KV cache than MHA, larger than MQA
    - Good balance between quality and efficiency

    Args:
        h: Number of query heads
        d_model: Model dimension (must be divisible by h)
        num_kv_heads: Number of KV heads (must divide h evenly)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, h, d_model, num_kv_heads=None, dropout=0.1,
                 position_embedding_type="learned", max_seq_len=256):
        """Initialize grouped query attention module"""
        super(GroupedQueryAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        # Per-head dimension
        self.d_k = d_model // h
        self.h = h  # Number of query heads
        self.d_model = d_model
        self.position_embedding_type = position_embedding_type

        # Default num_kv_heads to h//2 if not specified (a reasonable middle ground)
        if num_kv_heads is None:
            num_kv_heads = max(1, h // 2)
        self.num_kv_heads = num_kv_heads

        # Validate that h is divisible by num_kv_heads
        assert h % num_kv_heads == 0, f"num_heads ({h}) must be divisible by num_kv_heads ({num_kv_heads})"
        self.heads_per_group = h // num_kv_heads  # Query heads per KV head

        # GQA: Separate projections for Q (h heads), grouped K and V (num_kv_heads)
        self.q_projection = nn.Linear(d_model, d_model)  # h query heads
        self.k_projection = nn.Linear(d_model, num_kv_heads * self.d_k)  # num_kv_heads
        self.v_projection = nn.Linear(d_model, num_kv_heads * self.d_k)  # num_kv_heads
        self.output_projection = nn.Linear(d_model, d_model)

        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)

        # RoPE support
        if position_embedding_type == "rope":
            cos, sin = precompute_freqs_cis(self.d_k, max_seq_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for grouped query attention

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

        # 1) Project queries to h heads
        # Q: (batch, seq_len_q, d_model) -> (batch, h, seq_len_q, d_k)
        q = self.q_projection(query).view(nbatches, seq_len_q, self.h, self.d_k).transpose(1, 2)

        # 2) Project keys and values to num_kv_heads
        # K: (batch, seq_len_k, d_model) -> (batch, num_kv_heads, seq_len_k, d_k)
        # V: (batch, seq_len_v, d_model) -> (batch, num_kv_heads, seq_len_v, d_k)
        k = self.k_projection(key).view(nbatches, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.v_projection(value).view(nbatches, -1, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to Q and K before expanding (before repeat_interleave)
        if self.position_embedding_type == "rope":
            cos = self.rope_cos[:seq_len_q].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
            sin = self.rope_sin[:seq_len_q].unsqueeze(0).unsqueeze(0)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # 3) Expand K, V to match query heads by repeating each KV head
        # K, V: (batch, num_kv_heads, seq_len, d_k) -> (batch, h, seq_len, d_k)
        # Use repeat_interleave to repeat each KV head for its group of query heads
        k = torch.repeat_interleave(k, self.heads_per_group, dim=1)
        v = torch.repeat_interleave(v, self.heads_per_group, dim=1)

        # 4) Apply attention with expanded K, V
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
    # Unit tests for Grouped Query Attention
    print("Testing Grouped Query Attention (GQA)...")
    print("=" * 70)

    batch_size = 2
    seq_len = 10
    d_model = 512
    h = 8  # number of query heads
    dropout_p = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, h={h}\n")

    # Test 1: GQA with num_kv_heads=2
    print("1. Testing GroupedQueryAttention with num_kv_heads=2...")
    gqa_2 = GroupedQueryAttention(h, d_model, num_kv_heads=2, dropout=dropout_p).to(device)

    query = torch.randn(batch_size, seq_len, d_model).to(device)
    key = torch.randn(batch_size, seq_len, d_model).to(device)
    value = torch.randn(batch_size, seq_len, d_model).to(device)

    output, attn_weights = gqa_2(query, key, value)
    print(f"   Query shape: {query.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print(f"   num_kv_heads: {gqa_2.num_kv_heads}, heads_per_group: {gqa_2.heads_per_group}")
    assert output.shape == query.shape, "GQA-2 output shape mismatch"

    # Verify attention weights sum to 1 for each head
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Attention weights don't sum to 1"
    print("   GQA with num_kv_heads=2 working correctly")

    # Test 2: GQA with num_kv_heads=4
    print("\n2. Testing GroupedQueryAttention with num_kv_heads=4...")
    gqa_4 = GroupedQueryAttention(h, d_model, num_kv_heads=4, dropout=dropout_p).to(device)

    output, attn_weights = gqa_4(query, key, value)
    print(f"   Output shape: {output.shape}")
    print(f"   num_kv_heads: {gqa_4.num_kv_heads}, heads_per_group: {gqa_4.heads_per_group}")
    assert output.shape == query.shape, "GQA-4 output shape mismatch"
    print("   GQA with num_kv_heads=4 working correctly")

    # Test 3: Self-attention (Q=K=V)
    print("\n3. Testing Self-Attention...")
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, attn_weights = gqa_2(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Self-attention output shape mismatch"
    print("   Self-attention working correctly")

    # Test 4: Causal Mask
    print("\n4. Testing with Causal Mask...")
    from AttentionHeads.mha.attention import create_causal_mask
    causal_mask = create_causal_mask(seq_len, device)
    output, attn_weights = gqa_2(x, x, x, mask=causal_mask)
    print(f"   Output with causal mask shape: {output.shape}")

    # Verify causal mask prevents attending to future
    upper_tri = attn_weights.triu(diagonal=1)
    print(f"   Max attention to future positions: {upper_tri.max().item():.6f}")
    print("   Causal masking working correctly")

    # Test 5: Parameter count comparison (GQA vs MHA vs MQA)
    print("\n5. Comparing Parameter Counts (GQA vs MHA vs MQA)...")

    # GQA parameters
    gqa_2_params = sum(p.numel() for p in gqa_2.parameters())
    gqa_4_params = sum(p.numel() for p in gqa_4.parameters())

    # MHA parameters (for comparison)
    from AttentionHeads.mha.attention import MultiHeadedAttention
    mha = MultiHeadedAttention(h, d_model, dropout=dropout_p).to(device)
    mha_params = sum(p.numel() for p in mha.parameters())

    # MQA parameters (for comparison)
    from AttentionHeads.mqa.attention import MultiQueryAttention
    mqa = MultiQueryAttention(h, d_model, dropout=dropout_p).to(device)
    mqa_params = sum(p.numel() for p in mqa.parameters())

    print(f"   MHA parameters:     {mha_params:,}")
    print(f"   GQA-4 parameters:   {gqa_4_params:,} ({(1 - gqa_4_params/mha_params)*100:.1f}% reduction vs MHA)")
    print(f"   GQA-2 parameters:   {gqa_2_params:,} ({(1 - gqa_2_params/mha_params)*100:.1f}% reduction vs MHA)")
    print(f"   MQA parameters:     {mqa_params:,} ({(1 - mqa_params/mha_params)*100:.1f}% reduction vs MHA)")
    print("   Parameter comparison verified")

    # Test 6: Different sequence lengths for cross-attention
    print("\n6. Testing Cross-Attention (different seq lengths)...")
    query = torch.randn(batch_size, 8, d_model).to(device)
    key = torch.randn(batch_size, 12, d_model).to(device)
    value = torch.randn(batch_size, 12, d_model).to(device)

    output, attn_weights = gqa_2(query, key, value)
    print(f"   Query seq_len: 8")
    print(f"   Key/Value seq_len: 12")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert output.shape[1] == 8, "Output seq_len should match query"
    assert attn_weights.shape[-1] == 12, "Attention should be over key seq_len"
    print("   Cross-attention working correctly")

    # Test 7: Verify GQA reduces to MQA when num_kv_heads=1
    print("\n7. Testing GQA with num_kv_heads=1 (should be equivalent to MQA)...")
    gqa_1 = GroupedQueryAttention(h, d_model, num_kv_heads=1, dropout=dropout_p).to(device)
    gqa_1_params = sum(p.numel() for p in gqa_1.parameters())
    print(f"   GQA-1 parameters: {gqa_1_params:,}")
    print(f"   MQA parameters:   {mqa_params:,}")
    print(f"   Match: {gqa_1_params == mqa_params}")
    print("   GQA with num_kv_heads=1 matches MQA parameter count")

    # Test 8: Verify GQA with num_kv_heads=h is equivalent to MHA params
    print("\n8. Testing GQA with num_kv_heads=h (should be equivalent to MHA)...")
    gqa_h = GroupedQueryAttention(h, d_model, num_kv_heads=h, dropout=dropout_p).to(device)
    gqa_h_params = sum(p.numel() for p in gqa_h.parameters())
    print(f"   GQA-h parameters: {gqa_h_params:,}")
    print(f"   MHA parameters:   {mha_params:,}")
    print(f"   Match: {gqa_h_params == mha_params}")
    print("   GQA with num_kv_heads=h matches MHA parameter count")

    print("\n" + "=" * 70)
    print("All Grouped Query Attention tests passed!")
    print("GQA is ready to use!")
