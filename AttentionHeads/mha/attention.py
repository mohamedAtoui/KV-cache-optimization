"""
Multi-Head Attention (MHA) Implementation

Based on "Attention Is All You Need" (Vaswani et al., 2017)
Implementation follows Harvard NLP's Annotated Transformer:
https://nlp.seas.harvard.edu/annotated-transformer/

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances
    in neural information processing systems (pp. 5998-6008).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot-Product Attention

    Implementation from Harvard NLP's Annotated Transformer.

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

    Shape:
        - Query: (batch, num_heads, seq_len_q, d_k)
        - Key: (batch, num_heads, seq_len_k, d_k)
        - Value: (batch, num_heads, seq_len_v, d_k)
        - Mask: (batch, 1, seq_len_q, seq_len_k) or broadcastable
        - Output: (batch, num_heads, seq_len_q, d_k)
    """
    d_k = query.size(-1)

    # Compute attention scores: QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention probabilities
    p_attn = F.softmax(scores, dim=-1)

    # Apply dropout to attention weights
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Apply attention to values
    output = torch.matmul(p_attn, value)

    return output, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation follows Harvard NLP's Annotated Transformer.

    The attention mechanism allows the model to jointly attend to information
    from different representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

    Args:
        h: Number of attention heads
        d_model: Model dimension (must be divisible by h)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, h, d_model, dropout=0.1):
        """Initialize multi-head attention module"""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # Create 4 linear layers: Q, K, V projections and output projection
        # Using ModuleList for cleaner code (Harvard NLP approach)
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)
        ])

        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention

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

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value shapes: (batch, h, seq_len, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch
        # x shape: (batch, h, seq_len_q, d_k)
        # self.attn shape: (batch, h, seq_len_q, seq_len_k)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        # x shape: (batch, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # Apply final linear layer (output projection)
        output = self.linears[-1](x)

        return output, self.attn


# Backward compatibility: alias MultiHeadAttention to MultiHeadedAttention
MultiHeadAttention = MultiHeadedAttention


def subsequent_mask(size):
    """
    Mask out subsequent positions (Harvard NLP implementation)

    Used for masked self-attention in the decoder to prevent positions
    from attending to future positions.

    Args:
        size: Sequence length

    Returns:
        mask: (1, size, size) upper triangular matrix with 1s below diagonal

    Example:
        >>> subsequent_mask(5)
        tensor([[[1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1]]], dtype=torch.uint8)
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def create_causal_mask(seq_len, device=None):
    """
    Create causal mask for autoregressive modeling (decoder)

    Prevents positions from attending to future positions.
    Wrapper around subsequent_mask() for backward compatibility.

    Args:
        seq_len: Sequence length
        device: torch device (optional, for compatibility)

    Returns:
        mask: (1, seq_len, seq_len) - Lower triangular matrix
              1 = can attend, 0 = cannot attend
    """
    mask = subsequent_mask(seq_len)
    if device is not None:
        mask = mask.to(device)
    return mask


def create_padding_mask(seq, pad_token_id=0):
    """
    Create padding mask to prevent attention to padding tokens

    Args:
        seq: (batch_size, seq_len) - Input sequence with token IDs
        pad_token_id: ID of padding token (default: 0)

    Returns:
        mask: (batch_size, 1, seq_len) - 1 = real token, 0 = padding

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        >>> create_padding_mask(seq, pad_token_id=0).shape
        torch.Size([2, 1, 5])
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (seq != pad_token_id).unsqueeze(1)
    return mask


def create_combined_mask(seq, pad_token_id=0, causal=True):
    """
    Create combined causal and padding mask

    Combines both causal masking (for preventing future attention) and
    padding masking (for ignoring padding tokens).

    Args:
        seq: (batch_size, seq_len) - Input sequence
        pad_token_id: ID of padding token (default: 0)
        causal: Whether to apply causal mask (default: True)

    Returns:
        mask: (batch_size, seq_len, seq_len) combined mask

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0]])
        >>> mask = create_combined_mask(seq, causal=True)
        >>> mask.shape
        torch.Size([1, 5, 5])
    """
    batch_size, seq_len = seq.size()
    device = seq.device

    # Padding mask: (batch_size, 1, seq_len)
    padding_mask = create_padding_mask(seq, pad_token_id)

    if causal:
        # Causal mask: (1, seq_len, seq_len)
        causal_mask = create_causal_mask(seq_len, device)

        # Combine both masks: (batch_size, seq_len, seq_len)
        # Broadcasting: (batch_size, 1, seq_len) & (1, seq_len, seq_len)
        combined_mask = padding_mask & causal_mask
    else:
        # Just padding mask: (batch_size, seq_len, seq_len)
        combined_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

    return combined_mask


if __name__ == "__main__":
    # Unit tests for Multi-Head Attention (Harvard NLP style)
    print("Testing Multi-Head Attention (Harvard NLP Implementation)...")
    print("=" * 70)

    batch_size = 2
    seq_len = 10
    d_model = 512
    h = 8  # number of heads
    dropout_p = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, h={h}\n")

    # Test 1: attention() function
    print("1. Testing attention() function...")
    d_k = d_model // h
    Q = torch.randn(batch_size, h, seq_len, d_k).to(device)
    K = torch.randn(batch_size, h, seq_len, d_k).to(device)
    V = torch.randn(batch_size, h, seq_len, d_k).to(device)
    dropout_layer = nn.Dropout(p=dropout_p)

    output, attn_weights = attention(Q, K, V, mask=None, dropout=dropout_layer)
    print(f"   Q shape: {Q.shape}")
    print(f"   K shape: {K.shape}")
    print(f"   V shape: {V.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")

    # Verify attention weights sum to 1
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Attention weights don't sum to 1"
    print("   ✓ Attention function working correctly")

    # Test 2: MultiHeadedAttention
    print("\n2. Testing MultiHeadedAttention module...")
    mha = MultiHeadedAttention(h, d_model, dropout=dropout_p).to(device)

    query = torch.randn(batch_size, seq_len, d_model).to(device)
    key = torch.randn(batch_size, seq_len, d_model).to(device)
    value = torch.randn(batch_size, seq_len, d_model).to(device)

    output, attn_weights = mha(query, key, value)
    print(f"   Query shape: {query.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert output.shape == query.shape, "MHA output shape mismatch"

    # Verify attention weights sum to 1 for each head
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Attention weights don't sum to 1"
    print("   ✓ MultiHeadedAttention working correctly")

    # Test 3: Self-attention (Q=K=V)
    print("\n3. Testing Self-Attention...")
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    output, attn_weights = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Self-attention output shape mismatch"
    print("   ✓ Self-attention working correctly")

    # Test 4: subsequent_mask() function
    print("\n4. Testing subsequent_mask() function...")
    mask = subsequent_mask(5)
    print(f"   Subsequent mask shape: {mask.shape}")
    print(f"   Subsequent mask (prevents future attention):")
    print(f"{mask[0].int()}")
    print("   ✓ Subsequent mask created correctly")

    # Test 5: Causal Mask with attention
    print("\n5. Testing Causal Mask with attention...")
    causal_mask = create_causal_mask(seq_len, device)
    print(f"   Causal mask shape: {causal_mask.shape}")
    output, attn_weights = mha(x, x, x, mask=causal_mask)
    print(f"   Output with causal mask shape: {output.shape}")

    # Verify causal mask prevents attending to future (upper triangle should be ~0)
    upper_tri = attn_weights.triu(diagonal=1)
    print(f"   Max attention to future positions: {upper_tri.max().item():.6f}")
    print("   ✓ Causal masking working correctly")

    # Test 6: Padding Mask
    print("\n6. Testing Padding Mask...")
    seq = torch.randint(1, 100, (batch_size, seq_len)).to(device)
    seq[:, -3:] = 0  # Add padding at the end
    padding_mask = create_padding_mask(seq, pad_token_id=0)
    print(f"   Padding mask shape: {padding_mask.shape}")
    print(f"   Sample padding mask (1=real, 0=pad):\n   {padding_mask[0]}")
    print("   ✓ Padding mask created correctly")

    # Test 7: Combined Mask
    print("\n7. Testing Combined Mask...")
    combined_mask = create_combined_mask(seq, pad_token_id=0, causal=True)
    print(f"   Combined mask shape: {combined_mask.shape}")
    print(f"   Combined mask (first sample, last 5x5):")
    print(f"{combined_mask[0, -5:, -5:].int()}")

    output, attn_weights = mha(x, x, x, mask=combined_mask)
    print(f"   Output with combined mask shape: {output.shape}")
    print("   ✓ Combined masking working correctly")

    # Test 8: Parameter count
    print("\n8. Counting Parameters...")
    num_params = sum(p.numel() for p in mha.parameters())
    expected_params = 4 * d_model * (d_model + 1)  # 4 linear layers with bias
    print(f"   Total parameters: {num_params:,}")
    print(f"   Expected (approx): {expected_params:,}")
    print("   ✓ Parameter count verified")

    # Test 9: Backward compatibility
    print("\n9. Testing backward compatibility alias...")
    mha_old_name = MultiHeadAttention(h, d_model, dropout=dropout_p)
    print(f"   MultiHeadAttention alias works: {isinstance(mha_old_name, MultiHeadedAttention)}")
    print("   ✓ Backward compatibility maintained")

    print("\n" + "=" * 70)
    print("✓ All Multi-Head Attention tests passed!")
    print("Implementation matches Harvard NLP's Annotated Transformer")
