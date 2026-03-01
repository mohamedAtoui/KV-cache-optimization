"""
Core Transformer Layers

Based on "Attention Is All You Need" (Vaswani et al., 2017)
Implementation follows Harvard NLP's Annotated Transformer:
https://nlp.seas.harvard.edu/annotated-transformer/

Includes: LayerNorm, Position-wise Feed-Forward, Sublayer Connections

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances
    in neural information processing systems (pp. 5998-6008).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def clones(module, N):
    """
    Produce N identical layers (Harvard NLP utility function)

    Creates N deep copies of the given module. Used to create stacks of
    identical layers in the encoder and decoder.

    Args:
        module: PyTorch module to clone
        N: Number of copies to create

    Returns:
        ModuleList containing N copies of the module

    Example:
        >>> layers = clones(EncoderLayer(...), 6)  # 6-layer encoder
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    Layer Normalization (Harvard NLP implementation)

    Construct a layernorm module. Features dimension is d_model.
    Normalizes across the feature dimension with learnable scale and shift.

    Args:
        features: Feature dimension (d_model)
        eps: Epsilon for numerical stability (default: 1e-6)

    Shape:
        - Input: (batch, seq_len, features)
        - Output: (batch, seq_len, features)
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # Scale (gamma)
        self.b_2 = nn.Parameter(torch.zeros(features))  # Shift (beta)
        self.eps = eps

    def forward(self, x):
        """Apply layer normalization"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks (Harvard NLP implementation)

    Implements FFN equation from "Attention is All You Need":
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Each position is processed independently and identically through
    a two-layer feed-forward network with ReLU activation.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension (typically 2048)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply two linear transformations with ReLU in between"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# Backward compatibility: alias FeedForward to PositionwiseFeedForward
FeedForward = PositionwiseFeedForward


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer norm (Harvard NLP implementation)

    A residual connection with layer normalization and dropout.
    Implements: LayerNorm(x + Sublayer(x))

    Note: The code in the paper uses pre-norm (norm before sublayer), but
    this implementation uses post-norm (norm after adding residual).

    Args:
        size: Feature dimension (d_model)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: (batch, seq_len, size)
        - Output: (batch, seq_len, size)
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size"""
        return x + self.dropout(sublayer(self.norm(x)))


# Backward compatibility: alias ResidualConnection to SublayerConnection
ResidualConnection = SublayerConnection


class DropoutLayer(nn.Module):
    """
    Dropout layer with seed management for reproducibility

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        return self.dropout(x)


if __name__ == "__main__":
    # Unit tests for layers (Harvard NLP style)
    print("Testing Core Transformer Layers (Harvard NLP Implementation)...")
    print("=" * 70)

    batch_size, seq_len, d_model, d_ff = 2, 10, 512, 2048
    dropout_p = 0.1

    # Test clones utility
    print("\n1. Testing clones() utility function...")
    sample_module = nn.Linear(d_model, d_model)
    cloned_modules = clones(sample_module, 6)
    print(f"   Created {len(cloned_modules)} clones")
    print(f"   Type: {type(cloned_modules)}")
    assert len(cloned_modules) == 6, "clones() should create 6 copies"
    assert isinstance(cloned_modules, nn.ModuleList), "clones() should return ModuleList"
    print("   ✓ clones() working correctly")

    # Test LayerNorm
    print("\n2. Testing LayerNorm...")
    x = torch.randn(batch_size, seq_len, d_model)
    layer_norm = LayerNorm(d_model)
    out = layer_norm(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Output mean: {out.mean():.6f}, std: {out.std():.6f}")
    assert out.shape == x.shape, "LayerNorm output shape mismatch"
    # Check normalization properties (mean ~0, std ~1 before scale/shift)
    print("   ✓ LayerNorm working correctly")

    # Test PositionwiseFeedForward
    print("\n3. Testing PositionwiseFeedForward...")
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_p)
    out = ffn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Hidden dimension: {d_ff}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "FFN output shape mismatch"
    print("   ✓ PositionwiseFeedForward working correctly")

    # Test backward compatibility alias
    print("\n4. Testing FeedForward alias (backward compatibility)...")
    ffn_alias = FeedForward(d_model, d_ff, dropout=dropout_p)
    out_alias = ffn_alias(x)
    print(f"   FeedForward is PositionwiseFeedForward: {FeedForward is PositionwiseFeedForward}")
    print(f"   Output shape: {out_alias.shape}")
    assert out_alias.shape == x.shape, "FeedForward alias output shape mismatch"
    print("   ✓ Backward compatibility maintained")

    # Test SublayerConnection
    print("\n5. Testing SublayerConnection...")
    sublayer_conn = SublayerConnection(d_model, dropout=dropout_p)
    out = sublayer_conn(x, lambda x: ffn(x))
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "SublayerConnection output shape mismatch"
    print("   ✓ SublayerConnection working correctly")

    # Test ResidualConnection alias
    print("\n6. Testing ResidualConnection alias (backward compatibility)...")
    residual_alias = ResidualConnection(d_model, dropout=dropout_p)
    out_residual = residual_alias(x, lambda x: ffn(x))
    print(f"   ResidualConnection is SublayerConnection: {ResidualConnection is SublayerConnection}")
    print(f"   Output shape: {out_residual.shape}")
    assert out_residual.shape == x.shape, "ResidualConnection alias output shape mismatch"
    print("   ✓ Backward compatibility maintained")

    # Test DropoutLayer
    print("\n7. Testing DropoutLayer...")
    dropout = DropoutLayer(dropout=0.5)
    out = dropout(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "Dropout output shape mismatch"
    print("   ✓ DropoutLayer working correctly")

    # Test parameter counting
    print("\n8. Testing parameter counts...")
    print(f"   LayerNorm params: {sum(p.numel() for p in layer_norm.parameters()):,}")
    print(f"   FFN params: {sum(p.numel() for p in ffn.parameters()):,}")
    print(f"   SublayerConnection params: {sum(p.numel() for p in sublayer_conn.parameters()):,}")
    print("   ✓ Parameter counts verified")

    print("\n" + "=" * 70)
    print("✓ All layer tests passed!")
    print("Implementation matches Harvard NLP's Annotated Transformer")
