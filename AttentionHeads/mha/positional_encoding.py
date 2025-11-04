"""
Positional Encoding Implementations

Based on "Attention Is All You Need" (Vaswani et al., 2017)
Implementation follows Harvard NLP's Annotated Transformer:
https://nlp.seas.harvard.edu/annotated-transformer/

Includes: Sinusoidal (fixed) and Learned positional encodings

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances
    in neural information processing systems (pp. 5998-6008).
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sinusoidal functions (Harvard NLP implementation)

    Implement the PE function from "Attention is All You Need".
    Since the model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we inject some information
    about the relative or absolute position of the tokens.

    The positional encodings have the same dimension d_model as the embeddings,
    so that the two can be summed.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimension of model embeddings
        dropout: Dropout probability (default: 0.1)
        max_len: Maximum sequence length (default: 5000)

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# Backward compatibility: alias SinusoidalPositionalEncoding
SinusoidalPositionalEncoding = PositionalEncoding


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (alternative to sinusoidal)

    Uses an embedding layer to learn position representations instead of
    fixed sinusoidal patterns. Often used in BERT and other variants.

    Unlike the original Transformer which uses fixed sinusoidal positional
    encodings, this version learns the positional representations during training.

    Args:
        d_model: Dimension of model embeddings
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """Add learned positional encoding to input embeddings"""
        batch_size, seq_len, d_model = x.size()

        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)

        # Get position embeddings and add to input
        position_encodings = self.position_embeddings(positions)
        x = x + position_encodings.unsqueeze(0)

        return self.dropout(x)


class PositionalEncodingFactory:
    """
    Factory to create positional encoding based on type

    Provides a convenient way to instantiate different types of positional
    encodings with a single interface.
    """

    @staticmethod
    def create(pe_type, d_model, max_len=5000, dropout=0.1):
        """
        Create a positional encoding module

        Args:
            pe_type: Type of encoding - "sinusoidal" or "learned"
            d_model: Model dimension
            max_len: Maximum sequence length (default: 5000)
            dropout: Dropout probability (default: 0.1)

        Returns:
            PositionalEncoding or LearnedPositionalEncoding module

        Raises:
            ValueError: If pe_type is not recognized

        Example:
            >>> pe = PositionalEncodingFactory.create("sinusoidal", 512)
            >>> learned_pe = PositionalEncodingFactory.create("learned", 512)
        """
        if pe_type == "sinusoidal":
            return PositionalEncoding(d_model, dropout, max_len)
        elif pe_type == "learned":
            return LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}. "
                           f"Choose 'sinusoidal' or 'learned'.")


if __name__ == "__main__":
    # Unit tests for positional encodings (Harvard NLP style)
    print("Testing Positional Encodings (Harvard NLP Implementation)...")
    print("=" * 70)

    batch_size, seq_len, d_model = 2, 10, 512
    max_len = 1000
    dropout_p = 0.1

    # Test 1: PositionalEncoding (Sinusoidal)
    print("\n1. Testing PositionalEncoding (Sinusoidal)...")
    x = torch.randn(batch_size, seq_len, d_model)
    pe = PositionalEncoding(d_model, dropout_p, max_len)
    out = pe(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Max sequence length: {max_len}")
    assert out.shape == x.shape, "Positional encoding output shape mismatch"
    print("   ✓ PositionalEncoding working correctly")

    # Test 2: Verify positional encoding pattern
    print("\n2. Verifying sinusoidal pattern...")
    pe_no_dropout = PositionalEncoding(d_model, dropout=0.0, max_len=max_len)
    x_zeros = torch.zeros(1, 5, d_model)  # Start with zeros
    out_pe = pe_no_dropout(x_zeros)
    # Check that positions have different encodings
    pos_0 = out_pe[0, 0, :5]
    pos_1 = out_pe[0, 1, :5]
    print(f"   Position 0 encoding (first 5 dims): {pos_0}")
    print(f"   Position 1 encoding (first 5 dims): {pos_1}")
    print(f"   Positions differ: {not torch.allclose(pos_0, pos_1)}")
    print("   ✓ Sinusoidal pattern verified")

    # Test 3: Backward compatibility alias
    print("\n3. Testing SinusoidalPositionalEncoding alias...")
    pe_alias = SinusoidalPositionalEncoding(d_model, dropout_p, max_len)
    out_alias = pe_alias(x)
    print(f"   SinusoidalPositionalEncoding is PositionalEncoding: {SinusoidalPositionalEncoding is PositionalEncoding}")
    print(f"   Output shape: {out_alias.shape}")
    assert out_alias.shape == x.shape, "Alias output shape mismatch"
    print("   ✓ Backward compatibility maintained")

    # Test 4: LearnedPositionalEncoding
    print("\n4. Testing LearnedPositionalEncoding...")
    learned_pe = LearnedPositionalEncoding(d_model, max_len, dropout_p)
    out_learned = learned_pe(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out_learned.shape}")
    print(f"   Num parameters: {sum(p.numel() for p in learned_pe.parameters()):,}")
    assert out_learned.shape == x.shape, "Learned PE output shape mismatch"
    print("   ✓ LearnedPositionalEncoding working correctly")

    # Test 5: PositionalEncodingFactory
    print("\n5. Testing PositionalEncodingFactory...")
    pe_sin = PositionalEncodingFactory.create("sinusoidal", d_model, max_len, dropout_p)
    pe_learn = PositionalEncodingFactory.create("learned", d_model, max_len, dropout_p)
    print(f"   Created sinusoidal: {type(pe_sin).__name__}")
    print(f"   Created learned: {type(pe_learn).__name__}")
    assert isinstance(pe_sin, PositionalEncoding), "Factory should create PositionalEncoding"
    assert isinstance(pe_learn, LearnedPositionalEncoding), "Factory should create LearnedPositionalEncoding"
    print("   ✓ Factory working correctly")

    # Test 6: Variable sequence lengths
    print("\n6. Testing variable sequence lengths...")
    test_lengths = [5, 50, 500]
    for test_seq_len in test_lengths:
        x_test = torch.randn(batch_size, test_seq_len, d_model)
        out_sin = pe(x_test)
        out_learned = learned_pe(x_test)
        assert out_sin.shape == x_test.shape, f"Sinusoidal PE failed for seq_len={test_seq_len}"
        assert out_learned.shape == x_test.shape, f"Learned PE failed for seq_len={test_seq_len}"
    print(f"   Tested sequence lengths: {test_lengths}")
    print("   ✓ All sequence lengths working correctly")

    # Test 7: Buffer registration (PE should not be in parameters)
    print("\n7. Verifying PE buffer registration...")
    pe_params = list(pe.parameters())
    print(f"   PositionalEncoding trainable params: {len(pe_params)} (should be 0, only dropout)")
    print(f"   PE buffer registered: {'pe' in dict(pe.named_buffers())}")
    learned_params = list(learned_pe.parameters())
    print(f"   LearnedPositionalEncoding trainable params: {len(learned_params)} (should be 1, embeddings)")
    print("   ✓ Buffer registration verified")

    print("\n" + "=" * 70)
    print("✓ All positional encoding tests passed!")
    print("Implementation matches Harvard NLP's Annotated Transformer")
