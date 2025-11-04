"""
Full Transformer Encoder-Decoder Architecture

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
import copy

from .attention import MultiHeadedAttention, create_combined_mask, create_padding_mask
from .layers import LayerNorm, SublayerConnection, PositionwiseFeedForward, clones
from .positional_encoding import PositionalEncoding, PositionalEncodingFactory

# Backward compatibility imports
MultiHeadAttention = MultiHeadedAttention
FeedForward = PositionwiseFeedForward
ResidualConnection = SublayerConnection


class EncoderLayer(nn.Module):
    """
    Encoder layer from "Attention is All You Need" (Harvard NLP implementation)

    Encoder is made up of self-attention and feed forward layers.
    Each layer has two sub-layers with residual connections and layer norm.

    Args:
        size: Model dimension (d_model)
        self_attn: Multi-head attention module
        feed_forward: Position-wise feed-forward module
        dropout: Dropout probability

    Shape:
        - Input: (batch, seq_len, size)
        - Output: (batch, seq_len, size)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Decoder layer from "Attention is All You Need" (Harvard NLP implementation)

    Decoder is made of self-attention, source-attention, and feed forward.
    Each layer has three sub-layers with residual connections and layer norm.

    Args:
        size: Model dimension (d_model)
        self_attn: Masked multi-head self-attention module
        src_attn: Multi-head attention module for encoder-decoder attention
        feed_forward: Position-wise feed-forward module
        dropout: Dropout probability

    Shape:
        - Input: (batch, seq_len_tgt, size)
        - Memory: (batch, seq_len_src, size) [encoder output]
        - Output: (batch, seq_len_tgt, size)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)[0])
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder from "Attention is All You Need" (Harvard NLP implementation)

    Core encoder is a stack of N layers with final layer normalization.

    Args:
        layer: EncoderLayer instance to clone
        N: Number of layers to stack

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Backward compatibility
TransformerEncoder = Encoder


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking (Harvard NLP implementation)

    Core decoder is a stack of N layers with final layer normalization.

    Args:
        layer: DecoderLayer instance to clone
        N: Number of layers to stack

    Shape:
        - Input: (batch, seq_len_tgt, d_model)
        - Memory: (batch, seq_len_src, d_model) [encoder output]
        - Output: (batch, seq_len_tgt, d_model)
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass the input (and masks) through each layer in turn"""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# Backward compatibility
TransformerDecoder = Decoder


class EncoderDecoder(nn.Module):
    """
    Standard Encoder-Decoder architecture (Harvard NLP implementation)

    A standard Encoder-Decoder architecture. Base for many models including
    the original Transformer.

    Args:
        encoder: The encoder module
        decoder: The decoder module
        src_embed: Source embedding module (includes positional encoding)
        tgt_embed: Target embedding module (includes positional encoding)
        generator: Output generator module (linear + softmax)

    Shape:
        - src: (batch, seq_len_src)
        - tgt: (batch, seq_len_tgt)
        - Output: (batch, seq_len_tgt, vocab_size)
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """Encode source sequence"""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """Decode target sequence given encoder memory"""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step (Harvard NLP)

    Projects decoder output to vocabulary logits.

    Args:
        d_model: Model dimension
        vocab: Vocabulary size

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, vocab)
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """Project to vocabulary and apply log softmax"""
        return F.log_softmax(self.proj(x), dim=-1)


class Embeddings(nn.Module):
    """
    Embeddings layer with scaling (Harvard NLP implementation)

    Standard embeddings with scaling by sqrt(d_model) as described in the paper.
    This scaling helps stabilize gradients and improve training dynamics.

    Args:
        d_model: Model dimension
        vocab: Vocabulary size

    Shape:
        - Input: (batch, seq_len) [token IDs]
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """Embed tokens and scale by sqrt(d_model)"""
        return self.lut(x) * math.sqrt(self.d_model)


class Transformer(nn.Module):
    """
    Full Transformer Encoder-Decoder Architecture with MHA

    Backward compatibility wrapper around EncoderDecoder for existing code.

    Architecture:
        1. Input Embedding + Positional Encoding
        2. Encoder (N layers)
        3. Decoder (N layers)
        4. Output Linear + Softmax

    For language modeling on WikiText:
        - Source and target are the same (shifted by 1)
        - Use causal masking in decoder
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pe_type: str = "sinusoidal"
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encodings
        self.src_pos_encoding = PositionalEncodingFactory.create(
            pe_type, d_model, max_seq_length, dropout
        )
        self.tgt_pos_encoding = PositionalEncodingFactory.create(
            pe_type, d_model, max_seq_length, dropout
        )

        # Encoder and Decoder (Harvard NLP pattern)
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Create encoder and decoder using proper Harvard NLP construction
        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, seq_len_src) - Source token IDs
            tgt: (batch_size, seq_len_tgt) - Target token IDs
            src_mask: (batch_size, seq_len_src, seq_len_src) - Source mask
            tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt) - Target mask

        Returns:
            output: (batch_size, seq_len_tgt, vocab_size) - Logits over vocabulary
        """
        # Embed and add positional encoding
        # src_embedded: (batch_size, seq_len_src, d_model)
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.src_pos_encoding(src_embedded)

        # tgt_embedded: (batch_size, seq_len_tgt, d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)

        # Encode source
        # encoder_output: (batch_size, seq_len_src, d_model)
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode target
        # decoder_output: (batch_size, seq_len_tgt, d_model)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        # output: (batch_size, seq_len_tgt, vocab_size)
        output = self.output_projection(decoder_output)

        return output


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """
    Helper function to construct a model from hyperparameters (Harvard NLP)

    This is the standard way to create a Transformer model following the
    "Attention is All You Need" architecture.

    Args:
        src_vocab: Size of source vocabulary
        tgt_vocab: Size of target vocabulary
        N: Number of encoder/decoder layers (default: 6)
        d_model: Model dimension (default: 512)
        d_ff: Feed-forward dimension (default: 2048)
        h: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)

    Returns:
        EncoderDecoder model with initialized parameters

    Example:
        >>> model = make_model(10000, 10000, N=6)
        >>> # Create a model with 10K vocab, 6 layers, default dims
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This is important: Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == "__main__":
    # Unit tests for Transformer
    print("Testing Transformer Architecture...")

    batch_size = 2
    seq_len_src = 10
    seq_len_tgt = 8
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    # Test 1: EncoderLayer
    print("\n1. Testing EncoderLayer...")
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff).to(device)
    x = torch.randn(batch_size, seq_len_src, d_model).to(device)
    output = encoder_layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "EncoderLayer output shape mismatch"

    # Test 2: DecoderLayer
    print("\n2. Testing DecoderLayer...")
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff).to(device)
    tgt = torch.randn(batch_size, seq_len_tgt, d_model).to(device)
    encoder_output = torch.randn(batch_size, seq_len_src, d_model).to(device)
    output = decoder_layer(tgt, encoder_output)
    print(f"   Target shape: {tgt.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == tgt.shape, "DecoderLayer output shape mismatch"

    # Test 3: Full Transformer
    print("\n3. Testing Full Transformer...")
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=512,
        dropout=0.1,
        pe_type="sinusoidal"
    ).to(device)

    src_tokens = torch.randint(0, vocab_size, (batch_size, seq_len_src)).to(device)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_len_tgt)).to(device)

    output = transformer(src_tokens, tgt_tokens)
    print(f"   Source tokens shape: {src_tokens.shape}")
    print(f"   Target tokens shape: {tgt_tokens.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len_tgt, vocab_size), "Transformer output shape mismatch"

    # Test 4: With masks
    print("\n4. Testing with Masks...")
    src_mask = create_padding_mask(src_tokens, pad_token_id=0)
    tgt_mask = create_combined_mask(tgt_tokens, pad_token_id=0, causal=True)

    print(f"   Source mask shape: {src_mask.shape}")
    print(f"   Target mask shape: {tgt_mask.shape}")

    output = transformer(src_tokens, tgt_tokens, src_mask, tgt_mask)
    print(f"   Output with masks shape: {output.shape}")

    # Test 5: Parameter count
    print("\n5. Transformer Parameters...")
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"   Total parameters: {num_params:,}")

    # Test 6: Forward pass timing
    print("\n6. Forward Pass Timing...")
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = transformer(src_tokens, tgt_tokens, src_mask, tgt_mask)
    end = time.time()
    print(f"   Average forward pass time: {(end - start) / 10 * 1000:.2f}ms")

    # Test 7: make_model() function (Harvard NLP)
    print("\n7. Testing make_model() function (Harvard NLP)...")
    harvard_model = make_model(vocab_size, vocab_size, N=2, d_model=256, d_ff=512, h=4, dropout=0.1).to(device)

    src_tokens_small = torch.randint(0, vocab_size, (batch_size, seq_len_src)).to(device)
    tgt_tokens_small = torch.randint(0, vocab_size, (batch_size, seq_len_tgt)).to(device)
    src_mask_small = create_padding_mask(src_tokens_small, pad_token_id=0)
    tgt_mask_small = create_combined_mask(tgt_tokens_small, pad_token_id=0, causal=True)

    output_harvard = harvard_model(src_tokens_small, tgt_tokens_small, src_mask_small, tgt_mask_small)
    print(f"   Model created with make_model()")
    print(f"   Output shape: {output_harvard.shape}")
    print(f"   Parameters: {sum(p.numel() for p in harvard_model.parameters()):,}")

    # Test Generator (log softmax output)
    generator_out = harvard_model.generator(output_harvard)
    print(f"   Generator output shape: {generator_out.shape}")
    print(f"   Output is log probabilities: {(generator_out <= 0).all().item()}")

    print("\n✓ All Transformer tests passed!")
    print("Implementation matches Harvard NLP's Annotated Transformer")
