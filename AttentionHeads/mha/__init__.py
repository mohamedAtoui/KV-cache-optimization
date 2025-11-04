"""
Multi-Head Attention (MHA) Transformer Implementation

Based on "Attention Is All You Need" (Vaswani et al., 2017)
Implementation follows Harvard NLP's Annotated Transformer:
https://nlp.seas.harvard.edu/annotated-transformer/

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances
    in neural information processing systems (pp. 5998-6008).

Example usage (Harvard NLP style):
    from mha import make_model

    # Create standard transformer (Harvard NLP way)
    model = make_model(src_vocab=10000, tgt_vocab=10000, N=6)

Example usage (backward compatible):
    from mha import Transformer

    # Create transformer (legacy way, still supported)
    model = Transformer(vocab_size=50257, d_model=512, num_heads=8)
"""

__version__ = "1.0.0"
__author__ = "LLM-Journey Contributors"

# Harvard NLP Transformer Components
from .transformer import (
    # Harvard NLP classes
    EncoderDecoder,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    Embeddings,
    Generator,
    make_model,
    # Backward compatibility
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
)

# Attention mechanisms
from .attention import (
    # Harvard NLP
    attention,
    MultiHeadedAttention,
    subsequent_mask,
    # Backward compatibility
    MultiHeadAttention,
    # Mask utilities
    create_causal_mask,
    create_padding_mask,
    create_combined_mask,
)

# Layers
from .layers import (
    # Harvard NLP
    clones,
    LayerNorm,
    PositionwiseFeedForward,
    SublayerConnection,
    # Backward compatibility
    FeedForward,
    ResidualConnection,
    DropoutLayer,
)

# Positional encodings
from .positional_encoding import (
    # Harvard NLP
    PositionalEncoding,
    # Alternative
    LearnedPositionalEncoding,
    # Backward compatibility
    SinusoidalPositionalEncoding,
    PositionalEncodingFactory,
)

# Training utilities
from .utils import (
    # Harvard NLP
    rate,
    Batch,
    # Other utilities
    MetricsTracker,
    LabelSmoothing,
    Logger,
    CheckpointManager,
    AttentionVisualizer,
)

# Inference utilities
from .inference import (
    # Harvard NLP
    greedy_decode,
    # Text generation
    TextGenerator,
)

# Define what gets imported with "from mha import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Harvard NLP Transformer (RECOMMENDED)
    'make_model',
    'EncoderDecoder',
    'Encoder',
    'Decoder',
    'EncoderLayer',
    'DecoderLayer',
    'Embeddings',
    'Generator',

    # Backward compatibility
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',

    # Attention (Harvard NLP)
    'attention',
    'MultiHeadedAttention',
    'subsequent_mask',

    # Attention (backward compatibility)
    'MultiHeadAttention',
    'create_causal_mask',
    'create_padding_mask',
    'create_combined_mask',

    # Layers (Harvard NLP)
    'clones',
    'PositionwiseFeedForward',
    'SublayerConnection',

    # Layers (backward compatibility)
    'LayerNorm',
    'FeedForward',
    'ResidualConnection',
    'DropoutLayer',

    # Positional encodings
    'PositionalEncoding',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'PositionalEncodingFactory',

    # Training utilities (Harvard NLP)
    'rate',
    'Batch',

    # Other utilities
    'MetricsTracker',
    'LabelSmoothing',
    'Logger',
    'CheckpointManager',
    'AttentionVisualizer',

    # Inference (Harvard NLP)
    'greedy_decode',
    'TextGenerator',
]
