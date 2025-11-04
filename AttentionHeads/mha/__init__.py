"""
GPTNeo Decoder-Only Transformer Package

A complete implementation of GPTNeo-style decoder-only transformer
optimized for training on TinyStories dataset with A100 GPUs.

Key Features:
    - GPTNeo decoder-only architecture (~85-95M parameters)
    - BFloat16 mixed precision training
    - TinyStories dataset integration
    - A100-optimized training pipeline
    - Built-in text generation

Example:
    >>> from mha import GPTNeoForCausalLM, create_gptneo_model
    >>>
    >>> # Create model
    >>> config = {
    ...     'vocab_size': 50257,
    ...     'hidden_size': 768,
    ...     'num_layers': 8,
    ...     'num_heads': 12,
    ...     'intermediate_size': 3072,
    ...     'max_position_embeddings': 512,
    ...     'dropout': 0.2
    ... }
    >>> model = create_gptneo_model(config)
    >>>
    >>> # Train
    >>> from mha import GPTNeoTrainer
    >>> trainer = GPTNeoTrainer(full_config)
    >>> trainer.train()

Reference:
    Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be
    and Still Speak Coherent English? arXiv:2305.07759.
"""

__version__ = "1.0.0"
__author__ = "Attaimen"
__email__ = "wmis066@live.rhul.ac.uk"

# Core architecture
from .transformer import (
    GPTNeoBlock,
    GPTNeoModel,
    GPTNeoForCausalLM,
    create_gptneo_model,
)

# Training
from .train import GPTNeoTrainer

# Data loading
from .data_loader import (
    TinyStoriesDataset,
    TinyStoriesDataModule,
    load_config,
)

# Attention mechanism
from .attention import (
    MultiHeadedAttention,
    attention,
    subsequent_mask,
    create_causal_mask,
    create_padding_mask,
    create_combined_mask,
)

# Layers
from .layers import (
    LayerNorm,
    PositionwiseFeedForward,
    SublayerConnection,
    clones,
)

# Utilities
from .utils import (
    MetricsTracker,
    Logger,
    CheckpointManager,
    set_seed,
    count_parameters,
)

# Positional encoding (legacy - not used in GPTNeo but kept for compatibility)
from .positional_encoding import (
    PositionalEncoding,
    PositionalEncodingFactory,
)

# Define public API
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",

    # Core architecture
    "GPTNeoBlock",
    "GPTNeoModel",
    "GPTNeoForCausalLM",
    "create_gptneo_model",

    # Training
    "GPTNeoTrainer",

    # Data
    "TinyStoriesDataset",
    "TinyStoriesDataModule",
    "load_config",

    # Attention
    "MultiHeadedAttention",
    "attention",
    "subsequent_mask",
    "create_causal_mask",
    "create_padding_mask",
    "create_combined_mask",

    # Layers
    "LayerNorm",
    "PositionwiseFeedForward",
    "SublayerConnection",
    "clones",

    # Utilities
    "MetricsTracker",
    "Logger",
    "CheckpointManager",
    "set_seed",
    "count_parameters",

    # Positional encoding (legacy)
    "PositionalEncoding",
    "PositionalEncodingFactory",
]

# Package metadata
__doc_url__ = "https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT/blob/main/AttentionHeads/mha/README.md"
__source_url__ = "https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT"

# Print welcome message when imported
def _print_info():
    """Print package information"""
    print(f"GPTNeo TinyStories v{__version__}")
    print(f"Decoder-only transformer for causal language modeling")
    print(f"Documentation: {__doc_url__}")

# Uncomment to show info on import
# _print_info()
