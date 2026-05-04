"""
GPTNeo Decoder-Only Transformer with Multi-Query Attention (MQA)

Multi-Query Attention implementation based on "Fast Transformer Decoding:
One Write-head is All You Need" (Shazeer 2019).

Key Differences from mha/:
    - Uses MultiQueryAttention (shared K, V across heads) instead of MultiHeadedAttention
    - ~44% fewer attention parameters
    - Faster inference with reduced memory bandwidth
    - All other components reused from mha/

Benefits:
    - Faster decoding during generation
    - Smaller KV cache
    - Better for resource-constrained deployments
    - Minor quality degradation vs standard MHA

Example:
    >>> from mqa import GPTNeoForCausalLM, create_gptneo_model
    >>>
    >>> # Create MQA model
    >>> config = {
    ...     'vocab_size': 50257,
    ...     'hidden_size': 256,
    ...     'num_layers': 4,
    ...     'num_heads': 8,
    ...     'intermediate_size': 1024,
    ...     'max_position_embeddings': 256,
    ...     'dropout': 0.2
    ... }
    >>> model = create_gptneo_model(config)
    >>>
    >>> # Train (same as mha)
    >>> from mha import GPTNeoTrainer
    >>> trainer = GPTNeoTrainer(full_config)
    >>> trainer.train()

Reference:
    Shazeer, N. (2019). Fast Transformer Decoding: One Write-head is All You Need.
    arXiv preprint arXiv:1911.02150.
"""

__version__ = "1.0.0-mqa"
__author__ = "Attaimen"
__email__ = "wmis066@live.rhul.ac.uk"

# MQA-specific modules (only attention and transformer differ from mha)
from .transformer import (
    GPTNeoBlock,
    GPTNeoModel,
    GPTNeoForCausalLM,
    create_gptneo_model,
    GPTNeo,  # Alias
)

from .attention import (
    MultiQueryAttention,
    attention,
)

# Import shared modules from mha (everything else is the same)
from AttentionHeads.mha.train import GPTNeoTrainer
from AttentionHeads.mha.data_loader import (
    TinyStoriesDataset,
    TinyStoriesDataModule,
    load_config,
)
from AttentionHeads.mha.layers import (
    LayerNorm,
    PositionwiseFeedForward,
    SublayerConnection,
    clones,
)
from AttentionHeads.mha.utils import (
    MetricsTracker,
    Logger,
    CheckpointManager,
    set_seed,
    count_parameters,
    validate_config,
)

# Also import mask utilities from mha
from AttentionHeads.mha.attention import (
    create_causal_mask,
    subsequent_mask,
    create_padding_mask,
    create_combined_mask,
)

# Define public API
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",

    # Core architecture (MQA-specific)
    "GPTNeoBlock",
    "GPTNeoModel",
    "GPTNeoForCausalLM",
    "GPTNeo",
    "create_gptneo_model",

    # Attention (MQA-specific)
    "MultiQueryAttention",
    "attention",
    "create_causal_mask",
    "subsequent_mask",
    "create_padding_mask",
    "create_combined_mask",

    # Training (from mha)
    "GPTNeoTrainer",

    # Data (from mha)
    "TinyStoriesDataset",
    "TinyStoriesDataModule",
    "load_config",

    # Layers (from mha)
    "LayerNorm",
    "PositionwiseFeedForward",
    "SublayerConnection",
    "clones",

    # Utilities (from mha)
    "MetricsTracker",
    "Logger",
    "CheckpointManager",
    "set_seed",
    "count_parameters",
    "validate_config",
]

# Package metadata
__doc_url__ = "https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT/blob/main/AttentionHeads/mqa/README.md"
__source_url__ = "https://gitlab.cim.rhul.ac.uk/wmis066/PROJECT"
