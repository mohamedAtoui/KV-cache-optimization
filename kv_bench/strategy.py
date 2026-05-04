"""Abstract base class for KV cache strategies and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StrategyResult:
    """Results from running a single strategy benchmark."""
    name: str
    perplexity: float
    avg_loss: float
    num_tokens: int
    memory_peak_mb: float
    memory_kv_analytical_mb: float
    prefill_latency_ms: float
    decode_latency_ms_per_token: float
    compression_ratio: float
    extra: dict = field(default_factory=dict)


class KVCacheStrategy(ABC):
    """Abstract base class for KV cache strategies.

    Each strategy can patch a model to modify its KV cache behavior,
    report analytical memory usage, and cleanly unpatch when done.
    """

    name: str = "unnamed"

    @abstractmethod
    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        """Patch the model to use this strategy.

        Args:
            model: HuggingFace causal LM.
            model_config: The model's config object.
            device_config: DeviceConfig with GPU/memory info.

        Returns:
            The patched model (may be modified in-place).
        """
        ...

    def reset(self):
        """Clear any internal cache state (e.g., between eval windows)."""
        pass

    @abstractmethod
    def memory_bytes(self, seq_len: int, model_config) -> int:
        """Analytical KV cache memory for a given sequence length.

        Args:
            seq_len: Sequence length to compute memory for.
            model_config: Model config with num_layers, num_kv_heads, head_dim.

        Returns:
            Estimated KV cache memory in bytes.
        """
        ...

    def teardown(self, model: nn.Module) -> nn.Module:
        """Unpatch the model, restoring original forward methods.

        Returns:
            The restored model.
        """
        return model

    def needs_attention_weights(self) -> bool:
        """Whether this strategy needs attention weights from the model forward."""
        return False

    def on_step(self, layer_idx: int, attn_weights=None, key_states=None,
                value_states=None, **kwargs):
        """Hook called after each layer forward (for strategies that need weights).

        Only called if needs_attention_weights() returns True.
        """
        pass

    def get_keep_mask(self, seq_len: int, device) -> Optional[torch.Tensor]:
        """Return [seq_len] bool tensor: True = keep, False = evict.

        Returns None if strategy doesn't evict (single-pass eval).
        """
        return None

    def get_zone_masks(self, seq_len: int, device) -> Optional[dict[int, torch.Tensor]]:
        """Return per-layer zone assignments for quantization simulation.

        Returns:
            Dict mapping layer_idx -> [num_kv_heads, seq_len] int tensor of zone IDs
            (0=FP16, 1=INT8, 2=INT4, 3=EVICT), or None if not applicable.
        """
        return None
