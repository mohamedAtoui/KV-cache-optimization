"""Streaming attention strategy: streaming heads → recurrent state, retrieval heads → full KV."""

import logging
from typing import Optional

import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy

logger = logging.getLogger(__name__)


class StreamingAttentionStrategy(KVCacheStrategy):
    """Wraps the streaming_attention package's hybrid attention patching."""

    name = "StreamingAttention"

    def __init__(
        self,
        pattern_dir: Optional[str] = None,
        threshold: float = 0.5,
        decay_init: float = 0.99,
        checkpoint_path: Optional[str] = None,
    ):
        self.pattern_dir = pattern_dir
        self.threshold = threshold
        self.decay_init = decay_init
        self.checkpoint_path = checkpoint_path
        self._state_cache = None
        self._head_classification = None

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        from streaming_attention.head_classifier import load_duo_attention_patterns
        from streaming_attention.hybrid_attention import patch_model_for_streaming_attention, StreamingAttentionConfig

        if self.pattern_dir is None:
            raise ValueError(
                "StreamingAttentionStrategy requires pattern_dir pointing to DuoAttention patterns"
            )

        self._head_classification = load_duo_attention_patterns(
            self.pattern_dir, threshold=self.threshold
        )

        config = StreamingAttentionConfig(decay_init=self.decay_init)
        model, self._state_cache = patch_model_for_streaming_attention(
            model, self._head_classification, config
        )

        # Load calibrated weights if available
        if self.checkpoint_path:
            logger.info(f"Loading calibrated state from {self.checkpoint_path}")
            if hasattr(model, '_streaming_attention_modules'):
                import torch
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                model._streaming_attention_modules.load_state_dict(state_dict, strict=False)

        return model

    def reset(self):
        if self._state_cache is not None:
            self._state_cache.reset()

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        if self._head_classification is None:
            # Estimate: ~50% streaming
            n_streaming = num_layers * num_kv_heads // 2
            n_retrieval = num_layers * num_kv_heads - n_streaming
        else:
            mask = self._head_classification.mask
            n_retrieval = mask.sum().item()
            n_streaming = mask.numel() - n_retrieval

        # Streaming heads: fixed state matrix D×D + normalization vector D per head
        streaming_bytes = int(n_streaming * (head_dim * head_dim + head_dim) * 2)  # bf16
        # Retrieval heads: full KV cache
        retrieval_bytes = int(n_retrieval * seq_len * head_dim * 2 * 2)  # K+V, bf16

        return streaming_bytes + retrieval_bytes

    def teardown(self, model: nn.Module) -> nn.Module:
        # Restore original forwards
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, '_original_forward'):
                attn.forward = attn._original_forward
                del attn._original_forward
                if hasattr(attn, '_streaming_attention_patched'):
                    del attn._streaming_attention_patched

        # Clean up stored modules
        if hasattr(model, '_streaming_attention_modules'):
            del model._streaming_attention_modules
        if hasattr(model, '_streaming_attention_cache'):
            del model._streaming_attention_cache

        self._state_cache = None
        return model
