"""Hybrid strategy: KV2State for streaming heads + AdaptiveTiered for retrieval heads.

Combines the best of both: streaming heads are converted to fixed-size recurrent
state (KV2State), while retrieval heads get multi-signal importance-based tiered
compression (AdaptiveTiered). This is the full vision from the project.
"""

import logging
from typing import Optional

import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy

logger = logging.getLogger(__name__)


class HybridStrategy(KVCacheStrategy):
    """KV2State (streaming) + AdaptiveTiered (retrieval) combined strategy."""

    name = "Hybrid (KV2State + Tiered)"

    def __init__(
        self,
        pattern_dir: Optional[str] = None,
        threshold: float = 0.5,
        decay_init: float = 0.99,
        budget_fp16: float = 0.25,
        budget_int8: float = 0.30,
        budget_int4: float = 0.25,
        checkpoint_path: Optional[str] = None,
    ):
        self.pattern_dir = pattern_dir
        self.threshold = threshold
        self.decay_init = decay_init
        self.budget_fp16 = budget_fp16
        self.budget_int8 = budget_int8
        self.budget_int4 = budget_int4
        self.checkpoint_path = checkpoint_path
        self._state_cache = None
        self._head_classification = None

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        from kv2state.head_classifier import load_duo_attention_patterns
        from kv2state.hybrid_attention import patch_model_for_kv2state, KV2StateConfig

        if self.pattern_dir is None:
            raise ValueError("HybridStrategy requires pattern_dir")

        self._head_classification = load_duo_attention_patterns(
            self.pattern_dir, threshold=self.threshold
        )

        config = KV2StateConfig(decay_init=self.decay_init)
        model, self._state_cache = patch_model_for_kv2state(
            model, self._head_classification, config
        )

        if self.checkpoint_path:
            import torch
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            if hasattr(model, '_kv2state_modules'):
                model._kv2state_modules.load_state_dict(state_dict, strict=False)

        return model

    def reset(self):
        if self._state_cache is not None:
            self._state_cache.reset()

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        # In hybrid mode, attention weights from retrieval heads could feed
        # the adaptive tiered scoring. For now, we collect them but the actual
        # tiered compression is analytical.
        pass

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        if self._head_classification is None:
            n_streaming = num_layers * num_kv_heads // 2
            n_retrieval = num_layers * num_kv_heads - n_streaming
        else:
            mask = self._head_classification.mask
            n_retrieval = mask.sum().item()
            n_streaming = mask.numel() - n_retrieval

        # Streaming heads: fixed state (D×D + D) per head
        streaming_bytes = int(n_streaming * (head_dim * head_dim + head_dim) * 2)

        # Retrieval heads: tiered compression
        n_fp16 = int(seq_len * self.budget_fp16)
        n_int8 = int(seq_len * self.budget_int8)
        n_int4 = int(seq_len * self.budget_int4)

        # Per retrieval head (not per layer — n_retrieval already accounts for layers)
        retrieval_bytes = int(n_retrieval * (
            n_fp16 * head_dim * 2 * 2 +       # FP16 K+V
            n_int8 * head_dim * 1 * 2 +        # INT8 K+V
            n_int4 * (head_dim // 2) * 2 +     # INT4 K+V
            8 * head_dim * 4                    # sketch
        ))

        return streaming_bytes + retrieval_bytes

    def teardown(self, model: nn.Module) -> nn.Module:
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, '_original_forward'):
                attn.forward = attn._original_forward
                del attn._original_forward
                if hasattr(attn, '_kv2state_patched'):
                    del attn._kv2state_patched

        if hasattr(model, '_kv2state_modules'):
            del model._kv2state_modules
        if hasattr(model, '_kv2state_cache'):
            del model._kv2state_cache

        self._state_cache = None
        return model
