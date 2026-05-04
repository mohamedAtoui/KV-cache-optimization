"""TurboQuant strategy: uniform TurboQuant quantization of all KV entries.

Applies TurboQuant (random rotation + optimal scalar quant + QJL residual
correction) uniformly to all KV cache entries. This is the approach from
the original TurboQuant paper (Google, ICLR 2026).
"""

import logging

import torch
import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy
from streaming_attention.stratigraphic import ZONE_TQ3, ZONE_TQ4
from streaming_attention.turboquant import TurboQuantConfig, init_turboquant_state

logger = logging.getLogger(__name__)


class TurboQuantStrategy(KVCacheStrategy):
    """Uniform TurboQuant quantization of all KV cache entries.

    Applies TurboQuant round-trip noise via forward hooks on k_proj/v_proj
    so that PPL reflects real quantization error.
    """

    def __init__(self, bits_stage1: int = 2, qjl: bool = True):
        """
        Args:
            bits_stage1: Scalar quantizer bits (2 or 3).
            qjl: Whether to add +1 bit QJL residual correction.
        """
        self.bits_stage1 = bits_stage1
        self.qjl = qjl
        effective = bits_stage1 + (1 if qjl else 0)
        self.name = f"TQ{effective}-all"
        self._model_config = None

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        self._model_config = model_config
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        config = TurboQuantConfig(
            head_dim=head_dim,
            bits_stage1=self.bits_stage1,
            qjl_enabled=self.qjl,
        )
        model._tq_state = init_turboquant_state(
            config, device_config.device, device_config.dtype,
        )
        return model

    def needs_attention_weights(self) -> bool:
        return True

    def on_step(self, layer_idx: int, attn_weights=None, **kwargs):
        pass

    def get_keep_mask(self, seq_len, device):
        return torch.ones(seq_len, dtype=torch.bool, device=device)

    def get_zone_masks(self, seq_len, device):
        if self._model_config is None:
            return None
        num_layers = self._model_config.num_hidden_layers
        num_kv_heads = getattr(
            self._model_config, "num_key_value_heads",
            self._model_config.num_attention_heads,
        )
        zone_id = ZONE_TQ3 if (self.bits_stage1 == 2 and self.qjl) else ZONE_TQ4
        zone_masks = {}
        for l in range(num_layers):
            zone_masks[l] = torch.full(
                (num_kv_heads, seq_len), zone_id,
                dtype=torch.long, device=device,
            )
        return zone_masks

    def reset(self):
        pass

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        effective_bits = self.bits_stage1 + (1 if self.qjl else 0)
        # K + V entries: effective_bits per coordinate
        kv_bytes = num_layers * num_kv_heads * seq_len * head_dim * (effective_bits / 8) * 2
        # Per-vector scale: 1 float32 per vector (K and V each), always stored
        scale_bytes = num_layers * num_kv_heads * seq_len * 4 * 2
        # Residual norms: 1 float32 per vector (K and V each) if QJL enabled
        norm_bytes = 0
        if self.qjl:
            norm_bytes = num_layers * num_kv_heads * seq_len * 4 * 2
        # Global matrices (~128KB) amortized to 0
        return int(kv_bytes + scale_bytes + norm_bytes)

    def teardown(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "_tq_state"):
            del model._tq_state
        self._model_config = None
        return model
