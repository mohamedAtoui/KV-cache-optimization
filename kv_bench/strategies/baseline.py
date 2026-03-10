"""Full KV cache baseline — no compression, reference perplexity."""

import torch.nn as nn

from kv_bench.strategy import KVCacheStrategy


class FullKVBaseline(KVCacheStrategy):
    """No-op baseline: standard full-precision KV cache."""

    name = "FullKV (baseline)"

    def setup(self, model: nn.Module, model_config, device_config) -> nn.Module:
        return model

    def memory_bytes(self, seq_len: int, model_config) -> int:
        num_layers = model_config.num_hidden_layers
        num_kv_heads = getattr(model_config, "num_key_value_heads",
                               model_config.num_attention_heads)
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        # K + V, 2 bytes per bf16 element
        return num_layers * num_kv_heads * seq_len * head_dim * 2 * 2

    def teardown(self, model: nn.Module) -> nn.Module:
        return model
