"""Tests for KVCacheStrategy implementations.

Validates that each strategy follows the interface contract:
correct setup/teardown, memory_bytes returns sane values,
zone masks have correct shapes.
"""

import torch
import pytest

from kv_bench.strategies.baseline import FullKVBaseline
from kv_bench.strategies.uniform_quant import UniformQuantStrategy
from kv_bench.strategies.turboquant import TurboQuantStrategy


class TestFullKVBaseline:

    def test_name(self):
        s = FullKVBaseline()
        assert "baseline" in s.name.lower() or "fullkv" in s.name.lower()

    def test_memory_bytes_positive(self, fake_model_config):
        s = FullKVBaseline()
        mem = s.memory_bytes(1024, fake_model_config)
        assert mem > 0

    def test_needs_no_attention_weights(self):
        s = FullKVBaseline()
        assert s.needs_attention_weights() is False


class TestUniformQuantStrategy:

    def test_int8_name(self):
        s = UniformQuantStrategy(bits=8)
        assert "8" in s.name

    def test_int4_name(self):
        s = UniformQuantStrategy(bits=4)
        assert "4" in s.name

    def test_int8_memory_less_than_fp16(self, fake_model_config):
        baseline = FullKVBaseline()
        int8 = UniformQuantStrategy(bits=8)
        mem_fp16 = baseline.memory_bytes(1024, fake_model_config)
        mem_int8 = int8.memory_bytes(1024, fake_model_config)
        assert mem_int8 < mem_fp16

    def test_int4_memory_less_than_int8(self, fake_model_config):
        int8 = UniformQuantStrategy(bits=8)
        int4 = UniformQuantStrategy(bits=4)
        mem_int8 = int8.memory_bytes(1024, fake_model_config)
        mem_int4 = int4.memory_bytes(1024, fake_model_config)
        assert mem_int4 < mem_int8

    def test_needs_attention_weights(self):
        s = UniformQuantStrategy(bits=8)
        assert s.needs_attention_weights() is True

    def test_get_keep_mask_all_true(self, fake_model_config, fake_device_config):
        s = UniformQuantStrategy(bits=8)
        s.setup(None, fake_model_config, fake_device_config)
        mask = s.get_keep_mask(64, "cpu")
        assert mask.shape == (64,)
        assert mask.all()  # uniform quant keeps all tokens

    def test_zone_masks_shape(self, fake_model_config, fake_device_config):
        s = UniformQuantStrategy(bits=8)
        s.setup(None, fake_model_config, fake_device_config)
        zones = s.get_zone_masks(64, "cpu")
        assert len(zones) == fake_model_config.num_hidden_layers
        for l, z in zones.items():
            assert z.shape == (fake_model_config.num_key_value_heads, 64)

    def test_reset_no_error(self):
        s = UniformQuantStrategy(bits=8)
        s.reset()  # should not raise


class TestTurboQuantStrategy:

    def test_tq3_name(self):
        s = TurboQuantStrategy(bits_stage1=2, qjl=True)
        assert "3" in s.name  # TQ3 = 2+1

    def test_tq4_name(self):
        s = TurboQuantStrategy(bits_stage1=3, qjl=True)
        assert "4" in s.name  # TQ4 = 3+1

    def test_memory_less_than_fp16(self, fake_model_config):
        baseline = FullKVBaseline()
        tq3 = TurboQuantStrategy(bits_stage1=2, qjl=True)
        mem_fp16 = baseline.memory_bytes(1024, fake_model_config)
        mem_tq3 = tq3.memory_bytes(1024, fake_model_config)
        assert mem_tq3 < mem_fp16

    def test_needs_attention_weights(self):
        s = TurboQuantStrategy(bits_stage1=2, qjl=True)
        assert s.needs_attention_weights() is True
