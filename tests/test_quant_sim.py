"""Tests for quantization simulation (round-trip noise injection).

Validates INT8/INT4 symmetric quantization and TurboQuant produce
correct shapes, expected error magnitudes, and mathematical properties.
"""

import torch
import pytest

from kv_bench.quant_sim import (
    simulate_int8, simulate_int4,
    simulate_int8_per_channel, simulate_int4_per_channel,
)
from streaming_attention.turboquant import (
    TurboQuantConfig, init_turboquant_state,
    turboquant_quantize, turboquant_dequantize, simulate_turboquant,
)


class TestINT8:

    def test_output_shape(self):
        x = torch.randn(2, 8, 128)
        out = simulate_int8(x)
        assert out.shape == x.shape

    def test_output_dtype(self):
        x = torch.randn(2, 8, 128)
        out = simulate_int8(x)
        assert out.dtype == x.dtype

    def test_mse_is_small(self):
        x = torch.randn(2, 8, 128)
        out = simulate_int8(x)
        mse = (x - out).pow(2).mean().item()
        assert mse < 0.01  # INT8 should be very close

    def test_per_channel_shape(self):
        x = torch.randn(2, 8, 128)
        out = simulate_int8_per_channel(x)
        assert out.shape == x.shape


class TestINT4:

    def test_output_shape(self):
        x = torch.randn(2, 8, 128)
        out = simulate_int4(x)
        assert out.shape == x.shape

    def test_higher_error_than_int8(self):
        x = torch.randn(2, 8, 128)
        mse_int8 = (x - simulate_int8(x)).pow(2).mean().item()
        mse_int4 = (x - simulate_int4(x)).pow(2).mean().item()
        assert mse_int4 > mse_int8  # INT4 should have more error

    def test_per_channel_shape(self):
        x = torch.randn(2, 8, 128)
        out = simulate_int4_per_channel(x)
        assert out.shape == x.shape


class TestTurboQuant:

    @pytest.fixture
    def tq_state(self):
        config = TurboQuantConfig(head_dim=64, bits_stage1=2, qjl_enabled=True)
        return init_turboquant_state(config, "cpu", torch.float32), config

    def test_rotation_matrix_orthogonal(self, tq_state):
        state, _ = tq_state
        Pi = state.rotation_matrix
        I = Pi @ Pi.T
        assert torch.allclose(I, torch.eye(64), atol=1e-5)

    def test_codebook_levels(self, tq_state):
        state, _ = tq_state
        assert state.codebooks[2].shape == (4,)   # 2^2 = 4 levels
        assert state.codebooks[3].shape == (8,)   # 2^3 = 8 levels
        assert state.codebooks[4].shape == (16,)  # 2^4 = 16 levels

    def test_codebook_sorted(self, tq_state):
        state, _ = tq_state
        for bits in (2, 3, 4):
            cb = state.codebooks[bits]
            assert (cb[1:] >= cb[:-1]).all(), f"Codebook for {bits} bits not sorted"

    def test_round_trip_shape(self, tq_state):
        state, config = tq_state
        x = torch.randn(2, 8, 64)
        out = simulate_turboquant(x, state, config)
        assert out.shape == x.shape

    def test_round_trip_dtype(self, tq_state):
        state, config = tq_state
        x = torch.randn(2, 8, 64)
        out = simulate_turboquant(x, state, config)
        assert out.dtype == x.dtype

    def test_quantize_returns_codes_signs_norms_scales(self, tq_state):
        state, config = tq_state
        x = torch.randn(2, 8, 64)
        codes, signs, norms, scales = turboquant_quantize(x, state, config)
        assert codes.shape == (2, 8, 64)
        assert codes.dtype == torch.int8
        assert signs.shape == (2, 8, 64)
        assert signs.dtype == torch.int8
        assert norms.shape == (2, 8)
        assert scales.shape == (2, 8)
