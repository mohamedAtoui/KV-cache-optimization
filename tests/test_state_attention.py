"""Tests for DecayedLinearState (streaming attention recurrent state)."""

import torch
import pytest

from streaming_attention.state_attention import DecayedLinearState, StateCache


class TestDecayedLinearState:

    @pytest.fixture
    def state_module(self):
        return DecayedLinearState(head_dim=32, decay_init=0.99)

    def test_state_shape(self, state_module):
        state = torch.zeros(1, 32, 32)
        assert state.shape == (1, 32, 32)

    def test_decay_in_range(self, state_module):
        decay = torch.sigmoid(state_module.log_decay)
        assert (decay >= 0).all()
        assert (decay <= 1).all()

    def test_recurrent_forward_shape(self, state_module):
        batch = 2
        state = torch.zeros(batch, 32, 32)
        z = torch.zeros(batch, 32)
        # recurrent_forward expects [B, head_dim] (single token, no seq dim)
        q = torch.randn(batch, 32)
        k = torch.randn(batch, 32)
        v = torch.randn(batch, 32)
        output, new_state, new_z = state_module.recurrent_forward(
            q, k, v, state, z
        )
        assert output.shape == (batch, 32)
        assert new_state.shape == (batch, 32, 32)
        assert new_z.shape == (batch, 32)

    def test_decay_one_accumulates(self):
        """With λ=1.0, state should accumulate without decay."""
        module = DecayedLinearState(head_dim=4, decay_init=0.999)
        with torch.no_grad():
            module.log_decay.fill_(100.0)  # sigmoid(100) ≈ 1.0

        state = torch.zeros(1, 4, 4)
        z = torch.zeros(1, 4)
        # [B, head_dim] — no seq dimension for recurrent
        k1 = torch.ones(1, 4)
        v1 = torch.ones(1, 4)
        q = torch.ones(1, 4)

        _, state1, z1 = module.recurrent_forward(q, k1, v1, state, z)

        k2 = torch.ones(1, 4) * 2
        v2 = torch.ones(1, 4) * 2
        _, state2, z2 = module.recurrent_forward(q, k2, v2, state1, z1)

        # State should contain contributions from both tokens
        expected = k1.unsqueeze(-1) @ v1.unsqueeze(-2)
        expected = expected + k2.unsqueeze(-1) @ v2.unsqueeze(-2)
        assert torch.allclose(state2, expected, atol=1e-3)


class TestStateCache:

    def test_set_and_get(self):
        cache = StateCache()
        state = torch.ones(4, 4)
        z = torch.ones(4)
        cache.set(0, 0, state, z)
        got_state, got_z = cache.get(0, 0)
        assert torch.allclose(got_state, state)
        assert torch.allclose(got_z, z)

    def test_get_missing_returns_none(self):
        cache = StateCache()
        state, z = cache.get(0, 0)
        assert state is None
        assert z is None

    def test_reset_clears_state(self):
        cache = StateCache()
        cache.set(0, 0, torch.ones(4, 4), torch.ones(4))
        cache.reset()
        state, z = cache.get(0, 0)
        assert state is None
        assert z is None

    def test_memory_bytes(self):
        cache = StateCache()
        cache.set(0, 0, torch.ones(4, 4, dtype=torch.float32), torch.ones(4, dtype=torch.float32))
        # 16 floats (state) + 4 floats (z) = 20 × 4 bytes = 80
        assert cache.memory_bytes == 80
