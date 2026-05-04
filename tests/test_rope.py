"""Tests for Rotary Position Embeddings (RoPE)."""

import torch
import pytest

from AttentionHeads.mha.rope import precompute_freqs_cis, apply_rotary_emb


class TestRoPE:

    @pytest.fixture
    def rope_data(self):
        """Pre-compute RoPE cos/sin for dim=32, max_seq=64."""
        cos, sin = precompute_freqs_cis(dim=32, max_seq_len=64)
        return cos, sin

    def test_freqs_shape(self, rope_data):
        cos, sin = rope_data
        # cos and sin should be [max_seq_len, dim/2] or [max_seq_len, dim]
        assert cos.shape[0] == 64
        assert sin.shape[0] == 64

    def test_norm_preservation(self, rope_data):
        """RoPE should approximately preserve vector norms."""
        cos, sin = rope_data
        x = torch.randn(2, 8, 16, 32)
        x_rope = apply_rotary_emb(x, cos[:16], sin[:16])
        norms_before = x.norm(dim=-1)
        norms_after = x_rope.norm(dim=-1)
        assert torch.allclose(norms_before, norms_after, atol=1e-4)

    def test_different_positions_differ(self, rope_data):
        """Same vector at different positions should produce different outputs."""
        cos, sin = rope_data
        x = torch.ones(1, 1, 2, 32)
        x_rope = apply_rotary_emb(x, cos[:2], sin[:2])
        assert not torch.allclose(x_rope[0, 0, 0], x_rope[0, 0, 1], atol=1e-3)

    def test_deterministic(self, rope_data):
        """Same input should always produce same output."""
        cos, sin = rope_data
        x = torch.randn(1, 1, 4, 32)
        out1 = apply_rotary_emb(x, cos[:4], sin[:4])
        out2 = apply_rotary_emb(x, cos[:4], sin[:4])
        assert torch.allclose(out1, out2)

    def test_output_shape(self, rope_data):
        cos, sin = rope_data
        x = torch.randn(2, 8, 16, 32)
        x_rope = apply_rotary_emb(x, cos[:16], sin[:16])
        assert x_rope.shape == x.shape
