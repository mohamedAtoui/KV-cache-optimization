"""Tests for attention mechanism output shapes and properties."""

import torch
import pytest

from AttentionHeads.mha.attention import MultiHeadedAttention, attention
from AttentionHeads.mqa.attention import MultiQueryAttention
from AttentionHeads.gqa.attention import GroupedQueryAttention


class TestScaledDotProductAttention:

    def test_output_shape(self):
        q = torch.randn(2, 8, 16, 32)
        k = torch.randn(2, 8, 16, 32)
        v = torch.randn(2, 8, 16, 32)
        output, weights = attention(q, k, v)
        assert output.shape == (2, 8, 16, 32)
        assert weights.shape == (2, 8, 16, 16)

    def test_weights_sum_to_one(self):
        q = torch.randn(2, 8, 16, 32)
        k = torch.randn(2, 8, 16, 32)
        v = torch.randn(2, 8, 16, 32)
        _, weights = attention(q, k, v)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_causal_mask(self):
        seq_len = 8
        q = torch.randn(1, 1, seq_len, 32)
        k = torch.randn(1, 1, seq_len, 32)
        v = torch.randn(1, 1, seq_len, 32)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        _, weights = attention(q, k, v, mask=mask)
        upper = weights[0, 0].triu(diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


class TestMHA:

    def test_output_shape(self):
        mha = MultiHeadedAttention(h=8, d_model=256)
        x = torch.randn(2, 16, 256)
        mask = torch.ones(2, 1, 1, 16).bool()
        result = mha(x, x, x, mask)
        output = result[0] if isinstance(result, tuple) else result
        assert output.dim() == 3
        assert output.shape[0] == 2   # batch
        assert output.shape[2] == 256  # d_model preserved


class TestMQA:

    def test_output_shape(self):
        mqa = MultiQueryAttention(h=8, d_model=256)
        x = torch.randn(2, 16, 256)
        mask = torch.ones(2, 1, 1, 16).bool()
        result = mqa(x, x, x, mask)
        output = result[0] if isinstance(result, tuple) else result
        assert output.dim() == 3
        assert output.shape[0] == 2
        assert output.shape[2] == 256

    def test_kv_cache_smaller_than_mha(self):
        """MQA should use fewer KV parameters than MHA."""
        mqa = MultiQueryAttention(h=8, d_model=256)
        mha = MultiHeadedAttention(h=8, d_model=256)
        mqa_params = sum(p.numel() for p in mqa.parameters())
        mha_params = sum(p.numel() for p in mha.parameters())
        assert mqa_params < mha_params


class TestGQA:

    def test_output_shape(self):
        gqa = GroupedQueryAttention(h=8, d_model=256, num_kv_heads=4)
        x = torch.randn(2, 16, 256)
        mask = torch.ones(2, 1, 1, 16).bool()
        result = gqa(x, x, x, mask)
        output = result[0] if isinstance(result, tuple) else result
        assert output.dim() == 3
        assert output.shape[0] == 2
        assert output.shape[2] == 256

    def test_kv_heads_count(self):
        gqa = GroupedQueryAttention(h=8, d_model=256, num_kv_heads=4)
        assert gqa.num_kv_heads == 4
        assert gqa.heads_per_group == 2

    def test_kv_params_between_mha_and_mqa(self):
        """GQA should have parameter count between MHA and MQA."""
        mha = MultiHeadedAttention(h=8, d_model=256)
        mqa = MultiQueryAttention(h=8, d_model=256)
        gqa = GroupedQueryAttention(h=8, d_model=256, num_kv_heads=4)
        mha_p = sum(p.numel() for p in mha.parameters())
        mqa_p = sum(p.numel() for p in mqa.parameters())
        gqa_p = sum(p.numel() for p in gqa.parameters())
        assert mqa_p < gqa_p < mha_p
