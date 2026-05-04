"""Shared fixtures for the test suite.

All tests run on CPU with small tensors — no GPU or model downloads required.
"""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def small_tensor():
    """Random tensor shaped [batch=2, heads=4, seq=8, dim=32]."""
    return torch.randn(2, 4, 8, 32)


@pytest.fixture
def fake_model_config():
    """Minimal model config matching Llama-style architecture."""
    return SimpleNamespace(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=256,
    )


@pytest.fixture
def fake_device_config():
    """CPU device config for testing."""
    return SimpleNamespace(
        device="cpu",
        dtype=torch.float32,
    )
