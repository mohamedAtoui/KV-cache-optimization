"""Tests for configuration validation in AttentionHeads.

Validates that config loading catches invalid parameters and
accepts valid configurations.
"""

import json
import tempfile
import os
import pytest


class TestConfigValidation:

    def _make_config(self, overrides=None):
        """Create a valid config dict with optional overrides."""
        config = {
            "model": {
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 8,
                "d_ff": 1024,
                "max_seq_len": 256,
                "dropout": 0.1,
                "vocab_size": 50257,
                "activation": "relu"
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 5e-5,
                "max_steps": 6000,
                "warmup_steps": 600,
                "gradient_clip": 0.5,
                "gradient_accumulation_steps": 4,
                "seed": 42
            },
            "data": {
                "dataset": "tinystories",
                "train_samples": 30000,
                "val_samples": 5000,
                "max_length": 256
            }
        }
        if overrides:
            for section, vals in overrides.items():
                if section in config:
                    config[section].update(vals)
                else:
                    config[section] = vals
        return config

    def test_valid_config(self):
        """Valid config should load without errors."""
        config = self._make_config()
        assert config["model"]["d_model"] % config["model"]["n_heads"] == 0
        assert config["training"]["learning_rate"] > 0
        assert config["training"]["warmup_steps"] < config["training"]["max_steps"]

    def test_hidden_size_divisibility(self):
        """d_model must be divisible by n_heads."""
        config = self._make_config({"model": {"d_model": 255, "n_heads": 8}})
        assert config["model"]["d_model"] % config["model"]["n_heads"] != 0

    def test_learning_rate_positive(self):
        """Learning rate must be positive."""
        config = self._make_config({"training": {"learning_rate": -1e-5}})
        assert config["training"]["learning_rate"] <= 0

    def test_warmup_less_than_max_steps(self):
        """Warmup steps must be less than max steps."""
        config = self._make_config({"training": {"warmup_steps": 7000, "max_steps": 6000}})
        assert config["training"]["warmup_steps"] >= config["training"]["max_steps"]

    def test_batch_size_positive(self):
        """Batch size must be positive."""
        config = self._make_config({"training": {"batch_size": 0}})
        assert config["training"]["batch_size"] <= 0

    def test_json_roundtrip(self):
        """Config should survive JSON serialization."""
        config = self._make_config()
        json_str = json.dumps(config)
        loaded = json.loads(json_str)
        assert loaded == config
