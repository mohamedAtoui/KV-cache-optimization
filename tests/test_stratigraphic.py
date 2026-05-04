"""Tests for the Stratigraphic KV-Cache strategy.

Validates zone constants, zone assignment, monotonic downgrade
constraint, anchor detection, and inverse layer budget.
"""

import torch
import pytest

from streaming_attention.stratigraphic import (
    ZONE_FP16, ZONE_INT8, ZONE_INT4, ZONE_EVICT,
    ZONE_TQ3, ZONE_TQ4,
    StratigraphicConfig, HeadZoneAssigner, AnchorDetector,
)


class TestZoneConstants:

    def test_zone_ordering(self):
        """Zones should be ordered: FP16 < INT8 < INT4 < EVICT."""
        assert ZONE_FP16 < ZONE_INT8 < ZONE_INT4 < ZONE_EVICT

    def test_zone_values(self):
        assert ZONE_FP16 == 0
        assert ZONE_INT8 == 1
        assert ZONE_INT4 == 2
        assert ZONE_EVICT == 3
        assert ZONE_TQ4 == 4
        assert ZONE_TQ3 == 5


class TestHeadZoneAssigner:

    @pytest.fixture
    def assigner(self):
        return HeadZoneAssigner(StratigraphicConfig())

    def test_output_shape(self, assigner):
        """Zone assignment should return [num_heads, seq_len]."""
        scores = torch.randn(8, 64)  # [heads, seq_len]
        anchors = torch.zeros(64, dtype=torch.bool)
        zones = assigner.assign_zones(
            layer_idx=0, per_head_scores=scores,
            anchors=anchors, num_layers=16,
        )
        assert zones.shape == (8, 64)

    def test_zones_are_valid(self, assigner):
        """All zone values should be valid zone constants."""
        scores = torch.randn(8, 64)
        anchors = torch.zeros(64, dtype=torch.bool)
        zones = assigner.assign_zones(0, scores, anchors, 16)
        valid = {ZONE_FP16, ZONE_INT8, ZONE_INT4, ZONE_EVICT}
        for z in zones.unique().tolist():
            assert z in valid

    def test_monotonic_downgrade(self, assigner):
        """Tokens should only move to deeper compression, never back."""
        scores1 = torch.randn(4, 32)
        scores2 = torch.randn(4, 32)
        anchors = torch.zeros(32, dtype=torch.bool)

        zones1 = assigner.assign_zones(0, scores1, anchors, 8)
        zones2 = assigner.assign_zones(0, scores2, anchors, 8)

        # Second assignment should be >= first (deeper or same)
        assert (zones2 >= zones1).all()

    def test_anchors_pinned_at_fp16(self, assigner):
        """Anchor positions should always be FP16."""
        scores = torch.randn(4, 32)
        anchors = torch.zeros(32, dtype=torch.bool)
        anchors[0] = True
        anchors[5] = True
        anchors[10] = True

        zones = assigner.assign_zones(0, scores, anchors, 8)

        for h in range(4):
            assert zones[h, 0] == ZONE_FP16
            assert zones[h, 5] == ZONE_FP16
            assert zones[h, 10] == ZONE_FP16

    def test_inverse_layer_budget(self):
        """Later layers should get more FP16 budget than early layers."""
        config = StratigraphicConfig(zone_surface=0.20, lambda_=0.6)
        assigner = HeadZoneAssigner(config)

        scores = torch.randn(4, 100)
        anchors = torch.zeros(100, dtype=torch.bool)

        zones_early = assigner.assign_zones(0, scores, anchors, 16)
        assigner.clear()
        zones_late = assigner.assign_zones(15, scores, anchors, 16)

        fp16_early = (zones_early == ZONE_FP16).float().mean().item()
        fp16_late = (zones_late == ZONE_FP16).float().mean().item()

        assert fp16_late > fp16_early  # late layers preserve more

    def test_clear_resets_history(self, assigner):
        """Clear should allow fresh zone assignment."""
        scores = torch.randn(4, 32)
        anchors = torch.zeros(32, dtype=torch.bool)
        assigner.assign_zones(0, scores, anchors, 8)
        assigner.clear()
        assert len(assigner._zone_history) == 0


class TestAnchorDetector:

    def test_anchors_are_subset(self):
        """Number of anchors should not exceed budget."""
        config = StratigraphicConfig(anchor_budget=0.05)
        detector = AnchorDetector(config)
        attn = torch.randn(8, 100)  # [heads, seq_len]
        anchors = detector.detect_anchors(attn)
        assert anchors.shape == (100,)
        assert anchors.sum() <= int(0.05 * 100) + 1  # +1 for rounding

    def test_high_attention_detected(self):
        """Tokens with very high attention should be anchored."""
        config = StratigraphicConfig(anchor_budget=0.10)
        detector = AnchorDetector(config)
        attn = torch.zeros(4, 50)
        attn[:, 0] = 100.0  # token 0 has extremely high attention
        attn[:, 25] = 100.0  # token 25 too
        anchors = detector.detect_anchors(attn)
        assert anchors[0] == True
        assert anchors[25] == True

    def test_empty_sequence(self):
        """Should handle empty sequences gracefully."""
        config = StratigraphicConfig()
        detector = AnchorDetector(config)
        attn = torch.randn(4, 0)
        anchors = detector.detect_anchors(attn)
        assert anchors.shape == (0,)
