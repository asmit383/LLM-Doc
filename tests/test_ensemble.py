"""Metric-ensemble tests (Phase 6 robustness).

Culprit detection votes across independent metrics (cosine=direction, MSE=magnitude).
The key win: MSE catches damage cosine is blind to (magnitude blowup with intact
direction), and agreement across metrics yields a confidence signal.
"""

from __future__ import annotations

import torch

from quant_doctor.dumps import Dump
from quant_doctor.engine import diagnose_pair


def _dump(layers, model="test"):
    manifest = {"model": model, "n_layers": len(layers), "has_logits": False}
    return Dump(manifest=manifest, layers=layers, logits=None)


def test_mse_catches_magnitude_blowup_cosine_misses():
    """A layer scaled up 12x: direction identical (cosine≈1) but magnitude explodes.
    Cosine alone would pass it; the MSE voter must catch it."""
    torch.manual_seed(0)
    ref_layers = [torch.randn(16, 64) for _ in range(10)]
    quant_layers = [t.clone() for t in ref_layers]
    quant_layers[4] = ref_layers[4] * 12.0  # same direction, 12x magnitude

    diag = diagnose_pair(_dump(ref_layers), _dump(quant_layers))

    lm = diag.layers[4]
    assert lm.cosine > 0.99, "cosine should still look fine (its blind spot)"
    assert lm.is_culprit, "MSE voter must catch the magnitude blowup"
    assert 4 in diag.culprit_indices


def test_healthy_flags_nothing_and_passes():
    torch.manual_seed(1)
    ref_layers = [torch.randn(16, 64) for _ in range(10)]
    quant_layers = [t + 0.001 * torch.randn(16, 64) for t in ref_layers]
    diag = diagnose_pair(_dump(ref_layers), _dump(quant_layers))
    assert diag.culprit_indices == []
    assert diag.verdict.value == "PASS"


def test_corroborated_damage_is_high_confidence():
    """A scrambled layer trips both cosine (direction) and MSE (magnitude)."""
    torch.manual_seed(2)
    ref_layers = [torch.randn(16, 64) for _ in range(10)]
    quant_layers = [t.clone() for t in ref_layers]
    quant_layers[3] = torch.randn(16, 64)  # fully scrambled

    diag = diagnose_pair(_dump(ref_layers), _dump(quant_layers))
    lm = diag.layers[3]
    assert lm.is_culprit
    assert lm.metric_votes >= 1
    assert lm.confidence == "high"
