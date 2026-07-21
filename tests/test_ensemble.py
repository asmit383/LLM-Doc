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


def test_near_lossless_with_spread_passes():
    """The 8-bit pathology: every layer is near-lossless, but a couple carry a bit
    more (still negligible) error than the rest. The relative MSE voter would flag
    those as outliers and read DEGRADED; the absolute healthy-floor must suppress
    them so a clean quant earns a trustworthy PASS.

    All noise here stays well under MSE_HEALTHY_FLOOR (0.002 ≈ cosine 0.999), yet
    the two heavier layers are ~9x the others — pure relative outliers.
    """
    torch.manual_seed(7)
    ref_layers = [torch.randn(16, 64) for _ in range(12)]
    quant_layers = [t + 0.005 * torch.randn(16, 64) for t in ref_layers]  # MSE ~2.5e-5
    for i in (3, 8):
        quant_layers[i] = ref_layers[i] + 0.03 * torch.randn(16, 64)  # MSE ~9e-4, still < floor

    diag = diagnose_pair(_dump(ref_layers), _dump(quant_layers))

    assert all(lm.mse < 0.002 for lm in diag.layers), "sanity: every layer is near-lossless"
    assert diag.culprit_indices == [], "negligible-magnitude outliers must not be flagged"
    assert diag.verdict.value == "PASS"


def test_above_floor_outlier_still_flagged():
    """Guard against over-suppression: a layer with real, above-floor error must
    still be caught. Keeps the healthy-floor from silencing genuine degradation."""
    torch.manual_seed(8)
    ref_layers = [torch.randn(16, 64) for _ in range(12)]
    quant_layers = [t + 0.005 * torch.randn(16, 64) for t in ref_layers]
    quant_layers[5] = ref_layers[5] + 0.4 * torch.randn(16, 64)  # MSE ~0.16, well above floor

    diag = diagnose_pair(_dump(ref_layers), _dump(quant_layers))

    assert diag.layers[5].mse > 0.002
    assert 5 in diag.culprit_indices
    assert diag.verdict.value in ("DEGRADED", "BROKEN")


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
