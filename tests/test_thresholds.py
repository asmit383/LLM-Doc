"""Adaptive culprit-detection tests (Phase 6 robustness).

The adaptive rule must: catch relative outliers a fixed 0.90 floor misses, still
catch gross damage, and NOT fire on uniformly-healthy or uniformly-degraded models.
"""

from __future__ import annotations

from quant_doctor.diagnosis import find_culprit_indices


def test_absolute_floor_catches_gross_damage():
    # A clearly-dead layer is flagged regardless of the distribution.
    cosines = [0.99, 0.99, 0.20, 0.99, 0.99]
    assert find_culprit_indices(cosines) == [2]


def test_adaptive_catches_relative_outlier_a_fixed_floor_would_miss():
    # Healthy baseline 0.999, one layer at 0.95 — above the 0.90 floor, so a
    # fixed threshold misses it, but it's a clear anomaly vs this model.
    cosines = [0.999] * 10
    cosines[5] = 0.95
    assert 5 in find_culprit_indices(cosines)


def test_no_false_positive_on_uniformly_healthy_model():
    # Near-perfect with tiny natural variation — must flag nothing.
    cosines = [0.999, 0.998, 0.9985, 0.999, 0.997, 0.9982]
    assert find_culprit_indices(cosines) == []


def test_no_false_positive_on_uniformly_degraded_model():
    # Whole model sits ~0.94 (diffuse degradation) — that's a DEGRADED verdict,
    # not a set of per-layer culprits. Flag nothing.
    cosines = [0.94, 0.938, 0.942, 0.939, 0.941, 0.94]
    assert find_culprit_indices(cosines) == []


def test_gradual_decline_has_no_single_culprit():
    # Signal-degradation shape: smooth monotonic decline, no sharp outlier.
    cosines = [0.999 - 0.007 * i for i in range(12)]  # 0.999 -> ~0.92
    assert find_culprit_indices(cosines) == []


def test_localized_cliff_flags_only_the_cliff():
    cosines = [0.99] * 6 + [0.05] + [0.99] * 5
    assert find_culprit_indices(cosines) == [6]


def test_empty_is_safe():
    assert find_culprit_indices([]) == []
