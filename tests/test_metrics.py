"""Unit tests for the statistical + interpretability metrics."""

from __future__ import annotations

import torch

from quant_doctor.metrics.interpretability import residual_top1_concentration
from quant_doctor.metrics.propagation import cliff_gap, clean_prefix_len, depth_trend
from quant_doctor.metrics.statistical import layer_cosine, layer_mse, output_kl


def test_cosine_identical_is_one():
    x = torch.randn(8, 32)
    assert layer_cosine(x, x) == pytest_approx(1.0)


def test_cosine_opposite_is_minus_one():
    x = torch.randn(8, 32)
    assert layer_cosine(x, -x) == pytest_approx(-1.0)


def test_mse_identical_is_zero():
    x = torch.randn(8, 32)
    assert layer_mse(x, x) == pytest_approx(0.0)


def test_kl_identical_is_zero():
    x = torch.randn(8, 100)
    assert output_kl(x, x) == pytest_approx(0.0, abs=1e-5)


def test_depth_trend_declining_is_negative():
    cosines = [1.0 - 0.05 * i for i in range(10)]
    assert depth_trend(cosines) < -0.9


def test_depth_trend_flat_is_zero():
    assert depth_trend([0.99] * 10) == pytest_approx(0.0)


def test_cliff_gap_detects_single_dip():
    cosines = [0.99] * 5 + [0.01] + [0.99] * 5
    assert cliff_gap(cosines) > 0.9


def test_clean_prefix_counts_leading_healthy():
    cosines = [0.99, 0.99, 0.01, 0.99]
    assert clean_prefix_len(cosines, threshold=0.9) == 2


def test_subspace_concentration_bounds():
    ref = torch.randn(16, 64)
    quant = ref + 0.01 * torch.randn(16, 64)
    c = residual_top1_concentration(ref, quant)
    assert 0.0 <= c <= 1.0


def test_subspace_identical_is_zero():
    x = torch.randn(16, 64)
    assert residual_top1_concentration(x, x) == 0.0


# --- tiny local approx helper so we don't depend on pytest.approx import style ---
def pytest_approx(value, abs=1e-4):
    import pytest
    return pytest.approx(value, abs=abs)
