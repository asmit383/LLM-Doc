"""Ground-truth validation: the classifier must recover the injected failure mode.

Because the synthetic cases carry known damage, these assertions turn "5/5 correct"
into a repeatable claim rather than a one-off observation.
"""

from __future__ import annotations

import pytest

from quant_doctor.dumps import load_dump
from quant_doctor.engine import diagnose_pair
from quant_doctor.synthetic import CASES, COLLAPSE_AT, EXPECTED, make_case


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    for i, name in enumerate(CASES):
        make_case(name, root, seed=i)
    return root


@pytest.mark.parametrize("name", CASES)
def test_verdict_and_mode(cases, name):
    ref = load_dump(cases / name / "ref")
    target = load_dump(cases / name / "target")
    diag = diagnose_pair(ref, target)

    expected_verdict, expected_mode = EXPECTED[name]
    assert diag.verdict.value == expected_verdict, (
        f"{name}: verdict {diag.verdict.value} != {expected_verdict}"
    )
    assert diag.failure_mode == expected_mode, (
        f"{name}: mode {diag.failure_mode!r} != {expected_mode!r}"
    )


def test_collapse_is_localized(cases):
    """Computation collapse must be attributed to the injected layer."""
    ref = load_dump(cases / "computation_collapse" / "ref")
    target = load_dump(cases / "computation_collapse" / "target")
    diag = diagnose_pair(ref, target)

    assert diag.culprit_indices == [COLLAPSE_AT], (
        f"expected culprit=[{COLLAPSE_AT}], got {diag.culprit_indices}"
    )


def test_format_bug_flags_first_layer(cases):
    """A decode/format bug damages the model from layer 0 (no clean prefix)."""
    ref = load_dump(cases / "format_bug" / "ref")
    target = load_dump(cases / "format_bug" / "target")
    diag = diagnose_pair(ref, target)

    assert 0 in diag.culprit_indices
    assert len(diag.culprit_indices) == len(diag.layers)
