"""MoE per-expert diagnosis tests — the V4 layer-2 expert-blowup scenario."""

from __future__ import annotations

import pytest

from quant_doctor.dumps import load_dump
from quant_doctor.engine import diagnose_pair
from quant_doctor.synthetic import (
    BLOWN_EXPERT,
    BLOWN_LAYER,
    EXPECTED,
    MOE_CASES,
    make_moe_case,
)


@pytest.fixture(scope="module")
def moe(tmp_path_factory):
    root = tmp_path_factory.mktemp("moe")
    for i, name in enumerate(MOE_CASES):
        make_moe_case(name, root, seed=100 + i)
    return root


@pytest.mark.parametrize("name", MOE_CASES)
def test_moe_verdict_and_mode(moe, name):
    ref = load_dump(moe / name / "ref")
    target = load_dump(moe / name / "target")
    diag = diagnose_pair(ref, target)
    exp_verdict, exp_mode = EXPECTED[name]
    assert diag.verdict.value == exp_verdict
    assert diag.failure_mode == exp_mode


def test_blowup_localizes_the_exact_expert(moe):
    ref = load_dump(moe / "moe_expert_blowup" / "ref")
    target = load_dump(moe / "moe_expert_blowup" / "target")
    diag = diagnose_pair(ref, target)

    lm = diag.layers[BLOWN_LAYER]
    assert lm.culprit_experts == [BLOWN_EXPERT], (
        f"expected dead expert [{BLOWN_EXPERT}] at layer {BLOWN_LAYER}, got {lm.culprit_experts}"
    )


def test_blowup_hides_behind_healthy_block_average(moe):
    """The whole point: the block-level cosine looks fine; only per-expert reveals it."""
    ref = load_dump(moe / "moe_expert_blowup" / "ref")
    target = load_dump(moe / "moe_expert_blowup" / "target")
    diag = diagnose_pair(ref, target)

    lm = diag.layers[BLOWN_LAYER]
    assert lm.cosine > 0.9, "block average should look healthy"
    assert min(lm.expert_cosines) < 0.5, "the dead expert should be obvious per-expert"
