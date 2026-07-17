"""Recipe generation tests."""

from __future__ import annotations

from quant_doctor.dumps import load_dump
from quant_doctor.engine import diagnose_pair
from quant_doctor.recipe import generate_recipe
from quant_doctor.synthetic import make_case


def _diagnose(tmp_path, name):
    make_case(name, tmp_path, seed=0)
    ref = load_dump(tmp_path / name / "ref")
    target = load_dump(tmp_path / name / "target")
    return diagnose_pair(ref, target), target.manifest


def test_healthy_has_no_recipe(tmp_path):
    diag, manifest = _diagnose(tmp_path, "healthy")
    assert generate_recipe(diag, manifest) is None


def test_format_bug_refuses_bit_recipe(tmp_path):
    diag, manifest = _diagnose(tmp_path, "format_bug")
    recipe = generate_recipe(diag, manifest)
    assert recipe is not None
    assert recipe.overrides == []          # no bit-width prescription
    assert "dequant" in recipe.notes.lower()


def test_collapse_protects_onset_and_head(tmp_path):
    diag, manifest = _diagnose(tmp_path, "computation_collapse")
    recipe = generate_recipe(diag, manifest, base_bits=2, high_bits=4)
    names = {ov.name for ov in recipe.overrides}
    assert "lm_head" in names
    assert "model.layers.2" in names       # the injected onset
    assert recipe.est_vram_delta_gb > 0
    assert "LOW" in recipe.confidence


def test_signal_degradation_protects_worst_and_head(tmp_path):
    diag, manifest = _diagnose(tmp_path, "signal_degradation")
    recipe = generate_recipe(diag, manifest, base_bits=2, high_bits=4)
    names = {ov.name for ov in recipe.overrides}
    assert "lm_head" in names
    assert len(recipe.overrides) >= 2
    assert "HIGH" in recipe.confidence
