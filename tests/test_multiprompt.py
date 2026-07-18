"""Multi-prompt aggregation tests (Phase 6 robustness).

The point: real damage shows up consistently across prompts; a fluke on one prompt
does not. diagnose_multi must flag the consistent layer and reject the fluke.
"""

from __future__ import annotations

import torch

from quant_doctor.dumps import Dump
from quant_doctor.engine import diagnose_multi


def _dump(layers):
    return Dump(manifest={"model": "t", "n_layers": len(layers), "has_logits": False},
               layers=layers, logits=None)


def _make_prompt(n_layers=10, seed=0, break_layers=()):
    """One prompt's ref/target pair; `break_layers` are scrambled in the target."""
    g = torch.Generator().manual_seed(seed)
    ref = [torch.randn(16, 64, generator=g) for _ in range(n_layers)]
    tgt = [t.clone() for t in ref]
    for li in break_layers:
        tgt[li] = torch.randn(16, 64, generator=g)  # scrambled
    return _dump(ref), _dump(tgt)


def test_consistent_damage_is_flagged():
    # Layer 3 broken in ALL 5 prompts.
    pairs = [_make_prompt(seed=k, break_layers=(3,)) for k in range(5)]
    refs, tgts = zip(*pairs)
    diag = diagnose_multi(list(refs), list(tgts))
    assert 3 in diag.culprit_indices
    assert diag.layers[3].prompt_agreement == 1.0
    assert diag.layers[3].confidence == "high"


def test_single_prompt_fluke_is_rejected():
    # Layer 7 broken in only 1 of 5 prompts — must NOT be flagged.
    pairs = [_make_prompt(seed=k, break_layers=(7,) if k == 0 else ()) for k in range(5)]
    refs, tgts = zip(*pairs)
    diag = diagnose_multi(list(refs), list(tgts))
    assert 7 not in diag.culprit_indices
    assert diag.layers[7].prompt_agreement == 0.2  # flagged 1/5


def test_error_bars_reflect_variability():
    # A layer broken in some prompts but not others has high cosine std.
    pairs = [_make_prompt(seed=k, break_layers=(5,) if k % 2 == 0 else ()) for k in range(4)]
    refs, tgts = zip(*pairs)
    diag = diagnose_multi(list(refs), list(tgts))
    assert diag.layers[5].cosine_std > 0.1   # inconsistent -> wide error bar
    assert diag.layers[0].cosine_std < 0.01  # clean layer -> tight


def test_majority_damage_is_flagged():
    # Broken in 3 of 5 (majority) -> flagged, medium confidence (not all agree).
    pairs = [_make_prompt(seed=k, break_layers=(2,) if k < 3 else ()) for k in range(5)]
    refs, tgts = zip(*pairs)
    diag = diagnose_multi(list(refs), list(tgts))
    assert 2 in diag.culprit_indices
    assert diag.layers[2].confidence == "medium"
