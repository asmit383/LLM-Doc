"""Interpretability metrics on paired activations — Phase 2.

These go beyond "how much divergence" to "what kind". They are computable from
the per-layer hidden states we already capture (no attention weights needed), so
they work on both the live and dumps paths.
"""

from __future__ import annotations

import torch


def residual_top1_concentration(ref: torch.Tensor, quant: torch.Tensor) -> float:
    """Fraction of the error residual's energy in its single top singular direction.

    residual = ref - quant, shape [seq, hidden]. We SVD it and return
    sigma_1^2 / sum(sigma_i^2).

    High (→1)  : the quantization error is *structured* — it pushes activations
                 along one dominant direction. Characteristic of a specific
                 component malfunctioning (computation-collapse flavor).
    Low  (→0)  : the error is *diffuse*, spread across many directions — random
                 rounding noise (signal-degradation flavor).
    """
    r = ref.float() - quant.float()
    if r.numel() == 0 or torch.allclose(r, torch.zeros_like(r)):
        return 0.0
    # svdvals returns descending singular values, length min(seq, hidden).
    sv = torch.linalg.svdvals(r)
    energy = (sv**2).sum() + 1e-12
    return (sv[0] ** 2 / energy).item()
