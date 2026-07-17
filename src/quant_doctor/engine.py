"""Diagnostic engine — turns a pair of dumps into a Diagnosis.

Format-agnostic: it consumes paired activation tensors. The same function serves
the dumps path (V4/QTIP) and, later, the live-capture path (HF models).
"""

from __future__ import annotations

from .classifier import classify
from .diagnosis import CULPRIT_COSINE, Diagnosis, LayerMetrics, decide_verdict
from .dumps import Dump, check_pair
from .metrics.interpretability import residual_top1_concentration
from .metrics.statistical import layer_cosine, layer_mse, output_kl


def diagnose_pair(ref: Dump, target: Dump) -> Diagnosis:
    """Compare a reference dump against a quantized target dump."""
    check_pair(ref, target)

    layers: list[LayerMetrics] = []
    for i, (a, b) in enumerate(zip(ref.layers, target.layers)):
        cos = layer_cosine(a, b)
        mse = layer_mse(a, b)
        layers.append(LayerMetrics(index=i, name=f"layer_{i:02d}", cosine=cos, mse=mse))

    culprits = [lm.index for lm in layers if lm.cosine < CULPRIT_COSINE]
    for lm in layers:
        lm.is_culprit = lm.index in culprits

    # Error-subspace analysis is expensive (SVD) — run it only for culprit layers.
    for lm in layers:
        if lm.is_culprit:
            lm.subspace_top1 = residual_top1_concentration(ref.layers[lm.index], target.layers[lm.index])

    cosines = [lm.cosine for lm in layers]
    mean_cos = sum(cosines) / len(cosines)
    min_cos = min(cosines)

    kl = None
    if ref.logits is not None and target.logits is not None:
        kl = output_kl(ref.logits, target.logits)

    diag = Diagnosis(
        verdict=decide_verdict(min_cos, kl),
        mean_cosine=mean_cos,
        min_cosine=min_cos,
        output_kl=kl,
        layers=layers,
        culprit_indices=culprits,
        model=target.model,
    )

    cls = classify(diag)
    diag.failure_mode = cls.mode.value
    diag.signature = cls.signature
    diag.repair = cls.repair
    return diag
