"""Diagnostic engine — turns a pair of dumps into a Diagnosis.

Format-agnostic: it consumes paired activation tensors. The same function serves
the dumps path (V4/QTIP) and, later, the live-capture path (HF models).
"""

from __future__ import annotations

from .classifier import classify
from .diagnosis import (
    ABSOLUTE_FLOOR,
    Diagnosis,
    LayerMetrics,
    decide_verdict,
    find_culprit_indices,
    robust_high_outliers,
)
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

    # --- Ensemble culprit detection ---
    # Two independent signals vote, each via adaptive median/MAD outlier detection:
    #   cosine (direction) — find_culprit_indices, with absolute floor + ceiling
    #   MSE    (magnitude) — robust_high_outliers; catches cosine's blind spot
    #                        (a layer whose scale blows up but direction is intact)
    cos_flags = set(find_culprit_indices([lm.cosine for lm in layers]))
    mse_flags = set(robust_high_outliers([lm.mse for lm in layers]))

    culprits: list[int] = []
    for lm in layers:
        votes = (lm.index in cos_flags) + (lm.index in mse_flags)
        lm.metric_votes = int(votes)
        floored = lm.cosine < ABSOLUTE_FLOOR
        if floored or votes >= 1:
            culprits.append(lm.index)
            # High confidence when it's gross damage or corroborated by 2 signals;
            # a single-signal flag is surfaced but marked low-confidence.
            lm.confidence = "high" if (floored or votes >= 2) else "low"

    # --- MoE: per-expert diagnosis. A single dead expert can hide behind a
    # healthy-looking layer average, so we flag at expert granularity. ---
    for li in ref.moe_layers:
        if li not in target.experts:
            continue
        lm = layers[li]
        ref_experts, q_experts = ref.experts[li], target.experts[li]
        lm.expert_cosines = [layer_cosine(a, b) for a, b in zip(ref_experts, q_experts)]
        lm.culprit_experts = [e for e, c in enumerate(lm.expert_cosines) if c < ABSOLUTE_FLOOR]
        if li in ref.routers and li in target.routers:
            lm.router_kl = output_kl(ref.routers[li], target.routers[li])
        if lm.culprit_experts:
            # A dead expert is a strong signal even if the block average looks fine.
            lm.confidence = "high"
            if li not in culprits:
                culprits.append(li)

    for lm in layers:
        lm.is_culprit = lm.index in culprits

    # Error-subspace analysis is expensive (SVD) — run it only for culprit layers.
    for lm in layers:
        if lm.is_culprit:
            lm.subspace_top1 = residual_top1_concentration(ref.layers[lm.index], target.layers[lm.index])

    cosines = [lm.cosine for lm in layers]
    mean_cos = sum(cosines) / len(cosines)
    # A dead expert can leave the block-output cosine high; fold expert damage
    # into min_cosine so the verdict reflects it.
    expert_min = min(
        (min(lm.expert_cosines) for lm in layers if lm.expert_cosines), default=1.0
    )
    min_cos = min(min(cosines), expert_min)

    kl = None
    if ref.logits is not None and target.logits is not None:
        kl = output_kl(ref.logits, target.logits)

    diag = Diagnosis(
        verdict=decide_verdict(min_cos, kl, n_culprits=len(culprits)),
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
