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

    # --- MoE: per-expert diagnosis. A single dead expert can hide behind a
    # healthy-looking layer average, so we flag at expert granularity. ---
    for li in ref.moe_layers:
        if li not in target.experts:
            continue
        lm = layers[li]
        ref_experts, q_experts = ref.experts[li], target.experts[li]
        lm.expert_cosines = [layer_cosine(a, b) for a, b in zip(ref_experts, q_experts)]
        lm.culprit_experts = [e for e, c in enumerate(lm.expert_cosines) if c < CULPRIT_COSINE]
        if li in ref.routers and li in target.routers:
            lm.router_kl = output_kl(ref.routers[li], target.routers[li])
        if lm.culprit_experts and li not in culprits:
            culprits.append(li)  # dead expert => the layer is a culprit even if its mean looks fine

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
