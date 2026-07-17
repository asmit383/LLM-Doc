"""Failure-mode classifier — Phase 2.

Rule-based decision tree over the per-layer damage profile. The rules follow the
signatures named in the 2026 literature (ACL 2026: Signal Degradation vs
Computation Collapse) plus a Format-Bug class for decode errors that masquerade
as quantization damage.

Rule-based (not ML) on purpose: the decision boundaries are clean, the logic is
auditable for a report, and it needs no training corpus. Thresholds are initial
and calibrated against the synthetic ground-truth cases + real runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import median

from .diagnosis import CULPRIT_COSINE, Diagnosis, Verdict
from .metrics.propagation import cliff_gap, clean_prefix_len, depth_trend


class FailureMode(str, Enum):
    HEALTHY = "Healthy"
    SIGNAL_DEGRADATION = "Signal Degradation"
    COMPUTATION_COLLAPSE = "Computation Collapse"
    FORMAT_BUG = "Format Bug"
    GENERIC = "Generic Degradation"


_REPAIR = {
    FailureMode.HEALTHY:
        "No action needed — quantization preserved the model.",
    FailureMode.SIGNAL_DEGRADATION:
        "Training-free repair is viable. Keep the worst layers and lm_head at higher "
        "precision — the error is diffuse rounding noise, not lost information, so "
        "mixed-precision recovery should substantially help.",
    FailureMode.COMPUTATION_COLLAPSE:
        "Training-free repair will likely NOT fully recover this. One or more critical "
        "layers destroy information the downstream stack depends on. Keep the collapsed "
        "layer(s) at higher precision AND consider a LoRA / fine-tune; then re-quantize.",
    FailureMode.FORMAT_BUG:
        "This looks like a decode/format bug, not precision loss — every layer including "
        "the first is near-orthogonal to the reference. Check dequantization (e.g. a "
        "format misinterpretation like MXFP4-as-INT4) before touching bit-widths.",
    FailureMode.GENERIC:
        "Degradation is present but the pattern is ambiguous — inspect the per-layer heatmap.",
}

# Discrimination thresholds (initial; tuned against synthetic + real cases).
FORMAT_MEDIAN_COSINE = 0.35   # below this median, ~everything is destroyed
FORMAT_FRAC_CULPRIT = 0.60    # and most layers are culprits
COLLAPSE_GAP = 0.25           # median - min this large = a cliff
COLLAPSE_MIN_COSINE = 0.70    # and the worst layer is genuinely broken
DEGRADE_TREND = -0.30         # cosine declining with depth


@dataclass
class Classification:
    mode: FailureMode
    signature: list[str] = field(default_factory=list)
    repair: str = ""


def classify(diag: Diagnosis) -> Classification:
    cosines = [lm.cosine for lm in diag.layers]
    n = len(cosines)
    if n == 0:
        return Classification(FailureMode.GENERIC, ["no layers"], _REPAIR[FailureMode.GENERIC])

    mn = min(cosines)
    med = median(cosines)
    gap = cliff_gap(cosines)
    trend = depth_trend(cosines)
    culprits = [lm for lm in diag.layers if lm.is_culprit]
    frac = len(culprits) / n
    prefix = clean_prefix_len(cosines, CULPRIT_COSINE)
    worst = min(diag.layers, key=lambda lm: lm.cosine)

    # --- Healthy -----------------------------------------------------------
    if diag.verdict is Verdict.PASS:
        return _mk(FailureMode.HEALTHY, [f"min cosine {mn:.3f} — within healthy range"])

    # --- Format bug: uniform catastrophic, damage from the very first layer -
    if med < FORMAT_MEDIAN_COSINE and frac > FORMAT_FRAC_CULPRIT and cosines[0] < 0.5:
        return _mk(FailureMode.FORMAT_BUG, [
            f"median cosine {med:.3f} — nearly every layer destroyed",
            f"layer 0 cosine {cosines[0]:.3f} — damage present from the first layer",
            f"{len(culprits)}/{n} layers flagged, uniformly low (no clean prefix)",
        ])

    # --- Computation collapse: sharp localized cliff after a clean prefix ---
    if gap > COLLAPSE_GAP and mn < COLLAPSE_MIN_COSINE:
        sig = [
            f"cliff at {worst.name}: cosine {mn:.3f} vs median {med:.3f} (gap {gap:.2f})",
            (f"{prefix} clean layer(s) before the collapse"
             if prefix > 0 else "damage begins at the input"),
            f"localized: only {len(culprits)}/{n} layers flagged",
        ]
        if worst.subspace_top1 is not None:
            sig.append(f"error residual top-1 concentration {worst.subspace_top1:.2f} "
                       f"({'structured' if worst.subspace_top1 > 0.3 else 'diffuse'})")
        return _mk(FailureMode.COMPUTATION_COLLAPSE, sig)

    # --- Signal degradation: diffuse, worsening with depth, no catastrophe --
    if trend < DEGRADE_TREND or (mn >= COLLAPSE_MIN_COSINE and gap <= COLLAPSE_GAP):
        sig = [f"diffuse: worst layer only {mn:.3f} — no catastrophic cliff"]
        if trend < DEGRADE_TREND:
            sig.append(f"cosine declines with depth (r={trend:.2f}) — error accumulates")
        else:
            sig.append("damage spread across layers, not localized")
        sig.append(f"{len(culprits)}/{n} layers flagged")
        return _mk(FailureMode.SIGNAL_DEGRADATION, sig)

    # --- Fallback ----------------------------------------------------------
    return _mk(FailureMode.GENERIC,
               [f"min {mn:.3f}, median {med:.3f}, gap {gap:.2f}, trend {trend:.2f}"])


def _mk(mode: FailureMode, signature: list[str]) -> Classification:
    return Classification(mode=mode, signature=signature, repair=_REPAIR[mode])
