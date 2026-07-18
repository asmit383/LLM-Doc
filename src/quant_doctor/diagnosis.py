"""Diagnosis data structures and the verdict rules.

These are format-agnostic: a Diagnosis is built from paired activations regardless
of whether they came from live capture or dumps.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    PASS = "PASS"
    DEGRADED = "DEGRADED"
    BROKEN = "BROKEN"


@dataclass
class LayerMetrics:
    """Per-layer divergence between reference and quantized activations."""

    index: int
    name: str
    cosine: float          # mean per-token cosine similarity (1.0 = identical)
    mse: float             # mean squared error (normalized by ref energy)
    is_culprit: bool = False
    # Fraction of the quantization-error residual's energy in its top singular
    # direction (computed for culprit layers only; None otherwise). High = the
    # error is structured/low-rank; low = diffuse noise.
    subspace_top1: float | None = None
    # Ensemble: how many independent metrics flagged this layer (cosine, MSE, ...)
    # and the resulting confidence ("high" = floor/≥2 agree, "low" = single signal).
    metric_votes: int = 0
    confidence: str = ""
    # --- MoE (present only for expert layers) ---
    expert_cosines: list[float] | None = None   # per-expert cosine, len = n_experts
    culprit_experts: list[int] = field(default_factory=list)
    router_kl: float | None = None               # KL between ref/quant routing distributions


@dataclass
class Diagnosis:
    """Full diagnostic result for one ref/target comparison."""

    verdict: Verdict
    mean_cosine: float
    min_cosine: float
    output_kl: float | None                 # KL(ref_logits || quant_logits), nats
    layers: list[LayerMetrics] = field(default_factory=list)
    culprit_indices: list[int] = field(default_factory=list)
    model: str = ""
    notes: str = ""
    # Filled in by the classifier (Phase 2).
    failure_mode: str = ""
    signature: list[str] = field(default_factory=list)
    repair: str = ""

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# --- Verdict thresholds (initial, hand-set; calibrated against real dumps later) ---

# --- Adaptive culprit detection ---
# A layer below this absolute cosine is ALWAYS a culprit (catches gross/uniform
# damage regardless of the model's distribution).
ABSOLUTE_FLOOR = 0.90
# A layer at or above this cosine is NEVER a culprit (guards against flagging
# near-perfect layers as "anomalies" on an otherwise healthy model).
HEALTHY_CEILING = 0.98
# Robust z-score (median + MAD) cutoff for flagging a layer as anomalous
# *relative to this model's own layer distribution*.
OUTLIER_Z = 3.5

# Back-compat alias — clean_prefix_len and older callers use this as the
# "is this layer clean" threshold.
CULPRIT_COSINE = ABSOLUTE_FLOOR

# Verdict thresholds.
BROKEN_COSINE = 0.70
KL_BROKEN = 1.0
DEGRADED_COSINE = 0.95
KL_DEGRADED = 0.1


def find_culprit_indices(cosines: list[float]) -> list[int]:
    """Flag damaged layers adaptively.

    Three rules, in order:
      1. absolute floor   — cosine < ABSOLUTE_FLOOR is always a culprit
      2. healthy ceiling  — cosine >= HEALTHY_CEILING is never a culprit
      3. relative outlier — in the band between, a layer is a culprit if it's a
         robust-z outlier below the model's own median (architecture-agnostic).

    The median + MAD estimator makes this work across architectures whose healthy
    baseline differs, while the floor/ceiling guards prevent both missed gross
    damage and false positives on uniformly-healthy or uniformly-degraded models.
    """
    if not cosines:
        return []
    med = statistics.median(cosines)
    mad = statistics.median([abs(c - med) for c in cosines])
    scale = 1.4826 * mad  # MAD -> normal-consistent std estimate

    culprits: list[int] = []
    for i, c in enumerate(cosines):
        if c < ABSOLUTE_FLOOR:
            culprits.append(i)
        elif c >= HEALTHY_CEILING:
            continue
        elif scale < 1e-6:
            # Tight distribution: only flag if the model is otherwise healthy
            # (median near-perfect) but THIS layer dropped out of the band.
            # If the median itself is in the band, damage is uniform/diffuse —
            # not a per-layer culprit (the verdict handles it as DEGRADED).
            if med >= HEALTHY_CEILING:
                culprits.append(i)
        elif (med - c) / scale > OUTLIER_Z:
            culprits.append(i)
    return culprits


def robust_high_outliers(values: list[float], z: float = OUTLIER_Z) -> list[int]:
    """Indices that are outliers on the HIGH side, via median + MAD.

    For error metrics where *bigger = worse* (e.g. normalized MSE). Mirror of the
    low-side logic in find_culprit_indices. Returns [] when the spread is
    degenerate (all values ~equal) so a uniform model flags nothing.
    """
    if not values:
        return []
    med = statistics.median(values)
    devs = [abs(v - med) for v in values]
    scale = 1.4826 * statistics.median(devs)
    if scale < 1e-6:
        # Degenerate spread (majority of values identical, e.g. a lone spike
        # among near-zeros). MAD is 0, so fall back to the mean deviation — this
        # still isolates a dramatic single outlier while ignoring uniform data.
        scale = 1.4826 * (sum(devs) / len(devs))
        if scale < 1e-6:
            return []  # truly all identical
    return [i for i, v in enumerate(values) if (v - med) / scale > z]


def decide_verdict(min_cosine: float, output_kl: float | None, n_culprits: int = 0) -> Verdict:
    """Map the worst layer + output divergence + culprit count onto a verdict."""
    kl = output_kl if output_kl is not None else 0.0
    if min_cosine < BROKEN_COSINE or kl > KL_BROKEN:
        return Verdict.BROKEN
    # Any flagged culprit means it's at least DEGRADED — keeps the verdict
    # consistent with the layer table (no "PASS" with a flagged layer).
    if n_culprits > 0 or min_cosine < DEGRADED_COSINE or kl > KL_DEGRADED:
        return Verdict.DEGRADED
    return Verdict.PASS
