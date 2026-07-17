"""Diagnosis data structures and the verdict rules.

These are format-agnostic: a Diagnosis is built from paired activations regardless
of whether they came from live capture or dumps.
"""

from __future__ import annotations

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

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# --- Verdict thresholds (initial, hand-set; calibrated against real dumps later) ---

# A layer below this absolute cosine is flagged as a culprit.
CULPRIT_COSINE = 0.90
# Any layer below this, or output KL above KL_BROKEN, means BROKEN.
BROKEN_COSINE = 0.70
KL_BROKEN = 1.0
# Milder divergence -> DEGRADED.
DEGRADED_COSINE = 0.95
KL_DEGRADED = 0.1


def decide_verdict(min_cosine: float, output_kl: float | None) -> Verdict:
    """Map the worst layer + output divergence onto a coarse verdict."""
    kl = output_kl if output_kl is not None else 0.0
    if min_cosine < BROKEN_COSINE or kl > KL_BROKEN:
        return Verdict.BROKEN
    if min_cosine < DEGRADED_COSINE or kl > KL_DEGRADED:
        return Verdict.DEGRADED
    return Verdict.PASS
