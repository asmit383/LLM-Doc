"""Mixed-precision recipe generation — Phase 4.

Closes the loop from diagnosis to fix: given the culprit layers and failure mode,
emit a concrete per-layer bit-width plan, estimate its VRAM cost, and say how much
recovery to expect. This is the "keep lm_head + layers {12} at higher bits" output
that existing quantizers don't provide.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .classifier import FailureMode
from .diagnosis import CULPRIT_COSINE, Diagnosis, Verdict
from .metrics.propagation import clean_prefix_len


@dataclass
class LayerOverride:
    name: str          # module name, e.g. "model.layers.12" or "lm_head"
    bits: int
    reason: str


@dataclass
class Recipe:
    base_bits: int
    overrides: list[LayerOverride] = field(default_factory=list)
    est_vram_delta_gb: float = 0.0
    confidence: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "base_bits": self.base_bits,
            "overrides": {ov.name: ov.bits for ov in self.overrides},
            "reasons": {ov.name: ov.reason for ov in self.overrides},
            "est_vram_delta_gb": round(self.est_vram_delta_gb, 3),
            "confidence": self.confidence,
            "notes": self.notes,
        }


def _worst_layers(diag: Diagnosis, k: int):
    return sorted(diag.layers, key=lambda lm: lm.cosine)[:k]


def _vram_delta_gb(overrides: dict[str, LayerOverride], manifest: dict, base_bits: int) -> float:
    """Extra bytes from raising each override's layer above base_bits, in GB."""
    n = manifest.get("n_layers", 0)
    hidden = manifest.get("hidden_size", 0)
    vocab = manifest.get("vocab_size", 0)
    # Real counts if the dump carries them (live path); else approximate.
    layer_params = manifest.get("layer_params") or [12 * hidden * hidden] * n
    head_params = manifest.get("lm_head_params") or (hidden * vocab)
    embed_params = manifest.get("embed_params") or (hidden * vocab)

    total_bytes = 0.0
    for name, ov in overrides.items():
        if name == "lm_head":
            p = head_params
        elif name == "embed_tokens":
            p = embed_params
        else:
            idx = int(name.rsplit(".", 1)[-1])
            p = layer_params[idx] if 0 <= idx < len(layer_params) else 0
        total_bytes += p * (ov.bits - base_bits) / 8.0
    return total_bytes / 1e9


def generate_recipe(
    diag: Diagnosis, manifest: dict, base_bits: int = 4, high_bits: int = 8
) -> Recipe | None:
    """Produce a mixed-precision recipe from a diagnosis. None if nothing to fix."""
    if diag.verdict is Verdict.PASS:
        return None

    n = len(diag.layers)
    mode = diag.failure_mode

    # A format/decode bug is not a precision problem — refuse to prescribe bits.
    if mode == FailureMode.FORMAT_BUG.value:
        return Recipe(
            base_bits=base_bits,
            overrides=[],
            est_vram_delta_gb=0.0,
            confidence="n/a",
            notes="Format/decode bug detected — do NOT adjust bit-widths. Fix the "
                  "dequantization (e.g. format misinterpretation) first, then re-diagnose.",
        )

    overrides: dict[str, LayerOverride] = {}

    # Always protect the output head — its errors map straight to token logits.
    overrides["lm_head"] = LayerOverride(
        "lm_head", high_bits, "output head — quantization error maps directly to tokens"
    )

    if mode == FailureMode.COMPUTATION_COLLAPSE.value:
        expert_layers = [lm for lm in diag.layers if lm.culprit_experts]
        if expert_layers:
            # MoE expert blowup: target the offending MoE layer(s) directly.
            for lm in expert_layers:
                overrides[f"model.layers.{lm.index}"] = LayerOverride(
                    f"model.layers.{lm.index}", high_bits,
                    f"dead expert(s) {lm.culprit_experts} — keep this MoE layer high",
                )
        else:
            onset = clean_prefix_len([lm.cosine for lm in diag.layers], CULPRIT_COSINE)
            for idx in (onset - 1, onset, onset + 1):
                if 0 <= idx < n:
                    overrides[f"model.layers.{idx}"] = LayerOverride(
                        f"model.layers.{idx}", high_bits,
                        "collapse onset / neighbor — keep at high precision",
                    )
        confidence = ("LOW — training-free repair unlikely to fully recover; the onset "
                      "layer destroys information. Consider a LoRA / fine-tune too.")
    else:
        # Signal Degradation / Generic: protect the worst ~10% of layers.
        k = max(1, round(n * 0.1))
        for lm in _worst_layers(diag, k):
            overrides[f"model.layers.{lm.index}"] = LayerOverride(
                f"model.layers.{lm.index}", high_bits,
                f"elevated error (cosine {lm.cosine:.3f})",
            )
        confidence = "HIGH — diffuse noise; mixed-precision should recover most quality."

    return Recipe(
        base_bits=base_bits,
        overrides=list(overrides.values()),
        est_vram_delta_gb=_vram_delta_gb(overrides, manifest, base_bits),
        confidence=confidence,
        notes="",
    )
