"""Synthetic activation dumps with KNOWN, injected damage.

Used to build and validate the diagnostic engine without a GPU, and — because
the damage is controlled — as the ground-truth oracle for the classifier: each
case has an expected verdict and failure mode the test suite asserts against.

Cases:
  healthy               target ~= ref                  -> PASS,     Healthy
  signal_degradation    uniform noise growing w/ depth -> DEGRADED, Signal Degradation
  computation_collapse  one layer scrambled + fallout  -> BROKEN,   Computation Collapse (culprit = layer 2)
  format_bug            every layer wrecked uniformly  -> BROKEN,   Format Bug
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

SEQ = 64
HIDDEN = 256
N_LAYERS = 12
VOCAB = 1000
COLLAPSE_AT = 2

CASES = ["healthy", "signal_degradation", "computation_collapse", "format_bug"]

# MoE cases (per-expert dumps). BLOWN mimics the V4 layer-2 MXFP4-as-INT4 blowup.
MOE_LAYERS = [2, 6, 10]
N_EXPERTS = 8
BLOWN_LAYER = 2
BLOWN_EXPERT = 3
MOE_CASES = ["moe_healthy", "moe_expert_blowup"]

# Ground truth the test suite asserts against: (verdict, failure_mode).
EXPECTED = {
    "healthy": ("PASS", "Healthy"),
    "signal_degradation": ("DEGRADED", "Signal Degradation"),
    "computation_collapse": ("BROKEN", "Computation Collapse"),
    "format_bug": ("BROKEN", "Format Bug"),
    "moe_healthy": ("PASS", "Healthy"),
    "moe_expert_blowup": ("BROKEN", "Computation Collapse"),
}


def _cosine_noise(ref: torch.Tensor, ratio: float, gen: torch.Generator) -> torch.Tensor:
    """Perturb ref so that ||noise|| / ||ref|| ~= ratio per token."""
    noise = torch.randn(ref.shape, generator=gen)
    ref_norm = ref.norm(dim=-1, keepdim=True)
    noise_norm = noise.norm(dim=-1, keepdim=True) + 1e-8
    return ref + noise / noise_norm * ref_norm * ratio


def _make_reference(gen: torch.Generator) -> tuple[list[torch.Tensor], torch.Tensor]:
    layers = [torch.randn(SEQ, HIDDEN, generator=gen) * 2.0 for _ in range(N_LAYERS)]
    logits = torch.randn(SEQ, VOCAB, generator=gen) * 4.0
    return layers, logits


def _logit_noise(logits: torch.Tensor, scale: float, gen: torch.Generator) -> torch.Tensor:
    return logits + torch.randn(logits.shape, generator=gen) * scale


def _write_dump(out: Path, layers: list[torch.Tensor], logits: torch.Tensor, model: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(layers):
        save_file({"hidden": t.contiguous()}, str(out / f"layer_{i:02d}.safetensors"))
    save_file({"logits": logits.contiguous()}, str(out / "logits.safetensors"))
    manifest = {
        "model": model,
        "created_by": "quant_doctor.synthetic",
        "n_layers": len(layers),
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "seq_len": SEQ,
        "dtype": "float32",
        "is_moe": False,
        "has_logits": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def make_case(name: str, root: Path, seed: int = 0) -> Path:
    """Generate one case's ref/ and target/ dumps under root/<name>/. Returns root/<name>."""
    gen = torch.Generator().manual_seed(seed)
    ref_layers, ref_logits = _make_reference(gen)

    if name == "healthy":
        tgt_layers = [_cosine_noise(x, 0.03, gen) for x in ref_layers]
        tgt_logits = _logit_noise(ref_logits, 0.2, gen)

    elif name == "signal_degradation":
        tgt_layers = [_cosine_noise(x, 0.05 + 0.035 * i, gen) for i, x in enumerate(ref_layers)]
        tgt_logits = _logit_noise(ref_logits, 1.5, gen)

    elif name == "computation_collapse":
        tgt_layers = []
        for i, x in enumerate(ref_layers):
            if i == COLLAPSE_AT:
                tgt_layers.append(torch.randn(x.shape, generator=gen) * 2.0)
            elif i > COLLAPSE_AT:
                tgt_layers.append(_cosine_noise(x, 0.20, gen))
            else:
                tgt_layers.append(_cosine_noise(x, 0.03, gen))
        tgt_logits = _logit_noise(ref_logits, 3.0, gen)

    elif name == "format_bug":
        tgt_layers = [torch.randn(x.shape, generator=gen) * 2.0 for x in ref_layers]
        tgt_logits = _logit_noise(ref_logits, 5.0, gen)

    else:
        raise ValueError(f"unknown case: {name}")

    _write_dump(root / name / "ref", ref_layers, ref_logits, model=f"synthetic-{name}")
    _write_dump(root / name / "target", tgt_layers, tgt_logits, model=f"synthetic-{name}")
    return root / name


def _write_moe_dump(out, layers, logits, experts, routers, model) -> None:
    """Write a dense dump plus per-expert + router files for MoE layers."""
    _write_dump(out, layers, logits, model)
    for li, expert_list in experts.items():
        for e, t in enumerate(expert_list):
            save_file({"hidden": t.contiguous()},
                      str(out / f"layer_{li:02d}.expert_{e:03d}.safetensors"))
    for li, r in routers.items():
        save_file({"logits": r.contiguous()}, str(out / f"layer_{li:02d}.router.safetensors"))
    # Patch the manifest with MoE fields.
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update({"is_moe": True, "moe_layers": sorted(experts.keys()), "n_experts": N_EXPERTS})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def make_moe_case(name: str, root: Path, seed: int = 0) -> Path:
    """Generate a MoE case with per-expert dumps. Returns root/<name>."""
    gen = torch.Generator().manual_seed(seed)
    ref_layers, ref_logits = _make_reference(gen)
    ref_experts = {li: [torch.randn(SEQ, HIDDEN, generator=gen) * 2.0 for _ in range(N_EXPERTS)]
                   for li in MOE_LAYERS}
    ref_routers = {li: torch.randn(SEQ, N_EXPERTS, generator=gen) for li in MOE_LAYERS}

    # Block outputs stay healthy in both cases — the point of the blowup case is
    # that the damage is invisible at the block level and only shows per-expert.
    tgt_layers = [_cosine_noise(x, 0.03, gen) for x in ref_layers]
    tgt_routers = {li: _logit_noise(ref_routers[li], 0.1, gen) for li in MOE_LAYERS}
    tgt_experts = {li: [_cosine_noise(e, 0.03, gen) for e in ref_experts[li]] for li in MOE_LAYERS}

    if name == "moe_healthy":
        tgt_logits = _logit_noise(ref_logits, 0.2, gen)
    elif name == "moe_expert_blowup":
        # One expert in one layer is scrambled (the MXFP4-as-INT4 signature).
        tgt_experts[BLOWN_LAYER][BLOWN_EXPERT] = torch.randn(SEQ, HIDDEN, generator=gen) * 2.0
        tgt_logits = _logit_noise(ref_logits, 1.0, gen)
    else:
        raise ValueError(f"unknown MoE case: {name}")

    _write_moe_dump(root / name / "ref", ref_layers, ref_logits, ref_experts, ref_routers,
                    model=f"synthetic-{name}")
    _write_moe_dump(root / name / "target", tgt_layers, tgt_logits, tgt_experts, tgt_routers,
                    model=f"synthetic-{name}")
    return root / name


def make_all(root: Path, seed: int = 0) -> None:
    for i, case in enumerate(CASES):
        make_case(case, root, seed=seed + i)
    for i, case in enumerate(MOE_CASES):
        make_moe_case(case, root, seed=seed + 100 + i)
