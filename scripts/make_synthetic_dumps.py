"""Generate synthetic activation dumps with KNOWN, injected damage.

This lets us build and validate the diagnostic engine locally without an H100,
and doubles as the Phase 3 bug-injection harness: because we control the damage,
we have ground-truth labels to check the classifier against.

Cases produced (each a ref/ + target/ pair in the v1 dump format):
  healthy               target ~= ref                  -> expect PASS
  signal_degradation    uniform noise growing w/ depth -> expect DEGRADED, no single culprit
  computation_collapse  one layer scrambled + fallout  -> expect BROKEN, culprit = that layer
  format_bug            every layer wrecked uniformly  -> expect BROKEN, all layers culprit

Usage:
  python scripts/make_synthetic_dumps.py --out dumps/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

SEQ = 64
HIDDEN = 256
N_LAYERS = 12
VOCAB = 1000


def _cosine_noise(ref: torch.Tensor, ratio: float, gen: torch.Generator) -> torch.Tensor:
    """Return ref perturbed so that ||noise||/||ref|| ~= ratio (per token)."""
    noise = torch.randn(ref.shape, generator=gen)
    # scale each token's noise to `ratio` of that token's reference norm
    ref_norm = ref.norm(dim=-1, keepdim=True)
    noise_norm = noise.norm(dim=-1, keepdim=True) + 1e-8
    noise = noise / noise_norm * ref_norm * ratio
    return ref + noise


def _make_reference(gen: torch.Generator) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Structured-ish reference activations + logits."""
    layers = [torch.randn(SEQ, HIDDEN, generator=gen) * 2.0 for _ in range(N_LAYERS)]
    logits = torch.randn(SEQ, VOCAB, generator=gen) * 4.0  # peaky next-token dists
    return layers, logits


def _write_dump(out: Path, layers: list[torch.Tensor], logits: torch.Tensor, model: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(layers):
        save_file({"hidden": t.contiguous()}, str(out / f"layer_{i:02d}.safetensors"))
    save_file({"logits": logits.contiguous()}, str(out / "logits.safetensors"))
    manifest = {
        "model": model,
        "created_by": "make_synthetic_dumps.py",
        "n_layers": len(layers),
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "seq_len": SEQ,
        "dtype": "float32",
        "is_moe": False,
        "has_logits": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _logit_noise(logits: torch.Tensor, scale: float, gen: torch.Generator) -> torch.Tensor:
    return logits + torch.randn(logits.shape, generator=gen) * scale


def make_case(name: str, root: Path, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    ref_layers, ref_logits = _make_reference(gen)

    if name == "healthy":
        tgt_layers = [_cosine_noise(x, 0.03, gen) for x in ref_layers]
        tgt_logits = _logit_noise(ref_logits, 0.2, gen)

    elif name == "signal_degradation":
        # noise ratio grows gradually with depth -> smooth cosine decline
        tgt_layers = [
            _cosine_noise(x, 0.05 + 0.035 * i, gen) for i, x in enumerate(ref_layers)
        ]
        tgt_logits = _logit_noise(ref_logits, 1.5, gen)

    elif name == "computation_collapse":
        collapse_at = 2
        tgt_layers = []
        for i, x in enumerate(ref_layers):
            if i == collapse_at:
                tgt_layers.append(torch.randn(x.shape, generator=gen) * 2.0)  # scrambled
            elif i > collapse_at:
                tgt_layers.append(_cosine_noise(x, 0.20, gen))  # downstream fallout
            else:
                tgt_layers.append(_cosine_noise(x, 0.03, gen))  # clean before collapse
        tgt_logits = _logit_noise(ref_logits, 3.0, gen)

    elif name == "format_bug":
        # every layer near-orthogonal: a decode/format bug, not precision loss
        tgt_layers = [torch.randn(x.shape, generator=gen) * 2.0 for x in ref_layers]
        tgt_logits = _logit_noise(ref_logits, 5.0, gen)

    else:
        raise ValueError(f"unknown case: {name}")

    _write_dump(root / name / "ref", ref_layers, ref_logits, model=f"synthetic-{name}")
    _write_dump(root / name / "target", tgt_layers, tgt_logits, model=f"synthetic-{name}")


CASES = ["healthy", "signal_degradation", "computation_collapse", "format_bug"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dumps/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for i, case in enumerate(CASES):
        make_case(case, args.out, seed=args.seed + i)
        print(f"  wrote {args.out / case}/{{ref,target}}")
    print(f"\nDone. Try:\n  quant-doctor diagnose-dumps "
          f"--ref-dir {args.out}/computation_collapse/ref "
          f"--target-dir {args.out}/computation_collapse/target")


if __name__ == "__main__":
    main()
