"""Quantization ladder — the real-ground-truth validation.

Quantize ONE model across a matrix of methods x bit-widths, diagnose each against
the FP16 reference, and check the tool tracks the KNOWN monotonic degradation
(fewer bits -> more damage). Nobody injects anything here — the degradation is
real, which is what makes this the validation keystone.

Runs sequentially (capture ref once -> free -> each quant -> free), so peak GPU
memory is just the FP16 model size.

Usage:
  python scripts/quant_ladder.py --model Qwen/Qwen2.5-7B-Instruct --out reports/ladder
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from quant_doctor.capture import capture_dump
from quant_doctor.engine import diagnose_pair
from quant_doctor.loader import free_model, load_quantized, load_reference, load_tokenizer

EVAL_TEXT = (
    "The transformer architecture replaced recurrence with self-attention. "
    "Quantization reduces weight precision to save memory and speed inference, "
    "but can silently degrade quality. Photosynthesis converts sunlight into "
    "chemical energy. The capital of France is Paris, and water boils at 100 "
    "degrees Celsius at sea level. Large language models are trained on vast "
    "corpora of text scraped from the public internet over many years."
)

# The matrix. HQQ gives the full ladder; bnb adds cross-method points.
LADDER = [
    ("hqq8", "HQQ", 8),
    ("hqq4", "HQQ", 4),
    ("hqq3", "HQQ", 3),
    ("hqq2", "HQQ", 2),
    ("bnb8", "bitsandbytes", 8),
    ("bnb4", "bitsandbytes", 4),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("reports/ladder"))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.model)
    input_ids = tok(EVAL_TEXT, return_tensors="pt", truncation=True,
                    max_length=args.max_tokens).input_ids

    # --- Reference (FP16), captured once ---
    print(f"[ladder] capturing FP16 reference: {args.model}")
    ref_model = load_reference(args.model, device=args.device)
    input_ids = input_ids.to(next(ref_model.parameters()).device)
    ref_dump = capture_dump(ref_model, input_ids, model_name=f"{args.model} (fp16)")
    # Free in caller scope — free_model()'s `del` only drops its own arg binding,
    # not ours, so the model would stay resident and OOM the next load.
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for scheme, method, bits in LADDER:
        print(f"[ladder] {method} {bits}-bit ({scheme}) ...")
        try:
            qm = load_quantized(args.model, scheme=scheme, device=args.device)
            q_dump = capture_dump(
                qm, input_ids.to(next(qm.parameters()).device),
                model_name=f"{args.model} [{scheme}]",
            )
            del qm
            gc.collect()
            torch.cuda.empty_cache()
            diag = diagnose_pair(ref_dump, q_dump)
            rows.append({
                "method": method, "bits": bits, "scheme": scheme,
                "verdict": diag.verdict.value,
                "failure_mode": diag.failure_mode,
                "mean_cosine": round(diag.mean_cosine, 4),
                "min_cosine": round(diag.min_cosine, 4),
                "output_kl": round(diag.output_kl, 4) if diag.output_kl is not None else None,
                "n_culprits": len(diag.culprit_indices),
            })
            print(f"          -> {diag.verdict.value} | mean cos {diag.mean_cosine:.4f} "
                  f"| KL {diag.output_kl} | {diag.failure_mode}")
        except Exception as e:  # noqa: BLE001 - keep the sweep going, record the failure
            rows.append({"method": method, "bits": bits, "scheme": scheme, "error": str(e)})
            print(f"          -> ERROR: {e}")

    (args.out / "ladder.json").write_text(json.dumps(rows, indent=2))
    _write_markdown(args.out / "ladder.md", args.model, rows)
    print(f"\n[ladder] wrote {args.out}/ladder.json and ladder.md")
    _print_monotonicity(rows)


def _write_markdown(path: Path, model: str, rows: list[dict]) -> None:
    lines = [f"# Quantization Ladder — {model}", "",
             "Each row: the model quantized at that method/bit-width, diagnosed "
             "against the FP16 reference. Real degradation, no injected faults.", "",
             "| Method | Bits | Verdict | Mean cos | Min cos | KL (nats) | Culprits | Failure mode |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['method']} | {r['bits']} | ERROR | — | — | — | — | {r['error'][:40]} |")
        else:
            lines.append(
                f"| {r['method']} | {r['bits']} | {r['verdict']} | {r['mean_cosine']} | "
                f"{r['min_cosine']} | {r['output_kl']} | {r['n_culprits']} | {r['failure_mode']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _print_monotonicity(rows: list[dict]) -> None:
    """Sanity-check: within a method, does mean cosine drop as bits drop?"""
    print("\n[ladder] monotonicity check (mean cosine should fall as bits fall):")
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        if "error" not in r:
            by_method.setdefault(r["method"], []).append(r)
    for method, rs in by_method.items():
        rs.sort(key=lambda x: -x["bits"])
        seq = [(r["bits"], r["mean_cosine"]) for r in rs]
        mono = all(seq[i][1] >= seq[i + 1][1] - 1e-3 for i in range(len(seq) - 1))
        trail = "  ".join(f"{b}b={c:.4f}" for b, c in seq)
        print(f"  {method:14s} {'MONOTONIC ✓' if mono else 'NON-MONO ✗'}  {trail}")


if __name__ == "__main__":
    main()
