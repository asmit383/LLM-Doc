"""Convert Arc's V4 activation dumps (L{NN}.hidden.npy) into the quant-doctor dump format.

Arc's patched forward writes one .npy per decoder layer. This repackages them into
a quant-doctor dump directory (layer_NN.safetensors + manifest.json).

Two modes:
  reference : straight conversion of a dump dir
  target    : same, optionally scrambling given layers (post-hoc fault injection)
              to produce a contrast when only one real weight-state is available.

Usage:
  # real reference
  python scripts/arc_npy_to_dump.py --in dump_qtip2 --out qd_ref --model "V4-Flash qtip2"
  # target with an injected dead layer (real V4 activations, synthetic fault)
  python scripts/arc_npy_to_dump.py --in dump_qtip2 --out qd_target --model "V4-Flash qtip2 +fault" --scramble 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from safetensors.torch import save_file


def load_layers(in_dir: str):
    files = sorted(glob.glob(os.path.join(in_dir, "L*.hidden.npy")))
    layers = []
    for f in files:
        idx = int(re.search(r"L(\d+)\.hidden", os.path.basename(f)).group(1))
        arr = np.load(f)
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        # Collapse any leading dims (batch / 4-D mHC) into "tokens": [-1, hidden].
        t = t.reshape(-1, t.shape[-1])
        layers.append((idx, t))
    layers.sort(key=lambda x: x[0])
    return [t for _, t in layers]


def scramble(t: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(t)
    scale = t.norm(dim=-1, keepdim=True) / (noise.norm(dim=-1, keepdim=True) + 1e-8)
    return noise * scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--model", default="V4")
    ap.add_argument("--scramble", default="", help="comma-separated layer indices to scramble")
    args = ap.parse_args()

    layers = load_layers(args.in_dir)
    if not layers:
        raise SystemExit(f"no L*.hidden.npy found in {args.in_dir}")

    scr = {int(x) for x in args.scramble.split(",") if x.strip()}
    os.makedirs(args.out_dir, exist_ok=True)
    for i, t in enumerate(layers):
        if i in scr:
            t = scramble(t)
        save_file({"hidden": t.contiguous()}, os.path.join(args.out_dir, f"layer_{i:02d}.safetensors"))

    manifest = {
        "model": args.model,
        "created_by": "arc_npy_to_dump.py",
        "n_layers": len(layers),
        "hidden_size": layers[0].shape[-1],
        "seq_len": layers[0].shape[0],
        "dtype": "float32",
        "is_moe": True,
        "has_logits": False,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(layers)} layers to {args.out_dir}"
          + (f" (scrambled {sorted(scr)})" if scr else ""))


if __name__ == "__main__":
    main()
