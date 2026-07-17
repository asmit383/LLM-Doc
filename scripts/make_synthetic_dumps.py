"""Generate synthetic activation dumps with known, injected damage.

Thin CLI wrapper around quant_doctor.synthetic (the importable generator that the
test suite also uses). See that module for the case definitions.

Usage:
  python scripts/make_synthetic_dumps.py --out dumps/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_doctor.synthetic import CASES, make_case


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
