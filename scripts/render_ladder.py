"""Render a quant_ladder.py run (ladder.json) in the quant-doctor rich style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_VERDICT_STYLE = {"PASS": "bold green", "DEGRADED": "bold yellow", "BROKEN": "bold red"}
_BAR_W = 18


def _bar(cosine: float) -> Text:
    frac = max(0.0, min(1.0, cosine))
    filled = round(frac * _BAR_W)
    style = "green" if cosine >= 0.97 else "yellow" if cosine >= 0.90 else "red"
    bar = Text("█" * filled, style=style)
    bar.append("░" * (_BAR_W - filled), style="grey37")
    return bar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="reports/ladder/ladder.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    args = ap.parse_args()

    rows = json.loads(Path(args.json).read_text())
    console = Console(force_terminal=True, width=132)

    console.rule(f"[bold]quant-doctor · Quantization Ladder — {args.model}")
    console.print(Panel(
        Text.assemble(
            ("Real-ground-truth validation.  ", "bold"),
            "Each row = the model quantized at that method/bit-width, diagnosed against "
            "the FP16 reference. No injected faults — damage worsens as bits drop.",
        ),
        border_style="cyan", expand=False,
    ))

    table = Table(title="Method × Bit-width", header_style="bold", show_lines=False)
    table.add_column("Method", justify="left")
    table.add_column("Bits", justify="right")
    table.add_column("Verdict", justify="left")
    table.add_column("Mean-cosine", justify="left")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("KL (nats)", justify="right")
    table.add_column("Culprits", justify="right")
    table.add_column("Failure mode", justify="left")

    for r in rows:
        if "error" in r:
            table.add_row(r["method"], str(r["bits"]), Text("ERROR", style="red"),
                          "", "", "", "", "", r["error"][:32])
            continue
        v = r["verdict"]
        table.add_row(
            r["method"], str(r["bits"]),
            Text(v, style=_VERDICT_STYLE.get(v, "white")),
            _bar(r["mean_cosine"]),
            f"{r['mean_cosine']:.4f}", f"{r['min_cosine']:.4f}",
            f"{r['output_kl']:.4f}", str(r["n_culprits"]),
            r["failure_mode"],
        )
    console.print(table)

    # Monotonicity panel — mean cosine must fall as bits fall, per method.
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        if "error" not in r:
            by_method.setdefault(r["method"], []).append(r)
    body = Text()
    body.append("MONOTONICITY  ", style="bold")
    body.append("(mean cosine should fall as bits fall — the validation)\n\n")
    all_mono = True
    for method, rs in by_method.items():
        rs.sort(key=lambda x: -x["bits"])
        seq = [(r["bits"], r["mean_cosine"]) for r in rs]
        mono = all(seq[i][1] >= seq[i + 1][1] - 1e-3 for i in range(len(seq) - 1))
        all_mono = all_mono and mono
        body.append(f"  {method:14s} ")
        body.append("✓ MONOTONIC" if mono else "✗ NON-MONO",
                    style="bold green" if mono else "bold red")
        body.append("   " + "  ".join(f"{b}b={c:.4f}" for b, c in seq) + "\n")
    body.append("\nVerdict: ", style="bold")
    body.append("the tool tracks known degradation across every bit-width ✓"
                if all_mono else "non-monotonic — investigate", style="bold green")
    console.print(Panel(body, border_style="green", expand=False))


if __name__ == "__main__":
    main()
