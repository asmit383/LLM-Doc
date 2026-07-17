"""Report rendering — Phase 1.

Turns a Diagnosis into a CLI table (layer heatmap + verdict) or JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .diagnosis import Diagnosis, Verdict

_VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.DEGRADED: "bold yellow",
    Verdict.BROKEN: "bold red",
}

# Bar width in characters for the heatmap.
_BAR_W = 20


def _bar(cosine: float) -> Text:
    """A colored proportional bar for a cosine similarity in [0, 1]."""
    frac = max(0.0, min(1.0, cosine))
    filled = round(frac * _BAR_W)
    if cosine >= 0.95:
        style = "green"
    elif cosine >= 0.90:
        style = "yellow"
    else:
        style = "red"
    bar = Text("█" * filled, style=style)
    bar.append("░" * (_BAR_W - filled), style="grey37")
    return bar


def render_table(diag: Diagnosis, console: Console | None = None) -> None:
    console = console or Console()

    style = _VERDICT_STYLE[diag.verdict]
    header = Text()
    header.append("VERDICT: ", style="bold")
    header.append(diag.verdict.value, style=style)
    if diag.culprit_indices:
        header.append(f"   ({len(diag.culprit_indices)} culprit layer"
                      f"{'s' if len(diag.culprit_indices) != 1 else ''})", style="dim")

    summary = Text()
    summary.append(f"model      : {diag.model or '<unknown>'}\n")
    summary.append(f"mean cosine: {diag.mean_cosine:.4f}\n")
    summary.append(f"min cosine : {diag.min_cosine:.4f}")
    if diag.output_kl is not None:
        summary.append(f"\noutput KL  : {diag.output_kl:.4f} nats")

    console.print(Panel(summary, title=header, border_style=style, expand=False))

    table = Table(title="Layer Health", show_lines=False, header_style="bold")
    table.add_column("Layer", justify="right")
    table.add_column("Heatmap", justify="left")
    table.add_column("Cosine", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("", justify="left")

    for lm in diag.layers:
        flag = Text("← CRITICAL", style="bold red") if lm.is_culprit else Text("")
        table.add_row(
            lm.name,
            _bar(lm.cosine),
            f"{lm.cosine:.4f}",
            f"{lm.mse:.3e}",
            flag,
        )

    console.print(table)

    # --- Failure mode + signature + prescription (Phase 2) ---
    if diag.failure_mode:
        body = Text()
        body.append("FAILURE MODE: ", style="bold")
        body.append(f"{diag.failure_mode}\n", style=style)
        if diag.signature:
            body.append("\nSignature:\n", style="bold")
            for ev in diag.signature:
                body.append(f"  • {ev}\n")
        if diag.repair:
            body.append("\nPrescription:\n", style="bold")
            body.append(f"  {diag.repair}")
        console.print(Panel(body, border_style=style, expand=False))


def render_json(diag: Diagnosis) -> str:
    d = asdict(diag)
    d["verdict"] = diag.verdict.value
    return json.dumps(d, indent=2)
