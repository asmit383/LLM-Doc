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
        if lm.is_culprit:
            conf = f" ({lm.confidence} conf)" if lm.confidence else ""
            style = "bold red" if lm.confidence != "low" else "bold yellow"
            flag = Text(f"← CRITICAL{conf}", style=style)
        else:
            flag = Text("")
        table.add_row(
            lm.name,
            _bar(lm.cosine),
            f"{lm.cosine:.4f}",
            f"{lm.mse:.3e}",
            flag,
        )

    console.print(table)

    # --- MoE per-expert view for layers with a flagged expert (Phase 5) ---
    moe_layers = [lm for lm in diag.layers if lm.culprit_experts]
    if moe_layers:
        etable = Table(title="MoE Experts (flagged layers)", header_style="bold")
        etable.add_column("Layer", justify="right")
        etable.add_column("Expert", justify="right")
        etable.add_column("Heatmap", justify="left")
        etable.add_column("Cosine", justify="right")
        etable.add_column("", justify="left")
        for lm in moe_layers:
            for e, c in enumerate(lm.expert_cosines):
                dead = e in lm.culprit_experts
                flag = Text("← DEAD", style="bold red") if dead else Text("")
                etable.add_row(lm.name, str(e), _bar(c), f"{c:.4f}", flag)
        console.print(etable)

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


def render_recipe(recipe, console: Console | None = None) -> None:
    """Render a mixed-precision Recipe panel (Phase 4)."""
    console = console or Console()
    if recipe is None:
        return

    body = Text()
    body.append("RECIPE — mixed precision\n", style="bold cyan")
    body.append(f"\n  base: {recipe.base_bits}-bit\n")
    if recipe.overrides:
        body.append("  keep at higher precision:\n", style="bold")
        for ov in recipe.overrides:
            body.append(f"    {ov.name} → {ov.bits}-bit", style="cyan")
            body.append(f"   ({ov.reason})\n", style="dim")
        body.append(f"\n  est. VRAM delta: +{recipe.est_vram_delta_gb:.2f} GB\n")
    body.append(f"  confidence: {recipe.confidence}")
    if recipe.notes:
        body.append(f"\n\n  {recipe.notes}", style="yellow")

    console.print(Panel(body, border_style="cyan", expand=False))


def render_json(diag: Diagnosis) -> str:
    d = asdict(diag)
    d["verdict"] = diag.verdict.value
    return json.dumps(d, indent=2)
