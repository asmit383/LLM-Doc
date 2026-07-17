"""Command-line interface for quant-doctor.

Heavy imports (torch, transformers) are deferred into the command bodies so that
`--help` and arg parsing stay instant and work without a full ML stack installed.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__

app = typer.Typer(
    name="quant-doctor",
    help="Diagnose whether a quantized LLM is broken, where, and how to fix it.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


class QuantFormat(str, Enum):
    auto = "auto"
    gptq = "gptq"
    awq = "awq"
    bnb = "bnb"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"quant-doctor {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """quant-doctor — the missing QA layer for quantized LLMs."""


@app.command()
def diagnose(
    ref: str = typer.Option(
        ..., "--ref", help="Reference model (unquantized / pre-quant). HF id or local path."
    ),
    target: str = typer.Option(
        ..., "--target", help="Quantized model to diagnose. HF id or local path."
    ),
    eval_set: Optional[Path] = typer.Option(
        None, "--eval-set", help="Text file of eval prompts. Defaults to a built-in wikitext sample."
    ),
    fmt: QuantFormat = typer.Option(
        QuantFormat.auto, "--format", help="Quantization format of the target (auto-detected by default)."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.table, "--output", help="Report format."
    ),
    max_tokens: int = typer.Option(
        256, "--max-tokens", help="Number of eval tokens to run through both models."
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | cuda:N."
    ),
) -> None:
    """Compare a reference model against its quantized version and report per-layer damage."""
    # Phase 0: parse + echo the plan. Phase 1 wires in loader/capture/metrics/report.
    console.rule("[bold]quant-doctor diagnose")
    console.print(f"  reference : [cyan]{ref}[/cyan]")
    console.print(f"  target    : [cyan]{target}[/cyan]")
    console.print(f"  eval set  : {eval_set or '[dim]built-in wikitext sample[/dim]'}")
    console.print(f"  format    : {fmt.value}")
    console.print(f"  tokens    : {max_tokens}")
    console.print(f"  device    : {device}")
    console.print(f"  output    : {output.value}")
    console.print()
    console.print("[yellow]Phase 0 scaffold[/yellow] — loader/capture/metrics land in Phase 1.")


@app.command()
def diagnose_dumps(
    ref_dir: Path = typer.Option(..., "--ref-dir", help="Directory of reference activation dumps."),
    target_dir: Path = typer.Option(..., "--target-dir", help="Directory of quantized activation dumps."),
    output: OutputFormat = typer.Option(OutputFormat.table, "--output", help="Report format."),
) -> None:
    """Diagnose from pre-dumped activations (custom stacks: QTIP / Arc / MoE)."""
    # Deferred imports keep --help instant and torch out of the arg-parse path.
    from .dumps import load_dump
    from .engine import diagnose_pair
    from .report import render_json, render_table

    ref = load_dump(ref_dir)
    target = load_dump(target_dir)
    diag = diagnose_pair(ref, target)

    if output is OutputFormat.json:
        console.print_json(render_json(diag))
    else:
        render_table(diag, console)


if __name__ == "__main__":
    app()
