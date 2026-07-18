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


class QuantScheme(str, Enum):
    """How to produce the target from the reference (self-quantizer flow)."""
    none = "none"    # --target is a separate, already-quantized checkpoint
    bnb4 = "bnb4"    # self-quantize the ref to 4-bit NF4 on load
    bnb8 = "bnb8"    # self-quantize the ref to 8-bit on load


# A short built-in eval passage (used when --eval-set is not given).
_BUILTIN_EVAL = (
    "The transformer architecture revolutionized natural language processing by "
    "replacing recurrence with self-attention, allowing models to weigh the "
    "relevance of every token against every other token in parallel. Quantization "
    "reduces the numerical precision of a model's weights to shrink memory and "
    "accelerate inference, but doing so can silently degrade quality in ways that "
    "are hard to detect without careful, layer-by-layer measurement."
)


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
        ..., "--ref", help="Reference model (full precision). HF id or local path."
    ),
    target: Optional[str] = typer.Option(
        None, "--target", help="Already-quantized model (HF id/path). Omit when using --quantize."
    ),
    quantize: QuantScheme = typer.Option(
        QuantScheme.bnb4, "--quantize",
        help="Self-quantize the reference to produce the target (bnb4/bnb8), or 'none' to use --target.",
    ),
    eval_set: Optional[Path] = typer.Option(
        None, "--eval-set", help="Text file of eval text. Defaults to a built-in passage."
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
    inject_collapse: Optional[str] = typer.Option(
        None, "--inject-collapse",
        help="Fault injection: comma-separated layer indices to scramble in the target "
             "(for validation — manufactures a Computation-Collapse case on a real model).",
    ),
    base_bits: int = typer.Option(4, "--base-bits", help="Recipe: the target's quantization bit-width."),
    high_bits: int = typer.Option(8, "--high-bits", help="Recipe: bit-width for protected layers."),
    recipe_out: Optional[Path] = typer.Option(None, "--recipe-out", help="Write the recipe JSON here."),
) -> None:
    """Compare a reference model against its quantized version and report per-layer damage.

    Two modes:
      self-quantize : --ref MODEL --quantize bnb4     (we quantize; the true self-quantizer flow)
      compare       : --ref FP_MODEL --target QUANT_MODEL --quantize none
    """
    import torch

    from .capture import capture_dump
    from .engine import diagnose_pair
    from .loader import free_model, load_quantized, load_reference, load_tokenizer
    from .report import render_json, render_table

    if quantize is QuantScheme.none and target is None:
        raise typer.BadParameter("provide --target when --quantize none")

    corrupt = None
    if inject_collapse:
        corrupt = {int(x) for x in inject_collapse.split(",") if x.strip()}

    text = eval_set.read_text() if eval_set else _BUILTIN_EVAL

    console.rule("[bold]quant-doctor diagnose")
    console.print(f"  reference : [cyan]{ref}[/cyan]")
    console.print(f"  target    : [cyan]{target or f'self-quantized ({quantize.value})'}[/cyan]")
    console.print(f"  tokens    : {max_tokens}  device: {device}\n")

    tok = load_tokenizer(ref)
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids

    # --- Sequential capture: peak GPU memory is max(ref, quant), not the sum. ---
    with console.status("[dim]loading + capturing reference...[/dim]"):
        ref_model = load_reference(ref, device=device)
        input_ids = input_ids.to(next(ref_model.parameters()).device)
        ref_dump = capture_dump(ref_model, input_ids, model_name=f"{ref} (fp)")
        free_model(ref_model)

    with console.status("[dim]loading + capturing quantized target...[/dim]"):
        if quantize is QuantScheme.none:
            q_model = load_quantized(target, scheme="none", device=device)
            q_name = target
        else:
            q_model = load_quantized(ref, scheme=quantize.value, device=device)
            q_name = f"{ref} [{quantize.value}]"
        q_dump = capture_dump(
            q_model, input_ids.to(next(q_model.parameters()).device),
            model_name=q_name, corrupt=corrupt,
        )
        free_model(q_model)

    diag = diagnose_pair(ref_dump, q_dump)
    console.print()
    if output is OutputFormat.json:
        console.print_json(render_json(diag))
    else:
        render_table(diag, console)
        _emit_recipe(diag, q_dump.manifest, base_bits, high_bits, recipe_out)


def _emit_recipe(diag, manifest, base_bits: int, high_bits: int, recipe_out: Optional[Path]) -> None:
    """Generate, render, and optionally save a mixed-precision recipe."""
    import json as _json

    from .recipe import generate_recipe
    from .report import render_recipe

    recipe = generate_recipe(diag, manifest, base_bits=base_bits, high_bits=high_bits)
    render_recipe(recipe, console)
    if recipe is not None and recipe_out is not None:
        recipe_out.write_text(_json.dumps(recipe.to_dict(), indent=2))
        console.print(f"[dim]recipe written to {recipe_out}[/dim]")


@app.command()
def diagnose_dumps(
    ref_dir: Path = typer.Option(..., "--ref-dir", help="Directory of reference activation dumps."),
    target_dir: Path = typer.Option(..., "--target-dir", help="Directory of quantized activation dumps."),
    output: OutputFormat = typer.Option(OutputFormat.table, "--output", help="Report format."),
    base_bits: int = typer.Option(2, "--base-bits", help="Recipe: the target's quantization bit-width."),
    high_bits: int = typer.Option(4, "--high-bits", help="Recipe: bit-width for protected layers."),
    recipe_out: Optional[Path] = typer.Option(None, "--recipe-out", help="Write the recipe JSON here."),
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
        _emit_recipe(diag, target.manifest, base_bits, high_bits, recipe_out)


@app.command()
def diagnose_multi(
    root: Path = typer.Option(
        ..., "--root",
        help="Parent dir with one subdir per prompt, each containing ref/ and target/.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.table, "--output", help="Report format."),
) -> None:
    """Aggregate a diagnosis across several prompts (kills prompt-dependence).

    Layout:  <root>/<prompt-name>/ref  and  <root>/<prompt-name>/target
    A layer is flagged only if a majority of prompts agree it's damaged.
    """
    from .dumps import load_dump
    from .engine import diagnose_multi as _diagnose_multi
    from .report import render_json, render_table

    prompt_dirs = sorted(p for p in root.iterdir() if (p / "ref").is_dir() and (p / "target").is_dir())
    if not prompt_dirs:
        raise typer.BadParameter(f"no <prompt>/ref + <prompt>/target subdirs found under {root}")

    console.print(f"[dim]aggregating over {len(prompt_dirs)} prompts:"
                  f" {', '.join(p.name for p in prompt_dirs)}[/dim]\n")
    refs = [load_dump(p / "ref") for p in prompt_dirs]
    targets = [load_dump(p / "target") for p in prompt_dirs]
    diag = _diagnose_multi(refs, targets)

    if output is OutputFormat.json:
        console.print_json(render_json(diag))
    else:
        render_table(diag, console)


if __name__ == "__main__":
    app()
