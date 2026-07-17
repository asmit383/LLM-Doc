"""Load activation dumps written in the v1 dump format (see docs/dump-format.md).

A Dump is one model's per-layer activations for a fixed eval input. The engine
pairs a reference Dump against a target Dump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass
class Dump:
    manifest: dict
    layers: list[torch.Tensor]          # each [seq, hidden]
    logits: torch.Tensor | None         # [seq, vocab] or None

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def model(self) -> str:
        return self.manifest.get("model", "<unknown>")


def load_dump(path: str | Path) -> Dump:
    """Read a dump directory into memory."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {path} — is this a valid dump dir?")
    manifest = json.loads(manifest_path.read_text())

    n_layers = manifest["n_layers"]
    layers: list[torch.Tensor] = []
    for i in range(n_layers):
        f = path / f"layer_{i:02d}.safetensors"
        if not f.exists():
            raise FileNotFoundError(f"missing {f.name} (manifest declares {n_layers} layers)")
        layers.append(load_file(f)["hidden"])

    logits = None
    logits_path = path / "logits.safetensors"
    if logits_path.exists():
        logits = load_file(logits_path)["logits"]

    return Dump(manifest=manifest, layers=layers, logits=logits)


def check_pair(ref: Dump, target: Dump) -> None:
    """Validate that two dumps are comparable (same shape contract)."""
    if ref.n_layers != target.n_layers:
        raise ValueError(
            f"layer count mismatch: ref has {ref.n_layers}, target has {target.n_layers}"
        )
    for i, (a, b) in enumerate(zip(ref.layers, target.layers)):
        if a.shape != b.shape:
            raise ValueError(
                f"layer {i} shape mismatch: ref {tuple(a.shape)} vs target {tuple(b.shape)}"
            )
