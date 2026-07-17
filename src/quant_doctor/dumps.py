"""Load activation dumps written in the v1 dump format (see docs/dump-format.md).

A Dump is one model's per-layer activations for a fixed eval input. The engine
pairs a reference Dump against a target Dump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass
class Dump:
    manifest: dict
    layers: list[torch.Tensor]          # each [seq, hidden]
    logits: torch.Tensor | None         # [seq, vocab] or None
    # MoE: layer index -> list of per-expert outputs (each [seq, hidden]); and
    # layer index -> router logits [seq, n_experts]. Empty for dense models.
    experts: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    routers: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def model(self) -> str:
        return self.manifest.get("model", "<unknown>")

    @property
    def moe_layers(self) -> list[int]:
        return sorted(self.experts.keys())


def load_dump(path: str | Path) -> Dump:
    """Read a dump directory into memory (dense + optional MoE per-expert files)."""
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

    # MoE: load per-expert outputs + router logits for declared expert layers.
    experts: dict[int, list[torch.Tensor]] = {}
    routers: dict[int, torch.Tensor] = {}
    n_experts = manifest.get("n_experts", 0)
    for li in manifest.get("moe_layers", []):
        expert_tensors = []
        for e in range(n_experts):
            ef = path / f"layer_{li:02d}.expert_{e:03d}.safetensors"
            if ef.exists():
                expert_tensors.append(load_file(ef)["hidden"])
        if expert_tensors:
            experts[li] = expert_tensors
        rf = path / f"layer_{li:02d}.router.safetensors"
        if rf.exists():
            routers[li] = load_file(rf)["logits"]

    return Dump(manifest=manifest, layers=layers, logits=logits, experts=experts, routers=routers)


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
