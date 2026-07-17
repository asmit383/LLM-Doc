"""Activation capture — live HF path.

Registers forward hooks on every decoder block so a single forward pass records
the per-layer hidden states. Running the same input through the reference and
quantized models yields *paired activations* — the object the engine compares.

Output is packaged as a `Dump` (the same struct the dumps path produces), so the
engine and report code are shared verbatim between the live and dumps paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dumps import Dump


def find_decoder_layers(model) -> nn.ModuleList:
    """Locate the ModuleList of transformer decoder blocks.

    Covers the common HF layout `model.model.layers` (Llama/Qwen/Mistral/Phi) and
    falls back to the largest ModuleList whose name ends in `.layers`.
    """
    # Fast path: the standard layout.
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers") and isinstance(inner.layers, nn.ModuleList):
        return inner.layers

    # Fallback: search for a `*.layers` ModuleList and take the largest.
    best = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not locate decoder layers — unsupported architecture")
    return best


class ActivationCapture:
    """Hook every decoder block and record its output hidden state."""

    def __init__(self, model):
        self.model = model
        self.layers = find_decoder_layers(model)
        self._acts: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _hook(self, idx: int):
        def fn(_module, _inp, out):
            # Decoder blocks return either a tensor or a tuple whose [0] is hidden.
            h = out[0] if isinstance(out, tuple) else out
            # [batch, seq, hidden] -> take batch 0 -> [seq, hidden], float32 on CPU.
            self._acts[idx] = h.detach()[0].float().cpu()
        return fn

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, model_name: str = "") -> Dump:
        """Run one forward pass and return a Dump of paired-ready activations."""
        self._acts = {}
        out = self.model(input_ids)
        logits = out.logits.detach()[0].float().cpu()  # [seq, vocab]

        layers = [self._acts[i] for i in range(len(self.layers))]
        manifest = {
            "model": model_name,
            "created_by": "live-capture",
            "n_layers": len(layers),
            "hidden_size": layers[0].shape[-1] if layers else 0,
            "vocab_size": logits.shape[-1],
            "seq_len": logits.shape[0],
            "dtype": "float32",
            "is_moe": False,
            "has_logits": True,
        }
        return Dump(manifest=manifest, layers=layers, logits=logits)


def capture_dump(model, input_ids: torch.Tensor, model_name: str = "") -> Dump:
    """Convenience: hook, run one forward pass, unhook, return the Dump."""
    with ActivationCapture(model) as cap:
        return cap.capture(input_ids, model_name=model_name)
