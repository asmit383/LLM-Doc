"""Activation capture — Phase 1.

Registers forward hooks on every decoder block (and attention / FFN / lm_head)
so that a single forward pass records the per-layer hidden states. Running the
same input through the reference and quantized models yields *paired activations*
— the core object every downstream metric compares.
"""

from __future__ import annotations

# Phase 1: implement ActivationCapture(model) with register/run/collect.
