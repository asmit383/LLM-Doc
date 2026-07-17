"""Statistical metrics on paired activations — Phase 1.

All functions take a reference tensor and a quantized tensor of identical shape
(same input, same layer) and return a scalar or per-position measure of divergence.
"""

from __future__ import annotations

# Phase 1: layer_cosine, layer_mse, output_kl_divergence, perplexity_delta
