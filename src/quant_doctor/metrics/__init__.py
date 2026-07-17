"""Metrics operate on paired activations and are format-agnostic.

statistical    — Phase 1: cosine, MSE, KL divergence, PPL delta
interpretability — Phase 2: attention entropy, FFN sign-flip, logit lens, error subspace
propagation    — Phase 2: cross-layer error growth (QEP's Delta_m)
"""
