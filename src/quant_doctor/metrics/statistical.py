"""Statistical metrics on paired activations — Phase 1.

Every function takes a reference tensor and a quantized tensor of identical shape
(same input, same layer) and returns a scalar measure of divergence. Tensors are
upcast to float32 internally so bf16 dumps compare cleanly.
"""

from __future__ import annotations

import torch


def layer_cosine(ref: torch.Tensor, quant: torch.Tensor) -> float:
    """Mean per-token cosine similarity between two `[seq, hidden]` activations.

    1.0 = identical direction (quantization changed nothing); lower = more damage.
    """
    ref = ref.float()
    quant = quant.float()
    cos = torch.nn.functional.cosine_similarity(ref, quant, dim=-1)  # [seq]
    return cos.mean().item()


def layer_mse(ref: torch.Tensor, quant: torch.Tensor) -> float:
    """Mean squared error, normalized by the reference's mean energy.

    Scale-free so it's comparable across layers with different activation norms.
    """
    ref = ref.float()
    quant = quant.float()
    err = torch.mean((ref - quant) ** 2)
    energy = torch.mean(ref ** 2) + 1e-8
    return (err / energy).item()


def output_kl(ref_logits: torch.Tensor, quant_logits: torch.Tensor) -> float:
    """Mean per-token KL(ref || quant) over the vocabulary, in nats.

    Measures how far the quantized model's next-token distribution has drifted
    from the reference. 0 = identical distributions.
    """
    ref_logits = ref_logits.float()
    quant_logits = quant_logits.float()
    log_p = torch.log_softmax(ref_logits, dim=-1)
    log_q = torch.log_softmax(quant_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)  # [seq]
    return kl.mean().item()
