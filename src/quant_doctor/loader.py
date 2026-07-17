"""Model loading — live HF path.

Loads the reference and quantized models through HF `AutoModelForCausalLM`.
Loading is done *sequentially* by the caller (capture ref -> free -> load target)
so peak GPU memory is max(ref, quant), never the sum. This is what lets a
single GPU diagnose a model whose fp + quant copies wouldn't co-reside.

"Self-quantizer" flow: the target is usually the *same* base model quantized by
us on load (bitsandbytes), not a separate download — matching the real use case
where the user IS the quantizer.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_reference(model_id: str, device: str = "auto", dtype=torch.float16):
    """Load the full-precision reference model."""
    dev = _resolve_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=dev,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_quantized(
    model_id: str,
    scheme: str = "bnb4",
    device: str = "auto",
    dtype=torch.float16,
):
    """Load a model, self-quantizing it on the fly.

    scheme:
      bnb4 — bitsandbytes 4-bit NF4 (double-quant)
      bnb8 — bitsandbytes 8-bit
      none — load as-is (model_id is already a quantized checkpoint, e.g. GPTQ/AWQ)
    """
    dev = _resolve_device(device)
    kwargs = dict(torch_dtype=dtype, device_map=dev, trust_remote_code=True)

    if scheme in ("bnb4", "bnb8"):
        from transformers import BitsAndBytesConfig

        if scheme == "bnb4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def free_model(model) -> None:
    """Release a model's GPU memory before loading the next one."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
