"""Model loading — Phase 1.

Loads the reference and quantized models through a single unified path. Both go
through HF `AutoModelForCausalLM`, which resolves the quantization backend
(GPTQ / AWQ / bitsandbytes / ...) from the target model's config automatically.
"""

from __future__ import annotations

# Phase 1: implement load_pair(ref, target, device, fmt) -> (ref_model, quant_model, tokenizer)
