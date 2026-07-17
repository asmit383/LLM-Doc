# Activation Dump Format (v1)

The interface contract between an inference stack that produces activations
(Arc / QTIP, or quant-doctor's own live capture) and the quant-doctor diagnostic
engine that consumes them.

A **dump** is a directory holding one model's per-layer activations for a fixed
eval input. Diagnosis compares two dumps — a **reference** (pre-quant) and a
**target** (quantized) — produced from the *same input tokens*.

```
<dump_dir>/
├── manifest.json            # metadata (see below)
├── layer_00.safetensors     # key "hidden" -> tensor [seq, hidden]
├── layer_01.safetensors
├── ...
├── layer_NN.safetensors
└── logits.safetensors       # key "logits" -> tensor [seq, vocab]   (optional)
```

For MoE models (Phase 5), per-expert activations and the router logits are added
alongside the block output:

```
├── layer_02.safetensors              # key "hidden" -> [seq, hidden]   (block output)
├── layer_02.expert_000.safetensors   # key "hidden" -> [seq, hidden]   (expert 0, dense eval)
├── layer_02.expert_001.safetensors
├── ...
├── layer_02.router.safetensors       # key "logits" -> [seq, n_experts]
```

Per-expert tensors are the **dense evaluation** of each expert over all `seq`
tokens (every expert on every token), so `cosine(ref_expert_e, quant_expert_e)`
is well-defined regardless of routing. This is what catches a single dead expert
that a healthy-looking block average would hide. The manifest declares:

```json
{ "is_moe": true, "moe_layers": [2, 6, 10], "n_experts": 8 }
```

## manifest.json

```json
{
  "model": "deepseek-v4-flash",
  "created_by": "arc",
  "n_layers": 32,
  "hidden_size": 4096,
  "vocab_size": 129280,
  "seq_len": 256,
  "dtype": "float32",
  "is_moe": true,
  "has_logits": true,
  "notes": "post-QTIP 2-bit, RUN-161"
}
```

## Rules

1. **Same input.** Reference and target dumps MUST come from identical input
   tokens, in the same order. Any activation difference is then attributable to
   quantization alone.
2. **Deterministic.** No sampling, no dropout. Greedy / fixed forward pass.
3. **Matched layers.** `n_layers` and every layer's `[seq, hidden]` shape must
   match between the two dumps.
4. **Tensor layout.** Batch is folded out: each `layer_NN` tensor is `[seq, hidden]`
   (a single eval sequence). Multiple sequences → concatenate along `seq`.
5. **dtype.** float32 or bfloat16. The engine upcasts to float32 for metrics.

## Why dumps (not live loading) for large / custom models

A 236B MoE has no FP16 reference that fits in memory, and can't be loaded via HF
`transformers` at all. The dump path decouples the two forward passes in *time*:
run the reference weights → dump → free → quantize/run target → dump. Peak memory
is `max(ref, quant)`, not the sum. The two dumps (a few hundred MB) are then
compared offline on any machine.
