# Arc Integration — emitting V4 dumps

The bridge from quant-doctor to a real DeepSeek-V4 case study. V4-Flash is a custom
architecture that HF `transformers` cannot load, so its activations can only be
produced by Arc (the Rust inference engine). This spec defines what Arc must emit
so the dumps drop straight into `quant-doctor diagnose-dumps`.

## What's needed: two dumps, same input

| Dump | Weight state | How |
|------|--------------|-----|
| `activations_ref/` | pre-ISQ (FP4/FP8, dequantized to BF16) | run the forward pass **before** `model.quantize()` |
| `activations_quant/`| post-QTIP 2-bit | run the forward pass **after** ISQ |

Both on identical input tokens, deterministic (greedy, no sampling). There is no
FP16 V4 — the FP4/FP8 pre-ISQ state is the reference, which is the right question
anyway ("did QTIP make it worse than what DeepSeek shipped?").

## What to write (per the v1 dump format)

Env-gated (e.g. `QUANT_DIAG_DUMP=<dir>`), so normal serving is untouched:

- **Per decoder block** — after each layer, write `layer_NN.safetensors` (key
  `hidden`, `[seq, hidden]`, batch 0, upcast to f32).
- **Final logits** — `logits.safetensors` (key `logits`, `[seq, vocab]`).
- **Per expert (MoE layers)** — dense-evaluate each expert over all tokens and
  write `layer_NN.expert_MMM.safetensors`. If dense eval of all 256 experts is too
  costly, emit only the experts routed to ≥1 token (note the reduction in the
  manifest).
- **Router logits** — `layer_NN.router.safetensors` (key `logits`, `[seq, n_experts]`).
- **manifest.json** — `n_layers`, `hidden_size`, `vocab_size`, `is_moe: true`,
  `moe_layers`, `n_experts`, plus `layer_params` / `lm_head_params` for VRAM
  estimates if cheap to compute.

## Suggested implementation shape (Rust / mistralrs)

In `deepseek4.rs` forward, gated on the env var:

```rust
if let Some(dir) = std::env::var("QUANT_DIAG_DUMP").ok() {
    // after each decoder block:
    save_safetensors(&format!("{dir}/layer_{i:02}.safetensors"),
                     &[("hidden", hidden.i(0)?.to_dtype(F32)?)])?;
    // MoE: after computing expert outputs, dump each expert + router logits
}
```

Then two runs on the same prompt:

```bash
QUANT_DIAG_DUMP=activations_ref   ./mistralrs ... --isq none   # pre-ISQ reference
QUANT_DIAG_DUMP=activations_quant ./mistralrs ... --isq qtip2  # post-QTIP target
```

## ⚠️ Operational note

This requires **rebuilding Arc and stopping the running V4 server** for the two
dump runs. It is disruptive to a live serve and must be scheduled deliberately —
it is not a hot-path change. The quant-doctor side is already complete and
validated on synthetic V4-shaped dumps (see `docs/validation.md`); it will ingest
real dumps with zero code change the moment they exist.
