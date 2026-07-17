# Validation & Case Studies

How we show the diagnostic actually works. Two complementary methods: **synthetic
ground-truth** (controlled damage, known labels, repeatable via pytest) and **real
fault injection** (real model, real quantization, real error propagation on GPU).

## Method 1 — Synthetic ground-truth (`tests/`)

`quant_doctor.synthetic` injects four controlled damage patterns with known
expected labels. The test suite asserts the engine recovers each. Repeatable:

```bash
pytest            # 16 passed
```

| Case | Injected damage | Expected verdict | Expected mode | Result |
|------|-----------------|------------------|---------------|--------|
| healthy | ~0 noise everywhere | PASS | Healthy | ✅ |
| signal_degradation | noise growing with depth | DEGRADED | Signal Degradation | ✅ |
| computation_collapse | layer 2 scrambled + fallout | BROKEN | Computation Collapse (culprit=layer 2) | ✅ |
| format_bug | every layer scrambled | BROKEN | Format Bug | ✅ |

The suite also asserts *localization* (collapse culprit is exactly the injected
layer) and *format-bug shape* (damage from layer 0, no clean prefix).

## Method 2 — Real fault injection (H200, Qwen2.5-1.5B)

Both models are the *same* FP16 Qwen2.5-1.5B; the target is self-quantized to
4-bit NF4 (bitsandbytes) on load. No pre-made quant is involved.

### Case A — honest 4-bit quantization → Signal Degradation

```bash
quant-doctor diagnose --ref Qwen/Qwen2.5-1.5B --quantize bnb4
```

- **Verdict:** DEGRADED. mean cosine 0.990, min 0.978, output KL 0.205 nats.
- **Mode:** Signal Degradation — diffuse, no cliff, error spread across layers.
- **Real finding:** the two worst layers are **26–27** (just before `lm_head`),
  with MSE 20–45× the mid-stack. The tool auto-surfaces the "keep the late
  layers / `lm_head` at higher precision" heuristic — nobody told it that.

### Case B — injected collapse → Computation Collapse

Scramble layer 12's output in the quantized model; the corrupted activation then
propagates through the *real* downstream network:

```bash
quant-doctor diagnose --ref Qwen/Qwen2.5-1.5B --quantize bnb4 --inject-collapse 12
```

- **Verdict:** BROKEN. mean cosine 0.485, min −0.041, output KL 14.7 nats.
- **Mode:** Computation Collapse.
- **Localization:** *collapse onset at layer_12* after 12 clean layers; worst
  layer (20) correctly identified as **downstream fallout**, not the root cause.
- **Discriminator:** error residual at the onset is **structured (top-1
  concentration 1.00)** — the SVD metric cleanly separates a single malfunctioning
  component (collapse) from bnb4's diffuse rounding noise (Case A). This is the
  interpretability metric doing real work.

## Why this matters

The onset-vs-worst distinction and the structured-vs-diffuse discriminator are
exactly the signals a practitioner needs and no existing tool provides:

- **Signal Degradation (Case A):** cheap fix — keep a few layers higher-bit.
- **Computation Collapse (Case B):** the worst layer (20) is a red herring; the
  fix belongs at the onset (12), and training-free repair won't fully recover it.

## Scorecard

**6/6**: 4/4 synthetic ground-truth (pytest) + 2/2 real GPU case studies (correct
verdict *and* failure mode in every case).

## Still to validate (Phase 5)

The DeepSeek-V4 / QTIP case study via the dumps path — real MoE, real 2-bit
trellis quantization, per-expert diagnostics. Requires Arc to emit dumps in the
v1 format (`docs/dump-format.md`).
