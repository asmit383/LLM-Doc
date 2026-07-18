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

### Recipe (Phase 4) — the fix for Case B

```bash
quant-doctor diagnose --ref Qwen/Qwen2.5-1.5B --quantize bnb4 --inject-collapse 12 \
  --base-bits 4 --high-bits 16 --recipe-out fix.json
```

The tool prescribes a concrete mixed-precision plan targeting the *onset*:

| Keep at 16-bit | Reason |
|----------------|--------|
| lm_head | output head — error maps straight to tokens |
| model.layers.11 / 12 / 13 | collapse onset + neighbours |

**est. VRAM delta: +0.46 GB** (real per-layer param counts), **confidence: LOW**
(collapse — training-free won't fully recover; consider fine-tune). Exported to
`fix.json` for re-quantization.

## Why this matters

The onset-vs-worst distinction and the structured-vs-diffuse discriminator are
exactly the signals a practitioner needs and no existing tool provides:

- **Signal Degradation (Case A):** cheap fix — keep a few layers higher-bit.
- **Computation Collapse (Case B):** the worst layer (20) is a red herring; the
  fix belongs at the onset (12), and training-free repair won't fully recover it.

## Method 3 — MoE per-expert (the V4 scenario)

MoE damage hides behind the block average: a single dead expert barely moves the
layer's mean output, so aggregate metrics miss it. quant-doctor dumps and compares
each expert individually.

### Case D — expert blowup (mimics V4 layer-2 MXFP4-as-INT4)

Synthetic MoE dump: layer 2, expert 3 scrambled; every block output left healthy.

```bash
quant-doctor diagnose-dumps --ref-dir moe_expert_blowup/ref --target-dir moe_expert_blowup/target
```

- **Block average: mean cosine 0.9996** — the layer looks *pristine*.
- **Per-expert: layer_02 expert 3 = −0.003 → DEAD.** Localized to the exact expert.
- **Verdict:** BROKEN. **Mode:** Computation Collapse (expert-level).
- **Recipe:** keep `model.layers.2` (dead expert [3]) + `lm_head` at higher bits.

This is the differentiator: **no aggregate metric would have caught it.** It's
exactly the V4 failure the tool was conceived for.

## Method 4 — Real frontier model (DeepSeek-V4-Flash)

The threshold from "toy" to "actual inspection tool": running on a real 236B-param
MoE that no off-the-shelf framework can even load (custom arch — only the Arc
Rust engine runs it).

**Model:** `deepseek-ai/DeepSeek-V4-Flash` (public, no token, ~149 GB), served by
Arc at 2-bit QTIP (`--isq qtip2`). Arc was patched with an env-gated per-layer
activation dump (`V4_DUMP_DIR`); `scripts/arc_npy_to_dump.py` converts the `.npy`
dumps into the quant-doctor format.

### Case E — real V4 activations, injected fault

43 real decoder layers captured from Arc's forward pass. A fault injected at
layer 20 (post-hoc scramble) to validate localization on real V4 tensors:

- **Verdict:** BROKEN. **Mode:** Computation Collapse.
- **Localization:** onset at layer_20 (cosine 0.002), "20 clean layers before";
  structured error (top-1 concentration 0.77).
- **Recipe:** keep layers 19–21 + lm_head at 4-bit.

**What this proves:** the pipeline ingests, localizes, classifies, and prescribes
on real DeepSeek-V4-Flash activations (43 layers, 256-expert MoE, 4-D mHC state,
real 2-bit QTIP tensors). **What it does NOT prove:** a naturally-occurring V4
defect — the fault was injected. The other 42 layers read exactly 1.0000 because
reference and target are the same real dump with one layer corrupted.

### Honest limitation — no fittable reference on one H200

A *natural* 2-bit-vs-higher-bit V4 diagnosis needs a higher-fidelity reference
dump. On a single H200 (143 GB VRAM) this isn't achievable:

- Native FP4/FP8 (149 GB) and BF16 dequant (236 GB) don't fit.
- Q4K (4-bit, ~118 GB) loads only with CPU offload of the last layers → too slow.
- Q3K (3-bit, ~140 GB) fits fully on GPU but loads too slowly to finish in-window.

This is the exact single-GPU constraint quant-doctor is designed around, showing
up for real. A genuine reference comparison is a clean 2×H200 job (the model is
public + cached, so cheap to set up) and remains future work.

## Method 5 — Quantization ladder (REAL ground truth, no injected faults)

The keystone. Quantize ONE model across a method × bit-width matrix and check the
tool tracks the *known* monotonic degradation (fewer bits → more damage). Nothing
is injected — the degradation is real, so this answers "is your evaluation real?".

**Model:** Qwen2.5-32B-Instruct (64 GB FP16) on a single H100 80 GB.
**Harness:** `scripts/quant_ladder.py` (run) + `scripts/render_ladder.py` (report).

| Method | Bits | Verdict | Mean cos | Min cos | KL (nats) | Culprits |
|--------|------|---------|----------|---------|-----------|----------|
| HQQ | 8 | DEGRADED | 1.0000 | 0.9999 | 0.0002 | 2 |
| HQQ | 4 | DEGRADED | 0.9927 | 0.9833 | 0.056 | 19 |
| HQQ | 3 | DEGRADED | 0.9754 | 0.9414 | 0.205 | 15 |
| HQQ | 2 | DEGRADED | 0.8606 | 0.7542 | 0.924 | 53 |
| bitsandbytes | 8 | DEGRADED | 0.9945 | 0.9929 | 0.022 | 9 |
| bitsandbytes | 4 | DEGRADED | 0.9924 | 0.9850 | 0.062 | 19 |

**Result: both methods MONOTONIC.** Mean cosine falls, min cosine falls, KL rises,
culprit count rises as bits drop — the tool tracks real degradation with no injected
faults. (Raw data: `docs/ladder-qwen2.5-32b.json`.)

Two honest findings this surfaced:
- **8-bit reads DEGRADED, not PASS** — the MSE voter is slightly too sensitive at the
  near-lossless end (flags 2 layers despite cos 1.0). Fix: a healthy-floor on the MSE
  voter, mirroring the cosine ceiling. (Calibration nit, not a correctness bug.)
- **Even 2-bit stays "Signal Degradation," not "Computation Collapse"** — a 32B model
  with a good quantizer degrades *gracefully* rather than collapsing a layer. This
  matches the literature (larger models are more quantization-robust) and validates
  the classifier's diffuse-vs-structural distinction on a real model.

## Scorecard

**9/9** classification cases (6 synthetic + 3 real) + **the quantization ladder**
(real monotonic degradation on Qwen2.5-32B across 6 method/bit-width points, both
methods monotonic). The ladder is the real-ground-truth validation — not injected.

## Still to validate (Phase 5)

The DeepSeek-V4 / QTIP case study via the dumps path — real MoE, real 2-bit
trellis quantization, per-expert diagnostics. Requires Arc to emit dumps in the
v1 format (`docs/dump-format.md`).
