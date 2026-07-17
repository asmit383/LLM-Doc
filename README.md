<div align="center">

# 🩺 quant-doctor

### The missing QA layer for quantized LLMs.

**You quantized a model. Is it still good? If not — *where* did it break, *why*, and *how* do you fix it?**
`quant-doctor` answers all four. No existing tool does.

</div>

---

Everyone ships **quantizers** — GPTQ, AWQ, bitsandbytes, QTIP, llama.cpp, dozens more.
Nobody ships the thing that tells you your quantized model is **silently broken** and **why**.

The research community only *named* these failure modes in 2026
([*From Signal Degradation to Computation Collapse*](https://arxiv.org/abs/2604.19884), ACL 2026 Findings).
The intellectual foundation exists. The product didn't. This is the product.

> Built for the person **holding the quantizer** — fine-tuners, quantization researchers,
> infra teams — not the person downloading a pre-made quant. If you quantize your own
> model, you have no safety net. This is it.

---

## What it tells you

| Question | Answer it gives |
|----------|-----------------|
| **1. Is it broken?** | A verdict — `PASS` / `DEGRADED` / `BROKEN` — from perplexity-style KL divergence + per-layer cosine |
| **2. Where?** | A layer-by-layer damage heatmap, culprit layers flagged |
| **3. What *kind* of broken?** | **Signal Degradation** (cheap fix) vs **Computation Collapse** (needs retraining) vs **Format Bug** (your dequant is wrong) |
| **4. How do I fix it?** | A concrete mixed-precision recipe + its VRAM cost — *"keep lm_head + layers {11,12,13} at 16-bit"* |

---

## See it work

Take a healthy model, quantize it, and inject a fault at layer 12 — the tool localizes
the root cause and prescribes the fix. **Real run, real H200, real Qwen2.5-1.5B:**

```console
$ quant-doctor diagnose --ref Qwen/Qwen2.5-1.5B --quantize bnb4 --inject-collapse 12

╭─ VERDICT: BROKEN  (16 culprit layers) ─╮
│ model      : Qwen/Qwen2.5-1.5B [bnb4]  │
│ mean cosine: 0.4854   min: -0.0407     │
│ output KL  : 14.66 nats                │
╰────────────────────────────────────────╯
          Layer Health
  layer_11 ████████████████████ 0.9911
  layer_12 ░░░░░░░░░░░░░░░░░░░░ -0.0025  ← CRITICAL   ← injected fault
  layer_13 ██░░░░░░░░░░░░░░░░░░  0.0937  ← CRITICAL
  ...                                                 ← real error propagation
  layer_20 ░░░░░░░░░░░░░░░░░░░░ -0.0407  ← CRITICAL

╭─ FAILURE MODE: Computation Collapse ───────────────────────────────╮
│ Signature:                                                         │
│   • collapse onset at layer_12 (cosine 0.000) after 12 clean layers│
│   • worst layer layer_20 — downstream fallout, not the root cause  │
│   • error residual at onset is structured (low-rank), top-1 = 1.00 │
╰────────────────────────────────────────────────────────────────────╯

╭─ RECIPE — mixed precision ─────────────────────────────────────────╮
│   base: 4-bit                                                      │
│   keep at higher precision:                                        │
│     lm_head        → 16-bit   (errors map straight to tokens)      │
│     model.layers.12 → 16-bit  (collapse onset)                     │
│   est. VRAM delta: +0.46 GB    confidence: LOW (consider fine-tune)│
╰────────────────────────────────────────────────────────────────────╯
```

Notice what it got right: the **onset** (layer 12) is named as the root cause, while the
numerically-worst layer (20) is correctly labeled *downstream fallout*. That distinction —
and the structured-vs-diffuse error signature — is what tells you *where to actually apply the fix*.

---

## Quickstart

```bash
git clone https://github.com/asmit383/LLM-Doc.git && cd LLM-Doc
pip install -e ".[dev]"
pip install bitsandbytes        # or your quantization backend of choice
```

```bash
# Self-quantizer flow — you have one FP model, quantize it yourself, check your work
quant-doctor diagnose --ref Qwen/Qwen2.5-1.5B --quantize bnb4

# Compare against a checkpoint you already quantized (GPTQ/AWQ/…)
quant-doctor diagnose --ref meta-llama/Llama-3-8B --target ./llama-3-8b-gptq-2bit --quantize none

# Custom / huge / MoE stacks (QTIP, Arc, 200B+ models) — diagnose from activation dumps
quant-doctor diagnose-dumps --ref-dir activations_ref/ --target-dir activations_quant/
```

---

## How it works

```
  ref model ─┐                                                    ┌─ verdict
             ├─▶ capture ─▶ metrics ─▶ classify ─▶ recipe ─▶ report┤─ heatmap
  quant model┘   (paired    (cosine    (failure    (mixed-        └─ failure mode
                activations)  KL, MSE,   mode)       precision)      + prescription
                             subspace)
```

The engine operates on **paired activations** — the same input run through both models,
compared layer by layer. This one design choice makes it **format-agnostic**: whether the
activations come from live HF forward-hooks or `.safetensors` dumps written by a custom
Rust stack, the metrics, classifier, and recipe code are identical.

- **Sequential capture** → peak memory is `max(ref, quant)`, never the sum, so a single
  GPU can diagnose a model whose fp + quant copies wouldn't co-reside.
- **No FP16 reference needed** → compares whatever you *have* (e.g. FP4 weights) against the
  quantized output. Right for 236B MoE models that have no full-precision checkpoint.

### The failure modes

| | **Signal Degradation** | **Computation Collapse** | **Format Bug** |
|---|---|---|---|
| What broke | precision (noise) | function (a critical layer) | the decoder itself |
| Shape | gradual decline with depth | sharp cliff after a clean prefix | uniform, from layer 0 |
| Error | diffuse (high-rank) | structured (low-rank) | total |
| Fix | keep a few layers higher-bit — **cheap** | fine-tune / reconstruct — **expensive** | fix dequant, *not* bits |
| Typical | 4-bit | 2-bit | e.g. MXFP4-as-INT4 |

Telling these apart *before* you spend hours on the wrong fix is the entire point.

---

## Validation

Two methods, **6/6 correct** (verdict *and* failure mode):

- **Synthetic ground-truth** — controlled injected damage with known labels, asserted by `pytest` (20 tests).
- **Real fault injection** — real models on an H200, real error propagation on GPU.

```bash
pytest        # 20 passed
```

See [`docs/validation.md`](docs/validation.md) for the full case studies.

---

## Project layout

```
src/quant_doctor/
├── cli.py            # Typer CLI: diagnose, diagnose-dumps
├── loader.py         # sequential HF load + self-quantize (bnb 4/8-bit)
├── capture.py        # forward-hook activation capture (+ fault injection)
├── dumps.py          # load/validate activation dumps (custom stacks)
├── engine.py         # diagnose_pair() — the format-agnostic core
├── classifier.py     # rule-based failure-mode decision tree
├── recipe.py         # mixed-precision recipe generation
├── report.py         # rich CLI heatmap + panels
├── synthetic.py      # ground-truth bug-injection oracle
└── metrics/          # statistical · interpretability · propagation
docs/
├── dump-format.md    # the activation-dump contract
└── validation.md     # methodology + results
```

Roadmap and status live in [`PHASES.md`](PHASES.md). Design in [`quant-doctor-flowchart.md`](quant-doctor-flowchart.md).

---

## Why it's defensible

- **The gap is real and confirmed** — quantizers apply quantization; benchmarkers (LLMC)
  report aggregate perplexity. *Nothing* classifies failure modes, attributes damage to
  layers, or prescribes a mixed-precision fix.
- **The moat is calibration, not code** — a weekend wrapper can compute cosine similarity.
  The value is in the calibrated thresholds and per-architecture failure signatures —
  including **MoE per-expert diagnostics** no published tool offers.
- **The engine is format-agnostic** — one diagnostic core serves GPTQ, AWQ, bitsandbytes,
  and custom trellis quantizers (QTIP) alike.

## Non-goals

Not a quantizer (we diagnose; existing tools quantize) · not a fine-tune validator
(divergence there is *intended* — the logic inverts) · not a dashboard/SaaS · not
"any model, any quant" — scoped to transformer-family models and hookable formats.

## License

MIT
