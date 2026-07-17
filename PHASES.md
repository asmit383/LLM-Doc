# quant-doctor — Project Phases

The missing QA layer between "I quantized a model" and "is it actually still good, and if not, where and how do I fix it."

**Context:** Final-year project. Graded on novelty, technical depth, and demonstrated validation — not revenue. Validation leverages the Arc stack (real V4/QTIP failure cases) plus deliberately-injected bugs.

**Core thesis:** No shipped tool diagnoses *why* a quantized LLM is broken. Quantizers exist (dozens); benchmarkers exist (LLMC); nothing classifies failure modes, attributes damage to layers, or prescribes a fix. The research community just named the failure modes in 2026 (ACL 2026: "From Signal Degradation to Computation Collapse") but left productization as future work.

---

## Validation Strategy (how we prove it works — this is what earns the grade)

1. **Injected-bug detection** — take a healthy model, deliberately break the quantization, show the tool localizes the damage to the correct layer/component.
2. **Both failure modes** — one Signal Degradation case, one Computation Collapse case; show the classifier calls each correctly.
3. **Real case study (Arc/V4)** — retroactively detect the MXFP4-as-INT4 expert bug and the 2-bit tail collapse using activation dumps from the Arc stack.

---

## Status Legend
`TODO` · `WIP` · `DONE` · `BLOCKED` · `STRETCH`

---

## Phase 0 — Scaffold  `DONE`
Get a runnable package + CLI skeleton on the board before writing real logic.

- [x] `pyproject.toml` with deps (typer, torch, transformers, safetensors, rich, numpy)
- [x] Package structure `src/quant_doctor/`
- [x] `quant-doctor` CLI entry point — `diagnose` command that parses args and prints a plan
- [x] `--help` works end to end
- [x] README with quickstart

**Exit criteria:** `quant-doctor diagnose --help` runs. ✅

---

## Phase 1 — Core Diagnostic (MVP)  `WIP`
The minimum that answers "is it broken, and where?" End-to-end on the dumps path
(the V4/QTIP target). Live-HF capture deferred — same engine, different source.

- [x] Dump format spec (`docs/dump-format.md`) — the Arc↔quant-doctor contract
- [x] `dumps.py` — load + validate paired activation dumps
- [x] `metrics/statistical.py` — per-layer cosine, MSE, output KL divergence
- [x] `diagnosis.py` — data structures + threshold-based verdict
- [x] `engine.py` — `diagnose_pair()` builds a Diagnosis from two dumps
- [x] `report.py` — CLI table with layer heatmap + verdict (table + JSON)
- [x] Synthetic dump generator (`scripts/make_synthetic_dumps.py`) — 4 ground-truth cases
- [x] Verified end-to-end: healthy→PASS, degradation→DEGRADED, collapse→BROKEN (culprit localized), format_bug→BROKEN
- [x] `loader.py` + `capture.py` — live HF forward-hook capture (sequential load, self-quantize via bnb4/8)
- [x] Validated on real GPU (H200): Qwen2.5-1.5B fp16 vs bnb4 → DEGRADED, mean cosine 0.99
- [ ] PPL delta (needs eval labels — nice-to-have)

**Exit criteria:** `quant-doctor diagnose-dumps` localizes injected damage ✅ (dumps path);
`quant-doctor diagnose --ref M --quantize bnb4` runs live on a real model ✅ (HF path).

**Real-model finding (H200, Qwen2.5-1.5B bnb4):** quantization error concentrates in the
final layers (26–27): MSE 20–45× the mid-stack, cosine at run minimum. The tool
automatically surfaces the "keep late layers / lm_head at higher precision" heuristic.

---

## Phase 2 — Failure-Mode Classifier  `DONE`
Go from "it's broken" to "*why*, and what kind of broken."

- [x] `metrics/interpretability.py` — error-subspace top-1 concentration (structured vs diffuse error)
- [x] `metrics/propagation.py` — depth trend, cliff gap, clean-prefix length
- [x] `classifier.py` — rule-based decision tree: Healthy / Signal Degradation / Computation Collapse / Format Bug / Generic
- [x] Report surfaces failure mode + signature evidence + repair prescription
- [ ] attention entropy / FFN sign-flip / logit-lens (need extended capture; deferred to Phase 2.5)
- [ ] early-EOS detection from logits (needs eos_token_id in manifest; deferred)

**Exit criteria:** classifier correctly labels Signal Degradation and Computation Collapse. ✅
**Validation:** 4/4 on synthetic ground-truth (healthy/degradation/collapse/format_bug)
+ real H200 run (Qwen2.5-1.5B bnb4 → Signal Degradation, correct). 5/5.

Note: classifier keys off metrics computable from hidden states + logits alone, so it
works identically on the live and dumps paths (no attention weights required).

---

## Phase 3 — Validation & Case Studies  `TODO`
The section that earns the top grade. Prove it works.

- [ ] Bug-injection harness — corrupt a layer / mis-decode a format / force early-EOS
- [ ] Case study A: injected Signal Degradation → tool localizes + classifies
- [ ] Case study B: injected Computation Collapse → tool localizes + classifies
- [ ] Case study C (Arc/V4): `--from-dumps` path; retroactively detect MXFP4-as-INT4 expert bug
- [ ] Write-up: methodology, results, limitations

**Exit criteria:** three reproducible case studies documented with tool output.

---

## Phase 4 — Recipe Generation  `STRETCH`
Close the loop: not just "what's broken" but "here's the fix."

- [ ] `recipe.py` — mixed-precision config from culprit layers + failure mode
- [ ] VRAM-delta estimate
- [ ] Export GPTQ/AWQ/llama.cpp-compatible config
- [ ] `verify` command — re-diagnose after applying recipe

**Exit criteria:** tool emits a mixed-precision recipe that measurably improves a broken quant.

---

## Phase 5 — MoE / Arc Deep Integration  `STRETCH`
The differentiator nobody else has.

- [ ] Per-expert cosine (33k+ experts for V4)
- [ ] Router divergence + expert load skew
- [ ] Arc `--dump-activations` feature (Rust side)

**Exit criteria:** per-expert heatmap that isolates the V4 layer-2 expert blowup.

---

## Design Decisions (locked)

- **Language:** Python (ML ecosystem, pip-installable)
- **CLI:** Typer
- **Loading:** HF `transformers` + `auto-gptq`/`autoawq`/`bitsandbytes` for the zero-grind formats
- **Interpretability:** built from scratch (3–4 primitives), not TransformerLens (too heavy a dep)
- **Custom stacks (QTIP/Arc):** `--from-dumps` activation-file path, not live loading
- **Reference for V4:** the FP4/FP8 pre-ISQ weights (no FP16 exists), compared against post-QTIP
- **Classifier:** rule-based first (clean boundaries from ACL 2026), ML classifier only if time allows

## Non-Goals (explicitly out of scope)
- SaaS / dashboards / web UI
- Fine-tune validation (different problem — divergence is intended, logic inverts)
- Building our own quantizer (we diagnose; existing tools quantize)
- "Any model, any quant" universality — scoped to transformer-family + hookable formats
