# quant-doctor

**The missing QA layer between "I quantized a model" and "is it actually still good — and if not, where and how do I fix it?"**

Everyone ships quantizers (GPTQ, AWQ, QTIP, llama.cpp — dozens). Nobody ships the thing that tells you your quantized model is *silently broken* and *why*. `quant-doctor` is that tool: point it at a reference model and its quantized version, and it tells you:

1. **Is it broken?** — perplexity / KL-divergence delta vs a pass threshold
2. **Where?** — a layer-wise error heatmap flagging the culprits
3. **What kind of broken?** — Signal Degradation (recoverable) vs Computation Collapse (needs retraining)
4. **How to fix it?** *(stretch)* — a mixed-precision recipe: "keep `lm_head` + layers {2, 47} at 4-bit; rest fine at 2-bit"

Built for the person holding the quantizer — fine-tuners, quantization researchers, infra teams — not the person downloading a pre-made quant.

## Status

Early development. See [`PHASES.md`](PHASES.md) for the roadmap and [`quant-doctor-flowchart.md`](quant-doctor-flowchart.md) for the system design.

- **Phase 0** — scaffold (CLI runs) — *in progress*
- **Phase 1** — core diagnostic: load, capture, statistical metrics, heatmap
- **Phase 2** — failure-mode classifier
- **Phase 3** — validation case studies

## Install (dev)

```bash
pip install -e ".[dev]"
# plus the backend you use, e.g.:
pip install -e ".[gptq]"
```

## Usage

```bash
# Diagnose a quantized model against its reference
quant-doctor diagnose --ref meta-llama/Llama-3-8B --target ./llama-3-8b-gptq-2bit

# For custom stacks (QTIP / Arc / MoE) — compare pre-dumped activations
quant-doctor diagnose-dumps --ref-dir activations_ref/ --target-dir activations_quant/
```

## Why this is defensible

The failure modes were only named by the research community in 2026 ("From Signal
Degradation to Computation Collapse", ACL 2026 Findings) — the intellectual
foundation exists, but no tool productizes it. The diagnostic *engine* is
format-agnostic (it operates on paired activations); the moat is calibrated
thresholds and per-architecture failure signatures, including MoE per-expert
diagnostics that no published tool offers.

## License

MIT
