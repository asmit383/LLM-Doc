# quant-doctor — System Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER ENTRY POINT                            │
│                                                                     │
│  quant-doctor diagnose --ref <fp_model> --target <quant_model>      │
│                         [--eval-set <path>]                         │
│                         [--format auto|gptq|awq|bnb|gguf]           │
│                         [--output json|table|html]                  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: MODEL LOADING                         │
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │ Format        │    │ Load Reference │    │ Load Quantized       │  │
│  │ Auto-Detect   │───▶│ (FP16/BF16)   │    │ (GPTQ/AWQ/BNB/...)  │  │
│  │ from config   │    │               │    │                      │  │
│  └──────────────┘    └───────┬───────┘    └──────────┬───────────┘  │
│                              │                       │              │
│                              │   Both go through     │              │
│                              │   AutoModelForCausalLM│              │
│                              ▼                       ▼              │
│                     ┌─────────────────────────┐                     │
│                     │  Register Forward Hooks  │                    │
│                     │  on every decoder layer   │                    │
│                     │  + attention + FFN + head │                    │
│                     └────────────┬──────────────┘                    │
│                                  │                                  │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: ACTIVATION CAPTURE                       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Eval Set (small, ~256-1024 tokens)       │           │
│  │  Built-in: wikitext / gsm8k / code    OR  user BYO   │           │
│  └──────────────────────────┬───────────────────────────┘           │
│                             │                                       │
│              ┌──────────────┴──────────────┐                        │
│              ▼                             ▼                        │
│  ┌─────────────────────┐     ┌─────────────────────────┐           │
│  │  Forward Pass:       │     │  Forward Pass:           │           │
│  │  Reference Model     │     │  Quantized Model         │           │
│  │                      │     │                          │           │
│  │  Collect per-layer:  │     │  Collect per-layer:      │           │
│  │  • hidden states     │     │  • hidden states         │           │
│  │  • attn weights      │     │  • attn weights          │           │
│  │  • FFN outputs       │     │  • FFN outputs           │           │
│  │  • final logits      │     │  • final logits          │           │
│  └──────────┬──────────┘     └───────────┬──────────────┘           │
│             │                            │                          │
│             └─────────────┬──────────────┘                          │
│                           │                                         │
│                    Paired activation                                │
│                    tensors per layer                                │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: METRIC COMPUTATION                      │
│                                                                     │
│  For each layer i, compute:                                         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  STATISTICAL METRICS (Phase 1 — ship day one)               │    │
│  │                                                             │    │
│  │  • Cosine Similarity:  cos(h_ref[i], h_quant[i])           │    │
│  │  • MSE:                ||h_ref[i] - h_quant[i]||²          │    │
│  │  • Output KL Div:      KL(logits_ref || logits_quant)      │    │
│  │  • PPL Delta:          PPL_quant - PPL_ref                  │    │
│  │  • Error Growth:       Δ_m = ||h_ref[0:i] - h_quant[0:i]||²│    │
│  │                        (QEP's cross-layer propagation)      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  INTERPRETABILITY METRICS (Phase 2)                         │    │
│  │                                                             │    │
│  │  • Attention Entropy:  H(attn_weights) per head per layer   │    │
│  │  • FFN Sign Flip Rate: % of sign(FFN_ref) ≠ sign(FFN_quant)│    │
│  │  • Logit Lens Div:     project h[i] → vocab via unembed,   │    │
│  │                        compare ref vs quant ranked tokens   │    │
│  │  • Error Subspace:     top-k SVD of (h_ref - h_quant),     │    │
│  │                        check if error concentrates in few   │    │
│  │                        dimensions or spreads uniformly      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  MOE-SPECIFIC METRICS (Phase 2+, for MoE models)           │    │
│  │                                                             │    │
│  │  • Per-Expert Cosine:  cos(expert_ref[i][e], expert_q[i][e])│    │
│  │  • Router Divergence:  KL(routing_ref || routing_quant)     │    │
│  │  • Expert Load Skew:   do different experts get selected?   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│                           │                                         │
│                    Metric tensor per                                │
│                    layer/component                                  │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STAGE 4: FAILURE MODE CLASSIFICATION                  │
│                                                                     │
│  Input: all layer-wise metrics from Stage 3                         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    DECISION TREE                               │  │
│  │                                                               │  │
│  │  1. Is it broken?                                             │  │
│  │     PPL_delta > threshold  OR  mean(cosine) < threshold       │  │
│  │     OR  KL_div > threshold                                    │  │
│  │            │                                                  │  │
│  │     ┌──────┴──────┐                                           │  │
│  │     NO            YES                                         │  │
│  │     │              │                                          │  │
│  │     ▼              ▼                                          │  │
│  │  ┌──────┐   2. Where is the damage?                           │  │
│  │  │ PASS │      cosine[i] < layer_threshold                    │  │
│  │  │      │      → collect culprit_layers[]                     │  │
│  │  └──────┘         │                                           │  │
│  │                   ▼                                           │  │
│  │            3. What pattern?                                   │  │
│  │               │                                               │  │
│  │     ┌─────────┴──────────┐                                    │  │
│  │     │                    │                                    │  │
│  │     ▼                    ▼                                    │  │
│  │  SIGNAL                COMPUTATION                            │  │
│  │  DEGRADATION           COLLAPSE                               │  │
│  │                                                               │  │
│  │  Indicators:           Indicators:                            │  │
│  │  • Damage spread       • Damage concentrated in               │  │
│  │    uniformly across      1-3 layers (usually early)           │  │
│  │    many layers         • Attention entropy spikes             │  │
│  │  • Cosine degrades       (>3σ above mean)                     │  │
│  │    gradually           • FFN sign flip rate >15%              │  │
│  │    (0.95 → 0.90 → ..) • Logit lens: vocabulary               │  │
│  │  • Error grows           collapses at damaged layer           │  │
│  │    monotonically       • Error subspace: low-rank             │  │
│  │  • Computational         (few dimensions dominate)            │  │
│  │    patterns intact                                            │  │
│  │                                                               │  │
│  │  Repair:               Repair:                                │  │
│  │  CHEAP — training-     EXPENSIVE — needs fine-tuning          │  │
│  │  free mixed-precision  or full re-quantize with               │  │
│  │  recipe works          higher bits for collapsed layers       │  │
│  │  (64-81% recovery)     Training-free repair WILL NOT WORK     │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  ADDITIONAL FAILURE CLASSES (Phase 3+)                        │  │
│  │                                                               │  │
│  │  • FORMAT BUG:   cosine ≈ 0 for a component class            │  │
│  │                  (e.g., all experts dead = dequant bug)       │  │
│  │                  Your MXFP4-as-INT4 case                      │  │
│  │                                                               │  │
│  │  • EARLY EOS:    EOS probability mass > threshold             │  │
│  │                  in output logits (reference-free check)      │  │
│  │                                                               │  │
│  │  • ROUTER SKEW:  MoE routing diverges from reference          │  │
│  │                  (experts that should activate don't)          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│                           │                                         │
│                  Diagnosis object:                                  │
│                  {verdict, failure_mode,                            │
│                   culprit_layers, severity}                         │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STAGE 5: RECIPE GENERATION                           │
│                                                                     │
│  Input: culprit_layers + failure_mode + model architecture          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  if failure_mode == SIGNAL_DEGRADATION:                       │  │
│  │      recipe = {                                               │  │
│  │        default_bits: original_quant_bits,                     │  │
│  │        keep_higher: [                                         │  │
│  │          lm_head → 8-bit (always),                            │  │
│  │          embed_tokens → 8-bit (always),                       │  │
│  │          culprit_layers → original_bits * 2,                  │  │
│  │          first_N_layers → original_bits * 2  (if early        │  │
│  │                           damage pattern detected)            │  │
│  │        ],                                                     │  │
│  │        estimated_vram_delta: compute_delta(recipe, model),    │  │
│  │        confidence: "HIGH — training-free repair viable"       │  │
│  │      }                                                        │  │
│  │                                                               │  │
│  │  if failure_mode == COMPUTATION_COLLAPSE:                     │  │
│  │      recipe = {                                               │  │
│  │        recommendation: "FINE-TUNE or RE-QUANTIZE",            │  │
│  │        if re-quantize: {                                      │  │
│  │          collapsed_layers → max available bits,               │  │
│  │          rest → original_bits,                                │  │
│  │          estimated_vram_delta: ...,                            │  │
│  │          warning: "training-free recovery unlikely,           │  │
│  │                    expect partial improvement only"            │  │
│  │        },                                                     │  │
│  │        if fine-tune: {                                        │  │
│  │          target_layers: culprit_layers,                       │  │
│  │          suggested: "LoRA on collapsed layers + full          │  │
│  │                      precision head"                          │  │
│  │        }                                                      │  │
│  │      }                                                        │  │
│  │                                                               │  │
│  │  if failure_mode == FORMAT_BUG:                               │  │
│  │      recipe = {                                               │  │
│  │        recommendation: "FIX DEQUANT — not a quantization     │  │
│  │                         quality issue",                       │  │
│  │        evidence: "component class X has near-zero cosine     │  │
│  │                   uniformly — indicates decode bug, not       │  │
│  │                   precision loss"                             │  │
│  │      }                                                        │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Output formats:                                                    │
│  • JSON config (for re-quantization tools)                          │
│  • GPTQ/AWQ/llama.cpp compatible recipe                             │
│  • Human-readable prescription                                      │
│                                                                     │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 6: REPORT OUTPUT                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    CLI TABLE (default)                    │       │
│  │                                                          │       │
│  │  ╔══════════════════════════════════════════════════════╗ │       │
│  │  ║ quant-doctor v0.1.0 — Diagnosis Report              ║ │       │
│  │  ╠══════════════════════════════════════════════════════╣ │       │
│  │  ║                                                      ║ │       │
│  │  ║ VERDICT:  BROKEN — Computation Collapse              ║ │       │
│  │  ║ SEVERITY: CRITICAL (3 layers collapsed)              ║ │       │
│  │  ║                                                      ║ │       │
│  │  ║ Overall Metrics:                                     ║ │       │
│  │  ║   PPL delta:     +847.3 (6.2 → 853.5)              ║ │       │
│  │  ║   KL divergence: 4.73 nats                          ║ │       │
│  │  ║   Mean cosine:   0.891                              ║ │       │
│  │  ║                                                      ║ │       │
│  │  ║ Layer Heatmap:                                       ║ │       │
│  │  ║   Layer  0  ████████████████████  0.997              ║ │       │
│  │  ║   Layer  1  ████████████████████  0.994              ║ │       │
│  │  ║   Layer  2  ██████░░░░░░░░░░░░░░  0.623  ← CRITICAL ║ │       │
│  │  ║   Layer  3  ████████████████████  0.991              ║ │       │
│  │  ║   ...                                                ║ │       │
│  │  ║   lm_head  ███████████████████░  0.948              ║ │       │
│  │  ║                                                      ║ │       │
│  │  ║ Failure Signature:                                   ║ │       │
│  │  ║   • Layer 2 FFN sign flip rate: 47% (thresh: 15%)   ║ │       │
│  │  ║   • Layer 2 attn entropy: 3.2σ above mean           ║ │       │
│  │  ║   • Error subspace rank-1 dominant (93% variance)   ║ │       │
│  │  ║   → Computation Collapse: early-layer destruction   ║ │       │
│  │  ║                                                      ║ │       │
│  │  ║ Prescription:                                        ║ │       │
│  │  ║   Keep layers 0-3 + lm_head at 4-bit               ║ │       │
│  │  ║   VRAM increase: +1.2 GB (4.8 → 6.0 GB)            ║ │       │
│  │  ║   ⚠ Training-free recovery unlikely for collapsed   ║ │       │
│  │  ║     layers — consider LoRA fine-tune on layers 0-3  ║ │       │
│  │  ║                                                      ║ │       │
│  │  ║ Re-quantize: quant-doctor recipe generate \         ║ │       │
│  │  ║   --keep-layers 0-3,lm_head --target-bits 2         ║ │       │
│  │  ║ Verify:      quant-doctor verify --recipe recipe.json║ │       │
│  │  ╚══════════════════════════════════════════════════════╝ │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│  Also available: --output json  (machine-readable, CI integration)  │
│                  --output html  (shareable report with charts)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                     OPTIONAL VERIFY LOOP
═══════════════════════════════════════════════════════════════════════

  quant-doctor verify --recipe recipe.json --ref <model> --target <model>

  ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
  │ Load recipe   │────▶│ Re-quantize   │────▶│ Re-run diagnose  │
  │ (JSON config) │     │ with mixed    │     │ on new quant     │
  │               │     │ precision     │     │                  │
  └──────────────┘     └───────────────┘     └────────┬─────────┘
                                                       │
                                              ┌────────┴─────────┐
                                              │                  │
                                              ▼                  ▼
                                          ┌──────┐         ┌──────────┐
                                          │ PASS │         │ STILL    │
                                          │      │         │ BROKEN   │
                                          │ Ship │         │          │
                                          │ it   │         │ Iterate  │
                                          └──────┘         │ recipe   │
                                                           └──────────┘


═══════════════════════════════════════════════════════════════════════
                    SPECIAL PATH: ARC / MoE MODELS
═══════════════════════════════════════════════════════════════════════

  For models that don't load via HF (QTIP, Arc stack, custom):

  ┌──────────────────────────────────────────────────────────┐
  │  Arc Inference Engine (Rust)                              │
  │                                                          │
  │  1. --dump-activations flag on Arc serve/eval             │
  │  2. Forward pass dumps per-layer tensors to .safetensors │
  │  3. Run twice: pre-ISQ weights + post-QTIP weights       │
  │                                                          │
  │  Output: activations_ref/  activations_quant/            │
  │          ├── layer_00.safetensors                         │
  │          ├── layer_01.safetensors                         │
  │          ├── layer_02_expert_00.safetensors  (MoE)       │
  │          ├── layer_02_expert_01.safetensors               │
  │          └── ...                                          │
  └──────────────────────────┬───────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  quant-doctor diagnose --from-dumps                      │
  │    --ref-dir activations_ref/                            │
  │    --target-dir activations_quant/                       │
  │                                                          │
  │  Same Stage 3-6 pipeline — format agnostic               │
  │  Adds MoE-specific metrics:                              │
  │    • Per-expert cosine (33,792 experts for V4)           │
  │    • Router divergence                                    │
  │    • Expert load skew                                    │
  └──────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                        DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════

  models + eval_set
       │
       ▼
  ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌────────┐    ┌────────┐
  │  LOAD    │───▶│ CAPTURE  │───▶│  METRICS   │───▶│CLASSIFY│───▶│ RECIPE │
  │          │    │          │    │            │    │        │    │        │
  │ 2 models │    │ paired   │    │ per-layer  │    │ SD vs  │    │ mixed  │
  │ + hooks  │    │ activ-   │    │ cosine,mse │    │ CC vs  │    │ prec.  │
  │          │    │ ations   │    │ entropy,   │    │ format │    │ config │
  │          │    │          │    │ sign-flip  │    │ bug    │    │        │
  └──────────┘    └──────────┘    └────────────┘    └────────┘    └────────┘
                                                                       │
                                                                       ▼
                                                                  ┌────────┐
                                                                  │ REPORT │
                                                                  │        │
                                                                  │ table  │
                                                                  │ json   │
                                                                  │ html   │
                                                                  └────────┘
```
