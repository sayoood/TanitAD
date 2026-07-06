# KNOWLEDGE_BASE — Architecture & Inference

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-14] [arXiv 2512.24497] JEPA-WM planning ablation: faithful unroll ≠ planning success —
  **decode/probe quality is necessary but NOT sufficient**. So D1–D3 are instrument gates; closed-loop
  D4–D6 arbitrate. Also: AdaLN action-cond wins (our FiLM confirmed), +RoPE best; multistep rollout loss
  = data-aug vs compounding error (2-step sim / 6-step real); ViT-L enc + depth-12 pred optimal for
  complex real dynamics (validates base250); DINO > V-JEPA encoders (H4 arm-B data point) — impact:
  D1–D3 / H1 / H4 / H5 — https://arxiv.org/abs/2512.24497
- [2026-07-14] [Meta V-JEPA 2 AC] 300 M block-causal action-conditioned latent WM predicting next-frame
  representation — same family & envelope as our 261 M operative path (D-008 scale sanity) — impact:
  H1 — https://arxiv.org/html/2506.09985v1
- [2026-07-14] [LeWM] stable end-to-end action-conditioned JEPA, 2 loss terms, no EMA/stop-grad, no
  collapse — supports our LeJEPA/SIGReg-only anti-collapse (H3, D-003); field converging on
  regularize-don't-stopgrad — impact: H3 — https://medium.com/@adnanmasood/leworldmodel-and-the-case-for-stable-latent-world-models-0e4c33ca0f3c
- [2026-07-14] [DriveMoE CVPR2026 / GEMINUS] Vision-MoE routes camera VIEWS + skill Action-MoE, on a
  LEARNED scene router. Our differentiator (H15↔H2): route the tactical/sensor MoE on ImaginationField
  epistemic σ (gate a sensor/expert only where imagination uncertainty is low) — principled, not
  black-box — impact: H2 / H8 / WP4 — https://arxiv.org/abs/2505.16278 · https://arxiv.org/abs/2507.14456
- [2026-07-14] [SqueezeBits / ModelOpt] Native TensorRT ViT INT8 is a trap (MHA/RoPE block kernel
  fusion). Use OwLite (30 % latency, 0.7 % acc drop) / DFQ-ViT / ModelOpt PTQ instead. Batch-free
  LayerNorm (our I2 choice) enables TRT-LLM fused reduce-norm on the batch-1 streaming path → keep
  LayerNorm-only + static [6,256,256] input — impact: H5 / CNCE / deploy (ESTIMATE, no measured latency)
  — https://blog.squeezebits.com/how-to-quantize-transformerbased-model-for-tensorrt-deployment-55802
- [2026-07-14] [ReflectDrive-2] masked-discrete-diffusion trajectory planners allow revision but are
  heavier than our discrete tactical vocabulary + imagine-and-select (K batched passes, ms, no CEM/
  diffusion) → Phase-1 comparison target, not adoption — impact: H5 / WP4 — https://arxiv.org/html/2605.04647v1
- [2026-07-14] [repo/theory 2606.27014] `p0-spectral-sizing` tool (backlog #0, L2): fits the
  action-conditioned transition operator (z_t,a_t)→z_{t+1}, reports σ decay / entropy effective-rank
  (offline twin of live erank) / 99%-energy knee / trade-off-optimal k* / spectral tail, and an
  OVER-/UNDER-provisioning verdict vs the 2048 readout (D-008). Recovers a known rank-5-in-32 spectrum;
  real sizing awaits a TRAINED comma2k19 checkpoint (untrained latents degenerate, P8). 8 tests —
  impact: H3 / WP3 / D-008 — `../Implementation/incoming/2026-07-14-spectral-sizing-p0/`
- [2026-07-14] [repo] D1–D3 gate runner intake pkg: instrument-doctrine gating (BLOCKED vs FAIL),
  ADE/FDE, I3 episode split, D1 vs-global-pool & D3 probe_real/imag ablations, extra_metrics seam for
  Thursday's suite; 13 tests — impact: WP6 / D-004 —
  `../Implementation/incoming/2026-07-14-gate-runner-d1-d3/`
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`
