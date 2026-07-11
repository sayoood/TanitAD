# KNOWLEDGE_BASE — Architecture & Inference

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-11] [repo/measured] **D1 probe-capacity ladder + isotropy linkage** (step-6500 ckpt, 12 comma2k19
  val eps, 408 latents dim-2048, 4060, 36 s, $0). Advances the loop's live D1 discriminator (`0284a5c`).
  **(1) The raw-2048 probe is underdetermined (D≫N=204): `linear_ols` overfits 24.35 m vs ridge 10.31 m @1s
  — a 2.4× swing from REGULARISATION, not capacity → the loop's 12-ep `d1_probe_capacity.py` sits in the
  same regime. Fix: PCA-reduce to `active_k` (train-only basis) → N>D; `linear_ols` recovers to 10.77.**
  **(2) NO nonlinear advantage at 6500 — gap = best_linear−best_MLP is NEGATIVE everywhere (−15 %/−13 % raw,
  −15 %/−39 % PCA-19); best probe is a linear ridge (8.84 @1s vs 18.36 zero-motion floor → top-19 subspace
  carries ~52 % of the 1-s motion, linearly).** Disfavours the "less-linear" branch, BUT directional not
  decision-grade (MLP data-starved at n=204). **(3) Pre-registered prediction REFUTED (P8): anisotropy
  (`iso_active`=0.269, cross-checks the 07-10 instrument's 0.250) does NOT tax the linear trajectory probe —
  ridge already absorbs covariance anisotropy. The 2605.26379 orthogonality precondition governs latent-
  PLANNING regret (D4–D6), not external-target probe recoverability (D1).** Actionable: patch d1 script to
  PCA-reduce + ≥50 eps before any info-lost verdict; do NOT motivate a decode-capacity change from D1 (no
  gate, D-004) — impact: H3 / H5 / D1 / D-004 —
  `../Research/2026-07-11-d1-probe-capacity-ladder-and-isotropy-linkage.md` + `../Implementation/incoming/2026-07-11-d1-probe-capacity-ladder/`
- [2026-07-11] [arXiv 2605.09241] **Sub-JEPA — Subspace Gaussian Regularization** regularises the
  principal-variance subspace toward Gaussian instead of all embedding dims uniformly. Candidate REMEDY for
  our `iso_active` shortfall (full-dim SIGReg wastes budget on the dead 2027-dim tail; global iso 2e-8): a
  subspace variant targets the active-k directions the planner uses → could raise `iso_ratio_active` toward
  the 2605.26379 optimal-planning precondition. The gate it moves is the ISOTROPY admissibility gate, not D1.
  Abstract-level only (exact numbers not PDF-extractable; labelled ESTIMATE, G-AI2). Design-note-first,
  D-018 escalate before trained-config — impact: H3 / D-021 (isotropy remedy) — https://arxiv.org/pdf/2605.09241
- [2026-07-11] [arXiv 2606.09646] **Layerwise probing of video foundation models** (U. Amsterdam, Jun-2026)
  names the exact discriminator: LINEAR probe = "task-relevant info directly accessible"; MLP probe (GELU +
  LayerNorm hidden) = "present but not linearly encoded". V-JEPA strongest, esp. with temporal-dynamics
  probes; physics info weakest early / strongest at intermediate-late depth; frame-order disruption hurts.
  Methodological grounding for our D1 probe-capacity ladder (linear vs MLP = info-lost vs less-linear) —
  impact: D1 / H5 (probe protocol) — https://arxiv.org/abs/2606.09646
- [2026-07-11] [repo] **D-027 accepted: rollout_k=4 adopted for all post-30k training** (Sayed, `ff6a409`).
  Decision evidence = matched-compute K=4 arm (`859caa8`): 1-step `imag_rel` 8.13→1.03, recursive 14.5→1.11.
  This DECISION-GRADES my backlog P0 #2b (the operative-scale K sweep) — the K-step thread I opened
  (2026-07-09, −64 % @1-step, "K must match horizon") is settled at K=4. P0 #2b/#2c retire — impact: H5 / D-027
- [2026-07-09] [repo/measured] **K-step rollout bake-off — first measured arm** (backlog P0 #2; matched
  compute, 4060, 2×2000 steps, real comma2k19, 11.74 M reduced-but-real probe). K=2 vs K=1, OFAT-verified
  (`lever_diff==["train.rollout_k"]`). **(1) rollout ≈ free: +0.5 % wall-clock (749.4 vs 745.4 s), 0 extra
  params.** **(2) D2 P1 direction-acc SATURATED at 1.0 both arms** (probe fit ≈0.9999) → the backlog
  falsifier metric is ceiling-limited, NOT discriminative. **(3) discriminative signal = `imag_rel`: K=2
  cuts 1-step latent-pred error vs persistence 2.914→1.049 (−64 %) but does NOT help the 4-step horizon
  (I4 1.451→1.645, worse)** → K must cover the decode horizon (K≈4 for the 2-s D3 claim; 2512.24497 Pareto).
  D1 FAIL + D3 BLOCKED (I4>1) both → **no decision-grade claim** (D-004); decision-grade = operative-scale
  K∈{1,2,4} sweep from pod2 step-8k. No collapse (erank ~40 both) — impact: H5 / WP3 / D-018 —
  `../Research/2026-07-09-kstep-rollout-bakeoff-and-lejepa-identifiability.md` + `../Implementation/kstep_bakeoff_probe/`
- [2026-07-09] [arXiv 2605.26379] **When Does LeJEPA Learn a World Model? (LeCun/Klindt)** — LeJEPA
  (alignment + Gaussian reg = our SIGReg) **linearly & orthogonally identifies** world latents under
  stationary additive-noise transitions; **Gaussian is the UNIQUE prior** for which it holds; **"linear,
  orthogonal identifiability enables OPTIMAL latent-space planning"**; degrades gracefully; non-Gaussian
  breaks it. Translations: (a) grounds `p0-spectral-sizing`'s LINEAR transition proxy (why fit R²≈0.99–0.999)
  → D-021 sizing-to-the-knee is principled, not convenient; (b) upgrades H3 SIGReg-only anti-collapse from
  "empirically stable" (LeWM) to "provably identifiable → optimal planning" — the Epps–Pulley isotropic-
  Gaussian target IS the theorem's unique-prior condition; (c) named experiment: add an **orthogonality
  instrument** to `spectral.py` (readout covariance ~isotropic?) — makes the theorem falsifiable on our
  ckpt — impact: H3 / D-021 / D-008 — https://arxiv.org/abs/2605.26379
- [2026-07-09] [arXiv 2605.08567] ACWM action-conditioning ablation: **cross-attention beats AdaLN only for
  HIGH-dim action spaces; NO benefit for LOW-dim actions**; AdaLN (summed timestep+compressed-action
  modulation) is the standard low-cost injection. Our action space is **2-D (steer, accel) = low-dim** →
  keep AdaLN as the `adaln_conditioning` target (AdaLN>FiLM still holds) but **expected Δ is bounded**, and
  there is no reason to reach for cross-attention. Lowers my prior that the AdaLN lever clears the +2 %
  smoke bar (backlog P1 #3) — impact: H1 / H12 (adaln_conditioning planned lever) — https://arxiv.org/abs/2605.08567
- [2026-07-08] [repo/measured] **Spectral-sizing run #1 on a TRAINED ckpt (step-6500, 4060, 24 val eps,
  7,176 pairs):** fit R²=0.990 (linear proxy valid), operator effective rank ≈43, energy knee=31, k*=21 →
  **OVER-PROVISIONED**: the 2048 readout ≫ the ~tens-dim task-relevant transition rank. Rank still climbing
  (35→43 over steps 3k→6.5k) → re-measure at final Stage-0 ckpt; decision-grade evidence for **D-021** (keep
  2048 for now, keep measuring). No change executed (D-004/D-018) — impact: H3 / D-008 / D-021 —
  `Research/2026-07-08-spectral_step6500.json`
- [2026-07-08] [repo] Bake-off harness (WP3, backlog #2): OFAT one-lever-per-run driver — every variant is
  the base config with EXACTLY one field flipped (verified by a recursive dataclass `lever_diff`; a lever
  that lies about its fields raises), scored through the D1–D3 gate runner so a BLOCKED gate yields NO
  claim; multi-seed mean±95% CI; measured-params column (FLOPs/latency deferred to backlog #5, never mixed —
  G-AI2). 8 config-native runnable levers + 4 `planned` levers (AdaLN, RoPE, K-step, tactical-MoE-on-σ) that
  carry gate+hypothesis+WP pointer and refuse to run until model code lands. 16 tests; end-to-end on real
  smoke `WorldModel` → D3 BLOCKED / D2 MIXED on untrained latents (doctrine fires). Decision-grade sweep
  awaits trained ckpt — impact: WP3 / D-004 / H4·H5·H1·H15 — `../Implementation/incoming/2026-07-08-bakeoff-harness/`
- [2026-07-08] [arXiv 2606.31232] Delta-JEPA: reconstruction-free action-conditioned WM with a Latent
  Difference Action Decoder — reconstructs the executed action from the LATENT DISPLACEMENT between
  consecutive observations (= our A4 residual + A5 inverse-dynamics, arrived at independently); improves
  planning over JEPA/repr-learning baselines on 4 continuous-control tasks. Secondary summaries: AdaLN
  action injection, 6-layer causal predictor (not in abstract — flagged) — impact: H4 / H5 (residual +
  change-weight levers) — https://arxiv.org/abs/2606.31232v1
- [2026-07-08] [lit] Action-conditioned latent-predictor conditioning triangulated across 3 sources
  (2512.24497 AdaLN>FiLM +RoPE best; Delta-JEPA; OmniDreams 2606.03159 RoPE+AdaLN) + K-step rollout Pareto
  ≈ K=4 (2nd data point to 2512.24497's 6-step real). All entered as `planned` bake-off levers; each is a
  D-018 Tactic → escalate before touching the trained config — impact: H1 / H2 / H5 — see bake-off note
- [2026-07-08] [arXiv 2606.09311] FF-JEPA: hierarchical latent planners decompose long-horizon planning to
  beat compounding error + flat-CEM cost — Phase-1 comparison target; reinforces hierarchy (H1) as the
  compounding-error answer over flat rollout — impact: H1 (Phase-1 watch) — https://arxiv.org/html/2606.09311v1
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
