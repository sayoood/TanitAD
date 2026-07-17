# KNOWLEDGE_BASE — Architecture & Inference

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-17] [repo/measured] **Blind K-step belief rollout DISSIPATES uncertainty + collapses to an
  attractor — the H11/D8 σ-trigger is anti-calibrated past 1 step.** Rolled the trained 1-step
  ImaginationField fully blind on real comma2k19 (step-6500 base250cam ckpt, 4060, 2 seeds). Hidden-cell
  centered-cosine fidelity **0.357 (k1) → 0.011 (k4, = chance) → negative**; it **beats the persistence
  baseline only at k=1** and falls below it from k≥2. Epistemic σ (hidden log-variance) **falls
  monotonically −7.79 → −8.55** (more confident as predictions become worthless = FALSE confidence);
  belief energy **collapses ~11× by k4** (0.101→0.008) while inter-sample cosine **rises 0.21→0.57** →
  every sample drifts to a **common attractor** (true-token energy flat ~0.33, so it is the model, not
  the scene). **The cause is the recursion, not the 1-step prediction:** *freezing* the k=1 imagination
  and holding it retains **~0.25 cosine FLAT across all 8 horizons** and beats persistence throughout.
  Confirms the 2026-07-15 UWM-JEPA risk and the **"Biased Dreams" (2604.25416) attractor prediction**,
  measured. Two D-018 responses (escalate, don't execute): (A) train multi-step belief rollout (0b build);
  (B) operate imagination **parallel-horizon (non-autoregressive)** from the last real obs — freeze-1
  shows (B) recovers fidelity for free (recommended default). **Cap the operative H15 self-monitor at
  1-step until a multi-step σ is validated.** No H15 status change (P8, pre-reset directional ckpt) —
  impact: H15 / H11 / D8 / D9 — `../Research/2026-07-17-blind-rollout-uncertainty-dissipation-and-readout-orthogonality.md`
  + `../Implementation/belief_rollout_diagnostic/`
- [2026-07-17] [repo/measured] **Readout orthogonality — VERIFIED the stranded 2026-07-10 instrument
  (not a duplicate); D-021 = subspace ID, NOT "optimal planning" on the pre-reset ckpt.** While drafting a
  3b instrument I found a theoretically-superior one already built 2026-07-10 but **never merged** (branch
  `worktree-agent-arch-inf-20260710`); **withdrew my draft**, ran the prior `orthogonality_report`
  unchanged on the step-6500 ckpt (n=2600 real states > S=2048) → **reproduces exactly:** active_k=23,
  **iso_ratio_active 0.254** (< 0.5), cond_active 218, rms_offdiag 0.424, cov_eff_rank 26, verdict
  **NOT-YET-ADMISSIBLE**. Key correction: **global** isotropy ~0 is over-provisioning **by design**, NOT a
  failure — the theorem-relevant read is the **active-subspace** isotropy (my draft lacked this). My
  independent global read (isotropy 0.000, off-diagonal 0.999) corroborates over-provisioning from the
  coordinate angle. Two instruments now agree the readout is over-provisioned (op-rank ≈43, repr active-rank
  ≈23–26 ≪ 2048) AND not yet orthogonal → latent *capacity* is not a D1 bottleneck. SIGReg slice-Gaussianity
  ≠ active-subspace isotropy (cond 218) → whitening lever (D-018 escalate). **Process: flagged the stranded
  07-10 instrument for orchestrator merge (3rd week unmerged).** — impact: H3 / D-021 / D-008 —
  `../Implementation/orthogonality_verification/`
- [2026-07-15] [repo/measured] **The flagship H15 imagination edge is NOT dark — the log is unfaithful**
  (resolves the 2026-07-14 program-report §8 WATCH `h15=0.0`). GPU diagnostic on the exact code path
  (`train_flagship4b.h15_loss` + `flagship4b_smoke_config`): imagination module **built** (22.06 M params,
  `h15.enabled=True`), gradient **reaches** it (L1 44.6) **and the encoder** (L1 36.7), fire rate **0.4525**
  ≈ `mask_prob` 0.5, mean loss when fired **0.611**. `h15=0.0` is a **logging artifact** — `log["h15"]`
  records the LAST accum micro, 0.0 whenever its gate didn't fire; **46.3 % of all log rows falsely read
  `h15=0.0` while the edge trained**, true idle only 6.3 % (theory (0.5)⁴=0.0625 ✓). **Do NOT change the
  trained config** (would chase a phantom, D-018 restraint). Fix = observability: an accumulation-window
  meter (`h15`/`h15_fired`/`h15_fire_frac`) shipped as intake (6✓); `h15_fire_frac→0` is now the *real*
  dark-edge alarm — impact: H15 / D9 / D8 — `../Research/2026-07-15-h15-imagination-edge-not-dark-and-belief-space-rollout.md`
  + `../Implementation/incoming/2026-07-15-h15-logging-fidelity/`
- [2026-07-15] [repo/measured] **H15 imagination edge is affordable per tick** (CNCE/Efficiency moat).
  Batch-1, RTX 4060, flagship4b scale (263.44 M total, imagination 22.06 M = 8.4 % of params); latency
  weight-value-invariant so untrained instantiation valid for timing. **fp32:** encode 7.67 ms /
  imagination 2.25 ms / predictor 5.52 ms → core tick 13.18 ms → **H15 = 17.0 % of core**. **fp16:** 4.26
  / 1.35 / 6.40 → 10.66 ms → **12.7 %**. So the A9 self-monitor adds ~1.3–2.2 ms/tick (~roughly its param
  share), only when engaged → no efficiency-moat regression. Honest: **fp16 makes the small predictor
  SLOWER** (6.40 vs 5.52; batch-1 launch/convert-bound, not tensor-core-bound) — the fp16 win is entirely
  in the ViT tower (matches Prod-Opt "TRT-fp16 the tower"); eager un-fused, so absolute ms is an upper
  bound, the fraction is robust — impact: H5 / CNCE / H15 —
  `../Implementation/h15_logging_diagnostic/results/2026-07-15-h15_latency.json`
- [2026-07-15] [arXiv 2605.25313] **UWM-JEPA — belief-space imagination WM.** Density-matrix latent +
  learned unitary predictor imagine multiple compatible hidden futures; *"the construction preserves the
  joint-state spectrum exactly during rollout, so the predictor itself cannot dissipate the represented
  uncertainty."* Numbers: hidden-velocity 5-step forward-sim **0.77 vs 0.53** (LSTM-JEPA); blind rollout
  loses **<10** probe-R² pts short-horizon vs **41/68** baselines. Translations for our H15 (sector-mask
  1-step + advection + per-cell σ): (a) **we train imagination at 1 step only** → multi-step belief rollout
  (where object-permanence/OOD pays off) is untrained → new backlog P0; (b) **our epistemic σ may dissipate
  over the operative K-step rollout** — if it collapses with horizon, the H11/D8 self-monitor trigger
  silently dies where anticipation matters; UWM-JEPA gives the mechanism (spectrum preservation) + the
  falsifier (blind-rollout R²-retention by horizon) — impact: H15 / H11 / D8 / D9 — https://arxiv.org/abs/2605.25313
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
