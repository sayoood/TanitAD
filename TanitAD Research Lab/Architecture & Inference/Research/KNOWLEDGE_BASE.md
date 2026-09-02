# KNOWLEDGE_BASE — Architecture & Inference

> **Curated, deduplicated, newest first.** Format:
> `- [YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`
>
> This is the **Architecture & Inference** agent's findings log. The router across all areas is
> [`../../KNOWLEDGE_BASE.md`](../../KNOWLEDGE_BASE.md).
>
> ⛔ **Three layers, and they are not interchangeable.**
> **PAPER** (`Paper/TANITAD_PAPER.md`) = the scientific account of the frontier work — derivations,
> argument, results in narrative form.
> **KNOWLEDGE_BASE** (this file) = what we learned, written for an agent about to make a decision.
> **LIBRARY** (`../../Library/`) = the evidence. Every `[PUBLISHED]` entry cites a **library key**,
> not only a URL — bank it with `python tools/kb_add.py <arxiv-id> --tag <topic> --cited-by <report>`.

- [2026-08-31] [PUBLISHED lib `2605.25313` §4.2/§4.3 FULL TEXT] ⭐⭐⭐ **A FOURTH CAUSE OF ACTION-DEAFNESS, NOT ON THE
  GATE'S LIST, AND IT MAKES P2(b) THE WRONG FIRST MOVE.** UWM-JEPA: *"Under a teacher-forced JEPA objective the
  target encoder observes the trajectory that already contains the action's effect, which admits an
  action-invariant solution"* — measured **‖H₁‖/‖H₀‖ ≈ 0.03**, restored to **1.00 ± 0.15** by targets built from a
  counterfactual action sequence. ⭐ **Our O5 is teacher-forced in exactly that sense — MEASURED in our source**
  (`train_v6_staged.py:48` *"against its layer's stop-grad/EMA target"*, `:1765` *"O5's teacher side"*,
  `:4214/:4221` all three `--o5-target` modes encode the REAL future) and our ratio is 0.00408–0.00595 (MM-E10).
  ⛔ **The mechanism is independent of channel provenance ⇒ a genuine CAN command is not predicted to fix it**, and
  it retro-explains MM-E11's fall (0.40× — O1 is another realised-future term). ⚠️ Their fix needs *"simulator-state
  access during training"*, which we lack — the simulator-free variant is a negative-action contrastive target —
  impact: **V7 gate P2 candidate table (escalated), MM-E11 reading** — `2026-08-31-command-conditioning-and-l3/RESULT.md` F1
- [2026-08-31] [PUBLISHED lib `1710.02410`] ⭐⭐ **"THE NETWORK IS NOT FORCED TO TAKE THE COMMANDS INTO ACCOUNT" IS
  A 2018 RESULT WITH A NAMED FIX — our P2 is a re-discovery and the paper must say so.** Codevilla CIL, verbatim, on
  the concatenation-family *command input* architecture; the fix is **architectural branching** (*"the command acts
  as a switch"*), **64 % vs 52 %** success in the held-out town. Our FiLM/AdaLN modulation is the same family in the
  respect that matters — never forced — and **MM-E18's CONVERGED FiLM gain is the predicted equilibrium, not an
  anomaly**. ⭐ In a branched head the un-selected branch gets **no gradient**, so ignoring is unrepresentable.
  ⚠️ Needs a small DISCRETE vocabulary (theirs is a 4-way nav one-hot); does not apply to our continuous 2-D channel —
  impact: P2 conditioning mechanism, paper novelty downgrade — same RESULT.md F2
- [2026-08-31] [PUBLISHED lib `2607.27017` abstract-only] ⭐⭐ **L3's BAR HAS A PUBLISHED NAME AND OUR LADDER WAS
  ARRIVED AT INDEPENDENTLY.** *"only the full multimodal objective forecasts force beyond a **persistence
  baseline**"* ⇒ `zhat`-vs-`z_t` is precedented, not invented. Their *"certificate-gated protocol"* — certify
  recoverability from raw observations FIRST, *"so a null result can be attributed to the objective rather than to
  the environment"* — is our L1-floor→L2→L3 with our own rationale. ⭐ Two transfers: *"under single-step prediction
  a vision-only latent discards even perfectly visible object state"* (MM-E14 measured our predictor IS one-horizon,
  ‖W1‖ 8.8386 vs ‖W2‖ 0.0262) and stiffness **R² 0.50 when forecast vs −0.02 when merely fused into the input** ⇒
  the TARGET decides what is retained. ⚠️ ABSTRACT ONLY; protocol UNVERIFIED — impact: P5/L3 precedent + paper
  citation — same RESULT.md F3
- [2026-08-31] [PUBLISHED lib `2602.19634`+`2607.12547`] **P4's ladder is licensed AND its failure mode is named in
  advance.** Jumpy WMs across multiple timescales with *"a novel consistency objective that aligns predictions
  across timescales"* → **+200 % relative** on long-horizon tasks — ⭐ that consistency term is the piece MM-E16's
  1 : ~3 : ~15 ladder does not have. But *Mind the Gap*: *"Hierarchy does not automatically improve performance"*,
  best short-horizon config is a ONE-step high level, the bottleneck is **high-level subgoal generation**, and
  *"unconstrained search can select latent macro-actions that appear favorable under the learned model but produce
  poor control targets"* — constrained-to-training-trajectories search recovers **+11.3/+14.7 pts**. ⇒ judge a
  ladder at tactical/strategic, never operative. ⚠️ Both manipulation/navigation, not driving — impact: P4 /
  MM-E16 design — same RESULT.md F5
- [2026-08-31] [PUBLISHED lib `2606.09028`,`2605.31111` abstract-only] **Two instruments/mechanisms we built by hand
  exist in print.** ATM compares *"action information in real encoded transitions and model-predicted transitions
  through lightweight post-hoc probes"*, **>100× faster than CEM-coupled eval**, plus AITS turning
  action-identifiability into a training signal — ⛔ but *"when the true success gap is non-trivial"* makes it a
  SCREENING instrument, never a gate verdict. SD-JEPA carves an orthogonal **progression** subspace from the
  **content** subspace — the architectural remedy for P5's *"restating, not transporting"*; ⚠️ *"majority of its
  control benchmarks"*, no magnitude, not driving — impact: cheap checkpoint ranking; P5 mechanism candidate —
  same RESULT.md F4/F6
- [2026-08-30] [repo/MEASURED] ⭐⭐ **The v7-tiny T1 arms are action-conditioned BY CONSTRUCTION and action-free
  BY OBJECTIVE — a third state that neither literature branch covers.** `models/v6.py:5455` forward takes
  `actions` positionally; predictor built `action_dim=3` (`train_v6_staged.py:4122`); but O1 — documented as
  *"action-conditioned prediction with L_ctrl… the ANTI-ACTION-ECHO measure"*, `default=1.0` at `:7042` — was set
  to 0. ⇒ ⛔ *"never trained to drive"* is true of the objective, NOT the architecture, so `2411.04983`/`2412.03572`
  (closed-loop planning with zero planner objective) are NOT excluded and the comfortable reading is unproven —
  impact: D-T1-V7-READ scope line — `2026-08-30-t1-floor-action-sensitivity/RESULT.md` F1
- [2026-08-30] [derived/MEASURED] ⭐ **The closed-loop vs hold-action gap is only ~1 %** (emao14_30k ade +1.37 %,
  fde +0.64 %; o14fut30k ade +1.25 %, fde +0.94 %) — handing the model the wheel barely moves the rollout, which
  is the ACTION-INSENSITIVITY signature, not a bad-driving signature. ⚠️ HYPOTHESIS `H-ARCH-ACTINS` only: at
  heading MAE ~95° (chance) a degenerate rollout predicts the same small gap, and banked numbers cannot separate
  the two — impact: D-T1-V7-READ reading — same RESULT.md F2
- [2026-08-30] [PUBLISHED lib `2607.26712`+`2606.31232`] **"Non-echoing but non-predictive" IS a named, predicted
  failure mode**: ActSWM's **Context Collapse** = *"nearly indistinguishable futures under different action
  sequences"*; Delta-JEPA = *"reconstruction-free joint-embedding objectives can collapse to action-insensitive
  representations"*. ⇒ removing the echo (`echo_index 0.0000`) without an action-response term is the PREDICTED
  endpoint, not an anomaly — impact: D-T1-V7-READ, O1 weighting — same RESULT.md F3
- [2026-08-30] [PUBLISHED lib `2607.26712`,`2606.07687`; repo/MEASURED] ⭐ **The discriminating test is cheap and
  needs no training**: roll one initial state under DISTINCT action sequences, measure the spread of predicted
  futures — spread ≈ 0 ⇒ Context Collapse ⇒ the frozen-teacher lever is aimed at the wrong defect. Second test,
  action recoverability, needs only a PORT: `InverseDynamicsHead` exists (`inverse_dynamics.py`, used
  `fourbrain.py:461`) but is NOT wired in `models/v6.py` — impact: pre-registration before the L2 arm —
  same RESULT.md F4
- [2026-08-30] [MEASURED + PUBLISHED] **τ-ramp is NEUTRAL at 30k** (drift 0.6936 vs 0.6952, nrmse 0.7408 vs
  0.7466, cos 0.7513 vs 0.7524 — all ~20× under the measured seed spread) ⇒ EMA teacher stays at FIXED τ=0.996,
  cosine ramp DROPPED. And the frozen-vs-EMA question is genuinely open: the literature is three-way split
  (frozen `2411.04983`/`2509.10156` · EMA `2404.08471`/`2608.19085` · neither `2511.08544`/`2011.10566`) with
  **no head-to-head ablation in a driving WM found** — an argument FOR running L2 — same RESULT.md F5
- [2026-08-18] [PUBLISHED/library] ⭐⭐ **FROZEN ENCODERS SUCCEED IN EXACTLY TWO CONFIGURATIONS, AND
  REF-A WAS IN NEITHER.** (A) huge frozen VLM + wide interface + supervised head — FROST-Drive
  [`2601.03460`]: frozen 14 B **8.17 RFS / ADE@3s 1.04 m** BEATS the *same encoder fine-tuned*
  **8.13 / 1.47**, while a frozen **ImageNet** ViT is the WORST arm in the table **7.39 / 2.28** ⇒
  freezing is a **multiplier on pre-training quality, not an independent good**; interface width is
  its own lever (5120-d 8.17 vs 256-d 7.68). (B) moderate frozen encoder + **future-feature
  prediction** + **test-time planning** — DINO-WM [`2411.04983`] (frozen DINOv2 **patch** features,
  plain latent L2, *no* reconstruction/reward/terminal loss, **no policy head**, CEM+MPC; swapping
  patch features for global R3M/ResNet18/**CLS** "significantly degrades"), V-JEPA 2-AC
  [`2506.09985`] (<62 h of robot data), and in DRIVING: DeepSight [`2605.10564`] (frozen encoder +
  MSE against DINOv3 future BEV features, Bench2Drive DS **86.23**) and LAW [`2406.08481`].
  ⛔ Full fine-tuning is not the alternative — it DEGRADES pretrained structure (OpenVLA 36.7 % →
  12.1 % under paraphrase); the winning form is a **DUAL ENCODER** (frozen anchor ‖ trainable),
  35.03 → 55.55 → **78.46** [`2509.11417`]; CortexBench agrees from the other side [`2303.18240`].
  ⇒ REF-A had A's consumer on B's encoder class. **This is the evidence base for REF-A v1.**
  — impact: H4 / the encoder question / REF-A v1 — `Research/2026-08-18-frozen-encoder-literature/FROZEN_ENCODER_LITERATURE.md`, primaries in `Library/`
- [2026-08-03] [repo/MEASURED] **The LONGITUDINAL family's distance-keeping half is implemented and its
  gauge is ADMITTED.** `four_families.longitudinal` had returned `distance_keeping: UNAVAILABLE` since the
  binding rule landed (2026-08-02) because our ingest never read `obstacle.offline`. Now: `lead_metrics.py`
  (headway / time-gap / min-TTC) + `build_lead_tracks.py` (the rig→world→t0 frame composition, which is the
  genuinely new part — `lead_state_gate.lead_frame` answers "where is the lead NOW", scoring an arm needs
  "where would it have been relative to the path THIS ARM PREDICTED"). **Pre-registered D-LEAD-1 GT-vs-CV
  control PASSED on all three:** Δ min-TTC **+1.7474 s** [1.5813, 1.9218], Δ headway **+0.9769 m**
  [0.8830, 1.0758], Δ time-gap **+0.1641 s** [0.1499, 0.1786]; paired episode-cluster bootstrap, **14,027
  windows / 1,431 clip clusters**, B=2000, all separated with the correct sign. ⛔ Says NOTHING about any
  arm — it measures the gauge. ⛔ min-TTC is **censored** at 30 s on ~50 % of windows; quote `n_closing`.
  ⛔ **The eval path is not yet fed** — arm evals still report UNAVAILABLE until `win["lead"]` is built for
  the 40 val episodes (backlog P0 L1). — impact: the binding four-family rule / H-LONG / D1
  — `Research/2026-08-03-longitudinal-distance-keeping.md` + `…/incoming/2026-08-03-longitudinal-distance-keeping/raw/dlead1_discrimination.json`
- [2026-08-03] [PUBLISHED] **ADE does not predict closed-loop driving score: ρ = −0.36, p = 0.43 (n=8)**
  ([2605.00066](https://arxiv.org/html/2605.00066)); PDMS aggregate ρ = 0.90, Ego Progress alone ρ = 0.83.
  Direction is citable, magnitudes are not (n=8, p-values, no CI). ⇒ an arm ranking resting on ADE alone is,
  on the field's own evidence, uninformative about closed-loop driving — which is the empirical case for
  Sayed's binding four-family rule. — impact: all gates / instrument doctrine — `Research/2026-08-03-sota-scan/SOTA_SCAN.md` §2
- [2026-07-18] [repo/measured] **OPERATIVE flagship-speed @19k: the σ-dissipation + attractor collapse
  REPRODUCES (drops the pre-reset caveat); readout isotropy is CONVERGING toward admissibility.** Re-ran
  E1+E2 (backlog P0.1) on `flagship-speed` (WorldModel flagship4b, action_dim=3, step 19000) on the eval
  pod A40, canonical PhysicalAI val, 2 seeds, $0. **E1 (blind K-step rollout, 320 windows):** the P0.1
  falsifier "speed+jerk fixed it" is **NOT met** — cos_rollout dies to chance by **k3** (0.232→0.016→neg),
  σ_hidden nets **−9.461→−9.564** (more confident as it decays; *lower* absolute σ than the −7.8 pre-reset
  ckpt = worse temporal calibration), attractor inter-sample cos **0.219→0.805** (sharper than 0.57
  pre-reset). **freeze-1 holds 0.232→0.213 FLAT across 8 horizons, 7× persistence** → parallel-horizon is
  the safe operative mode, confirmed on the shipping model. **Refinement:** σ is *spatially* calibrated
  (calib_gap +0.37 hidden>visible; per-cell err↔var corr +0.29–0.43) but *temporally* anti-calibrated —
  the design target narrows to a **horizon-aware** σ, not a spatial rebuild. **E2 (orthogonality, 7,964
  latents):** `iso_ratio_active` **0.254→0.546** (crossed 0.5), `cond_active` **218→61**, `rms_offdiag`
  0.42→0.32 — SIGReg converging exactly as the 07-17 note predicted; still **NOT-YET-ADMISSIBLE** (offdiag
  0.32>0.1 → LeJEPA optimal-planning corollary still withheld). active_k≈19, cov_eff_rank≈30 ≪ 2048 →
  **readout capacity is NOT the D1 bottleneck (G1), reaffirmed on the operative model.** No config change
  (D-018); decision-grade re-run at @30k is turnkey (both scripts, ~2 min pod). — impact: H15 / H11 / D8 /
  H3 / D-021 / G1 — `../Research/2026-07-18-operative-flagship-blind-rollout-and-orthogonality.md`
  + `../Implementation/belief_rollout_diagnostic/blind_rollout_flagship.py`
- [2026-07-18] [ICLR2026 openreview pZuZWRuPyi] **HAUWM — "Learning to Be Uncertain: Pre-training World
  Models with Horizon-Calibrated Uncertainty" is the direct design anchor for backlog 0b-A** (the fix for
  E1's exact failure). A probabilistic-ensemble WM predicts frames at **randomly sampled future horizons**
  and a **Horizon-Calibrated Uncertainty (HCU) loss** shapes the latent so **predictive variance GROWS with
  horizon** — precisely the property my 07-18 E1 found MISSING on the flagship (σ dissipates instead). Two
  routes for 0b-A: (a) HCU-style horizon-sampled variance target on our single logvar head (cheaper, no
  ensemble; our ImaginationField already emits per-cell logvar — just supervise it against realized
  multi-horizon error so it must rise); (b) small ensemble à la **ELVIS** (2605.04709, ensemble-calibrated
  latent imagination for long-horizon visual MPC) if the single head can't express growing σ. **CAVEAT
  (P8): full text is behind OpenReview verification — mechanism is from the search abstract, the exact HCU
  loss form is UNVERIFIED; fetch the arXiv mirror before porting.** — impact: H15 / H11 / D8 / D9 (backlog
  0b-A) — https://openreview.net/forum?id=pZuZWRuPyi · https://arxiv.org/pdf/2605.04709
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
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`

- [2026-08-31] [PUBLISHED lib `2605.09701` FULL TEXT] **BOTH CURRENT NAVSIM LEADERS KEEP OUR 'DEFECTIVE' REALISED-MOTION ACTION CHANNEL AND FIX THE OBJECTIVE INSTEAD** - DriveFuture feeds (dx, dy, sin th, cos th); DriveWorld-VLA serialises ego actions to text; both target teacher-forced GT latents. Their mitigations are LatentAlign (sigmoid anneal grounded->self-predicted condition, ablation 32.1->34.6 EPDMS, e0 sensitivity +-0.12 = ~5.5 EPDMS) and an action-source mixture {p_gt, p_kin, p_null} with a LEARNED NULL TOKEN.
  impact: **corroborates today's P2(b) verdict (target, not channel) from SOTA SYSTEMS rather than mechanism; two pre-registerable anti-echo arms for the v7-tiny ladder** - `Frontier Scan/Daily/2026-08-31/RESULT.md` F1/F2
- [2026-08-31] [PUBLISHED lib `2602.06521` FULL TEXT] DriveWorld-VLA: **training STRATEGY alone moves PDMS 83.6 -> 91.3 (+7.7)** (non-progressive vs progressive), and task+feature supervision adds +3.4 over task-only - a larger delta than most architecture changes we have measured.
  impact: **schedule/curriculum is a first-class lever, not a tuning detail** - `Frontier Scan/Daily/2026-08-31/RESULT.md` F1

- [2026-08-31] [PUBLISHED-BLOG/waymo + MEASURED/ours] **OUR v6 READOUT-GEOMETRY CEILING IS A FIELD PROPERTY, NOT A TANITAD BUG** - Waymo reports frontier VLMs *lack sufficient spatial awareness*; we measured 4 azimuth bins over 120deg (30deg/bin, 2.1-7.8x too coarse for BEV localisation). Same defect class, theirs at Gemini scale.
  impact: **reframes the geometry fix from REPAIR to DIFFERENTIATOR; motivates B13 (SparseOcc++ `2607.04732`, persistent-3D-state `2603.03482`, both banked-unread)** - `Frontier Scan/LEDGER_B13_3d_geometry.md`
- [2026-08-31] [PUBLISHED lib `2606.21775` abstract-only] **Variable-Length Latent World Models: ONE predictor conditioned on VARIABLE-LENGTH action sequences evaluates plans across horizons** - our tactical-6s vs strategic-30s problem stated as an architecture; converges with today's 1.7M-param jump-model finding (1.02x GT motion vs *less than half* for single-pass).
  impact: **two independent lines now say long horizons need TEMPORAL ABSTRACTION, not a bigger model or a longer chain - and our MEASURED linear-in-K latency says the chain is unavailable anyway. Sequence AFTER the anti-echo arms P-1/P-2 or the result is uninterpretable** - `Frontier Scan/LEDGER_B12_memory_horizon.md`

- [2026-09-01] [PUBLISHED-PRIMARY lib `2310.16828` `2301.04104` `2606.18208` `2412.01522` `2608.25017`] **NO PUBLISHED RECIPE IN OUR REFERENCE CLASS BACK-PROPAGATES A 60-STEP FULL CHAIN** - TD-MPC2 plans at **H=3**, DreamerV3 imagines at **H=15**, Looped-WM **truncates** at `mu_bwd=ceil(mu_rec/2)`, InfinityDrive grows its window **16->32->64->128** by curriculum. Looped-WM states our failure verbatim: *"Training directly with a large K is unstable because gradients must back-propagate through K x T shared-parameter applications."* The stability lever is **AGC / latent normalisation / horizon curriculum - never a smaller scalar clip.**
  impact: **the planned `k60clip05p30k` mitigation (`--clip 0.5`) tests a lever no primary uses, and the prereg's own text says the problem is DIRECTION not magnitude (clip is PRE-clip, so updates were already bounded to 1.0); recommend a curriculum `K(step)=min(60, 8+floor(step/Delta))` instead** - `Architecture & Inference/Research/2026-09-01-longhorizon-bptt-stability/RESULT.md` (answers LAB_ASKS ASK-1)

- [2026-09-02] [MEASURED ours, re-analysis of two banked actdiv reads, all controls pass] **73.7% OF MM-E19's RATIO FALL IS THE DENOMINATOR, AND THE PRE-REGISTERED "10x" BAR SILENTLY DEMANDED 17.1x** - the action/scene ratio is a quotient of two quantities that BOTH move: one variable (`o5_k` 8->60) gives action factor 0.8256, scene factor 1.7126, ratio 0.4821 (reconstructed 0.4821). A ratio bar therefore floats with the denominator: 10x x 1.7126 = **17.13x** in action-side units.
  impact: **the statistic PENALISES world-model improvement - an arm that DOUBLED its action sensitivity would have read +17% and been written up as INERT. MM-E19's verdict stands (0.83x either way), but backlog row L-1's curriculum arm must NOT be pre-registered against the raw ratio** - `TanitAD Research Lab/Architecture & Inference/Research/2026-09-02-scene-matched-action-criterion/RESULT.md`
