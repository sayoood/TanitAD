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

- [2026-09-03] [PUBLISHED-PRIMARY lib `2607.26712` `2608.06706` `2606.31232`] **OUR ACTION FAILURE HAS A NAME AND A PUBLISHED SIZE - "CONTEXT COLLAPSE"**: ActSWM Table 5 step-31 cosine GT/zero-action/gap on LeWM: H=3 0.239/0.244/-0.005, H=32+multi-step 0.972/0.970/0.002 - every PREDICTION lever (context, multi-step, resolution) leaves the gap at 0; only an explicit zero-action separation term moves it (frozen readout 0.592, hinge+readout 0.760). Dueling WM: action separation collapses 1.28 -> 0.002 WHILE validation loss improves; val-loss checkpoint selection picks collapsed checkpoints 17/36.
  impact: **our MM-E10 ratio 0.004-0.006 is the published collapsed regime; k/O14/EMA/tau-ramp could never have moved it; T1 candidates selected by nrmse are exposed to the 17/36 error - select by the anchored sensitivity read (LAB-ACT-3)** - `Architecture & Inference/Research/2026-09-03-sota-action/RESULT.md`

- [2026-09-03] [PUBLISHED-PRIMARY lib `2608.06706`] **POST-HOC ACTION-MEAN CENTRING WORKS ON FROZEN MODELS**: zhat = B(z) + [Delta(z,a) - mean_a Delta(z,a)] (K=16 draws) read post-hoc on frozen RePo/TIA raises the action-delta probe from -0.002..+0.052 to 0.19-0.52; trained-in vs post-hoc parity on DMC (0.533/0.328/0.186 vs 0.528/0.323/0.192). On a realised-motion action channel the residual is exactly the motion NOT predictable from the scene.
  impact: **a 0-GPU read on all 32 census arms (LAB-ACT-1) decides "dead channel" vs "masked channel" BEFORE any arm is trained; the two branches lead to different first arms (AD-JEPA offset head vs ActSWM hinge)** - `Architecture & Inference/Research/2026-09-03-sota-action/PROPOSED_HYPOTHESES.md`

- [2026-09-03] [PUBLISHED-PRIMARY lib `2606.31232` `2606.20104` `2602.03604` `2607.02403`] **THE PUBLISHED FIXES PARTITION INTO ENCODER-SHAPING vs FROZEN-TRUNK-COMPATIBLE**: LDAD decodes the action from dz = z_{t+1} - z_t built from two ENCODER outputs on a ViT-Tiny trained from scratch (lambda=0 nearly collapses, 50 best; dz beats concat +4.07/+1.07/+12.60/+0.67; a realised end-effector decode target loses 16 pts, Table 3); SMWM and EB-JEPA (IDM omega=0 -> 1% success) likewise shape the encoder. ACID (planning-cost IDM on FROZEN WM latents: Le-WM 70/76/96 -> 74/88/100), ActSWM's hinge on predicted rollouts and AD-JEPA anchoring act on the predictor/planner.
  impact: **GS-2 is confirmed by content (not from a figure); the v7f freeze decision is the P2 decision ONLY for the encoder-shaping family - the first arm is drawn from the other family and does not re-open the trunk** - `Architecture & Inference/Research/2026-09-03-sota-action/RESULT.md` F2

- [2026-09-03] [PUBLISHED-PRIMARY lib `2606.07687`] **THE ACTION-CONDITIONING PARADOX IS PUBLISHED**: feeding the action INTO the trunk drops action-relevant R2 from 0.26 to -0.32 (Table 6), while an inverse-dynamics objective on FROZEN features multiplies it (V-JEPA 2 0.40 -> 0.85; Web-DINO -0.01 -> 0.16; SigLIP2 0.05 -> 0.17; k-step ID peaks at k=4). Actions on LIBERO are realised end-effector deltas.
  impact: **MM-E17/E18 (FiLM deafness is a converged optimum) is a known optimum of the objective, not a bug; the small multiplier on DINO-family features (0.16) is the prior for any adapter-IDM arm on refav1's frozen DINOv3 - theme 2's LAB-ENC-1 tests it at 0 trunk compute** - `Architecture & Inference/Research/2026-09-03-sota-action/RESULT.md` F3

- [2026-09-03] [PUBLISHED-PRIMARY lib `2608.24044` `2606.26217` `2512.24497` `2510.26782` `2504.03861`] **THE FIELD'S "DRIFT" IS ROLLOUT-VS-ENCODED-FUTURE DIVERGENCE, NOT OUR r(dz | z_t)**: JEPA-x drift(h) = E||zhat_{t+h} - z_{t+h}||^2 / E||z_t - z_{t+h}||^2 (persistence-normalised; fresh predictor on the frozen encoder; 20-step rollout on recorded actions); Fast-LeWM fits the open-loop error SLOPE; What-Drives-Success tracks embedding error at H=1..3; no primary defines increment predictability (grep over 36 extracted texts: 0 hits; 20 texts use "drift", 5 use "persistence").
  impact: **no published number may be quoted against our 0.669; L-17 (persistence-normalised divergence on the banked arms, LAB-DRIFT-1, 0 GPU) and L-18 (rename E-DEC-59) come BEFORE the v7f prereg quotes any drift** - `Architecture & Inference/Research/2026-09-03-sota-drift/RESULT.md` F1

- [2026-09-03] [PUBLISHED-PRIMARY lib `2608.24044`] **CROSS-PREDICTION TO A PRIVILEGED PHYSICAL STATE IS THE ONLY MEASURED ANTI-DRIFT LEVER, AND ITS CONTROL ARM IS OUR O14 NULL**: JEPA-x multi-task drift 0.361 -> 0.104 with control 53.6 -> 78.2%; single-task Two-Room 0.503 -> 0.221, Push-T 0.399 -> 0.258; a state-REGRESSION head raises decodability R2 0.978 -> 0.991 and leaves drift at 0.373; action-conditioned next-state/increment heads 0.386/0.359 (no gain); CROSS-ONLY 0.101 vs SHARE-ONLY 0.309 vs ALIGN-ONLY 0.158; the head is training-only, inference vision-only.
  impact: **E-DEC-67 (O14 absorbed pixels, drift unchanged) is the published REGRESS null; a training-only cross-prediction head to ego kinematics - admissible under labels-may-use-ego, Master Mind to confirm - has never been run on our line -> LAB-DRIFT-2 (one v7-tiny arm + the published-null control arm)** - `Architecture & Inference/Research/2026-09-03-sota-drift/PROPOSED_HYPOTHESES.md`

- [2026-09-03] [PUBLISHED-PRIMARY lib `2506.09985` `2512.24497` `2607.26712` `2603.19312`] **EMA IS NOWHERE AN ANTI-DRIFT LEVER; ROLLOUT LOSSES ARE UNIVERSALLY TRUNCATED AND SHORT; A DRIFT-FREE ROLLOUT CAN STILL BE ACTION-BLIND**: V-JEPA 2-AC differentiates through ONE recurrent step (T=2); What-Drives-Success: K=2 optimal in sim, K=6 on DROID, Remark 1 accuracy-robustness trade-off (the rollout loss lowers the effective Lipschitz constant along the rollout at the cost of one-step accuracy), and the best success proxy is the ONE-step embedding loss (Tables 13-16); ActSWM raises step-31 cosine 0.239 -> 0.972 with context + multi-step training while the action gap stays <= 0.002; LeWM trains with no EMA and no stop-gradient; no primary ablates EMA on a drift quantity (6 texts carry both words, none has the row).
  impact: **our EMA read (prediction +24.5% cos, drift +3.6%) AGREES with the field and backlog row 4 closes; the o5_k lever is small and non-monotone (E-DEC-66 and MM-E19 agree); P3 leaves the v7f gate and stays a diagnostic next to nrmse and the anchored action read** - `Architecture & Inference/Research/2026-09-03-sota-drift/RESULT.md` F3/F4

- [2026-09-03] [PUBLISHED-PRIMARY lib `2601.03460` `2602.12218`] **THE SIGN OF "FREEZE VS FINE-TUNE" DEPENDS ON THE TRUNK**: FROST-Drive - a frozen 14B VLM encoder scores RFS 8.17 / ADE@3s 1.04 and the SAME encoder fine-tuned 8.13 / 1.47, while a frozen ImageNet ViT 7.39 loses to its fine-tuned self 7.79; Observer Effect - full fine-tuning erases DYNAMIC variables (kinematic invariant rho 0.94 -> -0.03; frozen linear probe 0.91 vs full FT 0.05 vs last-layer 0.65; deep blocks B5-B10 move most, CKA < 0.2) and keeps static ones, and adaptation-based evaluation hallucinates competence (MAPE 140% -> 18%).
  impact: **E-DEC-64 (freeze = degenerate) is consistent with the field but is NOT the only datapoint - the v7f trunk decision has a published prior in both directions (freeze a strong pretrained trunk, train a tiny one); judge trunks ONLY by linear probes on frozen features (our ridge panel complies); L2's "agents up, range/ego down" is the published static-kept / dynamic-erased signature (LAB-ENC-2, 0 GPU)** - `Architecture & Inference/Research/2026-09-03-sota-encoder/RESULT.md` F1/F3

- [2026-09-03] [PUBLISHED-PRIMARY lib `2411.04983` `2512.24497` `2507.19468` `2608.29434`] **FROZEN DINO PATCH TOKENS SUFFICE FOR PREDICTION - WITH A PATCH-TOKEN READOUT, A TRAINED PREDICTOR AND A KNOWN METRIC CEILING**: DINO-WM patch 0.90 vs CLS 0.44 / R3M 0.42 / ResNet 0.20 (Push-T); What-Drives-Success: frozen DINOv2/v3 beat V-JEPA/2 at ViT-L, DINOv3 wins only on photorealistic data, proprioception is needed because metric quantities are implicit and patch-quantised, scaling helps only on real data; DINO-world: frozen DINOv2 + 1.1B predictor beats copy-last (VSPW mid 47.0 vs 42.1, Cityscapes 55.1 vs 39.7, KITTI 4.268 vs 4.745) where a jointly trained V-JEPA collapses (7.7); Utonia-WM (frozen PTv3) Push-T 46.6 vs a trained tiny encoder 83.6 - "the price of a representation that cannot adapt".
  impact: **refav1's frozen-DINOv3 design is field-supported IF its predictor consumes patch tokens (no registry row - verify before quoting a number); no frozen-trunk WM reports a numeric action-sensitivity metric and the inverse-dynamics multiplier on DINO features is small (Web-DINO 0.16) -> refav1 is PREDICTED action-insensitive: a predictor/target AND a feature problem** - `Architecture & Inference/Research/2026-09-03-sota-encoder/RESULT.md` F2

- [2026-09-03] [PUBLISHED-PRIMARY lib `2602.18639` `2606.07687` `2603.24581`] **FROZEN-COMPATIBLE SHAPING EXISTS WITH NUMBERS**: a 196x384 -> 196x32 bisimulation adapter on frozen DINOv2 patch tokens, trained jointly with the transition model, raises PointMaze success under shift 0.48 -> 0.78 (iBOT 0.72; end-to-end without a pretrained trunk 0.26 vs 0.86); an inverse-dynamics objective on frozen features multiplies action-relevant R2 (V-JEPA 2 0.40 -> 0.85, Web-DINO -0.01 -> 0.16); Latent-WAM: distillation INTO the trunk 89.3 > frozen-feature concatenation 88.0 > nothing 88.3, and LoRA is the worst row (Base-LoRA 68.5).
  impact: **GS-2 narrows to "nothing to shape IN the trunk"; LAB-ENC-1 tests the adapter route on refav1's banked fp8 features at 0 trunk compute (success bar: the published +0.17 R2); avoid LoRA-style partial unfreeze in the v7f prereg** - `Architecture & Inference/Research/2026-09-03-sota-encoder/PROPOSED_HYPOTHESES.md`

- [2026-09-03] [PUBLISHED-PRIMARY lib `2607.27017` `2507.19468`] **L3 IS CLEARED IN THE FIELD, AND OUR FAILURE MODE HAS A NAME - THE LAZY EQUILIBRIUM**: single-step vision-only latent world models discard even directly visible, perfectly predictable object state (position R2 0.04 against a recoverability certificate of 1.00); two escapes compose - cross-modal prediction TARGETS (0.58) and multi-horizon action-conditioned heads (0.89), together 0.98 - and direct long-horizon heads beat autoregressive composition at matched horizons (0.10 vs 0.19; static 0.21); force forecasting beats persistence only when touch is fused AND forecast (paired CI [+0.14,+0.28] at 4,258 episodes; camera-only -0.05/-0.11/-0.05); DINO-world's frozen-DINOv2 predictor beats copy-last on dense targets (VSPW mid 47.0 vs 42.1, Cityscapes 55.1 vs 39.7, KITTI 4.268 vs 4.745).
  impact: **P5 stays as written - it is achievable; our one-horizon predictor (MM-E14) with predictor deltas <= 0 at k=1/3/6 is the published equilibrium; the v7f arm is direct multi-horizon heads + a cross-modal ego-kinematics target (LAB-REP-2, to be MERGED with LAB-DRIFT-2 - same lever family), not another single-step objective change** - `Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md` F3

- [2026-09-03] [PUBLISHED-PRIMARY lib `2603.21546` `2607.27017` `2606.31232` `2608.29434`] **MOST FIELD PROBE TABLES LACK OUR CONTROLS, AND THE TWO THAT CARRY FLOORS SHOW THE FLOOR IS TARGET-SPECIFIC**: Delta-JEPA, LeWM, the point-cloud WMs and JEPA-x report no constant-only control and no raw floor; 2603.21546 (raw pixels / random model / shuffled labels) reads raw pixels at -1.31 on ball x and 0.9998 on score in the SAME table; 2607.27017 gates every claim behind a recoverability certificate (a recurrent probe on RAW observations) - the one rung our panel lacks; transition-level probes exist (dx-from-dz linear r 0.992 Delta-JEPA vs 0.765 LeWM, Table 5; joint velocity MLP R2 0.626 vs 0.102, decodable only from an LDAD-trained latent).
  impact: **no field number enters our tables as a bar; LAB-REP-1 (0 GPU) reads persistence / predictor / certificate plus the dz transition rung with a raw PAIR floor on five banked arms - the P3/P5 dual read - and LAB-REP-4 puts the certificate rung and the raw pair floor into the criteria registry** - `Architecture & Inference/Research/2026-09-03-sota-decodability/PROPOSED_HYPOTHESES.md`

- [2026-09-03] [PUBLISHED-PRIMARY lib `2411.04983` `2603.20327` `2603.24581` `2512.24497`] **TOKEN-LEVEL ATTENTION IS THE FIELD'S READOUT FOR FINE GEOMETRY; POOLING ERASES IT**: DINO-WM patch tokens 0.90 vs CLS 0.44 (Push-T); mean pooling over 196 spatial tokens collapses codebook use from 62.5% to 5% ("erases spatial local semantic differences"); Latent-WAM reads 16 learnable query tokens per view with attention on lane markings; What-Drives-Success: metric quantities are only implicitly encoded and patch-quantised.
  impact: **E-DEC-25 (the 128->64 projection lost lead range) is corroborated in kind and our 4x8 grid pool + projection is the outlier readout family; LAB-REP-3 separates readout from latent on the same banked tokens at 0 trunk compute and decides GS-4's ordering instead of assuming it** - `Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md` F5

- [2026-09-03] [PUBLISHED-PRIMARY lib `2608.29998` `2603.20327` `2607.27017` `2606.31232` `2608.06799`] **OUR PSEUDO-REPLICATION DEFECT IS THE FIELD'S NAMED DEFECT, AND THE FIELD PUBLISHES BOTH RESPONSES**: `2608.29998` states the rule verbatim — "the complete environment family is the uncertainty unit; checkpoints, roots, actions, coordinates, and imagined samples are never treated as independent replications" — and reports n=96 evaluation families with 4,096 whole-family bootstrap resamples, readouts and thresholds frozen on 48 calibration families, plus ten prespecified synthetic acceptance cases one of which is "halt on split leakage"; `2603.20327` self-declares the SAME defect ("token-level pseudo-replication": 1,568 tokens per video are not independent, effective n is ~9-10 videos) and downgrades its own p-values to "upper bounds on statistical significance". Correct clip-structured LOO = split by trajectory/episode (`2606.31232` "split train/test data by trajectory", 20,000 pairs, 3 seeds; `2607.27017` "probe-fit and probe-test windows come from disjoint episode sets"; `2608.06799` held-out test episodes) AND make the clip the unit of the INTERVAL, which only `2608.29998` does. Published n: 96 families / 20,000 pairs / 10,000 frames / 4,000 frames / 800-4,258 episodes.
  impact: **`H-LEAK-2` is not an idiosyncratic bug — our 24 clips are NOT small by the field's standard, our ESTIMATOR was wrong; and `2608.29998`'s three-gate order (capture -> real-effect resolvability -> propagation, interpreted only if 1 and 2 pass, against persistence / zero-effect / an environment-endpoint oracle) is what L3 must adopt before P5 is scored again — on `rdw8p30k` Gate 1 currently FAILS (`z_t` pooled cross-clip R2 is -0.05..-0.09) so no propagation number on that arm is interpretable** — `Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md` F1 + `PROPOSED_HYPOTHESES.md` LAB-REP-1

- [2026-09-03] [PUBLISHED-PRIMARY lib `2607.06925` `2607.27017` `2312.03031`] **THE ORACLE-INPUT GUARD IS PUBLISHED IN THREE FORMS, ONE OF THEM IN OUR OWN DOMAIN — AND OUR nav_zero COLLAPSE IS THE TEXTBOOK CASE**: `2607.06925` measures a goal-conditioned readout at 0.90 that collapses to 0.27 (chance 0.25) when the goal is withheld (3 seeds 0.900±0.009 -> 0.270±0.034) and follows a COUNTERFACTUAL goal 94.5% [0.914,0.969] of the time against the true scene 2.3% [0.008,0.047] (N=256), with a VALIDATED POSITIVE CONTROL (engineered-leaky model fires at 0.97, no-goal baseline 0.03) — and its dose-response REFUTES the "the other inputs are weak" explanation (alpha 1.0->0.0 gives 0.975->0.986: degrading the other input never increases leakage); `2607.27017` adds the PASSTHROUGH BOUND (an UNTRAINED copy of the architecture already reads force 0.94 / position 0.97, so the honest metric is passthrough-immune forecasting); `2312.03031` Table 2 is the system-level version in driving — blank images move planning L2 0.37->0.46 while detection NDS goes 45.5->0.0 and map mAP 47.0->0.0, whereas v x 0.0 moves L2 0.37->6.16, and Ego-MLP (ego status + driving command, NO perception) scores L2 avg 0.35 against VAD-Base's 0.37. The exposed/immune split is STRUCTURAL: goal in the planner COST is immune, goal conditioned into the PREDICTOR is exposed.
  impact: **`D-REFAV1-TAC-DECODER-PANEL` (AUC 0.873 -> 0.520 under `nav_zero`, nav-only predictor 0.684 > model 0.650 = constant 0.650) is the published class, and the fix is structural (nav out of the tactical dynamics, into the cost), NOT a loss weight; three controls we have never run are cheap and decisive — counterfactual nav, a validated positive control, and an untrained-passthrough bound (LAB-REP-4, 0 GPU). ⭐ `2607.06925` states TWICE that it could not find a public-weights world model conditioning its predictor on language and that demonstrating the leak in a released model is "the key remaining step" — refav1 is that model and the audit is ours** — `Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md` F4

- [2026-09-03] [PUBLISHED-PRIMARY lib `2210.02885` `2608.10145` `2607.27017`] **THERE IS NO MATCHED-REFERENCE CONVENTION FOR A RANK STATISTIC — THE FIELD'S OWN METRIC SHIPS WITH A SCOPING RULE INSTEAD, AND OUR RETIRED FLOOR IS CORROBORATED**: RankMe's authors write "RankMe should however only be used to compare different runs of a given method, since the embeddings' rank is not the only factor that affects performance", and its cross-dataset transfer rests on an explicit monotonic-link hypothesis that fails where source/target overlap is small (StanfordCars); `2608.10145` Table 2 measures the consequence on two checkpoints of one method on one task — effective rank (participation ratio of the covariance spectrum) 18.6 vs 67.8 of 192 while linear position decodability differs by 0.0006 and MLP decodability is IDENTICAL (0.9994), the authors' explanation for the extra dimensions did not survive, and the same number moved 16.5 -> 67.8 and 11.9 -> 18.6 on checkpoints WHOSE WEIGHTS NEVER CHANGED because it was read downstream of an unrepaired normalisation layer ("probe-style measurements are precisely the ones that will not warn you"); `2607.27017` shows the anti-collapse weight is a dose-response knob that trades spread against metric precision (60x lambda sweep, stiffness 0.12 -> 0.57), so a regulariser can raise rank and lower decodability at once. `2608.10145` also states the general form of our problem: "Probe values are protocol-dependent" — the same encoders read 0.9305-0.9974 across the authors' own logs and 0.0006 apart under one protocol on identical frames.
  impact: **`C-PARTICIPATION-FLOOR-RETIRED` and H-RANK-23 (3.51x by episode diversity alone) are field-corroborated at the same order (3.6x by pipeline alone); the field's practice is (i) compare rank within a run/method only, (ii) state the probe protocol beside every number, (iii) VERIFY THE NORMALISATION STATE before any latent-scale read — (iii) is a failure mode our instrument has never been checked for and it goes into the criteria registry with LAB-REP-4** — `Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md` F6

- [2026-09-03] [PUBLISHED-PRIMARY lib `2606.31232` `2608.06799` `2605.06388` `2608.10145`] **THE TRANSITION RUNG IS HARDER THAN THE STATE RUNG BY A LARGE, REPEATEDLY MEASURED MARGIN, AND THE PAIR-VS-DELTA LEAK IS REAL AND SMALL**: `2606.31232` Table 4->5 on the same models (Two-Room, linear r): state 0.950 -> transition 0.765 (LeWM), 0.960 -> 0.813 (PLDM), 0.907 -> 0.674 (Sub-JEPA), 0.998 -> 0.992 (Delta-JEPA, LDAD-trained); `2608.06799` Table 1->2 (held-out episodes, linear/MLP r): joint position 0.71/0.69 but gripper velocity 0.44/0.47 for LeWM and 0.20/0.12 for frozen DINOv2, against 0.69/0.76 for a pair-grounded model; `2608.29434`: joint velocity 0.021/0.102 vs 0.507/0.626. `2608.10145` measures the GS-1 leak directly on a near-linear latent: summed action decodes at R2 0.9290 from the PAIR `(z_t, z_{t+k})` and 0.8982 from the DIFFERENCE alone — +0.031 R2. `2605.06388` gives the construction that compares a predictor to its own encoder ceiling without retraining anything: train the readout on REAL latents, FREEZE it, apply it unchanged to PREDICTED latents (V-JEPA 2.1 encoder 0.829/0.865 -> WM 0.781/0.840; success classifier 0.905 -> 0.789). `2608.06799` argues AGAINST inverse dynamics as the transition target ("multiple action sequences can produce the same endpoint state change") and grounds pairs in the net state change at multiple horizons instead, training-only, heads discarded at inference.
  impact: **GS-9 is NOT TanitAD-novel — it has three protocols and three sets of priors; the leak audit's endpoint-shuffle finding is the same family as `2608.10145`'s +0.031 and argues for reporting BOTH columns rather than discarding the pair form; and `2605.06388`'s frozen-readout construction is what makes LAB-REP-1's predictor-vs-ceiling comparison immune to the retrained-readout gaming that ATM's AITS-P documents** — `Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md` F2
