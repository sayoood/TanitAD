# SPEC_WPB (DRAFT for the Master Mind to register) — does conditioning fan generation and selection on the tactical layer lift route following, on the frozen refcv7 trunk?

*Drafted 2026-10-04 by WP-B, BEFORE any WP-B arm was trained or scored. Nothing in it may run until the Master Mind
registers it (hash + time). Design: `DESIGN.md` (this package). Harness, cache, windows and dense labels: R1's
(`…/2026-10-04-refcv8-r1-head-only/`, SPEC_R1 REGISTERED sha256 `e9873e12…`), reused, not rebuilt.*

## 1. Question and the decision it feeds

R8-4 asks that the tactical layer learn actions with constraints and that it **condition fan generation and trajectory
selection**, with the planner consistent with it. D0 (registered, `SPEC_D0_DOSE_RESPONSE.md`) showed that conditioning
SELECTION pays only with a near-perfect lateral class (q\* 0.95) or an accurate progress constraint (σ\* 0.10), and that
refcv7's own progress predictor HURTS when conditioned on (+0.213 m). This SPEC asks: **with the trunk frozen, does the
WP-B conditioning (per-candidate hypothesis tags + constraint heads + factorised selection [+ allocation]) clear the
route gate where R1-H4 (dense labels + announced nav + listwise selector) alone does or does not — and are the fan and
the pick controllable and consistent?** Clears ⇒ the conditioning enters the refcv8 run with the components named.
Fails ⇒ per-criterion gaps, and the next lever is named from them.

## 2. Harness (fixed)

* **Model code:** the WP-B fix tree = tip (`50efa52`, or the Master Mind's newer landing tip, stated in the run record)
  + `code/fix/` (md5 manifest in `LANDING_READY_WPB.txt`), shipped to Thor as one overlay directory, md5-verified file
  by file, grep-verified for `enable_refcv8` before any launch. ⚠ It carries the tip's §26.1 ceiling-on-the-emitted-plan,
  so its baseline T0 is NOT bit-identical to R1-H4 (fec3a0d). Every WP-B comparison is therefore paired against **T0 on
  the WP-B tree**; R1's arms are cross-references only.
* **Checkpoint:** refcv7-r101-s0 step 50,400, loaded STRICTLY by the tree's eval loader; the refcv8 seams are attached
  POST HOC (`RefCV3Model.enable_refcv8`) — exactly the warm start.
* **Cache / windows / labels:** R1's Thor cache (fp16 kv, int8 BEV, slots), R1's 7,000 TRAIN windows, R1's 4,634
  scored EVAL windows (2,317 GT-turn + 2,317 sampled) + the 1,112 grid, R1's dense lat/lon labels (`r1_lib` literals,
  0–6 s; a PROBE of the trunk, not the v9 release), R1's announced nav table (H3). Constraint TARGETS: the window's own
  future (`refcv8_conditioning.constraint_targets`: terminal heading, turn onset, 6-s progress, v at 6 s).
* **Training:** identical to SPEC_R1 §4 for every arm: AdamW lr 5e-5, 100-step warmup, cosine, wd 0, batch 32, training
  seed 0 (T0r: 1), steps = R1's registered timing rule's value; the refcv8 dedicated generator seed 20261004.
* **Eval:** batch 1, sampler seeds 0 and 1, the 4,634 scored + 1,112 grid windows; the forced-condition passes of §4.
* **Identity control I-W (must PASS before any arm):** on the 1,112 grid windows, seed 0, the head path with the seams
  attached and NO training equals T0-untrained (no seams) bit for bit on `traj`, `sel_idx`, `anchor_traj`,
  `sel_score_v3`; with allocation (M 32) and the prior-free group attached and emission off, `traj`/`sel_idx`/base fan
  bit-identical and base scores within 1e-5 (the GEMM-shape rounding measured on the CPU rig: 9.5e-7). Fail ⇒ stop.

## 3. Arms (priority order = run order; each from the 50,400 weights)

| arm | what is ON (beyond T0) | role |
|---|---|---|
| **T0** | R1-H4's recipe on the WP-B tree: dense labels, announced nav, listwise selection (`SEL_V3_WEIGHT` 0 + the listwise term) | paired base |
| **T1** | seams: per-candidate tags on the base fan (tag + log p, no constraint), constraint heads (`w_cons` 0.05), factorised selection (β, γ), listwise through the r8 path (`w_listwise` 1.0); `cond_dropout` 0.15 | conditioning of generation (tags) + selection |
| **T1d** | T1 with the planner fed ANOTHER window's tactical output in training (`_r8_derange_feed`) | deliberate regression of T1 |
| **T2** | T1 + allocation M 32, top-4, **emitted**, GT hypothesis teacher-forced (≥ 4 candidates) with scheduled sampling 1.0 → 0.25, matched L1 on the best GT-hypothesis allocated candidate (`w_alloc_l1` 1.0) | conditioned generation proper |
| **T2d** | T2 with `_r8_derange_feed` | deliberate regression of T2 |
| **T0r** | T0, training seed 1 | the training-replicate floor (H-ESTIM-SEED-1) |
| **X1h** | T1 + Hydra-style sub-score critics (`w_subscore` 0.5) | X1 extension |
| **T2s** | T2 + L_sat (`w_sat` 0.1) | is L_sat needed for controllability? |
| **X2a** | T1 + lateral prior dropout 0.3 | X2 |
| **X2b** | T1 + the prior-free group (all anchors, emitted) | X2 |
| (R1-H5) | — | window-vector conditioning, R1's tree: cross-reference foil only |

Before T1: a **timing phase** (no scoring, nothing saved): 30 head-only training steps each for T0, T1, T2 and X2b;
`s/step` MEASURED.

## 4. Measures (every arm, both sampler seeds; per family, never pooled)

* **Route / LATERAL (deciding):** turn direction-correct pick (terminal heading class, τ 10.35°) on GT-turn windows;
  heading within 15° at 6 s; terminal heading error; cross-track at 6 s; curvature / yaw-rate MAE 0–2 s
  (`four_families._seq_geometry`).
* **LONGITUDINAL:** |along-track| and signed along-track at 6 s; speed MAE 0–2 s; the pick's 6-s progress relative
  error; ADE / FDE per class. Distance keeping: UNAVAILABLE (no lead tracks in this harness), reported as absent.
* **TACTICAL (R8-4 i):** the behaviour decoder's lat3 accuracy on GT-turn windows and macro-F1 on all classified, lon
  accuracy / macro-F1, each beside its majority control; constraint heads: progress median relative error (all
  classified windows with GT progress > 0.5 m), terminal-heading RMS error (GT-turn windows), each beside the
  TRAIN-median constant (`taniteval.tactical_conditioning.class_report` / `constraint_mae`).
* **STRATEGIC:** N/A — strategic layer OFF (PI ruling R5), reported as absent.
* **Controllability (R8-4 ii):** on every classified scored window, force the class TURN_L / TURN_R / LANE_KEEP
  (`set_r8_force`, one-hot posterior into every planner feed); and STOP-at-d with d ∈ {10, 20} m (lon BRAKE_TO, progress
  d, v_end 0) on windows with v0 ≥ 3 m/s and v0² / 2d ≤ 4 m/s². Read: the pick's direction class (LANE_KEEP: class 0);
  for T2-family the share of ALLOCATED candidates whose own path class equals the forced one; STOP: |stop distance of the
  pick − d| ≤ max(2 m, 0.1 d) (`taniteval.tactical_conditioning.controllability`).
* **Consistency (R8-4 iii):** share of allocated candidates whose `lat3_class` equals their tag; the pick vs its tag;
  the pick vs the tactical argmax (`consistency`).
* **Selection:** fan contains a correct candidate; oracle and random-in-reach picks; pick regret.
* **Cost:** projected refcv8 s/step = 9.9 s (refcv7, MEASURED) + (step_arm − step_T0) × 16 / 32 (refcv7 batch / harness
  batch) — ESTIMATED from MEASURED head-only steps.
* **Estimator:** paired episode-cluster bootstrap over the eval episodes (B 2000, R1's seed-0 draws), every arm vs T0 on
  the same windows and sampler seed. Seed 1 answers the INFERENCE question; T0r answers the TRAINING question.

## 5. Bars (committed now)

* **B-ROUTE (the DECIDING bar; MM 2026-10-04: "the real head's conditioned arm decides"):** on BOTH sampler seeds —
  turn direction-correct ≥ **0.95** with its paired gain vs T0 CI lower > 0; heading within 15° ≥ **0.70**; GT-straight
  ΔADE vs T0 ≤ **+0.05 m** with CI upper ≤ **+0.10 m**; AND the gain vs T0 on turn direction-correct and on heading-15
  each exceeds |T0r − T0| on the same metric.
* **B-TAC (MM 2026-10-04, from D0; necessary, not deciding):** lat3 accuracy on GT-turn windows ≥ **0.95**; progress
  constraint median relative error ≤ **0.0674** (D0's σ 0.10); terminal-heading constraint RMS ≤ **15°** (D0c, post hoc,
  committed here for this future experiment).
* **B-CTRL:** pick follows the forced class ≥ **0.95** for EACH of TURN_L, TURN_R, LANE_KEEP, both seeds; T2-family:
  allocated share ≥ 0.95 too; STOP-at-d within tolerance on ≥ **0.95** of eligible windows for each d. The regression arm
  (T1d / T2d) must fall below 0.95 on at least one forced class AND below its arm with a separated paired CI on the
  pooled pick-follow rate.
* **B-CONS (T2-family):** allocated-candidate consistency ≥ **0.95** on all classified windows.
* **B-COST:** projected refcv8 s/step ≤ **10.5** (the approved ceiling). Above it, the arm goes to the PI, whatever else
  it clears.

## 6. Reading rule (fixed now)

* **I-W fails** ⇒ stop; nothing is trained.
* **T1d or T2d passes B-CTRL, or clears B-ROUTE** ⇒ the instrument is broken; nothing from WP-B is quotable.
* **The first arm in §3's order that clears B-ROUTE, with its regression arm failing B-ROUTE and B-CTRL** ⇒ its
  components are the refcv8 conditioning recipe. T2 is adopted over T1 only if T2 − T1 on turn direction or heading-15
  is CI-separated on both seeds; X1h / T2s / X2a / X2b are adopted only on a CI-separated gain over their base arm on
  a route criterion with straight ΔADE ≤ +0.05 m.
* **B-ROUTE clears, B-TAC fails** ⇒ the conditioned planner already transmits what the head gets right; the B-TAC gap is
  reported per bar, and a refcv8 run with the trunk trainable is expected to raise it (not assumed).
* **B-TAC clears, B-ROUTE fails** ⇒ the head knows, the planner does not follow: read B-CTRL — if controllability passes,
  the selector arbitrates against the head (β / γ telemetry); if it fails, generation does not obey the condition (T2s
  then decides whether L_sat is the missing piece).
* **No arm clears B-ROUTE** ⇒ report each criterion's gap per arm and run R1's TRAIN-FIT diagnostic on the best arm: an
  arm that cannot reach the gate on its own training windows says the frozen features do not carry it ⇒ the refcv8
  full run trains the trunk; one that fits TRAIN but not EVAL says generalisation.
* **B-COST fails** on an otherwise clearing arm ⇒ a PI decision, named with the measured projection.

## 7. The v7-tiny ladder (trunk-dependent levers) — outline, detailed when the v9 release exists

Blocked on WP-A's v9 release (route checkpoint, nav args): **V-RC** (route checkpoint input, RC variant = WP-A E4's
recommendation, dropout ≥ 0.3) with RC-OFF, RC-shuffled and the trivial drive-to-checkpoint floor (WP-A E1) beside
it — bar: turn direction and heading-15 gains vs the no-RC arm separated; the shuffled arm loses them; the RC-OFF row of
the RC-trained arm within +0.05 m ADE of the no-RC arm (NavSim-legal robustness); a privileged and a legal NavSim row
for every NavSim result (MM binding). **V-NAVARGS** (nav-argument dropout 0.5, token kept). **V-X4** (gradient shares
of every loss family on the last shared layers at `w_tac_v6` ∈ {0.1, 0.5, 1.0}; the budget = the smallest weight giving
the tactical family ≥ 5 % of the shared gradient norm). **V-YAW** (yaw augmentation; needs the perception targets
rotated — with WP-C/WP-D). Each with a same-flag replicate (the v7-tiny separated-CI false-positive rate is 14.3 %,
`H-ESTIM-SEED-1`).

## 8. Stamps every WP-B number carries

OPEN-LOOP single-shot on logged frames; one TRAINING seed of the trunk (T0r is the head-training replicate only); nav
and dense labels are ego-future derived (optimistic on PhysicalAI); 38 GT-turn episodes, so a denser window set narrows
within-episode noise only.
