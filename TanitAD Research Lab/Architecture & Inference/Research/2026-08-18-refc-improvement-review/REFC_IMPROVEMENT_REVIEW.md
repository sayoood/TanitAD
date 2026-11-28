# REF-C Improvement Review — six binding axes, from source and from the measured record

**Date:** 2026-08-18 · **Stream:** arch-inf (`agent/arch-inf-20260803`) · **Status:** review — 0 GPU spent.
**PI ask (verbatim):** *"Deeply review the REF-C approach to dramatically improve regarding all our
binding criteria — especially and not limited to: the action space learnings from Alpamayo, optimizing
comfort and jerk, the correct 6s horizon, the dramatic improvement of longitudinal performance, the
selection of the right trajectory, combination with MPC."*

**Method.** For each axis: (a) what REF-C does **today**, from source with `file:line`; (b) what the
programme has **MEASURED** about it (registry / raw eval JSON / banked package, never a summary);
(c) the **concrete change** with its expected mechanism, each pre-registerable. Where a candidate
improvement is **already refuted or already built**, that is stated instead of re-proposed.

**Estimator rule (carried by every number here):** episode-cluster bootstrap over the canonical val
episodes (`taniteval/ci.py`), paired form for two arms on the same windows; `overlapping_holdout_se`
is never used. Tier stamps per `EVAL_DOCTRINE.md`: **T0** = teacher-forced WM diagnostic, **T1** =
action-closed-loop, the primary capability tier. ⚠️ C122 found a tier **name collision** — the metric
suite's `tier0/tier1` (metric availability tiers) is unrelated to the T0/T1 eval tiers; this document
never uses the bare word "tier" for the metric-suite sense.

---

## 0. What REF-C is, in one paragraph (source)

REF-C = **Anchored-Diffusion-C**, a DiffusionDrive-style discriminative trajectory head
(`stack/tanitad/refs/refc.py:1-120`): a ResNet trunk (base 104.2 M / XL 251.9 M, MEASURED registry §4)
emits an 8×8×F conv map; a fixed FPS **anchor vocabulary** (base 128 / XL 256, `refc.py:297`) of
ego-frame trajectories is refined by cross-attention decoder passes emitting per-anchor confidence +
offset (`refc.py:1275-1531`); `steps=0` is a pure classifier, `steps>0` runs 2 truncated-diffusion
refinement passes (`refc.py:309`). Selection is `argmax` over a score surface that D-SEL
(`stack/tanitad/refs/refc_select.py`) rebuilt with gated, zero-init levers (S1/S1b/S1c, S2/S2b, S3,
S4, S5, S6), all default OFF. Aux heads: LAW latent world model `[pooled, traj] → pooled_{t+0.5s}`
(`refc_train.py:114`, `LAW_AHEAD = 5`), 5-way manoeuvre head with gated factorised lat×lon replacement
(D-TAC1, `refc_tactical.py`), route head, gated predicted-goal head (S6, `refc.py:1767-1835`), and the
REF-C.1 target-speed class head (gated `refc1`, `refc.py:613-616`). Trained end-to-end from scratch;
no part of the flagship latent-WM stack is loaded.

**Where REF-C stands (registry §4.1, MEASURED, offline open-loop):** REF-C-XL `ade_0_2s` full-set
0.4714 vs flagship v1 0.4522 — **statistically indistinguishable** (Δ +0.0443, CI95 [−0.0544, +0.1465],
paired episode-cluster bootstrap; the "0.006 m behind" phrasing is RETRACTED). REF-C **proposes** ~2×
better than the flagship line (oracle-in-fan 0.1640 XL / 0.1914 base) and **selects** at 0.4714 — the
program's largest measured propose/select split.

---

## 1. ACTION SPACE — the Alpamayo learnings

### 1.1 What REF-C does today (source)

* **Trajectory output:** 4 ego-frame waypoints at t = {0.5, 1.0, 1.5, 2.0} s
  (`TrajectoryConfig.horizons = (5, 10, 15, 20)` @ 10 Hz, `refc.py:292`) — position-only `[S, 2]`,
  no heading, no speed profile, no acceleration channel in the emitted plan. Under `refc1` the same
  four slots re-read as fixed-**distance** checkpoints at (2, 5, 10, 20) m (`refc.py:613-614`).
* **Anchor vocabulary:** FPS over a synthetic **unicycle** pool — each anchor is a constant-(v₀,
  yaw-rate, accel) rollout, v ∈ [0, 30] m/s, yaw-rate ∈ ±0.35 rad/s, accel ∈ ±3 m/s²
  (`refc.py:159-190`). Within an anchor, jerk ≡ 0 and yaw-acceleration ≡ 0 by construction; all
  action-space richness beyond that lives in the **unbounded offset head** (the source of the 72.08 %
  unflyable fan, `refc_select.py:44-57`).
* **Tactical action vocabulary:** shipped head is the 5-way mixed softmax `(lane_keep, turn_left,
  turn_right, accelerate, brake_stop)` (`refc.py:142`, `N_MANEUVERS = 5`); the **factorised lat(3) ×
  lon(3)** replacement EXISTS, gated default-OFF (`refc.py:434-437` `factored_maneuver`;
  `refc_tactical.py:134-166` — `COLLAPSE_TABLE` is the exact priority collapse `turn > brake > accel >
  lane_keep`). The factorised head + split H19 grafts cost +897 params; the F1 speed-input arm +384
  (`refc.py:716-746`, `refc_f1only_config`).
* **Tactical label horizon:** `LABEL_HORIZON = 20` — 2 s endpoint kinematics (`refc_tactical.py:155`;
  labels minted by `refb_labels.window_maneuver_labels` at `horizon=max(cfg.trajectory.horizons)`,
  `refc_train.py:497-498`).

### 1.2 What is MEASURED

* **Dual-axis manoeuvres are 40.62 % of driving** (MEASURED, n = 4,729 Alpamayo clips, 0 unparsable:
  `simultaneous_lon_and_lat` = 1,921/4,729 = **40.62 %**, **98 distinct joint cells** —
  `…/Data Engineering/Implementation/incoming/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md:86-89`,
  raw `a1_alpamayo_taxonomy.json:719-726`). A single mutually-exclusive 5-way simplex cannot
  represent 4 in 10 real manoeuvres.
* **Alpamayo's meta_action structure (MEASURED from `records.parquet`):** **three axes × 7 values
  each** — Longitudinal {Gentle/Strong Decel, Maintain, Gentle/Strong Accel, Stop, Reverse}, Lateral
  {Go Straight, Steer L/R, Sharp Steer L/R, Reverse L/R}, Lane {Lane Keep, Turn L/R, Lane Change L/R,
  Shift L/R} — **one declaration per clip at t₀**, no per-timestep index, no declared horizon
  (`TACTICAL_LABEL_VALIDATION.md:59-84`). It is not a lat×lon product but three parallel enum lines,
  with **severity grades** our ±1.0 m/s 3-way lon label collapses. ⚠️ Known quirks, measured: `Stop`
  short-circuits the other axes (its 304 lateral nulls ARE the 304 Stop rows) and is a **state, not
  an action** (ego v(t₀) = 0.51 m/s rising to 2.95 by 2 s); cross-leg LON-token consistency is
  78.06 % ⇒ **≈22 % expected label error** on reason-typed longitudinal tokens.
* **Where the Alpamayo declaration binds in time (MEASURED, 201 clips):** agreement with executed
  kinematics peaks on a **ridge at 0.5–3.0 s** (LON κ 0.3655 / LAT κ 0.4694 at H = 2.0 s, best
  thresholds) and **falls in the tactical band** — at PRODUCTION thresholds the (2, 6] band values
  are LON κ **0.1428** [0.0540, 0.2250] / LAT **0.1777** [0.0658, 0.2953] (C89b restatement;
  `TACTICAL_LABEL_VALIDATION.md:256-293` + the §3.1 correction). ⇒ Alpamayo meta_action supervision
  is a **t₀/2-s** signal, not a band signal.
* **The 5-way head's measured failure is exactly what the algebra predicts** (MEASURED, REF-C-base
  30k, n = 1364 windows / 39 episodes, `refc_tactical.py:37-58`): accelerate recall **0.0000** (0 of
  146), brake_stop **0.0256**, turns emitted at true rate (165 vs 174, 114 vs 109). The failure is
  the within-lane_keep longitudinal comparison — the **mixing** — not lateral dominance. The same
  defect is measured **program-wide at T1** on the flagship rescore grid: longitudinal decision κ
  **0.0405 (chance)** vs lateral 0.3795, and the collapsed 5-way (κ 0.1404) *"sits between the two
  axes and reports neither"* (`MODEL_REGISTRY.md:1703-1715`).
* **The label itself destroys 9.68 %** of windows (132/1364 carry a live longitudinal manoeuvre AND
  are labelled a turn — unrecoverable by any decode rule; `refc_tactical.py:83-88`). And the decode
  patch cannot reach the rarest class: **`accelerate` is unrecoverable at ANY τ** (recall peaks 0.153
  at τ = 0.5 then falls; `DTAC1_RESULTS.md` τ frontier). Only the **retrain** (F2) can fix these.
* **The decode-rule patch is REFUTED as a shipped default** (D-TAC1B, LOEO): brake recall 0.0719 →
  0.4248 at precision 0.2340 → 0.1711; on the 1232 representable windows the patch is **NOT
  separated from doing nothing** (macro-F1 +0.0107 [−0.0418, +0.0665]) and costs a **separated**
  accuracy loss (−0.1129 [−0.1861, −0.0471] on ALL). Recommendation on record: optional reporting
  mode at τ = 0.5, never the default (`DTAC1B_RESULTS.md:261-266`). **Do not re-propose.**
* **D6 (D-TAC1 probe):** the head is **READOUT-limited, not input-limited** — `auc_lon_active`
  **0.7294** vs the ≥0.65 threshold fixed in advance (shuffled control 0.4933;
  `…/2026-08-03-dtac1-tactical-head/dtac1_probe_refc-base-30k.json`). The longitudinal information
  is in the image embedding; the 5-way readout discards it. (The prereg's own INPUT-limited
  prediction was REFUTED — R-2026-08-03-dtac1.)
* **Alpamayo-2's OWN action space (PUBLISHED, read off its `config.json` in the banked analysis
  `…/Benchmarks & Eval/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md:31-70`):**
  `action_space_cfg: UnicycleAccelCurvatureActionSpace` — **64 waypoints at dt 0.1 s = 6.4 s**,
  parameterised as per-step **(accel, curvature)** with bounds accel ±9.8 m/s², κ ±0.33 1/m,
  integrated through a unicycle ⇒ **every emitted plan is dynamically feasible by construction —
  not free XY**. Decoder: flow-matching, 10 Euler steps, **non-causal over the whole 64-step plan**;
  frozen VLM backbone (`cotrain_expert_vlm: false`); plus a second, discrete trajectory-token path
  (4,000-token vocab, 128 tokens/future traj). Full SE(3) per waypoint. What it does that ours does
  not, in one line: **plans in bounded control space over the full horizon; we regress unbounded XY
  offsets over 2 s.**
* ⛔ **Parity guard for any Alpamayo-supervised arm (C112/C113, registry §12.4):** 201 of 4,729
  Alpamayo clips ARE the aug120 cohort inside parity train — and the honest rate is over the
  BUILDABLE set: 201 of 257 w120-built clips = **78.21 % contaminated (REF-A I-JEPA scale)**; the
  4.3 % catalogue rate is the flattering denominator, do not quote it. Reverse direction: 6 of the
  40 canonical val episodes are inside the Alpamayo record set (blast radius today zero; trigger is
  the 4,472-clip build). The exclusion oracle is
  `stack/tanitad/data/parity.py:1962-2034` — `assert_eval_clips_disjoint_from_parity_train` raises
  `ParityViolation` (*"Provenance is not disjointness; ids are"*); the sanctioned removal path is
  `filter_eval_clips` (`parity.py:2037-2067`), and any `sanctioned_audit` bypass stamps
  `decision_grade: False`.

### 1.3 The concrete changes

**A1 — fund the factorised lat×lon arms (F2/F1), already built and pre-registered, never trained.**
Mechanism: the factorisation removes the product coupling (`refc_tactical.py:17-35`) that makes the
longitudinal argmax conditional on lateral certainty; the retrain re-mints the 9.68 % of labels the
collapse destroys and is the only route to `accelerate` (unrecoverable at any τ). Status from the
record: **F2** (`factored_maneuver`, +897 params) and **F1** (`refc_f1only_config`, +384 params =
exactly one speed column into `maneuver_head.0`) are staged with tests; the four pre-registered arms
(`refc-base-30k` / `dtac1-full` / `dtac1-f2only` / `dtac1-nolon-graft`, + `dtac1-f1only`) have **0
GPU spent**. F1's pre-committed thresholds already exist (load-bearing = LON macro-recall separated
and ≥ +0.03). What remains is funding, not design.

**A2 — adopt the Alpamayo meta_action as an auxiliary dual-axis supervision target at the 2-s label
horizon** (label-side only — labels may use anything; inference stays vision-only). Mechanism: the
Alpamayo declaration carries what our kinematic thresholds discard — **severity** (Gentle/Strong)
and **fine lateral intent**: the n = 39 gate sweep measured the arms moving in opposite directions
as the yaw gate tightens, Alpamayo declaring nudges our severity-free vocabulary has no slot for
(declared κ 0.4660 vs ours 0.1159 at gate 0.01 rad). Supervising the factored head with a 40.62 %
dual-axis-rate target directly supplies the joint classes `COLLAPSE_TABLE` deletes. Scope guards,
all measured: (a) it is a **2-s-ridge signal — do not use it to label the (2, 6] band** (band κ
0.14/0.18); (b) treat `Stop` as a state, exclude or remap; (c) expect ~22 % reason-token label
noise; (d) ⛔ run `parity.filter_eval_clips` first — any eval on buildable Alpamayo clips is 78.21 %
contaminated until filtered.

**A3 — the deeper Alpamayo lesson is the ACTION SPACE, not the label:** adopt the bounded
(a, κ)-control parameterisation for REF-C's plan output — this is axis 2/3's central change (§2.3
J1, §3.3 H1) because it simultaneously fixes comfort-expressibility, the 72.08 % unflyable fan, and
the 6-s horizon. Stated here because its provenance is Alpamayo-2's `UnicycleAccelCurvatureActionSpace`
— and because v6 §4b already adopted exactly this form (*"ONE 60-step (a, κ)@10 Hz control"*,
`v6.py:45`), so REF-C converging on it is convergence with both the reference system and our own
binding spec.

---

## 2. COMFORT / JERK

### 2.1 What REF-C does today (source)

* **No comfort term anywhere.** The training loss is `TRAJ(L1, GT-nearest anchor) + anchor-cls CE +
  refined-CE (gated) + LAW MSE + route CE + manoeuvre CE + refc1 speed CE`
  (`refc_train.py:76-114, 602-607`). No jerk, no accel, no curvature, no yaw-rate penalty exists in
  any REF-C loss, score, or selector — grep-verified, and the only accel-shaped object in the model
  is the **selection-side** bounded-acceleration band (S2/S2b, `sel_accel_max = 2.5` m/s²,
  `refc.py:550`), which is a *feasibility filter*, not a comfort cost.
* **The plan cannot even express jerk at emission resolution:** 4 waypoints at 0.5 s strides support
  at most a second difference — acceleration at ~3 estimates per plan; jerk needs a third difference
  of a 4-point position sequence: exactly **one** jerk estimate per plan, at 0.5 s resolution, per
  axis. Comfort scoring on today's emitted object is numerically starved by construction.
* **The anchors are jerk-free segments** (constant accel/yaw-rate unicycle, `refc.py:174-190`), so
  candidate-level comfort differences live entirely in the learned offsets.

### 2.2 What is MEASURED

* **C101 (T1, primary tier): comfort terms in a cost function did not save the CEM planner.** The
  P2 cost is, verbatim (`taniteval/taniteval/planner_p2.py:33-37`): `J(plan) = w_v·(v̂ − v_target)²
  + w_c·(accel² + jerk²) + w_s·steer_rate² − w_p·progress`, weights `v=1.0, c=0.10, s=50.0, p=0.02`
  (`planner_p2.py:139-140`, engineered, not fit). Closed-loop T1 result (registry
  `MODEL_REGISTRY.md:2637-2642`; raw `…/Benchmarks & Eval/Implementation/incoming/
  2026-08-18-planner-beats-cv-redrive/raw/planner_beats_cv_banked_analysis.json`, n = 221 windows /
  20 episodes, paired episode-cluster bootstrap): **planner − CV +0.2585 m [+0.0869, +0.4309],
  p(δ>0) = 0.9975 — the CEM planner is 35.8 % WORSE than constant velocity**, and it loses on
  LONGITUDINAL, its own target family (per-horizon RMSE @2 s **1.9062 vs CV 1.6705 m**; speed error
  0.9431 vs 0.7607 m/s; bias +0.2737 vs −0.0995 m/s). Meanwhile **operative-under-true-actions − CV
  = −0.3151 m [−0.6277, −0.0602]** — the WM rolls out *better* than CV when handed true actions.
  Registry `:2641-2642`: *"THE LOSS IS IN THE ACTION SEARCH, NOT IN THE WORLD MODEL."*
  ⚠️ Scope: this verdict is the **closed-loop T1** one; the older open-loop `planner_beats_cv`
  remains UNDECIDED (registry `:2650-2654`). ⇒ comfort/jerk as a *search-cost* term is downstream of
  a working search; REF-C's measured defect is the *ranking*, so a comfort term bolted onto today's
  ranking binds at the same broken surface.
* **Comfort instrumentation exists — outside the binding block.** taniteval computes jerk/comfort in
  five places (`closedloop.py:640-668` `_comfort` with `JERK_COMFORT = 2.0 m/s³`;
  `generalization.py:237-249` `jerk_max`; `tanitad_metrics.py:165-202` TMS; `bench.py:75-97`
  `tms_openloop`; `pseudosim.py:1206,1245-1256`) — but **`four_families.py` contains no jerk and no
  comfort** (grep-verified): comfort is deliberately NOT one of the four binding families.
* ⛔ **C46 — a comfort pass/fail composite REWARDS NOT DRIVING** (MEASURED, `RETRACTION_LOG.md:
  680-691`): the human's own logged path fails the comfort bounds on **16.60 %** of windows while
  `cv_holdv0` and `stand_still` both score a perfect 1.0000; every learned planner floors on the
  jerk clause. Weight set to 0.0, kept as diagnostic. Standing consequence: *validate any threshold
  against the ground truth before it carries weight* — and prefer barrier forms above the human p99
  (accel 2.689 m/s², jerk 6.369 m/s³, measured) over `λ·jerk²`, which punishes legitimate emergency
  braking (`…/2026-08-06-v1-defect-triage/UNICYCLE_RETRAIN_PLAN.md:43-53`).
* ⭐ **The strongest comfort result in the program is a PARAMETERISATION, not a cost** (MEASURED,
  v1.6 `unicycle-readout-v2-latentsonly`, registry `MODEL_REGISTRY.md:1179-1259`, 6,834 windows,
  paired episode-cluster bootstrap): a free per-step `dx` head produces **jerk RMS 52.13** (30.6×
  the human floor 1.71 — *"a delta head's natural output scale IS THE JERK"*), while the
  `UnicycleStepReadout` — controls integrated through a unicycle, with p99-barrier losses — lands
  **jerk RMS 1.1334 (Δ −35.03 [−46.69, −24.81], separated)**, accel RMS 2.95 → 0.72, replan accel
  jump 11× lower. Same lesson from the other side: retiming alone took 52.13 → 4.95.
* ⚠️ **Correction to a tempting mis-citation:** in the deployed `flagship4b-speedjerk-30k`, **jerk
  was never an action input** — `--jerk-weight 0.02` is a *loss* weight and `aux_accel` a 528,897-
  param aux head; **speed as the 3rd action channel is the proven lever** (D-A3: no-speed 2.918 vs
  speed 0.452, paired +2.21 m [2.04, 2.39]) and **no arm anywhere isolates the jerk loss**
  (`MODEL_REGISTRY.md:166-197, 2898`). Jerk-as-cost is unproven program-wide; jerk-as-
  parameterisation is proven (previous bullet).

### 2.3 The concrete changes

**J1 — comfort by construction: re-parameterise anchors AND offsets in bounded (a, κ) control space,
integrated through a unicycle (the Alpamayo/v6-§4b form; the v1.6-measured mechanism).**
Mechanism, all three legs measured: (i) REF-C's anchors are ALREADY unicycle rollouts at dt 0.1 s
(`refc.py:159-190`) — only the **unbounded XY offset head** breaks feasibility, and it is exactly
what makes 72.08 % of the fan unflyable (`refc_select.py:44-57`); (ii) moving the emitted object
from free XY to integrated controls is the mechanism v1.6 measured as **jerk RMS 52.13 → 1.13,
separated** (§2.2); (iii) it is the reference system's own action space
(`UnicycleAccelCurvatureActionSpace`) and our binding spec's own form (`v6.py:45`). Concretely: the
offset head emits per-step (Δa, Δκ) around the anchor's (a, κ), clamped to the physical bounds, and
the trajectory is the unicycle integral — S2's reachability band becomes **structural** (nothing
unflyable can be emitted), per-candidate jerk/accel become dense, honest quantities, and comfort
enters as **p99-barrier losses** (the v1.6 form), never a `λ·jerk²` and never a pass/fail composite
(C46). Pre-registered read: four families + jerk/accel RMS vs the human floor (~1.71 / ~0.91),
paired vs the XY incumbent at matched steps; the S2/S2b constants and the 72.08 %/2.78× figures are
re-measured, never inherited, on the new fan (`refc_select.py:210-213`).

**J2 — measure before optimising: report the comfort diagnostics for REF-C arms.** The instruments
exist (`closedloop._comfort`, `tms_openloop`, `generalization.jerk_max`) but no REF-C eval reports
them; comfort is *deliberately* outside the binding four families, so it is reported as a diagnostic
block beside them, never pooled and never gated (C46). 0-GPU on banked fans/dumps; precedes any
comfort training claim.

⚠️ Sequencing from C101: a comfort *cost* into today's broken ranking would re-run the CEM mistake.
J1 avoids this by construction (it changes what candidates ARE, not how they are scored); any
comfort *score term* (a graft on the rank surface) waits for the axis-5 selection repair verdict.

---

## 3. THE 6 s HORIZON

### 3.1 What REF-C does today (source)

* **REF-C decodes 2.0 s, four waypoints, 0.5 s strides** (`refc.py:292`), and every piece of its
  supervision is pinned to that: trajectory targets (`refc_train.py:411-412`), manoeuvre labels at 2 s
  (`refc_train.py:497-498`), LAW at +0.5 s (`refc_train.py:114`), selection-band `horizon_s` derived as
  `max(horizons) × 0.1 = 2.0` (`refc.py:635`). The anchor pool is integrated to `max_h` = 20 steps
  (`refc.py:181-190`). There is **no 6-s option anywhere in the REF-C stack** (grep: no horizons
  override in any config or script).
* The binding spec, verbatim (`stack/tanitad/models/v6.py:147-151`):
  ```
  PLAN_STEPS = 60
  DT = 0.1                       # 10 Hz tick — the dense-horizon contract
  HORIZON_S = PLAN_STEPS * DT    # 6.0 s
  OP_BAND_S = (0.0, 2.0)         # operative band — fine control authority
  TAC_BAND_S = (2.0, 6.0)        # tactical band — same controls, g_tac-shaped
  ```
  REF-C therefore covers **only the operative band**; the entire tactical band (2, 6] s is outside
  its plan, its labels, and its metrics — and **2.0 s is the SEAM, not the tactical band** (the whole
  point of the C89 correction).

### 3.2 What is MEASURED

* **C89b — the band's kinematic activity, measured AT THE BAND** (201 clips, PRODUCTION thresholds,
  statistic `mean_band` = mean in-band deviation from the band start, episode-cluster bootstrap;
  `RETRACTION_LOG.md:4741-4748`): **LON κ 0.1428 [0.0540, 0.2250] / LAT κ 0.1777 [0.0658, 0.2953]**
  (n = 201/193). The seam (0, 2] reads 0.3270/0.3132, the full horizon 0.2210/0.3806; paired
  band − seam is CI-separated on both axes (LON Δκ −0.1843 [−0.2746, −0.0961]). ⛔ C89's replacement
  numbers 0.2331/0.4040 are **not band values** — *"the highest-propagation-risk number in the log"*.
  Root-cause class: **a stratum boundary is a fit window** — the same trap as the learning-curve
  exponents, and the same lesson as C121.
* The oracle-in-fan numbers (0.1640/0.1914) and the 45.4 %/41.09 % mis-ranking rates are all **@2 s**;
  nothing is known about REF-C's fan quality in (2, 6] because **no 6-s REF-C fan has ever been
  decoded**.
* Reference points: DiffusionDrive (REF-C's design source, arXiv 2411.15139) plans NAVSIM's 4-s
  regime {PUBLISHED}; Alpamayo-2 plans **6.4 s at 10 Hz in bounded control space** {PUBLISHED, §1.2};
  our 2-s/4-slot choice is inherited from the REF-B tactical horizons (`refc.py:288-291`), not from
  any spec.

### 3.3 The concrete changes

**H1 (primary) — the 6-s decode in (a, κ) control space: 60 steps @ 10 Hz, unicycle-integrated** —
the same re-parameterisation as J1, extended to `PLAN_STEPS = 60`. One change buys three binding
axes at once (comfort, horizon, feasibility), lands REF-C exactly on the v6 §4b contract, and gives
the fan a tactical band for the first time — the manoeuvre/goal grafts finally act on a surface
whose consequences extend into the band they describe. Anchor vocabulary REBUILT as FPS over 60-step
control rollouts (the pool generator already integrates arbitrary `max_h`, `refc.py:181-190`); every
reachability constant re-measured on the new fan (`refc_select.py:210-213` — the 2.78×/72.08 %
figures are properties of *those* anchors, never inherited).
**H1-fallback (low-risk) — sparse-slot extension** `horizons = (5, 10, 15, 20, 30, 40, 50, 60)`,
keeping free-XY offsets: pure config surface (`refc.py:292`; `selection().horizon_s` follows at
`refc.py:635`; `waypoint_targets` horizon-generic), fully comparable on the shared @2 s slots — but
it inherits the unbounded-offset defect into a band where offsets are larger, so it is the fallback,
not the primary.
Pre-registered read for either arm: four families **per band** (OP (0, 2] and TAC (2, 6] separately;
C121: strata pre-registered — the band split here is edge-free by construction because §4b fixes the
edges), paired vs the 2-s incumbent on the shared @2 s slots.

**H2 — split trajectory supervision by band** with per-band weights declared in the prereg (an
unweighted mean over a 6-s plan silently re-weights the loss toward the far band by arc length).
Not tuned post hoc.

**H3 — tactical labels move to the band (= A3's label half):** mint lat×lon at the (2, 6] endpoint
(`window_factored_labels(…, horizon=60)` is parameter-generic, `refc_tactical.py:234-250`) so the
tactical head answers the band's question. C89b's band values (LON κ 0.1428 / LAT 0.1777) are the
measured prior for how much weaker the band signal is than the seam signal — and the reason A2's
Alpamayo supervision stays at its 2-s ridge while H3's kinematic labels own the band: **two labels,
two horizons, disjoint jobs.**

---

## 4. LONGITUDINAL PERFORMANCE

### 4.1 What REF-C does today (source)

* **Longitudinal supervision:** (a) the trajectory L1 (which pools lat+lon); (b) the manoeuvre CE whose
  longitudinal classes the 5-way argmax never emits (recall 0.0000/0.0256, `refc_tactical.py:42-47`);
  (c) the gated `refc1` target-speed class head (4 bins over [0, 30] m/s, `refc.py:615-616`,
  `SPEED_CLS_WEIGHT = 0.2`, `refc_train.py:81`). **No distance-keeping, headway, TTC, or lead-agent
  signal exists anywhere in REF-C's inputs, losses, or scores** (grep: no lead/TTC/gap symbol in
  `refs/refc*.py`, `scripts/refc_train.py`).
* **The tactical head is speed-blind at inference** unless the F1 arm is on: `maneuver_head(pooled)`
  reads the image embedding alone; v₀ reaches only the measurement encoder → decoder condition
  (`refc_tactical.py:71-82`). The F1 fix is built and gated (`tactical_speed_input`, `refc.py:438-448`).
* **Selection-side longitudinal:** the E-OBJ-1 `softade` objective is implemented
  (`refc.py:504-529`) and its measured recovery is **longitudinal** (`speed_abs` −0.1102 base /
  −0.1816 XL; `refc.py:507-511`) — but the incumbent objective is still the one-hot CE
  (`sel_ce_objective = "ce"`, `refc.py:522`).

### 4.2 What is MEASURED

* **88.7 % of the oracle gap is longitudinal** (registry `MODEL_REGISTRY.md:1582-1583`, verbatim:
  *"88.7 % of the oracle gap is longitudinal, and this is the missing longitudinal state variable"*
  — the P1 lead-gap resolution context, n = 266 vehicle-lead windows). **Sharpened at T1 to ~99 %**
  (`:1634-1635`: v5f along-track 23.8965 of 23.9837 total vs 0.9993 lateral). The longitudinal axis
  is not one axis among four; at the primary tier it is nearly the whole gap.
* **C122 (distance-keeping, MEASURED, banked at d2ede52b):** the flagship is indistinguishable from
  GT on all three distance-keeping metrics while **both frozen-DINOv2 arms are CI-separated from GT
  on all three, in the UNSAFE direction** — REF-A min-TTC **−5.82 s [−9.34, −2.06]** vs flagship
  **−0.16 s [−1.10, +0.71]**. Companion finding (same commit): the "91× reads better" lead-*readout*
  advantage buys **nothing on lead windows** (lead-vs-no-lead deficit contrast −0.0146
  [−0.5988, +0.5551], NOT separated) — reading the lead and *driving* the gap are different
  capabilities, so a lead-readout probe is not a substitute for gap supervision or gap metrics.
  The three metrics (`…/2026-08-18-refa-reconciliation/REFA_RECONCILIATION.md:238-244`, [metric-suite
  tier T0], paired episode-cluster bootstrap, n ≈ 218–240 windows / 19 episode clusters):
  **`headway_min` (m), `time_gap_min` (s), `min_TTC` (s)** vs GT — flagship not separated on any;
  REF-A separated on all three (headway −1.4180 [−2.3983, −0.3815]; time-gap −0.3152; min-TTC
  −5.8223). ⚠️ Caveats that travel: TTC is **dt-dependent** (banked spacing 0.5 s, scales 1/dt);
  censoring at `TTC_CAP_S = 30` on ~45 % of windows; headway/time-gap are dt-invariant.
* **The lead data EXISTS now, and the eval-side refusal is still in the tree:** `lead_source.py`
  turns `obstacle.offline` into the `win["lead"]` block (`lead_source.py:1-60`; strictly causal lead
  pick, LEAD/NO_LEAD/NO_LABEL three-state — val40: 270/551/60) and attaches row-for-row (881 = 881,
  episode partition identical, speed corr 1.0). **But `driving.py:606-608` still ships the refusal**
  `"headway_ttc_distance_keeping": "no lead-agent state exists (lead_state is a None stub)"` —
  unchanged since `df32781a` (2026-07-25). The stale blocker is *diagnosed* (C122), not yet *fixed*.
  Train-side: the obstacle join now covers the train corpus too — **2,308 episodes / 12,122,129
  agent boxes** (`…/2026-08-17-train-obstacle-join/TRAIN_OBSTACLE_JOIN.md:1,12`; the D1 slot-probe
  claim was withdrawn, the join itself unaffected).
* ⚠️ REF-C is in NONE of this: C122's arms are the flagship and the frozen-DINOv2 arms. **REF-C has
  never been scored on distance-keeping at all** — its longitudinal record is ADE-decomposition and
  the manoeuvre confusion only.
* **E-OBJ-1** (frozen 30k weights, 881 windows, LOEO, paired episode-cluster bootstrap): swapping the
  fitted ranker's objective CE → `softade` recovers **−0.0974 m (base) / −0.1670 m (XL)**, separated,
  and the recovery is longitudinal (`refc.py:507-511`). The `softce` control is separated WORSE
  (+0.0909 m base) — metric-awareness, not target-softness, is the active ingredient (`refc.py:512-517`).
* **REFUTED, do not re-propose:** a target-speed term in the *selection score* — REF-C v1.0 measured
  0.0 % recovered, best blend point is the untouched baseline, pure cost −171 % (registry §4.1,
  ~line 2218). The lever is the ranking's *objective*, not a bolt-on speed cost.

### 4.3 The concrete changes

**L1 — train a D-SEL arm with `sel_ce_objective = softade`** (S1+S1b+S1c+S2 on, per the D-SEL prereg
ladder). Mechanism: E-OBJ-1 measured the CE→softade swap recovers longitudinal error *on frozen
weights*; in-training the ranking objective additionally shapes the refined readout (the "different
in kind vs v1.2" argument the prereg commits to test). This is the single highest-leverage
longitudinal lever REF-C has that is measured, built, and unfunded. Note `sel_ce_weight` must be
explicitly decided — the objective changes units NATS→metres (`refc.py:519-529`).

**L2 — make the lead block first-class in REF-C's eval, then in its supervision.**
(a) **Instrument first (0-GPU): retire the `driving.py:606-608` refusal** by wiring `lead_source`'s
`win["lead"]` block into the distance-keeping family, then score REF-C's banked fans/dumps on
headway_min / time-gap_min / min-TTC exactly as C122's instrument did for the flagship arms (dense
0.1 s spacing where available; state the dt with any TTC). REF-C has never been measured on
distance-keeping — after C122, an ADE-only longitudinal story is known to be able to hide
unsafe-direction gap behaviour. (b) Then label-side supervision: a lead-conditioned longitudinal
target (time-gap class, or a gap-aware refinement of the 3-way lon label) minted from the 12.1 M-box
train join — labels may use ego+others; inference stays vision-only. Mechanism: C122 shows
lead-blind supervision can look fine on ADE while CI-separated unsafe on gap metrics; REF-C's
longitudinal path currently has zero gap-awareness end to end.

**L3 — fund the F1 speed-input arm** (built: `refc_f1only_config`, +384 params = one speed column) —
D6 measured the head readout-limited (`auc_lon_active` 0.7294), so F2 (structure) outranks F1; F1's
value is as the *attribution control* the 2026-08-03 decoupling explicitly created
(`refc.py:47-53`), with its thresholds already pre-committed (load-bearing = LON macro-recall
separated and ≥ +0.03). Run both, read per-axis. The measured negative control is stark: with the
flag off, `maneuver_logits` is **bit-identical from v0 = 0 to 25 m/s** — that IS the defect.

**L4 — refc1's target-speed head: never trained; pre-register or retire, do not tune.** Every
trained REF-C run carries `refc1: false` (registry §4.1 config row, `MODEL_REGISTRY.md:2162`), so
the head's value is UNKNOWN, not weak. Its current shape is also under-resolved for the job: 4 bins
over [0, 30] m/s = ±3.75 m/s quantisation against a longitudinal lever measured in fractions of a
m/s (v1's +0.66 m/s speed over-prediction). If ever funded, bin width is set from the measured error
scale in the prereg — but it ranks behind L1/L2, which attack the same axis with measured mechanisms.

---

## 5. TRAJECTORY SELECTION

### 5.1 What REF-C does today (source)

The **entire D-SEL surface exists in source, default-off** (`refc_select.py` module docstring is the
canonical defect list; `refc.py:1275-1531` the implementation):

| lever | what it does | status (source) |
|---|---|---|
| S1 `sel_refined` | rank the refined fan with the refined score (fixes D1: today the t=0 classifier score ranks post-denoise trajectories, `refc.py:1386-1407`) | built, off |
| S1b `sel_score_emitted` | score the fan that is actually EMITTED (today no head ever scores it — the last pass scores its own input, `refc.py:1409-1434`) | built, off |
| S1c `sel_ce_reach` | CE normalises over the survivor set the argmax ranks over (today: full-fan softmax over a 72-74 % unpickable fan, `refc.py:486-499`, `refc_train.py:446-467`) | built, off |
| S2/S2b reach band + anchor prefilter | bounded-accel candidate band; pre-decode variant is output-exact, selection index identical 881/881, 2.78× fewer decodes (`refc_select.py:191-226`) | built, off |
| S3 `graft_cons` | per-candidate consequence via `law_head`, zero new params (`refc_select.py:347-397`) | built, off |
| S4 `seam_clamp` | norm cap on graft totals (the instrument existed, no actuator — D4) | built, off |
| S5 `graft_route` / S6 `graft_goal` | route readout / predicted geometric goal reach the ranked score through zero-init gates (`refc.py:1440-1451`) | built, off |

Shipped behaviour remains: **argmax of the t=0 classifier confidence over the refined fan**
(`refc.py:1437` `base = refined if sel.refined else conf`, with `sel.refined = False`).

### 5.2 What is MEASURED

* **The selector is the defect, not the curve:** selected 0.4714 vs oracle-in-fan 0.1640 (XL; base
  0.4728/0.1914), `sel_gap` 0.3075 [0.2397, 0.3778], pick >2× worse on **45.4 %** of windows
  (`refc_select.py:18-33`; raw: `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`).
  The same shape holds on the flagship side: v5f's eval-grade **sel_gap 0.2036 m** (ade 0.4011 vs
  oracle 0.1975, [T0], 881 windows, `MODEL_REGISTRY.md:1039-1047`), registry §1.8 heading verbatim
  *"the fan is good and the SELECTOR is the defect"* (`:1096`; the trainer-log sel_gap series
  carries its own scope stamp, `:1112`), and v5.8f's whole deficit is selection (sel_gap 0.374,
  `:1425-1426`).
* ⛔ **The gap is NOT available headroom:** registry §4.1 standing caveat — ~92 % irreducible; v1.2's
  learned re-scorer over **47 trained arms** recovered ≤8.4 % and its headline is NOT separated
  (+0.00893 [−0.0062, +0.0250]). Any selection prereg is written under this adverse prior
  (`PREREG_D-SEL…` §1).
* **E-SEL-0:** the unsupervised refined readout ranks **0.8372 m (base) / 0.9187 m (XL) WORSE** than
  the shipped t=0 score — separated, both arms — while scoring 8.7×/16.6× chance
  (`refc.py:470-477`). S1 must **climb out** (supervise the refined readout), not harvest it.
* **E-S1-0 3.1:** under the one-hot CE, every fitted ranker is separated WORSE than the incumbent —
  including feature sets containing the incumbent's own score (C-leak gap −0.001 to −0.003 m, not
  overfitting; `refc.py:530-545`). The one-hot target over ~128 near-duplicates is itself the
  pathology; E-OBJ-1's `softade` is the measured repair (§4.2).
* **E-WC — the winner's-curse law, measured ON REF-C'S OWN FAN** (`V6F_PLANNER_DESIGN.md:46-71`,
  banked REF-C-XL fan, 881 windows / 256 candidates; replicated on base): the roll-consistency score
  (W7's quantity) has argmax error-rank **RISING** with N (0.241 → 0.286; lower-tail hit collapsing
  0.57 → 0.28) while **REF-C's supervised selector on the same fan moves the opposite way** (rank
  0.099 → **0.014**, lower-tail 0.77 → **0.99**). Top-m aggregation is refuted as a remedy (top-8
  medoid **+0.1294 m [+0.0645, +0.2029] WORSE**, paired-separated) — and so is the flagship-side
  kinematic top-k re-rank (W1: sel ADE 0.4011 → 0.4351, **−16.7 %**, `MODEL_REGISTRY.md:1307`), and
  REF-C v1.0's hand-cost re-rank (`taniteval/refc_rerank.py` reuses the P2 cost verbatim; 0.0 %
  recovered). **Do not re-propose any of these.**
* **SEL-1 is REFUSED — the admission gate FIRED 2026-08-16** (`V6F_PLANNER_DESIGN.md:645-686`,
  E-WC2): a `GoalDistanceScorer`-style selector is admissible only if a goal can be predicted at
  **σ* ≈ 0.8 m at 2 s** (≈1.7× selected ADE, per-axis); the measured σ of a ridge on **frozen REF-C
  latents** is **4.7104 m [3.8087, 5.6860] → σ/ADE 9.99 [7.45, 13.51]** — REFUSED at 2.48× the
  refusal threshold's lower bound. **But the estimand survives the surface:** a **0-parameter
  constant-yaw-rate kinematic extrapolation reaches σ(2 s) = 1.1888 m — 3.96× better than the
  latent ridge** (*"these latents are the wrong surface"*; still σ/ADE 2.52, not FUNDED). Scope:
  T0-DIAGNOSTIC, REF-C surface — the S-W-latent reopening (σ ≤ 0.80 m ⇒ FUNDED, > 1.41 ⇒ REFUSED
  and `ANCHOR_GOAL` is the line) awaits one ~10–25 GPU-min dump; the dumper landed 2026-08-18
  (`v6_dump_sw_latents.py`, roundtrip-verified on planted σ at all three verdicts) and is not yet on
  Thor.
* **C121:** a selection gate's verdict can move with stratum band edges — pooled P7 (ρ 0.4656,
  passes) hides a failing lead-interaction stratum (`LEAD_20_40m` ρ 0.0973 [−0.2664, +0.4400]), and
  the 3-band splits fail at 20/40, 15/35, 14/45 and the median cut. Any REF-C selection prereg
  pre-registers its strata; primary read edge-free (LEAD vs NO_LEAD, n = 270/21 ep).

### 5.3 The concrete changes

**S-A — fund the D-SEL ladder as pre-registered** (S1+S1b+S1c+S2 with `sel_ce_objective = softade`;
S3 as a separately-gated follow-on arm). Everything is built; the review's contribution is the
priority argument: E-OBJ-1 + E-SEL-0 + E-S1-0 together localise the defect to the *objective* and
the *scored object* — both fixable only in-training, which no post-hoc re-scorer (v1.2, 47 arms)
could reach — and E-WC certifies that REF-C's supervised selector is in the **good** structural
family (error-rank falls with N), so supervising it harder is aimed at a mechanism with the right
shape. Answering the brief's question directly: **which measured selector lesson does
`refc_select.py` violate? None in design — but the shipped default still violates all five measured
defects (D1–D5), because every repair is default-off. The violation is a funding state, not a code
state.**

**S-B — E-WC shape read as the standing admission test for any NEW rank graft.** The winner's-curse
instrument (`sel_winners_curse_law.py`) already ran on REF-C's fans (that is how E-WC was measured —
do not re-run it as if novel); what remains is applying it as the pre-registered admission check to
scores that have never had it: the **S3 consequence score** and the **S6 goal-compatibility terms**,
on the banked `fan_refc-*-30k.pt`, 0-GPU, strata pre-registered per C121 (edge-free LEAD/NO_LEAD
primary). A graft whose score shows the rising-rank shape is refused before it costs an arm.

**S-C — a param-free KINEMATIC goal prior on the rank surface, and an adverse prior on S6's
distance half.** E-WC2's climbout measured that on this substrate a constant-yaw-rate extrapolation
is the best available goal estimate (σ 1.1888 m, 3.96× better than a ridge on REF-C's own pooled
latents). Two consequences: (a) REF-C's S6 `goal_dist` half — predicted from `pooled` alone —
carries a **measured adverse prior** on top of the K7 confound its own docstring declares
(`refc.py:1904-1913`): pre-register it as expected-inert; the bearing half remains the live half.
(b) The cheap lever worth a 0-GPU banked-fan read first: a constant-yaw-rate goal point entering
the ranked score through the SAME param-free geometric compatibility LAN uses (`refc.py:1447-1448`),
gated zero-init and admissible — it is predicted (extrapolated), geometric, and carries no
situation-classifier output; v0/yaw are already model inputs (the measurement channel), not new
privileged signals. If the banked read shows the Goal-Distance shape, it graduates to an arm;
verdicts committed in advance. ⚠️ Its σ/ADE is 2.52 — *below* the σ* funding bar — so the
pre-registered expectation is **modest or null gain**; the read is cheap precisely because the prior
is adverse.

---

## 6. MPC COMBINATION

### 6.1 What REF-C does today (source)

* **REF-C has no planner loop and no plant model.** Its world model is `law_head`:
  `[pooled, traj] → pooled_{t+0.5s}` — consumes a trajectory, emits a pooled vector, **cannot be
  iterated** (no fmap to re-decode from; `refc_select.py:89-97`). A probe-style rollout is
  *structurally unavailable*; the only expressible imagination is one consequence per candidate (S3).
* Closed-loop today: the AlpaSim NuRec suite drove REF-C via `refc_driver.py` (registry §4.4) —
  **RECONSTRUCTION-OOD CONFOUNDED** (open-loop ADE on the reconstructions is 1.52 m vs 0.4714
  canonical); REF-C has **no T1 adapter** (`taniteval/tools/t1_eval.py` rolls the flagship/v5f latent
  stack; no refc reference in it — grep 2026-08-18).

### 6.2 What is MEASURED

* **C101's localisation:** the flagship's CEM lost to CV **in the action search, not the world
  model** — planner − CV **+0.2585 m [+0.0869, +0.4309]** (T1, separated) while
  operative-under-true-actions − CV = **−0.3151 m [−0.6277, −0.0602]** (§2.2 has the full block).
  The CEM's cost carries **no gap/TTC barrier — by explicit v0 decision, not oversight**
  (`planner_p2.py:36-37`: *"(+ gap/TTC barrier — SKIPPED in v0: no lead-agent labels in our
  front-cam+pose data)"*, and no such term exists in `cost_fn`, `planner_p2.py:177-194`) — there
  were no lead boxes then; **there are now** (12.1 M-box train join, val40 lead block).
* REF-C's fan is a **discrete, decoded, scoreable candidate set** — precisely the object a
  sampling-based MPC re-solves per step, and precisely what the CEM had to *construct* by sampling
  action sequences through a rollout.

### 6.3 The concrete change

**M1 — anchor-fan MPC: receding-horizon re-decode with warm-started selection, not action-space
search.** What the MPC optimises over: **the decoded fan** (anchors + offsets, post-H1 a 6-s fan),
scored by the D-SEL surface (incl. S3 consequence and, once leads are first-class, a gap/TTC barrier
term entering as a ranked-score graft with a zero-init gate — the term planner_p2 never had). The
plant model: **none beyond 1-step** — and that is the point: re-decoding from fresh perception each
0.5 s IS the plant update; the model never integrates its own dynamics forward, so it cannot inherit
the CEM's compounding-search failure, because there is no action-sequence search — selection is exact
argmax over N decoded candidates (S2b keeps it 2.78× cheap). Consistency across replans (the comfort
coupling): a hysteresis prior on the previously-selected candidate entering the ranked score as a
logged, clamped graft — replan jitter is exactly where jerk enters a receding-horizon system.
⚠️ **This mechanism already exists v6-side — port it, do not reinvent it:** F-8 is the
*"T5 temporal-consistency selection loss … + plan-switch-rate logging"* cell, built at **+0 params**
(`DIAGRAM_CONFORMANCE.md:213`, `F7_F8_CELLS.md`), with the measured warning that it is **degenerate
alone** (a flat plan scores exactly 0) — so the REF-C port carries F-8's guard, and the instrument
for the read exists (replan accel-jump was probed once: `o6_replan_accel_jump_gtframe`,
`t1_eval.py:33-35`; v1.6 measured replan accel jump 1.1310 → 0.1016 under the unicycle readout).
Why it would not inherit C101's failure, stated falsifiably: C101 localised the loss to *searching
the action space against the WM*; M1 searches nothing — it ranks decoded candidates under the same
selection surface whose in-training repair is axis 5. If axis-5 arms fail to close `sel_gap`, M1
inherits that failure — M1 is therefore sequenced AFTER S-A, not next to it.
Prerequisite instrument: a REF-C T1 adapter (closed-loop re-perception at 2 Hz replan over the val40
cache), because the binding capability tier for any "drives better" claim is T1 — today REF-C cannot
even be measured there.

---

## 7. Priority order (all pre-registerable; none hand-waved)

| # | change | axes | cost | gate it answers to |
|---|---|---|---|---|
| 1 | S-A = L1: fund the D-SEL ladder with `softade` | 5+4 | 1 arm (30 k) | D-SEL prereg §6 (written) |
| 2 | L2a: retire `driving.py:606-608`, distance-keeping on REF-C arms | 4 | 0 GPU | four-families rule + C122 |
| 3 | S-B: E-WC shape read on the S3/S6 scores (banked fans) | 5 | 0 GPU | new prereg, strata per C121 |
| 4 | J2: comfort diagnostics on banked REF-C fans/dumps | 2 | 0 GPU | C46-safe reporting block |
| 5 | J1+H1: (a, κ) 60-step control-space re-parameterisation | 2+3+1 | anchor rebuild + 1 arm | new prereg (v1.6 + Alpamayo-2 + v6 §4b) |
| 6 | A1+H3: D-TAC1 arms (F2/F1) with band labels | 1+3 | 1–2 arms (can share the H1 retrain) | D-TAC1/D-TAC1B preregs (written) |
| 7 | A2: Alpamayo dual-axis aux supervision at 2 s | 1 | label build + arm | new prereg + `parity.py` §10 filter first |
| 8 | L2b: lead-conditioned longitudinal supervision | 4 | label build + arm | new prereg (train join exists) |
| 9 | S-C: param-free kinematic goal prior, banked-fan read | 5 | 0 GPU | new prereg (adverse prior stated in advance) |
| 10 | M1: anchor-fan MPC + REF-C T1 adapter (F-8 port) | 6 | adapter + closed-loop eval | after the S-A verdict |

Dependencies stated once: M1 and any comfort/goal *score term* sit behind S-A, because C101 measured
that search/costs on top of a broken selection surface lose to CV — the selection repair is the
trunk of this tree. J1+H1 is independent of S-A (it changes what candidates ARE, not how they are
scored) and is the one change that moves three binding axes with a single retrain.

---

## 8. Deliverable manifest

| artifact | where |
|---|---|
| this review | `TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-refc-improvement-review/REFC_IMPROVEMENT_REVIEW.md` (repo, staged) |
| evidence dossier A (selection/longitudinal/horizon/MPC measured record) | same dir, `EVIDENCE_DOSSIER_A.md` (staged) |
| evidence dossier B (Alpamayo action space + tactical history) | same dir, `EVIDENCE_DOSSIER_B.md` (staged) |

Integration: escalated in the turn report (not buried here).
