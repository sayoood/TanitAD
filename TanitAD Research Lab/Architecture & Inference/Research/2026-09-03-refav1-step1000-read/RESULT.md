# refav1 — first T1 read: the retired fp32 incumbent at step 1,000 (20-clip slice)

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refav1-step1000-read/`
**Read:** 2026-09-03 00:22–02:54 Berlin (22:22–00:54Z), dev-box RTX 4060, detached (`run_read.cmd`), EXIT 0.
**Evidence class:** MEASURED — `raw/refav1_t1_step1000.json` (388,352 B), `raw/read_step1000.log`, dump manifest `raw/t1_dump_manifest.json`.
**Tier:** T1 for cl / ha / cl_navshuf (action-closed loop, the PRIMARY tier); T0 for ol (recorded actions) and cl_oraclegoal (true future field as goal).
**Estimator:** point = full-set pooled mean over windows; intervals = paired episode-cluster bootstrap (`taniteval/ci.py`, 2,000 resamples), never `overlapping_holdout_se`.
**n:** 20 episodes, 140 windows (stride 10), K = 10 steps (2.0 s), K_wm = 30. Nav shuffle changed 58/140 windows.
**Checkpoint:** the RETIRED incumbent (`refav1-b1-v72-1ep-21109`, fp32, no EMA, unanchored tactical target), step 1,000 of 21,109 (4.7 % of its epoch), md5 `45b9f4d82a3a7f15bda6eecf11e4fb71`; config from `ckpt['cfg']` (no `config.json` existed).
**Adapter:** `taniteval/tools/refav1_arm.py` at HEAD blob `de4660cb` (lead block ON, goal provenance, `--with-oracle-goal-arm`), plan budget samples 300 / iters 30 / elites 30 — the FULL budget; the tool projected 2.51 h against the 2 h rule and the Master Mind kept it deliberately (the first real T1 read deserves full search quality; the run was detached). Actual: 2 h 32 min.

⚠️ **NOT decision-grade.** A 20-clip SUBSET of the 141-clip eval split (non-parity local slice), a step-1,000 checkpoint of a run that has since been replaced, and the instrument's `_unverified` stamp ("validated on a random-init RefAV1; this box is forbidden from contacting Thor"). This is an instrument check and the first real number; the pre-registered read is the end-of-epoch checkpoint on the full split on Thor.

## 1. Per-arm ADE (m) with 95 % episode-cluster CIs

| arm | tier | ADE | distance-keeping (43 lead windows) |
|---|---|---|---|
| cl (plan(), true nav, own actions) | T1 | **0.547** [0.405, 0.714] | headway 33.40 m · time gap 3.55 s · min TTC 24.46 s |
| ha (hold observed (a, κ)) | T1 | 0.610 [0.410, 0.831] | 33.23 · 3.52 · 24.02 |
| ol (recorded future actions) | T0 | 0.499 [0.326, 0.688] | 33.28 · 3.53 · 23.93 |
| cl_navshuf (nav permuted) | T1 | 0.642 [0.492, 0.815] | 33.47 · 3.55 · 24.50 |
| cl_oraclegoal (true future field as goal) | T0 | 0.718 [0.608, 0.839] | 33.61 · 3.58 · 25.01 |

Strategic family: `families_unavailable=['strategic']` on every arm (the route-accuracy block needs the strategic head's declared route; 40 nav-labelled windows exist in the strategic block, `nav_valid_frac` 1.0).

## 2. Paired deltas, cl minus control (negative = better for errors; positive = better for the two correctness rates)

| metric | cl − ha (T1 − T1) | cl − navshuf (T1 − T1) | cl − oraclegoal (T1 − T0) | cl − ol (T1 − T0) |
|---|---|---|---|---|
| ADE m | −0.064 [−0.165, +0.035] | **−0.095** [−0.158, −0.040] | **−0.172** [−0.249, −0.085] | +0.047 [−0.063, +0.159] |
| FDE m | **−0.283** [−0.571, −0.001] | **−0.254** [−0.423, −0.107] | **−0.464** [−0.670, −0.229] | +0.079 [−0.221, +0.372] |
| LON speed MAE m/s | **+0.237** [+0.117, +0.375] | **−0.141** [−0.234, −0.058] | **−0.262** [−0.367, −0.143] | **+0.376** [+0.232, +0.546] |
| LON along MAE m | +0.059 [−0.027, +0.144] | **−0.107** [−0.177, −0.045] | **−0.192** [−0.271, −0.102] | **+0.137** [+0.044, +0.229] |
| LON accel MAE m/s² | **+0.189** [+0.086, +0.313] | **−0.143** [−0.238, −0.059] | **−0.285** [−0.382, −0.177] | **+0.405** [+0.263, +0.573] |
| LAT cross-track MAE m | **−0.179** [−0.281, −0.094] | 0.000 exactly | 0.000 exactly | **−0.129** [−0.209, −0.064] |
| LAT heading MAE ° | **−1.99** [−3.03, −1.13] | 0.000 exactly | +0.0008 [0, +0.0021] | **−1.59** [−2.36, −0.91] |
| LAT yaw-rate MAE rad/s | **−0.048** [−0.073, −0.027] | 0.000 exactly | 0.000 exactly | **−0.036** [−0.054, −0.021] |
| TAC lon manoeuvre correct | **−0.136** [−0.257, −0.036] | **+0.093** [+0.036, +0.164] | **+0.179** [+0.079, +0.264] | **−0.257** [−0.371, −0.150] |
| TAC lat manoeuvre correct | +0.064 [−0.021, +0.164] | 0.000 exactly | 0.000 exactly | +0.021 [−0.057, +0.107] |
| distance-keeping headway_min m | — | — | — | +0.127 [+0.022, +0.292] (7 episodes) |
| distance-keeping time_gap_min s | — | — | — | +0.024 [+0.005, +0.054] |
| distance-keeping min_ttc s | — | — | — | +0.528 [+0.058, +1.054] |

Bold = interval excludes 0 ("separated").

## 3. cl point values and provenance

- LON: speed MAE 0.488 m/s (bias −0.155), RMSE 0.831; within 0.5 / 1.0 / 2.0 m/s on 70.9 / 85.0 / 95.2 % of the 1,400 horizon steps; along MAE 0.377 m (final bias −0.236 m); accel MAE 0.483 m/s². Ego progress ratio mean 1.007 (median 0.993; under-progress on 65.7 % of 137 windows; 3 excluded for < 0.5 m GT progress).
- Distance-keeping (lead block `b1_eval_lead_block.npz`, gap = rig origin to lead rear face): 43/140 lead windows; 25 never close on the lead and are censored at TTC 30 s; every speed band UNPOWERED (3–11 lead windows < the 30-window floor) — PRESENT, not quotable per stratum.
- Planner provenance (`planner[cl]`): baseline won on **75.7 %** of windows (`baseline:hold_v0` 75.7 %, `cem` 24.3 %); goal source `tactical_imagined` on 100 %; goal action (lon) CRUISE 70 % / ADAPT_SPEED_FOR_CURVE 30 %. cl_navshuf: baseline 72.9 % (hold_v0 64.3 %, decel_1.5 8.6 %, cem 27.1 %). cl_oraclegoal: baseline 57.9 % (cem 42.1 %).
- Anti-echo block: `holdv0=NOT_SEPARATED` (ADE), `copy_detector=ECHO` (echo index 0.9929 vs the human's 0.1286), `shuffle=UNAVAILABLE` for that block.

## 4. Reading (T-tier stamped)

1. **The echo test is NOT passed on ADE at step 1,000** (T1): cl vs hold-action −6 cm, interval crossing 0; the copy detector reads ECHO because the deployed plan IS hold-v0 on 76 % of windows. This is the init-floor property of a 4.7 %-epoch checkpoint, not a verdict on the architecture.
2. **Lateral planning already beats holding** (T1, separated): heading −2.0°, cross-track −18 cm, yaw-rate −0.048 rad/s. **Longitudinal planning loses to holding** (T1, separated): speed +0.24 m/s, accel +0.19 m/s², longitudinal manoeuvre correctness −14 points. The planner's weak side is the speed channel — the same side the speed-channel repair targeted (v0 as an input); the search chooses worse speed profiles than "keep v0" where it overrides the baseline.
3. **Nav information reaches the plan, and only its longitudinal channel** (T1): the shuffle costs 9.5 cm ADE and every LON metric (separated), while every LAT delta is 0.000 on all 140 windows. The lateral control that reaches the road is nav-independent. `canonical_controls` (refa_v1.py:1677) builds proposals from lateral × longitudinal tactical tokens, so curvature IS in the proposal set; the chosen curvature never depended on nav or goal. Open code question: does `icem_plan`'s search move κ, or does the lateral profile always resolve to the baseline's held κ? (0-GPU read.)
4. **The oracle goal HURTS** (T0 vs T1): the true future field as the goal makes ADE 17 cm and speed 0.26 m/s worse than the true-nav arm. Attribution: "cannot search toward a goal field" (or the goal cost is mis-scaled), not "no goal information" — the imagined tactical goal (100 % of cl windows) does better than the truth.
5. **Closed vs open (T1 − T0):** LON worse (speed +0.38), LAT better (heading −1.6°), ADE not separated; distance-keeping deltas tiny and positive (the planner keeps 13 cm more minimum headway than the recorded human on 43 windows over 7 episodes).

## 5. What it changes

- The register carries refav1's first T1 number with its controls (D-REFAV1-STEP1000-READ); MODEL_REGISTRY gets NO row (a 20-clip slice of a retired run's step-1,000 checkpoint is not admissible for the registry's per-version record).
- The decisive comparison is the PAIRED read of the clean epoch's own step-1,000 checkpoint (EMA + bf16 + TF32; same 140 windows, arms, budget), rolling on the same box (ends ≈ 05:30 Berlin; one-shot 05:47 banks it): every delta above gets a paired counterpart.
- Two instruments to add before the end-of-epoch read: (i) the lateral-search question (κ moved by the search or not) as a 0-GPU probe on the dump; (ii) the strategic family's route block for refav1 (currently `families_unavailable`).
- The longitudinal weakness is the item to watch on the clean epoch's reads at 1,000 / 5,000 / 10,000 steps: if speed MAE vs hold-action does not cross 0 by mid-epoch, the plan's speed channel (proposal set, cost) is the lever, not the world model.

## 6. Deviations and caveats

- Budget: 2.51 h projected > the 2 h rule; kept deliberately (detached run, full search quality on the first real read). Recorded here and in the register row.
- 20-clip local slice (episodes chosen by the slice builder, non-parity): exploratory power only; n_episodes 7 for the distance-keeping pairs.
- `_unverified` stamp of the adapter on a real checkpoint: this read IS that verification for the plumbing (all five arms rolled, analyze() completed, all families PRESENT or REFUSED with reasons); the numbers are the first on a trained checkpoint.

## 7. AMENDMENT (2026-09-03 03:55 Berlin) — the closed-loop plan is the constant-velocity straight line on 140/140 windows

Re-read of the banked dumps (`raw/t1_dump_manifest.json`; per-episode `ep*.npz` trajectories, `tools/straight_line_probe.py`):

| arm | straight-line plans (y ≡ 0) | constant-speed plans | mean lateral error, human-straight windows (n 68) | human-curved windows (n 72) |
|---|---|---|---|---|
| cl | **140/140** | **140/140** | 0.035 m | 0.496 m |
| ha (hold observed a, κ) | 3/140 | 1/140 | 0.120 m | 0.763 m |
| ol (recorded a, κ replayed) | 1/140 | 1/140 | 0.069 m | **0.716 m** |
| cl_navshuf | 140/140 | 122/140 | 0.035 m | 0.496 m |
| cl_oraclegoal | 140/140 | 96/140 | 0.035 m | 0.496 m |

cl is identical to cl_navshuf on 122/140 windows and to cl_oraclegoal on 96/140; the differences are speed profiles only.

**What this corrects in §4:**
1. Reading 2 ("lateral planning already beats holding") is WRONG as an interpretation. Every cl plan has κ = 0: the "gain" is a straight line beating the hold-action control's held, noisy κ (which drifts 0.12 m even where the human drives straight). The lateral rows of the cl − ha comparison are VOID as evidence of planning. The echo test must be run against the STRONGEST trivial baseline — a constant-velocity straight line (`ha0`) — not only against hold-(a, κ).
2. Reading 3 is sharpened: curvature never moves under any nav or goal (0 of 140 windows), so the search's cost is flat in κ — the deployed policy at step 1,000 IS the CV baseline; the "cem" label on 24 % of windows names a CV profile chosen from the proposal set. Hypothesis H-REFAV1-LAT-INSENSITIVE: the imagination is insensitive to the lateral action (the P2 family), testable with the anchored displacement diagnostic on refav1's predictor (κ = ±0.05 vs 0).
3. NEW, instrument-level: the open-loop replay of the recorded (a, κ) misses the human's lateral position by 0.72 m on curved windows — MORE than the human's own excursion (0.50 m, the straight line's error). The kinematic contract ("recorded actions must reproduce GT") is not met laterally on this slice: a κ sign / timing / unit convention defect in the loader-to-unicycle path, or a slip component κ cannot carry. Until it is resolved, no LATERAL row of any refav1 T1 read is quotable, including the paired clean-epoch read.
