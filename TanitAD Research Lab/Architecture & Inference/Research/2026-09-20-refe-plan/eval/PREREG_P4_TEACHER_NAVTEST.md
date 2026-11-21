<title>PREREG P4 — our DriveRL teacher protocol on NAVSIM navtest</title>

# PREREG P4: does our teacher protocol cap the student? (written 2026-10-05, BEFORE any navtest teacher number exists)

**Status at writing:** no teacher trajectory has been rolled on any navtest token and no teacher seam has been scored.
The only navtest numbers that exist are the student's and the human's (REVIEW_7 §0). This file is written first and
is not edited after the first teacher score exists, except in the clearly dated **§9 decisions log** (token-set
decision from the smoke, made before scoring; any declared departure).

**Origin:** REVIEW_7_GAP_TO_PAPER.md §3.4 and §4 P4, PI-approved 2026-10-05.

## 1 · Question

The dirty fans (< 50 % of the 64 proposals NAVSIM-safe) cause 72 % of REFe's zero picks, and on held-out navtrain our
teacher's own targets fail NAVSIM's drivable-area rule on 4.97 % [1.30, 10.31] of samples (MEASURED,
`raw/2026-10-04-lane-discipline/teacher_check_heldout_summary.json`). The paper's DriveRL teacher row on navtest is
PDMS 95.8, NC 99.8, DAC 99.9, EP 91.5, TTC 99.1, C 99.0 (PUBLISHED, DriveZero arXiv 2609.06055, Table 4, banked PDF).
**We have never measured our teacher protocol on navtest.** This experiment measures it.

## 2 · The arm: "our teacher protocol" (exactly what built REFe's rank-0 targets)

* **Network:** the released DriveRL teacher `release/checkpoints/checkpoint_2400.pt` under `DZ_ROOT`
  (`C:/Users/Admin/dz/DriveZero/DriveRL`), local md5 `0b4236016856704b564bdbe33043185a` (MEASURED 2026-10-05).
  Control C3 (§4) is what establishes that this file reproduces the banked targets; no pod file is needed if C3 passes.
* **Planner / controller:** `refe/build_teacher_rollouts.py::build_planner` and `::build_controller`, imported
  unchanged — hydra instantiation of their shipped yaml with the runner's overrides (route goal horizon 12.0 s, min
  speed 5.0 m/s, pair mode `legacy`, TTS **off**; bicycle max accel 4.0, max steering rate 0.8); the exact
  map-feature / query / map-step caches ON (as in the bank build). 5 Hz policy inside the simulation loop.
* **Route:** rank 0 = THEIR unpatched route code (`route_rank.apply(0)`). The planner is initialised with the
  scenario's raw `get_route_roadblock_ids()` and DriveRL's feature builder applies `correct_route_roadblock_ids` itself
  on the first step (`driverl/nuplan/feature_builder.py:407-413`, reset per `initialize`, `planner.py:127`) — i.e.
  the corrected route, as in training.
* **Rate:** the simulation rate that reproduces the banked targets in C3. Expected 10 Hz (`REFE_SIM_HZ=10`,
  READINESS D9/C17: the rank-0 bank was rebuilt at 10 Hz by CPU-teacher shards). If C3 passes only at 20 Hz, 20 Hz is
  used and that is recorded. If it passes at neither, no teacher number is admissible.
* **Device:** CPU (the bank's teacher shards were CPU; no GPU is taken).
* **Scenario per token:** `navtrain_scenarios.build_scenarios_for_log` on the nuPlan **test** DBs, one scenario per
  navtest token (`initial_lidar_token = token`), exactly the seam's call (`history_rows=1, future_rows=80`).
* **Rollout:** `build_teacher_rollouts.rollout_from_frame(sc, planner, controller, 0, span)` unchanged: ego seeded at
  the logged state, 2.0 s history buffer read backward from the DB, background agents replaying the log
  (non-reactive), 4.0 s. ⚠️ Where a token sits less than 2.0 s after its log's first DB row, the buffer is shortened to
  the longest history the DB holds (nuPlan's devkit cannot read past row 0) and the token is **flagged**
  `history_clamped`; the count is reported.
* **Plan:** the target geometry of `make_row` — the rollout's rear-axle poses at 0.2 s (`poses[stride-1::stride][:20]`)
  in the logged ego frame — converted to NAVSIM's 8 poses at 0.5 s by `refe_navtest_seam.to_navsim` (the converter
  every REFe seam uses). Report-only: the rollout's own states at exactly 0.5 s multiples, and their max deviation from
  the converted plan.
* **Seam:** npz with `token`, `fingerprint` (W3's export), `poses` [N, 8, 3], `sampling` [8, 0.5], `arm`
  — the format `refe_navtest_seam.py` writes.

**Privileged inputs** (map, agents, corrected route) are allowed: this measures the teacher, not a student.

## 3 · Scoring

* Harness: the drive-repointed wrapper `score_e.py` around `eval/score_navtest_refe.py` (W3's `run_v1.py`
  `cmd_score`, unchanged, NAVSIM v1.1, navtest metric cache, guards C1–C3). Label starts with `refe`; `--tokens` with
  the exact token set; **the harness's own status line must read PASS** or nothing is quoted.
* **Paired human:** `score_navtest_refe.py --human-seam` on the same token set, scored the same way.
* **Tier:** NAVSIM v1.1 PDMS = a non-reactive pseudo-simulation of a 4 s open-loop plan against logged agents.
* **Estimator:** paired log-cluster bootstrap over the logs (`eval/proxy_eval.py::boot`, 10,000 resamples, seed
  20260927). It answers *another draw of LOGS* only. The teacher planner is deterministic given the device, so there is
  no inference-seed variance on this arm (C3 measures the device/rate sensitivity instead).

## 4 · Mandatory controls (each must read its known value, or no teacher number is admissible)

* **C1 frame/time:** per token, the logged human future from the SAME scenario on NAVSIM's 0.5 s grid vs W3's exported
  `human_future_poses`, bar **1 mm**. The 36 tokens in 4 logs already known to fail this bar on the student's full
  seam (`raw/2026-10-04-navtest-final/frame_control_navtest_full.json`, a nuPlan-DB vs OpenScene pose disagreement) are
  expected to fail again; **any OTHER token over 1 mm fails C1.**
* **C2 human:** the human seam on the token set must score within the harness's known value — on the full navtest
  94.55 [93.80, 95.24] (MEASURED, REVIEW_7 §0); on a subset, the human's PDMS must be ≥ 93.0 (a broken human seam reads
  far lower). Teacher vs human is read PAIRED.
* **C3 teacher pipeline:** on the 47 held-out navtrain samples of `raw/2026-10-04-lane-discipline/teacher_check_heldout.jsonl`
  whose banked teacher target is present in the on-policy queue (`E:/Projects/TanitAD/data/refe_m5r/queue`, field
  `teacher` [20, 3]), my rollout code must reproduce the banked target: **median max-|Δxy| ≤ 0.05 m AND ≥ 90 % of
  samples ≤ 0.25 m** (0.25 m = the fidelity tolerance of `build_teacher_rollouts.verify_fidelity`). Every sample over
  0.25 m is listed and explained, or C3 fails. Run BEFORE the navtest run.
* **C4 coverage:** misses (scenario not built, rollout exception, non-finite pose) ≤ 1 % of the token set. Above that
  the verdict is **INCOMPLETE (coverage)**.

## 5 · Token set (decided from the 24-token smoke, BEFORE scoring)

* Smoke: the existing `E:/Projects/TanitAD/data/refe_navtest/fullsmoke24_tokens.json` (24 tokens, 3 logs). Rollouts,
  C1 and timing only — **the smoke is not scored with the harness before the decision is written in §9.**
* Rule: **FULL navtest (12,146 tokens, 136 logs)** if the estimated wall clock of the rollouts at the worker count the
  free RAM allows (≥ 4 GB kept free) is **≤ ~8 h**; otherwise **Amendment 5's 923 confirmation tokens**
  (`a5_confirm_tokens.json`, 43 logs).
* **Primary set** = the chosen set minus the C1-known-failing tokens it contains (the student's final read did the same:
  12,110 of 12,146) minus misses. The all-tokens read is reported beside it.

## 6 · Decision rule (committed now, both outcomes; point estimates on the primary set, ×100)

* **"TEACHER CAPS"** iff teacher **PDMS < 93.0 OR DAC < 98.0** ⇒ the student inherits teacher failures; **P5 (clean or
  re-query failing teacher targets) is the next lever**, and the teacher protocol itself is reviewed.
* **"TEACHER DOES NOT CAP"** iff teacher **PDMS ≥ 94.0 AND DAC ≥ 99.0** ⇒ the dirty fans are a student learning
  problem; P5 is deprioritised; P1/P2 (scorer supervision) stay first.
* **"PARTIAL"** otherwise, with the sub-scores that miss named.
* The bootstrap interval is reported for every number; if a threshold lies inside the interval, the verdict carries
  the flag *"sensitive to log sampling"* (reported; it does not change the verdict as written).

## 7 · Reported (not gating)

1. Teacher sub-scores NC, DAC, EP, TTC, C, DDC (DDC weight 0 in v1.1 PDMS, reported separately), PDMS, zero-score rate.
2. **Paired teacher − human** per sub-score and PDMS, with intervals.
3. Against the paper: our teacher vs Table 4's DriveRL row (95.8 / 99.8 / 99.9 / 91.5 / 99.1 / 99.0) and human row
   (94.8 / 100 / 100 / 87.5 / 100 / 99.9) — PUBLISHED, no interval available.
4. Four families on the same tokens (families6, as `eval/stage2_sft.py` calls it): longitudinal, lateral, tactical;
   **strategic = UNAVAILABLE in NAVSIM** (no route/goal-setting label), stated per family.
5. Split by the review's dirty-fan scene types, using the review's own per-token inputs
   (`raw/2026-10-01-goal-trigger/goal_census_navtest_full.json`, as `raw/2026-10-04-gap-review/gap_dirty_fans.py`
   computes them): far-goal bearing (< 10°, 10–30°, 30–60°, ≥ 60°) and off-route (> 20 m) vs on-route; plus the
   teacher's own goal bearing.
6. **Teacher on the student's dirty-fan tokens vs clean-fan tokens** (fan safe share from
   `raw/2026-10-04-lane-discipline/lane_census.jsonl`, the review's definition) — the direct read of "the fan inherits
   teacher failures".
7. `history_clamped` count; the converted-plan vs exact-0.5 s deviation.

## 8 · What this does NOT license

* navtest has been read many times; every navtest number here is **exploratory** for the student programme. The teacher
  number is a **measurement of the target generator**, not a model selection.
* A teacher PDMS above the student's does not by itself show the student CAN reach it; a teacher below 94.55 does not by
  itself show the student cannot exceed it (the paper's student 94.55 sits below its teacher's 95.8).

## 9 · Decisions log (dated; appended only)

**2026-10-05, after C3 and the smoke, BEFORE any harness scoring of a teacher seam:**

* **C3 PASSED at 10 Hz** (MEASURED, `raw/2026-10-05-teacher-navtest/c3_heldout_10hz.json`): 47/47 held-out samples,
  median max-|Δxy| **8.1e-7 m**, p90 7.5e-6 m, max 0.026 m, 100 % ≤ 0.25 m. The local `checkpoint_2400.pt` + this code
  reproduce the banked rank-0 targets; no pod file was needed.
  **Mutation arm, same samples, `REFE_SIM_HZ=20`: FAILS** (median 0.63 m, max 2.99 m, 27.7 % ≤ 0.25 m;
  `c3_heldout_20hz_mutation.json`) ⇒ the control can discriminate, and the rate is **10 Hz**.
* **Smoke** (`fullsmoke24_tokens.json`, 24 tokens / 3 logs; `smoke24.log`, `smoke24/rows_s0of1.jsonl`): 24/24 rolled,
  0 errors, C1 max **7.1e-15 m** (24/24 under 1 mm), 0 `history_clamped`, plan-vs-exact-0.5 s max 0.016 m,
  **1.73 s/token** single process (2 torch threads), peak RSS ≈ 1.0 GB per process. Not scored.
* **Runtime estimate:** 12,146 × 1.73 s ≈ 5.8 h single-process; free RAM 9.0 GB at decision time allows 4 shard
  processes with ≥ 4 GB kept free ⇒ ≈ 1.5–2 h (ESTIMATED). **≤ 8 h ⇒ token set = FULL navtest (12,146), per §5.**
* Launch note: the 4 shards run `eval/teacher_navtest_seam.py` blob `a9532c19` (byte copy:
  `raw/2026-10-05-teacher-navtest/teacher_navtest_seam.launched_a9532c19.py`). A default-off `--tts N` flag was added to
  the working file AFTER launch for F1 below; the registered arm is `--tts 0`, i.e. the launched code path.

**2026-10-05, still BEFORE any teacher score exists — follow-on F1 (REPORT-ONLY, does not change §6):**
RULE ZERO: if §6 reads CAPS or PARTIAL, the next lever is P5 (clean / re-query the failing teacher targets), and the
cheapest test of P5's re-query arm is whether DriveRL's own value-guided test-time search (released `tts.py`, the
paper's nuPlan TTS lever) repairs the teacher's failures. Registered now:
* **Run iff** the registered teacher has **≥ 30** primary-set tokens with NC·DAC = 0 ("teacher-fail set").
* **Arm:** identical protocol with `--tts 16` (N = 16 candidates, their default seed 42, their conservative margin),
  on (a) the whole teacher-fail set and (b) a **harm control**: 200 teacher-pass primary tokens drawn by
  `random.Random(20261005).sample` over the sorted pass set. Scored with the same harness (status PASS required).
* **Read (paired with the registered teacher on the same tokens, log-cluster bootstrap):** repair rate = share of
  the fail set whose NC·DAC becomes 1; PDMS delta on (a); harm rate = share of (b) whose NC·DAC becomes 0.
* **Committed outcomes for P5's design:** repair ≥ 50 % AND harm ≤ 2 % ⇒ **re-query is a viable P5 arm**;
  repair < 20 % ⇒ **re-query is not the lever; P5 uses the paper-neutral DROP arm** (or a different repair);
  otherwise **MIXED**, both arms carried into P5. navtest is exploratory: this decides the P5 design, not a claim.
* **Sizing amendment (same day, still before any teacher score):** the TTS path check
  (`raw/2026-10-05-teacher-navtest/tts_check3.log`, 3 smoke tokens) MEASURED **≈ 120 s/token** on CPU at N = 16
  (70× the plain teacher), and it does change the rollout (max 0.12 m on 1 of 3 tokens, 0.0 on 2). So (a) is
  **capped at 120 tokens** — if the fail set is larger, `random.Random(20261005).sample` of 120 over the sorted fail
  set — and (b) is **80** pass tokens (same generator, drawn after (a)). Repair-rate precision at n = 120 is ≈ ±9 pp,
  enough to separate the 20 % / 50 % thresholds. CPU only (the GPU would change the device under the comparison).

**2026-10-05 ~09:15, AFTER the registered verdict (TEACHER CAPS) — diagnostic D1 and follow-on F2, both EXPLORATORY:**
* D1 (`eval/p4_dac_anatomy.py`, `raw/2026-10-05-teacher-navtest/d1_dac_anatomy.json`) found that of the teacher's 578
  NAVSIM DAC failures, **504 (87.2 %) are compliant in the teacher's OWN rollout and in the interpolated 8-pose plan,
  and fail only after NAVSIM's LQR tracker + bicycle re-simulates the plan**; 72 (12.5 %) leave the drivable area in
  the rollout itself; 2 from 0.5 s subsampling. Controls: human 0/778 violations; 200/200 pass-set agreement; 578/578
  fail-set agreement with the harness. ⇒ F1 (re-query) can reach at most ~12.5 % of the DAC loss.
* **F2 (registered now, before any compensated number exists):** tracker-aware plan compensation —
  the 8 reference poses are iteratively corrected (`r ← r + (p − track(r))` at the 8 plan times, 3 iterations, ego
  frame, wrapped heading) so that NAVSIM's OWN vendored tracker (`refe/navsim_pdm`, the simulator the harness runs)
  reproduces the intended plan `p`. Applied to (i) the registered teacher seam and (ii) the deployed student seam
  (`refe_navtest_final_routefix`), all 12,146 tokens, scored with the unchanged harness (status PASS), paired with the
  uncompensated seam on the 12,110 primary tokens.
  * (i) teacher: compensated **PDMS ≥ 94.0 AND DAC ≥ 99.0** ⇒ the teacher's cap is mostly a **plan↔tracker interface**
    effect, not teacher quality ⇒ P5 is reframed as **tracker-aware targets**; otherwise report the residual by sub-score.
  * (ii) student: paired Δ PDMS **lower bound > 0** AND no family separates adversely ⇒ a **zero-training test-time
    lever, escalated to the PI for adoption** (it puts NAVSIM's public LQR/bicycle model inside the agent, as
    PDM-Closed does — a departure from the paper's "no test-time post-processing" student); upper bound < 0 ⇒ refuted;
    else not proven.
  * Instrument-level repair rates (D2) are reported beside the harness read and never substituted for it.
