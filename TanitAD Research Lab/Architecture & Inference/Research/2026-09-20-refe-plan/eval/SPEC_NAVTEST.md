# SPEC — REFe on NAVSIM v1.1 `navtest` (PDMS + the four families) — PRE-REGISTERED

**Written 2026-09-24 ~02:10 Berlin, BEFORE any REFe navtest score exists.** The live run is
`vitl16_navtrain10_grow_tau0.3` (MODEL_REGISTRY §14.1), at step ~35 of 10,075 when this was written.
No REFe checkpoint has been evaluated on anything. Purpose: prove the evaluation path on an EARLY
checkpoint so the final number is not the first time the path runs, and read a learning curve.

## 0. What this SPEC stands on (MEASURED unless marked)

| fact | evidence |
|---|---|
| W3's harness reproduces the NAVSIM v1 reference on navtest: CV **20.6517** (leaderboard, 4 dp), HUMAN **94.55** (paper 94.8), **STOP 61.82** -- a do-nothing floor far above CV, because EP == 1 by rule when compliant progress <= 5 m | `…/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/RESULT.md` §1, §3.3 |
| The scorer reads trajectories from a **seam** npz (`token`, `fingerprint`, `poses [N,8,3]` at 0.5 s, `sampling`), fingerprint = sha1 over each ego status's pose/velocity/acceleration/driving_command -- `w3_agents_v1.SeamAgentV1` | `…/2026-09-19-navsim-v1-navtest/code/w3_agents_v1.py:35-43, :69-` |
| The exact per-token `AgentInput` (incl. fingerprints) is exported: `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz` | `…/raw/export_navtest.manifest.json` |
| navtest = **136 logs / 12,146 tokens**; **all 136 logs' nuPlan DBs are on this box** (`…/nuplan-v1.1/splits/test`) -- REFe's inputs can be built with the TRAINING code path | `navtrain_scenarios.load_split(navtest.yaml)` x `index_dbs()`, 2026-09-24 |

## 1. The REFe inputs at eval -- and the two declared departures from the NAVSIM agent contract

Built per token with the SAME functions that built the training rows (`navtrain_scenarios.build_scenarios_for_log`,
`build_teacher_rollouts` geometry, the teacher's own goal logic, `calib_table`), at the rate the run trained on
(`REFE_SIM_HZ=10`), from the nuPlan test DB + OpenScene `sensor_blobs/test` (4 cameras, the trainer's own decode:
`cv2.imdecode` -> resize 960x512 -> RGB -> /255):

| input | source at eval | vs the NAVSIM v1 agent contract |
|---|---|---|
| 4 cameras F0/L0/R0/B0 | OpenScene test blobs at the token's frame | same data a NAVSIM agent gets |
| per-sample camera rig | the log's DB camera table (`calib_table.read_db`) | same as training |
| ego (7-D: vx, vy, ax, ay, yaw rate, steering angle, speed) | the nuPlan DB state at the token | ⚠️ **DEPARTURE 1**: NAVSIM's `ego_status` carries velocity + acceleration only; yaw rate and tire steering angle are measured vehicle state at t0 (admissible under the 2026-09-02 v0 ruling's logic) but are NOT in the NAVSIM contract |
| goal (2 points, ego frame) | the teacher's route goal at the token -- the route is derived from the LOGGED drive | ⛔ **DEPARTURE 2**: NAVSIM gives a discrete `driving_command`; REFe consumes route goal points (the paper derives a navigation COMMAND from the goal point -- `PAPER_CONFORMANCE_REVIEW.md` D8). A route taken from the log is **optimistic by construction** (CLAUDE.md goal ruling). Every REFe navtest number carries this stamp until a command-conditioned arm exists |

⇒ **A REFe navtest PDMS is quotable as "REFe as trained, route-goal protocol" -- NOT as a leaderboard-comparable
number** against DriveZero's Table rows, until both departures are closed or shown not to matter.

## 2. Output and selection

The model returns 64 proposals x 20 poses (5 Hz, 4 s, ego frame) + a 6-component score per proposal. Selection:
the proposal with the highest aggregated scorer output (the rule `planner.py` uses -- re-read before running; if the
two differ, `planner.py` wins and this line is corrected before any number is quoted). Conversion to NAVSIM's
8 poses at 0.5 s: linear interpolation of x, y and wrapped heading on the 0.2 s grid (0.5 s -> between samples 2
and 3, …, 4.0 s == sample 20 exactly). ⛔ The converter is checked on an ANALYTIC target before use: a straight
line at constant speed must convert with max error < 1e-6 m, and a constant-curvature arc with max error <= its
chord SAGITTA R(1 - cos(w dt / 2)) + 1e-9 m (dt = 0.2 s).

> **AMENDMENT 1 -- 2026-09-24 ~02:40 Berlin, BEFORE any REFe navtest score existed.** The first version demanded
> < 1e-6 m on the arc too. Linear interpolation between 0.2 s samples cannot meet that on an arc: it lands on the
> chord, whose worst deviation from the arc is the sagitta (~2.5 cm at R = 20 m, 10 m/s). That bar would have
> failed a CORRECT converter, so it was a wrong expectation, not a strict one. The bar now carries the analytic
> bound. (Same interpolation the nuPlan simulator applies to a planner's trajectory.)

## 3. Bars -- written before the result, both outcomes committed

| id | bar | if it fails |
|---|---|---|
| E-0 | **harness live**: the HUMAN seam re-scored through THIS package's driver reproduces W3's HUMAN cells (max abs diff 0.0 on the subset) | stop -- the driver is wrong, no REFe number |
| E-1 | **converter exact** on the analytic targets (§2) | stop |
| E-2 | **pipeline live on an early checkpoint** (snap_epoch001, ~30 steps): a finite seam for every token of the subset, the scorer's C1-C3 guards pass | stop -- a path defect found on day 1 is the point of this SPEC |
| E-3 | a checkpoint's PDMS is reported WITH its STOP floor (61.82 full / the subset's own STOP) and the HUMAN ceiling, paired episode-cluster bootstrap vs STOP | a REFe PDMS below STOP is reported as below the do-nothing floor, never rounded into a "partial" result |
| E-4 | the four families (longitudinal / lateral / tactical / strategic) on the same tokens, reusing `…/2026-09-23-refcv6-standard-tests/navsim/code/families6.py` where it applies; any family that cannot be computed says so with its n | the eval is incomplete, not "ADE-only" |
| E-5 | **proposal diversity** (AMENDMENT 2): every learning-curve point reports, on the SAME tokens, the mean distance of the M proposal endpoints from their centroid, open-loop ADE (0.5 s grid, vs W3's human future) of the SELECTED / a RANDOM (mean over M) / the ORACLE-best proposal, and how many of the M are ever best | see the decision rule below -- a watch on the recipe, never a model pass/fail |
| E-6 | **selection diagnosis** (AMENDMENT 3): on the SAME tokens, EVERY one of the M proposals is scored by W3's unchanged harness as its own single-proposal seam, giving the PDMS of the planner's pick / a RANDOM pick (mean over M) / the ORACLE pick (best of M), the scorer's within-token ranking skill, and each scorer component's discrimination | see the decision rule below -- a diagnosis of the recipe, never a model pass/fail |

> **AMENDMENT 2 -- 2026-09-24 ~04:20 Berlin, BEFORE any 200-token learning-curve point existed.** The only proposal
> readout seen so far: the E-2 log (25 tokens, 1 log, `snap_epoch002`, ~90 optimiser steps) -- endpoint spread
> **0.04 m**, ADE oracle 3.115 / random 3.160 / selected 3.161 m, 7 of 64 proposals ever best. Near-identical proposals
> are EXPECTED this early in winner-takes-all training, and are also what a collapsed head looks like, so the reading is
> committed now. MEASURED on the same snapshots: the 64 queries are distinct (pairwise cosine 0.003) but small
> (|q| 0.33 -> 0.35 from epoch 1 to 2; `trunc_normal_(std=0.02)`) beside the ego token ADDED to every one of them
> (ego MLP weight norms ~9.3), and the init scale is OUR choice -- the paper does not state it (2 probes: DriveZero's
> release carries no student code; `../2026-09-20-drivezero-deep-analysis/RESULT.md` names only the mechanism).
> **Decision rule.** At the first learning-curve point taken after the goal-augmented twins have been in the training bank
> for at least one full epoch: if the endpoint spread is still **< 0.5 m** AND the oracle ADE is within **0.2 m** of the
> random ADE, the head is recorded **COLLAPSED** and the recipe question (query scale / init, a departure from our
> current reproduction) goes to the PI with this evidence; otherwise the watch closes as early-training behaviour.
> Either way the numbers are reported per point with the selected / random / oracle triple -- an oracle gap is never
> quoted without its random control.

> **AMENDMENT 3 -- 2026-09-26 ~09:52 Berlin, BEFORE any per-proposal PDMS existed.** Seen so far: the six learning-curve
> points (PDMS 31.33 / 27.51 / 29.60 / 53.48 / 49.28 / 43.83 after epochs 1 / 2 / 3 / 5 / 8 / 11); after epoch 11 the pick's
> open-loop ADE is 2.695 m against 2.442 m for a random proposal and 0.348 m for the oracle, and the pick drives fast (speed
> bias +1.15 m/s, progress 1.23x the human's). The planner's rule is PDM-shaped (`refe/planner.py` `aggregate`:
> NC x DAC x DDC x (5 EP + 5 TTC + 4 C)/14 over sigmoids), so a formula mismatch is NOT the hypothesis. The hypothesis is the
> scorer's SUPERVISION: `refe/train.py` attaches each of a frame's 8-9 FIXED banked candidates (teacher, lat +-2/+-4, lon
> x0.5/x1.5, stopped, jerky, reverse, over-curb) to its NEAREST model proposal and trains THAT proposal's six outputs on the
> candidate's scores; the trainer's `assign_d` reads 2.5-5.7 m on the last micro-batches of step 3549 while the 64 endpoints
> spread ~4 m, so most proposals are never supervised and the supervised ones inherit scores of paths metres away.
> **Measurement.** Each of the M = 64 proposals of every token is written as its own seam and scored by
> `eval/score_navtest_refe.py` UNCHANGED -- one run per proposal index, so EP is normalised against the PDM-Closed reference
> exactly as for a real submission and guards C1-C3 run on every table column. Table: PDMS and the six sub-scores per
> (token, proposal); the planner's raw six logits per (token, proposal) from the same forward pass.
> **Gates (any failure voids the readout):** G1 the dump re-run reproduces the landed seam's selected poses (per-token max
> abs difference reported; a token whose pick differs is counted and the readout uses the re-run's own pick); G2 the table
> at the pick reproduces the landed per-token score exactly wherever the poses are identical; G3 all 64 runs PASS with 200
> valid rows.
> **Readouts,** paired log-cluster bootstrap (93 logs, 10,000 resamples, 95 %): (a) PDMS actual / random / oracle;
> (b) within-token Spearman between the planner's aggregate and the true PDMS over tokens whose true PDMS is not constant,
> and the top-1 hit rate beside a random selector's expected hit rate; (c) per component, pooled AUC of the predicted
> probability against the true sub-score == 1 (NC, DAC, DDC, TTC, C) and Spearman for EP, plus the share of tokens on
> which that sub-score varies across the 64; (d) the pick's ego progress relative to the mean over proposals.
> **Decision rule.** SELECTION-BOUND iff actual - random has a 95 % upper bound below +2.0 PDMS AND oracle - actual exceeds
> 10 PDMS. A component is named a FAILING SCORER OUTPUT iff its pooled AUC is below 0.60 while its true value varies within
> token on at least 20 % of tokens. If SELECTION-BOUND, the recipe question (how the scorer is supervised) goes to the PI
> with this evidence and nothing in the recipe changes without the PI; if not, the decline is in the proposals themselves
> and the next probe is the oracle PDMS across snapshots. ONE alternative rule is confirmatory: **medoid** (the proposal with
> the smallest mean xy distance, over the 8 poses, to the other 63; no scorer) against the incumbent pick, paired, same
> tokens. Every other rule (logitsum, component-subset aggregates) is EXPLORATORY, labelled so, and is never claimed without
> a disjoint-token confirmation. The same table is re-taken at every later snapshot on the same tokens: from epoch 12 the
> trainer reads the full 189,858-frame scorer bank, and whether the scorer's skill moves with it is the question.

> **AMENDMENT 3a -- 2026-09-26 ~10:10 Berlin, BEFORE the per-proposal table was read** (25 of the 64 per-proposal
> score files existed on disk; none had been opened or aggregated). Amendment 3's first clause ("actual - random has a
> 95 % upper bound below +2.0 PDMS") CANNOT FIRE at n = 200: the readout's mutation self-test
> (`eval/selftest_selection_readout.py`, synthetic tables on these 200 tokens / 93 logs) gives a truly RANDOM selector a
> 95 % interval for actual - random of about [-4.1, +5.5] PDMS on three seeds, so the bar would call a coin flip "not
> selection-bound". Replaced by the SHARE OF THE AVAILABLE GAIN the pick realises: **skill = (actual - random) /
> (oracle - random)**, both sums over the same resampled logs (paired, 10,000). **SELECTION-BOUND** iff the skill's 95 %
> upper bound is below **0.25** AND oracle - actual exceeds 10 PDMS; **NOT SELECTION-BOUND** iff its lower bound is above
> 0.25 OR oracle - actual is at most 10 PDMS; otherwise **UNDETERMINED**, and the table is extended to more tokens before
> any claim. Self-test after the change: oracle-ranked scorer -> NOT (skill 1.0); random scorer -> SELECTION-BOUND (skill
> 0.010 [-0.059, 0.079]) while the old clause stays silent; noise on DAC alone -> DAC is the only failing output. The
> component rule, the gates, the medoid comparison and everything else in Amendment 3 are unchanged; the old clause's
> result is still reported, labelled unused.

**Subset first:** W3's `A1_sub200_tokens.json` (200 tokens) for E-0..E-2, then the full 12,146.
**Tier stamp:** NAVSIM v1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents, as W3 stamps it.
