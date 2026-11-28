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

> **AMENDMENT 4 -- 2026-09-26 ~12:15 Berlin, BEFORE the live switch and before any snapshot trained with it.** The PI chose
> option B, the paper's version (the scorer supervised by the student's OWN proposals, labelled by the teacher's scorer),
> plus a NAVSIM-faithful drivable-area label (`refe/navsim_dac.py`, 100.00 % agreement with NAVSIM on the 25,600 E-6
> proposals). Context already seen: selection skill 0.460 [0.376, 0.537] after epoch 5 (untrained head, an accidental
> preference for short plans) and 0.013 [-0.089, 0.121] after epoch 11 (fixed-candidate supervision).
> **Primary readout:** E-6 (Amendments 3 + 3a, unchanged) at the FIRST snapshot whose whole epoch trained with on-policy
> sets in its bank (the snapshot after the first epoch that STARTS after the switch). **SUCCESS** iff the selection
> skill's 95 % lower bound exceeds **0.25**; **FAILURE** iff its upper bound is below 0.25; otherwise UNDETERMINED and the
> next snapshot decides. **Secondary (reported, not gating):** the within-scene AUC of the drivable-area output (must
> rise above 0.60 to call the NAVSIM label learned), the pick's PDMS against the same tokens' STOP (62.6) and the
> pre-switch snapshots, and the training-side on-policy skill the trainer logs. A FAILURE goes to the PI with the table;
> nothing in the recipe changes again without the PI.

> **AMENDMENT 5 -- 2026-09-26 ~21:45 Berlin, BEFORE any selection on the confirmation tokens was computed.** The PI
> chose (in chat, after the Amendment 4 FAILURE) to match REFe's selection rule to the benchmark. Seen so far, on W3's
> 200 tokens only: `refe/planner.py:aggregate` selects with NAVSIM **v2**'s EPDMS shape (NC x DAC x DDC x (5 EP + 5 TTC
> + 4 C)/14) while this harness scores NAVSIM **v1** PDMS (NC x DAC x (5 EP + 5 TTC + 2 C)/12, driving direction at
> weight 0; `pdm_scorer.py:38-42`; RETRACTION_LOG R25). Re-selecting on the same stored logits with the v1 formula moved
> the pick by +4.79 / +0.04 / +1.54 PDMS after epochs 11 / 12 / 13 (`eval/rule_mismatch_diag.py`, EXPLORATORY).
> **Rule under test (ONE, fixed now):** the v1 formula over the scorer's sigmoids, `NC x DAC x (5 EP + 5 TTC + 2 C)/12`,
> argmax over the 64 proposals; ties broken by the lowest index, as the shipped rule.
> **Confirmation tokens:** navtest tokens from the 43 logs that W3's `A1_sub200_tokens.json` does NOT touch (its 200
> tokens span 93 logs); per log the first 24 tokens in sorted-token order (all of them where a log has fewer). None of
> these tokens has had a v1-rule pick computed. **Snapshot:** after epoch 13 (`snap_epoch013.pt`, md5
> `b59c688a4c3005a55fad0769098abb91`). **Measurement:** the unchanged pipeline (`eval/eval_checkpoint.py`) scores the
> shipped pick and dumps all 64 proposals with their logits; the v1-rule picks are written as their own seam from that
> dump and scored by the unchanged harness. **Statistic:** per-token PDMS(v1 pick) - PDMS(shipped pick), mean over the
> tokens, paired log-cluster bootstrap over the confirmation logs (10,000 resamples, 95 %).
> **Decision, both outcomes committed now:** the v1 formula IS the harness's own rule, so the switch is a correctness fix
> and this test guards against HARM. **REFUTED** iff the upper bound is below 0 -> keep the shipped rule and return to
> the PI. Otherwise the v1 rule becomes REFe's selection rule for every evaluation from this amendment on (a declared
> test-time change in `refe/planner.py`, recorded in MODEL_REGISTRY §14): **CONFIRMED GAIN** iff the lower bound is above
> 0, **NO MEASURABLE DIFFERENCE** iff the interval straddles 0. Earlier learning-curve points keep their shipped-rule
> values, with the v1-rule values beside them wherever an E-6 table allows. **Reported, not gating:** the same Delta
> for the v1 formula WITHOUT comfort (exploratory -- it compensates for the inverted comfort head that SPEC option 2,
> NAVSIM-faithful comfort labels, is meant to fix).

> **AMENDMENT 6 -- 2026-09-27 11:57 Berlin (time from `date`), BEFORE any slowed-copy result exists, exploratory or
> confirmatory.** The PI (in chat, 2026-09-27): implement every proposed measure, but use one only once its effectiveness
> and validity are proven. This amendment is the proof rule for the TEST-TIME measure. Seen so far, EXPLORATORY, on W3's
> 200 tokens only (`eval/raw/e6_sub200_ep015/`): the best of 64 fell 91.3 -> 84.2 from snapshot 012 to 015 because the
> whole proposal fan got faster (every slot ~4.3 m longer over 4 s at the first on-policy epoch); a STOP candidate lifted
> the pick +5.43 [+1.50, +9.44] but not above the STOP floor and at the cost of the longitudinal family (speed MAE
> 1.17 -> 2.52 m/s, progress 1.09 -> 0.76 of the human's), i.e. by stopping. **Measure under test:** time-rescaled copies
> of REFe's OWN proposals (same path, slower speed profile; ONE construction, `refe/slow_copies.py`) added to the candidate
> set and scored by REFe's own scorer, picked by the shipped v1 aggregate.
> **Variant (fixed by a rule stated now, before its inputs exist):** among the variants the exploratory probe on the 200
> tokens reports (scoring route: the 64 keep their shipped scores [masked] or the full set; the factor sets it runs), the
> one with the largest exploratory PDMS gain over the shipped pick AMONG THOSE THAT PASS THE FAMILY GUARD below on those
> 200 tokens. If none passes the guard, no confirmation is run and the measure is NOT adopted.
> **Family guard (fixed now; per-arm blocks from `families6.py` against the logged human future, variant vs shipped on
> the same tokens):** longitudinal -- mean progress ratio >= 0.90 AND speed MAE up by at most 0.30 m/s; lateral -- cross-
> track MAE up by at most 0.10 m AND heading MAE up by at most 0.5 deg; tactical -- goal-point error up by at most 1.0 m;
> strategic -- unavailable in NAVSIM (reported as such).
> **Confirmation tokens:** Amendment 5's 923 tokens (43 logs, none of them in W3's 200; `eval/raw/a5_confirm/
> a5_confirm_tokens.json`, md5 3098d178...). **Snapshot:** after epoch 15 (`snap_epoch015.pt`, md5
> `d7c59f4f2fbcbde3e2dec8f67d63a7e7`), the snapshot the exploration used. **Measurement:** the unchanged pipeline dumps
> all 64 proposals with their logits; the copies are built by `refe/slow_copies.py` and scored by REFe's own scorer on
> the chosen route; the variant's picks are written as their own seam and scored by the unchanged harness, beside the
> shipped seam. **Validity gates, all must pass before the statistic is read:** (a) the shipped pick is reproduced from
> the dump on every token and its seam score equals the pipeline's; (b) a factor-1.0 copy reproduces its original
> proposal bit-exactly and its harness score exactly; (c) tokens whose variant pick equals the shipped pick score
> identically (max |diff| 0.0); (d) every harness run PASSes with every token valid. **Statistic:** per-token PDMS(variant
> pick) - PDMS(shipped pick), mean over the tokens, paired log-cluster bootstrap over the 43 logs (10,000 resamples, 95 %,
> seed 20260927). It answers "another draw of episodes" only: one checkpoint, one deterministic forward.
> **Decision, both outcomes committed now:** **ADOPT (effectiveness and validity proven)** iff the lower bound is above 0
> AND the family guard holds on the confirmation tokens -> the variant becomes REFe's selection procedure for every
> evaluation from then on, the final full navtest included (a declared test-time change behind a flag in
> `refe/planner.py`, recorded in MODEL_REGISTRY §14.1; earlier points keep their values). **REFUTED** iff the upper bound
> is below 0. Otherwise **NOT PROVEN** -> not adopted. **Reported, not gating:** the same comparison on the latest
> snapshot's 200 tokens; how many picks switch to a copy and at which factor; the STOP floor on the confirmation tokens.

> **AMENDMENT 6 READOUT -- 2026-09-27 ~13:50 Berlin (applied as written).** The exploratory probe (`eval/raw/e6_sub200_ep015/slow_copies.json`) reported 16 scorer-selected variants; NONE passes the family guard on the 200 tokens (every 0.75x arm fails the heading bound, +0.52 to +0.66 deg; the 0.5x and mixed arms also fail progress or speed). By the rule: no confirmation is run and the measure is NOT adopted. The same probe found why the copies scored well: REFe's own heading at t = 4.0 s is corrupted (next amendment), and a copy with factor <= 0.95 never reaches that pose. Per-variant failures: masked_top16_075: fails heading<=+0.5; masked_top16_050: fails speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; masked_top16_both: fails speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; masked_top16_both_stop: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; plain_top16_075: fails heading<=+0.5; plain_top16_050: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; plain_top16_both: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; plain_top16_both_stop: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; masked_all_075: fails heading<=+0.5; masked_all_050: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; masked_all_both: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; masked_all_both_stop: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; plain_all_075: fails heading<=+0.5; plain_all_050: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; plain_all_both: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0; plain_all_both_stop: fails progress>=0.90, speedMAE<=+0.30, cross<=+0.10, heading<=+0.5, goal<=+1.0.

> **AMENDMENT 7 -- 2026-09-27 ~13:50 Berlin (time from `date`), BEFORE any confirmation-token result of the repair
> exists.** Seen so far, EXPLORATORY, on W3's 200 tokens only: REFe's OWN heading at the last pose (t = 4.0 s, native
> index 19) is corrupted in every evaluated snapshot (005..015): median |wrap(heading - path tangent)| 1.08-1.12 rad at
> that pose against 0.02-0.06 rad at the other 19; |heading| > pi on ~60 % of proposals; its raw value tracks the final
> x (correlation -0.94), while the pod's training targets are clean there (0.003 rad). Replacing ONLY that heading by
> the t = 3.8 s heading (positions and speed unchanged) lifted the shipped pick 61.18 -> 77.94 PDMS, +16.76
> [+11.72, +22.13] (`eval/raw/e6_sub200_ep015/slow_copies.json`, `repair_last_heading_shipped_pick`).
> **Repair under test (ONE, fixed now):** on the planner's native [20, 3] output, heading[19] := heading[18] for the
> executed plan, before the NAVSIM conversion; nothing else changes. **Confirmation tokens:** Amendment 5's 923 tokens
> (43 logs, none in W3's 200). **Snapshot:** after epoch 15 (md5 `d7c59f4f2fbcbde3e2dec8f67d63a7e7`). **Measurement:**
> the unchanged pipeline dumps the shipped picks; the repaired picks are written as their own seam and scored by the
> unchanged harness beside the shipped seam. **Validity gates, all must pass before the statistic is read:** (a) every
> repaired pose's x and y are bit-identical to the shipped seam's and only the t = 4.0 s heading differs; (b) the shipped
> seam reproduces the pipeline's pick and score on every token; (c) every harness run PASSes with every token valid.
> **Statistic:** per-token PDMS(repaired) - PDMS(shipped), mean over the tokens, paired log-cluster bootstrap over the
> 43 logs (10,000 resamples, 95 %, seed 20260927); it answers "another draw of episodes" only.
> **Decision, both outcomes committed now:** **ADOPT (effectiveness and validity proven)** iff the lower bound is above
> 0 -> the repair becomes part of REFe's planner output for every evaluation from then on, the final full navtest
> included (a declared test-time change in `refe/planner.py`, recorded in MODEL_REGISTRY §14.1; earlier learning-curve
> points keep their values, with repaired values beside them where an E-6 table allows). **REFUTED** iff the upper bound
> is below 0. Otherwise **NOT PROVEN** -> not adopted. The four metric families are identical by construction (the
> NAVSIM family adapter reads positions only) and are reported as such. **Reported, not gating:** the NAVSIM sub-score
> deltas; the repair with the path-tangent heading instead; the same repair on the latest snapshot's 200 tokens. The
> model-side cause of the corrupted heading is a SEPARATE question: any training change for it needs its own proof.

> **AMENDMENT 8 -- 2026-09-28 06:32 Berlin (time from `date`), BEFORE any model output on the confirmation tokens exists.**
> **Seen so far, EXPLORATORY, on the 1,123 proxy tokens (W3's 200 + Amendment 5's 923; the defect was FOUND on them,
> so none of it is admissible for adoption):**
> - The planner's goal input (`refe/planner.py:449-474` `_goal_for` -> `code/augment_routes.py:46-70`
>   `_route_with_lane_rank` -> DriveRL `goal_position_utils.py:512` `route_goal_positions`) has NO guard for a route
>   that does not reach the ego.
>   - On two logs the scenario's own `route_roadblock_ids` do not contain the ego's roadblock; the nearest route
>     roadblock is 373.5 m / 421.1 m ahead (`raw/2026-09-28-goal-clamp/goal_trace.json`).
>   - `_route_start_index` (`driverl_runtime_map_features.py:881`) takes the nearest roadblock at any distance.
>   - The projection (`goal_position_utils.py:555-566`) clamps to the route's first point at any distance, so the goal
>     becomes "route start + 30 / + 60 m", 349-480 m ahead. The re-derivation reproduces the cached goals exactly.
> - Census of the ego's distance to its own route polyline over the 1,123 (`route_cover_census.json`, CPU, no model):
>   median 0.29 m, p90 1.54 m; > 5 m: 60 tokens, > 10 m: 58, > 20 m: 53 (9 logs), > 50 m: 45, > 100 m: 27, > 300 m: 19.
> - PDMS of the live recipe (snapshot 015 + Amendment 7's repair) falls with that distance:
>
>   | ego-to-route distance | tokens | PDMS |
>   |---|---|---|
>   | < 2 m | 1,035 | 79.6 |
>   | 2-10 m | 30 | 65.6 |
>   | 10-20 m | 5 | 79.5 |
>   | 20-50 m | 8 | 56.4 |
>   | 50-100 m | 18 | 58.9 |
>   | 100-300 m | 8 | 41.2 |
>   | >= 300 m | 19 | 2.2 |
>
> - On the 20 tokens with goal p2 > 300 m, the WHOLE fan drives 57.6 m in 4 s against a GT of 12.6 m.
>   - The executed plan scores 2.08 PDMS with the repair ON, and 0-14 under every M6 / M6b arm.
>   - These tokens cost the 1,123-token mean 1.2-1.5 PDMS in every condition (`garbage_tokens_pdms.json`).
> - A CPU re-decode of those 20 tokens with the goal replaced (`goal_clamp_probe.json`; the untouched-goal control
>   reproduces the GPU picks 20/20) scored:
>
>   | goal on the 20 tokens | PDMS |
>   |---|---|
>   | untouched (control) | 2.08 |
>   | clamped to 219 m | 27.85 |
>   | clamped to 150 m | 35.35 |
>   | straight-route goal from the ego, max(v0, 5 m/s) x 12 s | 88.38 |
>
> - The training side carries the same defect at negligible frequency: the live bank has 9 of 33,704 r0 rows with goal
>   p2 > 300 m.
>
> **Sanitisation under test (ONE, fixed now; its threshold was read off the 1,123-token census above, before any
> confirmation token was read):**
> - **Trigger:** the ego's distance to the route polyline `_goal_for` builds (the 100-point polyline, segment
>   distance) exceeds **D = 20 m**.
>   - Why 20 m: about five lane widths, so unambiguously off the route.
>   - Tokens at 10-20 m score like on-route tokens (79.5, n = 5).
>   - The 2-10 m band (65.6, n = 30) is a DIFFERENT question (lane choice on a covering route) and is left alone.
> - **Below D:** nothing changes, bit for bit.
> - **Above D:** the goal is re-derived by the SAME `route_goal_positions` (horizon 12 s, min speed 5 m/s, 2 points),
>   on a **fallback route built from the ego's OWN lane** instead of the scenario route:
>   1. Candidates: the lanes and lane connectors within 10 m of the ego (`map_api.get_proximal_map_objects`).
>   2. The start is the candidate with the best `_route_edge_anchor_score` (distance + 5 x heading error).
>   3. The route extends through `outgoing_edges` until it is >= 150 m long. At each fork it takes the successor whose
>      exit heading change is the most counter-clockwise for command LEFT, the most clockwise for RIGHT, and the
>      smallest in magnitude for STRAIGHT and UNKNOWN.
>   4. It is resampled with `_fit_route_polyline`.
> - If no lane lies within 10 m (off-map), the fallback is the straight-route goal along the ego heading:
>   p2 = max(v0, 5) x 12 s, p1 = p2 / 2.
> - **Inputs, all admissible at inference:** the map, the ego pose, v0 at t0, and the benchmark's driving command,
>   which NAVSIM gives every agent. Never GT, never future ego.
> - **Why it is not optimistic by construction on turns:**
>   - The fallback follows the MAP's lane geometry and the COMMAND. It does not follow the logged future.
>   - On a turn it produces the turn only if the lane graph and the command say so.
>   - A wrong command or an ambiguous fork gives a WRONG goal, and that cost is in the statistic.
>   - The exploratory straight-route result (88.38) is the special case where all 20 tokens were STRAIGHT on straight
>     roads. It is NOT the number this amendment tests.
>
> **Confirmation tokens (FRESH):** every navtest token OUTSIDE the 1,123 on which the trigger fires, by the CPU census
> over the full 12,146 (`route_cover_census_navtest_full.json`, computed with no model output): **422 tokens in
> 32 logs** (commands L/S/R/U: 112 / 266 / 44 / 0).
> - The full census triggers on 475 of the 12,146 tokens; 53 of those are the selection tokens, which are excluded. The list is banked in `amendment8_fresh_set.json`.
> - The same census over the 1,123 reproduces the first census exactly (0 of 1,123 differ).
> - Distance bands of the fresh set: 122 at 20-50 m, 103 at 50-100 m, 171 at 100-300 m, 26 at > 300 m.
> - The fresh set contains 156 LEFT or RIGHT tokens, so the lane-following fallback is tested on turns, not only on the straight roads it was found on.
> - **Why a second, stricter read.** The defect is a property of a log's route, so 6 of the 32 logs also hold selection tokens; only 279 fresh tokens sit in the 26 logs that hold NO selection token.
> - **Estimator:** the paired log-cluster bootstrap (10,000 resamples, percentile 95 %, seed 20260927). It is computed TWICE, both committed now:
>   - **PRIMARY:** over all 422 tokens, clustered by their 32 logs;
>   - **FRESH-LOG:** over the 279 tokens of the 26 selection-free logs.
> - 32 and 26 clusters are enough for this estimator; the M6 / M6b analyses used 136. No smaller design is needed. If a harness run drops tokens, the design stays as written and the valid-token count is reported; fewer than 20 logs in either read is NOT PROVEN.
>
> **Snapshot:** after epoch 15 (md5 `d7c59f4f2fbcbde3e2dec8f67d63a7e7`), Amendment 7's repair ON, rule v1, in both arms.
>
> **Measurement:** the unchanged pipeline (`eval/refe_navtest_seam.py`, planner forward on GPU when the PI allows it)
> writes two seams on the confirmation tokens: goal as today (OFF) and goal sanitised (ON). Both are scored by the
> unchanged harness.
>
> **Validity gates, all must pass before the statistic is read:**
> - (a) on every confirmation token the trigger fires (census re-checked in the run);
> - (b) a unit test with a mutation arm:
>   - a covering route leaves the goal bit-identical;
>   - the recorded 373.5 m case is replaced;
>   - LEFT and RIGHT forks pick opposite successors on a synthetic Y junction;
>   - deleting the trigger check must go RED;
> - (c) the OFF seam reproduces the pipeline's own pick and score on every token;
> - (d) every harness run PASSes with every token valid;
> - (e) nothing but the goal differs between the arms' inputs.
>
> **Statistic:** per token PDMS(ON) - PDMS(OFF), mean over the confirmation tokens, with the estimator and CI named in
> the design paragraph. It answers "another draw of episodes" only.
>
> **Decision, both outcomes committed now:**
> - **ADOPT** iff BOTH reads' CI lower bounds are > 0 (PRIMARY and FRESH-LOG) AND no longitudinal or lateral family component separates adversely (a
>   named interval entirely on the worse side).
>   - The sanitisation then becomes part of REFe's planner goal path for every evaluation from then on: a declared
>     test-time change in `refe/planner.py`, recorded in MODEL_REGISTRY §14.1.
>   - Earlier points keep their values; sanitised values are reported beside them where a seam allows.
> - **REFUTED** iff the PRIMARY upper bound is < 0.
> - Otherwise **NOT PROVEN**, and nothing is adopted.
> - The four metric families (families6; strategic UNAVAILABLE in NAVSIM by design) are reported for both arms. Unlike
>   Amendment 7 they CAN differ here, because the goal changes positions.
>
> **Reported, not gating:**
> - the 150 m clamp;
> - the straight-route-only fallback;
> - the sanitised seam on the 53 triggered tokens of the 1,123 (exploratory, selection tokens);
> - the NAVSIM sub-score deltas.
>
> **The final full navtest (Thursday):**
> - The score of record uses the pipeline as it stands at the moment the run starts.
>   - If this amendment has read ADOPT by then, the final score uses the sanitised goal and ALSO reports the
>     unsanitised score beside it on the same run.
>   - If it has not been read, or reads NOT PROVEN / REFUTED, the final score uses today's goal path, and the sanitised
>     score is reported beside it as a non-gating readout.
> - The triggered tokens are named in the final report in either case.
> - ⛔ D, the fallback rule and the 10 m lane radius are NOT tuned after any confirmation token is read.
>
> **Training side (next model version, NOT a live change):** the same guard belongs in the bank builder
> (`build_targets.py`'s goal path).
> - A row whose ego lies > D from its route gets the SAME fallback goal. Today 9 of 33,704 r0 rows have p2 > 300 m.
> - It is a declared recipe change: a `--declare-change` identity key, a G-DVB entry in `launch_gate.py`, and the
>   launch-gate PASS token.
> - It keeps train and test goal definitions identical. Without it the model would still see ~0.03 % of rows with a
>   far goal at train time and none at test time: harmless at that frequency, but not symmetric.
> - The live run is not touched.

**Subset first:** W3's `A1_sub200_tokens.json` (200 tokens) for E-0..E-2, then the full 12,146.
**Tier stamp:** NAVSIM v1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents, as W3 stamps it.
