<title>Teacher review: 89.8 vs 95.8</title>

# TEACHER REVIEW: why our privileged teacher scores ~6 PDMS below the paper's, and what it is worth to REFe

Reviewer: Sonnet agent, 2026-11-11. 0 GPU, no pod, no HF, no new navtest read. Repo read from the mirror at tip `876ddcb8` (`agent/arch-inf-20260803`).
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Evidence classes: MEASURED (ours, artifact) / PUBLISHED (paper, page or table) / INHERITED / ESTIMATED / HYPOTHESIS / UNVERIFIED.
Paper = DriveZero arXiv 2609.06055, Library copy `Library/papers/2609.06055_...pdf`, sha256 re-checked here: `82a00048648b155e...` (matches P4). All navtest numbers are exploratory (navtest read many times).

## 0 · Headline

1. **The released teacher is very probably the paper's checkpoint, and the paper never says how Table 4's 95.8 was produced, so the 6-point gap is NOT evidence of a worse teacher.** About **72 % of it (4.30 of 5.99) is the plan-to-tracker interface**: the same teacher plans, made trackable, score 94.10 (MEASURED, F2, exploratory, not separated from the human). The paper's own text for Table 4 is one clause: "4-s rollouts from ground-truth symbolic inputs" (PUBLISHED, p. 11). The release contains no NAVSIM export or scorer (`grep -il navsim` over `DriveRL/` = 1 file, a reward-calculator comment), so the Table 4 protocol cannot be reproduced from the release. That is the unexplained core.
2. **The remaining ~1.7 PDMS is EP (about 1.1) and DAC (about 0.8), ESTIMATED**, and nothing banked separates checkpoint, rate and protocol there. Even a transparent tracker leaves our teacher at DAC <= 99.41, because 72 tokens (0.59 %) leave the drivable area in the rollout itself (MEASURED, D1), against the paper's 99.9.
3. **For REFe the teacher is no longer the binding cap.** P6 (91.20 on full navtest, `D-P6-GAP-1`) is already above our teacher (89.79 on the same 12,146 tokens, registry), and its DAC failures (2.05 %) are below the teacher's (4.77 %). Unpaired point comparison, no interval. A matched teacher or P5-C is worth **ESTIMATED 0 to +0.6 PDMS (central ~+0.2)** on top of P6/P7, below the rig's seed floor F = 0.33 as a central value. Ceiling from the DAC tail alone is +1.2.
4. **Recommendation:** do not spend A40-days on P5-C or a teacher re-roll before two CPU checks (section 6, E1 and E2). Do not touch P7, which is training on the old targets.

## 1 · Checkpoint identity

**Question: is `checkpoint_2400.pt` the checkpoint the paper reports?** Verdict: **SUPPORTED, not proven.**

| evidence | class | source |
|---|---|---|
| Paper: Single-Ego teacher "2,400 updates", 5.7 M parameters, "its reported 2,400-update checkpoint" (Table A7 caption) | PUBLISHED | paper p. 9 (3.1), p. 28 (Table A7) |
| Release: `driverl-teacher-u2400`, `checkpoint_update: 2400`, 63 tensors, **5,702,413 params** (exact). Our local file md5 `0b4236016856704b564bdbe33043185a`, 22,832,846 B | MEASURED | `DriveRL/release/driverl_checkpoints.yaml:9-14`; `RELEASED_ARTIFACT_TEARDOWN.md` R1; P4 result s1 |
| Our closed-loop nuPlan CLS with the release's own runner, Test14-hard NR: **89.755151** (272/272, 0 failures) against the paper's **89.97**. Gap -0.215, about 1.1 SE on an ESTIMATED SE of 0.20 taken from one 10-scenario replicate pair that moved 0.93. One draw, no interval | MEASURED | `PK/REVIEW_4_FINAL.md` s5 (lines 450-500) |
| The release self-declares `evaluation_status: artifact_bundled_pending_public_eval`, `evaluation_tts_enabled: false`, `evaluation_protocol: routefix-six-split-replay` | MEASURED | `driverl_checkpoints.yaml:22-24` |
| Release notes name no table or score for the checkpoint; `docs/driverl_repro_eval.md` lists release ids only | MEASURED | `docs/driverl_repro_eval.md:9-12` |

Reasons for caution (all read from the artifact and the paper, none resolved):
* **The released network is not exactly the described one.** The paper's critic has **eight** channels (hard, goal, six soft; Eq. 8, A.4, p. 26). The checkpoint's value head is `1024->9` (teardown R4, adds `soft_other`). Paper Table A2 gives **15-D** per actor frame; the checkpoint's `agent_mlp` takes **16**. `combiner_mlp` takes 1025 = 4 x 256 + one unexplained scalar. Parameter count is unaffected, so "5.7 M exact" cannot discriminate. PUBLISHED vs MEASURED.
* **Reward literals in the release's teacher yaml differ from the paper's "reported teacher" reward.** Paper A.4: off-road base **-2.0**, goal arrival within **1.5 m** worth **0.5** (p. 26). `driverl_teacher.yaml:65-69`: off-road **-1.0**, goal **3.0 m / 0.3**, `ttc_weight` 0.5. The *self-play* yaml carries exactly the paper's values (-2.0, 1.5 m, 0.5). The yaml comment says the block is "used by the inference/TTS runtime", so it may not be the training reward; **UNVERIFIED which**. Paper Table A6 describes the self-play variant as trained 4,800 updates, the release ships `checkpoint_3800.pt`, so at least one release artifact is not the paper's exact run (teardown R8).
* The checkpoint reproduced one of six published splits (Test14-hard NR) within noise. Val14 and Test14-random were never run (MEASURED absence in the banked set; UNVERIFIED elsewhere).

How to find out (CPU, no navtest): E3 in section 6.

## 2 · Evaluation protocol: from an RL policy to a NAVSIM 4 s plan

**What the paper says (PUBLISHED, p. 11, Table 4 and text):** DriveRL "achieves 95.8 PDMS with 4-s rollouts from ground-truth symbolic inputs", NC 99.8, DAC 99.9, TTC 99.1, C 99.0, EP 91.5. Nothing on rate, horizon encoding (NAVSIM wants 8 poses at 0.5 s), tracker, route, or whether rank-0/deterministic mode is used. The only related statement is for the *student's* supervision: the frozen teacher "is rolled out with all background actors following the driving log" per logged frame (p. 7, 2.2). PDM-Closed (89.1) in the same table is scored by NAVSIM's own simulator, so a tracker is the default for every row.

**What we run (MEASURED, `PK/eval/teacher_navtest_seam.py:1-30`, `PK/refe/build_teacher_rollouts.py:101-215`):** hydra-instantiated release yaml, TTS off, Beta mode, 5 Hz policy inside a **10 Hz** sim, release `DriveRLOneStageController` (bicycle, max accel 4.0, steering rate 0.8), ego seeded at the logged state, 2.0 s history from the nuPlan DB, background agents replay the log, 4.0 s, rank 0 route. The 20 rear-axle poses at 0.2 s are interpolated to NAVSIM's 8 poses (`to_navsim`, <= 0.036 m error). NAVSIM then re-simulates those 8 poses with its LQR tracker and bicycle model, as it does for every method.

**Does protocol explain most of 89.8 vs 95.8? Yes for about 72 %, and this is MEASURED, not hypothesised; what it cannot tell us is what the paper did.**

| step | PDMS | DAC | EP | source |
|---|---|---|---|---|
| ours, registered | 89.81 [88.56, 91.00] | 95.23 | 85.18 | `RESULT_P4` s2 |
| D1: of 578 DAC failures, class C (rollout and plan compliant, tracker pushes it off) | | 504 = 87.2 % | | `d1_dac_anatomy.json` |
| F2: same teacher plans, tracker-compensated (tracker deviation median 0.45 -> 0.05 m) | **94.10** [93.59, 94.60] | 99.01 | 88.76 | `f2_result.json` |
| paper | 95.8 | 99.9 | 91.5 | PUBLISHED Table 4 |

Residual after F2 = -1.70 PDMS. ESTIMATED one-at-a-time swap of the paper's sub-scores into the PDMS formula on our compensated values: DAC +0.8, EP +1.1, NC +0.2, TTC +0.1, comfort -0.1 (paper's comfort is lower, 99.0). EP among surviving tokens: ours about 89.9 (85.18 / (1 - 0.0534), ESTIMATED, assumes zeroed tokens contribute EP 0), paper about 91.8 (91.5 / 0.997). So after the tracker the teacher still reads ~1.8 EP points slower on alive tokens, and our teacher's EP gain over the human on alive tokens is about +3.0 against the paper's +4.0. Cause UNEXPLAINED.

Three facts that bound what the protocol could be:
* **Our 10 Hz sim is the published closed-loop protocol**, not an invention: the same planner and controller reproduce Test14-hard (section 1), and `target_sample_rate_hz: 5.0` is the training rate (`driverl_checkpoints.yaml:19`).
* **The teacher is rate-sensitive.** The same 47 held-out samples rolled at 20 Hz differ from the banked 10 Hz targets by a median 0.63 m, p90 1.85 m, max 2.99 m; only 27.7 % are within 0.25 m (MEASURED, `raw/2026-10-05-teacher-navtest/c3_heldout_20hz_mutation.json`). The teacher's class-C margin deficit against the human is 0.37 vs 0.60 m = 0.23 m, the same size as the median rate effect (`D-REFE-SIMRATE-1`: ADE 0.23 m, FDE 0.60 m). So an unknown rate or hold convention in the paper's navtest protocol could plausibly move the margin by the size of the effect. HYPOTHESIS, never tested directly.
* **Tracker-free ceiling is 99.41 DAC** (class A + B = 74 of 12,110; MEASURED, D1). To reach 99.9, a protocol needs rollouts with at most ~12 off-road tokens, which our rollouts do not give. So a pure tracker story cannot close DAC; something in the rollout itself differs, or the paper's DAC is computed on a different path.

⇒ Verdict on Q2: protocol is the leading explanation, quantified at 4.3 of 6.0 PDMS; the last 1.7 is open.

## 3 · Inputs: route/goal, map, agents

| input | paper (PUBLISHED) | ours (MEASURED) | match? |
|---|---|---|---|
| goal | two anchors, near and far, along the route polyline, recomputed each step; look-ahead L = min(L_remain, 12 max(v, 5)), anchors at midpoint and endpoint (A.2, p. 25) | `route_goal_horizon_s 12.0`, `route_goal_min_speed_mps 5.0`, `route_goal_pair_mode "legacy"` (`build_teacher_rollouts.py:~135-140`) | yes, by construction of the release code |
| route | formal nuPlan deployment: current ego + route polyline; release applies `correct_route_roadblock_ids` | scenario's raw `get_route_roadblock_ids()`; DriveRL corrects it itself (`feature_builder.py:407-413`); `route_roadblock_correction: true` in the release | yes. UNVERIFIED: which route NAVSIM-navtest tokens give the paper's Table 4 run |
| map | nuPlan vector map, 256 tokens, 64 lane centers, TL state attached | nuPlan maps v1.0 through the release feature builder; `enable_traffic_light_features: true` | yes |
| agents | boxes, 5 frames at 5 Hz, up to 96 | nuPlan DB tracks, log replay, 2.0 s history | yes for tracks; the **scorer** uses NAVSIM/OpenScene's own tracks; 36 tokens where DB and OpenScene disagree are excluded (primary set 12,110) |
| future leak | none | `future_steps: 0` (yaml:106), no oracle future | yes |
| ego state | 9-D | seeded at the logged state | yes |

Known departure: 24 tokens had less than 2.0 s of DB history and ran with the longest available buffer (`history_clamped`; MEASURED, P4 s1). No input difference found that would explain 6 points. Off-route goals (n = 475) are handled by the teacher's own route correction (PDMS 86.6 vs the student's pre-A9 55.5), not an input defect.

## 4 · Architecture and training: paper vs what we run

We run the released inference artifact only; no training code exists in the release (teardown R5: no optimiser, no backward, no GAE, no IDM). So "what we run" for training is the paper's text, not ours.

| item | paper (PUBLISHED) | release / ours (MEASURED) | difference |
|---|---|---|---|
| width / heads / layers | 256 / 4; ego-to-agent 2, ego-to-map 1 (Table 1) | yaml `embed_dim 256, num_heads 4, num_agent_attention_layers 2, num_lane_attention_layers 1` | none |
| capacity | 96 agents, 256 map tokens, 64 lane centers | `max_agents 96, max_lanes 256, max_lane_centers 64` | none |
| params | 5.7 M | 5,702,413 | none |
| critic | 8 channels (hard, goal, 6 soft) | `value_head` 9 outputs | **+1 channel** |
| actor input | 15-D per actor frame | 16-D `agent_mlp` | **+1 feature** |
| combiner | ego, goal, kinematics, map = 4 x 256 | 1025 inputs | **+1 scalar** |
| action | jerk [-8, 5], steering rate +-0.8, accel cap 4.0, steer cap pi/3 | yaml:21-34 match | none |
| dynamics | kinematic bicycle (Table A4; the extracted wheelbase cells are garbled in our text copy, UNVERIFIED) | `nuplan_bicycle_*`, wheelbase 3.089, accel tc 0.2, steer tc 0.05 | none visible |
| reward | off-road -2.0, goal 1.5 m / 0.5, six soft criteria with curb, TL not in the reward (A.4) | yaml: off-road -1.0, goal 3.0 m / 0.3, `curb_clearance_distance 0.7`, `comfort 0.5`, `ttc_weight 0.5` | **differs, but the yaml block is a TTS runtime block; UNVERIFIED if it is the training reward** |
| training | PPO, Muon, 2,400 updates, 96 GPUs, 5 Hz, 110-step rollouts, 922,703 scenes, 1:1 log/IDM (Table A1) | not releasable or reproducible | n/a |
| execution | Beta mode | Beta mode, TTS off | none |

Config differences to log: three architectural (critic channel, actor feature, scalar) and one reward block. None changes the policy forward pass in a way we can bound, and the curb-clearance threshold of 0.7 m in the yaml sits close to the 0.60 m / 0.37 m margins we measured (consistent with a policy trained to clear the curb by about 0.7 m and ending up inside it in turns; HYPOTHESIS).

## 5 · What it costs REFe

**Where REFe stands against the teacher (MEASURED, registry and `D-P6-GAP-1`; point values, no paired interval):**

| | PDMS | DAC fails | note |
|---|---|---|---|
| our teacher, plain protocol | 89.79 (12,146 tokens) | 4.77 % | `refe_p4_teacher_navtest` |
| P6, full navtest | 91.20 | 2.05 % DAC = 0 | 384 zeroed tokens, NC = 0 on 1.12 %, TTC = 0 on 4.92 % |
| paper student DriveZero, ViT-L, 4 cams | 94.55 (Table A13) | 0.75 % implied by DAC 99.25 | PUBLISHED |
| paper student on DriveRL-only supervision (ViT-S) | 93.61, DAC 98.60 | 1.4 % | PUBLISHED, Table 7 |

The paper's own student trained on **unaugmented teacher rollouts** reaches DAC 98.60 and 93.61 (Table 7). So teacher targets of this family can yield a student with DAC failures near 1.4 %, i.e. a thin-margin teacher is not by itself fatal. P6 sits at 2.05 %, not at the teacher's 4.77 %: the student is already better than its teacher on DAC and PDMS, so the teacher is not the binding cap for P6/P7.

**Value of fixing it, ESTIMATED (basis stated):**
* Ceiling from DAC alone: P6 DAC zeros cost about 2.05 % x ~0.9 = 1.85 PDMS; paper-level 0.75 % costs ~0.68; gap **+1.2 PDMS**, if the whole tail were teacher-inherited. It is not: ZT3 puts 60 % of u400's DAC-only zeros as near-twin grazes the scorer is blind to (selection side), and **ZT5 measured the execution-side net reach at about 0** (tracker saves as many picks as it breaks; best admissible plan-side fix on dev, heading-consistent plan, was -0.30 [-0.72, +0.04]; `D-REFE-ZEROTAIL-4`, `raw/2026-10-17-tracking-execution/`). That is the closest banked evidence on what P5-C targets, though P5-C changes training targets rather than post-hoc plans.
* Against P5-C: dual compliance of Tc20s falls 91.75 % -> 86.64 % (interp8 cost, cubic), learnability 1.69x acceleration and 1.76x heading rate against the 2x bar, families read pre-distorted references (P5, P5-S, `D-REFE-P5S-SMOOTH-1`). Only 29 % of the *student's* DAC failures are tracker-made (F2 ii), against 83 % of the teacher's.
* Against a matched teacher protocol (rate or margin change): EP alive gap of ~1.8 to the paper's teacher is worth about 0.4 PDMS if fully inherited; TTC is below the student's own failures.
* **Central ESTIMATE for P5-C on P6/P7: +0.2 PDMS, range 0 to +0.6**, probability of a separated gain (> F = 0.33 and CI > 0) about 20 %. A matched-teacher re-roll: same range, higher cost, same uncertainty. Both ESTIMATED from the arithmetic above, not measured.

**Cost and interaction with P7 (INHERITED where cited):**
* P5-C: fine-tune the WTA student on Tc20s, REVIEW_7 s4 P5 estimate "1-2 A40-days" plus CPU relabel (INHERITED, never timed); needs the PI's go, GPU and a pod (`RESULT_P5S` s5).
* A matched teacher protocol (new rate, margin-aware): re-roll the bank (the 3-C rebuild cost ~23 h CPU and an idle A40 on P6, `p6-newpod-rebuild`) plus a full retrain **~6.2 A40-days** (P7, `SPEC_P7.md:12`).
* **Interaction with P7:** P7 trains from scratch on the reserve-free bank of 150,603 tuples built from the *old* rank-0/rank-1 targets, with attribution already carried by one bundle (P7.1 head + 9.84 % fewer scenes). Changing targets now would void its launch-gate token and its attribution. Run nothing against P7; if P5-C is ever run, it is a **fine-tune of P7's checkpoint after P7 reads**, with P7 as its matched baseline, and a head-off replicate is not needed because the lever is the target.
* Seed floor F = 0.33 PDMS (one draw): any P5-C result below ~0.66 needs a replicate to be a lever claim (`H-ESTIM-SEED-1`).

## 6 · The cheapest discriminating experiments (max three, CPU first)

**E1. Rate/hold sensitivity of the margin and the tracker failure, on banked 10 Hz vs 20 Hz targets (CPU, hours, no navtest, banked only).**
Banked 20 Hz rows exist (4,265 pod + 18,122 dev box, "kept aside", `D-REFE-SIMRATE-1`; location UNVERIFIED), and 1,018 banked 10 Hz held-out targets exist (`d1_navtrain.json`). Step 1: compute the overlap by sample id; run only if >= 300 samples overlap. Step 2: NAVSIM sim8 DAC failure and footprint margin (`p4_dac_margin.py`, `p4_d1_navtrain.py`) on both rates, paired log-cluster bootstrap.
* Outcome A: 20 Hz (or a 5 Hz-native variant, if any rows exist) sim8 failure <= 3 % and class-C margin >= 0.55 m, with 10 Hz at 8.06 % / 0.37 m: **rate or hold convention drives the thin margin**; the cheap fix is a teacher re-roll at that rate, P5-C is unnecessary; the paper plausibly used it.
* Outcome B: both rates fail 6 to 10 % with margin ~0.4 m: the margin is the policy (curb clearance 0.7), **rate is eliminated**, P5-C or margin-aware selection stays the only lever.
* Outcome C: overlap < 300: report UNVERIFIED and use E2 alone. A new 20 Hz or 5 Hz roll on navtrain held-out logs is a proposal only (about 0.2-0.5 CPU-h per 100 samples, INHERITED from the 47-sample run: 90 s).

**E2. Tracker-transparent scoring of the banked held-out targets (CPU, hours, banked only).**
Score the 1,018 banked targets' own 20 rollout poses as "executed" (bypass the LQR) with NAVSIM's NC, DAC, TTC, EP, comfort calculators, and compare with the sim8 read already banked. Do not run it on navtest.
* Outcome A: transparent DAC >= 99.3 and PDMS-like within 1 point of the human on the same samples: the paper-level 95.8 needs only a tracker-transparent protocol; **the teacher is paper-grade, and the REFe-side action is to rescore, not retrain**. P5-C's expected value falls to ~0.
* Outcome B: transparent DAC < 98.5: the teacher's own rollout is the defect (class A, 0.59 % + more on navtrain); then the paper's 99.9 implies a checkpoint, yaml or input difference and E3 becomes the priority.

**E3. Checkpoint-identity replicate (CPU, nuPlan not navtest; proposal only, not run).**
Run `checkpoint_2400.pt` on the release's Test14-random NR (261 scenarios, published 94.50, Table A10) with the release runner, twice (identical config), plus Val14 NR (95.16) if time allows. Compare to the paper with the replicate spread as the noise unit.
* Outcome A: both within ~1 point of 94.50: checkpoint identity upgraded from SUPPORTED to near-confirmed; the 6-point navtest gap is then protocol and input only, and the paper's navtest teacher plans came from a different conversion (ask the authors; the release has no NAVSIM path).
* Outcome B: >= 2 points below on both: the released weights are not the paper's (compare the 9-channel critic and 16-D actor input), the teacher-fidelity question closes as "different teacher", and no protocol work can reach 95.8.
Cost: CPU hours (Test14-hard took UNVERIFIED hours; measure on 10 scenarios first, as REVIEW_4 advised). No GPU, no navtest.

Priority: E2 first (answers "protocol vs teacher" on banked data), then E1, then E3.

## 7 · Gaps and open items (stated, not narrowed)

* I did not open the checkpoint tensors; the 9-channel and 16-D facts are INHERITED from the teardown (R1, R4) re-checked against paper text, not against the weights.
* Table A4's constants did not extract cleanly from the PDF text (columns scrambled); I did not use them.
* Whether the paper's Table 4 route came from NAVSIM's `route_roadblock_ids` or a corrected one is not stated.
* The P6 vs teacher comparison is unpaired and unbootstrapped (different token sets, one seed).
* No paper page number was checked in the PDF viewer; pages are the printed page footers in the text layer (Table 4 p. 11, A.2 p. 24-25, A.4 p. 26, Table A6/A7 p. 27-28).

## 8 · Deliverable manifest

| artifact | where | copies |
|---|---|---|
| `TEACHER_REVIEW.md` (this file) | `repo-pending: E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/teacher_review/TEACHER_REVIEW.md` | **ONE place only**; the Master Mind lands it |
| scratch (`F:/tanitad_scratch/teacher_review/w/`: paper text, registers exported from the tip, 13.7 MB) | deleted at the end of this run | none kept; every file is re-creatable from the tip and the Library (see deletion log below) |

Integration escalation: none required, but the Master Mind should (a) decide whether to run E2 and E1 before any P5-C go, and (b) note that `D-REFE-TEACHER-NAVTEST-1` should carry the 72 % protocol attribution and the "paper protocol not in the release" fact, which it does not state now.

### Deletion log (CLAUDE.md rule 4)

Deleted by literal path, all scratch of this review, none the only copy: `F:/tanitad_scratch/teacher_review/w/dz.pdf` (9,392,899 B, Library copy exists), `dz.txt` (126,392 B), `goals.md` (3,796,906 B), `p5.md` (22,180 B), `p5s.md` (12,700 B), `p7.md` (22,463 B), `prereg4.md` (15,594 B), `reg.md` (670,970 B), then the empty dirs `F:/tanitad_scratch/teacher_review/w` and `F:/tanitad_scratch/teacher_review`. Total about 14.1 MB.
