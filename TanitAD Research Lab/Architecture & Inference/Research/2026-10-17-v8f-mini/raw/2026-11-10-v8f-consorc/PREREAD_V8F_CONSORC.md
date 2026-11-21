# PREREAD_V8F_CONSORC: on the adopted base DAG-OACC, does the tactical headroom still sit in T's continuous constraints, and in which kind? (Master Mind, 2026-11-10, written before any number of this read exists)

Status: rules fixed before the run. Zero training, CPU. Instrument `theory/consorc/cons_orc_read.py`, the loop of `theory/topk/topk_pair_read.py` (the T1-kin first-tick batch build of `theory/otband/read/topk_read.py`) with oracle decodes added. The sha256 of this card and of the script go in `PREREAD_V8F_CONSORC.sha256` before the run.

## Why
* **This is the committed trigger of `PREREAD_V8F_TOPK_PAIR`** (H-V8F-TOPK-PAIR-1) under DIVERSITY or MIXED: "the next tactical lever is the continuous constraints; the card reads T's constraint error on the adopted base: the H-V8F-TAC-2/-3 line, re-measured on DAG-OACC at T1-kin". Its first arm (DAG-OACC s0, MEASURED, T1-kin first tick) reads captured share S = 0.24 and a perfect manoeuvre classifier's ceiling of -0.132 m [-0.165, -0.101] against a best-of-3 headroom of -0.297 m on tactical ADE. Both outcomes that S can still reach at seed 1 lead here.
* **On M2b the constraint line is measured and ends at perception** (all T0 or T1-class open loop, M2b or its O-side variant): the constraints carry 0.88-0.90 of the tactical-code oracle and the decisions 0.07-0.12 (H-V8F-TAC-2); the cells interact and O cannot use a partly right code (H-V8F-KIND-1); once O consumes partial codes, the speed cells are the candidate carrier (H-V8F-KINDX-1, +0.716 m at s0); T's speed error is information-limited, not a head bias (H-V8F-SPEEDCAL-1). The levers in flight aim there: measured speed into the world model (EGOWM, box 2) and LoRA on the encoder (pod).
* **None of this was read on DAG-OACC.** Its DAgger rows train on the model's own rollouts and carry its tactical gain (H-V8F-OTBAND-1 ladder, tactical ADE -0.82 / -0.62 m), so the place of the remaining headroom may have moved.

## Arms (checkpoints as banked; no training; all dumped in one environment by the OTBAND read)
* `PARITY_DAGOACC_s0`, `PARITY_DAGOACC_s1`: the adopted base (the question's object).
* `INIT_M2b_s0`, `INIT_M2b_s1`: the reference (does DAgger move the headroom).
* `PARITY_DAGOACCREP_s0`, `PARITY_DAGOACCREP_s1`: DAG-OACC's argv at training seed 10 / 11 with a later trainer file (seed plus trainer version); beside only, as the run-to-run spread of every share below.

## Decodes per window (eval-141, 2,399 windows per arm, first tick, same batch build as the pair read)
T's own code `own` = `forward_topk`'s `code` (the code `forward_infer` executes). The GT code `gt` = `V8FMini._code_gt(lab_t)` (the code O is teacher-forced on in training; absent constraint slots 0), on windows with `lab_t.has`.
* **TOP1:** the plan on `own`. **IDENT:** `forward_diag(own)`, a control.
* **GT-ALL:** `forward_diag(gt)`: the full tactical-code oracle (on-distribution for O: DAG-OACC trains O on the whole GT code with p = 0.2, `code_mix whole`).
* **OWN-DEC:** `gt` with the decisions (goal, latdec, londec; 34 indices) set to `own`. **OWN-CONS:** `gt` with the 26 constraint cells set to `own`.
* **OWN-<kind>** for the six kinds of `CODE_KINDS` (speed, angle, tte, timegap, geom, leadgap): `gt` with that kind's cells set to `own` (leave-one-out, as H-V8F-TAC-3).
* **GT-SPEED-ONLY:** `own` with the speed cells set to `gt` (beside: off-distribution for an O trained on whole codes, H-V8F-KIND-1).
* **T accuracy:** per window and kind, the mean of (own - gt)^2 over the PRESENT constraint cells of that kind (normalised units, `cons_mu` / `cons_sd`), NaN where none is present.

## Quantities and estimator (tactical band, knots 11-30, is the decision band; operative and full bands beside)
H = GT-ALL - TOP1 (headroom). cost_x = OWN-x - GT-ALL for x in {DEC, CONS, each kind} (what T's own part x costs on a right background). share_x = mean(cost_x) / |mean(H)|, a ratio of MEASURED means (ESTIMATED, no interval). Every difference is paired, `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, on the windows with `lab_t.has` (n reported); it answers "another draw of episodes" only, the planner is deterministic, and the training-run spread is the REP pair beside. Rows per decode: ADE (decision row), FDE, speed, acceleration, along-track, cross-track, heading, yaw-rate, curvature, as `read_lib.band_rows`. T accuracy: DAG-OACC - M2b per kind, paired, same windows. STRATEGIC: PAUSED (PI item 25). This is an oracle decomposition at T1-kin first tick, never driving performance.

## Committed outcomes (tactical band ADE, DAG-OACC, per seed; both seeds must agree, else SEEDS-DISAGREE)
* **CONS-CARRY:** cost_CONS separated on the worse side and share_CONS >= 0.7.
* **DEC-CARRY:** cost_DEC separated on the worse side and share_DEC >= 0.3.
* Both fire: **BOTH**. Neither: **DIFFUSE**.

## What each outcome triggers (Rule Zero)
* **CONS-CARRY:** the headroom on the adopted base is in T's constraints, as on M2b. The kind with the largest cost_k (separated, at both seeds) is named. If it is speed: the speed-information levers in flight (EGOWM, LoRA) are the right aim on this base too, and each of their reads adds this decomposition on its own checkpoints. If it is another kind: a T-side card on that kind's information source (what T would need to see), drafted the same day.
* **DEC-CARRY:** DAgger moved the headroom into the decisions: a manoeuvre-classifier card on DAG-OACC, aimed at the GT-PAIR ceiling of the pair read. With the pair read's MIXED, its selector arm rides along.
* **BOTH:** both cards, the larger share first.
* **DIFFUSE:** the headroom is not in one part of T's code on this base; the next lever is O's use of the code: the card reads GT-ALL's own tactical residual (O with a perfect code) against TOP1 and against M2b's.
* **Beside, no bar:** GT-ALL's tactical ADE (what O plans from a perfect code); GT-SPEED-ONLY - TOP1; T accuracy DAG-OACC - M2b per kind; the same shares on M2b (against H-V8F-TAC-2's T0 values, no tolerance: different tier) and on the REP pair.

## Controls (the read is INVALID if one fails)
1. Each checkpoint's sha256 equals its bank manifest where one exists.
2. TOP1 equals the arm's dump `ol` at tick 0 (max |diff| <= 1e-4 m), the window set equals the dump's, and the GT future equals the dump's.
3. IDENT equals TOP1 (max |diff| <= 1e-5 m).
4. The replacement path is exact: `gt` with all 60 indices set to `own` (OWN-ALL) equals IDENT, max |diff| <= 1e-6 m; and the decision and constraint index sets partition the 60 indices.
5. PARITY_DAGOACC_s0's TOP1 tactical ADE equals the pair read's (`theory/topk/out/TOPK_PAIR.json`) to 1e-9 m.
6. Memory: 10 GB free before each arm, 7.5 GB before each chunk (the planner gates' 6 GB guard keeps its margin); the read starts only after the pair read has ended.

## Log
* 2026-11-10 written before the run.
