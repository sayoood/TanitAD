# SPEC_P1P2 AMENDMENT A2 -- P1' : the candidate-fan probe re-run on the FINAL checkpoint (step 50,400) with a state-level negative control K4s

Stream D6, 2026-10-04, written BEFORE any 50,400 candidate-fan number exists. To be registered (sha256 + time) by the Master Mind before the GPU export is allowed to start (the gate below
refuses without the registration marker). Amends nothing already registered: SPEC_P1P2.md (sha256 `4eaf655f8e000f5ae06a4b0ba93ff05be04a037064d1c16d108e7c11c1f5f4bd`) and A1
(`def9b4c24222d47dc49de0174690279c4819a7da1dc896954d7c6b582f1392bb`) stay as registered; A2 only adds a second, independent P1 on a different checkpoint.

## 0. Status of the step-30,000 P1 (the reason for this file)

P1 on step 30,000 ran to completion (2,403 scenes exported and scored, 0 errors) and its control K4 FAILED (registered literal: F_CLEAN_DAC <= 0.05 after a +30 m PLAN shift; measured 0.71 -- the devkit's LQR
tracker damps a plan jump, so the SIMULATED ego stayed near the road; a control-design error of mine). Under SPEC_P1P2 s4 the 30k reading is **INCONCLUSIVE**. The Master Mind ruled it stays so and that the numbers
(isolated in `raw/d6_p1_WITHHELD_controls_failed.json`, seen once by the author before the gate existed) are not quoted or used. **Nothing from that file enters P1'.** P1' is a new measurement on tokens whose
fans nobody has scored, on a different checkpoint, with a control that cannot be damped by the tracker.

## 1. What changes (everything else = SPEC_P1P2 + A1, verbatim)

| item | SPEC_P1P2 + A1 (30k) | P1' (A2) |
|---|---|---|
| checkpoint | `ckpt_30000.pt` md5 `ac4e4fab87e35b20de24d8d94910d910` | **`D:/refcv7_eval_kit/ckpt/ckpt_50400.pt` md5 `b418d0fc4a92a6848c246a6a7c50207b`**, step 50,400, config `D:/refcv7_eval_kit/ckpt/config.json` md5 `e6512a01b9c70f0e4a4dac581621984a` (the same file as at 30k; argv has no `--refcv7`, so the universe is FAN117 exactly as A1: the export falls back to `anchor_traj`, K6 asserts it) |
| bridge | `code/bridge_fix/` | the SAME files (insert-only copy, base blobs in `BASE_BLOBS.txt`); the base files are re-checked against the live package by `test_default_identical.py` check 2 before launch |
| arguments | banked step-30,000 navhard bridge command | the banked step-50,400 navhard bridge command (`.../step50400/bridge_navhard.log` line 1: `--device cuda --precision auto --threads 6`, no `--exact-dedup`, same banks / speed / road-plane / inputs) + `--export-fan` |
| GPU scope | 2,403 scenes of P1_all | **all 5,912 navhard scenes** (stage-1 450 + stage-2 5,462): the export has no dependency on the 50,400 NavSim scoring, which is still running on the CPU. Model load 4.3 s + 5,912 x ~0.56 s = ~55 min of lock time |
| token sets | built from the 30k scores | **derived by rule (below) from the OFFICIAL 50,400 R7_A1 navhard scoring**, after it is complete |
| K4 | +30 m PLAN shift, F <= 0.05 | **K4s: +30 m STATE shift, F <= 0.10** (s3) |
| geometry-class strata | from the 30k plans | dropped (a 30k plan class does not describe a 50,400 plan); strata = stage, command, max-speed known/unknown, route-in-horizon kind (plan-independent) |
| the 30k P1 reading | -- | stays INCONCLUSIVE; P1' neither confirms nor replaces it, it is a separate measurement of a different checkpoint |

## 2. Token sets: the derivation RULE (the tokens are not known yet, the rule is deterministic and committed)

Source: the official per-token frame `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/scores_navhard/score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_final_scores_frame.csv`,
used ONLY after `score_R7_A1__navhard_two_stage.counts.json` reads `"status": "PASS"` and `csv_token_rows == 5912` (read from the artifact; the runner's own exit status is not evidence). Code: `code/d6_make_spec_sets_p1prime.py`.

* **S1'** = A1-50,400 DAC = 0 AND the PDM-Closed reference proposal passes DAC (map-only, `code/d6_ref_dac.py`, validated against the 30k exact re-score: equal on the 1,563 scenes, control);
* **S1b'** = A1-50,400 DAC = 0 AND the reference fails;
* **S2'** = A1-50,400 NC = 0;
* **Cpass'** = 200 scenes drawn with `numpy.random.default_rng(20261004).choice(sorted(pop), 200, replace=False)` where `pop` = sorted scenes whose four multipliers (NC, DAC, DDC, TLC) are ALL exactly 1; `N_pass'` = |pop|, HT weight = N_pass' / 200;
* **K1fail'** = 100 scenes drawn with `default_rng(20261005).choice(sorted(S1'), 100, replace=False)`;
* **P1'_all** = S1' u S1b' u S2' u Cpass'. The sets, their sizes and sha256 are written to `raw/spec_p1x_token_sets.json` and sent to the Master Mind BEFORE any fan is scored; the CPU fan stage refuses to start without that file.

## 3. The new control K4s and its calibration (committed before any P1' reading)

**K4s (replaces K4 and nothing else).** For every Cpass' scene, simulate all 117 candidates with the devkit tracker, then TRANSLATE THE SIMULATED EGO STATES laterally by +30 m (global shift of
`(-sin h0, cos h0) x 30`, h0 = the initial rear-axle heading) before the map test, and score DAC / DDC map-only (`d6_fan_score.py --shift-states 30 --no-tierc`). **PASS iff F_CLEAN_DAC (any of the 117 candidates DAC = 1 AND DDC = 1)
over the 200 Cpass' scenes is <= 0.10.** FAIL -> every P1' read is INCONCLUSIVE and there is no further amendment without a new ruling.

**Calibration (MEASURED, on a control population with NO candidate evaluation; `raw/k4s_calibration_step30000_control.json`, `code/d6_k4s_calibrate.py`).** 1,000 navhard scenes drawn at random
(seed 20261006) from the 3,509 scenes outside every 30k P1 set, ONE plan per scene (the banked step-30,000 emitted pick; and separately the STOP plan), states translated, DAC and DDC scored map-only:

| plan | shift 0 m | +30 m | +100 m |
|---|---|---|---|
| banked pick: share DAC- and DDC-clean | 85.1 % (851 of 1,000) | **1.8 % (18)** | 2.0 % (20) |
| STOP plan | 89.4 % (894) | **2.6 % (26)** | 3.0 % (30) |

The residual is the single-plan "large drivable polygon" rate (2-3 %; Wilson 95 % upper bound for 26 of 1,000 is 3.8 %). Any-of-117 over one scene's candidates is a union of nearly identical footprints, so it can only exceed the single-plan
rate by the candidate multiplicity; a bar of 0.10 is ~2.6 x the upper single-plan bound and is committed here as the Master Mind's accepted line. (The 30k post-hoc K4b, any-of-117 on Cpass: 0.060 at +30 m, 0.040 at +100 m -- seen, not used to set the bar; it lies below it.)
The unshifted rows (85.1 % / 89.4 %) are the sanity check that the scorer is not simply returning 0.

## 4. Controls and reads (unchanged from SPEC_P1P2 s4 + A1.4, restated with the 50,400 binding)

* **K1** the exported pick `cands_poses[sel_idx]` equals the banked 50,400 emitted plan (official hooks `agent_poses`; the bridge rows `poses` as the cross-check) to max abs <= 1e-4 on >= 99 % of the scored scenes, and the exact re-score of that plan equals the official 50,400 row on the 300 control scenes (Cpass' u K1fail').
* **K2** the lexicographically first Cpass' token has a CLEAN_FULL candidate; F(Cpass') = 100 % at both levels.
* **K3** STOP as candidate 117: DAC / DDC / TLC equal the banked STOP floor frame on >= 99.9 % of stage-2 scenes; exact STOP row equals the banked STOP row on the 300 control scenes.
* **K4s** as above.  **K5** batch-vs-single DAC / DDC equal on 100+ random pairs.  **K6** `sel_idx` and `sel_idx_base` < 117 on 100 % and `derived_ok` >= 99.5 % in the 50,400 rows.
* **Reads** F, PTI (HT weights 1 for the failures, N_pass'/200 for Cpass'), RND, RANK (E9-ranked), R4' (decoder pick), SLOWER (S2'), their bands and the reading branches: SPEC_P1P2 s3 + s6 with A1.1-A1.3, thresholds unchanged. Class strata are replaced as stated in s1.
* The analysis code (`d6_p1_analyze.py`, parameterised for step 50,400) withholds every read -- writing it to a separate file and putting NO reading in the headline JSON -- unless K1, K2, K3, K4s, K5 and K6 all pass.

## 5. Gate, cost and order of events

1. A2 registered by the Master Mind -> `raw/SPEC_A2_REGISTERED.txt` containing the A2 sha256 (the GPU gate refuses without it).
2. GPU (detached, behind the battery's lock via `with_gpu_lock.py`, same chain with stage P1X): export for all 5,912 scenes -> `raw/p1x_fan/fan_R7_A1.jsonl`, `raw/p1x_bridge/`. ~55 min of lock time.
3. The official 50,400 R7_A1 navhard scoring completes (not mine; read its counts.json) -> sets built and sent to the Master Mind (`raw/spec_p1x_token_sets.json`).
4. CPU (<= 3 processes, RAM-gated): fan scoring of P1'_all (2.0-2.5 k scenes), K4s on Cpass', exact controls; then the analysis. ~1-1.5 h.

## 6. What this does not claim

Nothing about step 30,000 (INCONCLUSIVE stays). The 50,400 reading, if its controls pass, is a reading of the 117-anchor fan of THAT checkpoint on navhard; it licenses a selection / generation attribution, not a retrained-model claim (H-ESTIM-SEED-1 stays open).
