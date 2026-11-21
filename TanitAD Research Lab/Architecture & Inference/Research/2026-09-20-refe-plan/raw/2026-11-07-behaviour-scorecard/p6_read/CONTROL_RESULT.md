# CONTROL_RESULT -- the P6 read run on u400 (T0, CPU, banked data only, no P6 value read). Machine-readable: `CONTROL_RESULT.json`.

Same code path as the P6 read (`selc6_analyze.py`; `p6_control.py` drives it). Result: **PASS** (all data gates green, every deliberate-regression arm fails as required, u400-as-P6 verdict REFUTED).

## Reproduction of `u400_reference` (pinned content, sha `1ad5776a...`)  -- G6-REPRO, max abs diff **0.0** over every leaf (R1, R2 tie, R2 legacy, R3)
| quantity | this code | card |
|---|---|---|
| TTC head, pred vs speed, paired subset (n = 52) | -0.546 [-0.640, -0.449] | -0.546 |
| NC head, pred, paired subset (n = 38) | -0.491 [-0.606, -0.389] | -0.491 |
| TTC / NC pred, ALL 154 CRUISE tokens (section 4's S_h) | -0.479 / -0.443 | not in card table (AMBIGUITIES 4) |
| weighted CRUISE slow-pick share | 0.4124 [0.2747, 0.5690] | 0.412 |
| u400 without its lane factor (R3) | 0.3664 [0.2292, 0.5212]; paired delta -0.0460 [-0.1039, -0.0005]; 61 CRUISE picks change | 0.366 |
| legacy (argsort-of-argsort) estimator vs `p1_diag_all.json` Q4 (NC, DAC, EP, TTC, pred and true, n) | max abs diff 0.0 (<= 1e-12 required) | -- |

## Other gates on u400 (G6-1 and G6-3 use the banked u400 full-navtest seam and csv)
G6-PIN pass (3 pins accepted as AMENDED, AMBIGUITIES 1-2) | G6-0 1058/1058, RED argmin disagrees 1058, RED flipped-p3 disagrees 896 | G6-1 1058/1058 within 1e-3 m (max 0.0), RED cyclic shift fails 1058 |
G6-2 ego bit-equal 1058/1058, CRUISE formula 154 = file 154 on 1058/1058, perturbed ego equal 0 | G6-3 bit-equal 1058/1058 (RED arm here is a +0.01 stand-in; the real arm is a harness column) |
G6-4 finite, valid, PDMS recompute max abs diff 0.0 (bar 1e-9) | **G6-FP32 u400: 0.0 of 16,712 x[19] in [32, 64) m on the 0.25 m grid (bar 1 %)**.

## Deliberate regressions (each must FAIL; all do)
| arm | outcome |
|---|---|
| argsort-of-argsort ranks in the tie block (the bug behind the P1 retraction) | G6-REPRO max abs diff 116.0 vs 0.0 -> RED |
| predicted and true swapped | sign of every one of the 4 gaps flips -> RED |
| unweighted strata (0.5 / 0.5) | share differs -> RED |
| **trajectories snapped to the bf16 0.25 m grid** | G6-FP32: 100 % on grid -> RED |
| same grid fed to the slow-share (why the gate exists) | share 0.4124 (fp32) -> 0.3982 (on grid), a 1.4-point move from the grid alone |
| pick shifted by one index | G6-0 RED |
| ego v0 + 1 mm/s | G6-2 RED |
| EP sub-score + 0.01 | G6-4 RED |

## The verdict machinery on u400 treated as P6
`REFUTED`: G_TTC -0.451, G_NC -0.396 (pessimism persists), share 0.412 > 0.20, so (A), (B), (C) false. This is the expected sign: u400 is the reference the card calls pessimistic.
`test_p6_read.py` (22 tests, 56 s): verdict table incl. inclusive bars and CI-hi strictly < 0, tie-vs-index ranks against scipy, FP32 gate, control reproduction, a full synthetic P6 read in the real file formats (token order reversed), bf16 dump -> INSTRUMENT FAILURE, labeller bank -> H-P6-SPEED-1 VOID, missing inputs -> red.

## Not proven here (UNVERIFIED)
`selc6_dump.py` and `selc6_score.py` were not run on a forward / the harness (no P6 checkpoint; GPU held by another job; scoring is the card's 1.3-1.8 h). They import the proven selc code unchanged and only py_compile was run. RUNBOOK step (b) is the 3-token GPU smoke; the harness path is `selc_score.py`'s own, only `WORK`/`PREFIX` are overridden.
