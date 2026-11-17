# BUILD_AMEND4: build report for AMEND_HWMPLAN_4 (L2-IMG), 2026-11-17

Staged, nothing committed, nothing pushed. No SCORE window was planned under prim / prim23. The only SCORE windows planned are batch 0 of LON-PLAN's banked A0 plans, under `off`, as C-OFF's reuse condition (section 8.2). CPU only. Tier T0.

## Pieces (all in F:/TanitAD-scratch/actrsp/lonplan/amend4/build/)
| file | what it does |
|---|---|
| `l2img.py` | the shadow. `plan_unit` (l2img.py:163) = one batch of one arm. `off` (without record) is `planner.plan_windows` verbatim. prim / prim23: `PrimShadow.opt` (:92) replaces `st._optimize` for one call (instance attribute via `instance_attrs`, :62, restored on return or raise); `M2` / `M3` (:35 / :42); `donor_shift` (:52, the RS rule, M3); `image_residual` (:151, fresh recompute); mutation hooks no_shadow, decode_knots, noise_from_planner_gen, frozen_p (controls only; the wrapper refuses them) |
| `amend4_run.py` | run wrapper in LON-PLAN's layout. `split_l2` / `check_declared_l2` (:31 / :38, C-L4-IMG), `run_arm` (:106), `readback` (:147, `l2_override` is the regression arm). Reuses `lonplan_run` (declared==built, fresh folder, sha256s, thread rule) |
| `amend4_stage.py` | the one table of every arm's argv (TRAIN and SCORE), CLI `plan` / `argv` |
| `amend4_controls.py` | every section 8 control as a pure check plus the per-chassis runner and `merge` |
| `amend4_read.py` | section 5-7 readout: label rules (:20-:100), contrasts through LON-PLAN's `contrast_l1full_strat.py`, 2-0 rows, DIR-ACC (`lonplan_diracc`), REACH / IMG-residual / COST2 rows, TRAIN stage, `RESULT_AMEND4.json` + `.md` skeleton (`render_md`) |
| `run_amend4.sh`, `amend4.cmd`, `launch_amend4.ps1` | launcher (WMI, CreateFlags 0x08000200, `cmd /c X.cmd < NUL`, Git\usr\bin\bash.exe, PATH export). Last line `ZZ AMEND4 READ DONE ZZ` or `ZZ AMEND4 READ FAIL ZZ <why>` in `amend4/read.log` |
| `test_amend4.py`, `run_tests_amend4.cmd`, `env_amend4.sh` | 33 tests |

## 8.1 reachability trace (flag to loss)
argv `--l2-image` -> `amend4_run.split_l2` (:31) -> `check_declared_l2` (:38; not named = refuse, parsed must equal declared) -> `run_arm` -> `_unit` (:61) -> `l2img.plan_unit` (:163) -> `attrs["_optimize"] = sh.opt` -> `PrimShadow.opt` (l2img.py:92) -> `cost_p` -> the library's `_optimize` (AdamW, `backward`). C-REACH measures the gradient on p. Defaults: `--l2-image` has no default in the wrapper (must be named); `l2img.DEFAULT_MODE = off`, and C-OFF's regression arm turns it to prim and must go RED.

## Controls and their regression arms (real run on R35 TRAIN windows; smoke copy `CONTROLS_SMOKE_s*.json`)

Smoke run of the controls (the real controls code on TRAIN windows and A0 batch 0; `CONTROLS_SMOKE_s0.json`, `_s1.json`, 11 min per chassis at 2 threads, two processes side by side):
| control | s0 | s1 | regression arm (must read RED) |
|---|---|---|---|
| C-OFF (16-window smoke vs nothing installed; TRAIN batch 0 vs LON-PLAN lam_0; A0 SCORE batch 0, seeds 0 and 1, vs banked plans, plans / cost1 / a1 / decode_resid / iters1 bit-equal) | green | green | default prim: RED |
| C-PRIM-ID (knots 11-30 = p exactly; fresh image residual <= 1e-6; prim23 also level 3) | green (max 2.1e-7 / 1.2e-7) | green | shadow not installed: RED; knots from decode: RED |
| C-L3ID (level 3 bit-equal under prim; generator state equal in prim, prim23, rs) | green | green | noise from planner generator: RED |
| C-REACH (gradient > 0, zero control exactly 0.0, iterates move) | green | green | p frozen: RED |
| C-L4-IMG (argv to built for all 12 argvs; mismatch, flag not named, pre-planted plan file refuse) | green | green | the three refusals |
| C-READBACK (P file re-planned bit-equal; wrapper end to end on 16 TRAIN windows) | green | green | read back at off: RED (not equal) |
| C-TRAIN-ISO | green | green | SCORE windows as TRAIN: RED |
| C-RS on TRAIN batch 0 (RS - A0, 0-2 s acc_mae, in-sample) | +0.458 | +0.474 | |
| C-IN | RED | RED | altered expected: RED |

C-IN read RED in the smoke for ONE reason: the card writes the hwm_plan.py sha256 as `5a072272...`, while the file and LON-PLAN's SOLVER.json read `5a0722719f7c...` (equal to each other; C-IN against the phase-1 record is green, 10 of 10). The card has a slip in the 8th hex digit; `amend4_stage.CODE_SHA_PREFIX` now compares 7 digits (after the smoke had started), and re-checks green offline. **MM: the freeze text carries the slip; the real launch's C-IN passes with the 7-digit compare.**

TRAIN batch 0 (64 windows, in-sample, 0-2 s acc_mae vs logged): s0 off 1.180 / prim 1.217 / prim23 1.131 / rs 1.638; s1 off 1.107 / prim 1.173 / prim23 1.139 / rs 1.581. Image residual max 2.1e-7.

## Tests
`test_amend4.py`: 34 passed (150 s with other jobs running). Shadow: off identity, default-prim mutation, prim in the image and in the box, prim23 level 3, level 3 and generators untouched, three mutations, REACH with zero control, RS donor and the no-shift refusal, restore after a raise, batch cap; wrapper: layout, stamps, SOLVER.json module sha256s, C-L4-IMG refusals, C-READBACK and its regression, off through the wrapper equals the code-of-record run; C-IN / C-TRAIN-ISO regressions; label rules: every outcome row (CEILING, NARROWS, NULL, HARM, SEEDS-DISAGREE, INVALID), one mutation per rule (10), band label, base solver, cell extraction, MD skeleton. `l2_prim_raw` test: P and P23 files carry it with shape (n, h2*c2, 2) (new, MM request); the wrapper itself refuses a P / P23 / RS folder whose plan files lack it (amend4_run.py run_arm). The launcher re-runs test_amend4, LON-PLAN's test_lonplan (49) and the code-of-record test_hwm_plan (50). Not re-run: test_plantest2_controls (11 min, needs a renamed tree); the code of record is untouched and sha-pinned by C-IN.

## `l2_prim_raw` and the TRAIN stage (MM request)
`l2_prim_raw` (n, 30, 2), raw units, level 2's chosen primitives, is banked by `l2img.plan_unit` in every prim / prim23 / RS plan file, on SCORE and TRAIN. `l3_prim_raw` is added under prim23. A TRAIN stage EXISTS: P, P23, RS on LON-PLAN's 564 TRAIN windows (141 clips), both chassis, seed 0, files at `amend4/train/{P,P23,RS}_{s0,s1}/` (same layout); report `amend4/readout/TRAIN_STAGE_{k}.json`; A0 there is LON-PLAN's `sel_{k}/lam_0`. SCORE files: `amend4/score/{P,P23,RS}_{s0,s1}/`.

## Fills where the card is silent (kept `off` = code of record)
* Label combination: per-chassis label first; differing labels read SEEDS-DISAGREE with the less favourable one effective. Order CEILING, NARROWS, NULL, HARM. HARM first in precedence (a worse guard blocks CEILING / NARROWS). A CEILING lost only to D < 0.10 reads NARROWS and `hold_collapsed` is printed.
* 2-6 s band: ADDS = separated-better beyond floor; HURTS = WORSE; else NEUTRAL; chassis combined as above.
* RS "separated-worse" on SCORE = CI lower bound > 0 (RS has one seed, no floor).
* IMG nearest-neighbour rows are not computed in the readout (they need the model and cloud); the readout reports the banked fresh image residuals, REACH, COST2 (batch 0 against the A0 batch-0 re-plan saved by C-OFF). The probe's NN script (`l2img_probe.py`) remains the instrument for the NN rows. Deviation, stated.
* TRAIN stage: headway and STRATEGIC cells are n/a on TRAIN windows (no lead or navigation join), stated in the file.
* A0 and A1 are read in place from LON-PLAN's folders (not copied).

## Time estimate (ESTIMATED from the smoke: 0.93-1.05 s per window, 2 threads, box shared)
13.3k planned windows (TRAIN 3,384; SCORE 9,940) = about 3.9 h one process; controls 11 min per chassis; tests about 6 min; 10 readbacks about 10 min; 18 contrasts about 0.5 h. About 5 h sequential, about 3 h with the chassis side by side (launcher does this when 11 GB of commit is free, else one at a time at 5.5 GB). Card: 4.7 h.

## Not done / flagged
The full launcher was not run (MM launches after review). The readout `score()` was exercised only on the real LON-PLAN contrast format (label extraction reads NARROWS on A1 - A0, floor 0.0073, as LON-PLAN) and the TRAIN report on stand-in plans; its contrast subprocess loop is untested end to end until the SCORE arms exist.
