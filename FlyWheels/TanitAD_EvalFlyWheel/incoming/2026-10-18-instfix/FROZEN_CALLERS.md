# FROZEN_CALLERS -- invocations that now fail loudly (INSTFIX / HJ-UNITS-3, 2026-10-18)

Scan: `code/frozen_scan.py` over mirror tip `5448a05` (read with `git show`, never E:), raw rows in `raw/frozen_scan_*.json`.
It is a LOWER BOUND: a command built from an unresolvable variable is not counted. None of these files was edited -- they are the record of what ran.
**To reproduce a banked run, add the unit it effectively ran under (the OLD default):** `--action-units kappa` for `actdiv_anchored.py`, `--action-units steer` for `refcv3_arm.py` / `openloop_suite.py`.

## 1. `actdiv_anchored.py --family refav1` (old default `kappa`) -- 4 sites in 3 files, all frozen
`--family v7` invocations are not asked for a unit and are unaffected (`run_v7_cpu.cmd`). `run_units.ps1` passes the flag and is unaffected.

| file | line |
|---|---|
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-anchored-actdiv-and-transition-probe/run_refav1_cpu.cmd` | 10, 12 |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-anchored-actdiv-refav1/run_actdiv_anchored.ps1` | 13 |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-anchored-actdiv-refav1/SPEC.md` | 197 (a prose command) |

LIVE callers: **none** (0 rows in `stack/`, `taniteval/`, `tools/`; the only programmatic caller is `stack/tests/test_actdiv_anchored.py`, which calls `--family v7`).
The banked `raw/actdiv_anchored_refav1_step1000*.json` were produced under `kappa`; `run_units.ps1` produced both readings explicitly.

## 2. `refcv3_arm.py` rollouts (old default `steer`) -- 12 rows: 2 LIVE (fixed here) + 10 frozen
LIVE and FIXED in this package (each now passes `--action-units steer`, byte-identical to the old default):
`taniteval/taniteval/bench/internal_t1.py:47` (`build_cmd`, rollout branch only) and `taniteval/tools/run_hierarchy_panel.py:110` (`roll`, the rollout branch only; the `--analyze-only` recover branch is not asked).

| file | line | note |
|---|---|---|
| `Project Steering/GOALS_AND_CLAIMS.md` | 6347 | a register's prose command; the file mentions the flag elsewhere (UNVERIFIED) |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units/RESULT.md` | 83 | a prose command; flag mentioned elsewhere in the file (UNVERIFIED) |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-wpd-bev-aux/code/b17b_planner_seed_floor.py` | 48 | argv built from `TOOL` |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-s1-collision-gate/code/gap_runbook.sh` | 81 | |
| `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-arm/RESULT.md` | 235 | a prose command |
| `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/scripts/arm.sh` | 9 | |
| `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-21-e9-rate/code/e9_rate.sh` | 24 | |
| `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-21-e9-rate/code/e9_third.sh` | 13 | |
| `TanitAD Research Lab/Data Engineering/Research/2026-09-03-steer-curvature-interface/RESULT.md` | 341, 539 | prose commands; flag mentioned elsewhere (UNVERIFIED) |

## 3. `openloop_suite.py --ckpt` rollouts (old default `steer`) -- 4 rows: 2 LIVE (fixed here) + 2 frozen
LIVE and FIXED: `taniteval/tools/OPENLOOP_SUITE.md:150` (the elided `--stack` example) and the `--ckpt` usage example in `taniteval/tools/openloop_suite.py`'s module docstring (line 79).

| file | line | note |
|---|---|---|
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/RESULT.md` | 487 | a prose command |
| `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/raw/patch_eval_analyzeonly_arg.py` | 23 | a command string inside a patch script (UNVERIFIED that it is ever executed as a rollout) |

## 4. Not callers (looked at, deliberately not counted)
* `refcv3_arm` used as a MODULE (`load_model`, `build_corpus`, ...) by `refcv6_roll.py`, `r7_roll.py`, `rl_refcv3_min.py`, `ddv2_rl_refcv5.py`, `closedloop_drive.py`, `bench/navsim/bridge.py`, `s1_pass.py`, `render_refcv3_video.py`: none reaches `main` or the `--action-units` flag. `refcv3_arm.run_dump` (the one consumer of the unit) already refused `None` from a hand-built Namespace (`run_dump`: `units not in ("kappa", "steer")` -> `SystemExit`; tip line 2036, edited tree 2038), which is how `openloop_suite._arm_args` calls it.
* `render_refcv3_video.py --action-units` (default `steer`): a DEAD flag, proved dead by AST in `test_l4_...` -- see RESULT s5.
