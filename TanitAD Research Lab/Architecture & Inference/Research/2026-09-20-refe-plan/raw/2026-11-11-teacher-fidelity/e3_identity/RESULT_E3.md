# E3 result: the released DriveRL checkpoint reproduces the paper's Test14-random NR score (outcome A)

2026-11-12, Master Mind, Teacher Fidelity stream. CPU only, no GPU, no navtest.

## Setup

- **Design:** the outcomes were committed in advance in `PK/raw/2026-11-11-teacher-review/TEACHER_REVIEW.md` §6 E3, landed before any E3 number:
  - **A:** both runs within ~1 point of 94.50;
  - **B:** both runs ≥ 2 points below;
  - otherwise "in between", with no new bar.
- **What ran:** `checkpoint_2400.pt` (the DriveRL release) on nuPlan **Test14-random, closed-loop, non-reactive agents** (261 scenarios), with the release's own runner (`nuplan-devkit` from the DriveRL release).
  - Same overrides as REVIEW_4 §5, except `device=cpu` and no simulation-log callback.
  - Runner script `run_e3.sh`. The paper's value is **94.50** (DriveZero Table A10).
  - Two identical runs, A3 and B4.
- **Evidence class:** MEASURED.
- **Tier:** nuPlan CLS-NR, closed loop in the ego's own actions against log-replayed agents. The planner reads privileged object states, with no re-perception. This is **not NAVSIM** and is not navtest.
- **Estimator:** none. Replicate spread is the pre-registered noise unit. The 1-point band is the pre-registered criterion.

## Result

| run | aggregation | score ×100 | scenarios | runner failures | vs 94.50 |
|---|---|---|---|---|---|
| **A3** | post-hoc (`aggregate_posthoc.py`) | **94.848** | 261/261 | 0 | **+0.348** |
| **B4** | native nuPlan `MetricAggregatorCallback` | **94.848** | 261/261 | 0 | **+0.348** |

**Outcome A, as pre-registered.** Both runs are within 1 point of 94.50. Spread |A3 − B4| = **0.000**.

### Sub-metrics (identical in both runs)

| metric | value |
|---|---|
| drivable area | 0.9923 |
| driving direction | 0.9981 |
| no at-fault collision | 0.9808 |
| making progress | 1.0000 |
| progress along expert route | 0.9542 |
| time-to-collision within bound | 0.9655 |
| speed-limit compliance | 0.9835 |
| comfort | 0.9847 |

### Instrument check

- **The post-hoc score is the native score.** On run B4, `aggregate_posthoc.py` was run on the same metric files as the native aggregator. Max abs diff is **0.0** over all 276 rows × 23 columns, with the same NaN pattern and equal text columns. This validates A3's post-hoc number; A3 had no native aggregate because its output path lacked the challenge name the callback filters on (`metric_aggregator_callback.py:46`).
- **The two runs are bit-identical** on all 275 per-scenario rows (n differing = 0). On CPU the planner is deterministic.

## What it means

- **Checkpoint identity moves from SUPPORTED to near-confirmed**, which is TEACHER_REVIEW's wording for outcome A.
- The earlier evidence was update 2400 and 5,702,413 params matching, plus Test14-hard NR 89.755 vs the paper's 89.97 (one draw). Test14-random NR now matches too, at +0.35.
- ⇒ The ~6-point navtest gap between our teacher protocol and the paper's DriveRL row (95.8) is **protocol and input, not a different teacher**. The paper's navtest teacher plans come from a conversion the release does not ship; the release has no NAVSIM path.
- The tracker-aware target (Tc20s) is the protocol lever already localised on navtrain held-out:
  - TF-1: 94.36, DAC 99.02;
  - TF-2b fresh logs: 94.80, DAC 100.0.
- **N1** is the pre-registered navtest confirmation. It was launched 2026-11-12T07:20:28Z, right after this result.

## Caveats

1. **Spread 0.0 is not an interval.** The CPU runs are deterministic, so the replicate measures no inference variance. It says nothing about scenario sampling. No CI is quoted on +0.348, and the decision rests on the pre-registered 1-point band only.
2. **Device:** CPU here; the paper's device is not stated in the text layer we read. The +0.35 above the paper may be device numerics or devkit version, and is **not interpreted**.
3. **The 9-channel critic and the 16-D actor input** (release vs paper's 8 and 15-D) stay INHERITED from the teardown. Outcome A makes them unlikely to matter for behaviour but does not explain them.
4. **Val14 NR** (paper 95.16) was optional and was **not run**. Outcome A is decided without it.

## Four metric families

These are nuPlan sub-metrics on the closed-loop execution:

| family | reading |
|---|---|
| LONGITUDINAL | progress 0.9542, speed limit 0.9835, TTC 0.9655 |
| LATERAL | drivable area 0.9923, driving direction 0.9981 |
| TACTICAL | not computable: Test14-random carries no manoeuvre label (n = 261) |
| STRATEGIC | **PAUSED**, PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25) |

## Run record

| run | outcome | log |
|---|---|---|
| runA, runB | killed by WinError 1455 (commit memory, two concurrent runs) | `failed_runs/runA_1455.log`, `runB_1455.log` |
| runA2 | killed by a console window-CLOSE event (plain WMI launch) | `failed_runs/runA2_windowclose.log` |
| **runA3** | **complete**, launched with CreateFlags 0x08000200 | `runA3_bank/` |
| runB3 | killed at scenario 5 by the dev-box reboot at 00:33Z | `failed_runs/runB3_reboot.log` |
| **runB4** | **complete**; its output path contains the challenge name, so it aggregated natively | `runB4_bank/` |

## Artifacts (`e3_identity/`)

- **Scores:** `e3_score.json` (both runs, outcome), `e3_runA3_score.json`, `e3_runB4_native_score.json`.
- **Scripts:** `score_e3.py`, `aggregate_posthoc.py`, `run_e3.sh`, `go_b4.cmd`, `go_seq3.cmd`, `t10/` (10-scenario timing run).
- **Run banks:** `runA3_bank/` and `runB4_bank/`, each with:
  - `metrics/` (16 per-metric parquets);
  - `aggregator_metric/`, `runner_report.parquet`;
  - `log.txt`, `run.log`, `code/` (hydra config snapshot), `DONE.txt`.
- **Bank verification:** 37 data files, sha256 multiset equal to the source before it was deleted.
- `failed_runs/`, `HANDOFF.md`.
