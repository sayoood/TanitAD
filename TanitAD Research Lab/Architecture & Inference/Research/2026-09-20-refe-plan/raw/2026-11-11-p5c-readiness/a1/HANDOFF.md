# P5-C A1 (dual labels + P5-R gates) HANDOFF, 2026-11-12 (rev 2: MM rulings R-A..R-D)

HOLD: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/a1/`. Nothing run on a pod; nothing staged in git; nothing frozen (no `.sha256`). The MM lands and freezes.

## What changed in rev 2 (the MM's four rulings, implemented exactly)
- **R-D** runner: a line whose only failing token is `TB-6`, with that label's `gates.json` showing TB-6 rank 0 PASS / rank 1 FAIL / `rank_policy.decision == "rank0"`, is class `rank1only` = passing with policy rank0 (Tc20s line and P5-R line); `build_table` runs for it. Rank-0 TB-6 also failing stays the trigger/STOP path. Tested by extracting the 6b block and the two dispatch lines from the runner and running them under Git bash with fake gate lines (16 checks).
- **R-C** `p5c_dual_labels.py`: threadpoolctl optional; without it all of OPENBLAS/MKL/OMP_NUM_THREADS must be "1" or the labeller exits 2 with `ZZDUAL_REFUSE thread_pin`; DONE and repair records carry `thread_pin` + the env. Runner: import probe no longer needs threadpoolctl, exports all three vars and asserts them.
- **R-A** `p5c_table_gates.py --tc20s-tol` (CLI default 1e-9; 0 = frozen behaviour; library default 0.0). Tc20s TB-0/TB-1 and the Tc20s warm start inside the P5-R gate use it. sim8 compared where both sides carry `tc20s_sim8` (the Tc20s records carry none, so 0 rows on the pod).
- **R-B** R3's `--p5r-tol` and the 1e-6 default are gone. R-TOL-1/2/3 per rank on the EVAL control rows (R-TOL-3's reference is ROW-MATCHED (dev-box ratios of the same common rows from the eval records, MM ruling on caveat 1); `--p5r-verdict p5r/raw/verdict.json` is recorded context only, never gated; shipped + pinned, sha256 `7ae694a3...`).
- **Calibration** `calibrate_A1.py` -> `calibration_A1.json`, `CALIBRATION_A1.md`: control + noise arms pass R-TOL-1/2/3 (row-matched), regression arms (lam 1.0, 10.0) fail R-TOL-2 and R-TOL-3. sim8 is 0/33 everywhere, so R-TOL-1 is not exercised by the calibration.
- `PREREG_P5C_A1_DRAFT.md` updated: section 2 = the rulings, new section U (R-A is unconditional and pre-data), section 3 = calibration table, blob ids refreshed.

## Tests
`python test_p5c_table_gates.py --p5r-only` (interpreter `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 P5C_TMP=<scratch>`): `TEST_RUN_A1.log`, **163/163** (P5-R + R-A/R-B arms 138, thread pin 9, selection block 16). `TEST_RUN_P5R.log` is the superseded rev-1 log (87/87 with R3's 1e-6). The 94 Tc20s legacy checks still cannot run (input deleted); legacy gate_tb0/gate_tb1/run_all are checked by behaviour at tol 0 against the landed module instead of by source identity (their source gained the `tol` parameter). The legacy section of the test file was not edited and was not run.
Note: `driverl-eval` has NO threadpoolctl; calibration ran with BLAS env vars only.

## Deliverables
| file | state |
|---|---|
| `p5c_dual_labels.py`, `p5c_table_gates.py`, `p5c_pod_label.sh`, `test_p5c_table_gates.py` | edited, tests green, `bash -n` OK; blob ids in the draft |
| `calibrate_A1.py`, `calibration_A1.json`, `CALIBRATION_A1.md`, `TEST_RUN_A1.log` | new |
| `PREREG_P5C_A1_DRAFT.md`, `POD_SHIP_LIST.txt` (verdict.json added, runner pins for dual and gates updated), `HANDOFF.md` | updated |
| `fixtures/`, `TB1_control_1018.json`, `TEST_RUN_P5R.log` | unchanged |

## Open items for the MM
- Runner pins PIN_DUAL / PIN_GATES / PIN_VERDICT are the current sha256 values; any later edit to those files needs the pin refreshed.
- `p5r_eval/verdict.json` is still shipped and pinned (context value); R-A's sim8 half is vacuous for Tc20s (no sim8 in the records).
- The runner has never run; first pod run may need a path fix. P5-R on rank-1 rows and the pod itself are untestable here.
- Earlier measured facts (rev 1): per-row time 0.60 s dual, 25.1 CPU-h for 150,603 rows; BLAS 4 threads moves P5-R labels by 4.0e-4 max (re-measured in the calibration).
