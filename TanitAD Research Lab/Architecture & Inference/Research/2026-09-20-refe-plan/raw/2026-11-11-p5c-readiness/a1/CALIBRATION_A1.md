# A1 calibration of R-TOL-1/2/3 (2026-11-12)

Evidence class: MEASURED, dev box, CPU only. Inputs: the 33 real P5-R EVAL rows of `fixtures/` (rank 0). Instrument: `calibrate_A1.py` (one process per arm, BLAS env set before numpy imports; thresholds imported from `p5c_table_gates.py`, none changed). Raw: `calibration_A1.json`. Rows scored against the lam* = 3.0 EVAL labels (`fixtures/eval_control_33.jsonl`).

## Per-row max abs label difference (m / rad), 33 rows
| arm | bit-identical rows | median | p90 | p99 | max |
|---|---|---|---|---|---|
| control: 1 thread, dev-box Tc20s warm start | 33 | 0 | 0 | 0 | 0 |
| **noise (a)** BLAS 4 threads, same warm start | 0 | 6.0e-6 | 1.7e-4 | 3.4e-4 | 4.0e-4 |
| **noise (b)** warm start from a 1-thread Tc20s, 1 thread | 10 | 7.7e-7 | 4.2e-5 | 2.9e-4 | 3.4e-4 |
| **regression** lam = 1.0 | 0 | 1.1e-1 | 3.4e-1 | 1.26 | 1.59 |
| **regression** lam = 10.0 | 0 | 1.0e-1 | 3.9e-1 | 0.85 | 0.95 |

The max of noise (a) and (b) equal the 4.0e-4 and 3.4e-4 the previous agent measured.

## Criteria (R-TOL-3 row-matched: reference = dev-box ratios on the same 33 rows, MM ruling on caveat 1)
| arm | R-TOL-1 (sim8 identical >= 99.5 %) | R-TOL-2 (median <= 1e-3, p99 <= 5e-3) | R-TOL-3 (ratios +-0.01, sim8 rate +-0.25 pp) |
|---|---|---|---|
| control | 33/33 pass | pass | pass (acc 0, hr 0) |
| noise (a) | 33/33 pass | **pass** | **pass** (acc -1.2e-7, hr -3.5e-7, sim8 0.0 vs 0.0) |
| noise (b) | 33/33 pass | **pass** | **pass** (acc +1.6e-7, hr 0) |
| reg lam 1.0 | 33/33 pass | **FAIL** | **FAIL** (acc +0.0599, hr +0.7127) |
| reg lam 10.0 | 33/33 pass | **FAIL** | **FAIL** (acc -0.0585, hr -0.4413) |

Control and both noise arms pass all three; both regression arms still fail R-TOL-2 (medians 1.1e-1 and 1.0e-1, about 100x over the bound) and also fail R-TOL-3. No threshold was changed.

## Notes
1. The reference is row-matched: the reference ratios come from the control records themselves on the common rows (the fixture is slimmed to label + sim8, so here the per-row learnability was recomputed with the gate's `learn()`; with the real `eval_r0.jsonl` the records' own `learn` / `raw.learn` are used and equal the recomputed values exactly, max diff 0, tested). `verdict.json` (hr 1.5638 over 1,200 rows vs 1.9946 on these 33) is recorded as context only and not gated.
2. **sim8 is 0 failures in all five arms on these 33 rows**, so R-TOL-1 and the sim8 half of R-TOL-3 are not exercised by the calibration; the gate tests exercise them synthetically.
3. 33 rows: p99 is an interpolation between the top two rows. The noise max (4.0e-4) is 12.5x under the p99 bound 5e-3.
