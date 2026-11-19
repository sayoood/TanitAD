# HANDOFF_GAPS: tests for the STILL-GREEN mutations (2026-11-19)

## Gap tests
Method as in HANDOFF.md (driver `common/pod_mut_driver.py`, one exact replacement per run on the pod copy, CPU, one process at a time, 36 to 37 GB cgroup headroom at each start). New test files only, no lever code touched.
Files: `distrule/test_dist_rule_gaps.py` (6 tests, run from `tree_*/src/`), `leadsup/test_lead_sup_gaps.py` (2 tests, run from `tree_*/stack/tests/`); specs `spec_gaps.json` via `make_spec_gaps.py`; raw `pod_out_gaps.json`; records `MUTATIONS_DISTRULE_GAPS.json`, `MUTATIONS_LEADSUP_GAPS.json`; `PINS_*_GAPS.txt`.
Result: 8 of 8 RED, each through its own new test (`expected_all_red` true); control (unmutated tree, new files only) green for both levers (6 and 2 tests).
| id | new test | idea (expected value by hand, in the docstring) |
|---|---|---|
| DISTRULE w1 | test_w1 | RulePlanner on fakes: k=3 on the x axis, a_l 3.0, a_m 2.5, g0 = 10.5 (k=0 row: 10.0) |
| DISTRULE w5 | test_w5 | head probabilities 0.3, 0.49, 0.5, 0.9, 0.0 give present F, F, T, T, F |
| DISTRULE w8 | test_w8 | head gap NaN, closing NaN, logged speed missing: not present, src_bad logged |
| DISTRULE w9 | test_w9 | fake pinned tool with mode `report`: every non-off rule refused, off hands over |
| DISTRULE q5 | test_q5 | fit_arm on bins with label quantiles 6, 4, 15: pooled ys = [5, 5, 15] |
| DISTRULE q8 | test_q8 | 1850 rows over-read 1 m, 150 rows 10 m: p95 = 10.0 (a p90 read gives 1.0), tolerance 1e-6 |
| LEADSUP m2 | test_m2 | frame_ok [T,T,T,F,T], hist 3: one counted window, term = ln 2 (any: 1.9022) |
| LEADSUP m5 | test_m5 | 2 gap labels of 4 end frames: gap part 0.025 by hand (divided by 4: 0.0125); closing part 0.1 |

## No test wanted (equivalent or near-equivalent mutants)
* DISTRULE q2 (map not capped by min(g, Q)): `check_map` enforces ys <= xs (within 1e-12), so Q(g) <= g below the first knot (ys0*g/xs0), between knots (interpolation of ys against interpolation of xs = g) and above the last (g - (xs-ys)). On a map that passes `load_map` the min is the identity; it differs only for a map that fails check_map, or by at most the 1e-12 tolerance the check allows. No valid input separates it.
* DISTRULE d8 (unchanged steps not copied back): a step is unchanged when the new arc equals the knot arc exactly, and the sampled path then lands on the knot vertex. Measured with `distrule/measure_d8.py`: 2400 random 30-step plans per dtype (float32 and float64, about 1,400 of them changed by the rule), orig vs mutant: largest absolute difference 0.0 m in both dtypes. The copy is a guard against interpolation rounding that did not show up in this measurement.

## Temp files left (no deletes)
`distrule/make_spec_gaps.py`, `spec_gaps.json`, `pod_out_gaps.json`, `measure_d8.py`; `leadsup/make_spec_gaps.py`, `spec_gaps.json`, `pod_out_gaps.json`. Pod: `distrule/` and `leadsup/` gain `spec_gaps.json`, `out_gaps.json`, `run_gaps.log`, and the new test file in `tree_work` and `tree_pristine` of each. Nothing committed or staged.
