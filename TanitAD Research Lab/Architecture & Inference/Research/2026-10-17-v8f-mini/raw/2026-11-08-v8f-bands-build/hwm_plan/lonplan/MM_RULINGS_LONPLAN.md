# MM rulings on the LON-PLAN build (2026-11-16 ~14:55Z, before any TRAIN-selection or SCORE window is planned)

These rule on the frozen `AMEND_HWMPLAN_3.md` (sha256 7bbdc9fe…, commit 4dc4876c). No λ has been run or selected.

## The build agent's five interpretations (BUILD_REPORT_LONPLAN.md section 2): all CONFIRMED as written
1. The "running minimum" in the extension rules is the minimum rounded C over the admissible λ > 0 values already run, taken before the new value is added.
2. Extension downward applies when the chosen λ is the smallest positive value run. On the base grid that value is 0.3, so this matches the text.
3. D is measured against HOLD (the logged a0, held), the way `hold_distance` is called.
4. When no λ > 0 is admissible, the downward extension stops at the first admissible value.
5. The selection criterion C is the `band_metrics` expression. The test asserts it equals `band_metrics` on 30-knot data.

## Strategic-audit reader
`LONPLAN_READER_SRC` points at the repo's landed reader, `raw/2026-11-16-v8f-strategic-audit/reader/src/` (commit 7a571fc7). The build agent's `reader_src/` was a git export of it at 4dc4876c; it is not landed a second time.

## Before SCORE (not before the TRAIN selection)
* **Arm S and DIR-ACC.** `lonplan_shrink.py` (arm S, control C-SHRINK) and the SCORE-side DIR-ACC table are still to be built, each with its deliberate-regression arm, before any SCORE window is planned.
* **R35_s1 SCORE windows** (sha 8833e292…) are not on disk. Either fetch them from the programme's own private HF bank of the phase-1 tar, verified by sha256, or regenerate them with `run_hwm.py windows` and check the sha.
* **Code-of-record tests.** `test_hwm_plan.py` and `test_plantest2_controls.py` are to be re-run green on the dev box before SCORE.

## Order
1. TRAIN selection on dev-box CPU, after the 16:40Z memory window that another session asked for.
2. λ\*_0 and λ\*_1 written into the AMEND_HWMPLAN_3 Log.
3. The items above.
4. SCORE.
