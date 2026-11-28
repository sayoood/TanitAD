# MM rulings on the AMEND_7 build (2026-11-17 ~23:25Z, while the FIT read runs and before any FIT number exists)

Card `AMEND_HWMPLAN_7.md`, sha256 95c97b39, landed 55221e25. Build deviations: `build/BUILD_AMEND7.md` section 4.

1. **SCORE on PASS: the card governs.** The build runs FIT only, and `run_amend7.sh score` refuses unless `G7_FIT.json` reads R7-FIT-PASS. That guard stays. On R7-FIT-PASS the MM launches the SCORE read at once, as the card says, with no further decision. This keeps the card's order and adds a machine check.
2. **C-WM7 runs its roll check in float64. Accepted.** The card's 1e-6 bar is a numerical-identity check, and float32 rounding alone gives 1.4e-6 on the same computation. Float64 tests the identity, while float32 would test the dtype. The tolerance stays 1e-6.
3. **C-PLANT7 (i): "within 1e-6" is read as a final planted cost ≤ 1e-6. Accepted.** It is the only reading under which the planted arm can pass or fail on a number.
4. **Epoch snapshots in place of separate runs per epoch count. Accepted**, because they are bit-equal to separate runs (test in `test_amend7.py`) and change no number.
