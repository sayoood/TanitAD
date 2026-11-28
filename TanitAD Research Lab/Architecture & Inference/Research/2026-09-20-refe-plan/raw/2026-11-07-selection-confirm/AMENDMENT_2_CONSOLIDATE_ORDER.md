# AMENDMENT 2 to PREREG_SEL_CONFIRM.md: consolidate puts the rows in token order

**Date:** 2026-11-08 ~18:20Z, Master Mind. **Written before** any verdict number existed: the chain stopped at `consolidate`, before `verdict`, and no harness call on the confirmation tokens has been made.

**What failed.** `selc_dump.py --consolidate` stopped twice at its own assertion, `tok == sorted(doc["tokens"])` (`markers/selc_full.FAILED`, 2026-11-08 19:13 local). It concatenated the chunks log by log, so the rows were in log order and the assertion could never hold on a full run. `selc_score.py` requires the same token order (`assert tok == sorted(doc["tokens"])`). Smoke did not exercise it.

**The data is complete (MEASURED, read-only):** 75 chunks, 5,877 rows, 5,877 unique tokens, 0 missing, 0 extra, 0 duplicates against `tokens_confirm.json`.

**The change.** One step in `consolidate`: reorder every field by `np.argsort(token)` after concatenation and before the assertion. The assertion is unchanged. No rule, token, bar, estimator, gate or endpoint changes, and the dumped values are not recomputed.

**Test.** `test_consolidate_order.py`, on synthetic chunks through the real `consolidate`: interleaved logs give token order with every field still on its own row; a token missing from the chunks, and a duplicated token, must both raise.

**Re-freeze.** `CODE_PIN.json` re-pinned (old copy `CODE_PIN.before_am2.json`). `selc_score.check_pins` accepts a re-pin through the single value held in `AMENDMENT_1_CODE_PIN.sha256`, while Amendment 1's text still matches its sha. That file now holds the new `CODE_PIN.json` sha; the previous value is in `AMENDMENT_1_CODE_PIN.before_am2.sha256`. `selc_score.py` itself is not edited. The preflight stage re-runs in full (pins, G-RULE, G-ANA replay).

**Other package affected.** `p6_read/selc6_analyze.PINS` pins `selc_dump.py` and this `CODE_PIN.json`, so it gains the new values under its own amendment, before any P6 value is read. P6 does not use `consolidate` from `selc_dump.py`.
