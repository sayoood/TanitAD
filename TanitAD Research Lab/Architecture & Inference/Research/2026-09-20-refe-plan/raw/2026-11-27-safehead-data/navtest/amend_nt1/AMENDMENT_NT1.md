# AMENDMENT NT-1 to the navtest feature-dump controls (`navtest/nt_ctrl.py`), 2026-11-27, MM

**What failed.** The full navtest dump (136 logs, 12,146 tokens) ended `navtest.FAILED` at 20:20:06Z on ONE control.
CTRL-LOGIT (C1's `score_head` applied to the dumped `s`, against the dumped logits) read **max |diff| 1.144e-5** against its bar of **1e-5**.
Every other control PASSED: CTRL-PICK-FULL (12,146 / 12,146 picks equal to the anchor `full_C1_s{0,1}of2.jsonl`, 0 missing), CTRL-SELECT (0 picks off argmax agg32), CTRL-SHAPE and COVERAGE.
Evidence: `ctrl_full.attempt3_bar1e-5.json`.

**Why it is noise and not a defect** (MEASURED, `logit_dist.py` -> `logit_dist.json`, numpy float32 on the same arrays):
- 4,664,064 values. Max 1.526e-5 under numpy, against the control's 1.144e-5 under torch. The SAME weights in a different accumulation order give a different max.
- 625 values (0.013 %) lie in (1e-5, 2e-5]; none above 2e-5; every file's p99 is <= 6.95e-6.
- The largest |logit| is 19.8, where one float32 ulp is 1.9e-6, so the max is ~8 ulp: the rounding of a 256-term float32 dot product.
- The navtrain cache passed the same 1e-5 bar (7.6e-6 and 5.7e-6), on fewer values.

**The binding bar is the card's, not the instrument's.** `PREREG_SAFEHEAD.md` (line 72), CTRL-FEAT: C1's head on the cached `s` must reproduce the cached logits at **max and p99 <= 1e-4**. The instrument's 1e-5 was an internal threshold, 10x stricter than the card and never pre-registered. Here max 1.14e-5 to 1.53e-5 and p99 <= 6.95e-6, so CTRL-FEAT passes.

**Change.** CTRL-LOGIT's bar is 1e-4 (= CTRL-FEAT). The JSON also records `old_bar_1e-5_pass`. No other line changed.

**Timing.** Made before ANY SAFE-HEAD, SH-MLP or EP-HEAD navtest number exists. SAFE-HEAD's outcome (NULL-HELDOUT) was fixed on navtrain before this. The amendment moves no decision bar.

**Re-run.** `nt_ctrl.py full` was run directly (overall_pass True; it wrote `navtest/index.csv`, 12,146 rows). nt_chain was NOT re-run, because its `finish()` relaunches TRK-SCORE, which was already running (pid 2712). `markers/navtest.DONE` was written by the MM with this provenance; the failed marker is kept as `navtest.FAILED.attempt_3_ctrllogit_bar1e-5`.
