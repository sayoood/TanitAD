# PREREG_SELCOR AMENDMENT 1 -- K-PRIOR's 1e-5 m tolerance was below float32 path resolution; replaced BEFORE any pick, rate or bar (Master Mind, 2026-10-26 ~06:33 CET)

Base PREREG_SELCOR sealed dc1f9bc6 (~04:47). Sealed after the scorer's first run was STOPPED by the Master Mind and BEFORE any SCORE-side
(eval139) pick, rate, cell or bar exists. Arms, features, label, fit, bars and committed next steps are UNCHANGED.

## What happened (disclosed in full)
* The first scorer run (`code/selcor_score.py --out result/selcor`, started 06:31) printed its controls -- K-FRESH-ROLL, K-JOIN, K-ELIG,
  K-ID s0/s1, K-EVALJOIN s0/s1, K-ADE all PASS; **K-PRIOR FAIL** -- then built the FIT labels and printed six FIT-side CV lines (variant S,
  lambda 1e-3 .. 100, held-out-fold staged ADE_S6 on FRESH rows only). It was stopped at that point; `result/selcor/` is EMPTY. No eval139
  pick, rate or bar was computed. The six CV lines are FIT-side quantities the registered selection rule reads anyway; disclosed, not used to
  change anything.
* Diagnostic (`diag/kprior_diag.py` -> `diag/kprior_diag.out`, FIT rows only): |trace `prior_path` - loader `HP`| at the S6 slots: median
  **0**, 99.2 % of 7,147 rows <= 1e-5 m, max **6.10e-5 m**, growing with the slot index (3.8e-6 at slot 1 -> 6.1e-5 at slot 7); arc lengths
  agree to 6.1e-5 m; a one-slot misalignment would read 35.6 m. Both arrays are float32 paths built by cumulative sums along 8 steps, with
  |coordinates| up to O(100) m where one float32 ULP is 7.6e-6 .. 1.5e-5 m. A 1e-5 m absolute tolerance is below that resolution: the third
  instance of this definitional error in the programme (MAPPATH Amendment 2).

## The change (definition only)
* **K-PRIOR (replaced):** elementwise over every FIT row and S6 slot, `|a - b| <= 8 * ulp32(max(|a|, |b|))` with `a` = trace `prior_path`,
  `b` = loader `HP` (8 ULP = the rounding bound for two independent float32 accumulations over 8 steps, 4 ULP each); the same rule on the
  S6 arc lengths. **Mutation arm (must FAIL):** the identical check with the trace slots shifted by one (`S6_SLOTS - 1`).
* Descriptive, printed beside: the absolute max, the per-slot max, and the share of rows bit-equal and within 1e-5 m.
* The scorer gains exactly this check (disclosed code diff `code/selcor_score.AMEND1.DIFF.txt`); `result/selcor/` is re-created by the
  re-run; nothing from the stopped run is carried over.
