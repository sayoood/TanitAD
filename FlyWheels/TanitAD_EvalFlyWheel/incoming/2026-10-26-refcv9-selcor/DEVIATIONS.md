# SELCOR deviations / operationalisations (written 2026-10-26T04:50:56+0100, BEFORE any fit, pick or bar)

1. "ORACLE FD far below BASE" (PREREG_SELCOR section 6) is read mechanically as: the ORACLE arm's paired FD cell vs BASE-os|stage is
   separated below (CI hi < 0). The ORACLE is descriptive and never a bar.
2. The DER arm reuses the configuration (variant, k, lambda) that CV chose for SELCOR-G; it is not cross-validated again.
3. v0 in phi is the trace's own v0_in (the model's input) on FIT and SCORE rows alike.
4. The z_tac PCA inside CV is re-fitted on each fold's training rows (no held-out row informs the basis).
5. CV tie-break: lower CV ADE (9 dp), then S before Z, then smaller k, then larger lambda.
