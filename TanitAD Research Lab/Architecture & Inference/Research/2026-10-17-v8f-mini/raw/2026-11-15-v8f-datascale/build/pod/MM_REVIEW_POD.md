# MM review of the DSCALE10K pod prep (2026-11-16 ~21:40Z)

**Patches to the study tools: ACCEPTED.** The gate token binds the patched sha256s. Diffs: `study/PATCH_merge.diff`, `study/PATCH_check.diff`.

* **Merge.**
  * The real releases carry 10 more per-clip columns than the synthetic fixtures. They are carried by sid.
  * The two releases differ in key ORDER only, so the comparison is a set comparison; the union uses the corpus order.
  * The lead files carry a 0-d `meta` string. The union writes a deterministic merged `meta`, and C-MERGE checks it.
  * The merge reproduced byte-identically on a second run.
* **Checker.** `join_eq_problems` exempts only the 60 holdout clips. The skip set comes from the same holdout definition that C-HOLD pins to M2b's `excluded_sha256`, so it cannot hide any other missing clip.

**Review item for the gate (before R_s0 is minted).** The R derangement moves the label rows between 10k clips. The per-clip columns (`has_lc`, `n_turns`, `n_announced`, `has_lead`, ...) are not yet stated as moved or kept. State which, and why.
* A mismatch can only make R worse.
* R's role is to show that the new clips' labels reach the model.
* So a degradation caused by a flag/label mismatch must be named in the read, not counted as label reach.

**Gate items still open (from the pod handoff):** G-CLOCK, G-EVAL, G-CKPT, the stage-B `dagger` reach term, C-INIT, C-CACHE, C-SESS, and the evidence JSON for `mint`. They are done after the block-0 pilot and before block 1 is minted.

**Base G-SUITE binding.** The tree is `cdbff971`, verified by blob hash, and its manifest equals `/root/strat1/base`. Its junit `/root/strat1/junit/base/` (14,851 tests: 431 failed, 206 errors, 1,691 skipped, all MEASURED) is the baseline itself, so the code of record has 0 regressions by identity. Its failures are the pod environment's, and they are the same for every arm.
