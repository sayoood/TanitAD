# QUESTIONS: where the two preregs admit two readings, and the resolution the S1c auto-reader applies

Written 2026-10-16 ~10:10 Berlin (08:10Z), **BEFORE any S1c number exists**. Evidence that none does (MEASURED, this session): the driver's read-only live listing of `Sayood/tanitad-refe-s1aux` at 2026-10-16T08:08:15Z read 2359 files, arms present `H1-s0/s1/s2, H2-s0/s1/s2, N-s0/s1/s2`, and **`arms/H1u10k-s0/` = 0 files** (log: `dryrun_stdout2.log`). The only numbers the reader has produced are on S1's banked `H1-s0` artifacts (the dry run).

Nothing below edits a prereg. Each item is a *resolution proposal*: the Master Mind can veto one by telling the reader to change the corresponding line of `s1c_rules.py` / `s1c_autoread.py`; the unit tests pin the current resolution (`test_s1c_autoread.py`, grid + mutation checks). No item blocks the launch.

Sources quoted: `eval/PREREG_S1C.md` (S1C) and `eval/PREREG_AUXVETO.md` (AV, incl. amendment A1).

## Q1. The thresholds: rounded F, strict inequalities, K2 on the point estimate
* S1C section 3: *"F = the largest pairwise |difference| between S1's three H1 seeds ... = 0.035 (C3), 0.023 (NAT)"*; section 4: *"'Separated' = the paired CI excludes 0 **and** |Delta| > F"*; section 3: *"K2 is decided on the point estimate against 0.80 ... the interval is context."*
* **Resolution.** F is the prereg's stated pair `0.035` / `0.023` (not the unrounded 0.0347 / 0.0229; the rounded values are the stricter ones and are what the text commits). "Excludes 0" is strict (`lo > 0` or `hi < 0`); `|Delta| > F` is strict. K2 clears on `P1 >= 0.80` (the point estimate, `>=`). Boundary cases are pinned in the tests.

## Q2. Validity first; what INVALID covers; what INVALID writes
* S1C section 4: *"Decided by the arm's reads in this order"*; INVALID = *"C1 fails, or the chain's run-check refused | no S1c read"*; O3's rule ends *"and C1 holds"*; section 2 C1: *"A mismatch means ... the arm is **not paired with S1** and no budget claim is made (the init floor and absolute K2 may still be reported)."*
* **Resolution.**
  1. Validity is evaluated **first** (O3's own clause "and C1 holds" presupposes it): an INVALID read is INVALID whatever the Deltas say.
  2. INVALID = C1 fails (`step0` held-out `aux_eval` IoU outside 0.1586791569036157 +-1e-3) **or** `run_check.json` is not ok **or** the arm's evidence contradicts its label. *The third clause is an EXTENSION of the prereg's two:* a pushed arm whose PASS-token profile is not `refe_aux_s1c`, whose argv seed/updates/batch/accum/head flags are not the registered ones, whose `sft.jsonl` start/done/tags/lr-0-at-the-end/data audit/head config differ, is not "the arm" in substance. The checks are `check_evidence` (each registered fact has a deliberate-regression test). A mismatch between the token's argv and `ARGV_DIFF.json` (path drift) is only a WARNING; the substance flags are enforced independently.
  3. An INVALID read still writes `RESULT.md` (the INVALID row and its lever, quoted verbatim; the K2 / Delta numbers labelled "INFORMATIONAL ONLY", honouring "absolute K2 may still be reported") and `RESULT.json`, but the **marker is `S1CREAD_FAILED`** (reason names the failed check): *no S1c read exists* is a failure to read, not a recorded outcome.

## Q3. O1 needs BOTH populations; a K2 disagreement
* S1C section 4: O1 = *"P1_C3 >= 0.80 **and** P1_NAT >= 0.80"*; last line: *"If P1_C3 and P1_NAT disagree on K2, **the C3 read governs** ... and both numbers are reported with their population named."*
* The two sentences collide in one case: P1_C3 >= 0.80 > P1_NAT (or the reverse). The O1 rule says not O1; the last line says the C3 read governs K2.
* **Resolution.** The **row** follows the table: O1 only when both are >= 0.80, otherwise O2/O3/O4 by the Deltas. The **K2 block** of `RESULT.md` states the C3-governed verdict, names both populations' numbers, and sets `populations_disagree: true` with a visible flag. Nothing is averaged or dropped.

## Q4. O3 and O4 can both match
* S1C section 4: O3 = *"Delta_C3 or Delta_NAT not separated"*; O4 = *"Delta_C3 or Delta_NAT separated and < 0"*. One population adverse and the other flat satisfies both.
* **Resolution.** Table order: **O3** is the recorded row; the overlap is **flagged** (`overlap: ["O4"]`) and O4's committed next lever (`H1u2k-s0`, defect hunt first) is quoted verbatim beside O3's, so nothing is dropped. The marker carries `s1c_row = O3`; the MM reads the flag.

## Q5. The natural population's interval: weights fixed (primary) vs re-fitted per resample (sensitivity)
* S1C section 3 names the *"SHIFT_LOCATE weights: failure stratum 4.36 %, navtest's own city mix, ESS reported"* and the *"paired log-cluster 95 % CI"*; it does not say whether the weights are re-fitted inside each bootstrap resample.
* **Resolution.** **PRIMARY = weights fixed at the SHIFT_LOCATE value** (the form the 0.561 / 0.577 / 0.554 identity is reproduced in, to 0.0 abs diff; the CI therefore conditions on the frame weights). **SENSITIVITY (non-gating) = IPF re-fitted inside every resample**; both are reported, and if the sensitivity interval would change the *separated* class, `RESULT.md` prints "**CHANGES**". The decision uses the primary only. (MEASURED on S1's scratch preds: H1-s0 natural CI fixed [0.5217, 0.5988] vs re-raked [0.5451, 0.5790]; contrast H1-s1 minus H1-s0 fixed [0.00967, 0.02242] vs re-raked [0.01107, 0.02162].)

## Q6. The C3 Delta is computed twice; they must agree
* S1C section 3 names `head_fit_pairs["H1u10k-s0-minus-H1-s0"]` (the harness). The natural reader computes the C3 contrast too (same frames, same resample scheme).
* **Resolution.** The decision uses `nat_p1`'s C3 and natural contrasts (one instrument for both populations), and the driver **refuses** (FAILED, named) if `nat_p1`'s C3 P1 differs from the harness's by more than 1e-12 or its C3 contrast differs from `head_fit_pairs[...]["iou_0-40"]` (est or either CI end) by more than 1e-12. Both checks have mutation tests.

## Q7. O2's "rank by the measured remaining gap"
* S1C section 4 O2: *"rank by the measured remaining gap `0.80 - P1` on the worse population. No extrapolation beyond 2x the fitted range."*
* **Resolution.** `RESULT.md` prints `0.80 - P1` for C3 and NAT and names the worse population. It does not extrapolate and does not pick between candidates (a)/(b); that ranking is the MM's (S1C section 10 already ranks (a) first).

## Q8. The GPU: `auto` = GPU under the shared lock, then CPU; the identity tolerance
* The brief asked for the forward "on the dev-box GPU under the shared lock `devbox_gpu.lock` (wait, never break)". S1's banked reads were produced on **CPU** (bit-exact), and the lock is currently held by `refcv8-battery-step5000` (pid 64306), so the GPU may not be available for hours.
* **Resolution.** `--forward-device auto` (default for a real launch): try CUDA first (the forward tool itself takes/waits for the lock and exits **5** when busy; this driver waits with 60 -> 300 s backoff up to `--gpu-wait-min 30`, **never touches the lock file**), then fall back to **CPU** (the S1 precedent). `--forward-device cuda` waits the same and then FAILS (no CPU fallback); `cpu` skips the GPU. *This is a deviation from "GPU only": a stalled GPU must not stall the read for days.*
* **Comparator identity.** S1's `H1-s0`, read in the SAME forward call as the S1c head, must reproduce S1's banked navtest block (286 numeric leaves) and the SHIFT_LOCATE natural read: tolerance **1e-9 on CPU** (MEASURED 0.0 in the dry run), **1e-4 on CUDA**. *The CUDA figure is UNVERIFIED (no GPU run was possible: the lock was held); 1e-4 is 350x below the smallest F.* A CUDA miss falls back to CPU (`auto`) or fails (`cuda`). The real forward tool's busy contract was smoke-tested: with the lock held it printed `GPU BUSY (nothing loaded, nothing written)` and exited 5.

## Q9. AUXVETO A1: V-S1C in V-PRED's role; its next arm
* A1: *"V-S1C is read by the same section 6 decision list, as its own row, after V-PRED's."*; section 6 is written for V-PRED (*"V-PRED lower bound > 0 AND point >= +0.30"* ...).
* **Resolution.** V-S1C's stratified Delta PDMS x100 (unrounded) takes V-PRED's role in the list; V-GT is read by the same estimator in the same call. The row's next-arm cell is **quoted verbatim**; A1 adds no separate cell, so an item that names "S1c" (e.g. AV-PROMISING's *"S1c (a better raster) through this rule, plus the seed-1 replicate"*) is the arm just read, and the remaining items are the committed next arms. `RESULT.md` prints that note under the quote. AV-HARM's cell reads V-GT: the row prints which branch applies (GT ceiling positive => raster quality; else AV-NULL's cell is quoted too). V-PRED's own row is not recomputed and cannot change (A1: *"It cannot change V-PRED's row."*).
* **Validity.** The tool applies section 9's gates 1, 0, 3, 4, 5 and A1's replacement of gate 2 (*"the read reproduces S1c's own C3 P1 IoU [0, 40) from aux_s1_readout.py"*, tolerance 1e-6, the harness P1 of the same head from the same run). Any failure => the V-S1C row is **INVALID** (recorded, "commits NO next arm", the row it would have been is printed for the record only). A veto INVALID does **not** fail the S1c marker: the S1C row is independent (marker payload `veto_valid`).
* **Instrument identity** (mine, not the prereg's): the veto tool must reproduce the banked `AUXVETO_READ.json` V-GT and O-DAC blocks to 1e-9 (MEASURED 0.0); a miss refuses the TOOL (FAILED), never produces a verdict.

## Q10. The four families, STRATEGIC, and the descriptive arm-B read
* S1C section 8: *"The four families (LONGITUDINAL, LATERAL, TACTICAL) are NOT COMPUTED as paired reads ... Arm B's pick is read descriptively against the three 400-update N arms and stamped budget-confounded ... STRATEGIC: PAUSED."* AUXVETO section 8 defines the four-family table for the veto.
* **Resolution.** `RESULT.md` section 6 states LONGITUDINAL / LATERAL / TACTICAL "NOT COMPUTED as paired reads" with the reason; the V-S1C table (section 5) is the NAVSIM-tier four-family read of the head's raster (it needs no N arm; the base is the banked u400 scorer). The descriptive arm-B levels are produced by `s1c_desc_selection.py` (levels only, no paired difference, no interval, no classification, stamped BUDGET-CONFOUNDED; the instrument reproduces S1's banked H1-s0 90.7085 and N-s0 90.7633 exactly). **It is best-effort and non-gating:** if an input is unreadable the report says "NOT READ" loudly and the read continues. STRATEGIC is printed as PAUSED (PI 2026-09-19, item 25) in the headline, in section 5 and in section 6; never a pass, never omitted, never extended to the other three.

## Q11. The prereg files are pinned; a later amendment is surfaced, not applied
* S1C header: *"Nothing below may change after landing except by a dated amendment appended at the end."*
* **Resolution.** The driver hashes the prereg bodies before it reads anything: `PREREG_S1C.md` body (above the first `## Amendment`, LF-normalised, one trailing newline) must be `317584a3...`; `PREREG_AUXVETO.md` body above `## Amendment A1` (CRLF form, as `auxveto_read.py` asserted) `ef9f17fa...` and A1 itself `2b26ccad...`. An edited body or A1 => FAILED before any work. A later appended amendment is **listed verbatim (heading) in `RESULT.md`** but cannot be applied mechanically (the reader cannot interpret new rules): it is the MM's to read, and the line in the log says "AMENDMENTS PRESENT". The rows' text is parsed from the file at run time (never retyped), so a quoted lever cannot drift from the prereg.

## Q12. The markers
* `S1CREAD_DONE` = a RECORDED S1C row (O1-O4). The payload carries the AUXVETO row (or its INVALID) and `veto_valid`.
* `S1CREAD_FAILED` = anything else (INVALID, a refusal, a timeout, a bug), with phase + reason; a bug is "unexpected <Type>: ..." never a silent death. A relaunch clears FAILED and reuses every completed stage (stages are cached by input fingerprint; a GPU-busy outcome is never cached).
* `S1CREAD_DRYRUN_OK` = the dry run on S1's H1-s0 (plumbing only; never quote its rows).

## Q13. The operational check (S1C section 7) is reported, never gating
`s_per_update` mean and the train hours from `sft.jsonl` are compared with 1.24-1.8 s and 3.4-4.7 h and printed in `RESULT.md` with the prereg's own caveat (*"a reading outside it means the cost model, not the claim, is wrong"*). They do not enter any row.
