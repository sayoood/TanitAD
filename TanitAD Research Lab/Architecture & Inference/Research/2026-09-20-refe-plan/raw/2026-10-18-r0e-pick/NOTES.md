# NOTES: R0e-pick draft, open questions for the PI and the Master Mind (2026-10-18)

The companion file is `PREREG_R0E_PICK.draft.md`. It is a draft and is not registered. CPU only, nothing run, nothing committed.

## Decisions for the PI

1. **A40 spend.** Stage 1 is ≤ 4.95 A40-h: R0e-id 1.27 + R0e-pick 3.68. It drops to 3.34 if R0e-pick is K3-killed at u500. Stage 2 (R1e-pick + R1o) is ≤ 7.36, run only on O1, as in SAFEOBJ. The maximum is 12.31. All ESTIMATED from R0d's banked wall clock (§7).
2. **T = 0.01 (recommended) or 0.02.**
   * **0.02 is A2R2's committed T\*** (its rule: "the largest aligned grid T"). Its margin is thin: 0.208 vs the 0.25 bar.
   * **0.01 is a disclosed departure.** Margin 0.103. It is aligned at all five τ in A2R2's sensitivity and on A2R's near-tie-free subset, but those reads are descriptive.
   * A2R2 itself is post-hoc-motivated, and A2R, the pre-registered read, is UNDERPOWERED.
   * Switching to 0.02 changes only the T token and the registered-T constants. Every row of §6 stands; O2, O3-a and K2 would then name 0.01 or a hard-pick constraint as their next lever.
3. **R0e-id stop at u250** needs a supervisor stop-after-dump (build item B4). The alternative is running R0e-id to u1000 for +2.41 A40-h. That buys a full same-seed replicate of R0d at all four checks, the first training-rerun floor for a constrained arm.
4. **Bar G under FP-NONE.** G was kept unchanged ("prefer none"). Its premise, that a free deployable frontier point exists, is weaker on navtest: M25 read PDMS −0.47 [−1.03, +0.24] vs BASE (D-REFE-FREEPOINT-1). Should an O2 outcome, which is safe but below G, still earn Stage 2? The draft says no, matching SAFEOBJ's protocol.
5. **The DEPLOYABLE-SAFE vs-BASE line beside E** is reported and decides nothing, because FREEPOINT's FP-NONE consequence adds no clause. Should it become an E clause for a deployment decision? That would be a bar change and must be decided before data.
6. **DDC.** R0d's DDC was −0.92 [−1.45, −0.48], separated adverse on RL-val. No bar covers it, and the constraint does not touch it. If it persists, g5 + DDC as a third term is a candidate lever, and that is a PI scope question.

## Integration and escalation (Master Mind)

* **The outcome tool would NOT_READ R0e-pick as landed.** SO-PICKCON's `safeobj_outcome.py` fix accepts the nine-key `epo_con` block only when `temp` is exactly 0.1. A per-arm registered T is a launch blocker (B3). This is the "built, tested, unreachable" class.
* **`--max-updates 250` must not be used for the identity arm.** The cosine schedule runs over `total` (patched `scorer_finetune.py:1661-1662`), so the learning rate at u ≤ 250 would differ from R0d's.
* **Start the carve-feed build now, as a 0-GPU stream.** It is the committed next arm of row P-O4-PICK. Its feed half does not exist: the dual reads only `con_acc`, summed from training micro-batches (patched `:1685-1688`, `:1704-1705`). `--heldout-logs` is the RL-val list in the registered argv and must not be reused for it.
* **K2's grad-norm clause is not implemented.** `rls1_checks.py` kills only on a non-finite loss (`:488`, `:515`). `PREREG_RL_S1.md` §5's "grad norm > 20 × the control's median" is therefore unenforced, and that clause matters most at a low T. Logging the return value of `clip_grad_norm_` (patched `:1693`) would make it enforceable. It is optional for this arm because D2 uses `epo_J`, which is already logged.
* **Register rows.** Before registration, nothing goes into `GOALS_AND_CLAIMS.md`. On freeze, a `D-REFE-R0E-PICK` row with status PREREGISTERED.

## Facts verified while drafting (for the reviewer)

| fact | evidence |
|---|---|
| R0d's trainer blob = tip blob | `e0639288` at `8bb47ad`, `e2d4422` and `ac07b1a` (git ls-tree) |
| SO-PICKCON patched trainer | `869b1862` (git hash-object of `C:/Users/Admin/pickcon/out/.../refe/scorer_finetune.py`) |
| R0d timings | `sft.jsonl`: 9.70 s / update, 460 s / eval, 3.34 h wall |
| C0 gate + preflight | `rls1_chain.log`: 0.34 h |
| R0d post-ramp means (31 rows) | g −0.085 ± 0.026, p1gap −0.232 ± 0.069, epo_J 0.8171 (u25) → 0.8766 (mean u751–1000) |
| A2R2 committed rule and outcome | `PREREG_SOPICKCON_A2R2_TIEAWARE.md` §3; `a2r2_tieaware.json` `outcome` = `A2R2-ALIGNED(T*=0.02)` |
| FP-NONE consequence | `PREREG_FREEPOINT.md`: "the safety-objective arm proceeds exactly as pre-registered" |
