
## Amendment 2 (`A2-2026-10-26`), POST-DATA and CONTROL ONLY: the one-lever control compared two trainer record formats

**When, and what had been seen.** Written 2026-10-26 ~10:40.
* X_s0 (`M2b_s0_oacc`) had trained, been evaluated and been re-fitted on box 2.
* The read of seed 0 was then REFUSED by the B-vs-X built-lever control in `check_x_run`. The refusal message, in full:
  `built levers ['residual_zero_init', 'w'] differ from B_s0's`, for both `summary.json` and `selfcheck.json`.
* The reader refuses before it computes any row, and it wrote no read file. No outcome row, no family row and no eval number of X has been read
  by anyone. This amendment is written without them.

**The defect: a control comparing record formats, not levers.** MEASURED on the real records:
1. **`residual_zero_init` is a readout, not a lever.** It is True exactly while O's last residual layer `head_o.2` is all zero.
   * B (banked M2b, trained 2026-10-17 by an older trainer) writes it AFTER training in `summary.json` (False) and BEFORE training in
     `selfcheck.json` (True).
   * The current trainer writes the before-training value in both.
   * X's head trained: in its checkpoint, `head_o.2.weight` has an L1 norm of 164.4, against B's 137.4.
   * `test_v8f_grid.py` and `test_v8f_hist.py` already drop exactly this difference ("written BEFORE training (True) and AFTER it (False)").
2. **`w`: the current trainer records `w_tac` and `w_cons` inside it; B's older record lists only `w_pen`, `w_pen_self` and `w_wm`.**
   * B's own checkpoint cfg reads w_tac 1.0 and w_cons 0.5 at both seeds, and so does X's.
   * B's argv names neither.

**Fix, in `code/v8f_oacc_read.py`.** The control stays strict:
* `residual_zero_init` joins `lever_skip`.
* `w` is compared key by key. A weight that B's record lacks must equal B's BUILT config (`W_B_BUILT = {w_tac: 1.0, w_cons: 0.5}`). Any other
  difference is refused, by name.
* Proof that the fix does what it should:
  * the reader as it ran reproduces the refusal on the real records;
  * the fixed reader returns no defect on them;
  * a test refuses each real weight change (w_tac 2.0, w_cons 0.25, w_wm 0.2, a missing or added weight);
  * a new mutant turns the weight check off, and it must go RED.

**Root-cause class.** A cross-version control: an older banked arm compared field by field with a newer trainer's record. The generic rule
already lives in two tests and was not carried into this reader.

Nothing else moves: the rule's w_acc, the rows, the bars, the outcome table, the seed-1 condition and every committed next step.
