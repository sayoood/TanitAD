
## Amendment 2 (`A2-2026-10-26`), POST-DATA and CONTROL ONLY: C-1's float32 ulp is taken at the blend's largest operand

**When, and what had been seen: the exact sequence (dev-box clock, file mtimes).**
* 11:47: the full chain ended, 12 / 12 runs, `ZZHMC-DONE 12`.
* 11:54:49: the reader with Amendment 1 REFUSED on C-1 alone (`read/HMC_READ.json`). P-1, C-2, C-3, C-4 (as amended), C-5, C-6, C-6b and C-7
  passed. The reader stops before any row on a REFUSED, and `--force-rows-under-refusal` was not used.
* 11:56:40: the operand probe (`probe_c1_operand.json`, the table below) showed every run at ≤ 0.5 ulp at the largest operand. No outcome row
  existed yet.
* 11:57:33–11:57:46: the rule below was coded into the reader as `--c1-bar ulp32max`. This was its whole content: the unit moves to the largest
  operand, and the bar stays at 1.
* 11:59:30: the reader ran with it, passed every control, and computed and printed the outcome rows.
* ⚠️ **12:00:46: this text was written, AFTER the outcome had been printed.** The rule it records is the one already in the reader at 11:57,
  and nothing in it was changed after the outcome. A text written after the outcome is still a post-outcome document, and is disclosed as
  one. *(An earlier draft of this paragraph said no outcome row had been computed when it was written. That was false, and a review caught
  it.)*
* Why the outcome could not have shaped the rule: the probe at 11:56 had already shown that the new unit passes every run, and C-1 decides
  only whether rows are read, never which outcome they give.

**The defect.** Amendment 1 measured |d| in float32 ulps at |plan|. On a near-stationary tick, the blend's result is smaller than its
operands and they have opposite signs. MEASURED, MC-H s0, tick 4336 of m10, knot 3: plan −0.0017 m, from P +0.0046 m and τ −0.0267 m, with
d 1.35e-10 m = 1.16 ulp at |plan|. The float32 roundings inside `P + c (τ − P)` sit at the operands' scale, and a ulp taken at the result's
magnitude understates them under cancellation. The 24-window SMOKE that set Amendment 1's unit held no such tick.

MEASURED (`raw/2026-10-26-hmc-a2/probe_c1_operand.json`, the reader's own d):

| | MC-H s0 | MC-H s1 | MC-F s0 | MC-F s1 |
|---|---|---|---|---|
| max \|d\| / ulp at \|plan\| (A1) | 1.159 | 3.722 | 2.461 | 1.487 |
| tick-knots over 1 ulp at \|plan\| (of 86,364 at m10) | 1 | 1 | 3 | 1 |
| max \|d\| / ulp at max(\|P\|, \|τ\|, \|plan\|), m10 / m30 | 0.50000 / 0.49999 | 0.50000 / 0.49998 | 0.49999 / 0.49997 | 0.50000 / 0.50000 |
| regression, HACC's table on the MC-H ticks (m10): ulp at the largest operand | 5.19e6 (79,163 over) | 5.23e6 (79,160 over) | | |

**Fix.** C-1's bar becomes max |d| / spacing(float32(max(|P|, |τ|, |plan|))) ≤ 1: one float32 ulp at the blend's largest operand.
* The number 1 is Amendment 1's and does not move. Only the magnitude at which the ulp is taken moves, from the result to the operands.
  The arithmetic of a float32 blend sets that magnitude, not the data.
* The regression arm is unchanged and must still FAIL.
* `table_constant_over_knots_1_12` is still required.

**Root-cause class.** A float tolerance taken at the result's magnitude, blind to cancellation. This is the `R74` class again, now in the
amendment that was written to fix it, and its bar was calibrated on a smoke that held no near-zero result. ⇒ A rounding bar is set at the
largest operand of the operation that rounds, never at its output.

Nothing else moves: the arms, the TRAIN bar, the rows, F, the outcome table, the committed next steps, and C-4 as Amendment 1 amended it.

The reader applies this amendment only with `--c1-bar ulp32max --c1-amendment A2-2026-10-26`. The two earlier modes (`literal`, and A1's
`ulp32`) are kept and still REFUSE on these runs.
