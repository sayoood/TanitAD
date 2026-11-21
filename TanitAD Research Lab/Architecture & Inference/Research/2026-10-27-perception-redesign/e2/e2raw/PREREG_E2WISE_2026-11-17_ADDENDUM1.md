# PREREG E2-WISE — ADDENDUM 1: clause (b) (MM 4a3d05, 2026-11-17 ~05:15Z)

**Written after E2-TAIL's verdict and before any E2-WISE number exists.** WISE-PLAN stage A started at 05:05:42Z; its first EVAL is about
45 minutes away. This addendum narrows what WISE-PLAN can claim. It widens nothing.

## The fact

* **The E2-TAIL verdict** (`stageB/read/E2_VERDICT.md`, 05:05:04Z) is NO-GO on **(b)** and **(d)** at both seeds.
  * (b) box AP@2 m: K2DC 0.307623 vs K0 0.307665, a point miss of **−0.00004**.
  * The paired diagnostic gives box `ap2m_all` −0.0000 [−0.0077, +0.0080]: a tie, failed by the strict point rule as written.
* **WISE-PLAN does not touch the box head's inputs.** It interpolates only the planner subset, and `core.encoder`, `_perception` and
  `core.agent_head` are copied bit-exactly.
  * If perception is bit-equal, WISE-PLAN's stage A will reproduce E2-TAIL's (a) and (b), and (b) will fail again by the same −0.00004.

## What changes

1. **WISE-PLAN cannot produce a GO** under E2's full rule while (b) reproduces the tail tie. Outcome 1's "(a), (b), (c) and (e) still
   pass" is read as: **(a), (c) and (e) still pass, and (b) equals its E2-TAIL value.**
   * Equal means within 1e-6 of 0.307623 and 0.307665. Stage A recomputes it, which is the empirical check that perception is unchanged.
   * If (b) differs from E2-TAIL, perception was not untouched. Report that as a tool or read defect, not as a result.
2. **The decision WISE-PLAN informs is unchanged**: which training lever targets the lateral deficit, read from (d), the primary readout.
   * Planner drift points to the L2-SP anchor; trunk drift (via WISE-ENC) points to the 0.02 share target.
   * Neither is a GO claim. Any training lever still needs its own read, and the PI.
3. **(b) as its own item.** The box clause is a point tie under a strict rule. No E2 prereg branch names a box lever for a −0.00004 miss.
   * The roll's verdict text names the box levers B2/B3/B4.
   * This is reported to the PI as written. Nothing here changes the E2 rule.
