# AMEND_SLOWWM_1: deciding metrics, cells, and how cells combine (Master Mind, 2026-11-11 ~14:20, written before any SLOWWM basis, run or read exists)

**Card:** `PREREG_V8F_SLOWWM.md` (sha256 `7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca`). The card itself is unchanged; this file fills three gaps the read package found while being built (`read/report_slowwm.py`).

**State when written:**
* The SLOWWM chain (`C:/Users/Admin/slowwm_state`) is waiting behind the OBD seed-3 chain, which waits for STOPBAL.
* No basis has been fitted.
* No SLOWWM run, dump or margin exists.

## A1. The deciding metrics (card line 44: "WMPOOL's planner read verbatim, with one change: the deciding read is in trajectory space")
* WMPOOL decided on the harness's action-space MAEs (`acc_mae`, `yaw_mae`; `ladder_plantest/report_lp.py`). The harness reports its unicycle-integrated metrics beside (`report_lp.py:300`: speed error, heading error and ADE at 2.4 s, plan vs logged actions).
* The one change therefore makes the integrated per-family metrics deciding:
  * **LONGITUDINAL = `speed_err`** (m/s);
  * **LATERAL = `heading_err`** (rad).
* `acc_mae` / `yaw_mae` are reported beside, with no bar.
* ADE pools both families, so it stays beside.

## A2. The cells
The card reads "PLANTEST and PLANTEST2 forms … contexts 1 and 3 as labelled rows" (line 47). The read therefore has **four cells: {PLANTEST, PLANTEST2} × {context 1, context 3}**.
* The floor is per family and per cell.
* The card's outcome table is evaluated in each cell separately.

## A3. One cell's outcome
The card's table, evaluated per family, can give no outcome or more than one in a cell. Example: SLOW-PLANS on LONGITUDINAL and SLOW-COSTS on LATERAL.
* **Exactly one** outcome → the cell's outcome.
* **More than one** → the cell is **MIXED**. All of its outcomes and family rows are reported.
* **None** → the cell is **NONE-OF-FOUR**. Example: SLOW beats R35H beyond the floor but is not PLANNABLE, so neither SLOW-PLANS nor NO-EFFECT holds. Its rows are reported.

## A4. The read's outcome across the four cells
* **All four cells carry the same single outcome** → that is the read's outcome, and the card's next step for it is committed.
* **Otherwise the read is CELLS-DISAGREE.** Every cell is reported with its label. The committed next step is that of the **least favourable outcome in any cell**, in the order SLOW-PLANS > BASIS-ONLY > NO-EFFECT > SLOW-COSTS.
  * A NONE-OF-FOUR cell counts as NO-EFFECT.
  * A MIXED cell counts as its least favourable member.
  * NO-EFFECT and SLOW-COSTS share the card's next step.
* **Why the strict rule:** the four cells share the same five trained arms, so they are not independent evidence. The training-run floor rests on one replicate pair (SLOW_rep against SLOW_s0). A majority rule over correlated cells would read noise as agreement.
* **What a favourable cell still buys:** a SLOW-PLANS or BASIS-ONLY cell inside CELLS-DISAGREE is reported beside as a HYPOTHESIS for the next card, never as the read's outcome.

## A5. Unchanged
The arms, the order, the floor's definition, PLANNABLE (by the harness's own label rule), the estimator (paired episode-cluster bootstrap, B 2,000), the four outcomes and their next steps, TACTICAL n = 0, STRATEGIC PAUSED (PI item 25), and the beside reads.
