# DAC-SENSITIVITY: how good must a vision-predicted-map DAC checker be? (MM, 2026-11-28)

**Status:** DESCRIPTIVE. Tier T1, on C1's banked navtest table. It follows the PI ruling of 2026-11-28: the check runs against a vision-predicted drivable area, with no HD map at inference.

**Method:** the checker is the true DAC with i.i.d. per-proposal errors.
- A failing proposal passes with probability `e_miss`.
- A passing proposal is rejected with probability `e_fa`.
- The rule is FILTER-ORACLE's: argmax of C1's agg32 over the proposals that pass the checker, scored with `tr`.
- Each cell is the mean over 5 seeds; the seed-to-seed sd is ≤ 0.01. Script: `dac_sensitivity.py`; JSON: `DAC_SENSITIVITY.json`.
- ⚠ Real map errors are correlated in space within a token. Read this as a first-order design guide, never a bar.

**Delta PDMS by checked trajectory and error rates:**

| target | e_miss \ e_fa | 0 | 0.03 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| **tracked** trajectory | 0 | +1.532 | +1.537 | +1.527 | +1.536 |
| | 0.10 | +0.926 | +0.924 | +0.897 | +0.874 |
| | 0.25 | +0.573 | +0.567 | +0.532 | +0.492 |
| | 0.50 | +0.305 | +0.286 | +0.236 | +0.171 |
| **planned** proposal | 0 | +0.538 | +0.539 | +0.528 | +0.500 |
| | 0.10 | +0.313 | +0.304 | +0.276 | +0.221 |
| | 0.25 | +0.165 | +0.155 | +0.130 | +0.089 |

**Design rules that follow:**
1. **False alarms are nearly free.** Rejecting up to 20 % of the passing proposals keeps the whole gain, because C1's ranking among the survivors is good.
   - ⇒ Tune the predicted drivable area for **recall of violations**, conservatively: when uncertain, call the area non-drivable.
2. **Misses cost more than linearly.** A miss rate of 10 % keeps about 60 % of the gain, and 25 % keeps about 37 %, because a zeroed token carries many failing proposals and missing any high-scoring one undoes the switch.
   - ⇒ The bar to design to is the **miss rate on high-agg32 failing proposals**.
3. **Check the TRACKED trajectory.** It is worth about 3× the planned one: +1.53 against +0.54 at a perfect checker.
   - The TRK dump already carries the tracker's simulated trajectories (`trk20`, [12146, 64, 20, 3]).
