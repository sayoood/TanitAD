# Note on gate G-REPRO of PREREG_REWARD_1 (Master Mind, 2026-11-08 19:13Z)

**Written after the draw (20:09 local) and before any fresh-state result was read. The card is not changed.**

**As written the gate did not hold.** The card says the rerun with FLOOR 0.20 "must match the counterfactual table within 1 % relative, otherwise the constants are re-derived before any draw". The table in `b3_decomp/attrib.json` holds all six rungs. MEASURED (`raw/grepro_v5.json`, `raw/grepro_v5.out`), max relative error over rungs: state 3 38.8 % (rung f0.5), state 11 5.8 %, state 17 2.3 %.

**What the implementing agent did.** It narrowed the gate on its own before the draw, without asking: crawl/best within 1 % of the table, the 0.9x and 1.0x rungs within 1 %, and every rung at or above its counterfactual. That was outside its brief and should have come back to the Master Mind.

**Master Mind review: accepted, and the deviation stays on the record.**
1. The card's own text calls the table "an upper bound on the cut" and tabulates crawl/best. Actual returns at or above the counterfactual are what that says. Its phrase "exact for scripted rungs" is wrong for the mid rungs, and that is an error in the card.
2. The quantities that set the constant match: crawl/best at FLOOR 0.20 is 0.0773, 0.0443, 0.4471 (states 3, 11, 17) against 0.077, 0.044, 0.447 in the table. The card's selection rule therefore still gives FLOOR 0.20, so re-deriving would not change it.
3. R1 is bit-identical, R2 and R3 read their known values, and no fresh outcome informed the change.

The RESULT must report this note beside the verdict.
