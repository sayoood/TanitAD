# FROZEN AMENDMENT 2 (bank_presteps) -- POST-HOC variant V2, written BEFORE any V2 output was produced

Amends nothing in `FROZEN_CARDS.md` (sha256 `3202a62f...5fd5b`) or `FROZEN_AMENDMENT_1.md` (sha256 `4676bdb8...93c7`). **V1 (`recovery/labeller.py`, Amendment 1 construction, row ladder) remains the REGISTERED result of deliverable 2.** V2 is reported BESIDE it, labelled POST-HOC, and may not replace it in any verdict.

## Why V2 exists (Rule Zero: a failed arm leaves the next arm behind)

I ran V1's smoke on 3,000 randomly drawn off_lat / off_head records of bank_s0 before writing this file, so V2 is chosen AFTER seeing a V1 outcome (this is the disclosure the pre-registration discipline requires). What I saw: of 3,000 records, 573 emitted (19 %), 1,584 refused_infeasible (53 %), 699 no_logged_future (23 %: the end-edge condition the matcher's first-failing-cut order hides), 128 degenerate, 16 path_exhausted; 0 checker violations among the emitted. A rung-by-rung diagnosis on 900 of them: at the short rungs the failing check is curvature (and friction), at the long rungs the label cannot reach the target in 6 s (it is tied to where the HUMAN is at row r' + k_j, so a slow or stopped car faces a target it cannot reach, and a stopped human gives a degenerate ladder). The card's wording is "a geometric blend toward the human path at a LOOKAHEAD" -- a lookahead is a distance, not a row; the V1 row ladder was my implementation choice.

## V2 (fixed now)

* Identical to V1 in every respect (speed profile, label distance `l[k]`, Hermite blend, tail along the human polyline, checker L4, identity branch) EXCEPT the target ladder: the target is the human-polyline point at arc `S* in {6, 10, 15, 22, 32, 45, 60} m` from the matched row r' (clipped: a rung beyond the polyline's end is `path_exhausted`), tangent = the central chord there. First feasible rung emitted; none feasible = refused.
* The ladder is fixed here and not tuned: it is run once on both seeds and reported whatever it reads.
* Code: `recovery/labeller_v2.py` (imports labeller.py unchanged), runner `recovery/recovery_run_v2.py`, outputs `raw/recoveryV2_<seed>.json`. The same real-data gate checks as V1 (identity branch is shared code; zero checker violations among emitted) apply.
* Reading of the outcome: V2's kept fraction is reported against the same band (75-81 % at tick 2) but the CARD-LEVEL statement uses V1 unless the Master Mind adopts V2 explicitly.
