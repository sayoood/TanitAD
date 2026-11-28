# MM reading REPS1, addendum 1: the union also reads AHEAD of the baseline on STRATEGIC. Both sides belong in the headline (2026-11-19)

This addendum adds to `MM_READING_REPS1.md` (landed 37ebb758) after the coordinator's review. No number in the reading changes.

## Why this addendum exists
The reading's "Beside" section and its headline carried the union's TAC-LAT cost against BL with numbers. They carried the union's STRATEGIC lead against BL only as a list of cell names, and they left out UE_s0. Both are in the same result file, and an honest stage-A line carries both.

## The numbers
All cells are MEASURED, from `read/RESULT_REPS1.json` (sha256 708f943f), `families/STRATEGIC/cells_stage_A`.
- Tier: T0, first tick.
- Population: 141 episodes.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only.
- Floors are the card's per-cell F.

| pair | S2 route skill at 80 m | S1 route position error at 80 m | RC-A heading error at 80 m |
|---|---|---|---|
| UE_s0 - BL_s0 | **+0.1189 [+0.0513, +0.2016]**, gain | **-0.7246 m [-1.3748, -0.1470]**, gain | -1.320 deg [-2.791, -0.101], gain |
| UE_s1 - BL_s0 | **+0.0984 [+0.0193, +0.1909]**, gain | -0.2815 m [-0.7187, +0.1550], not separated | -0.724 deg [-1.841, +0.312], not separated |
| UM_s0 - BL_s0 | +0.0656 [-0.0453, +0.1702], not separated | -0.6331 m [-1.3560, +0.0080], not separated | -1.653 deg [-3.397, -0.143], gain |
| UM_s1 - BL_s0 | +0.0943 [+0.0305, +0.1660], gain | -0.6089 m [-1.1137, -0.1734], gain | -1.754 deg [-3.419, -0.401], gain |

Floors: S2 0.0164, S1 0.2070 m, RC-A 0.0613 deg. Each "gain" is separated and above its floor.

## The honest stage-A line
**At stage A, the union reads better than the corpus-only baseline on the strategic route read (T0) and worse on the lateral decision in the rollout (TAC-LAT, T1-kin). Neither is decision-grade.**

Reasons:
- BL exists at seed 0 only, so every seed-1 row against BL carries seed variance.
- The two families sit on different tiers: T0 first tick against T1-kin.
- Stage A is not the deployed recipe; stage B finishes it.

This is the case the stage-B card (`PREREG_DSCALE_STAGEB`, in draft, coordinator direction D9) must name. Its UNION-COST row prints a family gain at both seeds as a named trade-off (sub-label UNION-COST-TRADE), never netted against the cost.
