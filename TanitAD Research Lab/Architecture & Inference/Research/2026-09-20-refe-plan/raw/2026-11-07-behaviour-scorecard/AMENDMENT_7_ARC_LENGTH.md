# AMENDMENT 7 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): human-path arc length during a standstill

**Date:** 2026-11-08 (the commit carries the time).
**Written before** the 10 Hz arc length below was measured, and before any checkpoint was scored under c2. Every threshold is unchanged.

**Evidence (the sim session's numbers, INHERITED):** on `090d6a17`, in box centre at 2 Hz, the human travels 28.01 m to the first v < 0.5 tick and 0.099 m in the remaining 9 s, net 0.093 m; its path length to the last pose is 28.11 m. Ours is 31.87 m, on 171 ticks. Both sides agree on the line: 6.43 m and 6.49 m past the last pose. **HYPOTHESIS:** our 10 Hz path accumulates arc length from pose jitter during the standstill.

## The measurement, and both committed outcomes

For `090d6a17`, on the same path the selector and the scorer build, compute the arc length from the first v < 0.5 tick to the end (A) and the net displacement over that interval (N).

- **CONFIRMED if A − N > 0.5 m.** Apply the rulings below.
- **NOT CONFIRMED otherwise.** Change nothing, report A and N, and find the actual source of the 3.76 m. `MAP_FIELDS_A6` stays held for that token.

## Rulings, if confirmed

1. **Distance filter on every human path (selector, `MAP_FIELDS`, the scorer's `HumanPath`):** the first sample is kept; each later sample is kept only if it is at least 0.2 m from the previous kept sample. Arc length and stations are computed on the kept samples only.
2. The straight extension of AMENDMENT_6 starts at the last kept sample. Its direction rule is unchanged, and it is computed on the moving part as before.
3. A stopped pose takes its station by projection onto the kept polyline, so a pose in a standstill has one well-defined station.
4. **Scenes without slow or stopped samples must score byte-identically.** If a golden scene moves, name it and show that it has samples closer than 0.2 m. Re-baseline only those.
5. **Audit again:** recompute the stop-gap distributions over the TL-G and TL-S pools. If the candidate lists change, the new lists stand and the differences are reported, since no checkpoint has been scored. The 6 m cap does not change.
