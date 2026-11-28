# AMENDMENT 5 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): Master Mind rulings on the suite fill

**Date:** 2026-11-08 ~10:40Z.
**Ruled before** any checkpoint was scored on these scenes. No threshold of any scored rule changes. The sim session raised these while filling `SUITE_TOKENS.json`.

1. **QC 1, "no ego-lane discontinuity", is defined as: the human stays within 1.75 m (half a lane) of the route-lane centre for the first 6 s.** The lifter keeps one successor-connected lane chain, so a human lane change that early cannot be represented. The limit is applied strictly, with no margin. Token `9164913b` (1.76 m) is OUT, and later ranks fill as they fall. A 1 cm miss is not a reason to rule a token back in: choosing tokens by how close they came is the hand-picking section 3.1 forbids.
2. **A human who leaves the route lane after 6 s stays in the suite** (`adfdcd3c`, 3.55 m; `1a1fbb25`, 1.95 m), because QC 1 covers the first 6 s. But once the human has left, the human path is no longer a valid reference for an ego that follows the route lane, and a good ego would read as failing. So every rule that is measured against the human path (a1, a2, c4, e1, e2) is evaluated only up to `human_ref_valid_until_s`: the first tick at which the human is more than 1.75 m from the route-lane centre. Later ticks are NA for those rules and the count is reported. Rules that do not use the human path (b, c1 to c3, d, f, and EPDMS) keep the whole window. The sim session emits this field per scenario.
3. **Merge n = 4 in tier 2**, inside the table's 2–4. The total is 28 slots.
4. **Tier 1 takes the first-filled slots in fill order:** cruise 3, steady lead 3, lead braking 1 and stop-go 1, merge 1, J-UL 1 (10 scenarios).
