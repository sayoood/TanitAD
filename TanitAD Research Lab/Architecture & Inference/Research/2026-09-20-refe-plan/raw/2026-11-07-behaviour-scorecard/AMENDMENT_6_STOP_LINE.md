# AMENDMENT 6 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): the stop line and the path extension

**Date:** 2026-11-08 ~10:15Z.
**Ruled before** any checkpoint was scored under c2. The 6 m cap and every other threshold are unchanged.

**Evidence (the sim session's visual and numeric check on real-map renders, 5 TL-G candidates and 8 TL-S entries; `stopline_check.json` on thor3, INHERITED):**
- `stop_line_xy` coincides with the lifted connector start on 11 of 13 tokens (within 0.22 m); the exceptions are `c8d1d4dd` (−0.78 m) and `15715849` (+1.84 m).
- The polygon's true upstream edge cannot be the reference: the stopped human's front bumper is already past it on 9 of 13 tokens, by up to 4.5 m.
- On 4 TL-S tokens the stop point lies beyond the human's last position, and `stop_line_s` is wrong because the path was extended along a stopped human's noisy last heading. On `090d6a17` the stop point is 6.46 m from that extension but 0.21 m from the straight continuation of the moving path. Three stopped humans then read as past the line.

## Rulings

1. **The stop line is the START of the governing lane connector** (the first point of its baseline in the gpkg), for every token, with no per-token exceptions. AMENDMENT_3's wording "upstream edge of the stop polygon" is withdrawn. `stop_line_xy` and `stop_line_s` are recomputed on this definition, including `c8d1d4dd` and `15715849`.
2. **Straight path extension, everywhere (selector, `MAP_FIELDS_A3.json`, and the scorer's `HumanPath`):** the direction is the chord over the last 5.0 m of arc length of the path's MOVING part (ticks with v >= 0.5 m/s), never a stopped human's last heading or displacement. If the moving part is shorter than 5.0 m, use all of it. Selector and scorer use the same definition, because `stop_line_s` is a station on this path.
3. **Audit before use:** recompute the human stop gap to the new line over the whole TL-G and TL-S pools, not only the 13 checked tokens, and report the distribution. The earlier "p50 5.58 m" in AMENDMENT_3 was measured on a different population and against the polygon edge; it is withdrawn and must not be quoted. If any human-safe TL token exceeds the 6 m cap, report it; do not change the cap.
4. **If ruling 2 changes which tokens pass the TL-G or TL-S predicates, the new candidate list stands** and is reported with the differences, since no checkpoint has been scored.
