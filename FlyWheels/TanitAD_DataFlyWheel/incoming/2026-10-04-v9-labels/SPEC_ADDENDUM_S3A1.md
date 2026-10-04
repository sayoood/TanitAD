# SPEC addendum S3-A1 — implementation clarifications fixed BEFORE the Stage-3 build and validation

*Written 2026-10-04 by the Data FlyWheel, before any Stage-3 label or validation number exists. Hashed into
`raw/SPEC_SHA256.txt`. No bar is changed; each item makes a SPEC sentence computable.*

1. **"The segment holding τ\*" (SPEC §3.2, §3.3) → the band's DOMINANT segment.** τ\* (the argmax of the band
   heading excursion) usually falls on the heading plateau AFTER a turn, outside every sustained-yaw segment. The
   segment used for variant a's `is_turn` test and for every turn constraint is therefore: among the §3.1 segments
   overlapping the observed band `(NOW+2, NOW+min(h_obs, 8))`, the one whose heading change CLIPPED to that band is
   largest in magnitude with the sign of Θ (the v7 builder's own "largest in-plan segment" rule, `s2_geom_emit_v7.py
   :441-449`, applied to the v9 band). None overlapping → `lat_seg_found = 0`.
2. **Lane-change cross-section (SPEC §3.4).** The SAM3 map is in the rig frame and the near-field path is undefined at
   standstill, so `d_L`, `d_R` are read on the rig's own lateral axis at **x ∈ {5.05, 6.05, 7.05, 8.05} m** (fine
   rows 50, 60, 70, 80) instead of on the path normal. Error bound: on a bend of R ≥ 100 m the road's lateral drift at
   8 m is ≤ 0.32 m, shared by both lines (so `u` moves ≤ ~0.09); junction turns are excluded by the 10° not-a-turn
   test. A row counts when a code-2 cell exists on both sides within 4.5 m; `d_L`, `d_R` = medians over the counted
   rows; ≥ 2 rows required. Map frame index = the raw row k (`time_grid: axis0 = raw v2ep frame index`), clocked
   with the trainer's `now_s(k)`.
3. **FOLLOW lead classes (SPEC §3.5 "vehicle-class agent").** `join3d` classes admitted as a lead: `automobile,
   heavy_truck, bus, trailer, other_vehicle, train_or_tram_car, rider`. Excluded: `person, stroller, animal,
   protruding_object`. Lead geometry per frame k is computed in that frame's rig frame from the log pose at
   `now_s(k)` (the trainer's join key: raw frame = provider row + 2). A lead needs along-path `s ≥ 2.0 m`; a box whose
   centre lies in the ego footprint (−1.5 < x < 4.5 m, |y| < 1.2 m) is never a lead.
4. **Speed-limit proxy N2 / N3 (D-WPA-5)** is D4's code restated on the log: past window = [NOW − 20 s, NOW] (the
   available part near the clip start), ladder {20, 30, 50, 70, 80, 100, 120, 130} km/h, snap UP, urban floor 50 km/h
   (N2); N3 = urban if N2 ≤ 50, rural if ≤ 100, else motorway (`d4_vmax_nonoracle.py:73-75`).
5. **v_a, extrema, stop duration (SPEC §3.5).** `v_a = v(NOW + 2.0 s)` on the grid; local extrema use `v̄` (0.5 s
   centred moving average over the grid, truncated at the edges) with a 0.1 m/s plateau tolerance; `stop_dur_s` is read
   from the NATIVE log samples after the stop (first sample with v > 0.5), NaN past the log end or across a gap > 1.0 s.
