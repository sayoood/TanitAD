# HANDOFF_KINLAB (facts + census, no design)

**The label.** Under swap/add the own state enters only as INPUT: kin3 = (v, a, kappa = clamp(omega/max(v,2), +-0.3)) and p10f/base derived from it (FACTS 1). There is no yaw-rate channel and no yaw/accel/speed head. The target is `gt` [30,2] at 0.2 s (positions, human frame at r'), plus `s_h`, `lab*`, `lead_gap_h`, all human. Under `speed_v1` only the ARC TIMING is own (start at v_own, rate 3.0 m/s^2 toward the human's speed, then the human's); the first segment keeps the human's heading and curvature (RV:287-330). `act` (logged yaw rate, accel) stays human and feeds only the WM loss.

**Measured gap (shard 0, n = 372,791 ok, RH bank readout causal_g).** Yaw rate NOW p50 1.84, p95 23.5 deg/s; 30 % of records above 5 deg/s; human window-to-window scale p95 1.8 (p95 over 2 s lag 8.2). Acceleration NOW p50 1.55, p95 6.4 m/s^2; 41 % above 2 (scale p95 0.58). Speed control reproduces RECLAB exactly (p50 0.7725, p95 6.2210). Gaps are flat over the first 2 s (own NOW value fixed). So the gap is NOT negligible; it is 10x the natural variation at p50. Caveat: part may be own-readout noise (not separated).

**What the facts force or rule out.**
* `reanchor_rows` does NOT erase it: it moves only the pose history; kin3, p10f, base keep own yaw rate and acceleration (D:590-601).
* A yaw-rate or acceleration blend cannot be written as a target on any head (none exists); it is expressible only through `gt` (positions), i.e. as a label path whose initial heading/curvature and speed ramp start from the own state and merge into the human path. speed_v1 already does the speed ramp (A_LABEL 3.0) but never the heading; `recovery_label` (RV:156) has the only own-heading Hermite construction.
* Acceleration has no target other than the second difference of `gt` (`accel_term`, only when w_acc > 0) and the speed ramp.
* Envelope: FD has A_MAX 4.0, kappa 0.2, mu 0.7 (friction circle), no yaw-rate and no jerk limit; the own kappa input cap is 0.3 (above the 0.2 envelope), so an own-kappa start can already violate the curvature limit (own |kappa| p99 gap 0.22).
* The level label is the logged speed and the WM record is logged: a kinematic label for swapped rows would leave those two human.
