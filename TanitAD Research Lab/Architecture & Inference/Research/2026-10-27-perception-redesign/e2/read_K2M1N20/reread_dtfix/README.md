# refcv9 E2 re-read with the corrected waypoint clock (2026-11-12)

Re-run by the Master Mind (session 4a3d05) of the E2 judge (`r9_e2_verdict.py`, tip 0bbdc066) with **one change**:
`r9_e2_families.py` v2 reads the waypoint spacing from the dump's `manifest.json` `grid.dt_s` (**0.5 s**). v1 hard-coded
**0.1 s**. Everything else is the banked read (`../read/`): the same Thor dumps (`E2_K2_pf`, `E2_K0_pf`, `E2_K2_camblind`,
`dump_s{0,1}`; pulled, 866 files sha256-equal to Thor), the same lead block, distkeep, inputs and verdict code, and the
same EVAL and S6c files. The commands are `r9e2_roll.sh`'s CPU stage step for step (`reread_dtfix.sh`).

**Verdict unchanged: NO-GO.** (a), (c) and (d) FAIL at seeds 0 and 1; (b) and (e) PASS. STRATEGIC PAUSED.

## What changed (`read/families_s{0,1}.json`, T1, paired episode-cluster bootstrap over 70 DEV episodes, K2 − K0)

| row | v1 (0.1 s, wrong) s0 | **v2 (0.5 s) s0** | **v2 s1** | K0 level s0 |
|---|---|---|---|---|
| LON_accel_mae (m/s²) | +0.2445 [0.1019, 0.3944] | **+0.0098 [0.0041, 0.0158]** | +0.0108 [0.0048, 0.0169] | 0.3529 |
| LON_speed_mae (m/s) | +0.0129 | +0.0026 (ns) | +0.0031 (ns) | 0.2585 |
| LAT_yaw_rate_mae (rad/s) | +0.0042 | **+0.0011 [0.0005, 0.0017]** | +0.0009 [0.0004, 0.0015] | 0.0189 |
| LAT_curvature_mae (1/m) | +0.0003 (ns) | **+0.0003 [0.0001, 0.0006]** | +0.0003 [0.0001, 0.0005] | 0.0040 |
| LAT_heading_mae (deg) | +0.0571 | **+0.0588 [0.0204, 0.0984]** | +0.0431 [0.0123, 0.0743] | 2.0840 |
| LAT_cross_mae (m) | +0.0074 | +0.0074 [0.0037, 0.0111] | +0.0050 [0.0020, 0.0082] | 0.1007 |
| LAT_cross_abs_final (m) | +0.0166 | +0.0166 [0.0087, 0.0249] | +0.0116 [0.0046, 0.0189] | 0.2056 |

Position rows (`ade`, `along`, `cross`) are dt-free and did not move. Speed, yaw rate and accel were inflated ×5, ×5
and ×25; the tangent-validity gate and the tactical labeller's speed inputs moved too, so curvature is now separated
worse at both seeds and yaw rate at seed 1 as well. The v2 reader's own guard reads the GT spacing as
0.503368 s (`dt_check`, 2,184 moving windows).

**Rank by relative effect (s0 / s1):** LAT_cross_abs_final +8.1 % / +5.6 %, LAT_curvature +7.5 % / +7.3 %, LAT_cross
+7.3 % / +4.9 %, LAT_yaw_rate +5.8 % / +4.7 %, LON_accel +2.8 % / +3.1 %, LAT_heading +2.8 % / +2.1 %.
LON_dk_lead_progress_shortfall +36.7 % is separated at seed 0 only (37 episodes) and not at seed 1. ADE is not separated.
⇒ The consistent regression is **LATERAL**, not the LON accel row the first read ranked largest.

## Where the lateral regression lives (`read/lat_attrib.json`, DIAGNOSTIC, same estimator per stratum)

* **No side bias.** The signed cross error delta is +1.8 mm [−11.7, 16.7] (s0) and −2.6 mm (s1), not separated. K2 is
  laterally **noisier**, not offset.
* **Straight driving** (GT heading change < 2°): +7.6 mm [4.4, 10.9] (s0), +4.7 mm [1.4, 8.5] (s1) on a ~6 cm base,
  separated at both seeds. **Curving:** +6.9 / +5.7 mm, not separated.
* **High speed** (v0 13–33 m/s): +10.8 mm [6.0, 16.1] (s0), +6.6 mm [2.1, 11.4] (s1), separated at both seeds.

Reading: lane-keeping jitter on straight, fast roads. Not a turn-execution or calibration failure. Next lever: a
zero-training map-pathway ablation roll of K2 (W-MAP-2 projection input zeroed) to test whether the jitter enters
through the map path; then the W-MAP-2 / W-BOX gates (clause (c)).

One training run per arm (H-ESTIM-SEED-1); the intervals answer another draw of episodes, not another training run.
