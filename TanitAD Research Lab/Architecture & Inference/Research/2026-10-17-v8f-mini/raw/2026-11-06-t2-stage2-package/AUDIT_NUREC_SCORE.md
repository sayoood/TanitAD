# AUDIT_NUREC_SCORE: why P10 reads ~36 deg heading MAE on the two stage-1 starts (AGENT F, 2026-11-06)

**Verdict: (a) REAL DIFFICULTY. No scorer or alignment defect (b).** Nothing in `t2v8f/code` was changed; no re-score of the stage-1 pull is needed
(the pull's scoring is correct as written).
Evidence class: MEASURED (ours), CPU, `t2v8f/audit/` + `t2v8f/tests/test_t2v8f_audit.py`. Tier: T2 plumbing rollouts (never read for H1 or H2); the T1 comparison rows are the
reader's own banked T1-kin eval-141 output, not recomputed.

## 1. The "36 deg" is a 2-start mean of two unlike starts
`PLUMBING_SCORE.json` groups are means over 2 starts (one per scene). Per start, with the reader's own rows (`rr_lib.rows_of`, via `t2v8f_score.own_rows`):

| start | v0 | logged 6 s future | P10 heading MAE | P10 yaw-rate MAE | P10 ADE6 | valid steps (both / X / G of 30) |
|---|---|---|---|---|---|---|
| 00040136 f0 60 | **17.9 m/s (moving; NOT near standstill)** | slows 17.9 to 12 m/s, right curve to -26 deg yaw (84 m, y -13 m) | 10.25 deg | 5.1 deg/s | 10.80 m | 30 / 30 / 30 |
| 7c72937c f0 22 | 0.0008 m/s (**standstill launch**) | 0 to 5.9 m/s straight, 18.9 m | **61.43 deg** | **283.0 deg/s** | 4.26 m | 20 / 20 / 22 |

(10.25 + 61.43) / 2 = 35.84 = the "about 36" in `STAGE1_READ.md`; C0_s0(T2) is (1.04 + 77.05) / 2 = 39.05 = the pull's 39.046. The brief's premise "both starts sit near
standstill" is wrong for 00040136. The 30 to 45 deg level is the launch start alone; the moving start reads 1 to 10 deg (arms C0_s0 1.0, C0_s1 6.8, DAGOACC_s0 1.8, P10/ZERO 10.25,
BLIND 10.5). Which arm reads what is a stage-1 plumbing fact only.

## 2. Why the launch start reads 61 deg / 283 deg/s (the mechanism)
P10 (no camera) plans ~1 cm of noise from a stopped car. The playback executor sets the heading from the tangent of that noise (`EPS_M = 1e-3`, so a 1 cm tangent counts), the next
replan reads a history whose yaw rate is hundreds of deg/s, and the plan blows up (replan 2 last knot (-5, 44) m; replan 3 (244, -32) m; executed speeds to 46 m/s by the last knots).
The reader masks steps with ds <= 0.5 m/s x 0.2 s, which drops 10 of 30 knots; the remaining 20 carry random headings. This happens in T1 too (the executor is the T1 tool's own
functions).

## 3. The discriminating tests
| test | result |
|---|---|
| **Logged track as the plan** (T2g `ReplayPolicy` through `run_t2_rollout`, 3 replans x 10 knots), scored by the reader's own rows, both starts | max knot diff **2.8e-14 m / 3.6e-15 m**; ADE6, heading, yaw-rate, cross all **~1e-14 to 0.0**; 0 NaN rows, **including the standstill launch** |
| **Cross-check with the T1 reader's path** on the same logged rows (`tool.exec_in_ego` = `floors_kin.gt_waypoints`) | T2's `g` equals it **bit for bit** (max diff 0.0) on both |
| Start pose alignment with the real P10 policy | track row 2*t0 = logged row f0 (x, y, yaw), yaw equals travel direction (median diff -0.09 / +0.08 deg over v>1 m/s): frame and yaw sign are consistent; C0_s0 turns the right way on 00040136 (ADE6 4.8, heading 1.0) |
| **Mutation: shift by one tick** (executor `skew=1`) | ADE6 3.09 / 10.09 m, yaw-rate 179 / 185 deg/s: leaves ~0 |
| Mutation: logged future built 0.1 s late / 0.2 s late | ADE6 0.33 / 1.41 m and 0.67 / 2.81 m: leaves ~0 |
| Mutation: yaw sign flipped | RED on the moving start (ADE6 2.2 m); **insensitive on the launch start (yaw ~0; 1.4e-4 m)**, so it is not a launch-start test |

## 4. The T1 reader's own rows on the same start class (banked, not recomputed)
`dagland/motion_c/ri_read/raw/arrays/RIC0_s0.npz` (2399 eval-141 windows, `b_k` = P10 floor, `a_k` = C0 T1-kin):

| class (windows) | P10 heading MAE mean / median | P10 yaw-rate MAE | P10 ADE6 |
|---|---|---|---|
| standstill launch (v0 < 0.1 and logged 6 s extent > 10 m; n 18, 12 episodes) | 59.6 / **61.2** deg (IQR 55.7 to 62.0) | 278 / **283.0** deg/s (IQR 267.6 to 283.2) | 4.38 / 4.00 m |
| any standstill (v0 < 0.1; n 71; heading valid on 52) | 72.1 / 64.8 | 284 / 283 | 4.44 / 4.52 |
| v0 >= 2 (n 2188) | 6.3 / 3.5 | 8.8 / 2.9 | 3.32 / 2.64 |
| 00040136 analogue (v0 16 to 20, extent 70 to 95 m, lateral >= 8 m; n 4 windows, 3 episodes) | 17.9 / 18.1 | 8.5 / 8.2 | 7.9 / 8.1 |

The stage-1 launch start (61.43 deg, 283.03 deg/s) sits on the T1 median of its class. C0 on T1 launch windows reads 65.5 deg / 259 deg/s (the stage-1 C0 arms read 52 to 77 deg).
The moving start 00040136 is a hard start, not a defective one: P10 heading is at the 94th and ADE6 at the 98.7th percentile of T1 P10 windows with v0 16 to 20 (n 231), explained by a
5.7 m/s deceleration and a right curve that a constant-extrapolating prior cannot know (it accelerates 18 to 20 m/s).

## 5. Consequences for stage 2 (for the Master Mind; the frozen card is untouched)
1. **Never pool lateral rows over launch and moving starts.** A mean of 10 and 61 deg describes no start. Report the lateral and heading rows per start class (launch / moving), and
   keep M1 (crawl share, the deciding longitudinal row) as the card has it. Whether the card's section 6 needs an amendment is the MM's call; none is proposed by this audit because
   the card's deciding row is M1.
2. Launch starts carry an unstable executed heading (section 2): the lateral rows there are a stress test of the executor, not a vision reading. This is inherited from T1 (same
   tool functions), measured above, and is not a defect of the T2 harness.
3. The cheap per-scene gate used here (logged plan self-score, CPU, no render) is now pinned as `tests/test_t2v8f_audit.py` (3 tests) and
   `audit/run_mutations_audit.py` (9 of 9 as expected). It is suitable as a stage-2 CPU gate on every scene's rig trajectory.

## Files
`audit/selfscore_run.py` (executor side), `audit/selfscore_score.py` (scorer side), `audit/selfscore_stage1.json` (results), `audit/run_mutations_audit.py`,
`tests/test_t2v8f_audit.py`. New files only; `t2v8f/code` is untouched, so the stage-1 tar `1785337f...3421c` is still current.
