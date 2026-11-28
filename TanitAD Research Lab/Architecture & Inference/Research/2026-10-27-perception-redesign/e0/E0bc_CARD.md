# E0b + E0c card -- range census and per-frame pitch / roll spread (REFCV9_DESIGN §4.1)

**Instrument.** `e0bc_census.py`, run on Thor (CPU only, no GPU) by `run_e0bc_thor.sh` over the 4,372 cached TRAIN clips (4,369 read, 3 egomotion members missing), raw PhysicalAI egomotion at 100 Hz (`physicalai-b1/labels/egomotion`), v9 train label release
(`refcv8_v9labels/release_train4572/v9_labels_train.npz`, 909,651 per-frame rows). Result: `raw/E0BC_RESULT.json` (counts and distributions only, no ids). Tests `test_e0bc.py`, 4 PASS: straight 30 / 15 m/s end at 180 / 90 m; a radius-100 m circle at 20 m/s gives R sin(vT/R), R(1-cos(vT/R)) (literals) and a mid-curve window equals the first; pitch alternating 0 / 0.5° reads 0.25° and ignores yaw and roll;
**a rig that only yaws inside a 1.7°-tilted world frame reads a spurious pitch under the clip-mean statistic but ~0 under the 3 s high-pass** (the control that decided to report both).

## E0b — range census (MEASURED, decides the box decode range and the planner map extent)
Windows = every 0.4 s of a clip while t0 + 6 s fits (n = 1,140,339); GT 6 s path = the egomotion x, y in the ego frame at t0.

| clause | share of windows | note |
|---|---|---|
| GT path END beyond 60 m | **0.466** | path end x: p50 56.4 m, p90 140.1 m, p99 213.7 m |
| … beyond 80 m | **0.298** | |
| … beyond 100 m | **0.202** | |
| GT path max \|y\| beyond 16 m | **0.098** | |
| … beyond 24 m | **0.046** | |
| lead vehicle present (of 909,651 label rows) | 0.390 | |
| lead gap > 60 / 80 / 100 m (of rows WITH a lead) | 0.206 / 0.100 / **0.000** | gap p50 27.4 m, p90 80.0 m, p99 94.3 m; of ALL rows 0.080 / 0.039 / 0.000 |

**Reading (ESTIMATED consequence, not a bar).** The provisional box range x 0-80 m already covers every labelled lead (none beyond 100 m, 10 % of leads beyond 80 m); extending it to 100 m buys nothing for leads. The 6 s PATH is a different matter: 30 % of windows end beyond 80 m and 20 % beyond 100 m, so a planner-facing map that stops at 80 m cannot see the end of its own plan in 3 windows of 10; |y| ≤ 24 m covers 95.4 % of path extents (16 m: 90.2 %). The lead-gap definition is a LONGITUDINAL gap to an in-lane lead (lateral clauses are read on the path only).

## E0c — per-frame egomotion pitch / roll minus clip mean (MEASURED, decides R10: IN iff pitch p95 > 0.25°)
n = 13,029,449 samples. |deviation| in degrees:

| statistic | pitch p50 / p95 / p99 | roll p50 / p95 / p99 |
|---|---|---|
| minus CLIP MEAN (the design's definition) | 0.407 / **2.346** / 4.035 | 0.553 / 2.359 / 3.438 |
| minus 3 s centred moving average (dynamic part) | 0.117 / **0.829** / 1.827 | 0.151 / 1.040 / 1.975 |

Per-clip p95 pitch (clip-mean): p10 0.57°, p50 1.43°, p90 3.46°. **R10 is IN on both statistics** (p95 > 0.25°; the dynamic part alone is 3.3× the threshold).
⚠️ **UNVERIFIED whether this is physical pitch.** The clip-mean number is inflated by slope and by frame-tilt × yaw coupling (the first sample's quaternion has three equal imaginary parts, i.e. a tilted frame), which is why the 3 s high-pass is beside it; the high-pass value can still contain egomotion-estimator noise at 100 Hz. The cheap discriminator is the same statistic on straight, constant-speed windows, or the camera-measured horizon; neither was run.
**Evidence class.** MEASURED (ours, `raw/E0BC_RESULT.json`; superseded first run without the high-pass kept as `raw/E0BC_RESULT_v1_clipmean_only.json`, identical clip-mean numbers).
