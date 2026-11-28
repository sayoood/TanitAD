# KINLAB census: own-versus-human kinematic gap at NOW (RH bank s0, shard 0 of 4)

**What ran:** `census_kinlab.py` (CPU, one process, tree at df250a00) on `/root/rollhist/bank/s0/shard_000of004.npz` (sha256 aefbb46c..., pulled read-only with scp, 533 MB, sha matches RECLAB's record) and `/root/dscale10k/merged/native_tracks_union.pt` (sha256 9022952e..., 51 MB, matches). Bank generator (header): readout `causal_g`, acc_gain 1.0, path_exec true, kappa_max 0.2. Records 492,478; ok 372,791 (tick 1: 208,340, tick 2: 164,451); all ok records had a track (3,628 clips). Full JSON: `KINLAB_CENSUS.json`.
**Definitions.** Own = the record's kin3 at NOW (v, a, kappa); own yaw rate = kappa * max(v, 2). Human = the logged track at r'+k (k = 0..10 steps of 0.2 s), same one-frame backward definitions (`yaw_kin_from_track`). "first 1 s" pools k = 0..5, "first 2 s" k = 0..10 (pairs, not records). Gaps are absolute. Scale = human vs human, |x[r'+d] - x[r']|, d = 1 is the next logged window.
**Control:** |v_own - u_h[1]| p50 0.7725, p95 6.2210, n 372,791 = REAL_SHARD_CENSUS_2.json exactly (p10f v vs kin3 v max diff 2e-6).

| quantity (unit) | window | p50 | p90 | p95 | p99 | share > thresholds |
|---|---|---|---|---|---|---|
| yaw rate (deg/s) | NOW | 1.84 | 16.25 | 23.48 | 49.59 | >1: 65.1 %, >2: 48.1 %, >5: 29.9 % |
| | first 1 s | 1.87 | 16.47 | 23.83 | 49.94 | 65.3 / 48.5 / 30.3 % |
| | first 2 s | 1.90 | 16.70 | 24.19 | 50.16 | 65.5 / 48.9 / 30.6 % |
| | scale d=1 / d<=5 / d<=10 | 0.155 / 0.272 / 0.381 | 1.06 / 2.60 / 4.31 | 1.78 / 4.74 / 8.16 | 3.61 / 11.0 / 18.4 | >5: 0.3 / 4.6 / 8.6 % |
| acceleration (m/s^2) | NOW | 1.55 | 4.60 | 6.42 | 16.74 | >0.5: 80.1 %, >1: 63.9 %, >2: 41.4 % |
| | first 1 s | 1.60 | 4.59 | 6.40 | 16.69 | 80.2 / 64.2 / 42.3 % |
| | first 2 s | 1.60 | 4.59 | 6.38 | 16.66 | 80.2 / 64.3 / 42.4 % |
| | scale d=1 / d<=5 / d<=10 | 0.112 / 0.159 / 0.211 | 0.42 / 0.69 / 0.96 | 0.58 / 0.98 / 1.35 | 1.04 / 1.75 / 2.28 | >2: 0.07 / 0.6 / 1.6 % |
| speed (m/s) | NOW | 0.72 | 4.52 | 6.14 | 11.43 | >0.5: 59.5 %, >1: 41.2 %, >2: 24.3 % |
| | first 1 s / 2 s | 0.88 / 1.03 | 4.62 / 4.81 | 6.22 / 6.35 | 11.5 / 11.6 | |
| | scale d=1 / d<=10 | 0.070 / 0.295 | 0.29 / 1.68 | 0.38 / 2.34 | 0.54 / 3.74 | |

By bank tick at NOW (p50 / p95): yaw tick 1 1.65 / 19.2, tick 2 2.15 / 31.8; accel tick 1 1.16 / 5.04, tick 2 2.09 / 7.54; speed tick 1 0.42 / 1.90, tick 2 1.98 / 8.18. Per-k rows (k = 0..10) are in the JSON; the own-vs-human gap is nearly flat in k (the own NOW value is fixed, the human's drifts little): yaw p50 1.84 to 1.97.
Secondary yaw measure (0.2 s cache-grid polyline heading change, the brief's literal wording): NOW p50 1.835, p95 23.46, share >1/2/5 deg/s 65.0/48.1/29.9 %; same as the native-track one within 0.02. Kappa (1/m, the model's input channel): NOW p50 0.0058, p95 0.156; scale d=1 p50 0.0003, p95 0.0066.

**Not covered / caveats.** Shard 0 only (one of four); own state is only a snapshot at NOW (ticks 1, 2): the own FUTURE is not in the bank, so "over 1 s / 2 s" means own NOW against the human's value at r'+k, not an own-vs-human trajectory. The own yaw rate and acceleration come from the executed-knot readout (`causal_g`, path_exec): their noise level is not separated from true state difference here, and no human-vs-readout-of-human control was run (the own a gap p50 1.55 m/s^2 against a human one-window scale 0.11 may partly be readout noise; a replay of a logged track through the same readout would separate it). Pairs are pooled (k-steps of one record are correlated). No training, no GPU, nothing on the pod.
