# A2 RESULT: U3

**the head cannot fit frozen visual features even at 5x updates => the frozen trunk is the limit**  ->  next lever: the trunk: the graft (PI decision) and P6's trained trunk

| quantity | value |
|---|---|
| H = U10k held-out IoU [0,40) | 0.428047  (95% logs-only CI 0.3824..0.4778) |
| I = U10k in-sample IoU [0,40) (mean of 5 folds) | 0.433230  (pooled 0.433634) |
| H(V-s0, 2,000 updates), banked | 0.634891  (I 0.702434) |
| D = H(U10k) - H(V 2k), paired | -0.206843  (95% CI -0.2430..-0.1763); CI excludes 0: True; beyond floor {'V_seed_floor': 0.003552465097100832}: True; SEPARATED (probe rule): True |
| row if 'separated' meant floor-only | U3  (row depends on the CI clause: False) |
| per-fold held-out IoU | 0.4297, 0.4672, 0.3823, 0.4319, 0.3983 |
| per-fold in-sample IoU | 0.4313, 0.4197, 0.4466, 0.4323, 0.4362 |
| SDF MAE held-out overall / near-boundary (m) | 2.4525 / 3.6513  (2k: 1.6168 / 2.5109) |
| last-10% train-bce slope per 1,000 updates (5 folds) | -0.3749, -0.3634, +0.3859, +0.0275, -0.0522 |
| train wall time (5 units) | 231 s |

Instrument: identity test EXACT (V-s0 2k held-out IoU 0.6348906684369943 == banked); zero control exact 0.0, CI [0,0]. V seed floor used 0.003552465097100832.
Tier T0-like (teacher-forced head perception), never driving performance. Estimator: see A2_RESULT.json `estimator`.
