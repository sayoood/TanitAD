# A2 RESULT: U2

**longer training helps and the head can fit the features; generalisation is the limit**  ->  next lever: S1c with more updates AND more label logs, each as its own registered pair

| quantity | value |
|---|---|
| H = U10k held-out IoU [0,40) | 0.675218  (95% logs-only CI 0.6430..0.7076) |
| I = U10k in-sample IoU [0,40) (mean of 5 folds) | 0.894452  (pooled 0.894481) |
| H(V-s0, 2,000 updates), banked | 0.634891  (I 0.702434) |
| D = H(U10k) - H(V 2k), paired | +0.040327  (95% CI +0.0322..+0.0491); CI excludes 0: True; beyond floor {'V_seed_floor': 0.003552465097100832}: True; SEPARATED (probe rule): True |
| row if 'separated' meant floor-only | U2  (row depends on the CI clause: False) |
| per-fold held-out IoU | 0.6712, 0.7152, 0.6093, 0.6785, 0.6733 |
| per-fold in-sample IoU | 0.8972, 0.8931, 0.8928, 0.8937, 0.8954 |
| SDF MAE held-out overall / near-boundary (m) | 1.5083 / 2.2371  (2k: 1.6168 / 2.5109) |
| last-10% train-bce slope per 1,000 updates (5 folds) | -0.0017, -0.0010, -0.0009, -0.0001, -0.0004 |
| train wall time (5 units) | 5858 s |

Instrument: identity test EXACT (V-s0 2k held-out IoU 0.6348906684369943 == banked); zero control exact 0.0, CI [0,0]. V seed floor used 0.003552465097100832.
Tier T0-like (teacher-forced head perception), never driving performance. Estimator: see A2_RESULT.json `estimator`.
