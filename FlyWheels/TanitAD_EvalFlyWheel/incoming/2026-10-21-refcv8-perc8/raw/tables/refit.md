Fit set (TRAIN, never EVAL): TRAIN-DIAG: 139 TRAIN episodes x 8 windows of the refcv8 train4572 cache (sha12-sorted, first 139 with a 10 cm GT file); EMA-48,852 weights; unshifted pose clock; 1074 box windows. Tool `stack/scripts/refit_perception_thresholds.py` (sha256 5a6db392be352a07...). Checkpoint step 48852, md5 b4364029b68f29a33eeb4aebbbbd002b.

| head | P=R gate, no NMS | NMS radius (TRAIN-AP@2 m argmax) | gate re-fitted after NMS | train AP@2 m at that radius |
|---|---|---|---|---|
| box3d | 0.2660 | 2.5 m | 0.2165 | 0.4722 (no NMS 0.3462) |
| agent | 0.2394 | 3.0 m | 0.1844 | 0.4161 (no NMS 0.1865) |

box3d TRAIN grid (radius -> TRAIN AP@2 m / TRAIN P=R gate): none -> 0.3462 / 0.2660; 0.5 -> 0.3532 / 0.2638; 1 -> 0.3954 / 0.2540; 1.5 -> 0.4440 / 0.2391; 2 -> 0.4658 / 0.2264; 2.5 -> 0.4722 / 0.2165; 3 -> 0.4561 / 0.2080; 4 -> 0.4202 / 0.1943

refcv7@50,400's shipped values for reference (fit on ITS TRAIN-DIAG): box3d P=R gate 0.2567, NMS 2.5 m, gate after NMS 0.2145.