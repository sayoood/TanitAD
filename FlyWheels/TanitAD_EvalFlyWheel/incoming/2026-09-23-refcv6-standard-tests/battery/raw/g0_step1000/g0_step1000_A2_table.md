**A2 wrapper clause = PASS**; G0-A1 as registered = PASS; **G0-A2 = PASS** (reasons: none); step 1000, ckpt md5 `7c3ad3c1fbf3d30be5c7733e7b436c65`

| condition | settings | max Wrapper rel | max Floor rel | terms over bar | W1 detected | W2 detected |
|---|---|---|---|---|---|---|
| P1_as_run | cuDNN TF32 True, det False, trunk bf16 True | 8.443e-04 | 4.799e-05 | 26 of 63 | — | — |
| P2_cudnn_det | cuDNN TF32 False, det True, trunk bf16 True | 8.447e-04 | 4.808e-05 | 26 of 63 | — | — |
| P3_fp32_det | cuDNN TF32 False, det True, trunk bf16 False | 6.696e-07 | 1.179e-07 | 0 of 63 | 1 | 26 |

P3 worst terms (wrapper / floor / bar): `box3d_centre` 6.70e-07 / 0.00e+00 / 1.00e-05; `box3d_z` 5.86e-07 / 0.00e+00 / 1.00e-05; `sel_v3` 4.52e-07 / 0.00e+00 / 1.00e-05; `agent_centre` 3.95e-07 / 0.00e+00 / 1.00e-05; `box3d_h` 2.36e-07 / 1.18e-07 / 1.00e-05

P1 worst terms (wrapper / floor): `agent_centre` 8.44e-04 / 0.00e+00; `map_iou_drivable` 5.91e-04 / 0.00e+00; `box3d_centre` 5.06e-04 / 0.00e+00; `map_pred_drivable_frac` 4.02e-04 / 0.00e+00; `agent_yaw` 4.00e-04 / 0.00e+00
