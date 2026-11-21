**A2 wrapper clause = PASS**; G0-A1 as registered = FAIL; **G0-A2 = PASS** (reasons: none); step 5000, ckpt md5 `8a1e4da0dc6561353b28e3947146751e`

| condition | settings | max Wrapper rel | max Floor rel | terms over bar | W1 detected | W2 detected |
|---|---|---|---|---|---|---|
| P1_as_run | cuDNN TF32 True, det False, trunk bf16 True | 3.921e-03 | 2.785e-05 | 30 of 64 | — | — |
| P2_cudnn_det | cuDNN TF32 False, det True, trunk bf16 True | 3.921e-03 | 2.792e-05 | 30 of 64 | — | — |
| P3_fp32_det | cuDNN TF32 False, det True, trunk bf16 False | 1.891e-06 | 9.743e-08 | 0 of 64 | 1 | 27 |

P3 worst terms (wrapper / floor / bar): `traj` 1.89e-06 / 0.00e+00 / 1.00e-05; `box3d_z` 8.99e-07 / 0.00e+00 / 1.00e-05; `box3d_centre` 5.93e-07 / 0.00e+00 / 1.00e-05; `agent_centre` 5.15e-07 / 0.00e+00 / 1.00e-05; `cls` 3.90e-07 / 9.74e-08 / 1.00e-05

P1 worst terms (wrapper / floor): `box3d_yaw` 3.92e-03 / 1.12e-07; `goal2s_err_m` 2.01e-03 / 0.00e+00; `goal_tac` 1.54e-03 / 0.00e+00; `agent_centre` 1.06e-03 / 0.00e+00; `lon` 1.02e-03 / 0.00e+00
