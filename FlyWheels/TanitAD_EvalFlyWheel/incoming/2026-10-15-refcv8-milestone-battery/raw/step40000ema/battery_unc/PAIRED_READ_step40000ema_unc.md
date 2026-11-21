# refcv8 step-40,000 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 40,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.007 [-0.0082, 0.0227] n 4754/139 ep; floor 0.0008828341960906982 |
| &nbsp;&nbsp;seed 1 | delta 0.0079 [-0.0075, 0.0232] n 4754/139 ep; floor 0.0008828341960906982 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.6971 [-1.0214, -0.3796] n 3668/139 ep; floor 0.01825094223022461 |
| &nbsp;&nbsp;seed 1 | delta -0.6789 [-1.0007, -0.3676] n 3668/139 ep; floor 0.01825094223022461 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.29558 | 0.27842 | 0.0172 [0.006, 0.0276] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.639 | 0.5995 | 0.0395 [0.014, 0.0631] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.28097 | 0.25879 | 0.0222 [0.0096, 0.0339] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.2463 | 0.22603 | 0.0203 [0.009, 0.0308] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.54319 | 0.49906 | 0.0441 [0.0176, 0.0688] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.83051 | 0.85339 | -0.0229 [-0.0338, -0.012] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.39217 | 0.37153 | 0.0206 [0.0048, 0.0358] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4904 | 1.4512 | 0.0392 [-0.0028, 0.0833] | False | 4524/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0039859 | 0.0038545 | 0.0001 [-0.0001, 0.0004] | False | 4489/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019509 | 0.018966 | 0.0005 [-0.0003, 0.0014] | False | 4489/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10509 | 0.10301 | 0.0021 [-0.0025, 0.0064] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.2141 | 0.20714 | 0.007 [-0.0027, 0.0166] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.9594 | 0.96172 | -0.0023 [-0.0069, 0.0021] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.82289 | 0.83046 | -0.0076 [-0.0217, 0.0069] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.29646 | 0.27902 | 0.0174 [0.0062, 0.0282] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.64112 | 0.60077 | 0.0403 [0.0145, 0.065] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.28082 | 0.25954 | 0.0213 [0.0087, 0.0332] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.2463 | 0.22665 | 0.0197 [0.0083, 0.03] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.54248 | 0.50021 | 0.0423 [0.0163, 0.066] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.83367 | 0.85191 | -0.0182 [-0.0284, -0.0074] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.39138 | 0.37231 | 0.0191 [0.0038, 0.0334] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4996 | 1.4401 | 0.0594 [0.0132, 0.1083] | True | 4522/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.003996 | 0.0037019 | 0.00029 [5e-05, 0.00056] | True | 4491/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.019648 | 0.018668 | 0.001 [-0.0, 0.0021] | False | 4491/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10596 | 0.10306 | 0.0029 [-0.0019, 0.0078] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.2167 | 0.20736 | 0.0093 [-0.0011, 0.0195] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95793 | 0.96214 | -0.0042 [-0.0093, 0.0006] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.82331 | 0.83572 | -0.0124 [-0.0254, 0.0013] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.8507 | 2.5675 | 0.2833 [0.1441, 0.4245] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.1029 | 6.3328 | 0.7701 [0.3909, 1.1651] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0568 | 0.93285 | 0.124 [0.0625, 0.1866] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.451 | 2.19 | 0.261 [0.1239, 0.3996] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.0762 | 5.3853 | 0.6908 [0.318, 1.0669] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.49673 | 0.5368 | -0.0401 [-0.0579, -0.0215] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.48699 | 0.43063 | 0.0564 [0.0321, 0.0833] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.528 | 3.2663 | 0.2618 [0.0514, 0.4883] | True | 3554/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0056982 | 0.00539 | 0.0003 [-0.0001, 0.0007] | False | 3530/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.030447 | 0.028851 | 0.0016 [-0.0001, 0.0035] | False | 3530/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.90754 | 0.84125 | 0.0663 [0.0201, 0.1134] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.3082 | 2.1315 | 0.1767 [0.0396, 0.3112] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.86996 | 0.87704 | -0.0071 [-0.0213, 0.0074] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.5747 | 0.60387 | -0.0292 [-0.0611, 0.0003] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.869 | 2.5682 | 0.3007 [0.174, 0.432] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.1556 | 6.3389 | 0.8167 [0.464, 1.1972] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0601 | 0.93572 | 0.1244 [0.0663, 0.1844] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.4665 | 2.1892 | 0.2773 [0.1515, 0.4085] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.1336 | 5.3866 | 0.747 [0.398, 1.1181] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.49605 | 0.53376 | -0.0377 [-0.0558, -0.0197] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.48545 | 0.43221 | 0.0532 [0.0297, 0.0789] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.6007 | 3.2492 | 0.3516 [0.1176, 0.6155] | True | 3555/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0057411 | 0.0053087 | 0.00043 [1e-05, 0.00088] | True | 3530/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.030688 | 0.028738 | 0.002 [0.0001, 0.004] | True | 3530/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.91934 | 0.84705 | 0.0723 [0.0189, 0.1274] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.3284 | 2.1456 | 0.1828 [0.0406, 0.3294] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.86668 | 0.87677 | -0.0101 [-0.024, 0.0038] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.57252 | 0.60087 | -0.0284 [-0.0587, 0.0005] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
