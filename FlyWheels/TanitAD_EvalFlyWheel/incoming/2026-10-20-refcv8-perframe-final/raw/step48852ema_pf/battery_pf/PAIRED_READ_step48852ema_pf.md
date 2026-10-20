# refcv8 unlabelled pass: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* step unknown (pass dir not named step<N>[ema]_<pass>): read the root
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.007 [-0.0097, 0.0249] n 4754/139 ep; floor 0.0029102861881256104 |
| &nbsp;&nbsp;seed 1 | delta 0.0099 [-0.0056, 0.0267] n 4754/139 ep; floor 0.0029102861881256104 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.7665 [-1.1061, -0.4232] n 3668/139 ep; floor 0.016413450241088867 |
| &nbsp;&nbsp;seed 1 | delta -0.7501 [-1.0857, -0.4188] n 3668/139 ep; floor 0.016413450241088867 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "PASS"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.29563 | 0.27842 | 0.0172 [0.0043, 0.03] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.64073 | 0.5995 | 0.0412 [0.0115, 0.0713] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.28591 | 0.25879 | 0.0271 [0.0122, 0.0416] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.25027 | 0.22603 | 0.0242 [0.0113, 0.0368] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.55359 | 0.49906 | 0.0545 [0.0239, 0.0846] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.82951 | 0.85339 | -0.0239 [-0.035, -0.0129] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.39995 | 0.37153 | 0.0284 [0.0101, 0.0468] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4188 | 1.4525 | -0.0337 [-0.0959, 0.0206] | False | 4529/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0036545 | 0.0038683 | -0.0002 [-0.0005, 0.0001] | False | 4493/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.018053 | 0.018965 | -0.0009 [-0.0021, 0.0001] | False | 4493/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10074 | 0.10301 | -0.0023 [-0.0082, 0.0032] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.20441 | 0.20714 | -0.0027 [-0.0153, 0.009] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.96214 | 0.96172 | 0.0004 [-0.004, 0.0048] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.81742 | 0.83046 | -0.013 [-0.0282, 0.0019] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.29854 | 0.27902 | 0.0195 [0.0077, 0.0314] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.64632 | 0.60077 | 0.0456 [0.0184, 0.0726] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.28758 | 0.25954 | 0.028 [0.0147, 0.0413] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.25243 | 0.22665 | 0.0258 [0.0138, 0.0375] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.5571 | 0.50021 | 0.0569 [0.0289, 0.084] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.82904 | 0.85191 | -0.0229 [-0.0334, -0.0122] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.40013 | 0.37231 | 0.0278 [0.0109, 0.0446] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4287 | 1.4414 | -0.0127 [-0.0645, 0.0331] | False | 4528/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0036489 | 0.0037122 | -0.0001 [-0.0003, 0.0002] | False | 4494/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.01807 | 0.018668 | -0.0006 [-0.0016, 0.0003] | False | 4494/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10185 | 0.10306 | -0.0012 [-0.0065, 0.0036] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.20698 | 0.20736 | -0.0004 [-0.0118, 0.0104] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.96256 | 0.96214 | 0.0004 [-0.0038, 0.0048] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.81615 | 0.83572 | -0.0196 [-0.0345, -0.005] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.7813 | 2.5675 | 0.2139 [0.0806, 0.3535] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.8448 | 6.3328 | 0.512 [0.1792, 0.8815] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0375 | 0.93285 | 0.1047 [0.0526, 0.1634] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.4354 | 2.19 | 0.2453 [0.1155, 0.3824] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.9609 | 5.3853 | 0.5756 [0.2544, 0.9375] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.50441 | 0.5368 | -0.0324 [-0.0488, -0.0158] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.47162 | 0.43063 | 0.041 [0.0223, 0.0628] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.1059 | 3.2815 | -0.1757 [-0.3874, 0.02] | False | 3563/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0051487 | 0.0054983 | -0.00035 [-0.0007, -5e-05] | True | 3536/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.027857 | 0.029006 | -0.0011 [-0.0026, 0.0001] | False | 3536/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.83712 | 0.84125 | -0.0041 [-0.0588, 0.0476] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.1108 | 2.1315 | -0.0207 [-0.1636, 0.1202] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.88468 | 0.87704 | 0.0076 [-0.0035, 0.0197] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.60333 | 0.60387 | -0.0005 [-0.0245, 0.0227] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.7977 | 2.5682 | 0.2295 [0.1053, 0.3587] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.8851 | 6.3389 | 0.5462 [0.2329, 0.8898] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0413 | 0.93572 | 0.1056 [0.0543, 0.16] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.4491 | 2.1892 | 0.2599 [0.1377, 0.3949] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.9998 | 5.3866 | 0.6132 [0.2944, 0.9578] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.50418 | 0.53376 | -0.0296 [-0.0459, -0.0133] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.47048 | 0.43221 | 0.0383 [0.02, 0.0587] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.1492 | 3.2617 | -0.1125 [-0.3139, 0.0759] | False | 3563/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0051134 | 0.0053633 | -0.0002 [-0.0006, 0.0001] | False | 3536/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.027935 | 0.028826 | -0.0009 [-0.0023, 0.0004] | False | 3536/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.8444 | 0.84705 | -0.0027 [-0.0565, 0.0503] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.1243 | 2.1456 | -0.0213 [-0.1566, 0.1213] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.88304 | 0.87677 | 0.0063 [-0.0052, 0.0188] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.59678 | 0.60087 | -0.0041 [-0.0275, 0.018] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
