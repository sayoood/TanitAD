# refcv8 step-48,852 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **PASS** (A7 + W-R8-1; wrapper clause PASS under the fixed probe (max rel 3.702133517108737e-06)). G0 qualifies the instrument.

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* FINAL checkpoint (step 48,852 of 48,852): the pre-registered bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0244 [0.0075, 0.042] n 4754/139 ep; floor 0.0019494593143463135 |
| &nbsp;&nbsp;seed 1 | delta 0.0264 [0.0095, 0.0442] n 4754/139 ep; floor 0.0019494593143463135 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.4886 [-0.8234, -0.1408] n 3668/139 ep; floor 0.01534891128540039 |
| &nbsp;&nbsp;seed 1 | delta -0.4732 [-0.8126, -0.1294] n 3668/139 ep; floor 0.01534891128540039 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.31301 | 0.27842 | 0.0346 [0.0221, 0.0476] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.68232 | 0.5995 | 0.0828 [0.0531, 0.113] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.30342 | 0.25879 | 0.0446 [0.0302, 0.0595] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.26404 | 0.22603 | 0.038 [0.0252, 0.0509] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.58693 | 0.49906 | 0.0879 [0.0576, 0.1185] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81321 | 0.85339 | -0.0402 [-0.0524, -0.0285] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.42388 | 0.37153 | 0.0523 [0.0327, 0.0728] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.5048 | 1.4529 | 0.0518 [0.0024, 0.1026] | True | 4528/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0040756 | 0.0038731 | 0.0002 [-0.0001, 0.0005] | False | 4493/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.020044 | 0.018969 | 0.00108 [3e-05, 0.00218] | True | 4493/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10746 | 0.10301 | 0.0044 [-0.0008, 0.0094] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.2204 | 0.20714 | 0.0133 [0.0023, 0.0245] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95751 | 0.96172 | -0.0042 [-0.0088, 0.0004] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.80522 | 0.83046 | -0.0252 [-0.0414, -0.009] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.31496 | 0.27902 | 0.0359 [0.0234, 0.0484] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.6854 | 0.60077 | 0.0846 [0.0538, 0.1144] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.30274 | 0.25954 | 0.0432 [0.0279, 0.0581] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.26433 | 0.22665 | 0.0377 [0.0246, 0.0502] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.5857 | 0.50021 | 0.0855 [0.0543, 0.116] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81594 | 0.85191 | -0.036 [-0.0481, -0.024] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.42197 | 0.37231 | 0.0497 [0.0292, 0.0699] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.5261 | 1.4418 | 0.0843 [0.0375, 0.1347] | True | 4527/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0041305 | 0.0037134 | 0.0004 [0.0001, 0.0007] | True | 4494/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.020319 | 0.018671 | 0.0016 [0.0006, 0.0028] | True | 4494/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10998 | 0.10306 | 0.0069 [0.0016, 0.0119] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.2253 | 0.20736 | 0.0179 [0.0071, 0.0286] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95435 | 0.96214 | -0.0078 [-0.0135, -0.0023] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.80353 | 0.83572 | -0.0322 [-0.0488, -0.0151] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 3.0593 | 2.5675 | 0.4918 [0.3302, 0.6573] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.6189 | 6.3328 | 1.2861 [0.8529, 1.7229] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.1404 | 0.93285 | 0.2076 [0.1395, 0.2786] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.6523 | 2.19 | 0.4623 [0.3023, 0.6219] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.5708 | 5.3853 | 1.1855 [0.7561, 1.6147] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.47724 | 0.5368 | -0.0596 [-0.079, -0.0401] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.51769 | 0.43063 | 0.0871 [0.0598, 0.1166] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.6855 | 3.2838 | 0.4017 [0.1483, 0.7] | True | 3560/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0059828 | 0.0055026 | 0.00048 [3e-05, 0.00096] | True | 3536/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.032068 | 0.029016 | 0.0031 [0.0011, 0.0054] | True | 3536/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.95282 | 0.84125 | 0.1116 [0.0584, 0.1718] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.4291 | 2.1315 | 0.2976 [0.146, 0.4647] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.8626 | 0.87704 | -0.0144 [-0.0306, 0.0016] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.56189 | 0.60387 | -0.042 [-0.073, -0.0114] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 3.0747 | 2.5682 | 0.5064 [0.35, 0.6745] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.6605 | 6.3389 | 1.3216 [0.9017, 1.7714] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.1424 | 0.93572 | 0.2067 [0.1363, 0.2792] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.6585 | 2.1892 | 0.4694 [0.3115, 0.6333] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.6031 | 5.3866 | 1.2165 [0.7866, 1.6543] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.47973 | 0.53376 | -0.054 [-0.0737, -0.0343] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.51664 | 0.43221 | 0.0844 [0.0574, 0.1139] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.7061 | 3.2692 | 0.4369 [0.175, 0.7406] | True | 3560/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0059436 | 0.0053959 | 0.0005 [0.0001, 0.001] | True | 3534/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.031783 | 0.028898 | 0.0029 [0.0009, 0.0052] | True | 3534/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.96197 | 0.84705 | 0.1149 [0.0582, 0.1748] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.4345 | 2.1456 | 0.2889 [0.1336, 0.4493] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.86178 | 0.87677 | -0.015 [-0.0311, 0.0008] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.55671 | 0.60087 | -0.0442 [-0.0758, -0.0149] | True | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
