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
| &nbsp;&nbsp;seed 0 | delta 0.0581 [0.0432, 0.0738] n 4754/139 ep; floor 0.00034099817276000977 |
| &nbsp;&nbsp;seed 1 | delta 0.0585 [0.0433, 0.0744] n 4754/139 ep; floor 0.00034099817276000977 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.7094 [-1.0307, -0.3908] n 3668/139 ep; floor 0.0024361610412597656 |
| &nbsp;&nbsp;seed 1 | delta -0.707 [-1.0342, -0.394] n 3668/139 ep; floor 0.0024361610412597656 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "PASS"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.34672 | 0.27842 | 0.0683 [0.0536, 0.0834] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.74342 | 0.5995 | 0.1439 [0.1104, 0.1787] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.33253 | 0.25879 | 0.0737 [0.0577, 0.0906] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.30272 | 0.22603 | 0.0767 [0.0613, 0.0925] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.65753 | 0.49906 | 0.1585 [0.1244, 0.1937] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.79628 | 0.85339 | -0.0571 [-0.0712, -0.0434] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.3975 | 0.37153 | 0.026 [0.0114, 0.0414] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4512 | 1.4533 | -0.0021 [-0.0575, 0.0514] | False | 4528/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0037741 | 0.003889 | -0.0001 [-0.0004, 0.0001] | False | 4493/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.018269 | 0.018994 | -0.0007 [-0.0018, 0.0003] | False | 4493/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10316 | 0.10301 | 0.0002 [-0.0059, 0.0059] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.20892 | 0.20714 | 0.0018 [-0.0117, 0.0149] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.96256 | 0.96172 | 0.0008 [-0.0034, 0.0051] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.81994 | 0.83046 | -0.0105 [-0.0265, 0.0038] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.34706 | 0.27902 | 0.068 [0.0524, 0.0834] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.74284 | 0.60077 | 0.1421 [0.1079, 0.1767] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.33389 | 0.25954 | 0.0744 [0.0575, 0.0915] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.30389 | 0.22665 | 0.0772 [0.0613, 0.0934] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.65863 | 0.50021 | 0.1584 [0.122, 0.1939] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.79275 | 0.85191 | -0.0592 [-0.0737, -0.0453] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.39795 | 0.37231 | 0.0256 [0.0097, 0.0421] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4374 | 1.4427 | -0.0053 [-0.0539, 0.0406] | False | 4527/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0037203 | 0.0037345 | -0.0 [-0.0003, 0.0002] | False | 4493/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.017941 | 0.018704 | -0.0008 [-0.0017, 0.0001] | False | 4493/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.1025 | 0.10306 | -0.0006 [-0.0062, 0.0046] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.20714 | 0.20736 | -0.0002 [-0.0126, 0.0115] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.96235 | 0.96214 | 0.0002 [-0.004, 0.0042] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.817 | 0.83572 | -0.0187 [-0.0342, -0.0048] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.8385 | 2.5675 | 0.271 [0.1354, 0.4167] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.7981 | 6.3328 | 0.4653 [0.1435, 0.8085] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0297 | 0.93285 | 0.0968 [0.045, 0.1527] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.4937 | 2.19 | 0.3037 [0.1699, 0.4458] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.9136 | 5.3853 | 0.5283 [0.2067, 0.8682] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.46306 | 0.5368 | -0.0737 [-0.0992, -0.049] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.4494 | 0.43063 | 0.0188 [0.0038, 0.0356] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.2895 | 3.2971 | -0.0075 [-0.1986, 0.1846] | False | 3556/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0054747 | 0.0055677 | -0.0001 [-0.0004, 0.0002] | False | 3530/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.028429 | 0.02906 | -0.0006 [-0.0017, 0.0006] | False | 3530/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.82811 | 0.84125 | -0.0131 [-0.0714, 0.046] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.0885 | 2.1315 | -0.043 [-0.1737, 0.0939] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.8735 | 0.87704 | -0.0035 [-0.0125, 0.0049] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.58588 | 0.60387 | -0.018 [-0.0471, 0.0109] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.8409 | 2.5682 | 0.2727 [0.1349, 0.418] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.7971 | 6.3389 | 0.4582 [0.1249, 0.8093] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0352 | 0.93572 | 0.0994 [0.046, 0.1568] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.5056 | 2.1892 | 0.3164 [0.1794, 0.4577] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.9345 | 5.3866 | 0.5479 [0.2047, 0.8932] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.46379 | 0.53376 | -0.07 [-0.0953, -0.0449] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.44927 | 0.43221 | 0.0171 [0.001, 0.0344] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.2439 | 3.2961 | -0.0522 [-0.209, 0.1014] | False | 3557/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0053063 | 0.0054788 | -0.0002 [-0.0004, 0.0001] | False | 3528/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.028092 | 0.029021 | -0.0009 [-0.002, 0.0001] | False | 3528/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.82132 | 0.84705 | -0.0257 [-0.0762, 0.0253] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.0703 | 2.1456 | -0.0753 [-0.1895, 0.042] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.87514 | 0.87677 | -0.0016 [-0.0104, 0.0071] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.58424 | 0.60087 | -0.0166 [-0.0434, 0.0114] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
