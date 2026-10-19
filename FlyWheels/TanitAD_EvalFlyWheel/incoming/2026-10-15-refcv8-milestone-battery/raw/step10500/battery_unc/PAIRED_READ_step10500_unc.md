# refcv8 step-10,500 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **PASS** (A7 + W-R8-1; wrapper clause PASS under the fixed probe (max rel 3.264236823179439e-06)). G0 qualifies the instrument.

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 10,500 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.071 [0.0533, 0.0891] n 4754/139 ep; floor 0.0017468631267547607 |
| &nbsp;&nbsp;seed 1 | delta 0.0693 [0.0514, 0.0877] n 4754/139 ep; floor 0.0017468631267547607 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.6449 [-0.9381, -0.3577] n 3668/139 ep; floor 0.00024819374084472656 |
| &nbsp;&nbsp;seed 1 | delta -0.6447 [-0.9414, -0.3646] n 3668/139 ep; floor 0.00024819374084472656 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.35961 | 0.27842 | 0.0812 [0.065, 0.0972] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.72963 | 0.5995 | 0.1301 [0.0948, 0.1654] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.31935 | 0.25879 | 0.0606 [0.0447, 0.0767] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.3001 | 0.22603 | 0.0741 [0.0588, 0.0901] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.61146 | 0.49906 | 0.1124 [0.0788, 0.1461] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81368 | 0.85339 | -0.0397 [-0.0539, -0.0271] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.42298 | 0.37153 | 0.0515 [0.0325, 0.0691] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.6591 | 1.4141 | 0.245 [0.1838, 0.3134] | True | 4502/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0045269 | 0.0037421 | 0.0008 [0.0004, 0.0012] | True | 4464/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.022596 | 0.01894 | 0.0037 [0.0025, 0.005] | True | 4464/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.13079 | 0.10301 | 0.0278 [0.0216, 0.0336] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.26249 | 0.20714 | 0.0554 [0.0419, 0.0683] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95078 | 0.96172 | -0.0109 [-0.0179, -0.004] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83088 | 0.83046 | 0.0004 [-0.0133, 0.0149] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.35787 | 0.27902 | 0.0789 [0.0626, 0.0952] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.72481 | 0.60077 | 0.124 [0.0884, 0.1591] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.31738 | 0.25954 | 0.0578 [0.0421, 0.0742] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.29833 | 0.22665 | 0.0717 [0.0559, 0.0877] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.60685 | 0.50021 | 0.1066 [0.0729, 0.1409] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81726 | 0.85191 | -0.0347 [-0.0482, -0.0218] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.42228 | 0.37231 | 0.05 [0.0316, 0.0679] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.6577 | 1.4035 | 0.2541 [0.185, 0.3312] | True | 4504/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0044894 | 0.0036231 | 0.0009 [0.0005, 0.0013] | True | 4463/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.022491 | 0.018671 | 0.0038 [0.0025, 0.0053] | True | 4463/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.13108 | 0.10306 | 0.028 [0.0213, 0.0349] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.26265 | 0.20736 | 0.0553 [0.041, 0.0699] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95099 | 0.96214 | -0.0111 [-0.0183, -0.0042] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.83509 | 0.83572 | -0.0006 [-0.0131, 0.0128] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.9029 | 2.5675 | 0.3355 [0.1831, 0.4958] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.0856 | 6.3328 | 0.7528 [0.3483, 1.1699] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0545 | 0.93285 | 0.1217 [0.0536, 0.1911] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.4846 | 2.19 | 0.2945 [0.1467, 0.449] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.0327 | 5.3853 | 0.6474 [0.2429, 1.0644] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.49805 | 0.5368 | -0.0388 [-0.0567, -0.0204] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.48143 | 0.43063 | 0.0508 [0.022, 0.0803] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.7864 | 3.2194 | 0.567 [0.3497, 0.793] | True | 3513/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.005993 | 0.0052261 | 0.0008 [0.0004, 0.0011] | True | 3491/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.032269 | 0.028718 | 0.0036 [0.0023, 0.0049] | True | 3491/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.95216 | 0.84125 | 0.1109 [0.0608, 0.1602] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.3636 | 2.1315 | 0.2321 [0.1052, 0.3599] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.85142 | 0.87704 | -0.0256 [-0.0409, -0.0117] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.56516 | 0.60387 | -0.0387 [-0.0732, -0.0068] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.9032 | 2.5682 | 0.3349 [0.1756, 0.4954] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.0957 | 6.3389 | 0.7568 [0.3517, 1.1765] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0547 | 0.93572 | 0.119 [0.0509, 0.187] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.4846 | 2.1892 | 0.2955 [0.1447, 0.4508] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.0479 | 5.3866 | 0.6612 [0.2541, 1.0848] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.49459 | 0.53376 | -0.0392 [-0.057, -0.021] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.48093 | 0.43221 | 0.0487 [0.0204, 0.0773] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.7344 | 3.1698 | 0.5646 [0.3252, 0.823] | True | 3513/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0060262 | 0.0052096 | 0.0008 [0.0004, 0.0013] | True | 3492/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.032168 | 0.028671 | 0.0035 [0.002, 0.005] | True | 3492/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.95322 | 0.84705 | 0.1062 [0.0547, 0.16] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.3638 | 2.1456 | 0.2182 [0.0923, 0.3504] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.8536 | 0.87677 | -0.0232 [-0.0375, -0.0101] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.56189 | 0.60087 | -0.039 [-0.0723, -0.0073] | True | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
