# refcv8 step-30,000 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 30,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta -0.0008 [-0.0149, 0.0136] n 4754/139 ep; floor 0.002290278673171997 |
| &nbsp;&nbsp;seed 1 | delta 0.0015 [-0.0131, 0.0163] n 4754/139 ep; floor 0.002290278673171997 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.8146 [-1.139, -0.5132] n 3668/139 ep; floor 0.02442455291748047 |
| &nbsp;&nbsp;seed 1 | delta -0.7902 [-1.1101, -0.4921] n 3668/139 ep; floor 0.02442455291748047 |
| BAR-R8-NR | {"ADE": "FAIL: S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.28777 | 0.27842 | 0.0093 [-0.0016, 0.0197] | False | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.61989 | 0.5995 | 0.0204 [-0.0054, 0.0449] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.2696 | 0.25879 | 0.0108 [-0.0012, 0.0224] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.23778 | 0.22603 | 0.0117 [0.0009, 0.022] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.52321 | 0.49906 | 0.0241 [-0.0013, 0.0482] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84192 | 0.85339 | -0.0115 [-0.0216, -0.0011] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.37397 | 0.37153 | 0.0024 [-0.0128, 0.017] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.479 | 1.4521 | 0.027 [-0.012, 0.0649] | False | 4518/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0038559 | 0.0038449 | 0.0 [-0.0002, 0.0002] | False | 4486/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019155 | 0.018973 | 0.0002 [-0.0006, 0.001] | False | 4486/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10419 | 0.10301 | 0.0012 [-0.0035, 0.0055] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.21116 | 0.20714 | 0.004 [-0.0058, 0.0137] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95961 | 0.96172 | -0.0021 [-0.0065, 0.0023] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.82688 | 0.83046 | -0.0036 [-0.0164, 0.0095] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.29006 | 0.27902 | 0.011 [-0.0005, 0.022] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.62531 | 0.60077 | 0.0245 [-0.002, 0.0493] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.27222 | 0.25954 | 0.0127 [0.0004, 0.0244] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.24023 | 0.22665 | 0.0136 [0.0026, 0.024] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.52873 | 0.50021 | 0.0285 [0.0024, 0.0532] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.8405 | 0.85191 | -0.0114 [-0.0214, -0.0015] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.37686 | 0.37231 | 0.0045 [-0.0107, 0.0186] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4454 | 1.4013 | 0.0441 [0.0001, 0.0879] | True | 4517/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0038456 | 0.0036928 | 0.0002 [-0.0001, 0.0004] | False | 4486/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.019205 | 0.018681 | 0.0005 [-0.0004, 0.0014] | False | 4486/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10467 | 0.10306 | 0.0016 [-0.0033, 0.0062] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.21233 | 0.20736 | 0.005 [-0.0059, 0.0153] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95751 | 0.96214 | -0.0046 [-0.0092, -0.0002] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.82541 | 0.83572 | -0.0103 [-0.0227, 0.0027] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.7333 | 2.5675 | 0.1658 [0.0268, 0.3153] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.8091 | 6.3328 | 0.4763 [0.0876, 0.9005] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0108 | 0.93285 | 0.0779 [0.0166, 0.1424] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.3579 | 2.19 | 0.1678 [0.0285, 0.3127] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.8566 | 5.3853 | 0.4713 [0.0921, 0.8793] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.50741 | 0.5368 | -0.0294 [-0.048, -0.0114] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.46993 | 0.43063 | 0.0393 [0.0135, 0.0669] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.402 | 3.2725 | 0.1295 [-0.0405, 0.3237] | False | 3544/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0054112 | 0.0053668 | 0.0 [-0.0003, 0.0004] | False | 3522/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.029183 | 0.028854 | 0.0003 [-0.001, 0.0019] | False | 3522/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.86213 | 0.84125 | 0.0209 [-0.0251, 0.0685] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.1861 | 2.1315 | 0.0546 [-0.0678, 0.1827] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.87323 | 0.87704 | -0.0038 [-0.0155, 0.0071] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.58124 | 0.60387 | -0.0226 [-0.056, 0.0079] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.7577 | 2.5682 | 0.1895 [0.0464, 0.3391] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.8604 | 6.3389 | 0.5214 [0.1317, 0.9354] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0159 | 0.93572 | 0.0802 [0.0182, 0.1444] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.3739 | 2.1892 | 0.1848 [0.0442, 0.3237] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.8864 | 5.3866 | 0.4998 [0.1148, 0.8971] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.50591 | 0.53376 | -0.0279 [-0.0469, -0.0088] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.47068 | 0.43221 | 0.0385 [0.0127, 0.0665] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.4282 | 3.2348 | 0.1935 [0.0026, 0.3996] | True | 3544/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0054615 | 0.0052362 | 0.0002 [-0.0001, 0.0006] | False | 3520/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.029525 | 0.028648 | 0.0009 [-0.0005, 0.0024] | False | 3520/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.8732 | 0.84705 | 0.0261 [-0.0236, 0.0773] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.2118 | 2.1456 | 0.0662 [-0.0631, 0.2032] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.87568 | 0.87677 | -0.0011 [-0.012, 0.0093] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.58206 | 0.60087 | -0.0188 [-0.0522, 0.0112] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
