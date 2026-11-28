# refcv8 step-5,000 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 5,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta -0.0004 [-0.0161, 0.0159] n 4754/139 ep; floor 0.0017810165882110596 |
| &nbsp;&nbsp;seed 1 | delta -0.0022 [-0.0171, 0.0134] n 4754/139 ep; floor 0.0017810165882110596 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.9876 [-1.2924, -0.7085] n 3668/139 ep; floor 0.004557371139526367 |
| &nbsp;&nbsp;seed 1 | delta -0.9922 [-1.2926, -0.7247] n 3668/139 ep; floor 0.004557371139526367 |
| BAR-R8-NR | {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.28818 | 0.27842 | 0.0098 [-0.0007, 0.0203] | False | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.61669 | 0.5995 | 0.0172 [-0.007, 0.0424] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.26802 | 0.25879 | 0.0092 [-0.0026, 0.0219] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.23744 | 0.22603 | 0.0114 [0.0009, 0.0225] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.51967 | 0.49906 | 0.0206 [-0.0037, 0.0463] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.85197 | 0.85339 | -0.0014 [-0.0079, 0.0054] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.37511 | 0.37153 | 0.0036 [-0.0111, 0.0186] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4654 | 1.4521 | 0.0133 [-0.0161, 0.0443] | False | 4519/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0038443 | 0.0038222 | 0.0 [-0.0001, 0.0002] | False | 4487/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.01913 | 0.018942 | 0.0002 [-0.0003, 0.0007] | False | 4487/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10217 | 0.10301 | -0.0008 [-0.0044, 0.0026] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.20474 | 0.20714 | -0.0024 [-0.0101, 0.005] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95898 | 0.96172 | -0.0027 [-0.0063, 0.0006] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83298 | 0.83046 | 0.0025 [-0.008, 0.0133] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.2864 | 0.27902 | 0.0074 [-0.0024, 0.0176] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.61251 | 0.60077 | 0.0117 [-0.011, 0.0357] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.26615 | 0.25954 | 0.0066 [-0.0044, 0.018] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.23581 | 0.22665 | 0.0092 [-0.0002, 0.0189] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.51614 | 0.50021 | 0.0159 [-0.0063, 0.0391] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.85281 | 0.85191 | 0.0009 [-0.006, 0.0077] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.3732 | 0.37231 | 0.0009 [-0.0129, 0.0155] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4573 | 1.4416 | 0.0157 [-0.015, 0.0485] | False | 4519/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0037844 | 0.0037063 | 0.0001 [-0.0001, 0.0002] | False | 4487/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.018919 | 0.018693 | 0.0002 [-0.0003, 0.0008] | False | 4487/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10172 | 0.10306 | -0.0013 [-0.0051, 0.0021] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.2039 | 0.20736 | -0.0035 [-0.0117, 0.004] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.96024 | 0.96214 | -0.0019 [-0.0053, 0.0015] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.83782 | 0.83572 | 0.0021 [-0.0074, 0.0116] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.5602 | 2.5675 | -0.0072 [-0.1067, 0.0977] | False | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.2652 | 6.3328 | -0.0676 [-0.3076, 0.1808] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 0.91382 | 0.93285 | -0.019 [-0.0579, 0.0193] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.1852 | 2.19 | -0.0049 [-0.1058, 0.0968] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.3079 | 5.3853 | -0.0775 [-0.3167, 0.1645] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.5408 | 0.5368 | 0.004 [-0.0061, 0.0146] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.42128 | 0.43063 | -0.0094 [-0.0222, 0.0033] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.3618 | 3.2717 | 0.0901 [-0.0149, 0.2] | False | 3542/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0055353 | 0.0053823 | 0.0002 [-0.0, 0.0003] | False | 3518/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.029223 | 0.028827 | 0.0004 [-0.0004, 0.0012] | False | 3518/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.83004 | 0.84125 | -0.0112 [-0.0458, 0.0244] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.1117 | 2.1315 | -0.0198 [-0.1086, 0.0761] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.87132 | 0.87704 | -0.0057 [-0.0123, 0.0014] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.62077 | 0.60387 | 0.0169 [-0.0024, 0.0368] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.5557 | 2.5682 | -0.0126 [-0.1089, 0.0848] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.2576 | 6.3389 | -0.0814 [-0.3173, 0.156] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 0.91164 | 0.93572 | -0.0241 [-0.062, 0.0134] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.1805 | 2.1892 | -0.0086 [-0.1049, 0.0895] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.2972 | 5.3866 | -0.0894 [-0.3273, 0.1381] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.54153 | 0.53376 | 0.0078 [-0.0027, 0.0185] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.42045 | 0.43221 | -0.0118 [-0.0247, 0.0002] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.3867 | 3.266 | 0.1207 [0.0173, 0.2346] | True | 3544/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0054929 | 0.0052956 | 0.0002 [2e-05, 0.0004] | True | 3518/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.029158 | 0.028725 | 0.0004 [-0.0003, 0.0012] | False | 3518/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.83587 | 0.84705 | -0.0112 [-0.048, 0.0243] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.1267 | 2.1456 | -0.0189 [-0.1102, 0.0719] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.8705 | 0.87677 | -0.0063 [-0.0125, 0.0] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.61587 | 0.60087 | 0.015 [-0.006, 0.0359] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
