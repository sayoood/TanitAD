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
| &nbsp;&nbsp;seed 0 | delta -0.0044 [-0.0184, 0.0108] n 4754/139 ep; floor 0.004229247570037842 |
| &nbsp;&nbsp;seed 1 | delta -0.0002 [-0.0145, 0.015] n 4754/139 ep; floor 0.004229247570037842 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.9251 [-1.2477, -0.6118] n 3668/139 ep; floor 0.023540735244750977 |
| &nbsp;&nbsp;seed 1 | delta -0.9016 [-1.2167, -0.5892] n 3668/139 ep; floor 0.023540735244750977 |
| BAR-R8-NR | {"ADE": "PASS", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "PASS"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.28417 | 0.27842 | 0.0058 [-0.0049, 0.0161] | False | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.61188 | 0.5995 | 0.0124 [-0.0122, 0.0359] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.27054 | 0.25879 | 0.0117 [-0.0002, 0.0231] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.23787 | 0.22603 | 0.0118 [0.0011, 0.022] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.52266 | 0.49906 | 0.0236 [-0.0015, 0.0471] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84087 | 0.85339 | -0.0125 [-0.0221, -0.0029] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.37737 | 0.37153 | 0.0058 [-0.0089, 0.0193] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4142 | 1.4509 | -0.0367 [-0.098, 0.0174] | False | 4525/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0036022 | 0.0038632 | -0.0003 [-0.0006, 0.0] | False | 4490/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.017802 | 0.018974 | -0.0012 [-0.0024, -0.0001] | True | 4490/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.099662 | 0.10301 | -0.0034 [-0.0087, 0.0017] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.20144 | 0.20714 | -0.0057 [-0.0175, 0.0054] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.96424 | 0.96172 | 0.0025 [-0.0017, 0.0067] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.82878 | 0.83046 | -0.0017 [-0.0147, 0.0112] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.2884 | 0.27902 | 0.0094 [-0.0015, 0.0198] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.6216 | 0.60077 | 0.0208 [-0.0041, 0.0441] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.27463 | 0.25954 | 0.0151 [0.0035, 0.0264] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.24206 | 0.22665 | 0.0154 [0.0048, 0.0257] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.53155 | 0.50021 | 0.0313 [0.007, 0.0546] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.83935 | 0.85191 | -0.0126 [-0.0218, -0.0034] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.3804 | 0.37231 | 0.0081 [-0.0057, 0.021] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4218 | 1.4398 | -0.018 [-0.0727, 0.0298] | False | 4523/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0036149 | 0.0037007 | -0.0001 [-0.0004, 0.0002] | False | 4491/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.017841 | 0.018667 | -0.0008 [-0.0019, 0.0001] | False | 4491/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10007 | 0.10306 | -0.003 [-0.0082, 0.0019] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.20328 | 0.20736 | -0.0041 [-0.0155, 0.0065] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.96319 | 0.96214 | 0.0011 [-0.0032, 0.0053] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.82899 | 0.83572 | -0.0067 [-0.0198, 0.0061] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.6227 | 2.5675 | 0.0553 [-0.0521, 0.1651] | False | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.4412 | 6.3328 | 0.1084 [-0.1635, 0.3932] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 0.97113 | 0.93285 | 0.0383 [-0.0048, 0.0824] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.2779 | 2.19 | 0.0878 [-0.0122, 0.194] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.5655 | 5.3853 | 0.1802 [-0.0735, 0.4497] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.52013 | 0.5368 | -0.0167 [-0.032, -0.0016] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.4479 | 0.43063 | 0.0173 [0.0025, 0.0337] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.0431 | 3.2685 | -0.2254 [-0.4452, -0.028] | True | 3556/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0049868 | 0.0054011 | -0.0004 [-0.0008, -0.0001] | True | 3531/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.027198 | 0.028871 | -0.0017 [-0.003, -0.0004] | True | 3531/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.81663 | 0.84125 | -0.0246 [-0.0813, 0.0275] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.0587 | 2.1315 | -0.0728 [-0.2186, 0.0686] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.8874 | 0.87704 | 0.0104 [-0.0011, 0.023] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.61178 | 0.60387 | 0.0079 [-0.0139, 0.0292] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.6463 | 2.5682 | 0.0781 [-0.0224, 0.1805] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.4949 | 6.3389 | 0.1559 [-0.1037, 0.4198] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 0.97707 | 0.93572 | 0.0414 [0.0007, 0.0834] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.2999 | 2.1892 | 0.1107 [0.0126, 0.2118] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.6196 | 5.3866 | 0.233 [-0.0142, 0.495] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.51908 | 0.53376 | -0.0147 [-0.0289, -0.0001] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.44746 | 0.43221 | 0.0153 [0.0011, 0.0303] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.1008 | 3.257 | -0.1561 [-0.3389, 0.0111] | False | 3558/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.005029 | 0.0053196 | -0.0003 [-0.0006, 0.0] | False | 3532/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.027368 | 0.028759 | -0.0014 [-0.0026, -0.0002] | True | 3532/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.82528 | 0.84705 | -0.0218 [-0.0745, 0.0308] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.0732 | 2.1456 | -0.0724 [-0.2, 0.0644] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.88468 | 0.87677 | 0.0079 [-0.0033, 0.0201] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.60905 | 0.60087 | 0.0082 [-0.0133, 0.0293] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
