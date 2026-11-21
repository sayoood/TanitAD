# refcv8 step-5,000 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **NOT PASS** (-; 2 gate reason(s), not the wrapper clause alone). **PROVISIONAL: G0 is not PASS as coded nor under the registered amendments (2 gate reason(s), not the wrapper clause alone): per SPEC_REFCV8 section 5 no refcv8 battery number counts as bar evidence**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 5,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0735 [0.0509, 0.0987] n 4754/139 ep; floor 0.0014074742794036865 |
| &nbsp;&nbsp;seed 1 | delta 0.0749 [0.0511, 0.1001] n 4754/139 ep; floor 0.0014074742794036865 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.5563 [-0.8477, -0.2767] n 3668/139 ep; floor 0.024900197982788086 |
| &nbsp;&nbsp;seed 1 | delta -0.5314 [-0.8296, -0.2411] n 3668/139 ep; floor 0.024900197982788086 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.36209 | 0.27842 | 0.0837 [0.0608, 0.1092] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.74504 | 0.5995 | 0.1455 [0.0951, 0.1999] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.33033 | 0.25879 | 0.0715 [0.0482, 0.0962] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.30366 | 0.22603 | 0.0776 [0.056, 0.102] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.6302 | 0.49906 | 0.1311 [0.0832, 0.1821] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81526 | 0.85339 | -0.0381 [-0.0516, -0.0246] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.43034 | 0.37153 | 0.0588 [0.0299, 0.0888] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.7187 | 1.4181 | 0.3007 [0.2043, 0.4116] | True | 4479/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.004607 | 0.0037054 | 0.0009 [0.0005, 0.0014] | True | 4447/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.023991 | 0.018931 | 0.0051 [0.0034, 0.0069] | True | 4447/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.13319 | 0.10301 | 0.0302 [0.0214, 0.0393] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.26503 | 0.20714 | 0.0579 [0.0399, 0.0772] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.9411 | 0.96172 | -0.0206 [-0.0301, -0.012] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.81279 | 0.83046 | -0.0177 [-0.0338, -0.0011] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.36349 | 0.27902 | 0.0845 [0.0606, 0.1106] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.74746 | 0.60077 | 0.1467 [0.0955, 0.2025] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.33137 | 0.25954 | 0.0718 [0.0476, 0.0975] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.30462 | 0.22665 | 0.078 [0.0553, 0.1034] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.6319 | 0.50021 | 0.1317 [0.0831, 0.1849] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81505 | 0.85191 | -0.0369 [-0.0509, -0.023] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.43178 | 0.37231 | 0.0595 [0.0298, 0.0901] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.6951 | 1.3677 | 0.3274 [0.2323, 0.4322] | True | 4478/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0046328 | 0.003544 | 0.0011 [0.0007, 0.0015] | True | 4445/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.024305 | 0.018645 | 0.0057 [0.004, 0.0075] | True | 4445/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.13445 | 0.10306 | 0.0314 [0.0225, 0.0408] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.26753 | 0.20736 | 0.0602 [0.0419, 0.0802] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.94005 | 0.96214 | -0.0221 [-0.0317, -0.0128] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.81258 | 0.83572 | -0.0231 [-0.0402, -0.0057] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.9915 | 2.5675 | 0.4241 [0.2338, 0.626] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.2692 | 6.3328 | 0.9364 [0.4675, 1.4185] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.065 | 0.93285 | 0.1321 [0.0581, 0.2056] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.5424 | 2.19 | 0.3524 [0.1611, 0.5434] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.1345 | 5.3853 | 0.7492 [0.2934, 1.1985] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.50141 | 0.5368 | -0.0354 [-0.0531, -0.0174] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.47437 | 0.43063 | 0.0437 [0.018, 0.0691] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.7063 | 2.9881 | 0.7182 [0.4712, 0.9825] | True | 3483/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0061104 | 0.0050742 | 0.001 [0.0007, 0.0014] | True | 3464/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.03383 | 0.028593 | 0.0052 [0.0035, 0.0071] | True | 3464/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 1.005 | 0.84125 | 0.1637 [0.1013, 0.2312] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.5031 | 2.1315 | 0.3716 [0.1983, 0.5523] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.84515 | 0.87704 | -0.0319 [-0.0461, -0.0186] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.57306 | 0.60387 | -0.0308 [-0.0648, 0.0027] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 3.0164 | 2.5682 | 0.4482 [0.2486, 0.6615] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.3239 | 6.3389 | 0.985 [0.4987, 1.493] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0761 | 0.93572 | 0.1403 [0.063, 0.2184] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.5652 | 2.1892 | 0.376 [0.181, 0.5758] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.1781 | 5.3866 | 0.7915 [0.322, 1.2648] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.50082 | 0.53376 | -0.0329 [-0.0516, -0.0153] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.47766 | 0.43221 | 0.0455 [0.0193, 0.0709] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.7155 | 2.972 | 0.7434 [0.5022, 1.001] | True | 3483/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0060417 | 0.0049962 | 0.001 [0.0007, 0.0014] | True | 3463/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.033885 | 0.028511 | 0.0054 [0.0037, 0.0072] | True | 3463/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 1.0134 | 0.84705 | 0.1663 [0.1029, 0.2341] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.5267 | 2.1456 | 0.3811 [0.2092, 0.5605] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.84351 | 0.87677 | -0.0333 [-0.0477, -0.0202] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.57879 | 0.60087 | -0.0221 [-0.0552, 0.0112] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
