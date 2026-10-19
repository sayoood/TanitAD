# refcv8 step-15,000 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 15,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0003 [-0.015, 0.0163] n 4754/139 ep; floor 0.00019440054893493652 |
| &nbsp;&nbsp;seed 1 | delta 0.0001 [-0.0148, 0.0161] n 4754/139 ep; floor 0.00019440054893493652 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.9159 [-1.2255, -0.6377] n 3668/139 ep; floor 0.006027698516845703 |
| &nbsp;&nbsp;seed 1 | delta -0.9219 [-1.2306, -0.6417] n 3668/139 ep; floor 0.006027698516845703 |
| BAR-R8-NR | {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.2889 | 0.27842 | 0.0105 [-0.0016, 0.0224] | False | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.61853 | 0.5995 | 0.019 [-0.0088, 0.0471] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.26642 | 0.25879 | 0.0076 [-0.0054, 0.0205] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.23577 | 0.22603 | 0.0097 [-0.0021, 0.0213] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.51734 | 0.49906 | 0.0183 [-0.0093, 0.045] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84702 | 0.85339 | -0.0064 [-0.0163, 0.0042] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.36963 | 0.37153 | -0.0019 [-0.018, 0.0136] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4843 | 1.413 | 0.0713 [0.0277, 0.1167] | True | 4510/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0039644 | 0.0037953 | 0.0002 [-0.0001, 0.0004] | False | 4481/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019663 | 0.018936 | 0.0007 [-0.0001, 0.0016] | False | 4481/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10788 | 0.10301 | 0.0049 [-0.0001, 0.0097] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.21561 | 0.20714 | 0.0085 [-0.002, 0.0185] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95751 | 0.96172 | -0.0042 [-0.0101, 0.0017] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83256 | 0.83046 | 0.0021 [-0.0109, 0.016] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.28871 | 0.27902 | 0.0097 [-0.0019, 0.0218] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.61755 | 0.60077 | 0.0168 [-0.0107, 0.0443] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.26598 | 0.25954 | 0.0064 [-0.0068, 0.0192] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.23563 | 0.22665 | 0.009 [-0.0029, 0.0202] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.51661 | 0.50021 | 0.0164 [-0.012, 0.0422] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84976 | 0.85191 | -0.0022 [-0.0116, 0.0077] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.36912 | 0.37231 | -0.0032 [-0.0191, 0.0125] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4845 | 1.4018 | 0.0826 [0.0382, 0.1281] | True | 4511/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0039782 | 0.0036554 | 0.0003 [0.0001, 0.0006] | True | 4480/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.019694 | 0.018656 | 0.001 [0.0002, 0.0019] | True | 4480/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10804 | 0.10306 | 0.005 [-0.0001, 0.01] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.21555 | 0.20736 | 0.0082 [-0.0029, 0.0189] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95625 | 0.96214 | -0.0059 [-0.0116, -0.0004] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.83719 | 0.83572 | 0.0015 [-0.0103, 0.0137] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.632 | 2.5675 | 0.0645 [-0.074, 0.1998] | False | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.5016 | 6.3328 | 0.1688 [-0.1921, 0.5279] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 0.9559 | 0.93285 | 0.023 [-0.0356, 0.0816] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.2534 | 2.19 | 0.0634 [-0.0743, 0.1978] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.5339 | 5.3853 | 0.1485 [-0.2158, 0.5006] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.51913 | 0.5368 | -0.0177 [-0.0362, 0.0002] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.44721 | 0.43063 | 0.0166 [-0.0063, 0.0401] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.4169 | 3.2544 | 0.1624 [0.0164, 0.3155] | True | 3528/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0054761 | 0.0052488 | 0.00023 [3e-05, 0.00043] | True | 3506/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.029364 | 0.028728 | 0.0006 [-0.0003, 0.0017] | False | 3506/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.8447 | 0.84125 | 0.0035 [-0.0431, 0.0496] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.1367 | 2.1315 | 0.0052 [-0.1172, 0.1278] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.86914 | 0.87704 | -0.0079 [-0.0167, 0.0011] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.59597 | 0.60387 | -0.0079 [-0.04, 0.0237] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.6259 | 2.5682 | 0.0577 [-0.0811, 0.194] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.4904 | 6.3389 | 0.1514 [-0.2209, 0.5148] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 0.95216 | 0.93572 | 0.0164 [-0.0423, 0.075] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.2455 | 2.1892 | 0.0563 [-0.0771, 0.1933] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.5211 | 5.3866 | 0.1345 [-0.2284, 0.4915] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.52104 | 0.53376 | -0.0127 [-0.0317, 0.0064] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.44565 | 0.43221 | 0.0134 [-0.0088, 0.0362] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.4233 | 3.2339 | 0.1895 [0.0324, 0.3512] | True | 3528/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0054984 | 0.005165 | 0.0003 [0.0001, 0.0006] | True | 3504/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.029465 | 0.028602 | 0.0009 [-0.0002, 0.0018] | False | 3504/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.84722 | 0.84705 | 0.0002 [-0.049, 0.0506] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.1466 | 2.1456 | 0.001 [-0.1233, 0.1298] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.86805 | 0.87677 | -0.0087 [-0.0177, 0.0003] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.59188 | 0.60087 | -0.009 [-0.0414, 0.0218] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
