# refcv8 step-20,000 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **PASS** (A7 + W-R8-1 + W-R8-2; fixed probe FAIL by fp32 noise (max rel 1.0646966230152142e-05); float64 re-check: every term <= 1e-9 (max 1.9831653266307828e-14), W1/W2 detected). G0 qualifies the instrument.

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 20,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0146 [0.0003, 0.0293] n 4754/139 ep; floor 0.0010942816734313965 |
| &nbsp;&nbsp;seed 1 | delta 0.0135 [-0.0006, 0.0283] n 4754/139 ep; floor 0.0010942816734313965 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.8371 [-1.1368, -0.56] n 3668/139 ep; floor 0.002878904342651367 |
| &nbsp;&nbsp;seed 1 | delta -0.8342 [-1.1203, -0.557] n 3668/139 ep; floor 0.002878904342651367 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.30324 | 0.27842 | 0.0248 [0.0139, 0.0359] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.64035 | 0.5995 | 0.0408 [0.015, 0.0664] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.27574 | 0.25879 | 0.017 [0.0058, 0.0269] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.2472 | 0.22603 | 0.0212 [0.0106, 0.0312] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.53142 | 0.49906 | 0.0324 [0.0071, 0.0563] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.83914 | 0.85339 | -0.0143 [-0.0233, -0.005] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.36695 | 0.37153 | -0.0046 [-0.0199, 0.0087] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.494 | 1.41 | 0.084 [0.0404, 0.1291] | True | 4507/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0038524 | 0.0036887 | 0.0002 [-0.0, 0.0004] | False | 4470/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019962 | 0.018871 | 0.0011 [0.0002, 0.002] | True | 4470/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.113 | 0.10301 | 0.01 [0.0051, 0.0152] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.22714 | 0.20714 | 0.02 [0.0091, 0.0316] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95772 | 0.96172 | -0.004 [-0.0088, 0.0008] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83614 | 0.83046 | 0.0057 [-0.008, 0.0204] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.30215 | 0.27902 | 0.0231 [0.0125, 0.034] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.63723 | 0.60077 | 0.0365 [0.0114, 0.0613] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.27467 | 0.25954 | 0.0151 [0.0042, 0.0252] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.24578 | 0.22665 | 0.0191 [0.0091, 0.0291] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.52753 | 0.50021 | 0.0273 [0.0027, 0.051] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84166 | 0.85191 | -0.0103 [-0.0188, -0.0012] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.36706 | 0.37231 | -0.0053 [-0.0196, 0.0077] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4966 | 1.3993 | 0.0973 [0.0512, 0.1486] | True | 4508/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0038197 | 0.0035509 | 0.0003 [0.0001, 0.0005] | True | 4468/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.019996 | 0.01861 | 0.0014 [0.0005, 0.0024] | True | 4468/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.11315 | 0.10306 | 0.0101 [0.0047, 0.0156] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.22764 | 0.20736 | 0.0203 [0.0084, 0.0326] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95709 | 0.96214 | -0.005 [-0.0101, 0.0] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.83782 | 0.83572 | 0.0021 [-0.0109, 0.0162] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.7108 | 2.5675 | 0.1433 [0.0166, 0.2685] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.6585 | 6.3328 | 0.3257 [-0.0177, 0.6535] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 0.96777 | 0.93285 | 0.0349 [-0.0166, 0.085] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.2941 | 2.19 | 0.1041 [-0.0166, 0.2212] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.6228 | 5.3853 | 0.2375 [-0.0913, 0.547] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.51409 | 0.5368 | -0.0227 [-0.0378, -0.0064] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.44569 | 0.43063 | 0.0151 [-0.0057, 0.0357] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.5019 | 3.1207 | 0.3812 [0.1915, 0.5967] | True | 3518/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0054587 | 0.0050827 | 0.0004 [0.0001, 0.0007] | True | 3499/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.030782 | 0.028651 | 0.0021 [0.0008, 0.0037] | True | 3499/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.93101 | 0.84125 | 0.0898 [0.0351, 0.1448] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.3433 | 2.1315 | 0.2118 [0.0615, 0.3585] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.86123 | 0.87704 | -0.0158 [-0.0273, -0.0038] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.5976 | 0.60387 | -0.0063 [-0.0368, 0.0231] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.7137 | 2.5682 | 0.1455 [0.0247, 0.2652] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.669 | 6.3389 | 0.3301 [-0.0082, 0.6547] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 0.96486 | 0.93572 | 0.0291 [-0.0222, 0.0757] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.284 | 2.1892 | 0.0948 [-0.0204, 0.2069] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.5921 | 5.3866 | 0.2055 [-0.1041, 0.5021] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.5135 | 0.53376 | -0.0203 [-0.0352, -0.0051] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.44377 | 0.43221 | 0.0116 [-0.0087, 0.0309] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.5196 | 3.0943 | 0.4252 [0.1747, 0.7019] | True | 3520/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0054077 | 0.0049786 | 0.0004 [0.0001, 0.0008] | True | 3500/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.030875 | 0.028499 | 0.0024 [0.0007, 0.0043] | True | 3500/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.94597 | 0.84705 | 0.0989 [0.0448, 0.1554] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.3885 | 2.1456 | 0.2429 [0.0924, 0.3912] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.86341 | 0.87677 | -0.0134 [-0.0251, -0.0008] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.59542 | 0.60087 | -0.0055 [-0.0347, 0.0234] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
