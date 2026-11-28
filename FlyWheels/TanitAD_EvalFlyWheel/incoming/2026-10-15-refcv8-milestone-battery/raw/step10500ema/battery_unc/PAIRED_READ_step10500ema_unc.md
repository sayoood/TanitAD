# refcv8 step-10,500 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 10,500 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta -0.0039 [-0.0184, 0.0105] n 4754/139 ep; floor 0.0013476908206939697 |
| &nbsp;&nbsp;seed 1 | delta -0.0026 [-0.0168, 0.0121] n 4754/139 ep; floor 0.0013476908206939697 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.9957 [-1.2924, -0.7263] n 3668/139 ep; floor 0.03044891357421875 |
| &nbsp;&nbsp;seed 1 | delta -0.9652 [-1.2585, -0.695] n 3668/139 ep; floor 0.03044891357421875 |
| BAR-R8-NR | {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.28466 | 0.27842 | 0.0062 [-0.0047, 0.0172] | False | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.60751 | 0.5995 | 0.008 [-0.0179, 0.0336] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.26162 | 0.25879 | 0.0028 [-0.0092, 0.0143] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.23292 | 0.22603 | 0.0069 [-0.0037, 0.0172] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.50887 | 0.49906 | 0.0098 [-0.0153, 0.0337] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.85233 | 0.85339 | -0.0011 [-0.0094, 0.0079] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.36435 | 0.37153 | -0.0072 [-0.0226, 0.0074] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4557 | 1.413 | 0.0428 [-0.0006, 0.0868] | False | 4510/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0037984 | 0.0037996 | -0.0 [-0.0002, 0.0002] | False | 4481/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019252 | 0.01894 | 0.0003 [-0.0005, 0.0011] | False | 4481/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10531 | 0.10301 | 0.0023 [-0.0028, 0.0072] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.21061 | 0.20714 | 0.0035 [-0.0074, 0.0142] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95898 | 0.96172 | -0.0027 [-0.0076, 0.0021] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83572 | 0.83046 | 0.0053 [-0.0074, 0.0181] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.28601 | 0.27902 | 0.007 [-0.004, 0.0185] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.61019 | 0.60077 | 0.0094 [-0.0161, 0.0359] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.2636 | 0.25954 | 0.0041 [-0.0075, 0.0155] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.23461 | 0.22665 | 0.008 [-0.0026, 0.0183] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.51275 | 0.50021 | 0.0125 [-0.0118, 0.0367] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.85339 | 0.85191 | 0.0015 [-0.0068, 0.0099] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.36666 | 0.37231 | -0.0057 [-0.0203, 0.0086] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4526 | 1.4025 | 0.05 [0.0002, 0.0985] | True | 4510/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0037827 | 0.0036635 | 0.0001 [-0.0001, 0.0004] | False | 4479/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.019152 | 0.018673 | 0.0005 [-0.0004, 0.0014] | False | 4479/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10535 | 0.10306 | 0.0023 [-0.0029, 0.0075] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.21046 | 0.20736 | 0.0031 [-0.0084, 0.0144] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95772 | 0.96214 | -0.0044 [-0.0097, 0.0008] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.83614 | 0.83572 | 0.0004 [-0.0103, 0.0116] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.5522 | 2.5675 | -0.0153 [-0.1314, 0.1022] | False | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.2845 | 6.3328 | -0.0483 [-0.3413, 0.2588] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 0.91301 | 0.93285 | -0.0198 [-0.0657, 0.0276] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.1745 | 2.19 | -0.0155 [-0.1266, 0.098] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.3136 | 5.3853 | -0.0717 [-0.351, 0.2128] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.53449 | 0.5368 | -0.0023 [-0.0169, 0.012] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.42686 | 0.43063 | -0.0038 [-0.0213, 0.0133] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.3887 | 3.2564 | 0.1323 [-0.0417, 0.306] | False | 3525/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.005386 | 0.0052301 | 0.0002 [-0.0001, 0.0004] | False | 3509/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.029302 | 0.028697 | 0.0006 [-0.0005, 0.0017] | False | 3509/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.83475 | 0.84125 | -0.0065 [-0.0568, 0.0425] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.1213 | 2.1315 | -0.0102 [-0.1407, 0.1154] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.86832 | 0.87704 | -0.0087 [-0.018, 0.0] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.60033 | 0.60387 | -0.0035 [-0.0303, 0.0242] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.5826 | 2.5682 | 0.0144 [-0.0959, 0.1294] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.3638 | 6.3389 | 0.0249 [-0.2577, 0.3214] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 0.92502 | 0.93572 | -0.0107 [-0.055, 0.0358] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.203 | 2.1892 | 0.0138 [-0.0947, 0.1247] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.3872 | 5.3866 | 0.0006 [-0.2758, 0.2782] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.52826 | 0.53376 | -0.0055 [-0.0195, 0.0083] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.42943 | 0.43221 | -0.0028 [-0.0191, 0.0134] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.4348 | 3.2526 | 0.1822 [0.0011, 0.3508] | True | 3527/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0053751 | 0.0051549 | 0.0002 [-0.0, 0.0005] | False | 3507/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.029258 | 0.028628 | 0.0006 [-0.0005, 0.0018] | False | 3507/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.8393 | 0.84705 | -0.0078 [-0.0612, 0.0436] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.1362 | 2.1456 | -0.0094 [-0.1439, 0.1221] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.86859 | 0.87677 | -0.0082 [-0.0171, 0.0011] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.59951 | 0.60087 | -0.0014 [-0.0268, 0.0237] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
