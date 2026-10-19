# refcv8 step-20,000 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 20,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta -0.002 [-0.0168, 0.0131] n 4754/139 ep; floor 0.0004665851593017578 |
| &nbsp;&nbsp;seed 1 | delta -0.0025 [-0.0169, 0.0125] n 4754/139 ep; floor 0.0004665851593017578 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.8957 [-1.2084, -0.6024] n 3668/139 ep; floor 0.007934331893920898 |
| &nbsp;&nbsp;seed 1 | delta -0.8878 [-1.1999, -0.5935] n 3668/139 ep; floor 0.007934331893920898 |
| BAR-R8-NR | {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.28661 | 0.27842 | 0.0082 [-0.0042, 0.0202] | False | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.61342 | 0.5995 | 0.0139 [-0.0145, 0.0408] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.26612 | 0.25879 | 0.0073 [-0.0062, 0.02] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.23575 | 0.22603 | 0.0097 [-0.0025, 0.0208] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.51629 | 0.49906 | 0.0172 [-0.0113, 0.0429] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84308 | 0.85339 | -0.0103 [-0.0206, 0.0007] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.36499 | 0.37153 | -0.0065 [-0.0231, 0.0089] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4545 | 1.4122 | 0.0424 [0.0009, 0.0849] | True | 4514/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0038929 | 0.0038028 | 0.0001 [-0.0002, 0.0003] | False | 4484/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019176 | 0.018933 | 0.0002 [-0.0005, 0.0011] | False | 4484/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10457 | 0.10301 | 0.0016 [-0.003, 0.006] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.20905 | 0.20714 | 0.0019 [-0.0076, 0.0111] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.9594 | 0.96172 | -0.0023 [-0.0069, 0.0021] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83172 | 0.83046 | 0.0013 [-0.012, 0.0155] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.28614 | 0.27902 | 0.0071 [-0.005, 0.0188] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.61217 | 0.60077 | 0.0114 [-0.0159, 0.0387] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.26573 | 0.25954 | 0.0062 [-0.0068, 0.019] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.23552 | 0.22665 | 0.0089 [-0.0032, 0.0201] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.51544 | 0.50021 | 0.0152 [-0.0124, 0.0413] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.84623 | 0.85191 | -0.0057 [-0.0163, 0.0052] | False | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.36423 | 0.37231 | -0.0081 [-0.0241, 0.0073] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4616 | 1.4016 | 0.06 [0.0199, 0.1026] | True | 4514/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0039038 | 0.0036535 | 0.00025 [2e-05, 0.0005] | True | 4483/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.019194 | 0.018644 | 0.0005 [-0.0003, 0.0014] | False | 4483/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10504 | 0.10306 | 0.002 [-0.0028, 0.0067] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.20987 | 0.20736 | 0.0025 [-0.008, 0.0125] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95835 | 0.96214 | -0.0038 [-0.0084, 0.0004] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.83782 | 0.83572 | 0.0021 [-0.0105, 0.0152] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.6521 | 2.5675 | 0.0847 [-0.0568, 0.225] | False | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 6.5755 | 6.3328 | 0.2427 [-0.1404, 0.6219] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 0.97512 | 0.93285 | 0.0423 [-0.0209, 0.1047] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.2827 | 2.19 | 0.0926 [-0.0475, 0.2321] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 5.6388 | 5.3853 | 0.2534 [-0.1347, 0.6342] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.51354 | 0.5368 | -0.0233 [-0.0428, -0.0035] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.45567 | 0.43063 | 0.025 [-0.0001, 0.0512] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.3225 | 3.2503 | 0.0722 [-0.0628, 0.2124] | False | 3536/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0053987 | 0.0052469 | 0.0002 [-0.0002, 0.0005] | False | 3517/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.028862 | 0.028692 | 0.0002 [-0.0008, 0.0012] | False | 3517/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.83791 | 0.84125 | -0.0033 [-0.0432, 0.0364] | False | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.1217 | 2.1315 | -0.0098 [-0.1152, 0.0961] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.8735 | 0.87704 | -0.0035 [-0.0138, 0.0057] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.58997 | 0.60387 | -0.0139 [-0.047, 0.0175] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.6601 | 2.5682 | 0.0918 [-0.0534, 0.2344] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 6.6034 | 6.3389 | 0.2645 [-0.1334, 0.6547] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 0.97432 | 0.93572 | 0.0386 [-0.0239, 0.1017] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.2808 | 2.1892 | 0.0916 [-0.0506, 0.2294] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 5.6357 | 5.3866 | 0.2491 [-0.1346, 0.6295] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.51409 | 0.53376 | -0.0197 [-0.0399, 0.0003] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.45499 | 0.43221 | 0.0228 [-0.002, 0.0493] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.3649 | 3.2305 | 0.1344 [-0.0136, 0.2938] | False | 3536/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0053901 | 0.0051687 | 0.0002 [-0.0001, 0.0005] | False | 3514/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.029034 | 0.028584 | 0.0005 [-0.0006, 0.0016] | False | 3514/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.85092 | 0.84705 | 0.0039 [-0.0425, 0.048] | False | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.1606 | 2.1456 | 0.015 [-0.1065, 0.1347] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.87459 | 0.87677 | -0.0022 [-0.012, 0.0074] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.5916 | 0.60087 | -0.0093 [-0.0435, 0.0237] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
