# refcv8 step-48,852 EMA weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **NOT AVAILABLE** (amendment None), M1 detected: None; under the registered amendments: **n/a** (-). **no g0.json**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* FINAL checkpoint (step 48,852 of 48,852): the pre-registered bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0192 [0.0022, 0.0369] n 4754/139 ep; floor 0.0007080733776092529 |
| &nbsp;&nbsp;seed 1 | delta 0.0199 [0.0038, 0.0368] n 4754/139 ep; floor 0.0007080733776092529 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.5498 [-0.8872, -0.2035] n 3668/139 ep; floor 0.019716262817382812 |
| &nbsp;&nbsp;seed 1 | delta -0.5301 [-0.8641, -0.1934] n 3668/139 ep; floor 0.019716262817382812 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.30777 | 0.27842 | 0.0293 [0.0163, 0.0423] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.66966 | 0.5995 | 0.0702 [0.0391, 0.1002] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.29675 | 0.25879 | 0.038 [0.023, 0.0529] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.25844 | 0.22603 | 0.0324 [0.0192, 0.0453] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.57342 | 0.49906 | 0.0744 [0.043, 0.1052] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81863 | 0.85339 | -0.0348 [-0.0468, -0.0227] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.41495 | 0.37153 | 0.0434 [0.0237, 0.0635] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.5013 | 1.4528 | 0.0485 [0.0023, 0.0948] | True | 4528/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0040163 | 0.0038652 | 0.0002 [-0.0001, 0.0004] | False | 4492/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.019902 | 0.018962 | 0.0009 [-0.0, 0.0019] | False | 4492/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.10725 | 0.10301 | 0.0042 [-0.001, 0.0088] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.21996 | 0.20714 | 0.0128 [0.002, 0.023] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95751 | 0.96172 | -0.0042 [-0.0088, 0.0002] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.809 | 0.83046 | -0.0215 [-0.0372, -0.0057] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.30848 | 0.27902 | 0.0295 [0.0172, 0.0416] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.66952 | 0.60077 | 0.0688 [0.0394, 0.0975] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.29533 | 0.25954 | 0.0358 [0.0214, 0.0499] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.25815 | 0.22665 | 0.0315 [0.0188, 0.0433] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.57067 | 0.50021 | 0.0705 [0.0401, 0.0991] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.82283 | 0.85191 | -0.0291 [-0.04, -0.0181] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.41211 | 0.37231 | 0.0398 [0.0209, 0.0583] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.5169 | 1.442 | 0.0749 [0.0305, 0.124] | True | 4526/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.004056 | 0.0037121 | 0.0003 [0.0001, 0.0006] | True | 4493/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.020142 | 0.018671 | 0.0015 [0.0005, 0.0026] | True | 4493/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.10854 | 0.10306 | 0.0055 [0.0003, 0.0104] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.22233 | 0.20736 | 0.015 [0.0039, 0.0254] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95562 | 0.96214 | -0.0065 [-0.0118, -0.0015] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.80879 | 0.83572 | -0.0269 [-0.0422, -0.0112] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.9981 | 2.5675 | 0.4306 [0.2734, 0.5938] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.4718 | 6.3328 | 1.139 [0.7191, 1.574] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.1156 | 0.93285 | 0.1827 [0.1154, 0.2526] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.5902 | 2.19 | 0.4001 [0.246, 0.5565] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.4252 | 5.3853 | 1.0399 [0.6228, 1.4646] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.48237 | 0.5368 | -0.0544 [-0.0736, -0.0353] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.5087 | 0.43063 | 0.0781 [0.051, 0.1072] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.6296 | 3.2774 | 0.3522 [0.1189, 0.6246] | True | 3557/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0058257 | 0.0054425 | 0.0004 [-0.0, 0.0008] | False | 3532/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.031543 | 0.028953 | 0.0026 [0.0007, 0.0047] | True | 3532/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.94479 | 0.84125 | 0.1035 [0.0547, 0.1567] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.3983 | 2.1315 | 0.2669 [0.123, 0.4167] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.86478 | 0.87704 | -0.0123 [-0.028, 0.0038] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.56652 | 0.60387 | -0.0374 [-0.0678, -0.0068] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 3.0178 | 2.5682 | 0.4495 [0.2986, 0.6034] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.5271 | 6.3389 | 1.1882 [0.7912, 1.6188] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.1195 | 0.93572 | 0.1838 [0.1179, 0.2525] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.6022 | 2.1892 | 0.413 [0.262, 0.5656] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.4696 | 5.3866 | 1.083 [0.6792, 1.4984] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.48446 | 0.53376 | -0.0493 [-0.0688, -0.0304] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.50758 | 0.43221 | 0.0754 [0.0494, 0.1037] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.716 | 3.2638 | 0.4521 [0.1942, 0.7459] | True | 3559/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0058689 | 0.0053472 | 0.0005 [0.0001, 0.001] | True | 3533/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.031698 | 0.02884 | 0.0029 [0.0009, 0.0051] | True | 3533/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.95776 | 0.84705 | 0.1107 [0.053, 0.171] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.426 | 2.1456 | 0.2804 [0.1233, 0.4475] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.8626 | 0.87677 | -0.0142 [-0.0312, 0.0014] | False | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.56189 | 0.60087 | -0.039 [-0.0699, -0.009] | True | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
