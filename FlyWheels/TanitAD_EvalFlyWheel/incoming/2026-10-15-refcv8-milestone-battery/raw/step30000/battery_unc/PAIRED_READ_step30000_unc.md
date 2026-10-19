# refcv8 step-30,000 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **NOT PASS** (-; fixed probe reads FAIL and no qualifying W-R8-2 float64 re-check for this step (run raw/step20000/wrapper_fp64/wrapper_fp64_probe.py)). **PROVISIONAL: G0 is not PASS as coded nor under the registered amendments (fixed probe reads FAIL and no qualifying W-R8-2 float64 re-check for this step (run raw/step20000/wrapper_fp64/wrapper_fp64_probe.py)): per SPEC_REFCV8 section 5 no refcv8 battery number counts as bar evidence**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 30,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0242 [0.0078, 0.0403] n 4754/139 ep; floor 0.00015395879745483398 |
| &nbsp;&nbsp;seed 1 | delta 0.0244 [0.0082, 0.0408] n 4754/139 ep; floor 0.00015395879745483398 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.6289 [-0.9539, -0.3086] n 3668/139 ep; floor 0.0031118392944335938 |
| &nbsp;&nbsp;seed 1 | delta -0.6258 [-0.9564, -0.2987] n 3668/139 ep; floor 0.0031118392944335938 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.31283 | 0.27842 | 0.0344 [0.0213, 0.0473] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.67466 | 0.5995 | 0.0752 [0.044, 0.1058] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.29695 | 0.25879 | 0.0382 [0.0241, 0.0524] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.26097 | 0.22603 | 0.0349 [0.0222, 0.0476] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.57384 | 0.49906 | 0.0748 [0.0444, 0.1049] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.815 | 0.85339 | -0.0384 [-0.0516, -0.0261] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.40605 | 0.37153 | 0.0345 [0.0163, 0.0528] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4761 | 1.3734 | 0.1027 [0.0482, 0.1654] | True | 4511/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0041062 | 0.003721 | 0.0004 [0.0001, 0.0008] | True | 4472/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.020519 | 0.018875 | 0.0016 [0.0005, 0.0029] | True | 4472/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.11147 | 0.10301 | 0.0085 [0.0029, 0.0141] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.22603 | 0.20714 | 0.0189 [0.0066, 0.0315] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95456 | 0.96172 | -0.0072 [-0.0124, -0.0021] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.80879 | 0.83046 | -0.0217 [-0.0385, -0.0046] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.31299 | 0.27902 | 0.034 [0.0208, 0.0472] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.67439 | 0.60077 | 0.0736 [0.0421, 0.1035] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.29667 | 0.25954 | 0.0371 [0.0229, 0.0513] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.26028 | 0.22665 | 0.0336 [0.0208, 0.0464] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.57181 | 0.50021 | 0.0716 [0.0413, 0.1026] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81789 | 0.85191 | -0.034 [-0.0472, -0.0213] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.40569 | 0.37231 | 0.0334 [0.0157, 0.0515] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4407 | 1.3231 | 0.1176 [0.0605, 0.1799] | True | 4509/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0041459 | 0.0035917 | 0.0006 [0.0002, 0.0009] | True | 4473/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.020474 | 0.018608 | 0.0019 [0.0007, 0.0031] | True | 4473/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.11277 | 0.10306 | 0.0097 [0.0043, 0.0153] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.22841 | 0.20736 | 0.0211 [0.0089, 0.0336] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95456 | 0.96214 | -0.0076 [-0.0126, -0.0027] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.81069 | 0.83572 | -0.025 [-0.042, -0.0084] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.9189 | 2.5675 | 0.3514 [0.1817, 0.529] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.2282 | 6.3328 | 0.8954 [0.438, 1.3921] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0701 | 0.93285 | 0.1373 [0.0648, 0.2156] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.5095 | 2.19 | 0.3195 [0.1571, 0.4966] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.1805 | 5.3853 | 0.7952 [0.3578, 1.2799] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.48991 | 0.5368 | -0.0469 [-0.0684, -0.0266] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.49436 | 0.43063 | 0.0637 [0.0353, 0.0947] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.6618 | 3.211 | 0.4507 [0.2334, 0.6677] | True | 3532/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0057598 | 0.0051587 | 0.0006 [0.0003, 0.001] | True | 3507/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.031646 | 0.028743 | 0.0029 [0.0015, 0.0044] | True | 3507/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.936 | 0.84125 | 0.0947 [0.0328, 0.1594] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.3791 | 2.1315 | 0.2476 [0.0931, 0.4068] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.8596 | 0.87704 | -0.0174 [-0.0344, -0.0011] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.57688 | 0.60387 | -0.027 [-0.0621, 0.0049] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.922 | 2.5682 | 0.3538 [0.1854, 0.5314] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.2376 | 6.3389 | 0.8986 [0.4584, 1.3821] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0749 | 0.93572 | 0.1392 [0.0689, 0.2149] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.518 | 2.1892 | 0.3288 [0.1664, 0.5038] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.2088 | 5.3866 | 0.8222 [0.3933, 1.2886] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.48705 | 0.53376 | -0.0467 [-0.0678, -0.0264] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.49377 | 0.43221 | 0.0616 [0.0344, 0.0916] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.6743 | 3.1901 | 0.4842 [0.2469, 0.738] | True | 3535/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.005728 | 0.0050861 | 0.0006 [0.0002, 0.0011] | True | 3509/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.03117 | 0.028647 | 0.0025 [0.0011, 0.004] | True | 3509/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.92609 | 0.84705 | 0.079 [0.0164, 0.143] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.3455 | 2.1456 | 0.1999 [0.0401, 0.3645] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.85932 | 0.87677 | -0.0174 [-0.0327, -0.0011] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.57252 | 0.60087 | -0.0284 [-0.0626, 0.0025] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
