# refcv8 step-15,000 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **standalone**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **PASS** (A7 + W-R8-1; wrapper clause PASS under the fixed probe (max rel 2.2110427122060196e-06)). G0 qualifies the instrument.

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 15,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: standalone
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0452 [0.0275, 0.0625] n 4754/139 ep; floor 0.0004132091999053955 |
| &nbsp;&nbsp;seed 1 | delta 0.0448 [0.0276, 0.0614] n 4754/139 ep; floor 0.0004132091999053955 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.643 [-0.9576, -0.3396] n 3668/139 ep; floor 0.004254341125488281 |
| &nbsp;&nbsp;seed 1 | delta -0.6473 [-0.9657, -0.3405] n 3668/139 ep; floor 0.004254341125488281 |
| BAR-R8-NR | {"ADE": "FAIL: S2:ade_m, S6:ade_m", "LONGITUDINAL": "FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m", "LATERAL": "FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid"} |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] | 0.33383 | 0.27842 | 0.0554 [0.0401, 0.0713] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] | 0.6975 | 0.5995 | 0.098 [0.0619, 0.1337] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] | 0.30488 | 0.25879 | 0.0461 [0.0294, 0.0623] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] | 0.279 | 0.22603 | 0.053 [0.0377, 0.0688] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] | 0.58894 | 0.49906 | 0.0899 [0.0559, 0.1243] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.81963 | 0.85339 | -0.0338 [-0.0468, -0.0209] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] | 0.37314 | 0.37153 | 0.0016 [-0.0174, 0.0203] | False | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] | 1.4555 | 1.3294 | 0.1261 [0.0677, 0.1889] | True | 4495/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] | 0.0040732 | 0.0035896 | 0.0005 [0.0002, 0.0008] | True | 4454/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.020748 | 0.018801 | 0.0019 [0.0008, 0.0031] | True | 4454/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] | 0.1164 | 0.10301 | 0.0134 [0.0068, 0.02] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] | 0.23485 | 0.20714 | 0.0277 [0.0131, 0.0427] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] | 0.95435 | 0.96172 | -0.0074 [-0.0128, -0.0021] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.83193 | 0.83046 | 0.0015 [-0.0129, 0.0164] | False | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] | 0.33342 | 0.27902 | 0.0544 [0.0392, 0.0689] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] | 0.69643 | 0.60077 | 0.0957 [0.0617, 0.1296] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] | 0.30484 | 0.25954 | 0.0453 [0.0292, 0.0608] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] | 0.27911 | 0.22665 | 0.0525 [0.0368, 0.0675] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] | 0.58901 | 0.50021 | 0.0888 [0.0548, 0.1214] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.82126 | 0.85191 | -0.0307 [-0.043, -0.0186] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] | 0.37362 | 0.37231 | 0.0013 [-0.0168, 0.0198] | False | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] | 1.4618 | 1.3209 | 0.1409 [0.079, 0.2076] | True | 4496/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] | 0.0041537 | 0.0034882 | 0.0007 [0.0003, 0.0011] | True | 4457/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.02087 | 0.018567 | 0.0023 [0.0011, 0.0036] | True | 4457/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] | 0.1165 | 0.10306 | 0.0134 [0.007, 0.0197] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] | 0.23483 | 0.20736 | 0.0275 [0.0126, 0.0418] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] | 0.95414 | 0.96214 | -0.008 [-0.0132, -0.0032] | True | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.8313 | 0.83572 | -0.0044 [-0.0181, 0.0099] | False | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] | 2.9048 | 2.5675 | 0.3374 [0.152, 0.5167] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] | 7.146 | 6.3328 | 0.8132 [0.3226, 1.2961] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] | 1.0569 | 0.93285 | 0.124 [0.0437, 0.2041] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] | 2.4654 | 2.19 | 0.2753 [0.0974, 0.4515] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] | 6.0279 | 5.3853 | 0.6425 [0.1573, 1.1084] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.48519 | 0.5368 | -0.0516 [-0.0738, -0.0289] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] | 0.48171 | 0.43063 | 0.0511 [0.02, 0.0822] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] | 3.6987 | 3.187 | 0.5117 [0.3, 0.7373] | True | 3506/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] | 0.0055517 | 0.0049831 | 0.0006 [0.0003, 0.0009] | True | 3483/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] | 0.031192 | 0.028522 | 0.0027 [0.0013, 0.0041] | True | 3483/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] | 0.98142 | 0.84125 | 0.1402 [0.0731, 0.2067] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] | 2.4822 | 2.1315 | 0.3507 [0.1726, 0.5245] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] | 0.83642 | 0.87704 | -0.0406 [-0.0566, -0.0251] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] | 0.56979 | 0.60387 | -0.0341 [-0.0734, 0.003] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] | 2.9006 | 2.5682 | 0.3324 [0.1466, 0.5114] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] | 7.1463 | 6.3389 | 0.8074 [0.3218, 1.2938] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] | 1.0544 | 0.93572 | 0.1186 [0.0403, 0.1946] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] | 2.4681 | 2.1892 | 0.2789 [0.1005, 0.4487] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] | 6.051 | 5.3866 | 0.6644 [0.1939, 1.1247] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] | 0.48619 | 0.53376 | -0.0476 [-0.0694, -0.0253] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] | 0.48135 | 0.43221 | 0.0491 [0.019, 0.0796] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] | 3.6791 | 3.1723 | 0.5068 [0.2941, 0.7326] | True | 3507/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] | 0.0055704 | 0.0049188 | 0.0007 [0.0003, 0.001] | True | 3485/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] | 0.031163 | 0.028428 | 0.0027 [0.0013, 0.0042] | True | 3485/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] | 0.97808 | 0.84705 | 0.131 [0.0633, 0.198] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] | 2.4721 | 2.1456 | 0.3265 [0.1525, 0.4942] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] | 0.83779 | 0.87677 | -0.039 [-0.0555, -0.0233] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] | 0.56325 | 0.60087 | -0.0376 [-0.0773, 0.0003] | False | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.

## Controls (GT and model-free arms bit-identical to the banked refcv7 panel)

* S2_s0: all bit-identical = True; status OK
* S2_s1: all bit-identical = True; status OK

Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
