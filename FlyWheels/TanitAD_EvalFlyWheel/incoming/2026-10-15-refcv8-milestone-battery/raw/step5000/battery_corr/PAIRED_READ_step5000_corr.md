# refcv8 step-5,000 raw weights: paired four-family read (refcv8 `os` vs refcv7-50,400 `os`)

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; clock: **corrected**; estimator: episode-cluster bootstrap, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run (the inference floor is reported beside it); B 2000, seed 0.
* G0 as coded: **FAIL** (amendment A7), M1 detected: True; under the registered amendments: **NOT PASS** (-; 2 gate reason(s), not the wrapper clause alone). **PROVISIONAL: G0 is not PASS as coded nor under the registered amendments (2 gate reason(s), not the wrapper clause alone): per SPEC_REFCV8 section 5 no refcv8 battery number counts as bar evidence**

* single training seed; training-seed floor at full scale unmeasured (SPEC_REFCV8 section 9)
* EARLY READ at step 5,000 of 48,852: not the pre-registered final-checkpoint bar verdict
* pose clock: corrected
* privileged input: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction); LEGAL / NAVSHUF rows are NOT part of this read

## Bars (SPEC_REFCV8_DRAFT section 5)

| bar | verdict |
|---|---|
| BAR-R8-E1 (os - ha0_ext ADE 0-2 s < 0, separated, at both seeds) | FAILED |
| &nbsp;&nbsp;seed 0 | delta 0.0622 [0.0393, 0.0874] n 4754/139 ep; floor 0.0041465163230896 |
| &nbsp;&nbsp;seed 1 | delta 0.0664 [0.0427, 0.0914] n 4754/139 ep; floor 0.0041465163230896 |
| BAR-R8-E3 (S6: os - ha0_ext ADE 0-6 s < 0, separated, at both seeds) | PASS (single training seed; training-seed floor unmeasured) |
| &nbsp;&nbsp;seed 0 | delta -0.6147 [-0.9085, -0.3331] n 3668/139 ep; floor 0.037853240966796875 |
| &nbsp;&nbsp;seed 1 | delta -0.5768 [-0.8747, -0.2896] n 3668/139 ep; floor 0.037853240966796875 |
| BAR-R8-NR | NOT PAIRABLE on this clock (refcv7's GT is on the uncorrected clock; the paired difference would compare two eval sets) |

## Rows (refcv8 minus refcv7-50,400; negative = refcv8 better for an error metric)

| family | surface | seed | metric [unit] | refcv8 `os` | refcv7 `os` (or echo `ha0_ext` where marked) | delta [95 % CI] | sep. | n win / ep |
|---|---|---|---|---|---|---|---|---|
| ADE | S2 0-2 s | 0 | ADE (mean over the surface's instants) [m] (vs echo) | 0.35579 | 0.29354 | 0.0622 [0.0393, 0.0874] | True | 4754/139 |
| ADE | S2 0-2 s | 0 | FDE (final instant) [m] (vs echo) | 0.73176 | 0.64617 | 0.0856 [0.0372, 0.1395] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | speed MAE [m/s] (vs echo) | 0.32481 | 0.25804 | 0.0668 [0.0449, 0.0904] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | along-track MAE [m] (vs echo) | 0.29798 | 0.23792 | 0.0601 [0.0392, 0.0834] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | |along-track| at the final instant [m] (vs echo) | 0.61818 | 0.50436 | 0.1138 [0.071, 0.1613] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] (vs echo) | 0.82057 | 0.86327 | -0.0427 [-0.0551, -0.0309] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 0 | accel MAE [m/s2] (vs echo) | 0.42598 | 0.32103 | 0.105 [0.0813, 0.1306] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | heading MAE [deg] (vs echo) | 1.6637 | 1.4951 | 0.1687 [0.0519, 0.2918] | True | 4473/139 |
| LATERAL | S2 0-2 s | 0 | curvature MAE (masked) [1/m] (vs echo) | 0.0045234 | 0.0039095 | 0.0006 [0.0001, 0.0011] | True | 4442/139 |
| LATERAL | S2 0-2 s | 0 | yaw-rate MAE (masked) [rad/s] (vs echo) | 0.023858 | 0.024071 | -0.0002 [-0.0029, 0.0024] | False | 4442/139 |
| LATERAL | S2 0-2 s | 0 | cross-track MAE [m] (vs echo) | 0.1317 | 0.1107 | 0.021 [0.009, 0.0335] | True | 4754/139 |
| LATERAL | S2 0-2 s | 0 | |cross-track| at the final instant [m] (vs echo) | 0.26232 | 0.26867 | -0.0063 [-0.0383, 0.0236] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived lateral decision correct [frac] (vs echo) | 0.94194 | 0.94047 | 0.0015 [-0.0105, 0.0131] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 0 | trajectory-derived longitudinal decision correct [frac] (vs echo) | 0.81994 | 0.84308 | -0.0231 [-0.0368, -0.0095] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | ADE (mean over the surface's instants) [m] (vs echo) | 0.35993 | 0.29354 | 0.0664 [0.0427, 0.0914] | True | 4754/139 |
| ADE | S2 0-2 s | 1 | FDE (final instant) [m] (vs echo) | 0.74037 | 0.64617 | 0.0942 [0.0452, 0.1479] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | speed MAE [m/s] (vs echo) | 0.32838 | 0.25804 | 0.0703 [0.0483, 0.0945] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | along-track MAE [m] (vs echo) | 0.30156 | 0.23792 | 0.0636 [0.0415, 0.0876] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | |along-track| at the final instant [m] (vs echo) | 0.62531 | 0.50436 | 0.121 [0.0772, 0.1692] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] (vs echo) | 0.81821 | 0.86327 | -0.0451 [-0.058, -0.033] | True | 4754/139 |
| LONGITUDINAL | S2 0-2 s | 1 | accel MAE [m/s2] (vs echo) | 0.42951 | 0.32103 | 0.1085 [0.0849, 0.1341] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | heading MAE [deg] (vs echo) | 1.6848 | 1.4952 | 0.1895 [0.0688, 0.3177] | True | 4472/139 |
| LATERAL | S2 0-2 s | 1 | curvature MAE (masked) [1/m] (vs echo) | 0.0045943 | 0.0039099 | 0.0007 [0.0002, 0.0012] | True | 4441/139 |
| LATERAL | S2 0-2 s | 1 | yaw-rate MAE (masked) [rad/s] (vs echo) | 0.024175 | 0.024076 | 0.0001 [-0.0026, 0.0028] | False | 4441/139 |
| LATERAL | S2 0-2 s | 1 | cross-track MAE [m] (vs echo) | 0.13299 | 0.1107 | 0.0223 [0.0099, 0.0354] | True | 4754/139 |
| LATERAL | S2 0-2 s | 1 | |cross-track| at the final instant [m] (vs echo) | 0.2651 | 0.26867 | -0.0036 [-0.0358, 0.028] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived lateral decision correct [frac] (vs echo) | 0.94089 | 0.94047 | 0.0004 [-0.0126, 0.0134] | False | 4754/139 |
| TACTICAL | S2 0-2 s | 1 | trajectory-derived longitudinal decision correct [frac] (vs echo) | 0.81342 | 0.84308 | -0.0297 [-0.0446, -0.0147] | True | 4754/139 |
| ADE | S6 0-6 s | 0 | ADE (mean over the surface's instants) [m] (vs echo) | 2.9611 | 3.5758 | -0.6147 [-0.9085, -0.3331] | True | 3668/139 |
| ADE | S6 0-6 s | 0 | FDE (final instant) [m] (vs echo) | 7.1938 | 9.2626 | -2.0687 [-2.8241, -1.3517] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | speed MAE [m/s] (vs echo) | 1.0522 | 1.1017 | -0.0495 [-0.1228, 0.026] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | along-track MAE [m] (vs echo) | 2.5083 | 2.6655 | -0.1573 [-0.3747, 0.0626] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | |along-track| at the final instant [m] (vs echo) | 6.0462 | 6.8678 | -0.8216 [-1.3957, -0.2591] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] (vs echo) | 0.50463 | 0.52281 | -0.0182 [-0.0353, -0.0021] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 0 | accel MAE [m/s2] (vs echo) | 0.47097 | 0.50681 | -0.0358 [-0.0633, -0.0069] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | heading MAE [deg] (vs echo) | 3.6914 | 5.6738 | -1.9824 [-2.7042, -1.297] | True | 3479/139 |
| LATERAL | S6 0-6 s | 0 | curvature MAE (masked) [1/m] (vs echo) | 0.0058638 | 0.0071051 | -0.0012 [-0.0021, -0.0004] | True | 3442/139 |
| LATERAL | S6 0-6 s | 0 | yaw-rate MAE (masked) [rad/s] (vs echo) | 0.033409 | 0.048671 | -0.0153 [-0.0214, -0.0092] | True | 3442/139 |
| LATERAL | S6 0-6 s | 0 | cross-track MAE [m] (vs echo) | 1.0047 | 1.6824 | -0.6777 [-0.8811, -0.4817] | True | 3668/139 |
| LATERAL | S6 0-6 s | 0 | |cross-track| at the final instant [m] (vs echo) | 2.4997 | 4.4282 | -1.9285 [-2.4561, -1.416] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived lateral decision correct [frac] (vs echo) | 0.8476 | 0.78244 | 0.0652 [0.0423, 0.0867] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 0 | trajectory-derived longitudinal decision correct [frac] (vs echo) | 0.57225 | 0.55016 | 0.0221 [-0.0055, 0.0504] | False | 3668/139 |
| ADE | S6 0-6 s | 1 | ADE (mean over the surface's instants) [m] (vs echo) | 2.999 | 3.5758 | -0.5768 [-0.8747, -0.2896] | True | 3668/139 |
| ADE | S6 0-6 s | 1 | FDE (final instant) [m] (vs echo) | 7.2946 | 9.2626 | -1.968 [-2.7276, -1.2323] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | speed MAE [m/s] (vs echo) | 1.0679 | 1.1017 | -0.0339 [-0.1075, 0.0438] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | along-track MAE [m] (vs echo) | 2.5437 | 2.6655 | -0.1218 [-0.3418, 0.1075] | False | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | |along-track| at the final instant [m] (vs echo) | 6.1357 | 6.8678 | -0.7321 [-1.2919, -0.1508] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | target-speed accuracy (+-0.5 m/s, fraction of steps) [frac] (vs echo) | 0.50291 | 0.52281 | -0.0199 [-0.0372, -0.0038] | True | 3668/139 |
| LONGITUDINAL | S6 0-6 s | 1 | accel MAE [m/s2] (vs echo) | 0.47487 | 0.50681 | -0.0319 [-0.0595, -0.0023] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | heading MAE [deg] (vs echo) | 3.7161 | 5.6738 | -1.9577 [-2.678, -1.263] | True | 3479/139 |
| LATERAL | S6 0-6 s | 1 | curvature MAE (masked) [1/m] (vs echo) | 0.0059146 | 0.0071032 | -0.0012 [-0.002, -0.0003] | True | 3443/139 |
| LATERAL | S6 0-6 s | 1 | yaw-rate MAE (masked) [rad/s] (vs echo) | 0.033732 | 0.048658 | -0.0149 [-0.0211, -0.0089] | True | 3443/139 |
| LATERAL | S6 0-6 s | 1 | cross-track MAE [m] (vs echo) | 1.0159 | 1.6824 | -0.6665 [-0.8732, -0.4702] | True | 3668/139 |
| LATERAL | S6 0-6 s | 1 | |cross-track| at the final instant [m] (vs echo) | 2.5389 | 4.4282 | -1.8893 [-2.4156, -1.389] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived lateral decision correct [frac] (vs echo) | 0.84378 | 0.78244 | 0.0613 [0.0401, 0.0819] | True | 3668/139 |
| TACTICAL | S6 0-6 s | 1 | trajectory-derived longitudinal decision correct [frac] (vs echo) | 0.5777 | 0.55016 | 0.0275 [0.0003, 0.0546] | True | 3668/139 |

STRATEGIC: **NOT APPLICABLE**, n = 0: the strategic layer is OFF (--no-strategic in the run's argv; PI ruling R5): its route loss is gated off in training, the route head is untrained and is not scored. Route-level decisions are measured inside LATERAL / TACTICAL, not relabelled strategic.


Not covered by this read (their own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception.
