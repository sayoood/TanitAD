# Generated tables — closed-loop control suite

Regenerate: `python3 code/tables.py raw > artifacts/tables.md`

## T1 — DEMONSTRATED DYNAMIC RANGE (zero-bias reference arm `human_replay`)

| axis | control | zero-mean? | MDE down | MDE up | unit | both sides | monotone (beyond MDE) |
|---|---|:--:|--:|--:|---|:--:|:--:|
| `lon_track` | `lon_retime` | no | 0.050 | 0.050 | x speed (path preserved) | True | True / True |
| `lon_track` | `lon_scale` | no | 0.050 | 0.050 | x along-track scale | True | True / True |
| `lon_track` | `lon_jitter` | ⭐ yes | — | 0.250 | m sigma (zero-mean) | None | None / True |
| `lat_track` | `lat_shift` | no | 0.250 | 0.250 | m constant offset | True | True / True |
| `lat_track` | `lat_drift` | no | 0.010 | 0.010 | m per m (steering error) | True | True / True |
| `lat_track` | `lat_jitter` | ⭐ yes | — | 0.125 | m sigma (zero-mean) | None | None / True |
| `lat_track` | `yaw_bias` | no | 1.000 | 1.000 | deg plan rotation | True | True / True |
| `lat_track` | `yaw_jitter` | ⭐ yes | — | 0.500 | deg sigma (zero-mean rotation) | None | None / True |
| `lat_heading` | `yaw_bias` | no | 1.000 | 1.000 | deg plan rotation | True | True / True |
| `lat_heading` | `lat_drift` | no | 0.010 | 0.010 | m per m (steering error) | True | True / True |
| `lat_heading` | `yaw_jitter` | ⭐ yes | — | 0.500 | deg sigma (zero-mean rotation) | None | None / True |
| `recovery` | `lat_shift` | no | 0.250 | 0.250 | m constant offset | True | True / True |
| `recovery` | `lat_jitter` | ⭐ yes | — | 0.125 | m sigma (zero-mean) | None | None / True |
| `ego_progress` | `lon_retime` | no | 0.050 | 0.050 | x speed (path preserved) | True | True / True |
| `ego_progress` | `lon_jitter` | ⭐ yes | — | 0.250 | m sigma (zero-mean) | None | None / True |
| `recovery` | `yaw_jitter` | ⭐ yes | — | 0.500 | deg sigma (zero-mean rotation) | None | None / True |

## T2 — THE ADMISSION VERDICT

| axis | admissible | reason if refused |
|---|:--:|---|
| `lon_track` | ✅ |  |
| `lat_track` | ✅ |  |
| `lat_heading` | ✅ |  |
| `recovery` | ✅ |  |
| `ego_progress` | ✅ |  |

## T3 — THE SAME LADDER ON THE REAL ARMS (the confound, measured)

| arm | axis | control | MDE down | MDE up | both | monotone full ladder (dn/up) |
|---|---|---|--:|--:|:--:|:--:|
| `cv_holdv0` | `lon_track` | `lon_retime` | 0.050 | 0.050 | True | True / True |
| `cv_holdv0` | `lon_track` | `lon_scale` | 0.050 | 0.050 | True | True / True |
| `cv_holdv0` | `lon_track` | `lon_jitter` | — | 0.250 | None | None / True |
| `cv_holdv0` | `lat_track` | `lat_shift` | 0.250 | 0.250 | True | True / True |
| `cv_holdv0` | `lat_track` | `lat_drift` | 0.010 | 0.010 | True | True / False |
| `cv_holdv0` | `lat_track` | `lat_jitter` | — | 0.125 | None | None / True |
| `cv_holdv0` | `lat_track` | `yaw_bias` | 1.000 | 1.000 | True | True / False |
| `cv_holdv0` | `lat_track` | `yaw_jitter` | — | 0.500 | None | None / True |
| `cv_holdv0` | `lat_heading` | `yaw_bias` | 1.000 | 5.000 | True | True / True |
| `cv_holdv0` | `lat_heading` | `lat_drift` | 0.010 | 0.010 | True | True / True |
| `cv_holdv0` | `lat_heading` | `yaw_jitter` | — | 0.500 | None | None / True |
| `cv_holdv0` | `recovery` | `lat_shift` | 0.250 | 0.250 | True | False / False |
| `cv_holdv0` | `recovery` | `lat_jitter` | — | 0.125 | None | None / False |
| `cv_holdv0` | `ego_progress` | `lon_retime` | 0.050 | 0.050 | True | True / True |
| `cv_holdv0` | `ego_progress` | `lon_jitter` | — | 0.250 | None | None / True |
| `cv_holdv0` | `recovery` | `yaw_jitter` | — | 0.500 | None | None / False |
| `v1_tactical_follow` | `lon_track` | `lon_retime` | 0.050 | 0.050 | True | False / True |
| `v1_tactical_follow` | `lon_track` | `lon_scale` | 0.050 | 0.050 | True | False / True |
| `v1_tactical_follow` | `lon_track` | `lon_jitter` | — | 0.250 | None | None / True |
| `v1_tactical_follow` | `lat_track` | `lat_shift` | 0.250 | 0.250 | True | True / True |
| `v1_tactical_follow` | `lat_track` | `lat_drift` | 0.010 | 0.010 | True | True / True |
| `v1_tactical_follow` | `lat_track` | `lat_jitter` | — | 0.125 | None | None / True |
| `v1_tactical_follow` | `lat_track` | `yaw_bias` | 1.000 | 1.000 | True | True / False |
| `v1_tactical_follow` | `lat_track` | `yaw_jitter` | — | 0.500 | None | None / True |
| `v1_tactical_follow` | `lat_heading` | `yaw_bias` | 10.000 | 1.000 | True | True / True |
| `v1_tactical_follow` | `lat_heading` | `lat_drift` | — | 0.010 | False | False / True |
| `v1_tactical_follow` | `lat_heading` | `yaw_jitter` | — | 0.500 | None | None / True |
| `v1_tactical_follow` | `recovery` | `lat_shift` | 0.250 | 0.250 | True | False / False |
| `v1_tactical_follow` | `recovery` | `lat_jitter` | — | 0.125 | None | None / False |
| `v1_tactical_follow` | `ego_progress` | `lon_retime` | 0.050 | 0.050 | True | False / True |
| `v1_tactical_follow` | `ego_progress` | `lon_jitter` | — | 1.000 | None | None / True |
| `v1_tactical_follow` | `recovery` | `yaw_jitter` | — | 0.500 | None | None / False |

## T4 — ⛔ THE COMPOSITE REWARDS LATERAL DEGRADATION

| arm | injected lateral degradation | Δ`recovery` | Δ`ego_progress` | Δ`PSS@twosided_v2` | separated |
|---|---|---|---|---|:--:|
| `cv_holdv0` | `lat_shift(+2)` | +0.1379 | -0.0008 | **+0.0581** [+0.0473, +0.0691] | ⛔ SEP |
| `cv_holdv0` | `lat_shift(-2)` | +0.1189 | -0.0008 | **+0.0501** [+0.0386, +0.0620] | ⛔ SEP |
| `cv_holdv0` | `lat_jitter(+1)` | +0.0717 | -0.0003 | **+0.0303** [+0.0240, +0.0370] | ⛔ SEP |
| `cv_holdv0` | `yaw_bias(+5)` | +0.1784 | -0.0021 | **+0.0747** [+0.0618, +0.0872] | ⛔ SEP |
| `v1_tactical_follow` | `lat_shift(+2)` | +0.1190 | +0.0004 | **+0.0513** [+0.0409, +0.0610] | ⛔ SEP |
| `v1_tactical_follow` | `lat_shift(-2)` | +0.1137 | -0.0010 | **+0.0478** [+0.0346, +0.0607] | ⛔ SEP |
| `v1_tactical_follow` | `lat_jitter(+1)` | +0.0475 | -0.0002 | **+0.0205** [+0.0154, +0.0255] | ⛔ SEP |
| `v1_tactical_follow` | `yaw_bias(+5)` | +0.1614 | +0.0016 | **+0.0707** [+0.0594, +0.0818] | ⛔ SEP |

## T5 — ⛔ `recovery` IS FLOORED, AND THE FLOOR IS WHERE THE ARMS LIVE

| arm | `recovery` mean | defined | **frac at floor 0** | unclamped ratio median | p99 | max | frac ratio > 1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `refc_xl_produced` | 0.0259 | 0.8256 | **0.9219** | 1.367 | 3.686 | 13.0 | 0.9218 |
| `refc_xl_v0on` | 0.0259 | 0.8256 | **0.9219** | 1.367 | 3.686 | 13.0 | 0.9218 |
| `refc_base_produced` | 0.0293 | 0.8250 | **0.9160** | 1.373 | 3.827 | 16.0 | 0.9158 |
| `refc_base_v0on` | 0.0293 | 0.8250 | **0.9160** | 1.373 | 3.828 | 16.0 | 0.9158 |
| `refc_small_produced` | 0.0296 | 0.8253 | **0.9118** | 1.367 | 4.009 | 16.1 | 0.9114 |
| `refc_xl_v0off` | 0.0366 | 0.8235 | **0.9008** | 1.359 | 4.156 | 34.1 | 0.9005 |
| `refc_base_v0off` | 0.0383 | 0.8185 | **0.8993** | 1.364 | 4.364 | 31.8 | 0.8989 |
| `v4_oracle` | 0.0629 | 0.8377 | **0.7556** | 1.114 | 3.312 | 8.3 | 0.7535 |
| `v1_ego_double` | 0.0706 | 0.8234 | **0.7425** | 1.108 | 5.055 | 44.1 | 0.7401 |
| `v1_ego_v0` | 0.0653 | 0.8196 | **0.7287** | 1.095 | 5.228 | 28.4 | 0.7264 |
| `v1_ego_oracle_lon` | 0.0741 | 0.8234 | **0.7173** | 1.090 | 3.818 | 28.9 | 0.7149 |
| `nospeed_tactical_oracle` | 0.0800 | 0.8451 | **0.6988** | 1.091 | 3.838 | 10.5 | 0.6973 |
| `v1_tactical_follow` | 0.0785 | 0.8423 | **0.6973** | 1.085 | 3.672 | 9.5 | 0.6947 |
| `v1_tactical_oracle` | 0.0817 | 0.8419 | **0.6922** | 1.087 | 3.731 | 9.5 | 0.6896 |
| `v1_ego_half` | 0.0761 | 0.8116 | **0.6893** | 1.082 | 11.113 | 33.1 | 0.6870 |
| `v1_lat_straight` | 0.0747 | 0.8459 | **0.5635** | 1.004 | 5.915 | 14.3 | 0.5392 |
| `v4_blind` | 0.1159 | 0.5748 | **0.5579** | 1.042 | 9.252 | 19.9 | 0.5575 |
| `cv_holdv0` | 0.0776 | 0.8203 | **0.5565** | 1.007 | 7.036 | 20.2 | 0.5475 |
| `oracle_lon_straight` | 0.0816 | 0.8241 | **0.5503** | 1.006 | 5.707 | 9.1 | 0.5431 |

## T6 — ⛔ THE `comfort` AUDIT: the HUMAN fails the same bounds

Limits (PROPOSED): `{'a_lon_max_mps2': 3.0, 'a_lat_max_mps2': 3.0, 'jerk_max_mps3': 8.0, 'yaw_rate_max_radps': 0.95}`

| what | all four clauses | a_lon | a_lat | **jerk** | yaw_rate |
|---|--:|--:|--:|--:|--:|
| ⭐ **the HUMAN's own logged path** | **0.8340** | 0.9825 | 0.9150 | **0.9404** | 0.9812 |
| `cv_holdv0` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `stand_still` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `oracle_lon_straight` | 0.9361 | 0.9866 | 1.0000 | 0.9593 | 0.9877 |
| `v1_ego_half` | 0.6372 | 0.9936 | 0.9034 | 0.6396 | 0.9141 |
| `v1_ego_v0` | 0.2882 | 0.9585 | 0.7570 | 0.2908 | 0.8881 |
| `v1_ego_oracle_lon` | 0.2453 | 0.9423 | 0.7407 | 0.2644 | 0.8729 |
| `v1_ego_double` | 0.0987 | 0.8987 | 0.5169 | 0.1008 | 0.8826 |
| `v1_lat_straight` | 0.0099 | 0.3393 | 1.0000 | 0.0099 | 1.0000 |
| `refc_base_produced` | 0.0082 | 0.5157 | 0.4575 | 0.0181 | 0.7945 |
| `refc_base_v0on` | 0.0082 | 0.5158 | 0.4575 | 0.0180 | 0.7946 |
| `refc_xl_produced` | 0.0054 | 0.5092 | 0.4436 | 0.0161 | 0.7990 |
| `refc_xl_v0on` | 0.0053 | 0.5092 | 0.4436 | 0.0161 | 0.7990 |
| `refc_xl_v0off` | 0.0050 | 0.5343 | 0.4438 | 0.0176 | 0.7956 |
| `refc_base_v0off` | 0.0036 | 0.5258 | 0.4565 | 0.0155 | 0.7839 |
| `refc_small_produced` | 0.0027 | 0.4074 | 0.4467 | 0.0033 | 0.7856 |
| `nospeed_tactical_oracle` | 0.0007 | 0.4159 | 0.6010 | 0.0007 | 0.8234 |
| `v1_tactical_follow` | 0.0004 | 0.3622 | 0.6291 | 0.0004 | 0.8209 |
| `v1_tactical_oracle` | 0.0004 | 0.3505 | 0.6199 | 0.0004 | 0.8168 |
| `v4_blind` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0880 |
| `v4_oracle` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1666 |

## T7 — AXIS PURITY, MEASURED (response to the OTHER axis' control)

⛔ A PURITY RATIO IS NOT ARM-INDEPENDENT, and reading it off a biased arm alone is wrong in BOTH directions. `lat_heading` on `cv_holdv0` reports a ratio of 3.64 — apparently contaminated — but the cause is that `cv_holdv0` ALREADY has a large heading error, so its own control (a 5 deg rotation) barely moves the bounded score, while the same measurement on the zero-bias arm gives 0.025. That is the SAME 'a bounded score is not monotone in the raw error on a biased arm' effect documented for the ladders. The zero-bias arm answers 'is the metric pure?'; the real arm answers 'how does it behave where we actually evaluate?'. BOTH are published; neither alone is the answer. 

| arm | axis form | Δ under `lon_retime(0.5)` | Δ under its own control | contamination ÷ signal | own control dominates? |
|---|---|--:|--:|--:|:--:|
| `human_replay` | `lat_track` — widening corridor (**SHIPPED**) | +0.0000 | -0.5952 | **0.0000** | ✅ yes |
| `human_replay` | `lat_track` — FLAT tolerance (**REJECTED**) | +0.0000 | -0.5596 | **0.0000** | ✅ yes |
| `human_replay` | `lon_track` | -0.0171 | -0.5137 | **0.0333** | ✅ yes |
| `human_replay` | `lat_heading` | -0.0111 | -0.4363 | **0.0255** | ✅ yes |
| `cv_holdv0` | `lat_track` — widening corridor (**SHIPPED**) | +0.0410 | -0.1178 | **0.3477** | ✅ yes |
| `cv_holdv0` | `lat_track` — FLAT tolerance (**REJECTED**) | +0.1710 | -0.1379 | **1.2398** | ⛔ NO |
| `cv_holdv0` | `lon_track` | -0.0044 | -0.4536 | **0.0097** | ✅ yes |
| `cv_holdv0` | `lat_heading` | +0.0352 | -0.0097 | **3.6425** | ⛔ NO |

## T8 — THE ARM PANEL ON THE NEW AXES (panel-wide gate)

Gate admitted: `['ego_progress', 'lat_heading', 'lat_track', 'lon_track', 'recovery']` · dropped: `[]`

| arm | `lon_track` | rk | `lat_track` | rk | `lat_heading` | rk | `ego_progress` | `recovery` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `v1_ego_oracle_lon` | 0.9830 | 1 | 0.3676 | 4 | 0.3181 | 7 | 0.9799 | 0.0741 |
| `cv_holdv0` | 0.9347 | 2 | 0.3791 | 2 | 0.3379 | 2 | 0.9037 | 0.0776 |
| `refc_xl_produced` | 0.9322 | 3 | 0.3387 | 10 | 0.2676 | 14 | 0.9019 | 0.0259 |
| `refc_xl_v0on` | 0.9322 | 4 | 0.3387 | 11 | 0.2676 | 13 | 0.9019 | 0.0259 |
| `v1_ego_v0` | 0.9306 | 5 | 0.3656 | 5 | 0.3229 | 3 | 0.8970 | 0.0653 |
| `refc_base_produced` | 0.9279 | 6 | 0.3379 | 12 | 0.2688 | 11 | 0.8951 | 0.0293 |
| `refc_base_v0on` | 0.9279 | 7 | 0.3379 | 13 | 0.2688 | 10 | 0.8951 | 0.0293 |
| `v4_oracle` | 0.9256 | 8 | 0.3749 | 3 | 0.2949 | 8 | 0.9000 | 0.0629 |
| `refc_small_produced` | 0.9231 | 9 | 0.3355 | 16 | 0.2691 | 9 | 0.8894 | 0.0296 |
| `refc_xl_v0off` | 0.8087 | 10 | 0.3373 | 15 | 0.2665 | 15 | 0.7950 | 0.0366 |
| `refc_base_v0off` | 0.7844 | 11 | 0.3377 | 14 | 0.2678 | 12 | 0.7692 | 0.0383 |
| `v1_tactical_oracle` | 0.7079 | 12 | 0.3564 | 8 | 0.3194 | 5 | 0.6990 | 0.0817 |
| `v1_lat_straight` | 0.7011 | 13 | 0.3650 | 6 | 0.3401 | 1 | 0.6911 | 0.0747 |
| `v1_tactical_follow` | 0.7007 | 14 | 0.3557 | 9 | 0.3192 | 6 | 0.6902 | 0.0785 |
| `nospeed_tactical_oracle` | 0.6931 | 15 | 0.3571 | 7 | 0.3203 | 4 | 0.6802 | 0.0800 |
| `v4_blind` | 0.6073 | 16 | 0.5084 | 1 | 0.0807 | 16 | 0.5252 | 0.1159 |
