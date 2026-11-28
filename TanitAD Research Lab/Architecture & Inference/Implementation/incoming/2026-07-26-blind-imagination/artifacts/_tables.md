<!-- TABLES:GATE -->
| deployment | n windows | n episode clusters | `ade_0_2s` (new instrument) | committed | max abs diff vs unmodified `rollout.collect` |
|---|---:|---:|---:|---:|---:|
| **40 eps — CANONICAL** | 881 | 40 | **0.427109** | 0.4271 | 1.42e-05 m |
| **600 eps** | 13198 | 600 | **0.410807** | 0.4108 | n/a (cached-encode path) |

**`GATE_PASS = True`** · ckpt step 29999 · torch 2.4.1+cu124 · python 3.11.10
<!-- /TABLES:GATE -->

<!-- TABLES:CURVE -->
*One fixed window set: **599 windows / 596 episode clusters**, `K_max = 185`, stride 8, 600-episode clean val. Every horizon and every arm is scored on the SAME windows, so the whole curve is paired.*

### REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.065** | **0.188** | **0.814** | **1.775** | **4.281** | **8.230** | **20.486** | **38.498** | **96.645** |
| (b) FROZEN-LAST-FRAME | **0.131** | **0.354** | **1.164** | **2.481** | **5.402** | **9.198** | **19.114** | **32.067** | **69.558** |
| (c) FULL OBSERVATION | **0.122** | **0.319** | **1.014** | **2.131** | **4.499** | **7.471** | **14.620** | **23.552** | **47.947** |
| (c2) observed-pair odometry | **1.510** | **2.958** | **5.639** | **8.238** | **12.216** | **16.244** | **23.962** | **31.796** | **49.725** |
| (d) CONSTANT VELOCITY — the floor | **0.094** | **0.339** | **1.268** | **2.750** | **5.898** | **9.926** | **20.281** | **33.547** | **69.810** |
| (d2) hold-v0 go-straight | **0.088** | **0.327** | **1.244** | **2.717** | **5.851** | **9.854** | **20.139** | **33.339** | **69.520** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.066 | +0.166 | +0.350 | +0.705 | +1.121 | +0.968 | -1.372 | -6.431 | -27.087 |
| CI95 | [+0.057, +0.076] | [+0.144, +0.191] | [+0.273, +0.435] | [+0.543, +0.882] | [+0.775, +1.462] | [+0.355, +1.570] | [-2.777, -0.044] | [-9.038, -4.035] | [-33.892, -21.167] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⛔ | ⛔ | ⛔ |

### REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.127** | **0.439** | **2.158** | **4.954** | **10.202** | **16.570** | **33.919** | **58.438** | **132.328** |
| (b) FROZEN-LAST-FRAME | **0.154** | **0.424** | **1.345** | **2.841** | **6.094** | **10.307** | **21.431** | **36.096** | **78.228** |
| (c) FULL OBS (teacher-forced percept) | **0.149** | **0.400** | **1.204** | **2.432** | **4.998** | **8.174** | **15.957** | **25.790** | **52.664** |
| (d) CONSTANT VELOCITY — the floor | **0.094** | **0.339** | **1.268** | **2.750** | **5.898** | **9.926** | **20.281** | **33.547** | **69.810** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.027 | -0.015 | -0.813 | -2.113 | -4.107 | -6.263 | -12.488 | -22.342 | -54.100 |
| CI95 | [+0.018, +0.037] | [-0.041, +0.010] | [-0.937, -0.692] | [-2.397, -1.841] | [-4.711, -3.522] | [-7.269, -5.269] | [-14.482, -10.588] | [-25.610, -19.196] | [-61.421, -47.218] |
| imagination separated-better? | ✅ | — | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |

### REGIME (ii-0) — HELD LAST ACTION (deployable, no policy)

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.069** | **0.217** | **1.035** | **2.392** | **5.808** | **10.986** | **26.418** | **48.166** | **113.308** |
| (b) FROZEN-LAST-FRAME | **0.131** | **0.359** | **1.197** | **2.583** | **5.682** | **9.737** | **20.501** | **34.634** | **74.933** |
| (d) CONSTANT VELOCITY — the floor | **0.094** | **0.339** | **1.268** | **2.750** | **5.898** | **9.926** | **20.281** | **33.547** | **69.810** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.062 | +0.142 | +0.162 | +0.191 | -0.127 | -1.249 | -5.917 | -13.532 | -38.376 |
| CI95 | [+0.053, +0.072] | [+0.118, +0.166] | [+0.085, +0.247] | [+0.030, +0.370] | [-0.458, +0.227] | [-1.821, -0.681] | [-7.248, -4.694] | [-16.093, -11.120] | [-45.061, -32.593] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | — | ⛔ | ⛔ | ⛔ | ⛔ |

### CONVENTION CONTROL — actions from the TRUE motion through the same inverse

**`de_N` — displacement error AT horizon N (m)**

| arm | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| (a) IMAGINATION | **0.076** | **0.235** | **0.951** | **2.042** | **4.812** | **9.108** | **22.120** | **41.013** | **100.431** |
| (b) FROZEN-LAST-FRAME | **0.136** | **0.370** | **1.196** | **2.535** | **5.476** | **9.314** | **19.384** | **32.643** | **70.853** |
| (c) FULL OBSERVATION | **0.128** | **0.336** | **1.053** | **2.192** | **4.608** | **7.655** | **15.103** | **24.495** | **50.170** |

| paired contrast | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.060 | +0.135 | +0.245 | +0.493 | +0.663 | +0.206 | -2.735 | -8.370 | -29.578 |
| CI95 | [+0.051, +0.069] | [+0.113, +0.159] | [+0.172, +0.326] | [+0.337, +0.661] | [+0.330, +0.990] | [-0.356, +0.753] | [-4.055, -1.537] | [-10.815, -6.117] | [-36.047, -23.558] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | ✅ | — | ⛔ | ⛔ | ⛔ |

<!-- /TABLES:CURVE -->

<!-- TABLES:TBLIND -->
| regime | `T_blind` | CI95 | draws with `T_blind = 0` | first step where (b) is separated-BETTER | C14 saturated? |
|---|---:|---|---:|---:|---|
| REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable | **6.5 s** (65 steps) | [6.5, 8.9] s | 0.000 | 9.0 s | no |
| REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION | **0.8 s** (8 steps) | [0.8, 1.0] s | 0.000 | 1.1 s | no |
| REGIME (ii-0) — HELD LAST ACTION (deployable, no policy) | **3.2 s** (32 steps) | [3.2, 5.0] s | 0.000 | 5.1 s | no |
| CONVENTION CONTROL — actions from the TRUE motion through the same inverse | **5.4 s** (54 steps) | [5.4, 7.3] s | 0.000 | 7.4 s | no |
| A2 SENSITIVITY — readout = step['str'] (20-step-calibrated), true actions | **18.5 s** (185 steps) | [18.5, 18.5] s | 0.000 | — | ⚠️ YES — LOWER BOUND |

**Usefulness horizons for arm (a), and the CV floor crossing**

| regime | `de_N` < 1.0 m | < 1.391 m (corridor) | < 2.0 m (miss@2m) | beats CV floor until |
|---|---|---|---|---|
| REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable | 2.2 s | 2.6 s | 3.1 s | **0.1 s** [0.1, 0.1] |
| REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION | 1.4 s | 1.6 s | 1.9 s | **0.1 s** [0.1, 0.1] |
| REGIME (ii-0) — HELD LAST ACTION (deployable, no policy) | 1.9 s | 2.3 s | 2.7 s | **0.1 s** [0.1, 0.1] |
| CONVENTION CONTROL — actions from the TRUE motion through the same inverse | 2.0 s | 2.4 s | 2.9 s | **0.1 s** [0.1, 0.1] |
| A2 SENSITIVITY — readout = step['str'] (20-step-calibrated), true actions | 3.4 s | 3.9 s | 4.5 s | **0.1 s** [0.1, 0.1] |
<!-- /TABLES:TBLIND -->

<!-- TABLES:CONTROL -->
*Positive = the TRUE-action arm is better, i.e. the cost of routing the true motion through my kinematic inverse. A value indistinguishable from 0 means the inverse is faithful and any own-action penalty belongs to the model.*

| true_future − gt_kinematic, arm (a) | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| Δ (b − a), + = imagination better | +0.011 | +0.047 | +0.137 | +0.267 | +0.531 | +0.878 | +1.634 | +2.516 | +3.786 |
| CI95 | [+0.008, +0.014] | [+0.037, +0.057] | [+0.107, +0.174] | [+0.190, +0.354] | [+0.359, +0.727] | [+0.601, +1.209] | [+1.054, +2.276] | [+1.608, +3.516] | [+1.921, +5.794] |
| imagination separated-better? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
<!-- /TABLES:CONTROL -->

<!-- TABLES:LEVER -->
| contrast (positive = the ALTERNATE readout is better) | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `a_imagination__true__roTAC_vs_a_imagination__true` | -0.022⛔ | +0.050✅ | +0.501✅ | +1.032✅ | +2.130✅ | +3.780✅ | +8.799✅ | +15.981✅ | +39.626✅ |
| `a_imagination__true__roSTR_vs_a_imagination__true` | -0.038⛔ | +0.038✅ | +0.502✅ | +1.080✅ | +2.296✅ | +4.103✅ | +9.535✅ | +17.383✅ | +43.304✅ |
| `a_imagination__own__roSTR_vs_a_imagination__own` | -0.045⛔ | -0.025 | +0.341✅ | +1.101✅ | +2.062✅ | +2.567✅ | +4.417✅ | +9.088✅ | +29.823✅ |
| `a_imagination__hold__roSTR_vs_a_imagination__hold` | -0.037⛔ | +0.027✅ | +0.363✅ | +0.723✅ | +1.563✅ | +2.896✅ | +7.230✅ | +13.846✅ | +34.782✅ |
| `b_frozenlast__true__roSTR_vs_b_frozenlast__true` | -0.287⛔ | -0.520⛔ | -0.876⛔ | -1.130⛔ | -1.391⛔ | -1.667⛔ | -2.260⛔ | -2.749⛔ | -2.799⛔ |
| `c2_observedpair__true__roSTR_vs_c2_observedpair__true` | -0.002 | -0.008 | -0.029⛔ | -0.054⛔ | -0.103⛔ | -0.190⛔ | -0.481⛔ | -0.726⛔ | -1.269⛔ |
<!-- /TABLES:LEVER -->

<!-- TABLES:DECOMP -->
**(a) IMAGINATION** — `a_imagination__true`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 0.065 | 0.188 | 0.814 | 1.775 | 4.281 | 8.230 | 20.486 | 38.498 | 96.645 |
| along-track \|err\| (m) | 0.060 | 0.172 | 0.733 | 1.408 | 2.797 | 4.979 | 11.668 | 21.707 | 55.488 |
| cross-track \|err\| (m) | 0.015 | 0.047 | 0.248 | 0.808 | 2.547 | 5.371 | 14.305 | 27.448 | 70.532 |
| longitudinal share of squared error | 0.937 | 0.911 | 0.834 | 0.705 | 0.546 | 0.465 | 0.393 | 0.366 | 0.374 |
| DRIFT: signed along mean (m) | 0.008 | 0.059 | 0.483 | 0.917 | 1.529 | 1.996 | 1.809 | -0.744 | -15.815 |
| VARIANCE: along std (m) | 0.096 | 0.260 | 0.801 | 1.579 | 3.571 | 6.621 | 15.827 | 29.094 | 73.263 |
| DRIFT: signed cross mean (m) | 0.001 | -0.002 | -0.016 | -0.032 | -0.154 | -0.552 | -2.709 | -7.271 | -25.015 |
| VARIANCE: cross std (m) | 0.025 | 0.083 | 0.417 | 1.182 | 3.538 | 7.391 | 19.606 | 37.606 | 93.572 |
| drift share of along energy | 0.007 | 0.049 | 0.267 | 0.252 | 0.155 | 0.083 | 0.013 | 0.001 | 0.045 |
| Frenet cross-track p90 (m) | 0.03 | 0.10 | 0.58 | 1.89 | 6.25 | 14.04 | 38.47 | 73.43 | 189.22 |
| model's own predicted speed (m/s) | 12.97 | 13.20 | 13.30 | 13.29 | 13.26 | 13.29 | 13.37 | 13.49 | 13.42 |
| frac steps OUTSIDE the measured envelope | 0.000 | 0.000 | 0.002 | 0.007 | 0.072 | 0.200 | 0.395 | 0.516 | 0.673 |

**(a) IMAGINATION** — `a_imagination__own`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 33.919 | 58.438 | 132.328 |
| along-track \|err\| (m) | 0.118 | 0.368 | 1.254 | 2.718 | 5.524 | 9.098 | 19.018 | 32.852 | 75.644 |
| cross-track \|err\| (m) | 0.026 | 0.159 | 1.369 | 3.286 | 6.805 | 11.232 | 23.570 | 41.303 | 95.462 |
| longitudinal share of squared error | 0.936 | 0.771 | 0.363 | 0.332 | 0.355 | 0.381 | 0.404 | 0.401 | 0.406 |
| DRIFT: signed along mean (m) | 0.044 | 0.070 | 0.350 | 1.038 | 1.877 | 2.272 | 0.885 | -4.341 | -30.451 |
| VARIANCE: along std (m) | 0.165 | 0.489 | 1.617 | 3.464 | 7.564 | 13.038 | 27.353 | 45.432 | 96.679 |
| DRIFT: signed cross mean (m) | 0.004 | 0.026 | 0.491 | 0.964 | 1.347 | 1.300 | -0.379 | -4.541 | -21.053 |
| VARIANCE: cross std (m) | 0.044 | 0.268 | 2.136 | 5.036 | 10.421 | 16.815 | 33.243 | 55.559 | 120.914 |
| drift share of along energy | 0.066 | 0.020 | 0.045 | 0.083 | 0.058 | 0.029 | 0.001 | 0.009 | 0.090 |
| Frenet cross-track p90 (m) | 0.07 | 0.39 | 3.90 | 8.78 | 18.36 | 29.59 | 56.34 | 88.28 | 193.21 |
| model's own predicted speed (m/s) | 13.13 | 12.74 | 13.59 | 13.61 | 13.45 | 13.27 | 13.04 | 12.93 | 12.75 |
| frac steps OUTSIDE the measured envelope | 0.002 | 0.015 | 0.072 | 0.166 | 0.293 | 0.396 | 0.538 | 0.625 | 0.735 |

**(b) FROZEN-LAST-FRAME** — `b_frozenlast__true`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 0.131 | 0.354 | 1.164 | 2.481 | 5.402 | 9.198 | 19.114 | 32.067 | 69.558 |
| along-track \|err\| (m) | 0.126 | 0.326 | 1.005 | 2.041 | 4.261 | 7.099 | 14.215 | 23.259 | 48.205 |
| cross-track \|err\| (m) | 0.018 | 0.079 | 0.360 | 0.896 | 2.188 | 3.971 | 9.077 | 15.949 | 37.616 |
| longitudinal share of squared error | 0.975 | 0.923 | 0.847 | 0.802 | 0.751 | 0.714 | 0.663 | 0.632 | 0.595 |
| DRIFT: signed along mean (m) | 0.035 | 0.091 | 0.261 | 0.500 | 0.989 | 1.631 | 2.896 | 3.811 | 7.191 |
| VARIANCE: along std (m) | 0.190 | 0.473 | 1.451 | 2.994 | 6.204 | 10.216 | 20.062 | 32.242 | 65.940 |
| DRIFT: signed cross mean (m) | 0.002 | 0.008 | 0.022 | 0.056 | 0.173 | 0.366 | 0.813 | 1.204 | 2.440 |
| VARIANCE: cross std (m) | 0.031 | 0.139 | 0.626 | 1.506 | 3.609 | 6.544 | 14.443 | 24.737 | 54.722 |
| drift share of along energy | 0.032 | 0.036 | 0.032 | 0.027 | 0.025 | 0.025 | 0.020 | 0.014 | 0.012 |
| Frenet cross-track p90 (m) | 0.04 | 0.18 | 0.87 | 2.32 | 5.24 | 9.60 | 25.09 | 44.28 | 86.27 |
| model's own predicted speed (m/s) | 13.03 | 13.04 | 13.03 | 13.02 | 13.02 | 13.02 | 13.04 | 13.02 | 13.00 |
| frac steps OUTSIDE the measured envelope | 0.000 | 0.001 | 0.007 | 0.031 | 0.092 | 0.163 | 0.297 | 0.399 | 0.534 |

**(c2) observed-pair odometry** — `c2_observedpair__true`

| quantity | 0.5s | 1s | 2s | 3s | 4.5s | 6s | 9s | 12s | 18.5s |
|---|---|---|---|---|---|---|---|---|---|
| `de_N` (m) | 1.510 | 2.958 | 5.639 | 8.238 | 12.216 | 16.244 | 23.962 | 31.796 | 49.725 |
| along-track \|err\| (m) | 1.508 | 2.951 | 5.592 | 8.091 | 11.807 | 15.434 | 21.970 | 27.982 | 40.281 |
| cross-track \|err\| (m) | 0.025 | 0.097 | 0.387 | 0.830 | 1.803 | 3.080 | 6.194 | 10.117 | 21.292 |
| longitudinal share of squared error | 1.000 | 0.998 | 0.992 | 0.984 | 0.965 | 0.943 | 0.898 | 0.851 | 0.752 |
| DRIFT: signed along mean (m) | 0.118 | 0.269 | 0.617 | 1.015 | 1.395 | 1.884 | 2.760 | 3.238 | 3.586 |
| VARIANCE: along std (m) | 2.083 | 4.022 | 7.674 | 11.199 | 16.405 | 21.506 | 31.195 | 40.517 | 58.775 |
| DRIFT: signed cross mean (m) | -0.001 | -0.001 | 0.001 | 0.041 | 0.195 | 0.403 | 0.792 | 1.283 | 2.909 |
| VARIANCE: cross std (m) | 0.042 | 0.170 | 0.677 | 1.446 | 3.124 | 5.276 | 10.505 | 16.935 | 33.723 |
| drift share of along energy | 0.003 | 0.004 | 0.006 | 0.008 | 0.007 | 0.008 | 0.008 | 0.006 | 0.004 |
| Frenet cross-track p90 (m) | 0.07 | 0.26 | 1.08 | 2.38 | 4.91 | 8.70 | 15.82 | 24.05 | 52.74 |
| model's own predicted speed (m/s) | 13.29 | 13.04 | 13.10 | 13.20 | 12.92 | 13.15 | 13.15 | 13.17 | 13.15 |
| frac steps OUTSIDE the measured envelope | 0.003 | 0.010 | 0.025 | 0.045 | 0.090 | 0.143 | 0.256 | 0.352 | 0.501 |

*GT ego speed at window end: **12.90 m/s**. Envelope: |dlat| ≤ 3.0 m, |dyaw| ≤ 12.0°. ⚠️ The last horizon that is a genuine MEASUREMENT is 0.4 s. Every reading beyond it is EXTRAPOLATION. The OOD RATIO is deliberately NOT quoted: sup(ratio_arr)=1.298888 makes the <=1.30 test a tautology (C13). ENV_YAW_MAX=12deg was never measured; it is a grid terminus (C14).*
<!-- /TABLES:DECOMP -->

<!-- TABLES:DUTY -->
**UNIFORM peek-every-T′ (deployable)** — base arm: imagination + own actions

| policy | front-camera duty cycle | de@0.5s | de@1s | de@2s | de@3s | de@4.5s | de@6s | de@9s | de@12s | de@18.5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peek_own_uniform_T90` | **0.0108** | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 33.919 | 55.763 | 123.579 |
| `peek_own_uniform_T60` | **0.0162** | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 32.361 | 54.171 | 119.287 |
| `peek_own_uniform_T45` | **0.0216** | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.127 | 31.287 | 51.248 | 111.034 |
| `peek_own_uniform_T30` | **0.0324** | 0.127 | 0.439 | 2.158 | 4.954 | 9.954 | 15.936 | 30.493 | 49.081 | 103.075 |
| `peek_own_uniform_T20` | **0.0486** | 0.127 | 0.439 | 2.158 | 4.711 | 9.521 | 15.517 | 29.920 | 48.438 | 98.925 |
| `peek_own_uniform_T15` | **0.0649** | 0.127 | 0.439 | 2.036 | 4.572 | 9.452 | 15.470 | 30.932 | 50.604 | 104.886 |
| `peek_own_uniform_T10` | **0.0973** | 0.127 | 0.439 | 1.899 | 4.165 | 8.750 | 14.474 | 29.062 | 48.047 | 101.841 |
| `peek_own_uniform_T5` | **0.1946** | 0.127 | 0.361 | 1.254 | 2.677 | 5.599 | 9.244 | 18.579 | 30.628 | 64.142 |
| `peek_own_uniform_T3` | **0.3297** | 0.120 | 0.363 | 1.205 | 2.545 | 5.306 | 8.745 | 17.401 | 28.441 | 58.830 |
| `peek_own_uniform_T2` | **0.4973** | 0.133 | 0.364 | 1.182 | 2.468 | 5.139 | 8.472 | 16.746 | 27.259 | 56.187 |

**ORACLE peek ⚠️ privileged — reads the true error** — base arm: imagination + own actions

| policy | front-camera duty cycle | de@0.5s | de@1s | de@2s | de@3s | de@4.5s | de@6s | de@9s | de@12s | de@18.5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peek_own_oracle_e0.5` | **0.1529** | 0.127 | 0.450 | 2.095 | 4.910 | 10.809 | 18.436 | 38.775 | 64.995 | 134.338 |
| `peek_own_oracle_e0.2` | **0.4450** | 0.127 | 0.469 | 1.882 | 3.832 | 8.539 | 14.462 | 29.008 | 46.461 | 92.764 |
| `peek_own_oracle_e0.1` | **0.6199** | 0.128 | 0.446 | 1.609 | 3.200 | 6.573 | 10.866 | 21.517 | 34.633 | 69.688 |
| `peek_own_oracle_e0.05` | **0.7479** | 0.127 | 0.395 | 1.345 | 2.701 | 5.573 | 9.091 | 17.711 | 28.428 | 57.584 |
| `peek_own_oracle_e0.02` | **0.8691** | 0.136 | 0.382 | 1.217 | 2.489 | 5.134 | 8.383 | 16.410 | 26.518 | 53.874 |

**Anchors — the two ends of the duty-cycle axis**

| baseline | duty cycle | de@0.5s | de@1s | de@2s | de@3s | de@4.5s | de@6s | de@9s | de@12s | de@18.5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `a_imagination__own` | 0.0 | 0.127 | 0.439 | 2.158 | 4.954 | 10.202 | 16.570 | 33.919 | 58.438 | 132.328 |
| `a_imagination__hold` | 0.0 | 0.069 | 0.217 | 1.035 | 2.392 | 5.808 | 10.986 | 26.418 | 48.166 | 113.308 |
| `c_fullobs__own` | 1.0 | 0.149 | 0.400 | 1.204 | 2.432 | 4.998 | 8.174 | 15.957 | 25.790 | 52.664 |
| `c_fullobs__true` | 1.0 | 0.122 | 0.319 | 1.014 | 2.131 | 4.499 | 7.471 | 14.620 | 23.552 | 47.947 |
| `a_imagination__true` | 0.0 | 0.065 | 0.188 | 0.814 | 1.775 | 4.281 | 8.230 | 20.486 | 38.498 | 96.645 |

**ORACLE vs UNIFORM at matched duty cycle — the informative version of H2's efficiency claim**

| oracle policy | oracle duty | matched uniform | uniform duty | rel. Δde@0.5s | rel. Δde@1s | rel. Δde@2s | rel. Δde@3s | rel. Δde@4.5s | rel. Δde@6s | rel. Δde@9s | rel. Δde@12s | rel. Δde@18.5s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peek_own_oracle_e0.02` | 0.8691 | `peek_own_uniform_T2` | 0.4973 | -2.0% | -5.0% | -2.9% | -0.9% | +0.1% | +1.1% | +2.0% | +2.7% | +4.1% |
| `peek_own_oracle_e0.05` | 0.7479 | `peek_own_uniform_T2` | 0.4973 | +4.7% | -8.4% | -13.8% | -9.5% | -8.4% | -7.3% | -5.8% | -4.3% | -2.5% |
| `peek_own_oracle_e0.1` | 0.6199 | `peek_own_uniform_T2` | 0.4973 | +3.8% | -22.5% | -36.1% | -29.7% | -27.9% | -28.3% | -28.5% | -27.1% | -24.0% |
| `peek_own_oracle_e0.2` | 0.4450 | `peek_own_uniform_T2` | 0.4973 | +4.5% | -28.7% | -59.2% | -55.3% | -66.1% | -70.7% | -73.2% | -70.4% | -65.1% |
| `peek_own_oracle_e0.5` | 0.1529 | `peek_own_uniform_T5` | 0.1946 | +0.0% | -24.8% | -67.1% | -83.4% | -93.1% | -99.4% | -108.7% | -112.2% | -109.4% |
| `peek_hold_oracle_e0.02` | 0.8613 | `peek_hold_uniform_T2` | 0.4973 | -14.6% | -21.9% | -18.4% | -8.7% | -3.7% | -1.5% | +1.1% | +2.6% | +4.1% |
| `peek_hold_oracle_e0.05` | 0.7563 | `peek_hold_uniform_T2` | 0.4973 | +7.5% | -25.9% | -56.0% | -36.9% | -26.8% | -22.4% | -17.9% | -15.1% | -11.7% |
| `peek_hold_oracle_e0.1` | 0.5867 | `peek_hold_uniform_T2` | 0.4973 | +19.4% | -13.8% | -40.7% | -30.7% | -27.6% | -28.2% | -30.0% | -29.4% | -28.5% |
| `peek_hold_oracle_e0.2` | 0.3715 | `peek_hold_uniform_T3` | 0.3297 | +11.6% | +6.9% | -16.7% | -17.9% | -22.5% | -28.2% | -36.7% | -40.3% | -41.9% |
| `peek_hold_oracle_e0.5` | 0.1396 | `peek_hold_uniform_T10` | 0.0973 | +0.0% | +0.0% | +3.7% | -1.0% | -8.2% | -16.6% | -25.1% | -28.3% | -30.8% |
<!-- /TABLES:DUTY -->
