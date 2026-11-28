# GAINTRAJ -- does the best arc-length re-timing gain move toward 1 across refcv8's banked checkpoints?

Generated 2026-10-19T15:12:20. Registered text: `raw/gain_traj/PREREG_GAINTRAJ.md` (sha256 `4245baceb53d75fded31602fa996ad7b95ea5306e69c2056f0ed78ad80a6ddcc`). Script `code/gain_traj_r8.py` sha256 `c0cffb62ebdf74374bed0bd0fe6201f63a6b430532b5a6ea06ccfa856ba65076`.
Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Compliant windows (ceiling-compliant mask `4ebc6b86...`; S2 4,159 / S6 3,251 windows).

## LABEL (EMA, S2, both inference seeds): **STRUCTURAL**

* seed 0: g* by step 5000: 0.503, 10500: 0.498, 15000: 0.461, 20000: 0.496, 30000: 0.455; non-decreasing = False; Delta(30k - 5k) separated > 0 = False; 30k upper bound 0.534 (< 0.8: True)
* seed 1: g* by step 5000: 0.520, 10500: 0.494, 15000: 0.465, 20000: 0.508, 30000: 0.462; non-decreasing = False; Delta(30k - 5k) separated > 0 = False; 30k upper bound 0.536 (< 0.8: True)

Rule (registered): CONVERGING = Delta(30k-5k) separated > 0 at both seeds AND g* non-decreasing across the five EMA trees at both seeds; STRUCTURAL = not CONVERGING AND 30k upper bound < 0.8 at both seeds; MIXED otherwise. The label cannot tell a flat g* from a falling one; read the Delta table. **No extrapolation to the final step 48,852.**

## g* [95 % episode-cluster CI], EMA trees (the label's trees), S2

| step | seed 0 | seed 1 | seed0 - seed1 | n windows | n episodes |
|---|---|---|---|---|---|
| 5,000 | 0.503 [0.430, 0.585] | 0.520 [0.448, 0.598] | -0.0165 | 4,159 | 138 |
| 10,500 | 0.498 [0.415, 0.588] | 0.494 [0.410, 0.584] | +0.0047 | 4,159 | 138 |
| 15,000 | 0.461 [0.366, 0.565] | 0.465 [0.366, 0.571] | -0.0034 | 4,159 | 138 |
| 20,000 | 0.496 [0.401, 0.592] | 0.508 [0.408, 0.611] | -0.0123 | 4,159 | 138 |
| 30,000 | 0.455 [0.381, 0.534] | 0.462 [0.389, 0.536] | -0.0064 | 4,159 | 138 |

Paired Delta g* (EMA, S2; same episode draws):

| pair | seed 0 | seed 1 |
|---|---|---|
| 30000 - 5000 | -0.048 [-0.089, -0.010] sep<0 | -0.058 [-0.100, -0.020] sep<0 |
| 10500 - 5000 | -0.005 [-0.050, +0.037] | -0.026 [-0.076, +0.018] |
| 15000 - 10500 | -0.037 [-0.077, +0.002] | -0.029 [-0.064, +0.009] |
| 20000 - 15000 | +0.035 [-0.002, +0.071] | +0.044 [+0.006, +0.083] sep>0 |
| 30000 - 20000 | -0.041 [-0.097, +0.013] | -0.047 [-0.106, +0.014] |

## g* [95 % episode-cluster CI], EMA trees (the label's trees), S6

| step | seed 0 | seed 1 | seed0 - seed1 | n windows | n episodes |
|---|---|---|---|---|---|
| 5,000 | 0.759 [0.671, 0.852] | 0.764 [0.679, 0.854] | -0.0056 | 3,251 | 138 |
| 10,500 | 0.802 [0.705, 0.895] | 0.795 [0.697, 0.892] | +0.0072 | 3,251 | 138 |
| 15,000 | 0.724 [0.631, 0.810] | 0.728 [0.634, 0.817] | -0.0037 | 3,251 | 138 |
| 20,000 | 0.688 [0.595, 0.776] | 0.689 [0.598, 0.778] | -0.0014 | 3,251 | 138 |
| 30,000 | 0.619 [0.538, 0.700] | 0.619 [0.538, 0.699] | +0.0006 | 3,251 | 138 |

Paired Delta g* (EMA, S6; same episode draws):

| pair | seed 0 | seed 1 |
|---|---|---|
| 30000 - 5000 | -0.140 [-0.197, -0.089] sep<0 | -0.146 [-0.203, -0.097] sep<0 |
| 10500 - 5000 | +0.043 [+0.007, +0.081] sep>0 | +0.030 [-0.003, +0.064] |
| 15000 - 10500 | -0.078 [-0.116, -0.045] sep<0 | -0.067 [-0.104, -0.034] sep<0 |
| 20000 - 15000 | -0.036 [-0.057, -0.016] sep<0 | -0.038 [-0.063, -0.017] sep<0 |
| 30000 - 20000 | -0.069 [-0.097, -0.041] sep<0 | -0.071 [-0.099, -0.042] sep<0 |

## g* [95 % episode-cluster CI], RAW trees (beside, not in the label), S2

| step | seed 0 | seed 1 | seed0 - seed1 | n windows | n episodes |
|---|---|---|---|---|---|
| 5,000 | 0.213 [0.150, 0.293] | 0.214 [0.150, 0.296] | -0.0002 | 4,159 | 138 |
| 10,500 | 0.312 [0.257, 0.365] | 0.313 [0.255, 0.367] | -0.0010 | 4,159 | 138 |
| 15,000 | 0.307 [0.232, 0.396] | 0.319 [0.236, 0.410] | -0.0116 | 4,159 | 138 |
| 20,000 | 0.425 [0.362, 0.490] | 0.426 [0.362, 0.495] | -0.0012 | 4,159 | 138 |
| 30,000 | 0.328 [0.255, 0.406] | 0.336 [0.259, 0.422] | -0.0078 | 4,159 | 138 |

Paired Delta g* (RAW, S2; same episode draws):

| pair | seed 0 | seed 1 |
|---|---|---|
| 30000 - 5000 | +0.115 [+0.030, +0.188] sep>0 | +0.122 [+0.036, +0.198] sep>0 |
| 10500 - 5000 | +0.098 [+0.020, +0.167] sep>0 | +0.099 [+0.019, +0.168] sep>0 |
| 15000 - 10500 | -0.004 [-0.068, +0.070] | +0.007 [-0.060, +0.080] |
| 20000 - 15000 | +0.117 [+0.042, +0.187] sep>0 | +0.107 [+0.031, +0.184] sep>0 |
| 30000 - 20000 | -0.097 [-0.167, -0.028] sep<0 | -0.090 [-0.162, -0.017] sep<0 |

## g* [95 % episode-cluster CI], RAW trees (beside, not in the label), S6

| step | seed 0 | seed 1 | seed0 - seed1 | n windows | n episodes |
|---|---|---|---|---|---|
| 5,000 | 0.555 [0.437, 0.669] | 0.541 [0.426, 0.651] | +0.0140 | 3,251 | 138 |
| 10,500 | 0.592 [0.510, 0.665] | 0.598 [0.518, 0.672] | -0.0069 | 3,251 | 138 |
| 15,000 | 0.613 [0.525, 0.706] | 0.616 [0.523, 0.706] | -0.0025 | 3,251 | 138 |
| 20,000 | 0.680 [0.594, 0.764] | 0.676 [0.590, 0.761] | +0.0042 | 3,251 | 138 |
| 30,000 | 0.553 [0.473, 0.632] | 0.558 [0.477, 0.636] | -0.0049 | 3,251 | 138 |

Paired Delta g* (RAW, S6; same episode draws):

| pair | seed 0 | seed 1 |
|---|---|---|
| 30000 - 5000 | -0.001 [-0.099, +0.089] | +0.018 [-0.078, +0.106] |
| 10500 - 5000 | +0.037 [-0.060, +0.143] | +0.058 [-0.036, +0.157] |
| 15000 - 10500 | +0.022 [-0.036, +0.082] | +0.017 [-0.040, +0.077] |
| 20000 - 15000 | +0.067 [+0.004, +0.130] sep>0 | +0.060 [-0.001, +0.123] |
| 30000 - 20000 | -0.127 [-0.194, -0.061] sep<0 | -0.117 [-0.187, -0.051] sep<0 |

## Per-instant gains, EMA S2 (diagnostics, in-sample, no CI; (a) = RT.retime_fit (position loss along the os path), (b) = arc-length LS per instant)

| step | seed | instants (s) | (a) retime_fit | (b) LS per instant |
|---|---|---|---|---|
| 5,000 | 0 | [0.5, 1.0, 1.5, 2.0] | [0.924, 0.571, 0.49, 0.471] | [0.762, 0.5, 0.474, 0.51] |
| 5,000 | 1 | [0.5, 1.0, 1.5, 2.0] | [0.918, 0.57, 0.495, 0.482] | [0.758, 0.506, 0.492, 0.527] |
| 10,500 | 0 | [0.5, 1.0, 1.5, 2.0] | [0.957, 0.599, 0.525, 0.511] | [0.719, 0.486, 0.463, 0.507] |
| 10,500 | 1 | [0.5, 1.0, 1.5, 2.0] | [0.945, 0.599, 0.526, 0.506] | [0.71, 0.482, 0.459, 0.503] |
| 15,000 | 0 | [0.5, 1.0, 1.5, 2.0] | [1.023, 0.631, 0.524, 0.486] | [0.738, 0.478, 0.432, 0.466] |
| 15,000 | 1 | [0.5, 1.0, 1.5, 2.0] | [1.009, 0.625, 0.517, 0.479] | [0.739, 0.479, 0.433, 0.47] |
| 20,000 | 0 | [0.5, 1.0, 1.5, 2.0] | [1.042, 0.651, 0.532, 0.484] | [0.752, 0.511, 0.474, 0.499] |
| 20,000 | 1 | [0.5, 1.0, 1.5, 2.0] | [1.033, 0.648, 0.533, 0.48] | [0.745, 0.508, 0.482, 0.513] |
| 30,000 | 0 | [0.5, 1.0, 1.5, 2.0] | [1.036, 0.634, 0.49, 0.437] | [0.768, 0.502, 0.441, 0.454] |
| 30,000 | 1 | [0.5, 1.0, 1.5, 2.0] | [1.039, 0.638, 0.491, 0.434] | [0.769, 0.508, 0.449, 0.46] |

## Per-instant gains, EMA S6

| step | seed | instants (s) | (a) retime_fit | (b) LS per instant |
|---|---|---|---|---|
| 5,000 | 0 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.569, 0.47, 0.552, 0.631, 0.709, 0.76] | [0.496, 0.529, 0.647, 0.71, 0.747, 0.776] |
| 5,000 | 1 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.57, 0.486, 0.573, 0.653, 0.725, 0.773] | [0.5, 0.543, 0.658, 0.718, 0.753, 0.781] |
| 10,500 | 0 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.6, 0.523, 0.599, 0.699, 0.764, 0.795] | [0.487, 0.536, 0.683, 0.758, 0.794, 0.817] |
| 10,500 | 1 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.603, 0.512, 0.586, 0.671, 0.748, 0.78] | [0.482, 0.526, 0.669, 0.745, 0.785, 0.812] |
| 15,000 | 0 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.638, 0.498, 0.542, 0.602, 0.663, 0.683] | [0.476, 0.486, 0.619, 0.688, 0.718, 0.736] |
| 15,000 | 1 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.629, 0.484, 0.541, 0.604, 0.661, 0.695] | [0.474, 0.485, 0.623, 0.693, 0.722, 0.74] |
| 20,000 | 0 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.652, 0.485, 0.52, 0.559, 0.604, 0.623] | [0.499, 0.508, 0.617, 0.667, 0.685, 0.695] |
| 20,000 | 1 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.648, 0.478, 0.531, 0.57, 0.603, 0.632] | [0.488, 0.518, 0.628, 0.674, 0.687, 0.695] |
| 30,000 | 0 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.644, 0.451, 0.487, 0.518, 0.535, 0.554] | [0.508, 0.491, 0.571, 0.606, 0.616, 0.624] |
| 30,000 | 1 | [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] | [0.649, 0.445, 0.476, 0.514, 0.533, 0.558] | [0.513, 0.495, 0.572, 0.605, 0.616, 0.623] |

## Controls (all PASS, else the run is WITHHELD)

Overall: **PASS** on step30000ema_unc (compliant arrays, both seeds, both surfaces). Detail in GAINTRAJ.json `controls`.
* C1 (s_os = s_human) g* = 1.0 (want 1); C2 (prior + 2x) = 0.5 (want 0.5); C3 (prior + 4x) = 0.25 (want 0.25) [S2 seed 0]
* C-MUT swapped estimator reads C1 = 0.0000 (RED vs 1) and C3 = 0.7500 (RED vs 0.25); C2 is swap-invariant (0.5 <-> 0.5), which is why C3 was added
* C4 data path vs `FROZEN_GAINS_RETIME_ema30k.json` compliant gains: max abs diff S2 0.00e+00, S6 0.00e+00
* C5 vectorised vs plain-loop g* (S2 s0): abs err 2.78e-16

## Caveats (binding)

single training seed (the trees are checkpoints of ONE run); T1 open loop on logged eval139 frames; per-instant fits are in-sample per tree; eval episodes only; the interval answers 'another draw of EPISODES' only (not another training run, not another inference run); no extrapolation beyond step 30,000 (final = 48,852).
## Train-clip read (registered PREREG section 6; separate files, beside, no role in the label)

EMA-30k flat g* on the 139 TRAIN-clip rolls (`GAINTRAJ_TRAIN.json`; both seeds; 137 episodes with a compliant window): S2 compliant **0.686 [0.603, 0.772]** (s0) / 0.676 (s1), S2 all windows 0.496 / 0.501;
S6 compliant 0.833 / 0.827, S6 all 0.781 / 0.782. Against eval139 EMA-30k on the same window set (unpaired, two clip sets; post-hoc addendum `GAINTRAJ_TRAIN_vs_EVAL.md`, which also adds the eval all-windows row):
train is higher than eval by +0.21 to +0.23 (compliant, S2 and S6) and +0.11 to +0.19 (all windows), every difference separated. **Train and eval do not agree at 30k**, and the train value (0.68 S2 compliant) is itself well below 1.
The addendum is NOT registered; the label above does not depend on it.
