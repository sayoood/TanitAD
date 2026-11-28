# RESULT_CKPT120 (PREREG_V8F_CKPT120 read)

State: **FINAL**

Evidence class: MEASURED (ours; T0 evals and dumps, zero training). Estimator: taniteval.ci paired episode-cluster bootstrap, B=2000, seed 0 (F:\TanitAD-scratch\strat1\read\ckpt120/tree/taniteval/taniteval/ci.py); 'another draw of EPISODES'; another TRAINING RUN = the floors; another INFERENCE RUN does not apply. Every STRATEGIC number is worded "with the route supplied along the driven path"; H2 route following is worded "follows the supplied route text".

Side sign: +1 = left (79 scorable windows at 120 m), -1 = right (262); per-side keys read "1" and "-1".

## Arms

| arm | status | K-REACH-e |
|---|---|---|
| H1_s0 | READ | PASS |
| H1_s1 | READ | PASS |
| H2_s0 | READ | PASS |
| H2_s1 | READ | PASS |
| H1_rep | READ | PASS |
| R1_s0 | READ | PASS |
| R2_s0 | READ | PASS |

## Outcomes (T0 first tick for STRATEGIC; T1-kin for driving)

| rung | route following 120 m | route setting 120 m | route cost 30/50/80 | driving |
|---|---|---|---|---|
| H1 | S120-PARTIAL | FAR-SETS | none | KEEPS |
| H2 | S120-FOLLOWS | FAR-SETS | none | SEEDS-DISAGREE |

* H1: real - donor separated at both seeds; fails: raw point > 0 and > F120. Next: routed by stratum: shortfall where the side does not reach S -> H1: H2 (in the chain); H2: the far stratum's read decides the next lever (strategic token distance resolution, a card amendment). Shortfall where the side reaches S -> the zero-training probe on the arm's frozen S hidden (fit on train once, scored on eval once; a negative probe is not a negative about learnability); positive -> a 6,000-step arm; negative -> S reader depth 2
* H2: real - donor and balanced - 0.5 separated above 0 and F120, raw point > 0 and > F120, at both seeds. Next: the T-reads-far card (S's 120 m point as a fourth checkpoint token of T; PI go) and the derived-goals card

## STRATEGIC route following at 120 m (tier T0, first tick)

| arm | n | real | donor | zero | real - donor | balanced - 0.5 | raw (real - best constant) | echo share (UNVERIFIED) |
|---|---|---|---|---|---|---|---|---|
| H1_rep | 341/27 | +0.674 | +0.158 | +0.123 | +0.516 [+0.393, +0.619] SEP (n 341/27 ep) | +0.160 [-0.002, +0.299] not sep (n 341/27 ep) | -0.094 [-0.266, +0.099] not sep (n 341/27 ep) | +0.887 |
| H1_s0 | 341/27 | +0.689 | +0.179 | +0.147 | +0.510 [+0.388, +0.614] SEP (n 341/27 ep) | +0.240 [+0.132, +0.319] SEP (n 341/27 ep) | -0.079 [-0.278, +0.146] not sep (n 341/27 ep) | +0.885 |
| H1_s1 | 341/27 | +0.677 | +0.194 | +0.150 | +0.484 [+0.360, +0.595] SEP (n 341/27 ep) | +0.175 [+0.008, +0.299] SEP (n 341/27 ep) | -0.091 [-0.264, +0.106] not sep (n 341/27 ep) | +0.892 |
| H2_s0 | 341/27 | +0.927 | +0.182 | +0.141 | +0.745 [+0.647, +0.824] SEP (n 341/27 ep) | +0.435 [+0.383, +0.464] SEP (n 341/27 ep) | +0.158 [+0.010, +0.344] SEP (n 341/27 ep) | +0.658 |
| H2_s1 | 341/27 | +0.894 | +0.199 | +0.158 | +0.695 [+0.599, +0.778] SEP (n 341/27 ep) | +0.392 [+0.327, +0.437] SEP (n 341/27 ep) | +0.126 [-0.020, +0.306] not sep (n 341/27 ep) | +0.675 |
| R1_s0 | 341/27 | +0.000 | +0.000 | +0.000 | +0.000 [+0.000, +0.000] not sep (n 341/27 ep) | -0.500 [-0.500, -0.500] SEP (n 341/27 ep) | -0.768 [-0.910, -0.594] SEP (n 341/27 ep) | n/a |
| R2_s0 | 341/27 | +0.730 | +0.182 | +0.152 | +0.548 [+0.424, +0.643] SEP (n 341/27 ep) | +0.271 [+0.174, +0.345] SEP (n 341/27 ep) | -0.038 [-0.232, +0.180] not sep (n 341/27 ep) | +0.839 |

Strata (n windows, real level, real - donor):

* H1_rep: token_reaches_S: n 219, +0.932, +0.731 [+0.644, +0.812] SEP (n 219/24 ep); token_does_not: n 122, +0.213, +0.131 [-0.061, +0.342] not sep (n 122/22 ep); near_d<=60: n 244, +0.865, +0.672 [+0.564, +0.758] SEP (n 244/25 ep); far_d_60_100: n 97, +0.196, +0.124 [-0.019, +0.282] not sep (n 97/20 ep); t_6_30_s: n 304, +0.664, +0.510 [+0.382, +0.623] SEP (n 304/27 ep); t_above_30_s: n 37, +0.757, +0.568 [+0.321, +0.750] SEP (n 37/10 ep)
* H1_s0: token_reaches_S: n 219, +0.950, +0.735 [+0.639, +0.827] SEP (n 219/24 ep); token_does_not: n 122, +0.221, +0.107 [-0.082, +0.311] not sep (n 122/22 ep); near_d<=60: n 244, +0.848, +0.639 [+0.538, +0.732] SEP (n 244/25 ep); far_d_60_100: n 97, +0.289, +0.186 [+0.009, +0.376] SEP (n 97/20 ep); t_6_30_s: n 304, +0.688, +0.503 [+0.377, +0.613] SEP (n 304/27 ep); t_above_30_s: n 37, +0.703, +0.568 [+0.382, +0.765] SEP (n 37/10 ep)
* H1_s1: token_reaches_S: n 219, +0.941, +0.699 [+0.598, +0.789] SEP (n 219/24 ep); token_does_not: n 122, +0.205, +0.098 [-0.111, +0.350] not sep (n 122/22 ep); near_d<=60: n 244, +0.861, +0.631 [+0.522, +0.724] SEP (n 244/25 ep); far_d_60_100: n 97, +0.216, +0.113 [-0.029, +0.287] not sep (n 97/20 ep); t_6_30_s: n 304, +0.674, +0.484 [+0.348, +0.605] SEP (n 304/27 ep); t_above_30_s: n 37, +0.703, +0.486 [+0.269, +0.682] SEP (n 37/10 ep)
* H2_s0: token_reaches_S: n 219, +0.950, +0.753 [+0.667, +0.836] SEP (n 219/24 ep); token_does_not: n 122, +0.885, +0.730 [+0.529, +0.898] SEP (n 122/22 ep); near_d<=60: n 244, +0.943, +0.758 [+0.677, +0.834] SEP (n 244/25 ep); far_d_60_100: n 97, +0.887, +0.711 [+0.522, +0.882] SEP (n 97/20 ep); t_6_30_s: n 304, +0.931, +0.753 [+0.654, +0.835] SEP (n 304/27 ep); t_above_30_s: n 37, +0.892, +0.676 [+0.406, +0.865] SEP (n 37/10 ep)
* H2_s1: token_reaches_S: n 219, +0.941, +0.699 [+0.603, +0.786] SEP (n 219/24 ep); token_does_not: n 122, +0.811, +0.689 [+0.477, +0.880] SEP (n 122/22 ep); near_d<=60: n 244, +0.926, +0.701 [+0.613, +0.784] SEP (n 244/25 ep); far_d_60_100: n 97, +0.814, +0.680 [+0.477, +0.865] SEP (n 97/20 ep); t_6_30_s: n 304, +0.908, +0.707 [+0.608, +0.792] SEP (n 304/27 ep); t_above_30_s: n 37, +0.784, +0.595 [+0.405, +0.760] SEP (n 37/10 ep)
* R1_s0: token_reaches_S: n 219, +0.000, +0.000 [+0.000, +0.000] not sep (n 219/24 ep); token_does_not: n 122, +0.000, +0.000 [+0.000, +0.000] not sep (n 122/22 ep); near_d<=60: n 244, +0.000, +0.000 [+0.000, +0.000] not sep (n 244/25 ep); far_d_60_100: n 97, +0.000, +0.000 [+0.000, +0.000] not sep (n 97/20 ep); t_6_30_s: n 304, +0.000, +0.000 [+0.000, +0.000] not sep (n 304/27 ep); t_above_30_s: n 37, +0.000, +0.000 [+0.000, +0.000] not sep (n 37/10 ep)
* R2_s0: token_reaches_S: n 219, +0.954, +0.744 [+0.651, +0.830] SEP (n 219/24 ep); token_does_not: n 122, +0.328, +0.197 [-0.016, +0.423] not sep (n 122/22 ep); near_d<=60: n 244, +0.869, +0.660 [+0.562, +0.745] SEP (n 244/25 ep); far_d_60_100: n 97, +0.381, +0.268 [+0.055, +0.482] SEP (n 97/20 ep); t_6_30_s: n 304, +0.734, +0.549 [+0.422, +0.654] SEP (n 304/27 ep); t_above_30_s: n 37, +0.703, +0.541 [+0.261, +0.783] SEP (n 37/10 ep)

## Route setting at 120 m (position error vs RC-A120, arm - EXTRAP; negative = closer to the driven route)

* H1_rep: arm +16.615 m / +21.418 deg; EXTRAP +16.933 m; arm - EXTRAP pos -0.317 [-0.740, +0.062] not sep (n 2261/135 ep)
* H1_s0: arm +17.210 m / +21.706 deg; EXTRAP +17.613 m; arm - EXTRAP pos -0.403 [-0.728, -0.086] SEP (n 2261/135 ep)
* H1_s1: arm +16.629 m / +21.741 deg; EXTRAP +17.090 m; arm - EXTRAP pos -0.460 [-0.913, -0.071] SEP (n 2261/135 ep)
* H2_s0: arm +15.152 m / +17.992 deg; EXTRAP +15.685 m; arm - EXTRAP pos -0.533 [-0.969, -0.113] SEP (n 2261/135 ep)
* H2_s1: arm +15.419 m / +19.144 deg; EXTRAP +15.988 m; arm - EXTRAP pos -0.569 [-0.975, -0.223] SEP (n 2261/135 ep)
* R1_s0: arm +29.925 m / +28.789 deg; EXTRAP +17.832 m; arm - EXTRAP pos +12.093 [+8.251, +16.204] SEP (n 2261/135 ep)
* R2_s0: arm +17.109 m / +21.594 deg; EXTRAP +17.448 m; arm - EXTRAP pos -0.339 [-0.644, -0.009] SEP (n 2261/135 ep)

Floors at 120 m (F120, |H1_rep - H1_s0|): {'rd': 0.0058651026392961825, 'bal': 0.08027345637259631, 'raw': 0.014662756598240567, 'pos_err_vs_extrap': 0.08562280057379823, 'pair': '|H1_rep - H1_s0| on the same 120 m cell', 'available': True}

## Driving (24 cells; cost / gain with the floor)

* H1: ['KEEPS'] {'costs_both_seeds': [], 'costs_one_seed': [], 'gains_both_seeds': []} floor pending: False
* H2: ['SEEDS-DISAGREE'] {'costs_both_seeds': [], 'costs_one_seed': ['tactical_2_6s:accel_mae_mps2']} floor pending: False

## Controls

```
{
 "same_estimator_ci_sha256": true,
 "K1_driving_self_pair": {
  "delta": 0.0,
  "lo": 0.0,
  "hi": 0.0,
  "pass": true
 },
 "K7": {
  "hold": 0.0,
  "announced": 1.0,
  "mirrored": 0.0
 },
 "K7_pass": true,
 "K8": {
  "real": 1.0,
  "donor": 0.11436950146627566,
  "expected_donor": 0.11436950146627566,
  "pass": true
 },
 "K_PERFECT": {
  "raw": {
   "delta": 0.23167155425219943,
   "lo": 0.09010696183661304,
   "hi": 0.405579871414441,
   "separated": true,
   "n_windows": 341,
   "n_episodes": 27,
   "status": "OK"
  },
  "card": {
   "delta": 0.232,
   "lo": 0.09,
   "hi": 0.406
  },
  "matches_card_3dp": true
 },
 "K_NAVINF": {
  "raw": {
   "delta": -0.12609970674486803,
   "lo": -0.3014450410031993,
   "hi": 0.08029360010427529,
   "separated": false,
   "n_windows": 341,
   "n_episodes": 27,
   "status": "OK"
  },
  "balanced_minus_half": {
   "status": "OK",
   "delta": 0.14781621412696877,
   "lo": -0.01214003164556961,
   "hi": 0.29936544991511027,
   "separated": false,
   "n_windows": 341,
   "n_episodes": 27
  },
  "card": {
   "raw": -0.126,
   "balanced": 0.148
  }
 },
 "K1_route_self_pair": [],
 "R1_R2": {
  "R1_route_label_s0": "S120-IGNORES",
  "R1_not_FOLLOWS": true,
  "R1_far_pos_err_H1_s0_minus_R1_s0": {
   "delta": -12.71514582007611,
   "lo": -16.964644012397688,
   "hi": -8.68368229471934,
   "separated": true,
   "n_windows": 2261,
   "n_episodes": 135,
   "status": "OK"
  },
  "R1_pass": true,
  "R2_real_minus_donor_token_off": {
   "delta": 0.1967213114754098,
   "lo": -0.016156840934371534,
   "hi": 0.42344389844389835,
   "separated": false,
   "n_windows": 122,
   "n_episodes": 22,
   "status": "OK"
  },
  "R2_H2_s0_minus_R2_s0_real_token_off": {
   "delta": 0.5573770491803278,
   "lo": 0.37497614503816795,
   "hi": 0.7142857142857143,
   "separated": true,
   "n_windows": 122,
   "n_episodes": 22,
   "status": "OK"
  },
  "R2_pass": true
 }
}
```

K-LABEL: 341 windows / 27 episodes scorable (left 79 / right 262); the card says 341 / 27 (79 / 262): matches = True.
