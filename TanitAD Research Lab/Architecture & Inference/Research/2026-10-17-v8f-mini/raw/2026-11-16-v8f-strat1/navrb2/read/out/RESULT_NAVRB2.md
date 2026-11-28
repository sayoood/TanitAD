# NAVRB2: the roundabout nav channel (MEASURED, T1-kin; 141 episodes / 2,399 windows per arm; B 2000 seed 0; card sha256 e238ba8d)

Problems: none. K-RE rows: 96. Strata: {"RB": {"n_windows": 235, "n_episodes": 16}, "RB-TURN": {"n_windows": 43, "n_episodes": 6}, "RB-minus-TURN": {"n_windows": 192, "n_episodes": 16}, "zero_hist_Z-EVAL": {"n_windows": 2160, "n_episodes": 130}, "hmax": 16, "all": {"n_windows": 2399, "n_episodes": 141}}.

## Outcome

```
{
 "per_pair": {
  "N1R-N1": {
   "labels": [
    "RB-READ-NO-GAIN"
   ],
   "detail": {
    "d1_gain_by_seed": [
     false,
     false
    ],
    "d1_cost_by_seed": [
     false,
     false
    ],
    "guard_cost_cells_both_seeds": [],
    "guard_cost_cells_one_seed": [],
    "guard_holds": true,
    "route_cost": false,
    "z_use_separated_by_seed": [
     true,
     true
    ]
   }
  },
  "S1R-S1": {
   "labels": [
    "RB-SEEDS-DISAGREE",
    "RB-READ-NO-GAIN"
   ],
   "detail": {
    "d1_gain_by_seed": [
     false,
     false
    ],
    "d1_cost_by_seed": [
     false,
     false
    ],
    "guard_cost_cells_both_seeds": [],
    "guard_cost_cells_one_seed": [
     "operative_0_2s:curvature_mae_1pm",
     "operative_0_2s:dk_hw_abs_err_m",
     "tactical_2_6s:dk_tg_abs_err_s"
    ],
    "guard_holds": false,
    "route_cost": false,
    "z_use_separated_by_seed": [
     true,
     true
    ]
   }
  }
 },
 "cross_pair": []
}
```

## Next step

* RB-READ-NO-GAIN: the state is read but carries no side: RB-DIST card (exit number, countdown distance), zero-training first: the navecho2 nav-only classifier with the state + exit channels
* RB-SEEDS-DISAGREE: E4 (card 4.2): N1R / S1R at s10 / s11 (PI go on cost); COST4 / GAIN4 under F2

## D1 (TAC-LAT on RB) and floors

```
{
 "N1R-N1": {
  "0": {
   "status": "OK",
   "delta": 0.004255319148936176,
   "delta_rounded": 0.0043,
   "lo": -0.0274,
   "hi": 0.0504,
   "separated": false,
   "floor": 0.03829787234042559,
   "above_floor": false,
   "gain": false,
   "cost": false,
   "n_windows": 235,
   "n_episodes": 16
  },
  "1": {
   "status": "OK",
   "delta": -0.012765957446808529,
   "delta_rounded": -0.0128,
   "lo": -0.06,
   "hi": 0.0362,
   "separated": false,
   "floor": 0.03829787234042559,
   "above_floor": false,
   "gain": false,
   "cost": false,
   "n_windows": 235,
   "n_episodes": 16
  }
 },
 "S1R-S1": {
  "0": {
   "status": "OK",
   "delta": 0.029787234042553123,
   "delta_rounded": 0.0298,
   "lo": 0.0119,
   "hi": 0.0511,
   "separated": true,
   "floor": 0.03829787234042559,
   "above_floor": false,
   "gain": false,
   "cost": false,
   "n_windows": 235,
   "n_episodes": 16
  },
  "1": {
   "status": "OK",
   "delta": -0.008510638297872353,
   "delta_rounded": -0.0085,
   "lo": -0.0675,
   "hi": 0.0495,
   "separated": false,
   "floor": 0.03829787234042559,
   "above_floor": false,
   "gain": false,
   "cost": false,
   "n_windows": 235,
   "n_episodes": 16
  }
 }
}
{"F_rb": 0.03829787234042559, "F_rep_rb": 0.008510638297872353, "F2_rb": 0.03829787234042559, "definition": "the same estimator as F / F_rep, restricted to the 235 RB windows"}
```

## PRIMARY (T1-kin; * separated, G gain, C cost under F2)

| cell | N1R-N1_s0 | N1R-N1_s1 | S1R-S1_s0 | S1R-S1_s1 |
|---|---|---|---|---|
| operative_0_2s:ade_m | -0.0029 [-0.0047, -0.0013]*G | -0.0026 [-0.0046, -0.0005]*G | -0.0007 [-0.0023, +0.0008] | -0.0003 [-0.0019, +0.0013] |
| operative_0_2s:fde_m | -0.0076 [-0.0124, -0.0032]*G | -0.0040 [-0.0106, +0.0027] | -0.0011 [-0.0056, +0.0033] | +0.0008 [-0.0052, +0.0067] |
| tactical_2_6s:ade_m | -0.0296 [-0.0642, +0.0044] | -0.0191 [-0.0665, +0.0272] | +0.0277 [-0.0086, +0.0635] | +0.0244 [-0.0279, +0.0756] |
| tactical_2_6s:fde_m | -0.0532 [-0.1676, +0.0587] | -0.0216 [-0.1503, +0.1111] | +0.0707 [-0.0335, +0.1756] | +0.0442 [-0.1156, +0.2013] |

## LONGITUDINAL (T1-kin; * separated, G gain, C cost under F2)

| cell | N1R-N1_s0 | N1R-N1_s1 | S1R-S1_s0 | S1R-S1_s1 |
|---|---|---|---|---|
| operative_0_2s:speed_mae_mps | -0.0020 [-0.0039, -0.0003]* | -0.0019 [-0.0048, +0.0012] | +0.0009 [-0.0013, +0.0030] | +0.0011 [-0.0018, +0.0039] |
| operative_0_2s:along_mae_m | -0.0021 [-0.0039, -0.0006]*G | -0.0025 [-0.0043, -0.0006]*G | +0.0002 [-0.0014, +0.0016] | -0.0007 [-0.0023, +0.0009] |
| operative_0_2s:accel_mae_mps2 | +0.0011 [-0.0026, +0.0043] | -0.0047 [-0.0097, +0.0001] | +0.0014 [-0.0026, +0.0052] | +0.0052 [+0.0003, +0.0102]* |
| operative_0_2s:dk_hw_abs_err_m | +0.0012 [-0.0115, +0.0144] | -0.0015 [-0.0257, +0.0286] | +0.0010 [-0.0115, +0.0138] | +0.0149 [+0.0007, +0.0342]*C |
| operative_0_2s:dk_tg_abs_err_s | +0.0036 [+0.0001, +0.0089]* | +0.0008 [-0.0035, +0.0069] | +0.0006 [-0.0037, +0.0061] | +0.0004 [-0.0032, +0.0039] |
| tactical_2_6s:speed_mae_mps | +0.0066 [-0.0232, +0.0385] | +0.0529 [+0.0080, +0.1034]* | +0.0347 [-0.0063, +0.0753] | +0.0143 [-0.0380, +0.0712] |
| tactical_2_6s:along_mae_m | -0.0242 [-0.0599, +0.0131] | +0.0085 [-0.0444, +0.0584] | +0.0187 [-0.0193, +0.0559] | +0.0434 [-0.0114, +0.0962] |
| tactical_2_6s:accel_mae_mps2 | +0.0859 [-0.0115, +0.1947] | +0.1570 [+0.0125, +0.3195]* | +0.1402 [-0.0021, +0.2750] | -0.1099 [-0.3167, +0.0882] |
| tactical_2_6s:dk_hw_abs_err_m | -0.0868 [-0.2367, +0.0440] | +0.0973 [-0.0225, +0.2096] | +0.0558 [-0.0752, +0.2097] | +0.0978 [-0.0117, +0.2235] |
| tactical_2_6s:dk_tg_abs_err_s | -0.0137 [-0.0393, +0.0104] | -0.0049 [-0.0385, +0.0239] | +0.0073 [-0.0205, +0.0437] | +0.0386 [+0.0112, +0.0791]*C |

## LATERAL (T1-kin; * separated, G gain, C cost under F2)

| cell | N1R-N1_s0 | N1R-N1_s1 | S1R-S1_s0 | S1R-S1_s1 |
|---|---|---|---|---|
| operative_0_2s:cross_mae_m | -0.0014 [-0.0024, -0.0004]*G | -0.0007 [-0.0019, +0.0006] | -0.0012 [-0.0021, -0.0004]*G | -0.0000 [-0.0014, +0.0012] |
| operative_0_2s:heading_mae_deg | -0.0171 [-0.0371, +0.0010] | -0.0055 [-0.0288, +0.0199] | +0.0003 [-0.0139, +0.0150] | +0.0077 [-0.0187, +0.0316] |
| operative_0_2s:yaw_rate_mae_degps | +0.0156 [-0.0353, +0.0728] | -0.0753 [-0.2672, +0.0533] | +0.0489 [-0.0069, +0.1130] | +0.0355 [-0.0638, +0.1853] |
| operative_0_2s:curvature_mae_1pm | +0.0018 [+0.0003, +0.0036]* | -0.0009 [-0.0039, +0.0013] | +0.0022 [+0.0002, +0.0046]*C | -0.0002 [-0.0021, +0.0018] |
| tactical_2_6s:cross_mae_m | -0.0086 [-0.0331, +0.0157] | -0.0328 [-0.0648, -0.0022]* | +0.0157 [-0.0118, +0.0424] | -0.0236 [-0.0544, +0.0066] |
| tactical_2_6s:heading_mae_deg | +0.4925 [+0.1940, +0.8027]* | +0.0547 [-0.2398, +0.4108] | +0.4048 [+0.0019, +0.8474]* | -0.0214 [-0.3991, +0.3652] |
| tactical_2_6s:yaw_rate_mae_degps | +4.0582 [+2.4057, +5.8588]* | +2.0586 [+0.3923, +4.0042]* | +1.7010 [-0.0159, +3.5430] | +0.0994 [-1.7972, +2.0286] |
| tactical_2_6s:curvature_mae_1pm | +0.0262 [+0.0086, +0.0443]* | +0.0058 [-0.0168, +0.0299] | +0.0073 [-0.0053, +0.0216] | +0.0182 [+0.0019, +0.0357]* |

## TACTICAL (T1-kin; * separated, G gain, C cost under F2)

| cell | N1R-N1_s0 | N1R-N1_s1 | S1R-S1_s0 | S1R-S1_s1 |
|---|---|---|---|---|
| TAC-LAT | -0.0042 [-0.0163, +0.0067] | +0.0158 [-0.0013, +0.0375] | +0.0000 [-0.0092, +0.0088] | +0.0046 [-0.0125, +0.0208] |
| TAC-LON | -0.0096 [-0.0267, +0.0067] | +0.0088 [-0.0083, +0.0263] | -0.0046 [-0.0204, +0.0117] | -0.0125 [-0.0346, +0.0088] |

## STRATEGIC (T0)

See the JSON: route following (real / donor / zero), RC-A route setting, R - banked contrasts, detected info (frozen reader and no-fog).