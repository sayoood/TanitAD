# STRAT1 C0 same-seed replicates (MEASURED; T1-kin driving cells, T0 STRATEGIC; 141 episodes / 2,399 windows; paired episode-cluster bootstrap B 2000 seed 0, 'another draw of EPISODES')

Addendum PREREG_ADDENDUM_C0REP.md (sha256 prefix 2f41cf5c). Problems: none. K-RE rows compared: {'RESULT_STRAT1.json': 144, 'RESULT_STRAT1_SEEDS.json': 192, 'RESULT_STRAT1_N1S10.json': 336, 'RESULT_STRAT1_S1D.json': 288}. Pairs used for F_rep: 2 of 2.

## Label: COST-HOLDS

TAC-LAT is COST4_S1 under F2 at seeds [0, 10, 11]

Next step: S1's TAC-LAT cost is beyond same-seed training noise. The S1D label is re-read under F2 (CURED / PARTIAL / REMAINS as written); if it reads REMAINS, the next lever is the N2 card's attribution with the floor in hand.

## Floors

| cell | F | F_rep | F2 |
|---|---|---|---|
| operative_0_2s:ade_m | 0.0003 | 0.0005 | 0.0005 |
| operative_0_2s:fde_m | 0.0027 | 0.0019 | 0.0027 |
| operative_0_2s:speed_mae_mps | 0.0031 | 0.0007 | 0.0031 |
| operative_0_2s:along_mae_m | 0.0005 | 0.0003 | 0.0005 |
| operative_0_2s:accel_mae_mps2 | 0.0225 | 0.0013 | 0.0225 |
| operative_0_2s:cross_mae_m | 0.0004 | 0.0004 | 0.0004 |
| operative_0_2s:heading_mae_deg | 0.0170 | 0.0121 | 0.0170 |
| operative_0_2s:yaw_rate_mae_degps | 0.0385 | 0.1035 | 0.1035 |
| operative_0_2s:curvature_mae_1pm | 0.0014 | 0.0022 | 0.0022 |
| operative_0_2s:dk_hw_abs_err_m | 0.0039 | 0.0028 | 0.0039 |
| operative_0_2s:dk_tg_abs_err_s | 0.0150 | 0.0124 | 0.0150 |
| tactical_2_6s:ade_m | 0.0688 | 0.0030 | 0.0688 |
| tactical_2_6s:fde_m | 0.2619 | 0.0181 | 0.2619 |
| tactical_2_6s:speed_mae_mps | 0.0813 | 0.0219 | 0.0813 |
| tactical_2_6s:along_mae_m | 0.1219 | 0.0037 | 0.1219 |
| tactical_2_6s:accel_mae_mps2 | 0.1783 | 0.0491 | 0.1783 |
| tactical_2_6s:cross_mae_m | 0.0547 | 0.0112 | 0.0547 |
| tactical_2_6s:heading_mae_deg | 1.5731 | 0.2426 | 1.5731 |
| tactical_2_6s:yaw_rate_mae_degps | 8.1322 | 1.0831 | 8.1322 |
| tactical_2_6s:curvature_mae_1pm | 0.0747 | 0.0211 | 0.0747 |
| tactical_2_6s:dk_hw_abs_err_m | 0.0962 | 0.1087 | 0.1087 |
| tactical_2_6s:dk_tg_abs_err_s | 0.0219 | 0.0086 | 0.0219 |
| TAC-LAT | 0.0067 | 0.0046 | 0.0067 |
| TAC-LON | 0.0150 | 0.0096 | 0.0150 |

## COST4 / GAIN4 under F and F2

```
{
 "S1-C0": {
  "under_F": {
   "COST4": {
    "TAC-LAT": [
     0,
     10,
     11
    ]
   },
   "GAIN4": {}
  },
  "under_F2": {
   "COST4": {
    "TAC-LAT": [
     0,
     10,
     11
    ]
   },
   "GAIN4": {}
  },
  "changed_by_F2": {
   "COST4_dropped": [],
   "GAIN4_dropped": [],
   "COST4_added": [],
   "GAIN4_added": []
  }
 },
 "N1-C0": {
  "under_F": {
   "COST4": {},
   "GAIN4": {}
  },
  "under_F2": {
   "COST4": {},
   "GAIN4": {}
  },
  "changed_by_F2": {
   "COST4_dropped": [],
   "GAIN4_dropped": [],
   "COST4_added": [],
   "GAIN4_added": []
  }
 },
 "S1d-C0": {
  "under_F": {
   "COST4": {},
   "GAIN4": {}
  },
  "under_F2": {
   "COST4": {},
   "GAIN4": {}
  },
  "changed_by_F2": {
   "COST4_dropped": [],
   "GAIN4_dropped": [],
   "COST4_added": [],
   "GAIN4_added": []
  }
 },
 "S1d-S1": {
  "under_F": {
   "COST4": {},
   "GAIN4": {}
  },
  "under_F2": {
   "COST4": {},
   "GAIN4": {}
  },
  "changed_by_F2": {
   "COST4_dropped": [],
   "GAIN4_dropped": [],
   "COST4_added": [],
   "GAIN4_added": []
  }
 },
 "S1-N1": {
  "under_F": {
   "COST4": {},
   "GAIN4": {}
  },
  "under_F2": {
   "COST4": {},
   "GAIN4": {}
  },
  "changed_by_F2": {
   "COST4_dropped": [],
   "GAIN4_dropped": [],
   "COST4_added": [],
   "GAIN4_added": []
  }
 }
}
```

## S1D label and N1 attribution under F and F2

```
{
 "s1d_label": {
  "under_F": {
   "TAC-LAT": "PARTIAL"
  },
  "under_F2": {
   "TAC-LAT": "PARTIAL"
  }
 },
 "n1_attribution": {
  "under_F": {
   "TAC-LAT": "NEITHER"
  },
  "under_F2": {
   "TAC-LAT": "NEITHER"
  }
 }
}
```

## PRIMARY (T1-kin), replicate minus original

| cell | C0_s10r-C0_s10 | C0_s11r-C0_s11 |
|---|---|---|
| operative_0_2s:ade_m | +0.0005 [-0.0005, +0.0014] | -0.0004 [-0.0015, +0.0007] |
| operative_0_2s:fde_m | +0.0019 [-0.0015, +0.0052] | +0.0005 [-0.0028, +0.0034] |
| tactical_2_6s:ade_m | -0.0014 [-0.0321, +0.0303] | +0.0030 [-0.0278, +0.0305] |
| tactical_2_6s:fde_m | +0.0181 [-0.0662, +0.1038] | -0.0007 [-0.0990, +0.0862] |

## LONGITUDINAL (T1-kin), replicate minus original

| cell | C0_s10r-C0_s10 | C0_s11r-C0_s11 |
|---|---|---|
| operative_0_2s:speed_mae_mps | +0.0007 [-0.0010, +0.0023] | +0.0000 [-0.0011, +0.0010] |
| operative_0_2s:along_mae_m | +0.0003 [-0.0006, +0.0012] | -0.0003 [-0.0011, +0.0005] |
| operative_0_2s:accel_mae_mps2 | +0.0013 [-0.0018, +0.0042] | +0.0003 [-0.0026, +0.0029] |
| operative_0_2s:dk_hw_abs_err_m | -0.0012 [-0.0171, +0.0133] | +0.0028 [-0.0150, +0.0219] |
| operative_0_2s:dk_tg_abs_err_s | +0.0008 [-0.0016, +0.0046] | +0.0124 [-0.0017, +0.0405] |
| tactical_2_6s:speed_mae_mps | +0.0119 [-0.0214, +0.0490] | -0.0219 [-0.0574, +0.0102] |
| tactical_2_6s:along_mae_m | -0.0035 [-0.0359, +0.0300] | +0.0037 [-0.0225, +0.0297] |
| tactical_2_6s:accel_mae_mps2 | +0.0415 [-0.0473, +0.1409] | -0.0491 [-0.1713, +0.0806] |
| tactical_2_6s:dk_hw_abs_err_m | +0.1087 [+0.0001, +0.2332] COST | -0.0287 [-0.1194, +0.0570] |
| tactical_2_6s:dk_tg_abs_err_s | +0.0066 [-0.0190, +0.0302] | -0.0086 [-0.0480, +0.0238] |

## LATERAL (T1-kin), replicate minus original

| cell | C0_s10r-C0_s10 | C0_s11r-C0_s11 |
|---|---|---|
| operative_0_2s:cross_mae_m | +0.0004 [-0.0002, +0.0010] | -0.0002 [-0.0012, +0.0007] |
| operative_0_2s:heading_mae_deg | +0.0121 [+0.0020, +0.0246] | -0.0050 [-0.0231, +0.0099] |
| operative_0_2s:yaw_rate_mae_degps | -0.0033 [-0.0347, +0.0264] | +0.1035 [+0.0182, +0.2185] COST |
| operative_0_2s:curvature_mae_1pm | -0.0008 [-0.0027, +0.0006] | +0.0022 [+0.0001, +0.0054] COST |
| tactical_2_6s:cross_mae_m | +0.0112 [-0.0104, +0.0354] | -0.0016 [-0.0264, +0.0222] |
| tactical_2_6s:heading_mae_deg | -0.2426 [-0.5614, +0.0534] | +0.0773 [-0.2653, +0.4938] |
| tactical_2_6s:yaw_rate_mae_degps | -1.0831 [-2.6124, +0.4779] | +0.8928 [-0.6325, +2.6723] |
| tactical_2_6s:curvature_mae_1pm | -0.0211 [-0.0418, -0.0030] | +0.0137 [-0.0255, +0.0467] |

## TACTICAL (T1-kin), replicate minus original

| cell | C0_s10r-C0_s10 | C0_s11r-C0_s11 |
|---|---|---|
| TAC-LAT | +0.0013 [-0.0071, +0.0100] | -0.0046 [-0.0142, +0.0050] |
| TAC-LON | +0.0033 [-0.0112, +0.0167] | +0.0096 [-0.0033, +0.0221] |

## STRATEGIC (T0)

See the JSON: route following of the replicate and the original, the paired replicate-minus-original contrast per arc, RC-A route setting with F and F_rep; detected info is not computable for C0 (no context heads).