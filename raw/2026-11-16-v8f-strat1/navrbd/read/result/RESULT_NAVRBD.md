# NAVRBD read (MEASURED, T1-kin; D1-HO 5736 windows / 497 episodes; B 2000 seed 0; card sha256 881d7f3d)

Stand-in aliases: none (real dumps)
Problems: none. NOT-YET: ['Z-EVAL on the held-out grid (no --ho-zero-hist-mask); read on eval-141 below'].

## Outcome: ['RBD-NO-GAIN', 'RBD-FLOOR-HIGH'] (FINAL)

```
{
 "status": "FINAL",
 "missing": [],
 "labels": [
  "RBD-NO-GAIN",
  "RBD-FLOOR-HIGH"
 ],
 "labels_if_N1Dx_ignored": [
  "RBD-NO-GAIN",
  "RBD-FLOOR-HIGH"
 ],
 "detail": {
  "bar": 0.042364016736401666,
  "F_HO": 0.042364016736401666,
  "bar_rule": "max(0.0383, F_HO)",
  "route_cost": false,
  "d1_gain_by_seed": [
   false,
   false
  ],
  "d1_cost_by_seed": [
   false,
   false
  ],
  "d1_delta_by_seed": [
   0.029114365411436527,
   0.017085076708507674
  ],
  "z_use_separated_by_seed": [
   true,
   true
  ],
  "guard_cost_cells_by_seed": [
   [
    "tactical_2_6s:dk_tg_abs_err_s"
   ],
   []
  ],
  "guard_cost_cells_both_seeds": [],
  "guard_holds": false,
  "N1Dx_gain": false
 }
}
```

## Next

* RBD-NO-GAIN: the stacking-combiner check (CPU, zero-training): RB-JOINT vision probe out-of-fold probabilities stacked into the ED tree, on the same CV and on the held-out windows; then the vision lever: an RB window weight in the loss and/or the 8x20 grid input (RAW8 - S was +0.0266 separated, below the bar)
* RBD-FLOOR-HIGH: the cell cannot resolve this lever at these seeds: more seeds of the pair (PI go), never a lower bar

## D1-HO (kinematic TAC-LAT; bar 0.042364)

```
{
 "N1D_s0-N1R_s0": {
  "status": "OK",
  "delta": 0.029114365411436527,
  "lo": 0.0174,
  "hi": 0.0409,
  "separated": true,
  "bar": 0.042364016736401666,
  "gain": false,
  "cost": false,
  "acc_arm": 0.6195955369595537,
  "acc_ref": 0.5904811715481172,
  "n_windows": 5736,
  "n_episodes": 497
 },
 "N1D_s1-N1R_s1": {
  "status": "OK",
  "delta": 0.017085076708507674,
  "lo": 0.0044,
  "hi": 0.03,
  "separated": true,
  "bar": 0.042364016736401666,
  "gain": false,
  "cost": false,
  "acc_arm": 0.6499302649930265,
  "acc_ref": 0.6328451882845189,
  "n_windows": 5736,
  "n_episodes": 497
 },
 "N1Dx_s0-N1R_s0": {
  "status": "OK",
  "delta": 0.001394700139469962,
  "lo": -0.0089,
  "hi": 0.0113,
  "separated": false,
  "bar": 0.042364016736401666,
  "gain": false,
  "cost": false,
  "acc_arm": 0.5918758716875872,
  "acc_ref": 0.5904811715481172,
  "n_windows": 5736,
  "n_episodes": 497
 }
}
```

## eval-141 guard (T1-kin; * separated, G gain, C cost under F2)

| cell | N1D_s0-N1R_s0 | N1D_s1-N1R_s1 |
|---|---|---|
| operative_0_2s:ade_m | -0.0005 [-0.0018, +0.0009] | -0.0002 [-0.0017, +0.0014] |
| operative_0_2s:fde_m | -0.0010 [-0.0053, +0.0034] | -0.0018 [-0.0068, +0.0034] |
| operative_0_2s:speed_mae_mps | -0.0000 [-0.0016, +0.0016] | +0.0011 [-0.0010, +0.0034] |
| operative_0_2s:along_mae_m | +0.0002 [-0.0010, +0.0015] | +0.0002 [-0.0012, +0.0016] |
| operative_0_2s:accel_mae_mps2 | -0.0040 [-0.0075, -0.0007]* | +0.0075 [+0.0038, +0.0115]* |
| operative_0_2s:cross_mae_m | -0.0009 [-0.0021, +0.0000] | -0.0006 [-0.0024, +0.0007] |
| operative_0_2s:heading_mae_deg | -0.0132 [-0.0340, +0.0059] | -0.0034 [-0.0219, +0.0145] |
| operative_0_2s:yaw_rate_mae_degps | -0.0624 [-0.1262, -0.0082]* | -0.0160 [-0.0601, +0.0243] |
| operative_0_2s:curvature_mae_1pm | -0.0014 [-0.0037, +0.0003] | -0.0002 [-0.0015, +0.0008] |
| operative_0_2s:dk_hw_abs_err_m | -0.0118 [-0.0300, +0.0032] | -0.0176 [-0.0558, +0.0158] |
| operative_0_2s:dk_tg_abs_err_s | -0.0055 [-0.0123, -0.0006]* | +0.0079 [-0.0099, +0.0373] |
| tactical_2_6s:ade_m | -0.0137 [-0.0513, +0.0209] | -0.0361 [-0.0832, +0.0101] |
| tactical_2_6s:fde_m | -0.0503 [-0.1480, +0.0476] | -0.1811 [-0.3149, -0.0556]* |
| tactical_2_6s:speed_mae_mps | -0.0258 [-0.0633, +0.0092] | -0.0323 [-0.0735, +0.0055] |
| tactical_2_6s:along_mae_m | -0.0026 [-0.0338, +0.0315] | -0.0083 [-0.0515, +0.0364] |
| tactical_2_6s:accel_mae_mps2 | -0.0602 [-0.1618, +0.0368] | +0.0221 [-0.0958, +0.1298] |
| tactical_2_6s:cross_mae_m | -0.0178 [-0.0516, +0.0134] | -0.0341 [-0.0773, +0.0012] |
| tactical_2_6s:heading_mae_deg | -0.2967 [-0.5863, -0.0314]* | -0.2681 [-0.6220, +0.0569] |
| tactical_2_6s:yaw_rate_mae_degps | -0.4816 [-1.6755, +0.7638] | -1.4604 [-3.1556, +0.1856] |
| tactical_2_6s:curvature_mae_1pm | -0.0098 [-0.0210, +0.0009] | -0.0048 [-0.0331, +0.0242] |
| tactical_2_6s:dk_hw_abs_err_m | +0.0565 [-0.0741, +0.2145] | +0.0378 [-0.0823, +0.1668] |
| tactical_2_6s:dk_tg_abs_err_s | +0.0316 [+0.0061, +0.0617]*C | +0.0015 [-0.0235, +0.0261] |
| TAC-LAT | -0.0017 [-0.0117, +0.0079] | +0.0050 [-0.0079, +0.0171] |
| TAC-LON | +0.0104 [-0.0025, +0.0234] | +0.0054 [-0.0100, +0.0225] |
