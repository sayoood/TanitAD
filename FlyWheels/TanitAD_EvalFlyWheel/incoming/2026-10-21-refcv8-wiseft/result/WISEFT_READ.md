# R8-WISEFT + R8-CKPT-TRAJ read (registered e8911b1)

Tier: NavSim v1 open-loop PDMS, registered navtrain sample (2,089 tokens / 30 logs), Thor CPU scorer. Controls all pass: **True**.

| arm | PDMS s0 | PDMS s1 | minus EMA-48,852 s0 [interval] | s1 | floor | admissible |
|---|---|---|---|---|---|---|
| lam 0.0 | 71.02 | 71.36 | +7.63 [+4.77, +10.65] | +8.33 | 0.71 | True |
| lam 0.25 | 74.12 | 73.19 | +10.73 [+7.59, +13.95] | +10.17 | 0.56 | True |
| lam 0.5 | 76.33 | 76.44 | +12.94 [+10.54, +15.66] | +13.42 | 0.47 | True |
| lam 0.75 | 71.12 | 71.03 | +7.73 [+6.06, +9.75] | +8.01 | 0.28 | True |
| lam 1 (EMA-48,852) | 63.39 | 63.02 | -- | -- | -- | -- |

**Stage 1:** ADMISSIBLE -- lambda* = 0.5; next: stage 2: ONE navtest read of theta_lambda* (registered)

| EMA snapshot | PDMS s0 | minus PRIOR (73.84) [interval] | minus EMA-48,852 | straight / turn vs PRIOR |
|---|---|---|---|---|
| 5,000 | 74.61 | +0.77 [-3.23, +4.81] | +11.22 [+8.15, +14.61] | -5.92 / +15.77 |
| 15,000 | 71.12 | -2.72 [-8.19, +3.35] | +7.72 [+5.16, +10.88] | -11.71 / +17.39 |
| 30,000 | 63.73 | -10.10 [-16.23, -4.01] | +0.34 [-1.16, +2.16] | -18.36 / +8.37 |
| 40,000 | 64.49 | -9.35 [-15.23, -3.62] | +1.10 [+0.11, +2.20] | -17.91 / +9.81 |
| 48,852 | 63.39 | -10.45 [-16.21, -4.92] |  | -19.79 / +10.47 |

Trajectory reading (diagnostic): earliest snapshot below PRIOR (separated) = 30000; snapshots beating EMA-48,852 (separated) = [5000, 15000, 40000].
