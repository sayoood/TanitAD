# PRIVGEOM-CLEAN tables (generated 2026-10-19T21:22:45+02:00 REHEARSAL, NOT A RESULT; 4941 sets, 41 logs; paired log-cluster bootstrap, B = 10000, seed 20260927)

## DAC, sign-clean 9-column block: final **CONFIRM** (raw CONFIRM; companion CONFIRM; null fires [False, False, False, False, False]; void_null False; void_inputs False; admit_ok True; power fires True; small-reading False)

| seed | F-AUC L | F-AUC L+G (primary LGc) | F-AUC G-only (centred) | Delta_F primary | call | Delta_Q1-3 | G-only - L (F) | companion (LGu) Delta_F | companion call |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8322 [0.7929, 0.8722] | 0.8674 [0.8305, 0.8990] | 0.8042 [0.7661, 0.8435] | +0.0352 [+0.0194, +0.0495] | CONFIRM | +0.0155 [+0.0074, +0.0222] | -0.0281 [-0.0572, +0.0092] | +0.0276 [+0.0130, +0.0430] | CONFIRM |
| 1 | 0.8323 [0.7928, 0.8721] | 0.8676 [0.8306, 0.8991] | 0.8030 [0.7645, 0.8432] | +0.0352 [+0.0198, +0.0494] | CONFIRM | +0.0154 [+0.0074, +0.0220] | -0.0293 [-0.0587, +0.0078] | +0.0278 [+0.0135, +0.0427] | CONFIRM |

C-NULL (Delta_F of the within-set-permuted, centred block; fires iff the CALL's own rule holds on the replicate's seed):

- rep0 (seed 0): -0.0007 [-0.0014, -0.0002] fires=False
- rep1 (seed 1): -0.0001 [-0.0005, +0.0002] fires=False
- rep2 (seed 0): -0.0010 [-0.0020, -0.0003] fires=False
- rep3 (seed 1): -0.0007 [-0.0012, -0.0002] fires=False
- rep4 (seed 0): -0.0002 [-0.0009, +0.0003] fires=False

C-POWER (plant sigma 1.25, Delta_F by seed): +0.0276 [+0.0182, +0.0374] / +0.0276 [+0.0181, +0.0376]

F pairs: 332614

