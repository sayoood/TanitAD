# PRIVGEOM-CLEAN tables (generated 2026-10-20T01:36:57+02:00; 10867 sets, 76 logs; paired log-cluster bootstrap, B = 10000, seed 20260927)

## DAC, sign-clean 9-column block: final **INCONCLUSIVE** (raw INCONCLUSIVE; companion NULL; null fires [False, False, False, False, False]; void_null False; void_inputs False; admit_ok True; power fires False; small-reading True)

| seed | F-AUC L | F-AUC L+G (primary LGc) | F-AUC G-only (centred) | Delta_F primary | call | Delta_Q1-3 | G-only - L (F) | companion (LGu) Delta_F | companion call |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8985 [0.8672, 0.9285] | 0.9092 [0.8793, 0.9377] | 0.8256 [0.7882, 0.8646] | +0.0107 [+0.0004, +0.0252] | INCONCLUSIVE | +0.0056 [+0.0024, +0.0097] | -0.0729 [-0.0990, -0.0452] | +0.0078 [-0.0043, +0.0197] | NULL |
| 1 | 0.8987 [0.8673, 0.9285] | 0.9088 [0.8792, 0.9375] | 0.8258 [0.7882, 0.8647] | +0.0101 [+0.0001, +0.0244] | INCONCLUSIVE | +0.0057 [+0.0022, +0.0099] | -0.0729 [-0.0992, -0.0452] | +0.0075 [-0.0044, +0.0193] | NULL |

C-NULL (Delta_F of the within-set-permuted, centred block; fires iff the CALL's own rule holds on the replicate's seed):

- rep0 (seed 0): -0.0008 [-0.0014, -0.0003] fires=False
- rep1 (seed 1): -0.0005 [-0.0011, -0.0000] fires=False
- rep2 (seed 0): -0.0003 [-0.0006, +0.0001] fires=False
- rep3 (seed 1): -0.0009 [-0.0016, -0.0003] fires=False
- rep4 (seed 0): -0.0004 [-0.0007, -0.0001] fires=False

C-POWER (plant sigma 1.25, Delta_F by seed): +0.0159 [+0.0071, +0.0257] / +0.0157 [+0.0070, +0.0256]

F pairs: 404997

