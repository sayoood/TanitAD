# RECIPE_SELECTION - L2-OFFPROP (build agent, 2026-10-19; MM addendum 14:10)

**Chosen recipe `lam10_lr0.001_ep15_prox0.3`** (lam_bce 10, lr 0.001, epochs 15, prox 0.3; batch 256; Adam, no weight decay). **C-REACH A1-OOF (planted arm, real tokens): seed 0 0.9984, seed 1 0.9992** (bar 0.98 on both). In-fan clause satisfied in the semi-synthetic signal worlds: YES.

Evidence class: MEASURED on REAL tokens with SYNTHETIC labels (a property of the optimiser on a planted model, not of REFe); C-REACH reads the PLANTED arm only. **No real unplanted label was read and no real arm (H, H0, H-null) was trained.**

## 0. Why the recipe was re-selected
The chain's recipe `lam10.0_lr0.0003_ep5` (150 steps per fold) reached C-REACH A1-OOF **0.9048 < 0.98** (`raw/superseded/reach_chain_recipe_lam10_lr3e-4_ep5_FAIL.json`): the in-fan rule (R1 + R2 against P2) selected a recipe that barely moves the head off P2's init, so the real arm would have read FLAT by construction. The MM ruled that C-REACH is a HARD constraint of the selection.

## 1. The rule (committed in `code/design_sweep.py` before the sweep ran)
1. **Hard constraint - C-REACH:** the planted perfect feature (appended column = 10 x sign of the true EP difference, on the real tokens) gives A1-OOF >= 0.98 on **both** training seeds. Seed 0 is read for every setting; seed 1 is read walking down the ranked list (equivalent to filtering on both seeds first). If no setting clears it: **INSTRUMENT DEFECT**, no recipe.
2. **Subject to:** in-fan(H) >= in-fan(H0) - 0.01 in every semi-synthetic SIGNAL world (sigA, sigB), H0 = the same setting with the pairwise weight 0 (same init, folds, seed, minibatch stream).
3. **Maximise** the mean semi-synthetic A1-OOF (pooled lon + lat) of H over the signal worlds. **Ties** (within 0.005 of the best) go to the smaller lambda = pairwise / in-fan weight = 1 / lam_bce (so the larger lam_bce), then the larger prox, then fewer steps, then the smaller lr.
4. **Fallback:** if no C-REACH-passing setting satisfies the in-fan clause, take the one with the best in-fan margin and report that the clause may bind.
5. **Reported, not gating:** the null world (beta 0): A1(H) minus A1(P2) and the GLOBAL-null H-null minus P2.

**Worlds** (`synthetic.semi_synthetic`, REAL tokens): sigA (beta 0.5, alpha 0.25, seed 21), sigB (beta 0.5, alpha 0.5, seed 22), null (beta 0, alpha 0.4, seed 23); P2's in-fan top-8 EP concordance calibrated to 0.889 and P2's A1 at beta 0 to 0.685. In these worlds P2 is near-calibrated on the in-fan labels BY CONSTRUCTION, so H0 sits close to P2 here; in the real run P2 is near in-sample on its own in-fan labels and an out-of-fold H0 will sit lower, which is why the MM replaced the P2 comparator by H0.

## 2. Result
74 settings run (74 expected). **52** clear C-REACH on seed 0; of those **34** also satisfy the in-fan clause in both signal worlds. Rule outcome: C-REACH hard; max semi-synthetic A1-OOF subject to in-fan(H) >= in-fan(H0) - 0.01 in every signal world; ties -> smaller lambda.

P2 reference per world: sigA: A1-OOF 0.735, in-fan top-8 0.889; sigB: A1-OOF 0.785, in-fan top-8 0.889; null: A1-OOF 0.690, in-fan top-8 0.889

| setting | C-REACH s0 | sigA A1 | sigA in-fan H-H0 | sigB A1 | sigB in-fan H-H0 | mean signal A1 | worst H-H0 | in-fan clause | rank | |
|---|---|---|---|---|---|---|---|---|---|---|
| `lam10_lr0.001_ep15_prox0.3` | 0.9984 | 0.7800 | -0.0042 | 0.7899 | -0.0035 | 0.7849 | -0.0042 | yes | 1 | **PICK** |
| `lam10_lr0.001_ep50_prox0.3` | 1.0000 | 0.7796 | -0.0049 | 0.7911 | -0.0022 | 0.7854 | -0.0049 | yes | 2 |  |
| `lam10_lr0.003_ep50_prox0.3` | 1.0000 | 0.7800 | -0.0042 | 0.7914 | -0.0029 | 0.7857 | -0.0042 | yes | 3 |  |
| `lam10_lr0.001_ep5_prox0` | 0.9839 | 0.7881 | -0.0058 | 0.7909 | -0.0067 | 0.7895 | -0.0067 | yes | 4 |  |
| `lam10_lr0.001_ep5_prox0.3` | 0.9834 | 0.7763 | -0.0032 | 0.7906 | -0.0061 | 0.7835 | -0.0061 | yes | 5 |  |
| `lam10_lr0.003_ep5_prox0.3` | 0.9997 | 0.7764 | -0.0036 | 0.7901 | -0.0058 | 0.7833 | -0.0058 | yes | 6 |  |
| `lam10_lr0.003_ep15_prox0.3` | 1.0000 | 0.7782 | -0.0036 | 0.7906 | -0.0045 | 0.7844 | -0.0045 | yes | 7 |  |
| `lam100_lr0.001_ep50_prox0` | 1.0000 | 0.7596 | -0.0016 | 0.7870 | +0.0006 | 0.7733 | -0.0016 | yes | 8 |  |
| `lam100_lr0.003_ep50_prox0` | 1.0000 | 0.7637 | -0.0013 | 0.7877 | -0.0010 | 0.7757 | -0.0013 | yes | 9 |  |
| `lam1_lr0.001_ep5_prox3` | 0.9848 | 0.7608 | -0.0010 | 0.7901 | -0.0067 | 0.7754 | -0.0067 | yes | 10 |  |
| `lam1_lr0.003_ep5_prox3` | 0.9978 | 0.7596 | -0.0026 | 0.7913 | -0.0087 | 0.7755 | -0.0087 | yes | 11 |  |
| `lam1_lr0.001_ep15_prox3` | 0.9960 | 0.7603 | -0.0023 | 0.7905 | -0.0074 | 0.7754 | -0.0074 | yes | 12 |  |
| `lam1_lr0.003_ep15_prox3` | 0.9974 | 0.7589 | -0.0013 | 0.7904 | -0.0087 | 0.7746 | -0.0087 | yes | 13 |  |
| `lam1_lr0.001_ep50_prox3` | 0.9969 | 0.7601 | -0.0029 | 0.7906 | -0.0067 | 0.7753 | -0.0067 | yes | 14 |  |
| `lam1_lr0.003_ep50_prox3` | 0.9980 | 0.7620 | -0.0026 | 0.7917 | -0.0067 | 0.7768 | -0.0067 | yes | 15 |  |
| `lam100_lr0.003_ep5_prox0.3` | 0.9990 | 0.7457 | -0.0006 | 0.7863 | -0.0013 | 0.7660 | -0.0013 | yes | 16 |  |
| `lam100_lr0.001_ep15_prox0.3` | 0.9974 | 0.7496 | +0.0003 | 0.7869 | -0.0006 | 0.7683 | -0.0006 | yes | 17 |  |
| `lam100_lr0.003_ep15_prox0.3` | 1.0000 | 0.7501 | -0.0013 | 0.7871 | +0.0010 | 0.7686 | -0.0013 | yes | 18 |  |
| `lam100_lr0.001_ep50_prox0.3` | 1.0000 | 0.7529 | -0.0010 | 0.7864 | -0.0006 | 0.7697 | -0.0010 | yes | 19 |  |
| `lam100_lr0.003_ep50_prox0.3` | 1.0000 | 0.7537 | -0.0006 | 0.7879 | -0.0010 | 0.7708 | -0.0010 | yes | 20 |  |
| `lam100_lr0.003_ep5_prox0` | 0.9993 | 0.7462 | -0.0010 | 0.7865 | -0.0016 | 0.7664 | -0.0016 | yes | 21 |  |
| `lam100_lr0.001_ep15_prox0` | 0.9979 | 0.7520 | -0.0006 | 0.7871 | -0.0006 | 0.7696 | -0.0006 | yes | 22 |  |
| `lam100_lr0.003_ep15_prox0` | 1.0000 | 0.7525 | -0.0013 | 0.7872 | -0.0003 | 0.7699 | -0.0013 | yes | 23 |  |
| `lam10_lr0.001_ep5_prox3` | 0.9803 | 0.7504 | +0.0000 | 0.7888 | -0.0019 | 0.7696 | -0.0019 | yes | 24 |  |
| `lam10_lr0.003_ep5_prox3` | 0.9957 | 0.7511 | +0.0000 | 0.7895 | -0.0029 | 0.7703 | -0.0029 | yes | 25 |  |
| `lam10_lr0.001_ep15_prox3` | 0.9948 | 0.7520 | -0.0006 | 0.7889 | -0.0026 | 0.7705 | -0.0026 | yes | 26 |  |
| `lam10_lr0.003_ep15_prox3` | 0.9960 | 0.7511 | +0.0003 | 0.7881 | -0.0010 | 0.7696 | -0.0010 | yes | 27 |  |
| `lam10_lr0.001_ep50_prox3` | 0.9957 | 0.7516 | +0.0006 | 0.7883 | -0.0026 | 0.7699 | -0.0026 | yes | 28 |  |
| `lam10_lr0.003_ep50_prox3` | 0.9969 | 0.7523 | -0.0010 | 0.7894 | -0.0026 | 0.7708 | -0.0026 | yes | 29 |  |
| `lam100_lr0.003_ep5_prox3` | 0.9936 | 0.7419 | +0.0006 | 0.7857 | -0.0003 | 0.7638 | -0.0003 | yes | 30 |  |
| `lam100_lr0.001_ep15_prox3` | 0.9921 | 0.7428 | +0.0000 | 0.7862 | +0.0003 | 0.7645 | +0.0000 | yes | 31 |  |
| `lam100_lr0.003_ep15_prox3` | 0.9937 | 0.7423 | +0.0006 | 0.7868 | +0.0003 | 0.7646 | +0.0003 | yes | 32 |  |
| `lam100_lr0.001_ep50_prox3` | 0.9933 | 0.7444 | +0.0000 | 0.7861 | +0.0003 | 0.7652 | +0.0000 | yes | 33 |  |
| `lam100_lr0.003_ep50_prox3` | 0.9943 | 0.7443 | +0.0006 | 0.7871 | -0.0019 | 0.7657 | -0.0019 | yes | 34 |  |
| `lam100_lr0.001_ep15_prox30` | 0.9140 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.001_ep50_prox30` | 0.9159 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.001_ep5_prox0` | 0.9783 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.001_ep5_prox0.3` | 0.9780 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.001_ep5_prox3` | 0.9746 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.001_ep5_prox30` | 0.9160 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.003_ep15_prox30` | 0.9190 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.003_ep50_prox30` | 0.9205 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam100_lr0.003_ep5_prox30` | 0.9167 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.0003_ep15_prox0` | 0.9791 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.0003_ep50_prox0` | 0.9995 | 0.7963 | -0.0179 | 0.7911 | -0.0035 | 0.7937 | -0.0179 | no | - |  |
| `lam10_lr0.001_ep15_prox0` | 0.9991 | 0.7956 | -0.0117 | 0.7913 | -0.0058 | 0.7934 | -0.0117 | no | - |  |
| `lam10_lr0.001_ep15_prox30` | 0.9233 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.001_ep50_prox0` | 1.0000 | 0.8004 | -0.0192 | 0.7921 | -0.0090 | 0.7962 | -0.0192 | no | - |  |
| `lam10_lr0.001_ep50_prox30` | 0.9258 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.001_ep5_prox30` | 0.9254 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.003_ep15_prox0` | 1.0000 | 0.7966 | -0.0146 | 0.7928 | -0.0077 | 0.7947 | -0.0146 | no | - |  |
| `lam10_lr0.003_ep15_prox30` | 0.9258 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.003_ep50_prox0` | 1.0000 | 0.8033 | -0.0257 | 0.7945 | -0.0125 | 0.7989 | -0.0257 | no | - |  |
| `lam10_lr0.003_ep50_prox30` | 0.9281 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam10_lr0.003_ep5_prox0` | 0.9997 | 0.7908 | -0.0117 | 0.7911 | -0.0074 | 0.7910 | -0.0117 | no | - |  |
| `lam10_lr0.003_ep5_prox30` | 0.9281 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam1_lr0.001_ep15_prox0` | 0.9981 | 0.8176 | -0.1056 | 0.7935 | -0.0601 | 0.8056 | -0.1056 | no | - |  |
| `lam1_lr0.001_ep15_prox0.3` | 0.9999 | 0.8018 | -0.0260 | 0.7951 | -0.0212 | 0.7984 | -0.0260 | no | - |  |
| `lam1_lr0.001_ep15_prox30` | 0.9274 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam1_lr0.001_ep50_prox0` | 1.0000 | 0.8192 | -0.1235 | 0.7933 | -0.0736 | 0.8063 | -0.1235 | no | - |  |
| `lam1_lr0.001_ep50_prox0.3` | 1.0000 | 0.8004 | -0.0250 | 0.7946 | -0.0241 | 0.7975 | -0.0250 | no | - |  |
| `lam1_lr0.001_ep50_prox30` | 0.9280 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam1_lr0.001_ep5_prox0` | 0.9880 | 0.8156 | -0.0926 | 0.7953 | -0.0405 | 0.8054 | -0.0926 | no | - |  |
| `lam1_lr0.001_ep5_prox0.3` | 0.9904 | 0.8015 | -0.0279 | 0.7953 | -0.0218 | 0.7984 | -0.0279 | no | - |  |
| `lam1_lr0.001_ep5_prox30` | 0.9279 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam1_lr0.003_ep15_prox0` | 1.0000 | 0.8203 | -0.1180 | 0.7928 | -0.0713 | 0.8065 | -0.1180 | no | - |  |
| `lam1_lr0.003_ep15_prox0.3` | 1.0000 | 0.8015 | -0.0253 | 0.7955 | -0.0215 | 0.7985 | -0.0253 | no | - |  |
| `lam1_lr0.003_ep15_prox30` | 0.9264 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam1_lr0.003_ep50_prox0` | 1.0000 | 0.8175 | -0.1251 | 0.7924 | -0.0765 | 0.8050 | -0.1251 | no | - |  |
| `lam1_lr0.003_ep50_prox0.3` | 1.0000 | 0.8007 | -0.0286 | 0.7946 | -0.0183 | 0.7977 | -0.0286 | no | - |  |
| `lam1_lr0.003_ep50_prox30` | 0.9282 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |
| `lam1_lr0.003_ep5_prox0` | 0.9981 | 0.8179 | -0.1134 | 0.7932 | -0.0556 | 0.8056 | -0.1134 | no | - |  |
| `lam1_lr0.003_ep5_prox0.3` | 0.9999 | 0.8011 | -0.0273 | 0.7946 | -0.0212 | 0.7979 | -0.0273 | no | - |  |
| `lam1_lr0.003_ep5_prox30` | 0.9274 FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |

## 3. The pick
`lam10_lr0.001_ep15_prox0.3`: mean signal A1-OOF 0.7849 (best in the feasible set 0.7895); worst in-fan H-H0 margin -0.0042 (bar -0.01). Settings within 0.005 of the best A1 (the tie set, broken by smaller lambda = larger lam_bce, then larger prox, then fewer steps, then smaller lr): `lam10_lr0.001_ep15_prox0.3`, `lam10_lr0.001_ep50_prox0.3`, `lam10_lr0.003_ep50_prox0.3`, `lam10_lr0.001_ep5_prox0`.

Per-world numbers of the pick (seed 0): sigA: A1(H) 0.7800, A1(H0) 0.7358, in-fan H 0.8856 / H0 0.8898; sigB: A1(H) 0.7899, A1(H0) 0.7852, in-fan H 0.8837 / H0 0.8872.

C-REACH walk-down (seed 1 is read for ranked settings until one clears it): `lam10_lr0.001_ep15_prox0.3`: s0 0.9984, s1 0.9992, PASS.

Null world (beta 0), REPORTED not gating: A1(H) minus A1(P2) = +0.0200; in-fan H minus H0 = +0.0006; GLOBAL-null H-null (r100, r101) A1 0.6904, 0.6899 vs P2 0.6903 (max minus P2 = +0.0001; the VOID rule needs > +0.05 on two of five replicates).

**Interpretation notes.** (a) lambda was read as the pairwise / in-fan ratio, 1 / lam_bce (the MM's own phrase 'lambda pairwise / in-fan weight'); every setting in the tie set has lam_bce 10, so the convention is IMMATERIAL for this pick (the tie is decided by prox, then steps). (b) The tie width 0.005 was fixed in the code before the sweep ran; the MM ruling does not state one. (c) One world seed per world and one training seed per cell in the sweep; the sweep finds a recipe that is not on a cliff and says nothing about REFe. (d) C-REACH's planted feature is APPENDED, so passing it proves the optimiser and the read path can learn an ordering that exists in the input; it says nothing about whether the real token carries one.

