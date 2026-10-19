# RESULT FASTNEG-CENSUS: no row reads COVERAGE; DAC is FEATURE-limited on the fast band, NC has no fast deficit; the registered census outcome is INCONCLUSIVE (2026-10-19)

**Evidence class: MEASURED (ours; artifacts in `raw/`). Tier: SELECTION, fit-side OOF on SEEN scenes (`D-L1-G0-1`); no dev table, no navtest, 0 GPU. Estimator: the paired log-cluster bootstrap, 10,000 resamples, seed 20260927, 117 logs. It answers "would another draw of LOGS say this?" only; the refit's batch-order variance is sampled by two seeds and by the powered null; P2 itself is one training run (`H-ESTIM-SEED-1`) and is IN-SAMPLE on these sets.**

## 0. Headline (the decision-relevant facts first)

* **Outcome (mechanical, parent s5/s6 as amended by A1.5/A1.6): INCONCLUSIVE** (NC = NO DEFICIT, DAC = FEATURE, census not VOID: the powered null fired on **0 of 5** replicates for each row). The robustness read with HEADROW-PROBE's own L OOF gives the same class on both rows. **Committed next arm: re-run ONCE at f = 25 %, cost named: see section 4. It is BLOCKED (needs a feature bank that does not exist).**
* ⚠️ **Integration / decision for the Master Mind (not buried):** the INCONCLUSIVE comes from a cell the parent section 6 does not cover (FEATURE on one row, NO DEFICIT on the other). I resolved it as INCONCLUSIVE in Amendment 1 (A1.6), before any number existed. **If you instead read the table row by row, the two committed arms are available at 0 GPU:** DAC -> PRIVGEOM-PROBE (CPU EST 13.5 h); NC -> the pick-rule threshold audit (CPU minutes). I did NOT apply that reading; it is your call and it needs its own registration.
* **COVERAGE is not supported on either row.** DAC has a large fast-band deficit that does not move with the amount of fast training rows: AUC_F(100 %) - AUC_F(25 %) = **+0.0030 [0.0013, 0.0047]** (seed 0) and **+0.0033 [0.0012, 0.0054]** (seed 1), upper bounds 2 to 3 times below the 0.01 bar, with a 25 / 50 / 100 % series that rises by about 0.0015 per doubling. Our own cold-start refit reads the same (+0.0023, +0.0025), HEADROW-PROBE's 2-layer M arm is NOT better than its linear L arm in F (DAC F-AUC 0.861 / 0.857 vs 0.869 / 0.870), and P2's own frozen DAC row shows the same deficit. NC has no deficit at all.
* **Per row, per seed** (our 100 % L refit; AUC_F = within-token AUC of the fast stratum Q4, Q1-3 = Q1 u Q2 u Q3; deficit bar -0.02, slope bar +0.01):

| row | seed | AUC_F - AUC_Q1-3 | deficit call | AUC_F(100 %) - AUC_F(25 %) | slope call | row class |
|---|---|---|---|---|---|---|
| NC | 0 | +0.0323 [-0.0044, 0.0654] | NO DEFICIT | -0.0002 [-0.0021, 0.0014] | FEATURE (not used: no deficit) | **NO DEFICIT** |
| NC | 1 | +0.0323 [-0.0041, 0.0653] | NO DEFICIT | -0.0002 [-0.0021, 0.0015] | FEATURE (not used) | |
| DAC | 0 | **-0.0718 [-0.1005, -0.0419]** | DEFICIT | +0.0030 [0.0013, 0.0047] | FEATURE | **FEATURE** |
| DAC | 1 | **-0.0712 [-0.0996, -0.0416]** | DEFICIT | +0.0033 [0.0012, 0.0054] | FEATURE | |

* **The fast band is label-RICH, not label-sparse, inside this bank.** Q4 holds 25 % of the rows and **49.7 % [44.7, 54.5] of all unsafe rows** (NC = 0 or DAC = 0), 36.4 % unsafe against 5.5 to 17.5 % in Q1-Q3, and 74 % of all NC = 0 rows. (a) is descriptive; it is not what the rule tests.
* **The null:** 0 of 5 replicate pairs fire for NC and 0 of 5 for DAC; the largest |AUC_F(A) - AUC_F(B)| is 0.0014 (NC) and 0.0006 (DAC), against the 0.01 bar. The slope test therefore resolves differences about 7 times finer than its bar on this rig. It measures batch-order noise only; it cannot see a different training seed of P2 or a different fan.

## 1. Provenance and what was fixed when

| item | value |
|---|---|
| registration | `PREREG_FASTNEG_CENSUS.md` sha256 `b6cece236d5aa77dd412b62341ee1691177fd3dabad30b3c297881ad6d9e5634` (committed at `3a0c038d`, before any probe outcome was read) |
| Amendment 1 | `PREREG_FASTNEG_CENSUS_AMENDMENT1.md` sha256 `f53ae097d153681bd2de07ae552a5d48b0b2d47f996408d1a75588c618a40491`, sealed 07:54 BEFORE `strata.json` (08:04) and before any label was stratified. It fixes only what the parent left open (unit = a bank set; speed definition; AUC pair definition; the L-arm recipe copied from HEADROW-PROBE s3.2; the null's definition; outcome precedence A1.6; the descriptive controls A1.7) |
| `strata.json` | sha256 `67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6`; cut points c25 / c50 / c75 = 0.907631 / 1.045273 / 1.222830; S = 195 of 15,808 sets |
| label-free proof | the strata script reads only the members `log, token, step, rank, ckpt_step, logits` (bank) and `token, step, rank, ckpt_step, traj3` (stores); it refuses any member whose name contains "label"; it was run with `--labels-path` at a non-existent file; **the same script on copies of all 23 bank units and 117 stores with every label member REMOVED gives the identical content digest** (`raw/strata/strata_label_free_proof.json`); the spy test records no label member opened |
| trajectories | found in `C:/Users/Admin/srttc_s0/work/sets/<log>.npz` (`traj3 [S,64,20,3]`, 4 s at 5 Hz); the bank units carry none (lean dump). Join proof: the store's `labels6` equals the bank's EXACTLY on all 15,808 sets (0 mismatches). Independent cross-check: HEADROW-PROBE's separately coded speed export agrees with ours to 4.8e-7 (float32 rounding) on all proposals and base picks; its pick index and labels equal ours |
| data | S0 bank (23 units sha256-verified against `MANIFEST.sha256.json` `6296dfc0...` before `np.load`), f = 10 % (117 logs, 15,808 sets, 8,922 tokens), 5-fold grouped-by-log CV, seeds {0, 1}; P2 rows from `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` (row consistency 1.0e-3 / 1.3e-3) |
| gate | package-level gate PASS 4/4 (G-BANK, G-DVB, G-LIVE, G-SUITE); `raw/gate_token.json`. `stack/scripts/launch_gate.py` was NOT used (it is the stack-trainer gate); this is the precedent the sibling CPU head-fit packages set |
| controls | C-REPRO: our 100 % L refit reproduces HEADROW-PROBE's L OOF logits to a max abs difference of 1.9e-6 (55 to 63 % of entries bit-identical; the rest differ in summation order) => the recipe is the probe's; C-AUC: random scores read 0.497 / 0.495 (NC F / Q1-3) and 0.501 / 0.498 (DAC), oracle scores read 1.0000; tests: 20 synthetic unit tests + 2 end-to-end plumbing tests, all pass; all 21 RED mutation arms of the decision rule caught, the refit's parameters get a non-zero gradient with an exact-zero control |
| disclosure | two format probes printed the first 3 to 4 entries of one `labels6` array each before stratification (all 1.0); no stratum, count or statistic existed |

## 2. What the tables are

Pairs for the within-token AUC are formed inside one SET (a token has 1 to 2 sets, each its own 64-proposal fan), with BOTH members in the stratum (A1.3). Unsafe_row = label == 0, safe_row = label == 1; NC = 0.5 entries are excluded from pairs. Everything below is generated by `code/gen_tables.py` from `raw/results/census.json`; no number is typed by hand.

## T1. Strata (written by the label-free script before any label was read)

cut points of r = v_p / v_b over the 999,232 proposals of the non-S sets: **c25 = 0.907631, c50 = 1.045273, c75 = 1.222830**. S (v_b < 0.5 m/s) = 195 of 15,808 sets.

| stratum | rows |
|---|---|
| S | 12,480 |
| Q1 | 249,808 |
| Q2 | 249,808 |
| Q3 | 249,808 |
| Q4 | 249,808 |

## T2. (a) Label support per stratum (paired log-cluster bootstrap, 10,000, seed 20260927; MEASURED, selection tier)

| stratum | rows | NC = 0 | NC = 0.5 | DAC = 0 | NC = 0 or DAC = 0 | unsafe share of rows | share of ALL unsafe rows |
|---|---|---|---|---|---|---|---|
| S | 12,480 | 2,986 | 62 | 644 | 3,594 | 28.80 % [23.33, 34.55] | 1.96 % [1.15, 2.93] |
| Q1 | 249,808 | 3,847 | 470 | 10,071 | 13,782 | 5.52 % [4.55, 6.60] | 7.53 % [6.18, 8.89] |
| Q2 | 249,808 | 3,372 | 1,358 | 27,869 | 30,841 | 12.35 % [9.80, 15.53] | 16.86 % [14.05, 20.31] |
| Q3 | 249,808 | 7,918 | 2,006 | 36,991 | 43,815 | 17.54 % [14.37, 20.98] | 23.95 % [20.71, 27.33] |
| Q4 | 249,808 | 52,480 | 4,447 | 41,561 | 90,895 | 36.39 % [33.91, 39.07] | 49.69 % [44.66, 54.47] |

## T3. (b)(i) Within-token AUC of P2's FROZEN rows (in-sample for P2; both members of a pair in the stratum; Q1-3 = Q1 u Q2 u Q3)

| row | S | Q1 | Q2 | Q3 | Q4 | Q123 | pairs in F | F - Q1-3 |
|---|---|---|---|---|---|---|---|---|
| NC | 0.9864 [0.9697, 0.9972] | 0.9022 [0.8832, 0.9179] | 0.8720 [0.8267, 0.9131] | 0.8372 [0.7768, 0.9007] | 0.8910 [0.8747, 0.9054] | 0.8617 [0.8355, 0.8920] | 741,635 | **0.0293 [-0.0061, 0.0610]** |
| DAC | 0.9251 [0.8782, 0.9828] | 0.9645 [0.9469, 0.9772] | 0.9661 [0.9465, 0.9810] | 0.9140 [0.8888, 0.9411] | 0.8661 [0.8391, 0.8949] | 0.9408 [0.9224, 0.9582] | 737,611 | **-0.0747 [-0.1034, -0.0445]** |

HEADROW-PROBE OOF file: sha256 `fc86c4c07444e3b9...`, set keys equal to the bank's; cross-checks {"fold_equal_ours": true, "labels6_equal_bank": true, "mean_speed_max_abs_diff_vs_ours": 4.7683666082320997e-07, "p2_logits6_equal_bank": true, "pick_index_equal": true, "pick_mean_speed_max_abs_diff_vs_ours": 4.765901522318927e-07}

### T4. (b)(ii) HEADROW-PROBE L_s0 OOF within-token AUC

| row | S | Q1 | Q2 | Q3 | Q4 | Q123 | pairs in F | F - Q1-3 |
|---|---|---|---|---|---|---|---|---|
| NC | 0.9866 [0.9708, 0.9970] | 0.8970 [0.8779, 0.9134] | 0.8679 [0.8219, 0.9106] | 0.8373 [0.7772, 0.9009] | 0.8905 [0.8741, 0.9050] | 0.8582 [0.8310, 0.8895] | 741,635 | **0.0323 [-0.0044, 0.0654]** |
| DAC | 0.9154 [0.8591, 0.9827] | 0.9636 [0.9460, 0.9765] | 0.9665 [0.9469, 0.9812] | 0.9151 [0.8901, 0.9419] | 0.8690 [0.8423, 0.8970] | 0.9408 [0.9224, 0.9579] | 737,611 | **-0.0718 [-0.1005, -0.0419]** |

### T4. (b)(ii) HEADROW-PROBE L_s1 OOF within-token AUC

| row | S | Q1 | Q2 | Q3 | Q4 | Q123 | pairs in F | F - Q1-3 |
|---|---|---|---|---|---|---|---|---|
| NC | 0.9867 [0.9708, 0.9971] | 0.8964 [0.8771, 0.9128] | 0.8669 [0.8207, 0.9097] | 0.8370 [0.7768, 0.9010] | 0.8905 [0.8741, 0.9049] | 0.8582 [0.8309, 0.8894] | 741,635 | **0.0323 [-0.0041, 0.0653]** |
| DAC | 0.9128 [0.8626, 0.9818] | 0.9640 [0.9467, 0.9766] | 0.9665 [0.9473, 0.9810] | 0.9148 [0.8898, 0.9416] | 0.8696 [0.8434, 0.8973] | 0.9407 [0.9224, 0.9578] | 737,611 | **-0.0712 [-0.0996, -0.0416]** |

### T4. (b)(ii) HEADROW-PROBE M_s0 OOF within-token AUC

| row | S | Q1 | Q2 | Q3 | Q4 | Q123 | pairs in F | F - Q1-3 |
|---|---|---|---|---|---|---|---|---|
| NC | 0.9836 [0.9663, 0.9960] | 0.8886 [0.8672, 0.9081] | 0.8552 [0.8050, 0.8990] | 0.8361 [0.7830, 0.8953] | 0.8846 [0.8678, 0.8992] | 0.8516 [0.8238, 0.8843] | 741,635 | **0.0331 [-0.0061, 0.0672]** |
| DAC | 0.9203 [0.8893, 0.9887] | 0.9654 [0.9459, 0.9789] | 0.9658 [0.9469, 0.9802] | 0.9103 [0.8827, 0.9397] | 0.8610 [0.8357, 0.8886] | 0.9383 [0.9189, 0.9567] | 737,611 | **-0.0773 [-0.1050, -0.0469]** |

### T4. (b)(ii) HEADROW-PROBE M_s1 OOF within-token AUC

| row | S | Q1 | Q2 | Q3 | Q4 | Q123 | pairs in F | F - Q1-3 |
|---|---|---|---|---|---|---|---|---|
| NC | 0.9823 [0.9640, 0.9958] | 0.8884 [0.8652, 0.9077] | 0.8464 [0.7887, 0.8935] | 0.8307 [0.7747, 0.8920] | 0.8854 [0.8682, 0.9007] | 0.8513 [0.8248, 0.8830] | 741,635 | **0.0341 [-0.0052, 0.0677]** |
| DAC | 0.9078 [0.8736, 0.9857] | 0.9631 [0.9439, 0.9766] | 0.9624 [0.9406, 0.9790] | 0.9074 [0.8798, 0.9375] | 0.8569 [0.8300, 0.8860] | 0.9358 [0.9151, 0.9555] | 737,611 | **-0.0789 [-0.1080, -0.0469]** |

## T5. (c) Our L-arm refit, OOF within-token AUC of the fast stratum F = Q4 at F-training-row fractions, per row and seed

| row | seed | F AUC 25 % | F AUC 50 % | F AUC 100 % | Q1-3 AUC 100 % | F - Q1-3 at 100 % | deficit call | F(100) - F(25) | slope call |
|---|---|---|---|---|---|---|---|---|---|
| NC | 0 | 0.8907 [0.8742, 0.9053] | 0.8907 [0.8743, 0.9052] | 0.8905 [0.8741, 0.9050] | 0.8582 [0.8310, 0.8895] | **0.0323 [-0.0044, 0.0654]** | NO_DEFICIT | **-0.0002 [-0.0021, 0.0014]** | FEATURE |
| NC | 1 | 0.8907 [0.8742, 0.9054] | 0.8905 [0.8740, 0.9051] | 0.8905 [0.8741, 0.9049] | 0.8582 [0.8309, 0.8894] | **0.0323 [-0.0041, 0.0653]** | NO_DEFICIT | **-0.0002 [-0.0021, 0.0015]** | FEATURE |
| DAC | 0 | 0.8659 [0.8389, 0.8947] | 0.8675 [0.8407, 0.8960] | 0.8690 [0.8423, 0.8970] | 0.9408 [0.9224, 0.9579] | **-0.0718 [-0.1005, -0.0419]** | DEFICIT | **0.0030 [0.0013, 0.0047]** | FEATURE |
| DAC | 1 | 0.8662 [0.8395, 0.8948] | 0.8677 [0.8413, 0.8959] | 0.8696 [0.8434, 0.8973] | 0.9407 [0.9224, 0.9578] | **-0.0712 [-0.0996, -0.0416]** | DEFICIT | **0.0033 [0.0012, 0.0054]** | FEATURE |

Secondary slopes (descriptive): F(50) - F(25), F(100) - F(50), Q1-3 (100) - Q1-3 (25), and the sensitivity F - Q1-3 with unsafe := label < 1.

| row | seed | F(50) - F(25) | F(100) - F(50) | Q1-3(100) - Q1-3(25) | sensitivity: F - Q1-3 (unsafe := label < 1) |
|---|---|---|---|---|---|
| NC | 0 | 0.0001 [-0.0009, 0.0008] | -0.0003 [-0.0012, 0.0007] | -0.0025 [-0.0040, -0.0009] | 0.0067 [-0.0270, 0.0365] |
| NC | 1 | -0.0002 [-0.0011, 0.0007] | -0.0001 [-0.0010, 0.0008] | -0.0022 [-0.0038, -0.0006] | 0.0069 [-0.0267, 0.0367] |
| DAC | 0 | 0.0016 [0.0008, 0.0025] | 0.0014 [0.0005, 0.0024] | 0.0000 [-0.0002, 0.0003] | -0.0718 [-0.1005, -0.0419] |
| DAC | 1 | 0.0015 [0.0006, 0.0024] | 0.0018 [0.0005, 0.0031] | -0.0000 [-0.0003, 0.0003] | -0.0712 [-0.0996, -0.0416] |

## T6. The registered rule (parent s5) on our 100 % L refit

| row | deficit by seed | slope by seed | row class |
|---|---|---|---|
| NC | ['NO_DEFICIT', 'NO_DEFICIT'] | ['FEATURE', 'FEATURE'] | **NO_DEFICIT** |
| DAC | ['DEFICIT', 'DEFICIT'] | ['FEATURE', 'FEATURE'] | **FEATURE** |

Robustness: the same rule with the DEFICIT test read from HEADROW-PROBE's L OOF (slope from our refits):

| row | deficit by seed | slope by seed | row class |
|---|---|---|---|
| NC | ['NO_DEFICIT', 'NO_DEFICIT'] | ['FEATURE', 'FEATURE'] | NO_DEFICIT |
| DAC | ['DEFICIT', 'DEFICIT'] | ['FEATURE', 'FEATURE'] | FEATURE |

## T7. (c0) Powered null: two independent 100 % L refits differing only in the batch-order seed; slope test AUC_F(A) - AUC_F(B), fires iff lower CI > +0.01 on BOTH seeds

| row | rep | seed 0: A - B | seed 1: A - B | fires |
|---|---|---|---|---|
| NC | 1 | -0.00026 [-0.00076, 0.00025] | 0.00003 [-0.00059, 0.00070] | False |
| NC | 2 | 0.00055 [-0.00019, 0.00131] | 0.00077 [0.00008, 0.00146] | False |
| NC | 3 | -0.00143 [-0.00252, -0.00048] | -0.00051 [-0.00120, 0.00021] | False |
| NC | 4 | 0.00064 [-0.00055, 0.00190] | 0.00029 [-0.00032, 0.00085] | False |
| NC | 5 | -0.00030 [-0.00110, 0.00059] | 0.00061 [0.00007, 0.00109] | False |
| DAC | 1 | 0.00013 [-0.00055, 0.00075] | -0.00003 [-0.00121, 0.00108] | False |
| DAC | 2 | -0.00052 [-0.00168, 0.00057] | -0.00064 [-0.00208, 0.00063] | False |
| DAC | 3 | -0.00055 [-0.00131, 0.00031] | -0.00020 [-0.00113, 0.00073] | False |
| DAC | 4 | -0.00003 [-0.00106, 0.00100] | 0.00038 [-0.00054, 0.00126] | False |
| DAC | 5 | -0.00018 [-0.00130, 0.00080] | 0.00005 [-0.00118, 0.00103] | False |

| row | fires | fire rate | max abs point |
|---|---|---|---|
| NC | 0 of 5 | 0.00 | 0.00143 |
| DAC | 0 of 5 | 0.00 | 0.00064 |

Census VOID by the null: **False**.

## T8. Outcome (A1.6 precedence)

**INCONCLUSIVE** (NC = NO_DEFICIT, DAC = FEATURE, void = False; null: PRESENT). Committed next arm: re-run ONCE at f = 25 % (the next nested size), cost named before the run

Robustness (deficit from the probe's L OOF): INCONCLUSIVE; agrees with the primary: True.

## T9. D-PLACEBO (no authority): Q2 training rows at 25 %, F at 100 %; F's AUC change vs the 100 % refit, next to the F-subsample's own

| row | seed | F AUC (placebo) | F(100) - F(placebo) | F(100) - F(25, F-subsample) |
|---|---|---|---|---|
| NC | 0 | 0.8902 [0.8739, 0.9048] | 0.0002 [-0.0001, 0.0005] | -0.0002 [-0.0021, 0.0014] |
| NC | 1 | 0.8901 [0.8738, 0.9046] | 0.0004 [0.0001, 0.0006] | -0.0002 [-0.0021, 0.0015] |
| DAC | 0 | 0.8696 [0.8433, 0.8974] | -0.0006 [-0.0013, -0.0001] | 0.0030 [0.0013, 0.0047] |
| DAC | 1 | 0.8701 [0.8440, 0.8977] | -0.0006 [-0.0012, -0.0000] | 0.0033 [0.0012, 0.0054] |

## T10. D-COLD (no authority): the same fits from a ZERO start (no stored logit)

| row | seed | F AUC 25 % | F AUC 50 % | F AUC 100 % | F - Q1-3 at 100 % | would read | F(100) - F(25) | would read |
|---|---|---|---|---|---|---|---|---|
| NC | 0 | 0.8903 [0.8734, 0.9056] | 0.8907 [0.8739, 0.9057] | 0.8906 [0.8740, 0.9055] | 0.0323 [-0.0046, 0.0653] | NO_DEFICIT | 0.0002 [-0.0015, 0.0020] | FEATURE |
| NC | 1 | 0.8904 [0.8735, 0.9056] | 0.8904 [0.8737, 0.9054] | 0.8904 [0.8740, 0.9053] | 0.0323 [-0.0044, 0.0650] | NO_DEFICIT | 0.0000 [-0.0016, 0.0017] | FEATURE |
| DAC | 0 | 0.8653 [0.8381, 0.8941] | 0.8664 [0.8393, 0.8949] | 0.8676 [0.8407, 0.8957] | -0.0731 [-0.1020, -0.0424] | DEFICIT | 0.0023 [0.0011, 0.0034] | FEATURE |
| DAC | 1 | 0.8655 [0.8386, 0.8943] | 0.8666 [0.8398, 0.8950] | 0.8680 [0.8414, 0.8960] | -0.0726 [-0.1016, -0.0421] | DEFICIT | 0.0025 [0.0011, 0.0039] | FEATURE |

## T11. Controls

C-REPRO (our 100 % L refit vs HEADROW-PROBE's L OOF logits):

| row_seed | max abs logit diff | mean abs diff | fraction bit-identical |
|---|---|---|---|
| dac_s0 | 1.907e-06 | 1.839e-07 | 0.5499 |
| dac_s1 | 1.907e-06 | 1.612e-07 | 0.6020 |
| nc_s0 | 1.907e-06 | 1.355e-07 | 0.6122 |
| nc_s1 | 1.907e-06 | 1.267e-07 | 0.6265 |

C-AUC instrument (real labels and strata; random scores must bracket 0.5, oracle scores must read 1.0):

| row | random scores F | random scores Q1-3 | oracle F | oracle Q1-3 |
|---|---|---|---|---|
| NC | 0.4974 [0.4929, 0.5020] | 0.4954 [0.4898, 0.4997] | 1.0000 | 1.0000 |
| DAC | 0.5008 [0.4973, 0.5045] | 0.4983 [0.4951, 0.5014] | 1.0000 | 1.0000 |


## 3. Reading the descriptive controls (no decision authority; registered in A1.7 before any number)

* **D-PLACEBO** (75 % of Q2 training rows dropped instead of F's): F's DAC AUC moves -0.0006 (it does not fall), against +0.0030 / +0.0033 for the F-subsample. So the small F-slope is F-specific, not a total-size effect, and it is about 3 times below the +0.01 bar. NC moves 0.0002 to 0.0004 either way.
* **D-COLD** (the same fits from a zero start): the registered L arm starts at P2's row, which was trained on these sets and could mute a coverage slope. It does not: the cold start reads the same classes (NC no deficit; DAC deficit -0.0731 / -0.0726, slope +0.0023 / +0.0025 => FEATURE).
* **Sensitivity (unsafe := label < 1)**: NC F - Q1-3 = +0.0067 [-0.0270, 0.0365] / +0.0069 [-0.0267, 0.0367] (no deficit either way); DAC is unchanged (its labels are binary).
* NC: F reads *better* than Q1-3 (+0.032, interval touches 0). The 74 % of all NC = 0 rows that sit in Q4 are ranked well by the NC row among the fast proposals (F AUC 0.891); the fast-band failure the register describes is not visible in NC's within-set ranking.

## 4. Limits, and the committed next arm with its cost

* SEEN scenes, selection tier, P2 IN-SAMPLE (P2's frozen rows in (b)(i), and the refit starts at P2's row). One rig, two refit seeds. The interval does not see another P2 training run or another fan.
* **What F is.** F = r > 1.2228 relative to P2's frozen base pick inside P2's *own student fan at older checkpoints*. It is "faster than the pick the scorer would make", not the extreme fast, off-expert, lead-ward band SH1 would add. A COVERAGE gap in proposals that the fan never produced is outside what a census on this bank can see; this census says that the rows are not coverage-limited *within* the existing support, where F carries half of all unsafe rows.
* AUC ignores calibration; the pick rule thresholds the rows (parent s7). NC = NO DEFICIT therefore does not clear the pick rule: the parent's NO-DEFICIT arm (the 0-GPU threshold audit) is the test of that.
* **Committed next arm (INCONCLUSIVE): re-run ONCE at f = 25 %. Cost, named: the S0 bank exists only at f = 10 %** (`S0_SUMMARY.json`: 117 logs, 8,922 tokens, 15,808 sets; probed `srttc_s0/work/bank` (23 units), `work/sets` (117 stores), `work/sets_pod/sets/onpolicy_selected.jsonl` (trajectories and labels, no features)). f = 25 % is 292 logs, 23,906 tokens, 42,126 sets (`ranked_fit_logs.tsv`): **14,984 new tokens**. At the measured S0 dump rate of ~1.6 s per token that is **ESTIMATED ~6.7 dev-box GPU-h plus pod staging under the P6 overlap rule** (S0's measured end-to-end wall was 3.6 to 4.5 h for ~8,900 tokens, i.e. ~7 h for the new tokens), then ~3 h wall on two processes for the census fits (2.7 times the sets, ESTIMATED from this run's 72 min). **Blocked on dev-box GPU (another session's battery) and on the P6 overlap guard; nothing else blocks it.** Expected information is low: DAC's slope is already bounded above by 0.0047 and NC has no deficit, so a larger bank would only tighten intervals that are already one-sided relative to the bars.
* **Row-level arms available at 0 GPU if the Master Mind rules the FEATURE + NO DEFICIT cell row by row (needs a registration; I did not run them):** DAC -> PRIVGEOM-PROBE (CPU EST 13.5 h; privileged PDM geometry scalars, which scalars carry the jump names SH3 / SH5); NC -> a pick-rule audit on S0 (unsafe-pick rate of an EP-led composite versus the NC / DAC veto threshold, rows unchanged; CPU minutes).

## 5. Proposed register text (NOT applied; the Master Mind adopts, amends or rejects)

**`GOALS_AND_CLAIMS.md`, update to `H-REFE-FASTNEG-1`:** "UPDATE 2026-10-19: FASTNEG-CENSUS ran (RESULT in `raw/2026-10-19-fastneg-census/`; prereg `b6cece23...`, Amendment 1 `f53ae097...`, strata sha256 `67f43e55...`). **COVERAGE is NOT supported on either row.** NC: no fast-band deficit (AUC_F - AUC_Q1-3 = +0.032 [-0.004, 0.065] on both seeds). DAC: a large fast-band deficit (-0.0718 [-0.1005, -0.0419], -0.0712 [-0.0996, -0.0416]) that does NOT respond to fast training rows (AUC_F 100 % - 25 % = +0.0030 [0.0013, 0.0047], +0.0033 [0.0012, 0.0054], bar +0.01; cold start +0.0023 / +0.0025; HEADROW-PROBE's M arm no better than L in F) => DAC FEATURE-LIMITED. Powered null 0 of 5 fires on each row (max |dAUC_F| 0.0014). Registered outcome (A1.6 precedence): INCONCLUSIVE; committed next arm: re-run once at f = 25 %, which needs 14,984 new S0 tokens (EST ~6.7 dev-box GPU-h) and is blocked. The register's own falsifier ('FEATURE or NO DEFICIT') is met on one row each; the hypothesis as worded (both rows label-support-limited) is NOT supported at f = 10 %. Fast band is label-rich in this bank (Q4: 49.7 % [44.7, 54.5] of all unsafe rows). Open: coverage in proposals the student fan never produced is outside this census. MEASURED (ours), SELECTION tier, SEEN scenes, P2 in-sample; paired log-cluster bootstrap 10,000 / seed 20260927 (another draw of LOGS only)." Status: **PARTIALLY FALSIFIED (COVERAGE not supported); census outcome INCONCLUSIVE by the registered cell mapping.**

**`BACKLOG.md`:** (1) `FASTNEG-CENSUS`: DONE 2026-10-19, outcome INCONCLUSIVE (NC NO DEFICIT, DAC FEATURE, null 0/5), package `raw/2026-10-19-fastneg-census/`. (2) `FASTNEG-F25` (the committed re-run): BLOCKED on dev-box GPU, EST ~6.7 GPU-h for 14,984 new S0 tokens + pod staging under the P6 overlap rule, low expected information. (3) **PI / MM DECISION: read the FEATURE + NO DEFICIT cell row by row?** If yes: `PRIVGEOM-PROBE` on the DAC row (CPU EST 13.5 h) and `PICKRULE-AUDIT` on S0 for NC (CPU minutes, 0 GPU), each with its own prereg before any number. (4) Extend the PRIVGEOM-PROBE prereg so its readout includes the fast stratum F as a named stratum (strata cut points in `strata.json`).

## 6. Manifest

See `MANIFEST.md` (every artifact, its sha256 and where else it lives) and `paths.json` (the landing list). Held only on the dev box: the 30 OOF tables not staged and the 518 MB feature copy, all regenerable by `fastneg_census.py` (deterministic CPU fits, ~72 min wall on two processes) with shas in `raw/fits/*.json` and `raw/inputs/inputs_report.json`.
