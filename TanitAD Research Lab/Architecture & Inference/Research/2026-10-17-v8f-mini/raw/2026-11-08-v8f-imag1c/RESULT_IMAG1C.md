# RESULT_IMAG1C: is `lead_gap` readable in the encoded true future, and does the imagined gap beat Copy-Last, when IA-2 is read on frames with CAPLADDER's probe?
Trigger: CAPLADDER read outcome O-NONE (`../capladder/RESULT_CAPLADDER.md`). Spec `IMPL_CHOICES_IMAG1C.md` (sha256 `ed00b75c...`, in `IMPL_CHOICES_IMAG1C.sha256`, written before any IMAG1C row). Tier **T0** (teacher-forced diagnostic of a world model; never a driving claim). STRATEGIC: **PAUSED** (PI item 25). Evidence class on every number: **MEASURED (ours)**, artifacts `RESULT_IMAG1C.json`, `OLDPROBE_IMAG1C.json` (IMAG1B column: `../imag1b/RESULT_IMAG1B.json`, INHERITED as printed there and reproduced here by check iii). Estimators: episode-cluster bootstrap B 2,000 over 141 bank episodes, and the refit-under-permutation reference R = 199 (IMAG1B's two-part rule, unchanged). Window frames: 7,481 (old subset 1,976). Written by `report_imag1c.py`.

## Banner: VALID
| control | result | detail |
|---|---|---|
| 2 not-future (Oracle minus not-future separated in every cell with a non-UNREADABLE label; `speed` is the control-only target) | pass | 30 cells, 0 failed [] |
| 3 calibration (20 draws scored as if real, at most 5 % separated per kind) | pass | lead_gap|Oracle 0/360 = 0.0 %; READ3|Oracle 0/1080 = 0.0 %; lead_gap|pred-vs-Copy above 6/360 = 1.7 %; READ3|pred-vs-Copy above 20/1080 = 1.9 %; lead_gap|pred-vs-Copy below 0/360 = 0.0 %; READ3|pred-vs-Copy below 4/1080 = 0.4 %; READ3|pred-vs-Copy both sides 24/2160 = 1.1 %; lead_gap|pred-vs-Copy both sides 6/720 = 0.8 % |
| 4 planted / deranged / dual-equals-primal / regression arm (`TEST_IMAG1C.log`, synthetic) | pass | planted channel READABLE and IMAGINES; Predicted = Copy gives COPIES; deranged and Gaussian worlds give no READABLE; control 2 fails when not-future = Oracle |
| reproduction (i) rebuilt Z0 / ORA / PRED of the old windows vs the IMAG1B ctx | pass | max abs diff 3.6e-06 (bound 1e-4); (ii) labels equal exactly |
| reproduction (iii) IMAG1B's probe on the old windows vs `RESULT_IMAG1B.json` | pass | max abs diff of NMSE 1.2e-08 (bound 1e-4) |

IA-0 health (predicted k = 1 retrieves the encoded true future, derangement reference): L1 healthy, L2 healthy, L3 healthy.

## `lead_gap` on frames (deciding). NMSE: 1 = the scoring-fold mean; READABLE = bootstrap upper bound < 1 AND point below the 2.5th percentile of the 199 refit-permutation NMSEs of that row
Per seed: frames n (episodes), Oracle NMSE [95 % episode-cluster CI], reference 2.5th, Copy-Last NMSE, Predicted NMSE, Predicted minus Copy-Last NMSE [95 % CI] (negative = the imagined gap is closer). Label per IMAG1B's rule.

| level | k | label | seed | n (eps) | Oracle NMSE | ref 2.5th | Copy NMSE | Pred NMSE | Pred - Copy NMSE | pred vs Copy |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | UNREADABLE | s0 | 2027 (53) | 0.633 [0.447, 0.889] | 1.010 | 0.627 | 0.637 | 0.010 [-0.021, 0.037] | not separated |
| L1 | 1 | UNREADABLE | s1 | 2027 (53) | 0.690 [0.468, 1.006] | 1.031 | 0.690 | 0.663 | -0.027 [-0.078, 0.009] | not separated |
| L1 | 2 | UNREADABLE | s0 | 2024 (53) | 0.635 [0.447, 0.888] | 1.008 | 0.627 | 0.633 | 0.006 [-0.036, 0.047] | not separated |
| L1 | 2 | UNREADABLE | s1 | 2024 (53) | 0.697 [0.473, 1.014] | 1.029 | 0.697 | 0.655 | -0.042 [-0.114, 0.009] | not separated |
| L1 | 3 | UNREADABLE | s0 | 2020 (54) | 0.637 [0.454, 0.890] | 1.007 | 0.627 | 0.634 | 0.006 [-0.049, 0.063] | not separated |
| L1 | 3 | UNREADABLE | s1 | 2020 (54) | 0.705 [0.482, 1.024] | 1.027 | 0.708 | 0.656 | -0.052 [-0.143, 0.014] | not separated |
| L2 | 1 | COPIES | s0 | 2024 (53) | 0.622 [0.448, 0.885] | 1.019 | 0.622 | 0.622 | 0.000 [-0.023, 0.023] | not separated |
| L2 | 1 | COPIES | s1 | 2024 (53) | 0.575 [0.415, 0.801] | 1.015 | 0.570 | 0.574 | 0.003 [-0.012, 0.019] | not separated |
| L2 | 2 | COPIES | s0 | 2016 (54) | 0.629 [0.456, 0.887] | 1.015 | 0.635 | 0.639 | 0.005 [-0.039, 0.052] | not separated |
| L2 | 2 | COPIES | s1 | 2016 (54) | 0.576 [0.416, 0.794] | 1.010 | 0.568 | 0.577 | 0.009 [-0.022, 0.040] | not separated |
| L2 | 3 | COPIES | s0 | 2007 (55) | 0.633 [0.458, 0.889] | 1.012 | 0.649 | 0.659 | 0.010 [-0.053, 0.079] | not separated |
| L2 | 3 | COPIES | s1 | 2007 (55) | 0.577 [0.421, 0.789] | 1.003 | 0.569 | 0.584 | 0.015 [-0.030, 0.060] | not separated |
| L3 | 1 | COPIES | s0 | 2016 (54) | 0.628 [0.462, 0.862] | 0.984 | 0.623 | 0.629 | 0.006 [-0.024, 0.036] | not separated |
| L3 | 1 | COPIES | s1 | 2016 (54) | 0.708 [0.522, 0.962] | 0.977 | 0.687 | 0.698 | 0.011 [-0.017, 0.039] | not separated |
| L3 | 2 | COPIES | s0 | 1997 (55) | 0.641 [0.474, 0.877] | 0.975 | 0.643 | 0.648 | 0.004 [-0.050, 0.062] | not separated |
| L3 | 2 | COPIES | s1 | 1997 (55) | 0.732 [0.543, 0.988] | 0.968 | 0.704 | 0.712 | 0.008 [-0.035, 0.054] | not separated |
| L3 | 3 | COPIES | s0 | 1981 (55) | 0.667 [0.503, 0.899] | 0.972 | 0.711 | 0.694 | -0.016 [-0.100, 0.069] | not separated |
| L3 | 3 | COPIES | s1 | 1981 (55) | 0.752 [0.567, 0.997] | 0.956 | 0.754 | 0.740 | -0.014 [-0.071, 0.048] | not separated |

Labels on frames, `lead_gap`: UNREADABLE 3, COPIES 6, IMAGINES 0, WORSE-THAN-COPY 0 of 9 (level, k) cells.

## Beside: the older window-subset result and the attribution (no bar)
| level | k | IMAG1B label (old windows, lambda 10, 2 folds, ridge + MLP) | IMAG1B Oracle NMSE s0 / s1 | CAPLADDER ridge on the old windows: label | Oracle NMSE s0 / s1 | IMAG1B probe on all frames: Oracle NMSE s0 / s1 | CAPLADDER ridge on all frames: label | Oracle NMSE s0 / s1 |
|---|---|---|---|---|---|---|---|---|
| L1 | 1 | UNREADABLE | 0.780 / 0.929 | COPIES | 0.640 / 0.692 | 0.762 / 0.934 | UNREADABLE | 0.633 / 0.690 |
| L1 | 2 | UNREADABLE | 0.777 / 0.973 | UNREADABLE | 0.650 / 0.715 | 0.761 / 0.939 | UNREADABLE | 0.635 / 0.697 |
| L1 | 3 | UNREADABLE | 0.745 / 0.941 | UNREADABLE | 0.622 / 0.702 | 0.760 / 0.945 | UNREADABLE | 0.637 / 0.705 |
| L2 | 1 | UNREADABLE | 1.483 / 1.074 | COPIES | 0.623 / 0.582 | 1.477 / 1.066 | COPIES | 0.622 / 0.575 |
| L2 | 2 | UNREADABLE | 1.486 / 1.060 | COPIES | 0.638 / 0.566 | 1.475 / 1.066 | COPIES | 0.629 / 0.576 |
| L2 | 3 | UNREADABLE | 1.482 / 1.080 | COPIES | 0.635 / 0.582 | 1.468 / 1.072 | COPIES | 0.633 / 0.577 |
| L3 | 1 | UNREADABLE | 1.060 / 1.098 | COPIES | 0.634 / 0.709 | 1.034 / 1.095 | COPIES | 0.628 / 0.708 |
| L3 | 2 | UNREADABLE | 1.076 / 1.125 | COPIES | 0.653 / 0.736 | 1.051 / 1.124 | COPIES | 0.641 / 0.732 |
| L3 | 3 | UNREADABLE | 1.111 / 1.143 | COPIES | 0.686 / 0.763 | 1.081 / 1.149 | COPIES | 0.667 / 0.752 |

Frames per row (seed s0): old subset L1k1 529, L1k2 530, L1k3 532, L2k1 530, L2k2 534, L2k3 526, L3k1 534, L3k2 529, L3k3 524 windows with a finite horizon `lead_gap`; all frames L1k1 2027, L1k2 2024, L1k3 2020, L2k1 2024, L2k2 2016, L2k3 2007, L3k1 2016, L3k2 1997, L3k3 1981. Rows whose horizon frame has no `lead_gap` are dropped; the other windows of the 7,481 are not in the row.

## Beside, no bar: `lead_closing` (labels printed for reading only)
| level | k | label | seed | n (eps) | Oracle NMSE | ref 2.5th | Copy NMSE | Pred NMSE | Pred - Copy NMSE | pred vs Copy |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | UNREADABLE | s0 | 1867 (52) | 1.022 [0.944, 1.116] | 1.009 | 1.023 | 1.020 | -0.003 [-0.008, 0.001] | not separated |
| L1 | 1 | UNREADABLE | s1 | 1867 (52) | 0.997 [0.927, 1.096] | 1.019 | 1.000 | 0.999 | -0.001 [-0.006, 0.004] | not separated |
| L1 | 2 | UNREADABLE | s0 | 1898 (52) | 1.020 [0.944, 1.115] | 1.007 | 1.022 | 1.016 | -0.006 [-0.015, 0.002] | not separated |
| L1 | 2 | UNREADABLE | s1 | 1898 (52) | 0.997 [0.927, 1.096] | 1.017 | 1.000 | 0.999 | -0.001 [-0.010, 0.007] | not separated |
| L1 | 3 | UNREADABLE | s0 | 1926 (52) | 1.020 [0.945, 1.113] | 1.006 | 1.020 | 1.011 | -0.009 [-0.022, 0.003] | not separated |
| L1 | 3 | UNREADABLE | s1 | 1926 (52) | 0.996 [0.928, 1.096] | 1.017 | 0.999 | 0.998 | -0.001 [-0.013, 0.011] | not separated |
| L2 | 1 | UNREADABLE | s0 | 1898 (52) | 1.041 [0.970, 1.122] | 1.021 | 1.042 | 1.037 | -0.005 [-0.014, 0.003] | not separated |
| L2 | 1 | UNREADABLE | s1 | 1898 (52) | 1.031 [0.937, 1.170] | 1.008 | 1.035 | 1.033 | -0.002 [-0.011, 0.008] | not separated |
| L2 | 2 | UNREADABLE | s0 | 1924 (52) | 1.040 [0.970, 1.119] | 1.017 | 1.035 | 1.027 | -0.008 [-0.023, 0.006] | not separated |
| L2 | 2 | UNREADABLE | s1 | 1924 (52) | 1.030 [0.940, 1.165] | 1.006 | 1.032 | 1.028 | -0.003 [-0.021, 0.013] | not separated |
| L2 | 3 | UNREADABLE | s0 | 1914 (53) | 1.036 [0.968, 1.114] | 1.015 | 1.031 | 1.022 | -0.010 [-0.029, 0.009] | not separated |
| L2 | 3 | UNREADABLE | s1 | 1914 (53) | 1.025 [0.939, 1.154] | 1.006 | 1.025 | 1.020 | -0.005 [-0.029, 0.018] | not separated |
| L3 | 1 | UNREADABLE | s0 | 1924 (52) | 1.058 [0.995, 1.126] | 1.016 | 1.053 | 1.048 | -0.006 [-0.015, 0.003] | not separated |
| L3 | 1 | UNREADABLE | s1 | 1924 (52) | 1.024 [0.959, 1.105] | 1.012 | 1.022 | 1.018 | -0.004 [-0.012, 0.004] | not separated |
| L3 | 2 | UNREADABLE | s0 | 1903 (54) | 1.054 [0.991, 1.124] | 1.011 | 1.046 | 1.039 | -0.007 [-0.021, 0.008] | not separated |
| L3 | 2 | UNREADABLE | s1 | 1903 (54) | 1.020 [0.957, 1.097] | 1.003 | 1.016 | 1.007 | -0.009 [-0.025, 0.007] | not separated |
| L3 | 3 | UNREADABLE | s0 | 1890 (53) | 1.052 [0.990, 1.122] | 1.007 | 1.041 | 1.032 | -0.009 [-0.026, 0.011] | not separated |
| L3 | 3 | UNREADABLE | s1 | 1890 (53) | 1.020 [0.960, 1.100] | 1.000 | 1.016 | 0.998 | -0.018 [-0.041, 0.005] | not separated |

## Beside, no bar: `ttc` (labels printed for reading only)
| level | k | label | seed | n (eps) | Oracle NMSE | ref 2.5th | Copy NMSE | Pred NMSE | Pred - Copy NMSE | pred vs Copy |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | UNREADABLE | s0 | 1867 (52) | 1.106 [1.003, 1.238] | 1.034 | 1.107 | 1.100 | -0.007 [-0.012, -0.003] | not separated |
| L1 | 1 | UNREADABLE | s1 | 1867 (52) | 1.222 [1.002, 1.507] | 1.014 | 1.224 | 1.207 | -0.017 [-0.037, -0.001] | not separated |
| L1 | 2 | UNREADABLE | s0 | 1898 (52) | 1.102 [1.002, 1.233] | 1.029 | 1.103 | 1.090 | -0.013 [-0.021, -0.005] | not separated |
| L1 | 2 | UNREADABLE | s1 | 1898 (52) | 1.215 [1.000, 1.498] | 1.011 | 1.216 | 1.188 | -0.028 [-0.062, -0.001] | not separated |
| L1 | 3 | UNREADABLE | s0 | 1926 (52) | 1.097 [0.999, 1.226] | 1.024 | 1.097 | 1.080 | -0.018 [-0.030, -0.007] | not separated |
| L1 | 3 | UNREADABLE | s1 | 1926 (52) | 1.206 [0.996, 1.485] | 1.006 | 1.209 | 1.172 | -0.037 [-0.082, -0.002] | not separated |
| L2 | 1 | UNREADABLE | s0 | 1898 (52) | 1.079 [1.006, 1.176] | 1.007 | 1.080 | 1.068 | -0.013 [-0.020, -0.005] | not separated |
| L2 | 1 | UNREADABLE | s1 | 1898 (52) | 1.076 [0.961, 1.224] | 1.006 | 1.074 | 1.066 | -0.009 [-0.019, -0.003] | not separated |
| L2 | 2 | UNREADABLE | s0 | 1924 (52) | 1.072 [1.002, 1.169] | 1.002 | 1.073 | 1.052 | -0.021 [-0.035, -0.008] | not separated |
| L2 | 2 | UNREADABLE | s1 | 1924 (52) | 1.069 [0.961, 1.213] | 1.004 | 1.066 | 1.050 | -0.016 [-0.033, -0.005] | not separated |
| L2 | 3 | UNREADABLE | s0 | 1914 (53) | 1.065 [0.999, 1.162] | 0.998 | 1.069 | 1.041 | -0.027 [-0.044, -0.010] | not separated |
| L2 | 3 | UNREADABLE | s1 | 1914 (53) | 1.059 [0.960, 1.199] | 1.003 | 1.057 | 1.035 | -0.022 [-0.045, -0.007] | not separated |
| L3 | 1 | UNREADABLE | s0 | 1924 (52) | 1.061 [0.991, 1.156] | 1.005 | 1.064 | 1.039 | -0.026 [-0.040, -0.010] | not separated |
| L3 | 1 | UNREADABLE | s1 | 1924 (52) | 1.093 [0.949, 1.283] | 0.999 | 1.100 | 1.086 | -0.014 [-0.030, -0.002] | not separated |
| L3 | 2 | UNREADABLE | s0 | 1903 (54) | 1.054 [0.990, 1.148] | 0.995 | 1.067 | 1.020 | -0.047 [-0.074, -0.019] | not separated |
| L3 | 2 | UNREADABLE | s1 | 1903 (54) | 1.072 [0.951, 1.247] | 0.993 | 1.089 | 1.058 | -0.031 [-0.059, -0.008] | not separated |
| L3 | 3 | UNREADABLE | s0 | 1890 (53) | 1.046 [0.986, 1.134] | 0.991 | 1.071 | 1.005 | -0.066 [-0.105, -0.027] | not separated |
| L3 | 3 | UNREADABLE | s1 | 1890 (53) | 1.055 [0.955, 1.207] | 0.990 | 1.076 | 1.035 | -0.041 [-0.077, -0.015] | not separated |

## Reading
* **Readable on frames:** the encoded true future's `lead_gap` is READABLE (both seeds) in 6 of 9 (level, k) cells: L2 k1, L2 k2, L2 k3, L3 k1, L3 k2, L3 k3. IMAG1B's rows were UNREADABLE in 9 of 9.
* **The imagined lead gap does NOT beat Copy-Last by the two-part rule at any (level, k):** no cell is IMAGINES. Readable cells that are COPIES: L2 k1, L2 k2, L2 k3, L3 k1, L3 k2, L3 k3 (the Predicted minus Copy-Last NMSE intervals are in the table).
* **What the labels change about IMAG1B's row:** IMAG1B's `lead_gap` row read UNREADABLE in 9 of 9 cells, which fed its trigger E3 (CAPLADDER). On frames with CAPLADDER's probe the row reads 3 UNREADABLE, 6 COPIES, 0 IMAGINES, 0 WORSE-THAN-COPY. The attribution table separates frames from probe (the IMAG1B probe on all frames against the CAPLADDER ridge on the old windows).
* Tier T0; one probe family (ridge); two seeds of one recipe (M2b); eval-141 frames from 141 episodes (56 with a lead). STRATEGIC: PAUSED. The 16-step windows start at every cache step and overlap, so rows share frames; the episode-cluster bootstrap accounts for that and not for training variance (`H-ESTIM-SEED-1`).

## Files
All under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/imag1c/`: `IMPL_CHOICES_IMAG1C.md/.sha256`, `imag1c_lib.py`, `build_imag1c.py` (+ `.report.json`, `.log`), `run_imag1c.py`, `oldprobe_imag1c.py`, `report_imag1c.py`, `test_imag1c_controls.py`, `TEST_IMAG1C.log`, `RESULT_IMAG1C.json`, `LABELS_IMAG1C.json`, `OLDPROBE_IMAG1C.json`, `draws_imag1c/` (220 draws), `DELETIONS_IMAG1C.log`, `HANDOFF_IMAG1C.md`. Model-side arrays: `F:/TanitAD-scratch/imag1c_ctx.pkl`.
