# HEADDIAG: does SAFE-HEAD carry detection information beyond C1's AUC 0.763? (DIAGNOSTIC, not a result)
Evidence class: MEASURED, `E:/Projects/TanitAD-artifacts/headdiag/HEADDIAG.json` (script `headdiag.py`). T1 navtest table, 12,146 tokens / 136 logs, 336 zeroed ships (vetodiag definition, reused). Estimator: paired log-cluster bootstrap (10,000 draws, seed 20260927), tau by 5-fold log CV, exactly as vetodiag Q3. Answers "another draw of LOGS" only; one seed per arm.

**Head application (reproduced as the read does):** `safehead_read.py:246` `D.delta_logits(BL, S, W, b, W0, b0, ROWS=(0,1,3,5))`; rows `ROW_NAMES=(nc,dac,ep,ttc,c,ddc)` (`safehead_data.py:34`) so row 0 = NC, row 1 = DAC. p_fail = 1 - sigma(row0)*sigma(row1) (vetodiag Q2 form). Checks: navtest cache pick == C1 ship 12,146/12,146, cache logits == trk_c1 logits max diff 0.0; read's `picks.npz` C1 == ship 12,146/12,146; step-0 head reproduces vetodiag exactly (AUC 0.7633, veto OOF -0.0255, 16 switched); sh_red (label-permuted) falls to AUC 0.50-0.58.

**Caveat that frames everything:** the held-out rule selected **step 0 for both seeds** (heldout_select.json; read outcome NULL-HELDOUT), i.e. the SELECTED head IS C1. The only informative comparison is the trained steps 250-2000, reported below as descriptive.

## Q1 AUC (zeroed vs non-zeroed, shipped pick)
| arm | step 0 (=C1) | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| s0 head | 0.7633 | 0.7661 | 0.7668 | 0.7684 | 0.7683 |
| s1 head | 0.7633 | 0.7657 | 0.7655 | 0.7679 | 0.7690 |
| red (permuted) | 0.7633 | 0.5849 | 0.5433 | 0.5082 | 0.5025 |

max(head, C1) combination: 0.7633-0.7690, identical to the head to 3 decimals. The head gains +0.003 to +0.006 AUC over C1: no new detection information.

## Q2 veto-only rule, head p_fail, OOF (dPDMS x100 [95 % CI]; n switched / precision)
- Step 0: -0.0255 [-0.0567, -0.0013]; 16 / 0 %.
- s0 step 2000: -0.0298 [-0.0607, -0.0031]; 26 / 3.8 % (1 zeroed hit). s1 step 2000: -0.0391 [-0.0811, +0.0007]; 40 / 7.5 % (3 hits). Recall 0.3-0.9 %.
- Steps 250/500 (s0 209, s1 219/243 switched): -0.0656, -0.0642, -0.0876. Step 1000: -0.0173 (9 sw) / -0.0073 (1 sw).
- EP falls separated (-0.03 to -0.06) in the 2000-step arms; NC/DAC/TTC/C intervals contain 0. In-sample best tau gains at most +0.002. red: 0 switches, 0.0.

## Q3 precision at fixed recall of zeroed (base rate 2.77 %)
| recall | C1 p_fail | s0 head @2000 | s1 head @2000 |
|---|---|---|---|
| 10 % | 20.0 % (170 flagged) | 13.5 % (251) | 14.3 % (237) |
| 25 % | 11.5 % (731) | 13.3 % (632) | 12.5 % (674) |
| 50 % | 8.1 % (2,069) | 8.6 % (1,951) | 8.9 % (1,881) |

## One-line implication
An independent head trained on navtrain features does NOT carry detection information beyond C1: AUC 0.768-0.769 vs 0.763, precision at 10/25/50 % recall equal to C1's within noise, OOF veto negative-to-zero; precision tops out near 10-20 %, so a high-precision veto lever is not supported by this head and is not worth pre-registering on it (a different information source would be needed; the +2.34 perfect-detector ceiling stays unreachable from C1-feature heads).
