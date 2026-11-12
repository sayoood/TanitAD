# QUICK_OFFSET: logit-offset arithmetic on BANKED STOPDEC files (not the weight edit)

MEASURED, T1-kin first tick (diagnostic), eval-141 STOPDEC windows (117 stop-point, 2,234 other labelled); from `theory/stopbal/{read,dose2/read}/out/stopdec/*.npz` by `bias_quick.py`; no bootstrap.
It previews the offset arithmetic p -> sigmoid(logit p + b) of addendum 4 (frozen) on the parent's banked stop-goal probabilities. ln W1 = 1.5177 (W1 4.5616, BAL-G), ln W2 = 3.0354 (W2 20.81, BAL-G2) from the banked argv. The deciding read is the weight-edit run (`bias_read.py`); this table previews its offset rows, its b_match and the replicate gap only.

Replicate gap |NULL_rep - NULL_s0|: hit 0.0684, fpr 0.0103 (NULL_s0 0.0598 / 0.0054, NULL_rep 0.1282 / 0.0157).

## seed 0 (parent NULL_s0)

| arm | hit | fpr | note |
|---|---|---|---|
| parent NULL_s0 (b = 0) | 0.0598 | 0.0054 | |
| parent + ln W1 (b = 1.5177) | 0.2564 | 0.0416 | offset arithmetic |
| parent + ln W2 (b = 3.0354) | 0.5299 | 0.1468 | offset arithmetic |
| trained BAL-G (G_s0) | 0.2222 | 0.0278 | banked; b_match = 1.086 -> parent offset reads hit 0.2137, fpr 0.0278; hit gap to trained 0.0085 (replicate gap 0.0684) |
| trained BAL-G2 (G2_s0) | 0.4530 | 0.1195 | banked; b_match = 2.760 -> parent offset reads hit 0.4786, fpr 0.1195; hit gap to trained 0.0256 (replicate gap 0.0684) |

## seed 1 (parent NULL_s1)

| arm | hit | fpr | note |
|---|---|---|---|
| parent NULL_s1 (b = 0) | 0.1197 | 0.0085 | |
| parent + ln W1 (b = 1.5177) | 0.3590 | 0.0609 | offset arithmetic |
| parent + ln W2 (b = 3.0354) | 0.5897 | 0.1652 | offset arithmetic |
| trained BAL-G (G_s1) | 0.2906 | 0.0506 | banked; b_match = 1.382 -> parent offset reads hit 0.3077, fpr 0.0506; hit gap to trained 0.0171 (replicate gap 0.0684) |
| trained BAL-G2 (G2_s1) | 0.5299 | 0.1258 | banked; b_match = 2.564 -> parent offset reads hit 0.5214, fpr 0.1258; hit gap to trained 0.0085 (replicate gap 0.0684) |

## Reading (offset arithmetic only)

- G_s0: at ln W the offset reads hit 0.2564 / fpr 0.0416 vs trained 0.2222 / 0.0278 (differences +0.0342 / +0.0139; gaps 0.0684 / 0.0103); b_match 1.086 vs ln W 1.518, hit at b_match 0.2137 vs trained 0.2222 (within the replicate hit gap).
- G2_s0: at ln W the offset reads hit 0.5299 / fpr 0.1468 vs trained 0.4530 / 0.1195 (differences +0.0769 / +0.0273; gaps 0.0684 / 0.0103); b_match 2.760 vs ln W 3.035, hit at b_match 0.4786 vs trained 0.4530 (within the replicate hit gap).
- G_s1: at ln W the offset reads hit 0.3590 / fpr 0.0609 vs trained 0.2906 / 0.0506 (differences +0.0684 / +0.0103; gaps 0.0684 / 0.0103); b_match 1.382 vs ln W 1.518, hit at b_match 0.3077 vs trained 0.2906 (within the replicate hit gap).
- G2_s1: at ln W the offset reads hit 0.5897 / fpr 0.1652 vs trained 0.5299 / 0.1258 (differences +0.0598 / +0.0394; gaps 0.0684 / 0.0103); b_match 2.564 vs ln W 3.035, hit at b_match 0.5214 vs trained 0.5299 (within the replicate hit gap).

Evidence class: MEASURED arithmetic on banked probabilities. It is not a model run and does not read the edited checkpoint; BIAS-EQUALS-DOSE / MATCHES-ROC / DIFFERS are decided by `bias_read.py` only.
