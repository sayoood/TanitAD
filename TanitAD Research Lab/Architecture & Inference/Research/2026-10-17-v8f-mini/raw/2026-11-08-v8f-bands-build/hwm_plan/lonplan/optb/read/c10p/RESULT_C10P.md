# C10' on R35: C10' PASS

Tier T0, MEASURED. 141 EVAL clips, IA-3's folds, acceleration channel, K = 10 (2 s). Zero training, CPU.

Estimator: episode-cluster bootstrap over the 141 eval clips, B 2000, seed 0; answers another draw of episodes with R35's weights and the cross-fitted probe FORM held fixed (not another training run).

| chassis | n anchors | s_swap (point [95% CI]) | CI wholly above 0 | cross-fit R2 | train-clip s_swap (Amend 1) |
|---|---|---|---|---|---|
| R35_s0 | 12827 | 0.2280 [0.2020, 0.2520] | True | 0.243 | 0.201 |
| R35_s1 | 12827 | 0.2223 [0.1952, 0.2473] | True | 0.234 | 0.203 |

Beside (step 0 of the fine-tune, no bar): the term's response on the eval clips.

| chassis | IA-3P-K gain (cross-fit, s_swap) | cycle gain through the frozen train-fit RA | Amend 1 train-clip G_aa |
|---|---|---|---|
| R35_s0 | 0.9902 [0.8756, 1.1289] | 1.2551 [1.1338, 1.3738] | 1.244 |
| R35_s1 | 0.8910 [0.7847, 1.0141] | 1.2460 [1.1253, 1.3664] | 1.202 |

Source artifacts: F:/TanitAD-scratch/actrsp/optb/read/step0/ia3pk_R35_s0.json; F:/TanitAD-scratch/actrsp/optb/read/step0/ia3pk_R35_s1.json
Code: read/ia3pk_read.py (read_core / crossfit_ra / s_swap_from_moments).
