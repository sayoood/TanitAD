# L2-FRESH-STEPA read

**SEEN-SCENE, SELECTION tier (L1 A1.3); token-fresh, LOG-SHARED: B is NOT a fresh-log test (its 24 logs are Step A's and the heads' fit logs); ONE training seed per head (H-ESTIM-SEED-1)**

Estimator: paired log-cluster bootstrap over 24 logs, B=10000, seed 20260927; it answers 'another draw of LOGS' only (H-ESTIM-SEED-1: one training seed per head; the two L2 seeds are near-replicates).

| head | A1 all | A1 lon | A1 lat | A2 | A3 | A4 | A5 | in-fan top-8 EP | A1 - P2 (paired) | A3 - P2 (paired) | outcome: interval_aware (DECIDES) / literal (record only) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L2_s0 | 0.840 [0.795, 0.891] | 0.839 [0.792, 0.892] | 0.867 [0.741, 0.946] | 0.846 [0.784, 0.919] | 0.719 [0.670, 0.766] | 0.016 [0.011, 0.023] | 0.010 [0.007, 0.014] | 0.916 [0.880, 0.947] | 0.122 [0.074, 0.164] | -0.048 [-0.067, -0.029] | O3 / O3 |
| L2_s1 | 0.846 [0.803, 0.894] | 0.845 [0.799, 0.896] | 0.864 [0.740, 0.943] | 0.846 [0.784, 0.919] | 0.706 [0.655, 0.755] | 0.019 [0.013, 0.029] | 0.010 [0.007, 0.014] | 0.917 [0.881, 0.949] | 0.127 [0.078, 0.170] | -0.061 [-0.086, -0.036] | O1 / O1 |
| P2 | 0.719 [0.647, 0.806] | 0.714 [0.638, 0.805] | 0.867 [0.746, 0.943] | 0.846 [0.784, 0.919] | 0.767 [0.717, 0.814] | 0.014 [0.010, 0.020] | 0.008 [0.005, 0.011] | 0.866 [0.815, 0.925] | - | - | O3 / O2 |
| LOGOOF_s0 | 0.841 [0.798, 0.888] | 0.840 [0.794, 0.890] | 0.864 [0.738, 0.943] | 0.846 [0.784, 0.919] | 0.732 [0.685, 0.777] | 0.016 [0.012, 0.024] | 0.010 [0.007, 0.013] | 0.913 [0.875, 0.948] | 0.122 [0.072, 0.166] | -0.035 [-0.052, -0.022] | reported-only, no outcome |
| LOGOOF_s1 | 0.838 [0.793, 0.889] | 0.838 [0.789, 0.891] | 0.860 [0.738, 0.937] | 0.846 [0.784, 0.919] | 0.727 [0.678, 0.774] | 0.015 [0.012, 0.020] | 0.010 [0.007, 0.013] | 0.914 [0.875, 0.947] | 0.120 [0.071, 0.163] | -0.040 [-0.058, -0.025] | reported-only, no outcome |

**L2 outcome (interval_aware, most severe of the two seeds): O3** (per seed {'L2_s0': 'O3', 'L2_s1': 'O1'}). Literal reading, for the record only, decides nothing: O3.

**A3 (reported, ungated):** paired dA3 upper CI end per final-head seed {'L2_s0': -0.0290567454033583, 'L2_s1': -0.03559068774853207}; pre-stated consequence: EP-RELIAB-CLASS RISK RECORDED: the Step B amendment must report the aggregate concordance of the search outputs beside dPDMS.

**Log-OOF companion (reported-only, never gated):** reported-only, never gated (MM ruling 2026-10-19)
Fold heads 00ce7e84d0fa; rows LOGOOF_* above.

Gates: A1 >= 0.80, A4 <= 0.10, A5 <= 0.05 (L1 prereg section 4, Amendment 1 `interval_aware`).
STRATEGIC: PAUSED (PI ruling 2026-09-19, ITEM 25); reported as PAUSED, not as a pass
Exposure: "SEEN-SCENE, SELECTION tier (L1 A1.3); token-fresh, LOG-SHARED: B is NOT a fresh-log test (its 24 logs are Step A's and the heads' fit logs); ONE training seed per head (H-ESTIM-SEED-1)"
