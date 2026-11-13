# VERDICT WSP (H-V8F-PROOF-1 SOUP, H-V8F-PROOF-2 P2b), generated from the JSON; MEASURED, T1-kin, eval-141

Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0; ci.py from F:/TanitAD-scratch/v8proof/wsp/tree/taniteval/taniteval/ci.py. It answers 'another draw of EPISODES' only; one member pair and one soup give no run-to-run interval
Controls: 0 problem(s): none

STRATEGIC family: PAUSED (PI item 25).

## H1 SOUP: **SOUP-REALISES**
| primary cell | members (s0 / rep) | member mean -> soup | delta [CI] | class | kappa (ADE) [CI] |
|---|---|---|---|---|---|
| tactical_2_6s:ade_m | 3.2375 / 3.4202 | 3.3289 -> 3.2535 | -0.0754 [-0.1029, -0.0469] | BETTER | 0.738 [0.480, 0.988] |
| full_0_6s:ade_m | 2.2311 / 2.3542 | 2.2926 -> 2.2418 | -0.0508 [-0.0693, -0.0318] | BETTER | 0.738 [0.480, 0.986] |

| band | metric | G_ens | G_soup | kappa [CI] | VC-G1 residual |
|---|---|---|---|---|---|
| operative_0_2s | ADE | 0.00215 | 0.00163 | 0.756 [0.419, 1.058] | 3.6e-15 |
| operative_0_2s | SQ | 0.00306 | 0.00241 | 0.789 [0.403, 1.141] | 3.6e-15 |
| tactical_2_6s | ADE | 0.10218 | 0.07538 | 0.738 [0.480, 0.988] | 2.3e-13 |
| tactical_2_6s | SQ | 2.09291 | 1.34026 | 0.640 [0.304, 1.088] | 2.3e-13 |
| full_0_6s | ADE | 0.06884 | 0.05080 | 0.738 [0.480, 0.986] | 2.3e-13 |
| full_0_6s | SQ | 1.39629 | 0.89431 | 0.640 [0.304, 1.088] | 2.3e-13 |

Named costs (soup minus member mean, separated worse): none; lateral / tactical separated costs: none.
Named gains: ['full_0_6s:ade_m', 'full_0_6s:along_mae_m', 'full_0_6s:fde_m', 'full_0_6s:heading_mae_deg', 'full_0_6s:speed_mae_mps', 'operative_0_2s:accel_mae_mps2', 'operative_0_2s:ade_m', 'operative_0_2s:along_mae_m', 'operative_0_2s:cross_mae_m', 'operative_0_2s:fde_m', 'operative_0_2s:heading_mae_deg', 'operative_0_2s:speed_mae_mps', 'tactical_2_6s:ade_m', 'tactical_2_6s:along_mae_m', 'tactical_2_6s:fde_m', 'tactical_2_6s:heading_mae_deg', 'tactical_2_6s:speed_mae_mps'].
Four families (soup minus member mean; delta [CI]; replicate gap):

| family | cell | delta [CI] | gap |
|---|---|---|---|
| PRIMARY | operative_0_2s:ade_m | -0.0016 [-0.0024, -0.0008] gain | 0.0040 |
| PRIMARY | operative_0_2s:fde_m | -0.0057 [-0.0087, -0.0029] gain | 0.0057 |
| LONGITUDINAL | operative_0_2s:speed_mae_mps | -0.0031 [-0.0046, -0.0018] gain | 0.0117 |
| LONGITUDINAL | operative_0_2s:along_mae_m | -0.0016 [-0.0024, -0.0008] gain | 0.0044 |
| LONGITUDINAL | operative_0_2s:accel_mae_mps2 | -0.0037 [-0.0061, -0.0013] gain | 0.0465 |
| LATERAL | operative_0_2s:cross_mae_m | -0.0005 [-0.0009, -0.0001] gain | 0.0002 |
| LATERAL | operative_0_2s:heading_mae_deg | -0.0103 [-0.0209, -0.0009] gain | 0.0041 |
| LATERAL | operative_0_2s:yaw_rate_mae_degps | -0.0180 [-0.0478, 0.0107] | 0.1647 |
| LATERAL | operative_0_2s:curvature_mae_1pm | -0.0007 [-0.0018, 0.0003] | 0.0048 |
| LONGITUDINAL | operative_0_2s:dk_hw_abs_err_m | 0.0170 [-0.0035, 0.0420] | 0.0009 |
| LONGITUDINAL | operative_0_2s:dk_tg_abs_err_s | 0.0027 [-0.0006, 0.0070] | 0.0001 |
| PRIMARY | tactical_2_6s:ade_m | -0.0754 [-0.1029, -0.0469] gain | 0.1826 |
| PRIMARY | tactical_2_6s:fde_m | -0.2521 [-0.3321, -0.1696] gain | 0.5450 |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps | -0.0485 [-0.0761, -0.0221] gain | 0.1637 |
| LONGITUDINAL | tactical_2_6s:along_mae_m | -0.0679 [-0.0970, -0.0404] gain | 0.2268 |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2 | -0.0462 [-0.1369, 0.0446] | 0.3984 |
| LATERAL | tactical_2_6s:cross_mae_m | -0.0178 [-0.0356, 0.0022] | 0.0184 |
| LATERAL | tactical_2_6s:heading_mae_deg | -0.4437 [-0.8326, -0.1110] gain | 1.8144 |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps | -0.6991 [-2.0320, 0.5798] | 10.0949 |
| LATERAL | tactical_2_6s:curvature_mae_1pm | -0.0063 [-0.0194, 0.0065] | 0.0795 |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m | -0.0079 [-0.1383, 0.1203] | 0.0160 |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s | 0.0052 [-0.0163, 0.0327] | 0.0170 |
| PRIMARY | full_0_6s:ade_m | -0.0508 [-0.0693, -0.0318] gain | 0.1231 |
| PRIMARY | full_0_6s:fde_m | -0.2521 [-0.3321, -0.1696] gain | 0.5450 |
| LONGITUDINAL | full_0_6s:speed_mae_mps | -0.0334 [-0.0519, -0.0156] gain | 0.1131 |
| LONGITUDINAL | full_0_6s:along_mae_m | -0.0458 [-0.0653, -0.0272] gain | 0.1527 |
| LONGITUDINAL | full_0_6s:accel_mae_mps2 | -0.0299 [-0.0898, 0.0291] | 0.2753 |
| LATERAL | full_0_6s:cross_mae_m | -0.0120 [-0.0239, 0.0014] | 0.0123 |
| LATERAL | full_0_6s:heading_mae_deg | -0.4094 [-0.7836, -0.1133] gain | 1.3002 |
| LATERAL | full_0_6s:yaw_rate_mae_degps | -0.9169 [-2.0788, 0.0976] | 6.6665 |
| LATERAL | full_0_6s:curvature_mae_1pm | -0.0069 [-0.0180, 0.0035] | 0.0541 |
| TACTICAL | TAC-LAT | 0.0033 [-0.0060, 0.0125] | 0.0158 |
| TACTICAL | TAC-LON | 0.0085 [-0.0098, 0.0263] | 0.0054 |
| STRATEGIC | | PAUSED (PI item 25) | |

## H2 P2b dose linearity: **DOSE-NONLINEAR**
| metric | NULL_s0 | G2_s0 (dose 2) | G_s0 (dose 1, trained) | DOSE050 | replicate gap | between | within gap of dose 1 |
|---|---|---|---|---|---|---|---|
| hit | 0.0598 | 0.4530 | 0.2222 | 0.2479 | 0.0684 | True | True |
| fpr | 0.0054 | 0.1195 | 0.0278 | 0.0385 | 0.0103 | True | False |

| tag | hit [CI] | fpr [CI] |
|---|---|---|
| NULL_s0 | 0.0598 [0.0083, 0.1406] | 0.0054 [0.0018, 0.0104] |
| NULL_rep | 0.1282 [0.0385, 0.2400] | 0.0157 [0.0071, 0.0264] |
| G_s0 | 0.2222 [0.0970, 0.3726] | 0.0278 [0.0160, 0.0421] |
| G2_s0 | 0.4530 [0.3167, 0.6000] | 0.1195 [0.0833, 0.1586] |
| DOSE050 | 0.2479 [0.1196, 0.3971] | 0.0385 [0.0231, 0.0565] |
| DOSE100 | 0.4530 [0.3167, 0.6000] | 0.1195 [0.0833, 0.1586] |

Control DOSE100 = G2_s0: {'DOSE100_decisions_equal_G2_s0': True, 'DOSE100_goal_prob_max_abs_diff_to_G2_s0': 0.0}
Own dumps vs banked STOPDEC decisions: {'NULL_s0': {'decisions_equal_banked': True, 'banked_file': 'E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/read/out/stopdec/stopdec_BAL-NULL_s0.npz'}, 'NULL_rep': {'decisions_equal_banked': True, 'banked_file': 'E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/read/out/stopdec/stopdec_BAL-NULL_rep.npz'}, 'G_s0': {'decisions_equal_banked': True, 'banked_file': 'E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/read/out/stopdec/stopdec_BAL-G_s0.npz'}, 'G2_s0': {'decisions_equal_banked': True, 'banked_file': 'E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/dose2/read/out/stopdec/stopdec_BAL-G2_s0.npz'}}
