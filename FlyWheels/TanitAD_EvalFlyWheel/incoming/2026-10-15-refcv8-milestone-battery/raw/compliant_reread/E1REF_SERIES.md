# E1REF_SERIES -- BAR-R8-E1 / E3 under the ha0_ext_pose reference (D-R8-E1-REF), capped plans on ceiling-compliant windows

**Reference:** ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19) -- refcv8's own causal prior (measured v0, backward-difference acceleration, kappa0 = omega0 / max(v0, floor) from the backward yaw difference at t0). The registered `ha0_ext` read (steer-channel kappa0, curvature: inadmissible) stays BESIDE and is never dropped.

**DISCLOSURE: this reference switch FOLLOWED the step-30,000 reads.** It was ruled by the PI on 2026-10-19 (~05:40 Berlin) after the 30k cells below had been seen; every row before step 40,000 is an EARLY READ re-scored under it, not a pre-registered test. Steps >= 40,000 are the first reads scored under it by construction.

**TIER:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). **ESTIMATOR / QUESTION:** episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run. Single training seed, two inference seeds; cell = `os - reference` ADE, negative = better than the reference, verdict = bar_E (separated at BOTH inference seeds, NOT PROVEN inside 2x the inference floor). Capped plans (A-R8-CLAMP-1/2), compliant windows. The paired bootstrap answers 'another draw of EPISODES' only; `H-ESTIM-SEED-1` and the inference floor are NOT covered by the interval.

| tree | inference seed | E1 S2 `os - pose` | E1 S2 `os - ha0_ext` (beside) | E3 S6 `os - pose` | E3 S6 `os - ha0_ext` (beside) | verdict E1 / E3 (pose) | verdict E1 / E3 (ha0_ext, beside) | inference floor S2 / S6 (m) | n windows S2 / S6 |
|---|---|---|---|---|---|---|---|---|---|
| step5000_unc | 0 | +0.0440 [+0.0224, +0.0667] | +0.0491 [+0.0273, +0.0718] | -0.6652 [-0.9481, -0.3964] | -0.6365 [-0.9194, -0.3714] | FAILED / PASS | FAILED / PASS | 0.0011 / 0.0217 | 4159 / 3251 |
| step5000_unc | 1 | +0.0451 [+0.0229, +0.0686] | +0.0502 [+0.0279, +0.0737] | -0.6436 [-0.9265, -0.3714] | -0.6148 [-0.8948, -0.3493] | FAILED / PASS | FAILED / PASS | 0.0011 / 0.0217 | 4159 / 3251 |
| step5000ema_unc | 0 | -0.0170 [-0.0327, -0.0015] | -0.0119 [-0.0269, +0.0037] | -1.0243 [-1.3274, -0.7539] | -0.9956 [-1.2920, -0.7328] | PASS / PASS | FAILED / PASS | 0.0010 / 0.0025 | 4159 / 3251 |
| step5000ema_unc | 1 | -0.0180 [-0.0331, -0.0030] | -0.0129 [-0.0272, +0.0020] | -1.0269 [-1.3297, -0.7620] | -0.9981 [-1.2975, -0.7411] | PASS / PASS | FAILED / PASS | 0.0010 / 0.0025 | 4159 / 3251 |
| step10500_unc | 0 | +0.0568 [+0.0401, +0.0736] | +0.0619 [+0.0450, +0.0786] | -0.6563 [-0.9641, -0.3667] | -0.6275 [-0.9337, -0.3463] | FAILED / PASS | FAILED / PASS | 0.0017 / 0.0078 | 4159 / 3251 |
| step10500_unc | 1 | +0.0551 [+0.0381, +0.0720] | +0.0602 [+0.0434, +0.0774] | -0.6641 [-0.9728, -0.3740] | -0.6354 [-0.9396, -0.3521] | FAILED / PASS | FAILED / PASS | 0.0017 / 0.0078 | 4159 / 3251 |
| step10500ema_unc | 0 | -0.0189 [-0.0331, -0.0049] | -0.0137 [-0.0273, +0.0003] | -1.0105 [-1.3104, -0.7440] | -0.9817 [-1.2813, -0.7238] | PASS / PASS | FAILED / PASS | 0.0004 / 0.0210 | 4159 / 3251 |
| step10500ema_unc | 1 | -0.0185 [-0.0324, -0.0040] | -0.0134 [-0.0271, +0.0011] | -0.9895 [-1.2914, -0.7241] | -0.9607 [-1.2597, -0.6978] | PASS / PASS | FAILED / PASS | 0.0004 / 0.0210 | 4159 / 3251 |
| step15000_unc | 0 | +0.0238 [+0.0066, +0.0408] | +0.0289 [+0.0125, +0.0458] | -0.6361 [-0.9590, -0.3238] | -0.6073 [-0.9278, -0.3011] | FAILED / PASS | FAILED / PASS | 0.0010 / 0.0094 | 4159 / 3251 |
| step15000_unc | 1 | +0.0228 [+0.0066, +0.0395] | +0.0279 [+0.0121, +0.0442] | -0.6455 [-0.9672, -0.3324] | -0.6168 [-0.9335, -0.3089] | FAILED / PASS | FAILED / PASS | 0.0010 / 0.0094 | 4159 / 3251 |
| step15000ema_unc | 0 | -0.0159 [-0.0309, -0.0001] | -0.0108 [-0.0248, +0.0049] | -0.9222 [-1.2351, -0.6390] | -0.8935 [-1.1955, -0.6150] | FAILED / PASS | FAILED / PASS | 0.0007 / 0.0008 | 4159 / 3251 |
| step15000ema_unc | 1 | -0.0152 [-0.0296, +0.0002] | -0.0101 [-0.0241, +0.0051] | -0.9230 [-1.2366, -0.6370] | -0.8942 [-1.1985, -0.6137] | FAILED / PASS | FAILED / PASS | 0.0007 / 0.0008 | 4159 / 3251 |
| step20000_unc | 0 | +0.0042 [-0.0115, +0.0200] | +0.0093 [-0.0061, +0.0250] | -0.8089 [-1.1180, -0.5290] | -0.7801 [-1.0870, -0.5060] | FAILED / PASS | FAILED / PASS | 0.0001 / 0.0069 | 4159 / 3251 |
| step20000_unc | 1 | +0.0043 [-0.0117, +0.0200] | +0.0094 [-0.0062, +0.0253] | -0.8020 [-1.1015, -0.5256] | -0.7733 [-1.0690, -0.5023] | FAILED / PASS | FAILED / PASS | 0.0001 / 0.0069 | 4159 / 3251 |
| step20000ema_unc | 0 | -0.0179 [-0.0322, -0.0029] | -0.0127 [-0.0265, +0.0021] | -0.8807 [-1.2036, -0.5862] | -0.8519 [-1.1654, -0.5641] | PASS / PASS | FAILED / PASS | 0.0000 / 0.0001 | 4159 / 3251 |
| step20000ema_unc | 1 | -0.0179 [-0.0320, -0.0025] | -0.0127 [-0.0263, +0.0022] | -0.8808 [-1.2011, -0.5867] | -0.8520 [-1.1691, -0.5612] | PASS / PASS | FAILED / PASS | 0.0000 / 0.0001 | 4159 / 3251 |
| step30000_unc | 0 | +0.0123 [-0.0043, +0.0299] | +0.0174 [+0.0014, +0.0349] | -0.6028 [-0.9347, -0.2914] | -0.5741 [-0.9035, -0.2681] | FAILED / PASS | FAILED / PASS | 0.0014 / 0.0110 | 4159 / 3251 |
| step30000_unc | 1 | +0.0109 [-0.0056, +0.0283] | +0.0160 [+0.0002, +0.0329] | -0.6138 [-0.9520, -0.2995] | -0.5851 [-0.9183, -0.2759] | FAILED / PASS | FAILED / PASS | 0.0014 / 0.0110 | 4159 / 3251 |
| step30000ema_unc | 0 | -0.0136 [-0.0286, +0.0018] | -0.0085 [-0.0229, +0.0065] | -0.7823 [-1.1159, -0.4748] | -0.7535 [-1.0832, -0.4506] | FAILED / PASS | FAILED / PASS | 0.0009 / 0.0106 | 4159 / 3251 |
| step30000ema_unc | 1 | -0.0127 [-0.0272, +0.0024] | -0.0076 [-0.0216, +0.0073] | -0.7717 [-1.1024, -0.4582] | -0.7430 [-1.0699, -0.4363] | FAILED / PASS | FAILED / PASS | 0.0009 / 0.0106 | 4159 / 3251 |

