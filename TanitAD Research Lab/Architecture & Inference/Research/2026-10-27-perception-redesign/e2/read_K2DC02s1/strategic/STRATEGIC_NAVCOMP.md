# STRATEGIC family (route following) -- refcv9 E2, report-only

STRATEGIC (route following vs the TRUE nav command; refcv9 E2 arms are --no-strategic)

Tier: T1 self-action open loop. Estimator: episode-cluster bootstrap; cross-arm = PAIRED on windows informative for both.
Caveat: one training run per arm: a separated interval answers 'another draw of EPISODES' only (H-ESTIM-SEED-1).
Control: PASS: REF - REF reads exactly 0.0 on every readout and seed.

| arm:seed | readout | n inf | compliance TRUE (mean [lo, hi]) | true-shuffled | true-zero | true-flipped |
|---|---|---|---|---|---|---|
| K0:s0 | plan | 270 | +0.7037 [+0.6259, +0.7778] | +0.2296 | +0.2889 | +0.4519 |
| K0:s0 | gstr | 459 | +0.2985 [+0.1391, +0.4570] | +0.0000 | +0.0000 | +0.0000 |
| K0:s0 | anchor | 213 | +0.6995 [+0.6119, +0.7763] | +0.1737 | +0.2066 | +0.3239 |
| K0:s1 | plan | 270 | +0.6889 [+0.6124, +0.7610] | - | - | - |
| K0:s1 | gstr | 459 | +0.2985 [+0.1391, +0.4570] | - | - | - |
| K0:s1 | anchor | 251 | +0.6932 [+0.6127, +0.7653] | - | - | - |
| K2DC:s0 | plan | 270 | +0.6926 [+0.6077, +0.7706] | +0.2185 | +0.2667 | +0.4370 |
| K2DC:s0 | gstr | 459 | +0.3007 [+0.1366, +0.4617] | +0.0000 | +0.0000 | +0.0000 |
| K2DC:s0 | anchor | 228 | +0.6667 [+0.5764, +0.7489] | +0.1535 | +0.2061 | +0.3114 |
| K2DC:s1 | plan | 270 | +0.6741 [+0.5948, +0.7468] | - | - | - |
| K2DC:s1 | gstr | 459 | +0.3007 [+0.1366, +0.4617] | - | - | - |
| K2DC:s1 | anchor | 260 | +0.6846 [+0.6065, +0.7612] | - | - | - |
| K2DC02:s0 | plan | 270 | +0.6963 [+0.6178, +0.7660] | +0.2222 | +0.2778 | +0.4296 |
| K2DC02:s0 | gstr | 459 | +0.2941 [+0.1307, +0.4546] | +0.0000 | +0.0000 | +0.0000 |
| K2DC02:s0 | anchor | 232 | +0.6681 [+0.5797, +0.7457] | +0.1422 | +0.2026 | +0.3060 |
| K2DC02:s1 | plan | 270 | +0.6815 [+0.6073, +0.7481] | - | - | - |
| K2DC02:s1 | gstr | 459 | +0.2941 [+0.1307, +0.4546] | - | - | - |
| K2DC02:s1 | anchor | 261 | +0.6897 [+0.6199, +0.7550] | - | - | - |
| K2DC02s1:s0 | plan | 270 | +0.7037 [+0.6217, +0.7825] | +0.2333 | +0.2741 | +0.4444 |
| K2DC02s1:s0 | gstr | 459 | +0.2941 [+0.1326, +0.4510] | +0.0000 | +0.0000 | +0.0000 |
| K2DC02s1:s0 | anchor | 218 | +0.7064 [+0.6233, +0.7825] | +0.1697 | +0.2156 | +0.3303 |
| K2DC02s1:s1 | plan | 270 | +0.6778 [+0.6000, +0.7549] | - | - | - |
| K2DC02s1:s1 | gstr | 459 | +0.2941 [+0.1326, +0.4510] | - | - | - |
| K2DC02s1:s1 | anchor | 252 | +0.6865 [+0.6058, +0.7625] | - | - | - |

## ARM - K0, paired on windows informative for both

| pair:readout:seed | delta [lo, hi] | separated | n |
|---|---|---|---|
| K2DC-K0:plan:s0 | -0.0111 [-0.0543, +0.0330] | False | 270 |
| K2DC-K0:gstr:s0 | +0.0022 [-0.0177, +0.0179] | False | 459 |
| K2DC-K0:anchor:s0 | -0.0101 [-0.0542, +0.0263] | False | 199 |
| K2DC-K0:plan:s1 | -0.0148 [-0.0494, +0.0178] | False | 270 |
| K2DC-K0:gstr:s1 | +0.0022 [-0.0177, +0.0179] | False | 459 |
| K2DC-K0:anchor:s1 | -0.0040 [-0.0359, +0.0283] | False | 248 |
| K2DC02-K0:plan:s0 | -0.0074 [-0.0422, +0.0246] | False | 270 |
| K2DC02-K0:gstr:s0 | -0.0044 [-0.0272, +0.0197] | False | 459 |
| K2DC02-K0:anchor:s0 | -0.0150 [-0.0502, +0.0155] | False | 200 |
| K2DC02-K0:plan:s1 | -0.0074 [-0.0361, +0.0202] | False | 270 |
| K2DC02-K0:gstr:s1 | -0.0044 [-0.0272, +0.0197] | False | 459 |
| K2DC02-K0:anchor:s1 | +0.0000 [-0.0300, +0.0290] | False | 249 |
| K2DC02s1-K0:plan:s0 | +0.0000 [-0.0340, +0.0317] | False | 270 |
| K2DC02s1-K0:gstr:s0 | -0.0044 [-0.0301, +0.0169] | False | 459 |
| K2DC02s1-K0:anchor:s0 | +0.0207 [-0.0184, +0.0592] | False | 193 |
| K2DC02s1-K0:plan:s1 | -0.0111 [-0.0489, +0.0261] | False | 270 |
| K2DC02s1-K0:gstr:s1 | -0.0044 [-0.0301, +0.0169] | False | 459 |
| K2DC02s1-K0:anchor:s1 | +0.0082 [-0.0244, +0.0423] | False | 244 |
