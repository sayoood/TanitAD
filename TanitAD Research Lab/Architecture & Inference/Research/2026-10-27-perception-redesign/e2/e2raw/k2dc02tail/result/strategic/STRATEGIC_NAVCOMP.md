# STRATEGIC family (route following) -- refcv9 E2, report-only

STRATEGIC (route following vs the TRUE nav command; refcv9 E2 arms are --no-strategic)

Tier: T1 self-action open loop. Estimator: episode-cluster bootstrap; cross-arm = PAIRED on windows informative for both.
Caveat: one training run per arm: a separated interval answers 'another draw of EPISODES' only (H-ESTIM-SEED-1).
Control: PASS: REF - REF reads exactly 0.0 on every readout and seed.

| arm:seed | readout | n inf | compliance TRUE (mean [lo, hi]) | true-shuffled | true-zero | true-flipped |
|---|---|---|---|---|---|---|
| K0t:s0 | plan | 270 | +0.6778 [+0.6071, +0.7424] | +0.2148 | +0.2519 | +0.4185 |
| K0t:s0 | gstr | 459 | +0.3007 [+0.1401, +0.4583] | +0.0000 | +0.0000 | +0.0000 |
| K0t:s0 | anchor | 220 | +0.6818 [+0.6034, +0.7544] | +0.1818 | +0.2091 | +0.3273 |
| K0t:s1 | plan | 270 | +0.6963 [+0.6277, +0.7588] | - | - | - |
| K0t:s1 | gstr | 459 | +0.3007 [+0.1401, +0.4583] | - | - | - |
| K0t:s1 | anchor | 249 | +0.6948 [+0.6203, +0.7640] | - | - | - |
| K2DC02t:s0 | plan | 270 | +0.6704 [+0.5861, +0.7511] | +0.2296 | +0.2481 | +0.3815 |
| K2DC02t:s0 | gstr | 459 | +0.2854 [+0.1207, +0.4538] | +0.0000 | +0.0000 | +0.0000 |
| K2DC02t:s0 | anchor | 222 | +0.7117 [+0.6239, +0.7921] | +0.1982 | +0.2297 | +0.3243 |
| K2DC02t:s1 | plan | 270 | +0.6333 [+0.5443, +0.7121] | - | - | - |
| K2DC02t:s1 | gstr | 459 | +0.2854 [+0.1207, +0.4538] | - | - | - |
| K2DC02t:s1 | anchor | 250 | +0.6600 [+0.5760, +0.7381] | - | - | - |
| K2DCt:s0 | plan | 270 | +0.6741 [+0.5789, +0.7638] | +0.2370 | +0.2704 | +0.4037 |
| K2DCt:s0 | gstr | 459 | +0.3072 [+0.1513, +0.4622] | +0.0000 | +0.0000 | +0.0000 |
| K2DCt:s0 | anchor | 186 | +0.5968 [+0.4952, +0.6995] | +0.1720 | +0.1720 | +0.2312 |
| K2DCt:s1 | plan | 270 | +0.6593 [+0.5684, +0.7500] | - | - | - |
| K2DCt:s1 | gstr | 459 | +0.3072 [+0.1513, +0.4622] | - | - | - |
| K2DCt:s1 | anchor | 234 | +0.5897 [+0.5000, +0.6814] | - | - | - |

## ARM - K0t, paired on windows informative for both

| pair:readout:seed | delta [lo, hi] | separated | n |
|---|---|---|---|
| K2DC02t-K0t:plan:s0 | -0.0074 [-0.0517, +0.0388] | False | 270 |
| K2DC02t-K0t:gstr:s0 | -0.0153 [-0.0774, +0.0321] | False | 459 |
| K2DC02t-K0t:anchor:s0 | +0.0205 [-0.0201, +0.0600] | False | 195 |
| K2DC02t-K0t:plan:s1 | -0.0630 [-0.1345, -0.0041] | True | 270 |
| K2DC02t-K0t:gstr:s1 | -0.0153 [-0.0774, +0.0321] | False | 459 |
| K2DC02t-K0t:anchor:s1 | -0.0169 [-0.0676, +0.0301] | False | 236 |
| K2DCt-K0t:plan:s0 | -0.0037 [-0.0647, +0.0550] | False | 270 |
| K2DCt-K0t:gstr:s0 | +0.0065 [-0.0231, +0.0380] | False | 459 |
| K2DCt-K0t:anchor:s0 | -0.0242 [-0.0724, +0.0310] | False | 165 |
| K2DCt-K0t:plan:s1 | -0.0370 [-0.1013, +0.0268] | False | 270 |
| K2DCt-K0t:gstr:s1 | +0.0065 [-0.0231, +0.0380] | False | 459 |
| K2DCt-K0t:anchor:s1 | -0.0682 [-0.1140, -0.0137] | True | 220 |
