verdict (mechanical): **S1-WEAK** - both fail, in-sample IoU 0.7864 < 0.85

| bar | value | 95 % CI | bar | holds |
|---|---|---|---|---|
| S1-P1 | 0.7802 | [0.7587, 0.7974] | >= 0.8 | False |
| S1-Q | 0.8332 | [0.7951, 0.8691] | >= 0.9 | False |
| S1-F | P1 lower 0.5120, Q lower 0.0891 | - | both > 0 | True |
| C-SHUF | 0.4468 | - | < 0.5 | True |
| C-FLIP | 0.7056 | - | < 0.8 | True |

per city (S1-P1 [0,40) IoU, HA-s0): LV 0.8914, BOS 0.7394, PIT 0.7234, SG 0.6960

| band IoU, HA-s0 (city-reweighted) | value | 95 % CI |
|---|---|---|
| 0-20 m | 0.8090 | [0.7886, 0.8300] |
| 20-40 m | 0.7498 | [0.7216, 0.7686] |
| 40-72 m | 0.6608 | [0.6288, 0.6822] |
| 0-40 m | 0.7802 | [0.7587, 0.7974] |

HA-SHUF IoU [0,40): 0.4468 CI [0.4223, 0.4703]; HA-SHUF S1-Q 0.6677
HA-s0 near-boundary SDF channel MAE 1.6297 m, RMS 2.3626 m (cells 1236457)
sigma_b 6.5905 m; true-field 2 m AUC 0.9408; y-flipped S1-Q 0.7056

### HB-s0: verdict (mechanical, frozen v1 table): **S1-BOUND** - S1-P1 holds, S1-Q fails; A5 row **X-BOUND**

| paired X minus REF | S1-P1 delta | 95 % CI | S1-Q delta | 95 % CI | X-LIFT |
|---|---|---|---|---|---|
| HB-s0 - HA-s0 | 0.0311 | [0.0224, 0.0359] | 0.0043 | [-0.0046, 0.0119] | False |
