# VERDICT_HIERWHERE2 (PREREG_V8F_PROOF_ADD6B, sha256 3a0a6771; MEASURED, tier T1-kin on logged frames, a diagnostic, never driving performance)

Reader: hier/hierwhere.py unchanged, driven by hier/pair2/hierwhere2.py. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0; 2,399 windows, 141 episodes per seed. One training run per arm per seed and the two arms start from the same pretrained parents as the first pair: a separated interval is necessary and not sufficient. Floors: F = 0.12075 m, tactical replicate floor 0.287 m. STRATEGIC: PAUSED (PI item 25).

## Outcome: **CRIT-PARTIAL**

* P1 (selection-critical difference negative, interval excluding 0, both seeds): NOT HELD: -0.1817 [-0.3054, -0.0686] sep / -0.1196 [-0.2619, +0.0243].
* P2 (critical minus bulk negative at both seeds, separated at one or more): HELD: -0.4997 [-0.9106, -0.1566] sep / -0.6124 [-0.8915, -0.3434] sep.
* P3 (BOTH-MISS the largest term where D is separated): held. P4 (ACCEL and LEFT negative at both seeds): held: ACCEL -0.1361 [-0.3462, +0.0761] / -0.1055 [-0.3500, +0.1573]; LEFT -0.2609 [-0.5484, +0.0407] / -0.2388 [-0.6026, +0.1123].

## DAG0 minus FLAT-DAG-0 (own-state swap off on both sides), ADE6 in metres, s0 / s1

| row | s0 | s1 |
|---|---|---|
| D | +0.0462 [-0.1259, +0.2419] | +0.1597 [+0.0089, +0.3145] sep |
| WITHIN | -0.0117 [-0.0327, +0.0073] | -0.0085 [-0.0406, +0.0188] |
| SELECTION | -0.0404 [-0.0812, +0.0021] | +0.0760 [+0.0223, +0.1350] sep |
| BOTH-MISS | +0.0983 [-0.0596, +0.2917] | +0.0922 [-0.0253, +0.2289] |
| selection-critical (1,305 windows) | -0.1817 [-0.3054, -0.0686] sep | -0.1196 [-0.2619, +0.0243] |
| bulk (1,094 windows) | +0.3180 [-0.0079, +0.7169] | +0.4928 [+0.2488, +0.7552] sep |
| critical minus bulk | -0.4997 [-0.9106, -0.1566] sep | -0.6124 [-0.8915, -0.3434] sep |
| logged STOP | -0.0120 [-0.4390, +0.3512] | -0.1955 [-0.6608, +0.1674] |
| logged DECEL | -0.1470 [-0.2847, -0.0024] sep | +0.0264 [-0.1535, +0.2080] |
| logged ACCEL | -0.1361 [-0.3462, +0.0761] | -0.1055 [-0.3500, +0.1573] |
| logged KEEP | +0.1669 [-0.0954, +0.4821] | +0.3202 [+0.1119, +0.5439] sep |
| logged LEFT | -0.2609 [-0.5484, +0.0407] | -0.2388 [-0.6026, +0.1123] |
| logged RIGHT | -0.3732 [-0.5889, -0.1559] sep | -0.2263 [-0.5080, +0.0627] |
| logged STRAIGHT | +0.1874 [-0.0351, +0.4469] | +0.3041 [+0.1252, +0.4970] sep |
| joint mode error, DAG0 / FLAT-DAG-0 | 0.6536 / 0.6565 | 0.7140 / 0.6044 |

On bulk windows the hierarchy without the swap is worse than its flat control (separated at seed 1); on selection-critical windows it is better (separated at seed 0). Over all windows the difference is not in the hierarchy's favour at either seed.

## Controls

* h1a s0: PASS; (DAG, FLAT-DAG-0) reproduced -0.4973 [-0.6075, -0.3874] against banked -0.4973 [-0.6075, -0.3874], deviation 0.0 m.
* h1a s1: PASS; (DAG, FLAT-DAG-0) reproduced -0.2450 [-0.3553, -0.1316] against banked -0.2450 [-0.3553, -0.1316], deviation 0.0 m.
* s0: h2 PASS (residual 2.1e-17 m); h4c both intervals include 0 for 18 of 20 swap seeds (bar 15): PASS.
* s1: h2 PASS (residual 0.0e+00 m); h4c both intervals include 0 for 15 of 20 swap seeds (bar 15): PASS.
* The reproduction check inside the reader has no banked value for (DAG0, FLAT-DAG-0); it was handed the reader's own value and is self-consistency only, not a control (h1a is the control).

## Beside it, not an outcome of this card: the pass used for h1a, DAG against FLAT-DAG-0 (swap on for the hierarchy only, so the comparison is confounded)

| row | s0 | s1 |
|---|---|---|
| D | -0.4973 [-0.6075, -0.3874] sep | -0.2450 [-0.3553, -0.1316] sep |
| WITHIN | -0.0492 [-0.0772, -0.0234] sep | -0.0540 [-0.0882, -0.0229] sep |
| SELECTION | -0.1316 [-0.1757, -0.0928] sep | -0.0508 [-0.0986, -0.0030] sep |
| BOTH-MISS | -0.3165 [-0.4096, -0.2242] sep | -0.1403 [-0.2202, -0.0596] sep |
| selection-critical | -0.5044 [-0.6658, -0.3546] sep | -0.2623 [-0.4300, -0.0909] sep |
| bulk | -0.4888 [-0.6394, -0.3446] sep | -0.2243 [-0.3464, -0.1045] sep |
| critical minus bulk | -0.0155 [-0.2367, +0.1868] | -0.0381 [-0.2346, +0.1730] |

There the difference is the same size on both window groups (critical minus bulk not separated).

## Limits
Second fine-tune run per seed from the same pretrained parents, recipe without the swap: not an independent pretrain, not a replicate arm with equal flags. The partition thresholds are those of addendum 6. The interval answers another draw of episodes.
