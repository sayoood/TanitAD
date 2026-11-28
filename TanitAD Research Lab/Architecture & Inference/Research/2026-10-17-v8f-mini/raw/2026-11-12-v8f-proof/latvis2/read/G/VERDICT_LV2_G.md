# VERDICT_LV2_G: the token-level read of what the rollout transports (card PREREG_V8F_PROOF_ADD38)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: PAUSED (PI item 25).

## Controls (all must pass or no name is printed)

* **k1**: PASS - held-out skill of the token map with the states permuted across frames <= 0.02
* **k2**: PASS - training-fold skill > held-out skill in every fold, and held-out skill of the true present state on the present tokens > 0 in every fold
* **k3**: PASS - with the present state given as the imagined state D is exactly 0
* **k4**: PASS - with the true future state given as the imagined state D equals the headroom and its two-level interval lies above 0
* **k5**: FAIL - the rollout of the tool equals the banked rollout (pred_2s) of the same checkpoint to 0.0001
* **k6**: PASS - the windows, folds and episodes equal those of the r-b read (same ids, same count)

controls_pass: False (k5 as written; all controls as written: False)

## Statistic D (error of the present state minus error of the imagined state; above 0 = the imagination is closer to the true tokens of t + 2.0 s than the copy)

1948 windows, 139 episodes, seeds [0, 1]; moving cells = per window the 10 of 40 with the largest true token change. Units: squared error in the 64-component token space, per cell.

| statistic | per seed | pooled | pooled 95 % interval (two-level) |
|---|---|---|---|
| D, all cells | +1.2297, +1.3117 | +1.2707 | [+1.0887, +1.4518] |
| D, moving cells | +4.7832, +4.8661 | +4.8247 | [+4.4655, +5.1694] |
| headroom, all cells | +3.8752, +3.7454 | +3.8103 | [+3.6013, +4.0144] |
| headroom, moving cells | +9.4516, +9.0786 | +9.2651 | [+8.7620, +9.7596] |
| share of headroom, all | +0.3173, +0.3502 | +0.3335 | [+0.2923, +0.3731] |
| share of headroom, moving | +0.5061, +0.5360 | +0.5207 | [+0.4931, +0.5493] |

D per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +2.2862, +2.3963 | +2.3413 | [+2.0248, +2.6393] |
| far | +1.8352, +2.0091 | +1.9221 | [+1.6792, +2.1520] |
| near | +1.1566, +1.2506 | +1.2036 | [+0.9940, +1.4342] |
| hood | -0.3590, -0.4091 | -0.3841 | [-0.5203, -0.2476] |

## Name

* NONE (a control failed: k5)

Limits: linear maps show what is linearly readable; the token grid is a pooled 4 x 10 read of the picture; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance). No picture is evidence for a claim.
