# VERDICT_LV2_RR: the token-level read of what the rollout transports (card PREREG_V8F_PROOF_ADD38)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: PAUSED (PI item 25).

## Controls (all must pass or no name is printed)

* **k1**: PASS - held-out skill of the token map with the states permuted across frames <= 0.02
* **k2**: PASS - training-fold skill > held-out skill in every fold, and held-out skill of the true present state on the present tokens > 0 in every fold
* **k3**: PASS - with the present state given as the imagined state D is exactly 0
* **k4**: PASS - with the true future state given as the imagined state D equals the headroom and its two-level interval lies above 0
* **k5**: FAIL - the rollout of the tool equals the banked rollout (pred_2s) of the same checkpoint to 0.0001
* **k6**: PASS - the windows, folds and episodes equal those of the r-b read (same ids, same count)
* **k5p**: PASS - k5 in parts (note ADD38_N2): rollout function on the banked inputs to 1e-4; same windows and actions; frames aligned (share >= 0.99); pod and banked states agree (mean cosine >= 0.999); the probe is tied to the dump (1e-6, same checkpoint)

controls_pass: True (k5 in parts (note PREREG_V8F_PROOF_ADD38_N2); k5 as written: FAIL; all controls as written: False)

## Statistic D (error of the present state minus error of the imagined state; above 0 = the imagination is closer to the true tokens of t + 2.0 s than the copy)

1948 windows, 139 episodes, seeds [0, 1, 2, 3]; moving cells = per window the 10 of 40 with the largest true token change. Units: squared error in the 64-component token space, per cell.

| statistic | per seed | pooled | pooled 95 % interval (two-level) |
|---|---|---|---|
| D, all cells | +1.7650, +1.7705, +1.7609, +1.7870 | +1.7709 | [+1.6879, +1.8539] |
| D, moving cells | +4.5856, +4.5624, +4.5068, +4.6590 | +4.5785 | [+4.3662, +4.7882] |
| headroom, all cells | +3.8184, +3.7931, +3.6860, +3.7734 | +3.7677 | [+3.6288, +3.9034] |
| headroom, moving cells | +9.2339, +9.1616, +8.8874, +9.1382 | +9.1053 | [+8.7742, +9.4219] |
| share of headroom, all | +0.4622, +0.4668, +0.4777, +0.4736 | +0.4700 | [+0.4581, +0.4815] |
| share of headroom, moving | +0.4966, +0.4980, +0.5071, +0.5098 | +0.5028 | [+0.4923, +0.5132] |

D per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +2.8097, +2.7500, +2.7921, +2.9035 | +2.8138 | [+2.6430, +2.9810] |
| far | +2.0688, +2.1255, +2.1120, +2.0948 | +2.1003 | [+1.9970, +2.2044] |
| near | +1.7441, +1.7765, +1.7288, +1.7282 | +1.7444 | [+1.6412, +1.8449] |
| hood | +0.4376, +0.4300, +0.4105, +0.4215 | +0.4249 | [+0.3958, +0.4548] |

## Name

* TOKENS-IMAGINED (D on the moving cells above 0 at both seeds and the two-level interval above 0)

Limits: linear maps show what is linearly readable; the token grid is a pooled 4 x 10 read of the picture; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance). No picture is evidence for a claim.
