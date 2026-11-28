# VERDICT_LV2_G: the token-level read of what the rollout transports (card PREREG_V8F_PROOF_ADD38)

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

1948 windows, 139 episodes, seeds [2, 3]; moving cells = per window the 10 of 40 with the largest true token change. Units: squared error in the 64-component token space, per cell.

| statistic | per seed | pooled | pooled 95 % interval (two-level) |
|---|---|---|---|
| D, all cells | +1.1024, +1.4372 | +1.2698 | [+0.9869, +1.5469] |
| D, moving cells | +4.4361, +5.0166 | +4.7264 | [+4.2218, +5.2341] |
| headroom, all cells | +3.6281, +3.8222 | +3.7252 | [+3.5003, +3.9465] |
| headroom, moving cells | +8.7764, +9.2439 | +9.0102 | [+8.4836, +9.5319] |
| share of headroom, all | +0.3039, +0.3760 | +0.3409 | [+0.2769, +0.3960] |
| share of headroom, moving | +0.5055, +0.5427 | +0.5246 | [+0.4926, +0.5548] |

D per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +2.1223, +2.3946 | +2.2585 | [+1.9086, +2.6096] |
| far | +1.7713, +2.0696 | +1.9205 | [+1.6441, +2.2102] |
| near | +0.9829, +1.4390 | +1.2109 | [+0.8540, +1.5756] |
| hood | -0.4667, -0.1545 | -0.3106 | [-0.5468, -0.0898] |

## Name

* TOKENS-IMAGINED (D on the moving cells above 0 at both seeds and the two-level interval above 0)

Limits: linear maps show what is linearly readable; the token grid is a pooled 4 x 10 read of the picture; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance). No picture is evidence for a claim.
