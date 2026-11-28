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

1948 windows, 139 episodes, seeds [0, 1, 2, 3]; moving cells = per window the 10 of 40 with the largest true token change. Units: squared error in the 64-component token space, per cell.

| statistic | per seed | pooled | pooled 95 % interval (two-level) |
|---|---|---|---|
| D, all cells | +1.2297, +1.3117, +1.1024, +1.4372 | +1.2703 | [+1.1088, +1.4344] |
| D, moving cells | +4.7832, +4.8661, +4.4361, +5.0166 | +4.7755 | [+4.4543, +5.0808] |
| headroom, all cells | +3.8752, +3.7454, +3.6281, +3.8222 | +3.7677 | [+3.6088, +3.9262] |
| headroom, moving cells | +9.4516, +9.0786, +8.7764, +9.2439 | +9.1376 | [+8.7460, +9.5136] |
| share of headroom, all | +0.3173, +0.3502, +0.3039, +0.3760 | +0.3371 | [+0.3024, +0.3725] |
| share of headroom, moving | +0.5061, +0.5360, +0.5055, +0.5427 | +0.5226 | [+0.5020, +0.5438] |

D per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +2.2862, +2.3963, +2.1223, +2.3946 | +2.2999 | [+2.0628, +2.5321] |
| far | +1.8352, +2.0091, +1.7713, +2.0696 | +1.9213 | [+1.7447, +2.1000] |
| near | +1.1566, +1.2506, +0.9829, +1.4390 | +1.2073 | [+0.9964, +1.4214] |
| hood | -0.3590, -0.4091, -0.4667, -0.1545 | -0.3473 | [-0.4877, -0.2016] |

## Name

* TOKENS-IMAGINED (D on the moving cells above 0 at both seeds and the two-level interval above 0)

Limits: linear maps show what is linearly readable; the token grid is a pooled 4 x 10 read of the picture; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance). No picture is evidence for a claim.
