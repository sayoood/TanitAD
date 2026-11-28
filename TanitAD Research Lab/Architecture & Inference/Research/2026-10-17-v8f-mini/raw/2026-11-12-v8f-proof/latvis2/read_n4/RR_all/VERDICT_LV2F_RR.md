# VERDICT_LV2F_RR: the token-level read at equal calibration (note PREREG_V8F_PROOF_ADD38_N3 to card PREREG_V8F_PROOF_ADD38)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: PAUSED (PI item 25).

Each state is the input of its own out-of-fold linear forecast of the true tokens of t + 2.0 s. C1 = E(present) minus E(imagined); C2 = E(present + actions) minus E(present + actions + imagined).

## Controls

* **f1**: PASS - with the present state given as the imagined state C1 is exactly 0
* **f2**: PASS - known positive: E(P) minus E(T) above 0 at both seeds with the two-level interval above 0 (all cells and moving cells)
* **f3**: PASS - with the imagined states permuted across windows the held-out skill of the forecast from I is at most 0.02 in every fold
* **f4**: PASS - for every input and fold the skill on the training windows is above the held-out skill, and the held-out skill of P is above 0
* **f5**: PASS - k6 of the card and k5 in parts of note N2 on this dump

controls_pass: True

| statistic | per seed | pooled | pooled 95 % interval (two-level) |
|---|---|---|---|
| C1 moving cells | +0.7656, +0.7087, +0.7813, +0.7537 | +0.7523 | [+0.6909, +0.8145] |
| C1 all cells | +0.5001, +0.4573, +0.5302, +0.4859 | +0.4934 | [+0.4554, +0.5340] |
| C1 ceiling E(base) minus E(T), moving cells | +4.0953, +3.9531, +3.8054, +3.9197 | +3.9434 | [+3.7859, +4.0963] |
| C1 share of the ceiling, moving cells | +0.1869, +0.1793, +0.2053, +0.1923 | +0.1908 | [+0.1765, +0.2064] |
| C2 moving cells | +0.2623, +0.2238, +0.2938, +0.2611 | +0.2603 | [+0.2163, +0.3051] |
| C2 all cells | +0.1903, +0.1745, +0.2398, +0.1894 | +0.1985 | [+0.1688, +0.2329] |
| C2 ceiling E(base) minus E(T), moving cells | +4.0990, +3.9552, +3.8090, +3.9250 | +3.9471 | [+3.7910, +4.0999] |
| C2 share of the ceiling, moving cells | +0.0640, +0.0566, +0.0771, +0.0665 | +0.0659 | [+0.0554, +0.0774] |

C1 per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +0.6992, +0.6247, +0.7555, +0.6825 | +0.6905 | [+0.6252, +0.7621] |
| far | +0.5459, +0.5282, +0.5672, +0.5566 | +0.5494 | [+0.5096, +0.5914] |
| near | +0.5015, +0.4226, +0.4508, +0.4263 | +0.4503 | [+0.4021, +0.5005] |
| hood | +0.2538, +0.2539, +0.3475, +0.2782 | +0.2833 | [+0.2448, +0.3305] |

Held-out skill of each forecast (mean over folds, per seed) and the chosen penalty factors:

* P: skill 0.4026, 0.4112, 0.4060, 0.4115; penalty [1.0]; [1.0]; [1.0]; [1.0]
* PA: skill 0.4027, 0.4113, 0.4060, 0.4115; penalty [1.0]; [1.0]; [1.0]; [1.0]
* I: skill 0.4354, 0.4413, 0.4409, 0.4433; penalty [1.0]; [1.0]; [1.0]; [1.0]
* PAI: skill 0.4152, 0.4227, 0.4218, 0.4239; penalty [1.0]; [1.0]; [1.0]; [1.0]
* T: skill 0.5505, 0.5529, 0.5447, 0.5497; penalty [0.1]; [0.1]; [0.1]; [0.1]

## Names

* C1: TOKENS-FORECAST-FROM-IMAGINATION
* C2: ROLLOUT-BEYOND-LINEAR

Limits: linear forecasts on about 1,560 training windows per fold; the input I is a function of the present state and the logged actions, so C1 includes what the actions tell and C2 is the stricter read; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance).
