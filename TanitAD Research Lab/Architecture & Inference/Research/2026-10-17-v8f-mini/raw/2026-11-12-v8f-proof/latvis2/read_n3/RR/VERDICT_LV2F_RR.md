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
| C1 moving cells | +0.7656, +0.7087 | +0.7371 | [+0.6510, +0.8298] |
| C1 all cells | +0.5001, +0.4573 | +0.4787 | [+0.4276, +0.5313] |
| C1 ceiling E(base) minus E(T), moving cells | +4.0953, +3.9531 | +4.0242 | [+3.8161, +4.2396] |
| C1 share of the ceiling, moving cells | +0.1869, +0.1793 | +0.1832 | [+0.1655, +0.2004] |
| C2 moving cells | +0.2623, +0.2238 | +0.2430 | [+0.1836, +0.3069] |
| C2 all cells | +0.1903, +0.1745 | +0.1824 | [+0.1498, +0.2162] |
| C2 ceiling E(base) minus E(T), moving cells | +4.0990, +3.9552 | +4.0271 | [+3.8192, +4.2445] |
| C2 share of the ceiling, moving cells | +0.0640, +0.0566 | +0.0604 | [+0.0468, +0.0738] |

C1 per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +0.6992, +0.6247 | +0.6619 | [+0.5755, +0.7500] |
| far | +0.5459, +0.5282 | +0.5370 | [+0.4799, +0.5958] |
| near | +0.5015, +0.4226 | +0.4621 | [+0.3874, +0.5418] |
| hood | +0.2538, +0.2539 | +0.2538 | [+0.2254, +0.2838] |

Held-out skill of each forecast (mean over folds, per seed) and the chosen penalty factors:

* P: skill 0.4026, 0.4112; penalty [1.0]; [1.0]
* PA: skill 0.4027, 0.4113; penalty [1.0]; [1.0]
* I: skill 0.4354, 0.4413; penalty [1.0]; [1.0]
* PAI: skill 0.4152, 0.4227; penalty [1.0]; [1.0]
* T: skill 0.5505, 0.5529; penalty [0.1]; [0.1]

## Names

* C1: TOKENS-FORECAST-FROM-IMAGINATION
* C2: ROLLOUT-BEYOND-LINEAR

Limits: linear forecasts on about 1,560 training windows per fold; the input I is a function of the present state and the logged actions, so C1 includes what the actions tell and C2 is the stricter read; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance).
