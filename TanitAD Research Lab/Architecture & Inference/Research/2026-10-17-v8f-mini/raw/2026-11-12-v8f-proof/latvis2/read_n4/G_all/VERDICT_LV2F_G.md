# VERDICT_LV2F_G: the token-level read at equal calibration (note PREREG_V8F_PROOF_ADD38_N3 to card PREREG_V8F_PROOF_ADD38)

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
| C1 moving cells | +0.2782, +0.3176, +0.1422, +0.1894 | +0.2319 | [+0.1261, +0.3365] |
| C1 all cells | -0.0790, -0.0070, -0.1399, -0.1157 | -0.0854 | [-0.1560, -0.0144] |
| C1 ceiling E(base) minus E(T), moving cells | +4.1647, +3.9164, +3.6648, +4.0361 | +3.9455 | [+3.7234, +4.1536] |
| C1 share of the ceiling, moving cells | +0.0668, +0.0811, +0.0388, +0.0469 | +0.0588 | [+0.0332, +0.0831] |
| C2 moving cells | +0.2433, +0.2557, +0.1644, +0.1859 | +0.2123 | [+0.1530, +0.2738] |
| C2 all cells | +0.1310, +0.1724, +0.1149, +0.1153 | +0.1334 | [+0.0942, +0.1723] |
| C2 ceiling E(base) minus E(T), moving cells | +4.1705, +3.9186, +3.6700, +4.0427 | +3.9504 | [+3.7296, +4.1599] |
| C2 share of the ceiling, moving cells | +0.0583, +0.0653, +0.0448, +0.0460 | +0.0537 | [+0.0398, +0.0676] |

C1 per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +0.1021, +0.0815, -0.0761, -0.1210 | -0.0034 | [-0.1335, +0.1158] |
| far | +0.0036, +0.1049, -0.1827, -0.0711 | -0.0363 | [-0.1617, +0.0858] |
| near | -0.1521, -0.0246, -0.1063, -0.0728 | -0.0889 | [-0.1666, -0.0125] |
| hood | -0.2694, -0.1899, -0.1944, -0.1981 | -0.2130 | [-0.2824, -0.1471] |

Held-out skill of each forecast (mean over folds, per seed) and the chosen penalty factors:

* P: skill 0.4197, 0.4054, 0.4109, 0.4085; penalty [1.0]; [1.0]; [1.0]; [1.0]
* PA: skill 0.4196, 0.4055, 0.4108, 0.4084; penalty [1.0]; [1.0]; [1.0]; [1.0]
* I: skill 0.4147, 0.4050, 0.4016, 0.4009; penalty [1.0]; [1.0]; [1.0]; [1.0]
* PAI: skill 0.4282, 0.4168, 0.4184, 0.4160; penalty [1.0]; [1.0]; [1.0]; [1.0]
* T: skill 0.5673, 0.5441, 0.5422, 0.5502; penalty [0.1]; [0.1]; [0.1]; [0.1]

## Names

* C1: TOKENS-FORECAST-FROM-IMAGINATION
* C2: ROLLOUT-BEYOND-LINEAR

Limits: linear forecasts on about 1,560 training windows per fold; the input I is a function of the present state and the logged actions, so C1 includes what the actions tell and C2 is the stricter read; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance).
