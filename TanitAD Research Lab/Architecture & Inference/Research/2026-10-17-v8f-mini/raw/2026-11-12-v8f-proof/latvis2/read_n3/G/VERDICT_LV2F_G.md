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
| C1 moving cells | +0.2782, +0.3176 | +0.2979 | [+0.1900, +0.4052] |
| C1 all cells | -0.0790, -0.0070 | -0.0430 | [-0.1281, +0.0348] |
| C1 ceiling E(base) minus E(T), moving cells | +4.1647, +3.9164 | +4.0406 | [+3.7895, +4.3037] |
| C1 share of the ceiling, moving cells | +0.0668, +0.0811 | +0.0737 | [+0.0470, +0.1006] |
| C2 moving cells | +0.2433, +0.2557 | +0.2495 | [+0.1793, +0.3191] |
| C2 all cells | +0.1310, +0.1724 | +0.1517 | [+0.1016, +0.2018] |
| C2 ceiling E(base) minus E(T), moving cells | +4.1705, +3.9186 | +4.0445 | [+3.7911, +4.3086] |
| C2 share of the ceiling, moving cells | +0.0583, +0.0653 | +0.0617 | [+0.0447, +0.0782] |

C1 per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +0.1021, +0.0815 | +0.0918 | [-0.0227, +0.2004] |
| far | +0.0036, +0.1049 | +0.0543 | [-0.0584, +0.1615] |
| near | -0.1521, -0.0246 | -0.0884 | [-0.2115, +0.0332] |
| hood | -0.2694, -0.1899 | -0.2297 | [-0.3295, -0.1233] |

Held-out skill of each forecast (mean over folds, per seed) and the chosen penalty factors:

* P: skill 0.4197, 0.4054; penalty [1.0]; [1.0]
* PA: skill 0.4196, 0.4055; penalty [1.0]; [1.0]
* I: skill 0.4147, 0.4050; penalty [1.0]; [1.0]
* PAI: skill 0.4282, 0.4168; penalty [1.0]; [1.0]
* T: skill 0.5673, 0.5441; penalty [0.1]; [0.1]

## Names

* C1: TOKENS-FORECAST-FROM-IMAGINATION
* C2: ROLLOUT-BEYOND-LINEAR

Limits: linear forecasts on about 1,560 training windows per fold; the input I is a function of the present state and the logged actions, so C1 includes what the actions tell and C2 is the stricter read; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance).
