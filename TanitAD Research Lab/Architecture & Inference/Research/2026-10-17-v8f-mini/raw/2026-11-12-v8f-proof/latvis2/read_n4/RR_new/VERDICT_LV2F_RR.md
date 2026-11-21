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
| C1 moving cells | +0.7813, +0.7537 | +0.7675 | [+0.6871, +0.8451] |
| C1 all cells | +0.5302, +0.4859 | +0.5081 | [+0.4530, +0.5588] |
| C1 ceiling E(base) minus E(T), moving cells | +3.8054, +3.9197 | +3.8626 | [+3.6638, +4.0611] |
| C1 share of the ceiling, moving cells | +0.2053, +0.1923 | +0.1987 | [+0.1795, +0.2176] |
| C2 moving cells | +0.2938, +0.2611 | +0.2775 | [+0.2173, +0.3361] |
| C2 all cells | +0.2398, +0.1894 | +0.2146 | [+0.1673, +0.2605] |
| C2 ceiling E(base) minus E(T), moving cells | +3.8090, +3.9250 | +3.8670 | [+3.6670, +4.0668] |
| C2 share of the ceiling, moving cells | +0.0771, +0.0665 | +0.0718 | [+0.0567, +0.0864] |

C1 per row of the grid (all cells of the row):

| row | per seed | pooled | pooled interval |
|---|---|---|---|
| sky | +0.7555, +0.6825 | +0.7190 | [+0.6327, +0.8009] |
| far | +0.5672, +0.5566 | +0.5619 | [+0.5060, +0.6136] |
| near | +0.4508, +0.4263 | +0.4386 | [+0.3793, +0.4976] |
| hood | +0.3475, +0.2782 | +0.3128 | [+0.2591, +0.3733] |

Held-out skill of each forecast (mean over folds, per seed) and the chosen penalty factors:

* P: skill 0.4060, 0.4115; penalty [1.0]; [1.0]
* PA: skill 0.4060, 0.4115; penalty [1.0]; [1.0]
* I: skill 0.4409, 0.4433; penalty [1.0]; [1.0]
* PAI: skill 0.4218, 0.4239; penalty [1.0]; [1.0]
* T: skill 0.5447, 0.5497; penalty [0.1]; [0.1]

## Names

* C1: TOKENS-FORECAST-FROM-IMAGINATION
* C2: ROLLOUT-BEYOND-LINEAR

Limits: linear forecasts on about 1,560 training windows per fold; the input I is a function of the present state and the logged actions, so C1 includes what the actions tell and C2 is the stricter read; two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance).
