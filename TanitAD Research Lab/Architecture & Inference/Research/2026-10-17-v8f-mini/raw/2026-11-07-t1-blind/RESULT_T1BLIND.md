# RESULT_T1BLIND (T1-kin, eval-141, scene `reindexed`)

**Outcome by the card's section 5: VISION-USED.**  Tier T1-kin. Estimator: paired episode-cluster bootstrap B 2,000 seed 0 over 141 episodes / 2,399 windows. F = 0.12074 m (training-run spread of M2b on ADE6).
Reading of the card's disjunction: 'At both seeds' applies to the whole disjunction: (ADE6 B separated-worse with point > F) or (M1 B separated either direction) must hold at EACH seed; pinned in this reader before any BLIND1 window was scored.

## Gates (card section 3): all PASS

| seed | G-ID | G-MEAN (frames received / equal to the pinned mean) | G-EGO (tick-0 windows equal / compared) |
|---|---|---|---|
| s0 | PASS | PASS (115,152 / 115,152) | PASS (2,399 / 2,399) |
| s1 | PASS | PASS (115,152 / 115,152) | PASS (2,399 / 2,399) |

G-EGO compares the tick-0 ego inputs (the logged history) window for window; ticks 1 and 2 are the executed motion, which differs when the plan differs (the effect under test).

## The two deciding rows, B = BLIND1 - RIC0

| seed | ADE6 B (m) [95 % CI] | status | point above F | M1 crawl share B [95 % CI] | status |
|---|---|---|---|---|---|
| s0 | +0.8313 [+0.5576, +1.1530] | worse | True | -0.0246 [-0.0428, -0.0085] | better |
| s1 | +0.8158 [+0.5309, +1.1454] | worse | True | -0.0415 [-0.0654, -0.0208] | better |

Alternative readings, beside: {"primary_reading": "VISION-USED iff (ADE6 cond or M1 cond) at BOTH seeds", "strict_reading_ADE_both_or_M1_both": "VISION-USED", "at_least_one_seed_reading": "a USED condition holds at >= 1 seed"}

## All 19 rows, per seed (status: better / worse = CI excludes 0 on that side of the row's direction; n.s. = CI contains 0)

### s0

| family | row | BLIND1 - RIC0 [CI] | status | BLIND1 - P10 [CI] | status | RIC0 - P10 status |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.8313 [+0.5576, +1.1530] | worse | -0.2814 [-0.5157, -0.0132] | better | better |
| crawl | M1 crawl share | -0.0246 [-0.0428, -0.0085] | better | +0.0287 [+0.0171, +0.0424] | worse | worse |
| LONGITUDINAL | speed MAE | +0.1075 [+0.0186, +0.1907] | worse | -0.6266 [-0.8054, -0.4680] | better | better |
| LONGITUDINAL | along-track MAE | +0.2352 [+0.0932, +0.3868] | worse | -0.4135 [-0.5658, -0.2580] | better | better |
| LONGITUDINAL | accel MAE | +0.0695 [-0.0373, +0.1675] | n.s. | -0.7281 [-1.0641, -0.4383] | better | better |
| LONGITUDINAL | DK-VIOL | +0.0235 [-0.0017, +0.0539] | n.s. | +0.0134 [-0.0081, +0.0376] | n.s. | n.s. |
| LONGITUDINAL | DK-HW | +0.2899 [+0.0249, +0.5707] | worse | -0.0692 [-0.4247, +0.2447] | n.s. | better |
| LONGITUDINAL | DK-TG | +0.0968 [+0.0195, +0.2088] | worse | -0.0234 [-0.1037, +0.0733] | n.s. | better |
| LATERAL | heading MAE | +1.8475 [+0.9933, +2.7709] | worse | +0.0060 [-0.6768, +0.6942] | n.s. | better |
| LATERAL | yaw-rate MAE | -1.7633 [-3.7448, +0.1243] | n.s. | +0.7378 [-0.6630, +2.0577] | n.s. | worse |
| LATERAL | curvature MAE | -0.0328 [-0.0578, -0.0105] | better | +0.0513 [+0.0323, +0.0743] | worse | worse |
| LATERAL | cross-track MAE | +0.7835 [+0.5266, +1.0907] | worse | -0.0769 [-0.3116, +0.1950] | n.s. | better |
| LATERAL | L1 lateral-accel MAE | +0.2599 [-0.1564, +0.6641] | n.s. | -1.9708 [-2.9570, -1.1078] | better | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0207 [-0.0487, +0.0008] | n.s. | +0.0197 [+0.0106, +0.0308] | worse | worse |
| TACTICAL | TAC-LAT | -0.0738 [-0.1205, -0.0296] | worse | +0.1088 [+0.0633, +0.1563] | better | better |
| TACTICAL | TAC-LON | -0.0108 [-0.0496, +0.0295] | n.s. | +0.1371 [+0.0771, +0.1982] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@30 | +0.7695 [+0.3980, +1.1508] | worse | -0.7985 [-1.2727, -0.3762] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@50 | +1.8563 [+1.1466, +2.5964] | worse | -2.2913 [-3.3901, -1.3049] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@80 | +4.1746 [+2.7571, +5.6127] | worse | -4.2213 [-6.1246, -2.4728] | better | better |

Separated-better in B: M1 crawl share, curvature MAE.  Separated-worse in B: ADE6, DK-HW, DK-TG, S-POS@30, S-POS@50, S-POS@80, TAC-LAT, along-track MAE, cross-track MAE, heading MAE, speed MAE.

### s1

| family | row | BLIND1 - RIC0 [CI] | status | BLIND1 - P10 [CI] | status | RIC0 - P10 status |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.8158 [+0.5309, +1.1454] | worse | -0.2454 [-0.4762, +0.0045] | n.s. | better |
| crawl | M1 crawl share | -0.0415 [-0.0654, -0.0208] | better | +0.0181 [+0.0072, +0.0306] | worse | worse |
| LONGITUDINAL | speed MAE | +0.0607 [-0.0207, +0.1409] | n.s. | -0.6971 [-0.8725, -0.5417] | better | better |
| LONGITUDINAL | along-track MAE | +0.2164 [+0.0577, +0.3854] | worse | -0.3662 [-0.5086, -0.2228] | better | better |
| LONGITUDINAL | accel MAE | +0.0015 [-0.0773, +0.0861] | n.s. | -0.9429 [-1.3047, -0.6380] | better | better |
| LONGITUDINAL | DK-VIOL | +0.0387 [+0.0104, +0.0760] | worse | +0.0202 [-0.0033, +0.0468] | n.s. | better |
| LONGITUDINAL | DK-HW | +0.4313 [+0.1499, +0.7302] | worse | +0.0574 [-0.3116, +0.4002] | n.s. | better |
| LONGITUDINAL | DK-TG | +0.2261 [+0.0877, +0.4113] | worse | +0.1438 [+0.0028, +0.3320] | worse | better |
| LATERAL | heading MAE | +1.1704 [+0.1929, +2.2390] | worse | -0.6704 [-1.3253, -0.0383] | better | better |
| LATERAL | yaw-rate MAE | -6.3333 [-10.4576, -2.8914] | better | -3.4564 [-6.1164, -1.1929] | better | worse |
| LATERAL | curvature MAE | -0.0685 [-0.1100, -0.0326] | better | +0.0302 [+0.0162, +0.0472] | worse | worse |
| LATERAL | cross-track MAE | +0.8043 [+0.5528, +1.1108] | worse | -0.0609 [-0.2745, +0.1898] | n.s. | better |
| LATERAL | L1 lateral-accel MAE | -0.2402 [-0.4895, -0.0104] | better | -2.7925 [-4.1326, -1.6535] | better | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0252 [-0.0588, +0.0062] | n.s. | +0.0206 [+0.0068, +0.0391] | worse | worse |
| TACTICAL | TAC-LAT | -0.0125 [-0.0638, +0.0383] | n.s. | +0.1251 [+0.0804, +0.1681] | better | better |
| TACTICAL | TAC-LON | -0.0042 [-0.0483, +0.0417] | n.s. | +0.1359 [+0.0796, +0.1958] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@30 | +0.9054 [+0.4556, +1.4018] | worse | -0.2695 [-0.5331, -0.0083] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@50 | +1.9338 [+1.0611, +2.8744] | worse | -1.4765 [-2.2004, -0.7642] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@80 | +3.9655 [+2.3943, +5.5887] | worse | -3.9829 [-5.5803, -2.3657] | better | better |

Separated-better in B: L1 lateral-accel MAE, M1 crawl share, curvature MAE, yaw-rate MAE.  Separated-worse in B: ADE6, DK-HW, DK-TG, DK-VIOL, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE.

## Next, committed by the card for this outcome

RENDER-GAP-BINDS (read with T2's BLIND ~ C0): the LoRA dominance card gains a gate before LC's T2 read counts (L-BLIND at T2 separated from L on ADE6 or M1); a render-robustness lever gets its own card (adapt the encoder to NuRec renders, or render-domain augmentation), aimed at T2.

STRATEGIC family: PAUSED (PI item 25), listed above, decides nothing. The estimator answers another draw of EPISODES only; BLIND1 is a deterministic function of C0, so the seed question is read by the two seeds side by side.

Artifacts: `raw/READ_T1BLIND.json` (sha256 96049e35cff646a85b11eb07882e9b966af199a17f736d443b562460f5ef4a56), `raw/GATES.json` (sha256 e2b86944ea324551ab503ad5c4deafedf3ab9325c24d719a516bdb40f541d780), `raw/GATE_ID_s*.json`, rollouts `F:/TanitAD-scratch/motion_c/t1_blind/out/`.
