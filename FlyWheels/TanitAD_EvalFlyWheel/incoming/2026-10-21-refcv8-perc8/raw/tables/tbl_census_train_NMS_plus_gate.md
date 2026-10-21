gate / NMS: `train_NMS_plus_gate` -- refcv8_ema48852: gate 0.2165, NMS {'radius_m': 2.5, 'gate': 0.21648472547531128, 'p_floor': 0.05}; refcv7_50400: gate 0.2145, NMS {'radius_m': 2.5, 'gate': 0.21447688341140747, 'p_floor': 0.05}

| metric | refcv8_ema48852 | refcv7_50400 | paired |
|---|---|---|---|
| recall | 0.460 [0.425, 0.497] | 0.385 [0.350, 0.420] | +0.075 [0.060, 0.090] SEP |
| precision | 0.454 [0.433, 0.475] | 0.397 [0.379, 0.417] | +0.057 [0.046, 0.068] SEP |
| F1 | 0.457 [0.432, 0.482] | 0.391 [0.367, 0.414] | +0.066 [0.056, 0.076] SEP |
| conf_ratio (n_conf / n_pos) | 1.013 [0.945, 1.085] | 0.970 [0.888, 1.048] | +0.043 [0.002, 0.086] SEP |
| boxes per detected object | 1.088 [1.075, 1.104] | 1.111 [1.097, 1.124] | -0.022 [-0.035, -0.010] SEP |
| frac. objects with >=2 boxes | 0.087 [0.075, 0.101] | 0.110 [0.097, 0.122] | -0.023 [-0.036, -0.012] SEP |
| class accuracy on TPs | 0.930 [0.911, 0.947] | 0.881 [0.856, 0.904] | +0.049 [0.033, 0.065] SEP |
| counts (tp / n_conf / n_pos / detected objects) | 6713 / 14782 / 14586 / 7142 | 5622 / 14152 / 14586 / 5966 | |

recall by class at this gate: automobile 0.541 / 0.454; heavy_truck 0.461 / 0.396; bus 0.343 / 0.030; other_vehicle 0.117 / 0.117; trailer 0.135 / 0.095; person 0.234 / 0.207; rider 0.281 / 0.191; stroller 0.316 / 0.105; animal 0.286 / 0.000; protruding_object 0.250 / 0.345  (refcv8_ema48852 / refcv7_50400)