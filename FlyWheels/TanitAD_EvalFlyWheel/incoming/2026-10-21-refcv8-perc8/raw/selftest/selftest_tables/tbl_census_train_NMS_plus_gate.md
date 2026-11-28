gate / NMS: `train_NMS_plus_gate` -- refcv8_ema48852: gate 0.2145, NMS {'radius_m': 2.5, 'gate': 0.21447688341140747, 'p_floor': 0.05}; refcv7_50400: gate 0.2145, NMS {'radius_m': 2.5, 'gate': 0.21447688341140747, 'p_floor': 0.05}

| metric | refcv8_ema48852 | refcv7_50400 | paired |
|---|---|---|---|
| recall | 0.385 [0.338, 0.435] | 0.385 [0.338, 0.435] | +0.000 [0.000, 0.000] ns |
| precision | 0.397 [0.375, 0.428] | 0.397 [0.375, 0.428] | +0.000 [0.000, 0.000] ns |
| F1 | 0.391 [0.357, 0.426] | 0.391 [0.357, 0.426] | +0.000 [0.000, 0.000] ns |
| conf_ratio (n_conf / n_pos) | 0.970 [0.895, 1.073] | 0.970 [0.895, 1.073] | +0.000 [0.000, 0.000] ns |
| boxes per detected object | 1.111 [1.100, 1.118] | 1.111 [1.100, 1.118] | +0.000 [0.000, 0.000] ns |
| frac. objects with >=2 boxes | 0.110 [0.099, 0.117] | 0.110 [0.099, 0.117] | +0.000 [0.000, 0.000] ns |
| class accuracy on TPs | 0.881 [0.859, 0.894] | 0.881 [0.859, 0.894] | +0.000 [0.000, 0.000] ns |
| counts (tp / n_conf / n_pos / detected objects) | 5622 / 14152 / 14586 / 5966 | 5622 / 14152 / 14586 / 5966 | |

recall by class at this gate: automobile 0.454 / 0.454; heavy_truck 0.396 / 0.396; bus 0.030 / 0.030; other_vehicle 0.117 / 0.117; trailer 0.095 / 0.095; person 0.207 / 0.207; rider 0.191 / 0.191; stroller 0.105 / 0.105; animal 0.000 / 0.000; protruding_object 0.345 / 0.345  (refcv8_ema48852 / refcv7_50400)