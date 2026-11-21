gate / NMS: `train_PR_gate_noNMS` -- refcv8_ema48852: gate 0.2660, NMS None; refcv7_50400: gate 0.2567, NMS None

| metric | refcv8_ema48852 | refcv7_50400 | paired |
|---|---|---|---|
| recall | 0.360 [0.325, 0.394] | 0.296 [0.260, 0.330] | +0.064 [0.053, 0.075] SEP |
| precision | 0.359 [0.347, 0.371] | 0.304 [0.294, 0.315] | +0.054 [0.043, 0.064] SEP |
| F1 | 0.359 [0.338, 0.379] | 0.300 [0.280, 0.318] | +0.059 [0.052, 0.066] SEP |
| conf_ratio (n_conf / n_pos) | 1.005 [0.908, 1.091] | 0.973 [0.846, 1.087] | +0.032 [-0.018, 0.085] ns |
| boxes per detected object | 2.142 [2.033, 2.274] | 2.191 [2.078, 2.292] | -0.049 [-0.124, 0.026] ns |
| frac. objects with >=2 boxes | 0.678 [0.651, 0.703] | 0.686 [0.650, 0.714] | -0.007 [-0.032, 0.018] ns |
| class accuracy on TPs | 0.939 [0.920, 0.956] | 0.894 [0.871, 0.916] | +0.045 [0.030, 0.061] SEP |
| counts (tp / n_conf / n_pos / detected objects) | 5254 / 14653 / 14586 / 5362 | 4319 / 14191 / 14586 / 4427 | |

recall by class at this gate: automobile 0.421 / 0.346; heavy_truck 0.209 / 0.161; bus 0.075 / 0.000; other_vehicle 0.021 / 0.032; trailer 0.054 / 0.027; person 0.219 / 0.187; rider 0.164 / 0.115; stroller 0.211 / 0.105; animal 0.571 / 0.286; protruding_object 0.417 / 0.369  (refcv8_ema48852 / refcv7_50400)