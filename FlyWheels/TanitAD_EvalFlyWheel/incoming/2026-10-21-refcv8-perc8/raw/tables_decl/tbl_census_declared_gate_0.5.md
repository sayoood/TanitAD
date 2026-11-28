gate / NMS: `declared_gate_0.5` -- refcv8_ema48852: gate 0.5000, NMS None; refcv7_50400: gate 0.5000, NMS None

| metric | refcv8_ema48852 | refcv7_50400 | paired |
|---|---|---|---|
| recall | 0.013 [0.008, 0.019] | 0.008 [0.004, 0.013] | +0.005 [0.003, 0.008] SEP |
| precision | 1.000 [1.000, 1.000] | 0.983 [0.938, 1.000] | +0.017 [0.000, 0.062] ns |
| F1 | 0.026 [0.016, 0.037] | 0.016 [0.008, 0.026] | +0.010 [0.006, 0.015] SEP |
| conf_ratio (n_conf / n_pos) | 0.013 [0.008, 0.019] | 0.008 [0.004, 0.013] | +0.005 [0.003, 0.008] SEP |
| boxes per detected object | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [0.000, 0.000] ns |
| frac. objects with >=2 boxes | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| class accuracy on TPs | 0.985 [0.967, 1.000] | 0.966 [0.894, 1.000] | +0.019 [-0.022, 0.084] ns |
| counts (tp / n_conf / n_pos / detected objects) | 195 / 195 / 14586 / 214 | 117 / 119 / 14586 / 123 | |

recall by class at this gate: automobile 0.017 / 0.011; heavy_truck 0.000 / 0.000; bus 0.000 / 0.000; other_vehicle 0.000 / 0.000; trailer 0.000 / 0.000; person 0.004 / 0.000; rider 0.000 / 0.000; stroller 0.000 / 0.000; animal 0.000 / 0.000; protruding_object 0.012 / 0.048  (refcv8_ema48852 / refcv7_50400)