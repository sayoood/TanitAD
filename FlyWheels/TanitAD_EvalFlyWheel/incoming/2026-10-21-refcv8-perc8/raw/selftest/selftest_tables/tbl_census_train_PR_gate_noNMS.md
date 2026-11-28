gate / NMS: `train_PR_gate_noNMS` -- refcv8_ema48852: gate 0.2567, NMS None; refcv7_50400: gate 0.2567, NMS None

| metric | refcv8_ema48852 | refcv7_50400 | paired |
|---|---|---|---|
| recall | 0.296 [0.254, 0.343] | 0.296 [0.254, 0.343] | +0.000 [0.000, 0.000] ns |
| precision | 0.304 [0.292, 0.311] | 0.304 [0.292, 0.311] | +0.000 [0.000, 0.000] ns |
| F1 | 0.300 [0.273, 0.323] | 0.300 [0.273, 0.323] | +0.000 [0.000, 0.000] ns |
| conf_ratio (n_conf / n_pos) | 0.973 [0.862, 1.126] | 0.973 [0.862, 1.126] | +0.000 [0.000, 0.000] ns |
| boxes per detected object | 2.191 [2.126, 2.281] | 2.191 [2.126, 2.281] | +0.000 [0.000, 0.000] ns |
| frac. objects with >=2 boxes | 0.686 [0.671, 0.714] | 0.686 [0.671, 0.714] | +0.000 [0.000, 0.000] ns |
| class accuracy on TPs | 0.894 [0.879, 0.904] | 0.894 [0.879, 0.904] | +0.000 [0.000, 0.000] ns |
| counts (tp / n_conf / n_pos / detected objects) | 4319 / 14191 / 14586 / 4427 | 4319 / 14191 / 14586 / 4427 | |

recall by class at this gate: automobile 0.346 / 0.346; heavy_truck 0.161 / 0.161; bus 0.000 / 0.000; other_vehicle 0.032 / 0.032; trailer 0.027 / 0.027; person 0.187 / 0.187; rider 0.115 / 0.115; stroller 0.105 / 0.105; animal 0.286 / 0.286; protruding_object 0.369 / 0.369  (refcv8_ema48852 / refcv7_50400)