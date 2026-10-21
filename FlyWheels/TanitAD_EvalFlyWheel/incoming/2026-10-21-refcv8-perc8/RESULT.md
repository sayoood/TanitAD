# PERC8_EVAL -- eval-grade perception read of refcv8 EMA-48,852, against refcv7@50,400 (same instrument, same windows)

*EvalFlyWheel, 2026-10-21 (Berlin local). Nothing here is staged or committed; the orchestrator lands the files (manifest at the end).*

| | |
|---|---|
| **Evidence class** | MEASURED (ours). Every number traces to a file under `raw/`; the refcv7 side is the banked eval-grade dump `D:/refcv7_eval_kit/perc/step50400_a6text/` (`perc_det.pkl`, `perc_map_table.npz`) -- re-scored here, reproducing its banked numbers (section 1). |
| **Tier** | **T0** -- the perception outputs of the model's forward on the S2 eval windows (the box and map heads read the frames and see no future frame; no closed loop). A perception read, NOT a driving-performance claim. |
| **Estimator** | episode-cluster bootstrap over the eval episodes (`taniteval.ci` draw stream, seed 0, 1,000 draws); PAIRED = the same draws for both arms; 95 % percentile interval. |
| **n** | S2 = 4,754 windows over 139 eval episodes (the refcv7 window list, md5 a3a78dba..., identity checked per `(sha12, ws)`): **4,532 box windows / 139 episodes / 14,586 VIS-1 positives** (identical on both arms) and **4,686 map windows / 137 episodes**. TRAIN-DIAG (fit only): 1,112 windows / 139 train episodes, 1,074 carry box labels. |
| **Seeds** | ONE training seed per arm (refcv8-r101-s0, refcv7-r101-s0). |
| **What the CI answers** | **another draw of EPISODES only.** It does not answer "would another TRAINING run say this?" (`H-ESTIM-SEED-1`: a separated interval from a one-seed arm is necessary, not sufficient). The perception forward is deterministic (no sampling), so there is no inference-seed variance to measure. |
| **Clock** | PRIMARY read = the unshifted pose clock (`R8_NO_POSE_SYNC=1`), i.e. the SAME ground-truth definition refcv7 was scored on. refcv8 was TRAINED on the pose-synced clock (`--pose-sync-sidecar`, mean shift 16.5 ms, max 35.4 ms), so this read is, if anything, conservative for refcv8's boxes at the tight thresholds. |
| **Headline** | refcv8 EMA-48,852 vs refcv7@50,400, box AP@2 m **0.328 [0.301, 0.354] vs 0.249 [0.227, 0.270], paired +0.079 [0.071, 0.087]**; after the TRAIN-fitted F4b NMS (2.5 m) + P=R gate **0.447 vs 0.345 (+0.102 [0.089, 0.115])**, F1 0.457 vs 0.391; the declared 0.5 gate is still unusable (recall 0.013, conf_ratio 0.013); duplicate boxes unchanged (2.14 per object before NMS, 1.09 after); the 10 cm map moved by only -0.001..+0.013 IoU. Controls all PASS. Corrected-clock read NOT run. |
| **Compute** | Thor (Jetson), `/home/nvidia/r8ns/perc8/`, `OMP_NUM_THREADS=6`, GPU taken through the Thor `flock` convention after the other agent's chain had finished; only `torch.cuda.max_memory_allocated()` is quoted for memory: pass A 1.124 GiB (batch 1; 4,754 windows, 1,955 s loop, 2,085 s wall, 0.41 s/window), TRAIN-DIAG 8.11 GiB (batch 8; 453 s loop). Dev-box GPU untouched. |


## 1. Controls (each must read its known value)

| control | known value | read | verdict |
|---|---|---|---|
| GT as the box prediction (programme `summarise`, 300 windows, 947 positives) | every defined AP cell = 1.0 | 156/156 cells = 1.0 | PASS |
| GT as the map prediction (structured 1000x600 grid) | pooled IoU = 1.0 in every (class, band) with GT | 30/30 cells = 1.0 | PASS |
| all-one-class map prediction (negative control) | IoU of every other class = 0 | max 0.0 | PASS |
| re-score refcv7's banked dump with THIS instrument (AP@2 m pooled) | banked 0.24906 (within 1e-3) | 0.249064; max abs diff over 23 AP/AUROC cells = 1.0e-12 | PASS |
| re-summarise refcv7's banked map WindowTable | banked IoU cells | 40 IoU cells, max abs diff 0.0 | PASS |
| weighted-AP kernel == `detection_metrics._ap` (4 thr x 2 arms, 1e-9) | equal | see `raw/box_read_declared.json` controls | PASS |
| bootstrap draw stream == `ci.paired_episode_cluster_bootstrap` (recall paired CI) | equal | see `raw/box_read_declared.json` controls | PASS |
| weighted AUROC == `detection_metrics.auroc` (1e-9) | equal | see `raw/box_read_declared.json` controls | PASS |
| LAW-frame control (perc pass, 2 windows: zero LAW frame == real LAW frame for every perception tensor) | equal | all equal, law term differs | PASS |
| checkpoint md5 | b4364029b68f29a33eeb4aebbbbd002b (battery record `E:/refcv8_eval_kit/ckpt/ckpt_48852.md5`) | b4364029b68f29a33eeb4aebbbbd002b | PASS |

## 2. Box read: refcv8 EMA-48,852 vs refcv7@50,400, declared protocol (S2 windows, unshifted pose clock)

| metric (pooled over 4532 windows / 139 episodes) | refcv8_ema48852 | refcv7_50400 | paired refcv8_ema48852 - refcv7_50400 |
|---|---|---|---|
| box AP@0.5 m (all classes) | 0.065 [0.051, 0.080] | 0.030 [0.023, 0.037] | +0.035 [0.027, 0.044] SEP |
| box AP@1 m | 0.199 [0.177, 0.222] | 0.129 [0.112, 0.148] | +0.069 [0.060, 0.078] SEP |
| box AP@2 m | 0.328 [0.301, 0.354] | 0.249 [0.227, 0.270] | +0.079 [0.071, 0.087] SEP |
| box AP@4 m | 0.410 [0.378, 0.441] | 0.332 [0.300, 0.363] | +0.078 [0.070, 0.085] SEP |
| class-mean AP@0.5 m | 0.014 [0.012, 0.018] | 0.006 [0.004, 0.008] | +0.009 [0.007, 0.011] SEP |
| class-mean AP@1 m | 0.051 [0.043, 0.069] | 0.028 [0.023, 0.035] | +0.023 [0.017, 0.038] SEP |
| class-mean AP@2 m | 0.102 [0.086, 0.125] | 0.070 [0.054, 0.091] | +0.032 [0.015, 0.050] SEP |
| class-mean AP@4 m | 0.144 [0.121, 0.170] | 0.102 [0.080, 0.128] | +0.042 [0.026, 0.059] SEP |
| AP@2 m, 0-20 m | 0.539 [0.503, 0.572] | 0.418 [0.387, 0.451] | +0.121 [0.106, 0.136] SEP |
| AP@2 m, 20-40 m | 0.234 [0.214, 0.253] | 0.166 [0.151, 0.181] | +0.068 [0.058, 0.078] SEP |
| AP@2 m, 40-60 m | 0.092 [0.077, 0.107] | 0.067 [0.056, 0.078] | +0.025 [0.018, 0.032] SEP |
| AUROC (matched slots) | 0.931 [0.921, 0.941] | 0.914 [0.902, 0.925] | +0.017 [0.015, 0.020] SEP |
| AUROC (objectness, 2 m) | 0.890 [0.880, 0.900] | 0.907 [0.898, 0.917] | -0.017 [-0.023, -0.012] SEP |

| per-class AP@2 m | refcv8_ema48852 | refcv7_50400 | paired |
|---|---|---|---|
| automobile | 0.360 [0.331, 0.389] | 0.273 [0.248, 0.298] | +0.087 [0.077, 0.097] SEP |
| heavy_truck | 0.144 [0.097, 0.229] | 0.125 [0.062, 0.212] | +0.019 [-0.069, 0.110] ns |
| bus | 0.109 [0.019, 0.211] | 0.042 [0.004, 0.095] | +0.067 [0.014, 0.133] SEP |
| other_vehicle | 0.018 [0.006, 0.147] | 0.014 [0.000, 0.030] | +0.005 [-0.006, 0.145] ns |
| trailer | 0.011 [0.000, 0.064] | 0.002 [0.000, 0.014] | +0.010 [0.000, 0.054] SEP |
| person | 0.177 [0.117, 0.225] | 0.111 [0.066, 0.152] | +0.065 [0.048, 0.078] SEP |
| rider | 0.094 [0.061, 0.147] | 0.046 [0.033, 0.072] | +0.048 [0.025, 0.077] SEP |
| stroller | 0.008 [0.000, 0.016] | 0.003 [0.000, 0.010] | +0.005 [-0.002, 0.012] ns |
| animal | 0.079 [0.000, 0.130] | 0.081 [0.000, 0.272] | -0.002 [-0.140, 0.061] ns |
| protruding_object | 0.021 [0.001, 0.052] | 0.005 [0.000, 0.020] | +0.016 [-0.001, 0.047] ns |

### Census at the DECLARED gate (sigma >= 0.5)

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

## 3. Map read: 10 cm IoU per class x distance band

10 cm map IoU per class x distance band, 4686 windows / 137 episodes (cells with GT; n_gt = refcv8_ema48852 GT cells)

| class | band | n_gt | refcv8_ema48852 IoU | refcv7_50400 IoU | paired refcv8_ema48852 - refcv7_50400 |
|---|---|---|---|---|---|
| seen, no map class | 0-20m | 225303440 | 0.617 [0.586, 0.647] | 0.596 [0.564, 0.625] | +0.022 [0.013, 0.031] SEP |
| seen, no map class | 20-40m | 228852225 | 0.667 [0.633, 0.697] | 0.659 [0.625, 0.688] | +0.008 [0.004, 0.012] SEP |
| seen, no map class | 40-60m | 206382013 | 0.613 [0.575, 0.647] | 0.608 [0.570, 0.640] | +0.006 [0.002, 0.009] SEP |
| seen, no map class | 60-80m | 171569778 | 0.581 [0.538, 0.618] | 0.575 [0.532, 0.612] | +0.006 [0.002, 0.010] SEP |
| seen, no map class | 80-100m | 136444084 | 0.567 [0.522, 0.606] | 0.561 [0.516, 0.600] | +0.006 [0.001, 0.010] SEP |
| drivable | 0-20m | 115850784 | 0.605 [0.586, 0.624] | 0.606 [0.587, 0.624] | -0.001 [-0.005, 0.004] ns |
| drivable | 20-40m | 116023385 | 0.613 [0.590, 0.637] | 0.604 [0.580, 0.628] | +0.009 [0.005, 0.013] SEP |
| drivable | 40-60m | 97583637 | 0.548 [0.522, 0.575] | 0.538 [0.511, 0.566] | +0.009 [0.005, 0.014] SEP |
| drivable | 60-80m | 78581503 | 0.517 [0.485, 0.549] | 0.504 [0.469, 0.537] | +0.013 [0.008, 0.018] SEP |
| drivable | 80-100m | 61443291 | 0.490 [0.457, 0.523] | 0.478 [0.445, 0.511] | +0.012 [0.007, 0.017] SEP |
| lane / road line | 0-20m | 5268358 | 0.199 [0.170, 0.224] | 0.195 [0.169, 0.220] | +0.004 [-0.002, 0.008] ns |
| lane / road line | 20-40m | 5328444 | 0.060 [0.041, 0.079] | 0.046 [0.032, 0.061] | +0.013 [0.004, 0.024] SEP |
| lane / road line | 40-60m | 4772631 | 0.026 [0.014, 0.042] | 0.017 [0.009, 0.027] | +0.009 [0.003, 0.017] SEP |
| lane / road line | 60-80m | 3951487 | 0.005 [0.001, 0.012] | 0.001 [0.000, 0.003] | +0.004 [0.000, 0.009] SEP |
| lane / road line | 80-100m | 3362928 | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | +0.001 [-0.000, 0.003] ns |
| crosswalk | 0-20m | 2022495 | 0.149 [0.113, 0.185] | 0.140 [0.105, 0.175] | +0.009 [0.003, 0.017] SEP |
| crosswalk | 20-40m | 1912634 | 0.001 [0.000, 0.001] | 0.000 [0.000, 0.000] | +0.001 [0.000, 0.001] SEP |
| crosswalk | 40-60m | 1405290 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| crosswalk | 60-80m | 962264 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| crosswalk | 80-100m | 728875 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| arrow / text | 0-20m | 228468 | 0.088 [0.068, 0.111] | 0.082 [0.060, 0.104] | +0.006 [-0.001, 0.014] ns |
| arrow / text | 20-40m | 233068 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [-0.000, 0.000] ns |
| arrow / text | 40-60m | 148631 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| arrow / text | 60-80m | 116940 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| arrow / text | 80-100m | 89959 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| non-drivable edge | 0-20m | 2075376 | 0.001 [0.000, 0.002] | 0.001 [0.001, 0.002] | -0.000 [-0.001, 0.001] ns |
| non-drivable edge | 20-40m | 2033703 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.000 [-0.000, 0.000] ns |
| non-drivable edge | 40-60m | 1627600 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| non-drivable edge | 60-80m | 1251428 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| non-drivable edge | 80-100m | 957614 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| hatched area | 0-20m | 233513 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| hatched area | 20-40m | 221940 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| hatched area | 40-60m | 128673 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| hatched area | 60-80m | 105690 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| hatched area | 80-100m | 99317 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] ns |
| sidewalk / verge | 0-20m | 187323924 | 0.459 [0.427, 0.488] | 0.460 [0.428, 0.487] | -0.001 [-0.009, 0.006] ns |
| sidewalk / verge | 20-40m | 187054717 | 0.529 [0.491, 0.562] | 0.517 [0.480, 0.549] | +0.012 [0.007, 0.016] SEP |
| sidewalk / verge | 40-60m | 159962359 | 0.456 [0.417, 0.494] | 0.447 [0.409, 0.485] | +0.009 [0.004, 0.014] SEP |
| sidewalk / verge | 60-80m | 128364038 | 0.419 [0.378, 0.462] | 0.408 [0.368, 0.451] | +0.010 [0.005, 0.016] SEP |
| sidewalk / verge | 80-100m | 99923030 | 0.401 [0.357, 0.447] | 0.391 [0.346, 0.437] | +0.011 [0.004, 0.017] SEP |

## 4. Zero-training inference fixes, fitted on TRAIN-DIAG only, frozen, re-scored on EVAL

Fit set (TRAIN, never EVAL): TRAIN-DIAG: 139 TRAIN episodes x 8 windows of the refcv8 train4572 cache (sha12-sorted, first 139 with a 10 cm GT file); EMA-48,852 weights; unshifted pose clock; 1074 box windows. Tool `stack/scripts/refit_perception_thresholds.py` (sha256 5a6db392be352a07...). Checkpoint step 48852, md5 b4364029b68f29a33eeb4aebbbbd002b.

| head | P=R gate, no NMS | NMS radius (TRAIN-AP@2 m argmax) | gate re-fitted after NMS | train AP@2 m at that radius |
|---|---|---|---|---|
| box3d | 0.2660 | 2.5 m | 0.2165 | 0.4722 (no NMS 0.3462) |
| agent | 0.2394 | 3.0 m | 0.1844 | 0.4161 (no NMS 0.1865) |

box3d TRAIN grid (radius -> TRAIN AP@2 m / TRAIN P=R gate): none -> 0.3462 / 0.2660; 0.5 -> 0.3532 / 0.2638; 1 -> 0.3954 / 0.2540; 1.5 -> 0.4440 / 0.2391; 2 -> 0.4658 / 0.2264; 2.5 -> 0.4722 / 0.2165; 3 -> 0.4561 / 0.2080; 4 -> 0.4202 / 0.1943

refcv7@50,400's shipped values for reference (fit on ITS TRAIN-DIAG): box3d P=R gate 0.2567, NMS 2.5 m, gate after NMS 0.2145.

### Census at the TRAIN P=R gate, no NMS

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

### Census with the TRAIN NMS + its re-fitted gate

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

### Within-arm effect of the fixes (same arm, same draws)

**refcv8_ema48852** (variant minus declared; paired = the same draws within the arm)

| effect | delta [95 % CI] |
|---|---|
| nms_ap2_all_minus_ap2_all | +0.119 [0.096, 0.139] SEP |
| nms_map2_minus_map2 | +0.031 [0.019, 0.043] SEP |
| train_PR_gate_noNMS_minus_declared__prec | -0.641 [-0.653, -0.629] SEP |
| train_PR_gate_noNMS_minus_declared__rec | +0.347 [0.311, 0.381] SEP |
| train_PR_gate_noNMS_minus_declared__f1 | +0.333 [0.311, 0.353] SEP |
| train_PR_gate_noNMS_minus_declared__conf_ratio | +0.991 [0.894, 1.080] SEP |
| train_PR_gate_noNMS_minus_declared__boxes_per_object | +1.142 [1.033, 1.274] SEP |
| train_PR_gate_noNMS_minus_declared__frac_objects_ge2 | +0.678 [0.651, 0.703] SEP |
| train_NMS_plus_gate_minus_declared__prec | -0.546 [-0.567, -0.525] SEP |
| train_NMS_plus_gate_minus_declared__rec | +0.447 [0.411, 0.483] SEP |
| train_NMS_plus_gate_minus_declared__f1 | +0.431 [0.405, 0.457] SEP |
| train_NMS_plus_gate_minus_declared__conf_ratio | +1.000 [0.931, 1.072] SEP |
| train_NMS_plus_gate_minus_declared__boxes_per_object | +0.088 [0.075, 0.104] SEP |
| train_NMS_plus_gate_minus_declared__frac_objects_ge2 | +0.087 [0.075, 0.101] SEP |

**refcv7_50400** (variant minus declared; paired = the same draws within the arm)

| effect | delta [95 % CI] |
|---|---|
| nms_ap2_all_minus_ap2_all | +0.096 [0.080, 0.113] SEP |
| nms_map2_minus_map2 | +0.016 [0.012, 0.021] SEP |
| train_PR_gate_noNMS_minus_declared__prec | -0.679 [-0.703, -0.632] SEP |
| train_PR_gate_noNMS_minus_declared__rec | +0.288 [0.251, 0.323] SEP |
| train_PR_gate_noNMS_minus_declared__f1 | +0.284 [0.262, 0.303] SEP |
| train_PR_gate_noNMS_minus_declared__conf_ratio | +0.965 [0.838, 1.079] SEP |
| train_PR_gate_noNMS_minus_declared__boxes_per_object | +1.191 [1.078, 1.292] SEP |
| train_PR_gate_noNMS_minus_declared__frac_objects_ge2 | +0.686 [0.650, 0.714] SEP |
| train_NMS_plus_gate_minus_declared__prec | -0.586 [-0.618, -0.536] SEP |
| train_NMS_plus_gate_minus_declared__rec | +0.377 [0.343, 0.412] SEP |
| train_NMS_plus_gate_minus_declared__f1 | +0.375 [0.350, 0.400] SEP |
| train_NMS_plus_gate_minus_declared__conf_ratio | +0.962 [0.880, 1.039] SEP |
| train_NMS_plus_gate_minus_declared__boxes_per_object | +0.111 [0.097, 0.124] SEP |
| train_NMS_plus_gate_minus_declared__frac_objects_ge2 | +0.110 [0.097, 0.122] SEP |


## 5. WiSE decoder-only blend (DEC) as a control

DEC = WiSE decoder-only blend (md5 dad4cb5b5fb316922b4e9a7e85cee8f8; build record: outside_set_is_theta8 = true), 297 windows (every 16th S2 window), unshifted clock, vs the EMA-48,852 pass A on the same (sha12, ws) keys.

| head | windows compared | bit-equal | differ | max abs diff (logit / xy / pair_err) |
|---|---|---|---|---|
| box3d slots | 281 | 281 | 0 | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| agent slots | 281 | 281 | 0 | 0.00e+00 / 0.00e+00 / 0.00e+00 |
| 10 cm map rows | 293 | 293 (order-preserving embedding into A's table, bit-identical statistics) | 0 | |

Control verdict: **PASS** (box and map must read identically).

## 6. Reading

**What the read says (T0, one training seed per arm, CIs = another draw of episodes only).** No pre-registered bar was supplied for this read, so no bar is judged; these are measured deltas.

1. **refcv8's box head is better than refcv7's on the declared protocol, at every threshold and in every distance band.** Pooled AP@2 m 0.328 [0.301, 0.354] vs 0.249 [0.227, 0.270] -- paired delta +0.079 [0.071, 0.087]; AP@4 m +0.078, AP@1 m +0.069, AP@0.5 m +0.035 (0.065 vs 0.030). Near field gains most (AP@2 m 0-20 m 0.539 vs 0.418). Positives are identical (14,586 on both sides), so the delta is not a different-label artifact. Class-mean AP@2 m 0.102 vs 0.070 (+0.032 [0.015, 0.050]): the gain is carried by automobile (+0.087), person (+0.065), rider (+0.048), bus (+0.067); of the six rare classes (heavy_truck 0.144, animal 0.079, protruding_object 0.021, other_vehicle 0.018, trailer 0.011, stroller 0.008) five have paired intervals that span zero and trailer is only marginally separated (+0.010 [0.000, 0.054]) -- unresolved, not improved. The trainer's own in-run EMA AP@2 m at the same step (0.314, in-run subset, corrected clock; INHERITED from `E:/refcv8_eval_kit/thor_reads/metrics_20261020T210033.jsonl`, not quotable) points the same way.
2. **Ranking quality is mixed.** AUROC over matched slots rose (0.931 vs 0.914, +0.017 [0.015, 0.020]) but AUROC over the objectness label (a slot within 2 m of a positive) FELL (0.890 vs 0.907, -0.017 [-0.023, -0.012]). Both are separated against episode resampling; I have no mechanism for the second and do not offer one.
3. **The declared 0.5 gate is still unusable on refcv8**: recall 0.013 [0.008, 0.019] vs 0.008, conf_ratio 0.013 (the A10 alarm band is [0.5, 1.5]) -- 195 confident slots over 14,586 positives. Any recall / precision statement needs the F4 gate. At the TRAIN P=R gate (0.2660, refcv7's shipped 0.2567) refcv8 reads recall 0.360 / precision 0.359 / F1 0.359 vs 0.296 / 0.304 / 0.300, conf_ratio 1.005 (inside the band).
4. **The duplicate-box defect is unchanged**: 2.14 boxes per detected object at the P=R gate (refcv7 2.19; delta -0.049 [-0.124, 0.026], not separated), 67.8 % of detected objects carry >= 2 boxes (refcv7 68.6 %). The F4b NMS fitted on refcv8's TRAIN-DIAG picks the same radius as refcv7's (2.5 m; TRAIN-AP@2 m 0.472 vs 0.346 without NMS) and gate 0.2165 (refcv7 0.2145). Frozen and re-scored on EVAL it lifts refcv8's AP@2 m 0.328 -> 0.447 (within-arm +0.119 [0.096, 0.139]; refcv7 0.249 -> 0.345, +0.096), F1 0.026 (declared gate) -> 0.457, boxes per object 1.088, conf_ratio 1.013. Under the NMS the refcv8 - refcv7 AP@2 m gap is +0.102 [0.089, 0.115].
5. **The map is essentially unchanged**: IoU deltas against refcv7 are -0.001 to +0.013 for the map classes (the background cell 'seen, no map class' +0.022 at 0-20 m, +0.006..+0.008 beyond) (drivable 0-20 m -0.001 ns, 20-100 m +0.009..+0.013; lane +0.004..+0.013 in the first two bands; sidewalk +0.012; crosswalk 0-20 m +0.009), far-band thin classes remain at ~0. "Separated" here is a necessary, not sufficient, condition (the paired bootstrap is blind to training variance, `H-ESTIM-SEED-1`; no replicate of either arm exists); do not quote any of these map deltas as a lever effect.
6. **Controls all read their known values** (section 1), including the WiSE control: the decoder-only blend's 281 box windows and 293 map rows are bit-identical to EMA-48,852 (0 differences).

**Caveats / what this read does NOT establish.**
* *Clock.* The primary read uses the unshifted pose clock so the ground truth is the one refcv7 was scored on. refcv8 trained on the pose-synced clock (shift mean 16.5 ms, max 35.4 ms), so the tight-threshold numbers (AP@0.5 m) may be conservative for refcv8. The corrected-clock read was NOT run (time budget): it is one command on Thor (`touch /home/nvidia/r8ns/perc8/status/RUN_A_SYNC`, then `bash code/chain_perc8.sh`; ~35 min, or `RUN_SYNC_SUB` for the every-4th-window subset, ~10 min). Blocker: wall clock only.
* *One seed per arm.* refcv8 was warm-started from refcv7 (`--init-from`); the paired delta is a one-seed difference between two trained runs. A separated interval is necessary, not sufficient. The cheapest sufficient next experiment is a replicate arm (a second refcv8 seed), not more windows.
* *T0.* This is a perception read on teacher-forced windows; it says nothing about driving.
* *refcv7 checkpoint identity.* refcv7's shipped gate/NMS files name checkpoint md5 d5f104ee... (the Thor run's `ckpt.pt`), the banked eval-grade dump was produced from the kit's `ckpt_50400.pt` (md5 b418d0fc...). Both are step 50,400 per their own provenance; equality of weights is INHERITED, not re-verified here. It affects only the refcv7 side of the NMS/gate rows.
* *Eval inputs.* The 139 eval-cache files on Thor are md5-identical to the dev-box kit's (139/139 by the other agent's Thor md5 list against my dev-box md5s; 4/4 re-measured by me on Thor). The S2 window list is byte-identical to the one in `windows_equality.json` (md5 a3a78dba...), and the pass reproduces refcv7's window counts exactly (4,754 / 4,686 map / 4,532 box).
* *Agent head.* The agent-slot packs are in the dump (`raw/A_nosync/perc_det.pkl`, key `agent`) and were refit (gate 0.2394, NMS 3.0 m / 0.1844) but are not scored here.


## 7. Deliverable manifest

Everything below is **unstaged and uncommitted** (the orchestrator lands it). Dev-box root `C:/Users/Admin/refcv8_pf/perc8/` (not on G:/D:); Thor root `/home/nvidia/r8ns/perc8/`.

| artifact | where |
|---|---|
| `PERC8_EVAL.json` (merged: provenance, controls, box read, map read, refit, re-score, WiSE control) | dev box `C:/Users/Admin/refcv8_pf/perc8/PERC8_EVAL.json` |
| `RESULT.md` (this file) | dev box `C:/Users/Admin/refcv8_pf/perc8/RESULT.md` |
| code (all scripts, the verbatim originals in `code/orig/`, the two Thor remap JSONs, the chain / waiter / smoke launchers, `READING.md`, `MANIFEST.md`) | dev box `.../perc8/code/` and Thor `.../perc8/code/` (the Thor copy is the one that ran) |
| pass A, EMA-48,852, S2 (4,754 windows): `perc.json`, `perc_det.pkl` (per-window box/agent packs), `perc_map_table.npz` (per-window map statistics), log | dev box `.../raw/A_nosync/`; Thor `.../out/A_nosync/` |
| TRAIN-DIAG pass (1,112 windows / 139 train episodes): `train_diag.json`, `train_diag.packs.pkl`, refit outputs `refcv8_det_presence_gates_train.json`, `refcv8_det_nms_train.json`, logs incl. the FAILED first attempt | dev box `.../raw/T_nosync/`; Thor `.../out/T_nosync/` (+ `train_diag.acc.pt`, 6 MB, Thor only) |
| WiSE DEC control pass (297 windows) + comparison `wise_control.json` | dev box `.../raw/W_dec_sub/`, `.../raw/wise_control.json`; Thor `.../out/W_dec_sub/` |
| box read: baseline (all AP/AUROC cells, controls C1-C3) `box_read_declared.json`; variants (refit gate / NMS) `box_read_final.json`; per-window rows `rows/*.npz` (regenerable with `code/perc8_rows.py`) | dev box `.../raw/` |
| map read `map_read.json`; GT control `gt_control.json`; refcv7 re-score control `rescore_refcv7_control.json`; tables `tables/*.md`; self-tests `selftest/` | dev box `.../raw/` |
| inputs: `stage/s2_windows.json` (md5 a3a78dba889e70aa69a05271559794ad), `s2_every16.json`, `s2_every4.json`, refcv7's shipped gate / NMS JSONs | dev box `.../stage/`; Thor `.../inputs/` |
| md5 evidence for the eval cache (dev box 139, other agent's Thor 139, my Thor sample 4) | dev box `.../raw/md5_*.txt` |

Not run: the corrected-clock (pose-synced) read -- the launcher supports it (`status/RUN_A_SYNC`, `status/RUN_SYNC_SUB`); blocker is wall clock only.
Modified originals: `code/r8_repoint.py` (one line: the remap file name now comes from env `R8_REMAP_FILE`, default `refcv8_eval_remap_thor.json`; verbatim original in `code/orig/`), `code/run_diag.py` (the map-loss wrapper forwards extra keywords: the launch tree's `map_hires_loss_row` gained `class_thresholds` after the wrapper was written; the first TRAIN-DIAG attempt died on it). Nothing under the repo tree was edited.

Script provenance: `raw/box_read_declared.json` = `code/perc8_box_read.py` (full AP/AUROC grid + controls C1-C3, 1,000 draws); `raw/box_read_final.json` = `code/perc8_box_read2.py --light` (a copy of the former with a variants-only mode: AP/mAP@2 m raw + NMS and the census at the refit gate / NMS; 1,000 draws, same seed 0 draw stream, controls skipped because the baseline run already holds them).
