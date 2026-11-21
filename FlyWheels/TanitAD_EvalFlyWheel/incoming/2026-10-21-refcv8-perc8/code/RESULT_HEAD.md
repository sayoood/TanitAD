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
