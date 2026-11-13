> LEAD NOTE (PREREG_V8F_PROOF_ADD13_N2.md, N2a): S-1/4 runs on the restricted cache root WITH THE FULL TRAINING JOIN (--join-train /workspace/v8proof/data/inputs/join_train.npz), not with join_S_quarter.npz. Where a line below names join_S_quarter.npz, the full join is used.

# ARMS: the 10 arms of PREREG_V8F_PROOF_ADD13 (5 arms x seeds 0, 1), pod argv

Built by dataeff_prep.py. Pod paths follow the HIERREP chain (`W=/workspace/v8proof`, cache `data/cache_pod`, inputs `data/inputs`). `<OUT>` = the run root. One line per run, executed as
`python <TREE>/stack/scripts/v8f_mini_train.py <argv>` at commit 3e264bbf, each under its own selfcheck gate (`--selfcheck-only` appended, `--out` as in the chain).
Differences from the HIERREP argv: `--exclude 540:600` becomes `540:B`; `--o-interface cut` for F-1/4; `--steps 19164` added (it wins over `--steps-mult 4`, train.py:2703); S-1/4 data flags (below).

## Selfcheck expectations (from the tool, the same rules as the trainer)

| arm | --exclude | n_clips | n_clips_all | n_windows (= windows of clips_all) | S = ceil(n_windows/64) | steps |
|---|---|---|---|---|---|---|
| R-1/2 | 540:2856 | 2256 | 2256 | 153284 | 2396 | 19164 |
| R-1/4 | 540:3984 | 1128 | 1128 | 76631 | 1198 | 19164 |
| R-1/8 | 540:4548 | 564 | 564 | 38313 | 599 | 19164 |
| F-1/4 | 540:3984 | 1128 | 1128 | 76631 | 1198 | 19164 |
| S-1/4 | none (restricted manifest) | 1128 | 1128 | 76663 | 1198 | 19164 |

Full arm (HIERREP): n_clips 4,512, n_windows 306,566, S 4,791, steps 19164.

## S-1/4 needs two inputs, not one (departure 1, DATAEFF_DEFS.md R9)
The trainer takes its clip set from `<cache-dir>/train/manifest.jsonl` (train.py:2560-2561), and `D.train_windows` indexes the join for every manifest clip (data.py:611-619). A restricted join alone would raise KeyError, so the arm runs on a restricted cache root built by `s_cache_setup.py` from the pod's own manifest and the list `lists/S_quarter.txt`:

`python s_cache_setup.py --src-cache /workspace/v8proof/data/cache_pod --list lists/S_quarter.txt --dst-cache /workspace/v8proof/data/cache_S_quarter`  (writes `train/manifest.jsonl` with only the 1,128 clips in the source file order and a link `train/clips` to the full clip folder; refuses if a listed clip is missing).
The join `join_S_quarter.npz` goes to `/workspace/v8proof/data/inputs/join_S_quarter.npz`. Do NOT pass `--exclude` to S-1/4 (a 540:600 would cut S's own positions). Expected selfcheck: n_clips 1,128, n_excluded 0.

## Argv

### R-1/2 seed 0
```
--arm M2b --seed 0 --grid pool4x10 --out <OUT>/R1_2_s0 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:2856 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### R-1/2 seed 1
```
--arm M2b --seed 1 --grid pool4x10 --out <OUT>/R1_2_s1 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:2856 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### R-1/4 seed 0
```
--arm M2b --seed 0 --grid pool4x10 --out <OUT>/R1_4_s0 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:3984 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### R-1/4 seed 1
```
--arm M2b --seed 1 --grid pool4x10 --out <OUT>/R1_4_s1 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:3984 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### R-1/8 seed 0
```
--arm M2b --seed 0 --grid pool4x10 --out <OUT>/R1_8_s0 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:4548 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### R-1/8 seed 1
```
--arm M2b --seed 1 --grid pool4x10 --out <OUT>/R1_8_s1 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:4548 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### F-1/4 seed 0
```
--arm M2b --seed 0 --grid pool4x10 --o-interface cut --out <OUT>/F1_4_s0 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:3984 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### F-1/4 seed 1
```
--arm M2b --seed 1 --grid pool4x10 --o-interface cut --out <OUT>/F1_4_s1 --cache-dir /workspace/v8proof/data/cache_pod --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_train.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --exclude 540:3984 --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### S-1/4 seed 0
```
--arm M2b --seed 0 --grid pool4x10 --out <OUT>/S1_4_s0 --cache-dir /workspace/v8proof/data/cache_S_quarter --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_S_quarter.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

### S-1/4 seed 1
```
--arm M2b --seed 1 --grid pool4x10 --out <OUT>/S1_4_s1 --cache-dir /workspace/v8proof/data/cache_S_quarter --v9-train /workspace/v8proof/data/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train /workspace/v8proof/data/v9/lead_train4572.npz --speed-train /workspace/v8proof/data/v9/refcv6_speed_max_v8_train.jsonl --join-train /workspace/v8proof/data/inputs/join_S_quarter.npz --p10 /workspace/v8proof/data/inputs/p10_prior.pt --kin-mode yaw --tracks /workspace/v8proof/data/inputs/native_tracks.pt --batch 64 --steps-mult 4 --steps 19164 --device cuda
```

