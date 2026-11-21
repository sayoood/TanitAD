# HANDOFF_LORAPOD1 (2026-11-06, pod clock 21:35 UTC = dev box 22:35)

Pod address: `C:/Users/Admin/lorapod/pod_direct.txt`; `podenv.sh` gives `pssh` / `pscp`. Nothing in the repo or artifacts holds it.

## Result (MEASURED, `identity_pod/IDENTITY_POD.{json,md}`)

* **Decision by the fixed rule: B' ROUTE.** `b vs c` rel-L2 = 2.42e-2 (8f387977), 2.63e-2 (bf5ec76e), 2.25e-2 (dca157a0); bar 1e-3 on all 3. The builder's own `encode_episode` path gives the same numbers bit for bit.
* `a vs c` (byte-identical banked frames, only the GPU differs) already reads 2.1e-2 to 2.3e-2, so byte-identical frames would not rescue B on this GPU (the handoff's option B is not a remedy). `a vs b` (route alone) reads 2.2e-2 to 2.6e-2. Same-device batch 16 vs 8 reads 2.2e-2: the fp8 token grid turns any last-ULP bf16 change into about 2e-2.
* Max abs diff is 0.25 (one fp8 step); 74 to 80 % of fp8 values are bitwise equal, 0 % of whole tokens.
* Controls: repeat encode bit-equal; arm encoder at B = 0 equals the builder bit for bit; driver sha256 equals the banked AR-2 run 3 driver.
* Throughput (item 5): builder forward **10.56 ms/frame** at batch 8, peak 1.28 GiB; 470k frames = **1.38 h** GPU. LoRA module at B = 0: 12.73 ms/frame. A whole clip end to end runs about 2.8 s (CPU-bound: PNG decode and pooling).

## Item 7: the detached trailing re-encode is RUNNING (launched 21:29:57 UTC)

* Output `/root/data/cache_pod/{train,eval}` (clips/, manifest.jsonl, meta.json, runs.jsonl, `pod_provenance.jsonl`, `compare_box2.jsonl`). Same layout and manifest-row keys as box 2's `cache_full`. Smoke (2 clips) opened by M2b's `ClipStore`/`read_manifest`; file size equals box 2's.
* Log `/root/logs/cache_pod.log`; markers `ZZ CACHEPOD DONE ZZ` / `ZZ CACHEPOD FAIL ZZ`. It finishes only after the build's `ZZ V2EP DONE ZZ` AND all 4,713 ledger clips are encoded. pid in `/root/logs/cache_pod.pid`.
* Progress: `tail -n 2 /root/logs/cache_pod.log; wc -l /root/data/cache_pod/train/manifest.jsonl /root/data/cache_pod/eval/manifest.jsonl; grep -a PROGRESS /root/logs/v2ep_run.log | tail -1`
* Relaunch (safe, resumes by manifest + v2ep sha256): `timeout 30 ssh <pod> 'setsid nohup nice -n 10 bash /root/identity_pod/run_cache_pod.sh < /dev/null > /dev/null 2>&1 &'`
* **ETA (ESTIMATED): done about 00:30 to 01:00 UTC pod clock (01:30 to 02:00 dev box).** Basis: build 1,640 new clips/h from 700/4,713 at 21:34, so build done about 00:00 UTC; encoder 2.76 s/clip under load (1,300/h), backlog about 1,500 clips at build end, then 1.5 to 2 s/clip once the CPU is free. A second encoder process would roughly double the rate if the CPU quota frees earlier.
* The per-clip comparison against box 2 exists only for the 6 pulled clips (`compare_box2.jsonl`; box 2's full cache is about 98 GB and was not pulled).

## Exact next step

1. MM: write Amendment 4 (venue RTX 4000 Ada, torch 2.11.0+cu128, transformers 5.15.0; Linux CPU frame route; B' ROUTE with the numbers above) before any training step.
2. When `ZZ CACHEPOD DONE ZZ` appears: check manifest counts (train 4,572, eval 141), run `loader_check.py` on one train and one eval clip (`PYTHONPATH` needs `/root/tad/taniteval`), then bank frames and cache to HF private (container disk only, no network volume; about 174 GB frames + about 98 GB cache).
3. Then retrain M2b = B' at seeds 0 and 1 on `/root/data/cache_pod` (launch gate PASS first); F' comes from B'.

## Paths

Dev box `E:/Projects/TanitAD-artifacts/dagland/motion_c/lora_pod/identity_pod/`: IDENTITY_POD.json/.md, ident_extra.json, run.log, loader_check_smoke.json, smoke_compare_box2_*.jsonl, scripts (identity_pod.py, ident_extra.py, cache_pod.py, run_cache_pod.sh, loader_check.py, ident_env.sh). Pod: `/root/identity_pod/` (code, frames_a, cache_c = 6 box-2 files, site = transformers overlay), `/root/data/cache_pod`, `/root/data/cache_pod_smoke`. Scratch on dev box: box 2's manifests in the session scratchpad only.
