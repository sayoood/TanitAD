# Thor cleanup 2026-10-25 — the records of what left the disk

Master Mind, 2026-10-25 ~08:45 CET. PI in chat: *"delete the smoke checkpoints too … You can delete the rest. Assure that at
least the refav1 experments are documented. let tanit astra and sam3 there"*. Trigger: the NT A1 launch preflight needs >= 60 GB
free on Thor and read 52 GB (`PF-FAIL disk-60G`, gate on 009d1cb, 08:13 CET).

## 1. Compressed in place (PI: "Compress in place") — nothing lost
25 superseded launch-kit scratch items in `/home/nvidia/r8ns/nt/` (8 `suite_*`, 3 `gate_*`, 7 `launch_tree_aside_*`, 7 old
`launch_tree_*.tar`) -> `/home/nvidia/r8ns/nt/nt_superseded_20261025.tar.gz`, 11,085,776,437 B, md5
`d726a7a46bf029ce5d7d5d046d9ec677`; `gzip -t` OK and the member list EQUALS a `find` of the originals (108,622 entries) before the
originals were removed. Free space 52 -> 62 GB.

## 2. Approved for deletion — what each was, and where its documentation lives
`ckpt_md5.txt`: md5, bytes, date and path of every checkpoint involved (13). `dir_manifest.txt`: bytes, file count, date per
directory (file names are NOT listed, because several of them are raw clip ids). `runs/<name>/`: the small run records
(config, summary, train logs, drift checks, supervisor logs) of every experiment dir, copied BEFORE deletion.

| item | what it was | documentation |
|---|---|---|
| `experiments/refav1-b1-v72-ep2-ema-bf16/` (2 ckpts) | refav1 epoch-2 EMA / bf16 run, superseded by ep3 | `runs/refav1-b1-v72-ep2-ema-bf16/` |
| `experiments/refav1-b1-v72-1ep-21109/` | refav1 first full epoch (DRIFT_ALARM file present) | `runs/refav1-b1-v72-1ep-21109/` |
| `refav1_lon/` | refav1 longitudinal study; its ckpt is BYTE-IDENTICAL to ep3-speed's (md5 `1189bc02…`) | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/`, `…/2026-09-05-refav1-lonshift-t1/` |
| `experiments/TSC-A-adapter/`, `TSC-B-frozen/` | TSC adapter / frozen arms | `runs/TSC-*/train_log.jsonl` |
| `experiments/wpd-D0-4k/` | WP-D D0 arm | `runs/wpd-D0-4k/` |
| smoke ckpts: `refcv8_run/gate_fb47a583763f/smoke`, `refcv8_run/gate_bc2816862b8c/smoke`, `refcv8_smoke/run`, `r8ns/nt/labels/a1_trainer_smoke/run_smoke`, `refcv8_run/runs/refcv8-r101-s0_gsmeasure` | launch-gate / trainer smoke runs, regenerable | md5 in `ckpt_md5.txt` |
| `qwendrive/` | Qwen-Drive-1.0-4B weights (re-downloadable) | `qwendrive_config.json`, `qwendrive_top.txt` |
| `wpd_a3/bank`, `percprobe/`, `v7tiny/`, `refcv8_wpd/out/pgrad`, `data/_b1stage416` | old experiment banks / staging | sizes in `dir_manifest.txt`; results in their own research packages |

## 3. KEPT on purpose (a deviation from "delete the rest", stated)
`/home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt` — **the registered REF-A v1 model** (MODEL_REGISTRY sec. 2.4;
2,122,997,633 B, md5 `1189bc020018c2c67ce03d566c390285`). The handover's dev-box copy
`C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt` no longer exists (MEASURED 2026-10-25; an exact-size search of C: and E:
found nothing; F: was offline after USB I/O errors and could not be searched) and there is no HF copy, so this file is the last
verifiable copy. 2.1 GB. Delete it only after a second copy exists.

Kept by the PI's instruction: `tanit-astra-research/`, `sam3map/`.

## 4. CORRECTION (09:3x CET): `data/_b1stage416` was NOT deletable — 4 of its files were pinned by the NT bind
The gate re-run on 009d1cb refused at 09:30 CET: `G-MAP-OVERFIT: 4 data file(s) the run read are absent` — four
`r0/camera_front_wide/*.timestamps.parquet` under `data/_b1stage416/`. The bind records (`bind_f5262ea3c01d_nt-a1-s0/{map,box}_closure.json`)
carry a `data` list of `[path, sha256]` (17 map + 34 box) that the gate re-hashes; I had checked only the code `modules` before
calling the directory "staging leftovers". All 4 were restored from the dev box (`C:/Users/Admin/tanitad-data/refav1-train600-256_stage/r0/camera_front_wide/`,
sha256 matched before copy and again on Thor, 4/4), then all 51 recorded data files were verified (map 17/17, box 34/34) before
the gate was re-run (09:36). Rule recorded: before deleting on a host that holds a bind, union every bind record's `data` paths and
never touch them. The rest of the PI-approved list is unaffected (none of it is in either `data` list).
