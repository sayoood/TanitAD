# FEASIBILITY -- train-fitted RETIME gains for refcv8 (EvalFlyWheel, 2026-10-19, dev-box clock ~03:15 Berlin)

Evidence classes: MEASURED = read/executed by me today, path given; INHERITED = another artifact, not re-verified here.
**Verdict: FEASIBLE.** Every input the roll needs for a TRAIN clip exists on the dev box except one 264 MB file (the v9 TRAIN label release), which I pulled from Thor over the LAN and md5-verified. The only new GPU work is **2 os-only rolls (inference seeds 0, 1) over 139 train clips**, ~45 min each by the eval-139 precedent (priced below; the smoke result is in section 6). Everything after the rolls is CPU and re-uses `a0k0_headroom_retime_r8.py` unchanged.

## 1. How the battery's roll reads episode frames for eval139 (traced, MEASURED by reading the files)
`run_battery_r8.py main()` -> `r8_battery_patch.load_battery_module` = the refcv7 `run_battery_r7.main()` (launch tree `C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/`) -> `windows = RR.s2_window_list(P.BASELINES["b_refcv4b"], kit_ids)` (the S2 window list is the **banked refcv4b EVAL dump's `ws` set**; it does not exist for train clips) -> a child `roll_seed_r8.py` (`r8_repoint.apply()` then `roll_seed_r7.main()`) -> `run_battery_r7.roll_one(ckpt, config, seed, dump_dir, windows)` -> `r7_roll.install_patches(...)` + `refcv3_arm.main(["--ckpt", ..., "--config", ..., "--episodes", KIT_EVAL, "--labels", KIT_LABELS, ...])`.
* `KIT_EVAL = L.KIT/data/refcv6-b1-416x1024-eval139` and `KIT_LABELS = .../v8labels/labels/s2_labels_v8_eval.jsonl.gz` are **module constants** of `run_battery_r7.py:42-43`; `r7_roll.build_corpus` additionally refuses unless `--episodes == targs.eval_cache`.
* `targs.eval_cache` comes from the checkpoint's `config.json` argv: `--eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139` (Thor path), mapped to the dev box by the loader's `PATH_REMAP_SPEC` (`refcv7_loader.py:104-139`) plus the RECORDED override file `code/refcv8_eval_remap_devbox.json` (set via `r8_repoint.REMAP_OVERRIDES`). The loader supports exactly this override for "a train-episode roll" (`refcv7_loader.py:134-138`), and the refcv6 A6 amendment used it (`2026-09-23-refcv6-standard-tests/battery/code/a6_roll.py`, `raw/a6/remap_overrides_train.json`).
* The model is trained on `--v2-cache /home/nvidia/data/refcv8-b1-416x1024-train-4572` (Thor; `config.json` `v2_cache`, `v2_parity.clips_present = 4572`). The eval roll never reads it.
* Clock policy of the trees I must be comparable to: `battery_summary.json["r8"]["pose_sync_policy"]` of `C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc` reads **OFF (R8_NO_POSE_SYNC): the standalone, refcv7-comparable clock**. The train roll therefore also runs with `R8_NO_POSE_SYNC=1` (so the pose-sync sidecar is mapped but not applied).

## 2. The train sample (MEASURED)
**The A6 sample already on the dev box: 139 clips, `E:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6/` (11.37 GB, 139 `*.v2ep.pt`).** Selection rule, recorded 2026-09-26 for a different question (refcv6 A6; `.../2026-09-23-refcv6-standard-tests/battery/raw/a6/select_record.json`): *"139 smallest sha12(clip_id) among train-cache clips not in the eval 139, with a train label record and a train max-speed sidecar row"*, candidate pool 4,369 (the refcv6 train cache). It is deterministic, content-hash based (not seeded by any driving property), and was fixed three weeks before this question existed.
* In refcv8's train4572: **139/139** are among the 4,572 records of `s2_labels_v8_train.jsonl.gz` (md5 `b45377a1f25263b5c0f3d318c126b1ac` = Thor's file = the refcv8 argv's `--v7-labels`) and **139/139** are listed in Thor `/home/nvidia/data/refcv8-b1-416x1024-train-4572/` (4,572 entries). **0/139** are in the eval139 cache (probe: set intersection of the two cache directories).
* Same bytes: both Thor train caches (`refcv8-...-4572`, 4,572 entries, and `refcv6-...-train`, 4,369) hold SYMLINKS into `/home/nvidia/data/physicalai-b1-w120-416x1024cyl/`; `stat -L` sizes of all 139 equal the local files' sizes (139/139), and the A6 pull verified md5 against Thor (`pull_record.json`: 0 bad). So these are the very frames refcv8 trained on.
* Disclosed limit: the pool was refcv6's 4,369 train clips; the 203 clips added for refcv8 are not sampled.

## 3. Every input a train-clip roll needs (MEASURED unless marked)
| input (loader flag) | eval139 value | train value used | covers the 139? |
|---|---|---|---|
| frames cache (`--eval-cache`) | `E:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139` | `E:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6` | yes (section 2) |
| labels (`--eval-labels`) | `.../v8labels/labels/s2_labels_v8_eval.jsonl.gz` | `E:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz` | yes, 139/139 of 4,572 |
| max-speed sidecar (`--speed-max-sidecar-v6-eval`) | `refcv6_speed_max_v8_eval.jsonl` | `E:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl` (md5 `bab41adf5d08aef89ebd4596ed48cead` = Thor = refcv8 argv `--speed-max-sidecar-v6`) | yes, 139/139 by sid |
| v9 labels (`--r8-v9-labels-eval`) | `E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz` | **was NOT on the dev box.** Pulled from Thor read-only: `E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz` (264,220,433 B, md5 `83e30bd3791f0767707eade5f4671759` = the refcv8 argv's `--r8-v9-md5`, verified locally after `scp`) + its `.manifest.json` (md5 `7faeec88733c2aa9a29608e3ef5b9c0c`, verified). The loader checks the npz against `args.r8_v9_eval_md5` (argv carries the EVAL md5) so `roll_seed_r8_train.py` sets it to the TRAIN md5 for that file only. | yes (train4572 release) |
| clip-clock sidecar | `E:/refcv8_eval_kit/data/refcv8_clip_clock_sidecar_ts33.jsonl` | same file | 139/139 by sid (probe) |
| pose-sync sidecar | `.../refcv8_pose_sync_sidecar.jsonl` | same file, mapped but NOT applied (standalone clock, section 1) | 139/139 |
| extrinsics, vis1, nav-compliance tau | files named `*train4572_eval139*` / `*train4572*` | same files | extrinsics 139/139 (4,711 clips); vis1 npz covers 4,711 clips; tau is a global scalar file |
| `--join-defect-masks` | `C:/lgs/Lfb47a58376/stack/tanitad/configs/refcv8_join_label_defects_train4572.json` | same | train4572 file |
| agent join / join3d | `b1_train_plus_eval_agents*.jsonl.xz` | same files (train + eval) | yes (INHERITED from the name; the roll builds the dataset with `with_perception_targets=False`, so these are not read in the roll) |
| map GT root | `sam3_gt_v3_eval` (eval clips only) | not needed: `roll` builds the eval dataset with `with_perception_targets=False` (`r7_roll.py:~400`) | n/a |
| window list | banked refcv4b EVAL grid (stride 5, gap 5 in 4,615/4,615 steps; per-clip first ws 7..11) | rule: the dataset index with `t % 5 == 0` (A6 precedent), S6 cut by the battery's own `build_s6_dump` | rule, not a file |
| checkpoint / config | `E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt`, `ckpt/config.json` | same | -- |
| fed-ceiling inputs for the compliance mask | `v_max_ms`, `v_max_valid`, `gt_max_speed_6s` from the roll's `refcv6_extras.npz` | same extras, written by the same `save_extras` | produced by the roll |

Sidecars where the gain fit and the compliance rule read the model's inputs (`v0`, `v_max_ms`) are therefore the SAME channels the eval windows get; nothing is substituted by an eval-side file.

## 4. Where the downstream (CPU) side reads
Fit/score re-use `a0k0_headroom_retime_r8.py` unchanged: `references()`, `load_decisions()`, `retime_fit_pre()`, `forward(tree)`. The train tag directory is laid out like an archive tree (`dump_s{0,1}`, `panel_s{0,1}`, `s6_s{0,1}`) by the battery's own `build_panel_dump(baselines={})` / `build_s6_dump(baselines={})` (no banked baselines exist for train clips; the eval r8 trees were also built with no refcv4b pairing on the corrected clock). The existing eval trees are at `C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc` (89 MB) and `.../step30000_unc`.

## 5. Pricing (eval precedent, MEASURED from `roll_s{0,1}.json` of `step30000ema_unc`)
* eval139 EMA-30k: seed 1 **os-only** roll = **2,730.6 s (45.5 min)** for 4,754 windows, `cuda_max_memory_allocated_gib` **1.222**; seed 0 full-arm roll = 8,537 s (not needed here: os + model-free only).
* train: ~139 clips x ~34 windows (stride 5) ~ **4.7k windows** per seed -> **~45 min GPU per seed, ~1.5 h for both seeds** (ESTIMATED from the eval rate; the smoke below measures the per-window CPU path only).
* RAM: eval os-only rolls ran under the dev box's COMMIT limit; free commit now is 8.1-8.6 GB. The queue's memory gate is `FreeVirtualMemory >= 10 GB` twice 60 s apart (`kdraw_queue_r8.sh`); I use the same gate. Peak process RAM is measured in the smoke (section 6).
* Disk: 2 dumps (~100 MB each; the eval `step30000ema_unc` tree incl. panels is 89 MB) + the 264 MB v9 file + 11.4 GB cache already on disk. Negligible.
* After the rolls: panels, fit, freeze, two `forward` calls: CPU, minutes (`retime` pipeline precedent: 20-30 min total incl. the heavy compute; the train fit is one surface-pair fit, far smaller).

## 6. Smoke (CPU, 2 clips x 3 windows, EMA-30k, seed 0) -- see the section appended below once it finishes

### 6. Smoke result (MEASURED, `E:/refcv8_eval_kit/work/trainfit_smoke/{smoke.log,roll_s0.json}`; CPU, `--max-clips 2 --max-windows-per-clip 3`, EMA-30k, seed 0)
* `ZZRTTRAINROLLOKZZ`: the whole path runs on TRAIN clips: remap -> dataset (139 clips; labels md5 `b45377a1...`; `nav_from_v7` missing 0; road-type speed 23,751/23,751 windows fed) -> model load (strict) -> os-only dump -> extras with `gt_max_speed_6s`, `v_max_ms`, `v_max_valid`, `v0`, `ws`. Wall 60.2 s for 6 windows (CPU, dominated by the 8.2 s first forward and model load; not a GPU rate).
* The train dataset index has **23,751 windows**; at `t % 5 == 0` that is **~4,750 windows** (the eval S2 list is 4,754) -> the GPU price above (~45 min per os-only seed) stands. Label clock: 139/139 clips from the clip-clock sidecar, G3 PASS (417 reads, worst error 0.0 s), 0 clips excluded.
* First attempt failed loudly (missing `v9_labels_train.manifest.json` beside the npz); fixed by pulling that 6 KB file from Thor (md5-verified). No other blocker.
* Peak process RAM / commit was NOT measured in the CPU smoke (UNVERIFIED); the GPU eval precedent is `cuda_max_memory_allocated 1.222 GiB` for the os-only roll, and the chain below gates on FreeVirtualMemory >= 10 GB.
