# P1 -- where the train-split inputs live, and whether the dev box can run the capture without a large copy

Probed 2026-10-22 00:21-00:27 +0200 (Berlin). Dev box: `ls`/`du`/size reads. Thor (`thor6`): read-only `ssh -n tanitad-thor 'ls ...'` and one `stat` loop over 139 file names; **nothing was written on Thor today**
(one earlier 1.7 KB `scp` into Thor `/tmp`, since deleted and confirmed absent, is disclosed in the hand-back). No download, no GPU, no compute on Thor.

## Verdict

**Yes: the dev box can run the TRAIN139 capture with no large copy.** Every heavy input is already on `E:`:
checkpoint (403 MB), the 139-file A6 episode cache (**11.37 GB**, 139 files, byte sizes equal to Thor's), labels, speed-max sidecar and the agent join (327 MB).
The **one** gap is the SAM3 10 cm map codes for the 139 train clips (**610,535,121 B = 0.61 GB, 139/139 present on Thor**, absent on the dev box, which holds only `sam3_gt_v3_eval`).
Options: (1) LAN copy of those 139 files Thor -> dev box (~0.6 GB; the LAN rate is ~63 MB/s, INHERITED from the 2026-09 Thor-LAN measurement in memory, so ~10 s) and set `L7C_MAP_ROOT`; or (2) run the dev-box job without it, in which case the label bank carries
the agent futures only (`map_*` keys absent) and the DAC target abstains for those windows. Neither is done here (no downloads, no copy was authorised).
What the dev box does **not** give: a measured GPU rate for refcv8 (CPU is ~9 s/window, unusable) and the Thor-GPU numerics that K3 needs -- see `RESULT.md` sections 3 and 6.

## Table (size = MEASURED today unless marked INHERITED)

| input | dev box | Thor | note |
|---|---|---|---|
| EMA checkpoint 48,852 | `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt` (403,200,712 B) | `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt` (403,200,712 B; md5 `b4364029...` INHERITED from the fanceil run, checked at launch) | same bytes on both |
| config | `E:/refcv8_eval_kit/ckpt/config.json` (88,390 B) | `/home/nvidia/r8ns/ckpt/config.json` (md5 `4ce2c772...` INHERITED, checked at launch) | |
| **A6 / TRAIN139 episode cache** (139 x `.v2ep.pt`) | `E:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6` (**11.37 GB**, `du` 11 G) | built at launch as **139 symlinks** into `/home/nvidia/data/refcv6-b1-416x1024-train` (4,372 files) by `make_a6cache_thor.py`, which asserts 139/139 present, 139/139 byte-size equal to the dev-box files, 3/3 md5 probes equal | symlink step itself UNVERIFIED (Windows dev box has no symlink privilege; the test is verify-only) |
| train labels | `E:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz` | `/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz` (2,686,219 B; md5 `b45377a1...`) | |
| speed-max sidecar | `E:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl` | `/home/nvidia/data/refcv6_speed_max_v8_train.jsonl` (880,186 B; md5 `bab41adf...`) | |
| v9 labels (TRAIN, md5 `83e30bd3...`) | via the train roll's own dev re-point (`roll_seed_r8_train.py`, `hold_r8batt`) -- the same path the banked train-fit used | `/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz` (264,220,433 B) | |
| agent join (train + eval) | `E:/refcv6_eval_kit/data/joins/b1_train_plus_eval_agents.jsonl.xz` (327,041,136 B) | `/home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz` (327,041,136 B; md5 `0c31a3a6...`) | log: 139/139 episodes, 22,946/23,751 windows labelled (96.6 %), pad 397 |
| **SAM3 map codes, train139** | **absent** (`E:/refcv6_eval_kit/data/sam3_gt_v3_eval` is eval only) | `/home/nvidia/data/sam3_gt_v3` (4,509 files, 19 GB total); **the 139 train139 files: 139/139 present, 610,535,121 B** | the only copy needed for a dev-box run |
| train pool list (raw ids, 4,369) | not banked | `/home/nvidia/data/refcv6_train_clips.txt` (161,653 B, md5 `bb0e2263...`) | raw ids stay on Thor / in session scratch; never in `l7cap/` |
| eval139 windows / remap (for the K3 captures) | `E:/refcv8_eval_kit/windows/...` | `/home/nvidia/r8ns/fanceil/inputs/windows_s1.json`, `/home/nvidia/r8ns/fanceil/code/fanceil/refcv8_eval_remap_thor_fanceil.json` | md5 literals in `run_l7cap_thor.sh` |
| code tree the Thor run imports | `C:/lgs/Lfb47a58376/stack` (+ taniteval) | `/home/nvidia/r8ns/fanceil/tree/stack`, `.../taniteval`; PF code `/home/nvidia/r8ns/fanceil/code/pf/code` | the banked fanceil run's own tree |
| python | `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (torch 2.11.0+cu128) | `/home/nvidia/venvs/tanitad-train/bin/python` (numpy 2.5.1, torch 2.13.0+cu130) | the same venv the banked fanceil Thor run used |

## Thor state at probe time (read-only)

* `/home/nvidia/r8ns/l7cap` **does not exist yet** (nothing shipped). `/home/nvidia/r8ns/fanceil` exists with `code/ inputs/ out/ ship/ tree/`.
* Disk: `/dev/nvme0n1p1` 937 G, 786 G used, **103 G free (89 %)**. The pilot writes ~1.6 GB per seed; the full plan (~25 GB) fits but should be sharded.
* The GPU lock file `/home/nvidia/refcv7_post/thor_gpu.lock` exists (0 bytes, the flock target). At the earlier probe Thor was running another GPU job; `run_l7cap_thor.sh` therefore **waits on the flock and fails closed** on a non-empty
  `nvidia-smi --query-compute-apps`. I did not re-read `nvidia-smi` today.

## What is NOT probed

The md5 of Thor's checkpoint/config/labels/join were not recomputed today (hashing 400 MB-class files was not asked for); the launcher does it at start and exits 2 on any mismatch.
Thor-side sizes of the 139 A6 files **were** read earlier in this task: `ZZSIZECHKZZ ids_missing=0 files_checked=278 size_equal=278 size_differ=0 (2 roots x 139 expected)` -- every one of the 139 files has the dev-box byte size in each of two Thor cache roots (MEASURED; the dev-box sizes are in `inputs/a6_sizes_md5_sha12.json`).
The 3 md5 probes (`001e7d520dd0`, `03ef04f0d748`, `086bb25ad2fd` are hashes, not clip ids) were computed on the dev-box files only; their Thor-side equality is checked by `make_a6cache_thor.py` at launch and is **UNVERIFIED** until then.
