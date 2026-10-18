# INVENTORY_THOR -- disk inventory and safe cleanup plan, 2026-10-04

Host `thor6` (`ssh tanitad-thor-wifi`, user nvidia). `/` = 937 GiB NVMe ext4 (`df -B1`: size 1,005,441,994,752 B, used 891,963,289,600 B, **free 62,329,552,896 B = 62.3 GB**). `df -h` prints GiB, this document uses decimal GB (1 GB = 1e9 B) unless it says GiB.

**Status of this work: READ-ONLY.** Nothing on Thor was deleted, moved, truncated or modified (checks used `ls/du/find/stat/md5sum/sha256sum/tar --compare/python read-only scans`; no remote file was created). The two scripts next to this file were **written but not run**; they are meant to be reviewed and run by the PI on Thor.

Deliverables (all in `Project Steering/Ops/disk_cleanup_2026-10-04/`): `cleanup_thor_A.sh`, `cleanup_thor_B.sh`, `B_sha256_sam3map_corpus_out.txt`, `B_sha256_sam3_corpus_gt.txt` (the B script needs the two manifests beside it), this `INVENTORY_THOR.md`. Nothing is staged or committed.

## 0. Bottom line

| category | meaning | entries | reclaimable | free after (cumulative) |
|---|---|---:|---:|---:|
| now | | | | 62.3 GB |
| **A** | safe / re-creatable: git-identical code trees + tars, smoke checkpoints, caches | 221 paths | **90.5 GB** actually freed (the 221 lines add up to 91.7 GB: hardlinked twin trees are counted twice) | **152.8 GB** |
| **B** | redundant copy, hash-proven, surviving copy named | 13 entries (11 single paths + 14,006 manifest files) | **36.7 GB** | **189.5 GB** |
| **C** | probably obsolete, PI decides (listed in section 6, no script) | 29 entries | **310.2 GB** | **499.7 GB** |
| **D** | keep | see section 3 | 0 | |

A+B alone take the disk from 94 % full to about 80 % full by df's definition (free 189 GB of 954 GB). **One decision dominates C: `data/physicalai-b1-w120-256x640cyl` = 178.0 GB (166 GiB)** -- the older 256x640 B1 cache, which the refcv7 line does not read.

Sizes are allocated bytes (`du -B1`) except the two manifest-driven B rows (apparent bytes of 14,006 files, within 1-2 % of allocated). Free-space arithmetic assumes the freed bytes add 1:1 to `Avail` (ext4 reserved blocks do not change).

### The 10 biggest single entries (reclaimable side)

| # | cat | size | entry |
|---:|---|---:|---|
| 1 | C | 178.03 GB | C01 `data/physicalai-b1-w120-256x640cyl` -- B1 4,713-episode PNG v2ep cache at 256x640 cylindrical (built 2026-08-30..09-07) |
| 2 | C | 24.05 GB | C02 `sam3map` -- SAM3 map experiments: ~455 variant dirs (<seq>_v6*/spd*/prod*, ~17 GB) + data/ (5.1 GB) + native7 (1.5 GB) + corpus remainder |
| 3 | C | 18.15 GB | C15 `tanit-astra-research` -- TanitAstra research project (touched 2026-10-03 19:13): .venv 4.9 GB, models (sam3 3.3 GB, dinov3_l 1.2 GB), data (navsim maps/pilot 4.5 GB), experiments, backups |
| 4 | B | 14.37 GB | `sam3map/corpus/out/{*.sam3mapgt.npz,*.worldmap.npz}` -- HF-identical SAM3 map products (sha256 = publisher ledger) |
| 5 | C | 13.45 GB | C14 `qwendrive` -- Qwen-Drive-1.0-4B model (9.2 GB) + v2 sequence outputs (3.3 GB) |
| 6 | C | 12.54 GB | C11 `wpd_a3` -- WPD A3 token banks tok_D0/D0b/D0c/D1.npy (2.98 GB each) + run |
| 7 | C | 12.36 GB | C12 `percprobe` -- perception-probe banks (bank 9.8 GB, smokebank 1.5 GB, raw 0.45 GB; 2026-09-06) |
| 8 | B | 9.98 GB | `data/sam3_corpus/semantic_maps/gt/*.sam3mapgt.npz` -- HF-identical SAM3 map products (sha256 = publisher ledger) |
| 9 | C | 7.79 GB | C13 `v7tiny` -- v7-tiny rig runs (o1ctrl30k 1.7 GB + ~25 x 130 MB; 2026-08-22..09-05) |
| 10 | C | 6.79 GB | C20 `bx_0252/full/shards` -- vis1 pre-computation shards (96 j2d_*.jsonl, 6.5 GB) from which vis1_sidecar_refcv6b1_train4369_eval139.npz was built |

Biggest keep: `data/physicalai-b1-w120-416x1024cyl` = 393 GB (the refcv7 corpus). Biggest A *groups*: 107 code trees = 42.4 GB, 50 tars = 19.4 GB, 22 smoke/scratch checkpoints = 22.8 GB.

## 1. Where the 892 GB are

| area | size | note |
|---|---:|---|
| `/home/nvidia/data` | 585,143 MiB = 614 GB | two B1 caches (416x1024: 393 GB keep; 256x640: 178 GB C), sam3_gt_v3 20 GB keep, sam3 pulls 10 GB (B), obstacle 4.3 GB, ... |
| `/home/nvidia/sam3map` | 38,489 MiB = 40 GB | SAM3 map production + ~455 experiment dirs |
| `/home/nvidia/refcv6_run` | 21,343 MiB = 22 GB | 11 smoke ckpts (A), refcv6 ckpts (B), 5+5 code trees (A) |
| `/home/nvidia/tanit-astra-research` | 17,309 MiB = 18 GB | active sibling project (C) |
| `/home/nvidia/{experiments,qwendrive,wpd_a3,percprobe,v7tiny,...}` | ~60 GB | finished experiments (C) |
| 63 gate-family dirs (`gate_*`, `bx_*`, `nb2*`, `refcv7_gate_*`, `refcv7_bind*`, `wgate_0505`, `bha_2325`, `gmo_*`) | ~72 GB (independent sizes; `du` single-pass undercounts hardlinked twins) | 2026-09-26..28 launch-gate development copies (A), plus `bx_0252` (7.6 GB, mostly C20 shards) |
| `/home/nvidia/venvs` | 10,999 MiB = 12 GB | keep |
| `/home/nvidia/.cache` | 5,984 MiB = 6 GB | uv 1.7 G + pip 0.36 G (A), HF hub 3.8 G (sam3 3.3 G is B, timm resnet101 keep) |
| `/home/nvidia/refcv7_run` | 8,017 MiB = 8.4 GB | run 2.7 G keep; 7 code trees + 7 tars = 5.2 G (A) |
| `/` outside `/home` | 21.3 GB | `/usr` 15 G, `/var` 3.4 G (journal 0.62, snapd 2.1), `/opt` 0.28, `/tmp` 1.5 (A) -- docker is installed but EMPTY (0 images/volumes), no core dumps |

## 2. Method and evidence classes

Everything below is **MEASURED (ours, 2026-10-04)** unless tagged otherwise.

* **Code trees (A).** 159 directories containing `stack/` were walked on Thor (353,038 files). Every regular file <= 20 MB was hashed as a git blob (sha1 of `blob <n>\0<bytes>`), 5,949 unique hashes, and checked for membership in `git rev-list --objects origin/agent/arch-inf-20260803` of `C:/Users/Admin/tanitad-push` (47,860 objects). `git ls-remote origin` returned `a9b981dd` = the local tracking ref, so "on origin" is a live fact. Result: **73 of the 107 heavy trees are 100 % git blobs; 11 are 100 % git except overlay files that stay in the sibling `overlay/`; 23 carry 1-14 WIP files (<= 1.4 MB per tree) that are NOT in git -> the A script tars exactly those files into `~/_cleanup_salvage_2026-10-04/` before it removes the tree (and refuses to remove the tree if the salvage did not verify).** Worktree-style copies (`refcv6_run/pre_*`, `35be81285e`, `4ca6168822`, `bx_0412/tree`, `gmo_early_0327/code`) were CRLF checkouts; compared after CRLF->LF normalisation they match git (the residue counted is after normalisation).
* **Tars (A).** Each `git archive` tar carries its commit id in the pax header: 33 distinct ids, **all 33 are ancestors of `origin/agent/arch-inf-20260803`**. The 7 `refcv7_run/*.tar` were compared with their extracted dirs (`tar --compare`: 0 content/size/missing differences, only uid/gid). Tars without a pax id (`nb2r2/3/5 cand_*.tar`, `nb2_gate_4797ffb/cand_4797.tar`, `cand_r4.tar`, `cand_b4a5.tar`) were stream-hashed member by member against git: 0 non-git members except the salvaged WIP files. `gate_r4_0928/gate_r4.tar` is a wrapper of exactly its three sibling files (sizes equal).
* **Checkpoints (B).** md5 (refcv6) / sha256 (others) computed on Thor and compared with (a) the dev-box copy, whose md5 was also computed now, or (b) the Hugging Face LFS sha256 read through the Hub API (read-only, token read in place, never printed). The refcv7 checkpoints on Thor were also md5-compared with D: (see section 7).
* **SAM3 map products (B).** `sam3map/corpus/publish/published.jsonl` is the publisher ledger (sha256 + bytes per published file, 'sha256 verified on the Hub'). All 9,354 + 4,652 local files were re-hashed on Thor against it: 0 mismatch, 0 missing. A live Hub check of 120 random clips (240 files) matched sha256+bytes 240/240 (2 files had moved `gt_flagged -> gt` in the S4 re-tier and match at the new path). Both manifests were also replayed on Thor with `sha256sum -c --quiet < manifest` (read-only): rc=0, no output.
* **Absence claims** (e.g. 'no HF/D: copy of the refav1 checkpoints') are exact-byte-size matches only: Hub files >= 50 MB of the account Sayood (256 files) vs Thor files >= 100 MB (485 files), and `find -size` over `D:/Projects, D:/Archive, D:/TanitAstra*, D:/thor_archive, D:/refcv6_eval_kit, D:/refcv7_eval_kit, D:/tmp_fixtree`. Not a content search; treat 'no copy found' as 'not proven redundant'.
* **Hardlinks (checked because `du` gave two different sizes for the same dir).** 12,256 file entries (1.6 GB apparent) under `~` are hardlinked, mostly between twin trees (`nb2a19_thor/code` <-> `nb2r5_2ac0/code`, 3,058 files). Freed space was therefore computed **per inode**: every link of every inode inside the A set also lies inside the A set (0 inodes held by outside links), so A frees exactly **90.454 GB**, while the 221 per-line sizes add up to 91.672 GB; the same check for the 11 single B paths gives 12.355 GB (no outside links). The KEEP tree `refcv7_run/fec3a0dccf` has no hardlinks (0 files with nlink>1). Both scripts print `ACTUALLY FREED` from `df` at the end of a real run.
* No process of user nvidia was running (only the desktop/system), GPU idle, load 0.16. `thor_gpu.lock` exists (0 bytes) and was not touched; the scripts refuse to run for real while it is held.

## 3. D -- KEEP (the scripts refuse to touch any of these)

The refcv7 argv (`runs/refcv7-r101-s0/config.json`, 154 tokens) and the PASS token (`refcv7_gate_fec3a0dccf/PASS_fec3a0dccfda.json`) name these paths; all exist.

| path (under /home/nvidia) | size | why |
|---|---:|---|
| `data/physicalai-b1-w120-416x1024cyl (+ symlink views refcv6-b1-416x1024-train / -eval139)` | 392.97 GB | the refcv7 training + eval corpus named in argv (--v2-cache, --eval-cache); 4,713 v2ep.pt + MANIFEST; built from HF Sayood/tanitad-v7-training-corpus @a0cf20df (rebuild is a multi-hour job) |
| `data/sam3_gt_v3` | 19.99 GB | --map-gt-root of the refcv7 run (EXPORT_OK verdict PASS, 4,506 files, 19.98 GB); derived export, no HF copy |
| `venvs/tanitad-train, venvs/tanitad-edge` | 11.53 GB | the interpreters of the run (argv/gate token: venvs/tanitad-train/bin/python); rebuilding risks the torch/driver trap |
| `refcv7_run/runs (refcv7-r101-s0: ckpt.pt final + ckpt_{5000,15000,20000,30000}.pt, config, metrics, summary)` | 2.82 GB | the finished run; all 5 ckpts md5-identical to the D: copies (MEASURED, see B notes) |
| `refcv7_run/fec3a0dccf (+ runs.d, sup_refcv7-r101-s0.out)` | 0.41 GB | the launch code tree (commit fec3a0dccfda, byte-identical to git) -- needed for a BRANCH run |
| `refcv7_post (diag + video; video/bank/final 1.38 GB is listed under C22)` | 0.72 GB | tonight's diagnostics and video banks (PI instruction: keep) |
| `refcv7_gate_fec3a0dccf (PASS_fec3a0dccfda.json, final_verdict, evidence/, suite_bound, tree_manifest ...)` | 0.03 GB | launch-gate PASS token + evidence = provenance (its 1.17 GB smoke ckpt is listed under C23) |
| `refcv7_bind/run_map, refcv7_bind/run_box, refcv7_dry_thor/q4c_grid_vs_egolog_ALLTRAIN.json, bx_0252/calib` | 0.04 GB | gate-input records that the PASS token names by path (g_map_overfit.json, gbo_binding.json, closures, clock reference) + vis1 calib; the heavy code trees beside them go in A |
| `data/{joins, join3d, refcv7, anchors, v8labels, v72, physicalai-b1-EVAL6-w120-256x640cyl} + sidecar/extrinsics/clock files + the two symlink views` | 1.62 GB | every other path named in the run's argv (agent joins 0.33 + 0.82 GB, vis1 sidecar, tau/class-weight json, anchors, v8 labels, speed-max sidecars, clip-clock sidecar) |
| `.cache/huggingface/hub/models--timm--resnet101.a1_in1k` | 0.18 GB | the pretrained resnet101.a1_in1k trunk weights (argv --trunk-name); a branch run may re-load them from this cache |
| `data/physicalai-b1 (r0, calibration, clip_index), data/b1-bundle (rest), refcv6 (rest), refcv6_v2` | 0.56 GB | small inputs / build provenance |

**Surprise worth acting on: the PASS token is not self-contained.** It references gate-input records outside its own directory: `refcv7_bind/run_map/{out/g_map_overfit.json,gmo_closure.json}`, `refcv7_bind/run_box/{gbo_binding.json,gbo_closure.json}` and `refcv7_dry_thor/q4c_grid_vs_egolog_ALLTRAIN.json`. Deleting the directories `refcv7_bind/` or `refcv7_dry_thor/` wholesale would orphan the provenance of the launch. The scripts delete only the heavy `tree*/` subdirectories there.

## 4. A -- safe / re-creatable (90.5 GB, 221 paths) -> `cleanup_thor_A.sh`

The group sizes below are the per-line sizes (sum 91.67 GB); the space really freed is 90.45 GB.

| group | paths | size | rule |
|---|---:|---:|---|
| caches / temp | 40 | 6.21 GB | `~/.cache/uv` 1.7 G, `~/.cache/pip`, `~/.nv`, `/tmp/torchinductor_nvidia`, 34 `/tmp/v4smoke_*`, `data/_measure_tmp`, `sam3map/corpus/scratch` |
| smoke / profile / gate-scratch checkpoints | 22 | 22.82 GB | only the big `ckpt.pt` (or `g_eval_trainer_state.pt`/`mt.pt`) is removed, config/metrics/logs beside it stay |
| gate dev-iteration eval output | 2 | 0.81 GB | `arms_r6_eval` of `gate_lg_0415` / `gate_lg_2317` |
| code tars | 50 | 19.45 GB | pax commit on origin, or stream-hashed against git |
| extracted code trees | 107 | 42.38 GB | blob-identical to git (see section 2); 23 with salvage |

Trees + tars by family:

| family | paths | size |
|---|---:|---:|
| gate_* | 87 | 36.18 GB |
| bx_* | 23 | 8.81 GB |
| refcv7_run + refcv7_bind/dry/probe | 18 | 7.19 GB |
| nb2* | 12 | 4.84 GB |
| refcv6_run | 10 | 2.01 GB |
| gmo_* | 4 | 1.61 GB |
| bha_2325 | 2 | 0.79 GB |
| wgate_0505 | 1 | 0.40 GB |

What stays in those directories (so nothing evidential is lost): `overlay/` (the candidate files), `results/`, `evidence/`, `*.json` verdicts, `plan_thor.json`, logs, `md5*.txt`, `FAIL_*.json` of the superseded gate attempts, the PASS dir in full. MEASURED: the 62 top-level gate-family dirs other than `bx_0252` hold 64.4 GB today and 1.7 GB after A, of which 1.17 GB is the PASS gate's own smoke ckpt (C23) -- the evidence stubs are about 0.5 GB in total.

`refcv7_run`: the six superseded trees `02ffcebfc3, 0fcb1c1cbb, 14907a0a5a, 53ecf9b806, 60f4c068a8, 6fa5e8bba4` (dir + tar each) and `fec3a0dccf.tar` are A. `0fcb1c1cbb` was the R4-restart candidate and was **never launched** (run log, `sup_refcv7`, `train.log` contain only `fec3a0dccf`). `fec3a0dccf/` itself is KEEP.

`.nv` and `/tmp/torchinductor_nvidia` are compile caches: the next run re-JITs (a few minutes with `--trunk-compile`).

## 5. B -- redundant copies, hash-proven (36.7 GB, 13 entries) -> `cleanup_thor_B.sh`

The B script **re-hashes every file on Thor immediately before removing it** (`VERIFY=1`, also in the dry run; reads ~36 GB, nice/ionice) and deletes nothing on a mismatch. The two manifest rows use `sha256sum -c` over the whole list and delete only if all entries verify.

| size | what (path under /home/nvidia) | proof | surviving copy |
|---:|---|---|---|
| 0.40 GB | `refcv6_run/runs/refcv6-r101-s0/ckpt_5000.pt` | md5 `8a1e4da0dc656135..` -- refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04) | D:/refcv6_eval_kit/ckpt/ckpt_5000.pt |
| 0.40 GB | `refcv6_run/runs/refcv6-r101-s0/ckpt_15000.pt` | md5 `46eb21857234a9dc..` -- refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04) | D:/refcv6_eval_kit/ckpt/ckpt_15000.pt |
| 0.40 GB | `refcv6_run/runs/refcv6-r101-s0/ckpt_20000.pt` | md5 `f4320c8971dcc88b..` -- refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04) | D:/refcv6_eval_kit/ckpt/ckpt_20000.pt |
| 0.40 GB | `refcv6_run/runs/refcv6-r101-s0/ckpt_30000.pt` | md5 `0c5c67b3ecf28495..` -- refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04) | D:/refcv6_eval_kit/ckpt/ckpt_30000.pt |
| 1.18 GB | `refcv6_run/runs/refcv6-r101-s0/ckpt.pt` | md5 `5a2e7222a9f5f8c7..` -- refcv6 rolling ckpt at the PI stop (step 38000); Thor md5 == D: md5 == STOPPED_BY_PI.json; Thor sha256 de34332d.. == HF LFS sha256 | D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt + D:/refcv6_eval_kit/ckpt_final/ckpt_step38000.pt + HF Sayood/tanitad-refcv6-r101-s0/ckpt_step38000.pt |
| 1.18 GB | `refcv6_run/snapshots/refcv6-r101-s0/ckpt_step35000.pt` | md5 `68a4ef3bbb0a3ec7..` -- refcv6 step-35000 rolling snapshot; Thor md5 == D: md5 == snapshot MD5SUMS | D:/refcv6_eval_kit/ckpt/ckpt_35000.pt |
| 1.29 GB | `refcv4b/ckpt_40284_FINAL.pt` | sha256 `090ee8d2584acc81..` -- refcv4b FINAL ckpt; Thor sha256 == HF LFS sha256 (private repo) | HF Sayood/tanitad-refc-v4b/ckpt_40284_FINAL.pt |
| 1.25 GB | `tmp/refc-base-30k/ckpt.pt` | sha256 `6b7002c42fab31eb..` -- refc-base 30k ckpt; Thor sha256 == HF LFS sha256 (public repo) | HF Sayood/tanitad-refc-base/ckpt.pt |
| 0.43 GB | `models/refcv3-b1-v72-40284/ckpt.pt` | sha256 `5a163eaf57674e52..` -- refcv3 ckpt; Thor sha256 == HF LFS sha256 | HF Sayood/tanitad-refc-v3/ckpt_40284.pt |
| 1.97 GB | `data/b1-bundle/egomotion/egomotion_alpamayo.tar` | sha256 `d38592bed40cec2a..` -- B1 egomotion bundle tar; Thor sha256 == HF LFS sha256 (private corpus repo) | HF Sayood/tanitad-v7-training-corpus:egomotion/egomotion_alpamayo.tar |
| 3.45 GB | `.cache/huggingface/hub/models--facebook--sam3` | sha256-of-snapshot-file `9999e2341ceef5e1..` -- HF-cache copy of facebook/sam3 (3.45 GB = 3.29 GiB); its sam3.pt sha256 9999e234.. is IDENTICAL to tanit-astra-research/models/sam3/sam3.pt | /home/nvidia/tanit-astra-research/models/sam3/sam3.pt (same sha256) and the gated HF repo facebook/sam3 |
| 14.37 GB | `sam3map/corpus/out/{*.sam3mapgt.npz,*.worldmap.npz}` (9,354 files) | SAM3 map products of 4,677 clips (gt 4,569 + gt_flagged 108 + worldmap 4,677 = 9,354 files, 14.37 GB): every file's sha256 equals the publisher ledger entry (MEASURED on Thor 2026-10-04: 9,354 ok / 0 bad / 0 missing); the ledger records 'sha256 verified on the Hub'; live Hub sample 240/240 sha256+bytes match (2 had been re-tiered gt_flagged -> gt) | HF dataset Sayood/tanitad-v7-training-corpus (private): semantic_maps/{gt,gt_flagged,worldmap}/ |
| 9.98 GB | `data/sam3_corpus/semantic_maps/gt/*.sam3mapgt.npz` (4,652 files) | verbatim HF pull (_pull_record.json: revision 355f4adf, 4,652 files); all 4,652 sha256 match the publisher ledger (MEASURED on Thor 2026-10-04). It is the input of the sam3_gt_v3 export, which is KEPT | HF dataset Sayood/tanitad-v7-training-corpus @355f4adf (private): semantic_maps/gt/ |

Consequences to know before running B: **`D:` is 100 % full (7.1 GB free) and `C:` 99 % (14 GB free)**, so after B the `D:` file is the only *local* copy of the refcv6 checkpoints (HF `Sayood/tanitad-refcv6-r101-s0` holds `ckpt_step38000.pt`, 1.18 GB, sha256 `de34332d..` = the Thor file; the milestone ckpts 5k/15k/20k/30k and the step-35k snapshot exist only on `D:`). refcv6 was STOPPED by the PI on 2026-09-26; a branch run would copy a ckpt back to Thor over the LAN at ~63 MB/s (~20 s per GB).

HF cache `models--facebook--sam3`: the only reason it is B and not A is that `tanit-astra-research/models/sam3/sam3.pt` (same sha256 `9999e234..`) must stay -- if you ever delete `tanit-astra-research` (C15), keep one sam3.pt.

## 6. C -- probably obsolete, PI decides (310.2 GB over 29 entries, **no script**)

Ordered by size. 'Risk' says what you lose if you are wrong. Sizes of nested entries are not double counted except C22/C23 (inside KEEP dirs, listed because the PI asked / for completeness).

| id | size | path (under /home/nvidia) | what | evidence / risk |
|---|---:|---|---|---|
| C01 | 178.03 GB | `data/physicalai-b1-w120-256x640cyl` | B1 4,713-episode PNG v2ep cache at 256x640 cylindrical (built 2026-08-30..09-07) | Used by the v7-tiny / refav1 / refcv3-5 lines, NOT by refcv7 (its argv names only the 416x1024 cache). Rebuildable from HF Sayood/tanitad-v7-training-corpus camera mp4s (3.4-4.6 h, PNG encode is 70 %). HF Sayood/tanitad-physicalai-w120-256x640cyl holds the PARITY caches (2,400+600), NOT this B1 4,713 cache (GOALS_AND_CLAIMS D-REFCV3-CORPUS, 2026-09-02). The refav1-fp8-* cache views already dangle (dinov3-b1-fp8-w120-256x640cyl is gone). |
| C02 | 24.05 GB | `sam3map` | SAM3 map experiments: ~455 variant dirs (<seq>_v6*/spd*/prod*, ~17 GB) + data/ (5.1 GB) + native7 (1.5 GB) + corpus remainder | Not in D:/thor_archive (the 217 dirs archived there were different ones, already removed from Thor). The production products are on HF (see B). Variants are experiment outputs of 2026-09-13..16. |
| C15 | 18.15 GB | `tanit-astra-research` | TanitAstra research project (touched 2026-10-03 19:13): .venv 4.9 GB, models (sam3 3.3 GB, dinov3_l 1.2 GB), data (navsim maps/pilot 4.5 GB), experiments, backups | ACTIVE sibling project; its owner decides. Holds the surviving sam3.pt used by the B item. |
| C14 | 13.45 GB | `qwendrive` | Qwen-Drive-1.0-4B model (9.2 GB) + v2 sequence outputs (3.3 GB) | Model is an HF download; v2 outputs are derived. |
| C11 | 12.54 GB | `wpd_a3` | WPD A3 token banks tok_D0/D0b/D0c/D1.npy (2.98 GB each) + run | Derived banks (2026-09-09); rebuild needs wpd_trunks + the 256x640 cache. |
| C12 | 12.36 GB | `percprobe` | perception-probe banks (bank 9.8 GB, smokebank 1.5 GB, raw 0.45 GB; 2026-09-06) | Derived banks; GOALS_AND_CLAIMS cites percprobe. |
| C13 | 7.79 GB | `v7tiny` | v7-tiny rig runs (o1ctrl30k 1.7 GB + ~25 x 130 MB; 2026-08-22..09-05) | GOALS_AND_CLAIMS cites o1ctrl30k and the seed-floor replicate; needs the 256x640 cache to re-run. |
| C20 | 6.79 GB | `bx_0252/full/shards` | vis1 pre-computation shards (96 j2d_*.jsonl, 6.5 GB) from which vis1_sidecar_refcv6b1_train4369_eval139.npz was built | The sidecar (186 MB) is KEPT in data/refcv7; rebuild = 44 min wall (log: 'wall 2648.7 s'). |
| C24 | 4.32 GB | `data/obstacle` | obstacle.offline + egomotion_alpamayo per-clip parquet inputs (4.1 GB) of the agent joins | The joins (KEPT) are built from it; rebuild = pod-side join step. |
| C03 | 4.25 GB | `experiments/refav1-b1-v72-ep2-ema-bf16` | refav1 ep2 ema-bf16 arm (ckpt.pt + ckpt_step1000.pt, 2.12 GB each) | No byte-size match on HF or in the D: dirs searched -> possibly the ONLY copy. |
| C16 | 3.45 GB | `TanitAD` | Thor-side repo clone (HEAD 30d6d601 of 2026-08-15, branch agent/arch-inf-20260803, 92 modified/untracked files) incl. the old 'TanitAD Research Hub' dir | 92 uncommitted files = possible unique WIP; cache MANIFESTs record /home/nvidia/TanitAD/stack/... as digest_source for future cache builds. Bundle git diff + untracked files to the dev box first. |
| C09 | 2.15 GB | `refav1_lon` | refav1 longitudinal arm: ckpt.pt 2.12 GB + code | No HF/D: copy found by size. |
| C04 | 2.12 GB | `experiments/refav1-b1-v72-ep3-speed` | refav1 v1 arm (MODEL_REGISTRY 2.4 names this Thor path as the artifact) | Registry-named location; no HF/D: copy found by size. |
| C05 | 2.09 GB | `experiments/refav1-b1-v72-1ep-21109` | refav1 1-epoch arm ckpt (2.09 GB) | No HF/D: copy found by size. |
| C07 | 2.09 GB | `experiments/TSC-B-frozen` | TSC-B frozen arm (2.09 GB) | No HF/D: copy found by size. |
| C06 | 2.08 GB | `experiments/TSC-A-adapter` | TSC-A adapter arm (2.08 GB) | No HF/D: copy found by size. |
| C26 | 2.02 GB | `data/_b1stage416` | 416x1024 build staging (egomotion_all.zip 1.96 GB, r0, dl, logs) | Build finished 2026-09-23 (cache MANIFEST present); small provenance files inside. |
| C25 | 1.96 GB | `data/physicalai-b1/labels/egomotion` | 1,411 upstream egomotion chunk zips (1.87 GB) | Public PhysicalAI-AV chunks; re-downloadable. |
| C10 | 1.81 GB | `wpd_trunks` | 5 WPD trunk checkpoints (362 MB each) | No HF/D: copy found by size. |
| C22 | 1.38 GB | `refcv7_post/video/bank/final` | refcv7 video bank (1.32 GB; single copy), re-derivable by the video renderer | Inside the KEEP refcv7_post tree; listed C as instructed. |
| C08 | 1.30 GB | `experiments/wpd-D0-4k` | WPD D0 4k arm (1.30 GB ckpt) | No HF/D: copy found by size. |
| C23 | 1.17 GB | `refcv7_gate_fec3a0dccf/smoke/20260927T211913/ckpt.pt` | G-LIVE smoke ckpt of the PASSING gate (1.17 GB); metrics/summary beside it | The dir holds the PASS token (KEEP); this single file has no provenance role. |
| C18 | 1.17 GB | `refcv6_run/snapshots/refcv6-r101-s0/ckpt_step10000.pt` | refcv6 rolling snapshot @10000 (1.17 GB, with optimizer) | Not on D: (only model-only 5000/15000/20000/30000 are). |
| C19 | 1.17 GB | `refcv6_run/snapshots/refcv6-r101-s0/ckpt_step25000.pt` | refcv6 rolling snapshot @25000 (1.17 GB, with optimizer) | Not on D:. |
| C17 | 0.67 GB | `v6F_snap_fp16.pt` | v6F fp16 snapshot (2026-08-16, 673 MB) | D:/thor_archive/2026-09-14/ckpt_snaps holds other v6F_sw_step*.fp16.pt; no size match -> maybe the only copy. |
| C29 | 0.54 GB | `.rustup` | Rust toolchain (520 MB; .cargo another 51 MB) | Needed only to build Rust wheels. |
| C27 | 0.52 GB | `trt_c3` | TensorRT plan + onnx (497 MB, 2026-08-03) | Edge/deploy experiment. |
| C28 | 0.45 GB | `nurec-gsplat` | NuRec/gsplat experiments (425 MB, 2026-08-03) | AlpaSim/NuRec asset -> PI decides (never A). |
| C21 | 0.30 GB | `bx_0252/control/shards` | vis1 control-run shards (288 MB) | Control run of the same builder. |

Suggested order if you want space without regret: (1) **C01** (178 GB) after confirming no 256x640 line (v7-tiny / refav1 / refcv3-5 re-runs) is planned -- it is rebuildable from HF in 3.4-4.6 h, and its fp8 views are already dangling; (2) **C02** the 455 SAM3 variant dirs (the products are on HF); (3) C11/C12/C13/C20 derived banks and rig runs; (4) C14 `qwendrive/Qwen-Drive-1.0-4B` (re-download); (5) the checkpoints C03-C10/C17 only after copying them somewhere that is not a full `D:` (HF private is the cheap way: ~19 GB against the 1 TB Pro quota, of which 183 GB private was in use on 2026-09-22 per the publisher log; **check the quota rule first**); (6) leave C15 to its owner, C28 NuRec/AlpaSim assets untouched.

## 7. Things that surprised me

1. **107 code trees and 50 tars (60.6 GB) are development copies of a few dozen git commits** (33 distinct commit ids in the tar headers) -- the gate ran on fresh extracts every time. 73 trees are byte-identical to git, so this is the cheapest 60 GB on the box.
2. **The PASS token depends on files outside its directory** (section 3); a tidy-minded `rm -rf refcv7_bind refcv7_dry_thor` would have broken provenance.
3. **The refcv7 checkpoints are NOT only on Thor**: all five (`ckpt.pt` final = `ckpt_50400_full.pt`, 5000, 15000, 20000, 30000) are md5-identical to `D:/refcv7_eval_kit/ckpt/` (actual md5 recomputed on both sides today: `d5f104ee..`, `06eb9dfd..`, `2e7fd26c..`, `4c0a78e7..`, `ac4e4fab..`). HF `Sayood/tanitad-refcv7-r101-s0` holds only model-only 5000/20000/30000/50400 (no 15000, **no optimizer state**). So the full rolling ckpt (opt + data_pos, needed for a true resume) exists on Thor and `D:` only.
4. **`D:` and `C:` are full** -- the dev box cannot absorb anything from C, and B makes `D:` the sole local holder of the refcv6 milestones.
5. **SAM3 products exist three times on Thor** (`corpus/out` 14.4 GB, `data/sam3_corpus` 10.0 GB verbatim HF pull, `sam3_gt_v3` 20 GB derived export) plus once on HF; the first two are verbatim duplicates of HF (B). A re-export of `sam3_gt_v3` needs the HF pull again (or `sam3_corpus/semantic_maps/gt`), 948 s measured in `pull_gt.log`.
6. **The 482 `archive_manifest_*.md5/.links` files in `~` (5.1 MB) are stale**: they describe the 2026-09-14 archive to `D:/thor_archive/2026-09-14` (217 sam3map dirs, models, ckpt_snaps, nurec, alpasim, ...), all of which were already removed from Thor. None of the 461 dirs now in `sam3map/` is in that archive, and none of the current C/B items is covered by those manifests -- do not use them as a deletion proof.
7. **`refav1-fp8-train/eval` cache views dangle** (`dinov3-b1-fp8-w120-256x640cyl` no longer exists): the refav1 line cannot be re-run on this box without a rebuild regardless of C01.
8. **The Thor repo clone (`~/TanitAD`, 3.4 GB) is 7 weeks old with 92 uncommitted files**; the two venvs do not point at it (no editable install; I did not grep scripts for references), and the cache MANIFESTs record its `stack/` as `digest_source`.
9. **Experiment checkpoints of the refav1 / TSC / WPD arms (~19 GB: C03-C10, C17) have no Hub or `D:` match by size**; `MODEL_REGISTRY.md` 2.4 names the Thor path as the artifact location of refav1 v1. They may be the only copies.
10. **/tmp holds 10,243 entries / 1.48 GB**: 34 `v4smoke_*` dirs (0.95 GB) and the inductor cache (0.37 GB) are the bulk; docker is installed but empty; no core dumps; journald 623 MB.

## 8. Optional system-level cleanup (needs sudo, NOT in the scripts) -- about 1.4 GB

| size | action | note |
|---:|---|---|
| ~0.52 GB | `sudo journalctl --vacuum-size=100M` | journal is 623 MB |
| 0.78 GB | `sudo snap remove core24 --revision=1644; sudo snap remove firefox --revision=8968; sudo snap remove mesa-2404 --revision=1166; sudo snap remove snapd --revision=27740; sudo snap remove thunderbird --revision=1261` | the disabled old revisions (`snap list --all`); current revisions untouched |
| 0.12 GB | `sudo apt-get clean` | `/var/cache/apt` |
| -- | leave `/opt/ota_package` (266 MB) | Jetson OTA payload |

## 9. Runbook

```bash
# on the dev box
cd "/d/Projects/TanitAD/Project Steering/Ops/disk_cleanup_2026-10-04"
ssh -n tanitad-thor-wifi 'mkdir -p /home/nvidia/cleanup_2026-10-04'
scp cleanup_thor_A.sh cleanup_thor_B.sh B_sha256_sam3map_corpus_out.txt B_sha256_sam3_corpus_gt.txt tanitad-thor-wifi:/home/nvidia/cleanup_2026-10-04/
# on Thor (user nvidia)
cd /home/nvidia/cleanup_2026-10-04
bash cleanup_thor_A.sh        | tee A_dry.log      # DRY RUN: read it; sum of per-line sizes ~ 91.7 GB (really freed 90.5 GB), refused-by-guard = 0
DRYRUN=0 bash cleanup_thor_A.sh | tee A_real.log
bash cleanup_thor_B.sh        | tee B_dry.log      # re-hashes ~36 GB; every 'verified' line must appear, no HASH-MISMATCH
DRYRUN=0 bash cleanup_thor_B.sh | tee B_real.log
df -h /      # expect ~189 GB free after A+B
```

* Both scripts are `DRYRUN=1` by default, print `WOULD-RM <size> <path>` per line and a total, and refuse (guard) any path inside or containing a KEEP entry, outside `/home/nvidia` + `/tmp`, or with glob characters / `..`. A real run also refuses while `thor_gpu.lock` is held or a python job of user nvidia is alive (`FORCE=1` overrides the second check only).
* Run A before B; they are independent. Nothing in them touches `/var`, `/usr`, snap or journald.
* After the real runs `~/_cleanup_salvage_2026-10-04/*.tgz` (23 small tarballs of 119 WIP files, 14.5 MB uncompressed) is the only new content on Thor. Copy it to the dev box if you want it as an artifact.
* **Sandbox-tested only**: the guard (KEEP, parent-of-KEEP, outside-root, glob, `..`), `rm_rf`, `rm_if_hash` (wrong hash refuses), `rm_dir_if_same_file` (differing sha256 refuses) were exercised in a throw-away directory on the dev box; `rm_manifest`'s two ingredients were tested separately -- the line-shape regex (0 non-matching lines in both manifests) and `sha256sum -c` replayed on Thor (rc=0) -- but the function as a whole could not be sandboxed (it only accepts the real `/home/nvidia/...` paths). The scripts themselves were **never executed against Thor**.

## 10. Not verified / limits

* The Hub check of the SAM3 products is a 240-file live sample; the other 9,114 rest on the publisher ledger plus the local re-hash (INHERITED for the Hub side).
* `tar --compare` covers tar members against the dir, not extra files in the dir (e.g. `fec3a0dccf/` holds 109 more files than its tar -- `__pycache__`; they go with the tree and are regenerated on import).
* 'Origin-reachable' means reachable from `origin/agent/arch-inf-20260803` (a9b981dd); a blob only on `main` or another branch would count as 'non-git' (conservative).
* C-entry sizes are `du -B1` taken in one pass; entries nested inside others were taken first, so a few parents (`refcv7_post`, `data/physicalai-b1`, ...) are listed in section 3 with hand-corrected sizes.
* I did not look inside `venvs`, `tanit-astra-research/.venv` or the ~100 home directories below 100 MB beyond their names/sizes.
* The home reconciliation is informational: A 90.5 GB + B 36.7 GB + C 310.2 GB + D 430.9 GB = 868.2 GB of `/home/nvidia` (870.6 GB); the remaining 2.4 GB are the gate evidence stubs (~0.5 GB), the hardlink double count of the A lines (1.2 GB) and ~100 small dirs.

## Appendix A -- every A path (script order)

| group | size | path (under /home/nvidia unless absolute) | proof / note |
|---|---:|---|---|
| cache | 0.371 GB | `.cache/pip` | pip http cache |
| cache | 1.815 GB | `.cache/uv` | uv package cache; MEASURED no inode in it is hardlinked from outside (venvs keep their own copies); re-downloaded on demand |
| cache | 0.272 GB | `.nv` | NVIDIA JIT ComputeCache; kernels recompile on next use (minutes) |
| cache | 0.447 GB | `data/_measure_tmp` | 2026-08-30 measurement scratch: 12 loose 17-42 MB *.v2ep.pt sample episodes, no consumer (the real caches are physicalai-b1-w120-*) |
| cache | 1.939 GB | `sam3map/corpus/scratch` | SAM3 corpus production per-launch scratch (100 dirs of intermediate *.npz); run finished 2026-09-22 (DONE), outputs are in corpus/out and on HF |
| cache | 0.387 GB | `/tmp/torchinductor_nvidia` | torch inductor compile cache; recompiled on next --trunk-compile (a few minutes) |
| cache | 0.029 GB | `/tmp/v4smoke_0wvs52sm` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_105wfszz` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_16qah13r` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_1b02_69g` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_30ez15_2` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_4zhvalf9` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_54bvyz_g` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_5kg8xyfo` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_6pv5uq8h` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_6wdgg5s0` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_6wkqebvm` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_8b_gozb4` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_8kabur3m` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_9exjidvo` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_9oeb5jl1` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_ax56op1d` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_dvkviwk9` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_g00f5tlo` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_h5k_ym4n` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_huu6p_zv` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_jcewkq7y` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_kepq63z1` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_kmmri5li` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_o3pknx_t` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_p0fyakbc` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_pzoaobxt` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_s2_7bugz` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_v_cvopes` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_vc4owzr0` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_w9kx4j3g` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_wxh1czp1` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_y457bocr` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_z0hfvgny` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| cache | 0.029 GB | `/tmp/v4smoke_z_hc3iae` | /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27 |
| smoke | 1.300 GB | `refcv6/raw/smoke-V0/ckpt.pt` | refcv6-V0 smoke checkpoint (1.3 GB); config/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/prof/S8/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S10_b16_fold_dedup/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S11b_b16_fold_dedup_compile/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S3_b8_ce10_ck8/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S4_b8_ce10_ck8_bf16_cl/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S5_ck24/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S6_ck8_bench/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S7_b4_nockpt/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S8_ck8_bf16_cl_fold/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/S9_ck8_bf16_cl_fold_dedup/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/ck_b4_nocd/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 1.171 GB | `refcv6_run/smoke/prof_S3/ckpt.pt` | refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept |
| smoke | 0.403 GB | `refcv7_gate_02ffcebfc3/scratch/g_eval_trainer_state.pt` | gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it |
| smoke | 1.173 GB | `refcv7_gate_02ffcebfc3/smoke/20260927T194710/ckpt.pt` | launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept |
| smoke | 0.403 GB | `refcv7_gate_14907a0a5a/diag_scratch/mt.pt` | gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it |
| smoke | 1.173 GB | `refcv7_gate_14907a0a5a/smoke/20260927T203203/ckpt.pt` | launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept |
| smoke | 0.403 GB | `refcv7_gate_53ecf9b806/scratch/g_eval_trainer_state.pt` | gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it |
| smoke | 1.173 GB | `refcv7_gate_53ecf9b806/smoke/20260927T190203/ckpt.pt` | launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept |
| smoke | 0.403 GB | `refcv7_gate_6fa5e8bba4/scratch/g_eval_trainer_state.pt` | gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it |
| smoke | 1.173 GB | `refcv7_gate_6fa5e8bba4/smoke/20260927T180523/ckpt.pt` | launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept |
| smoke | 1.173 GB | `refcv7_probe/run/ckpt.pt` | refcv7 argv-probe smoke checkpoint; train.log/probe_result.json are kept |
| evalscratch | 0.405 GB | `gate_lg_0415/ev/arms_r6_eval` | launch-gate dev-iteration eval output (387 MB), regenerated by launch_gate.py; the PASS evidence is in refcv7_gate_fec3a0dccf (kept) |
| evalscratch | 0.405 GB | `gate_lg_2317/out/arms_r6_eval` | launch-gate dev-iteration eval output (387 MB), regenerated by launch_gate.py; the PASS evidence is in refcv7_gate_fec3a0dccf (kept) |
| tar | 0.393 GB | `bha_2325/code/bha_tree_a756e81.tar` | git-archive tar; pax header commit a756e81522ba is an ancestor of origin/agent/arch-inf-20260803 (it is the only content of bha_2325/code) |
| tar | 0.113 GB | `bx_0252/tip_68e8ec6_stack.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `bx_0412/tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `bx_0524/tip_cef9709.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `bx_1600/tip_2ac0bfb.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `bx_gate_1136/tip_2374cd2.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `bx_gate_1251/tip_1b8170e.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `gate_b2_0201/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `gate_b3_0258/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_bxa17_1626/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_bxa2_1252/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_bxa_1156/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_c3_2200/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_c4_2245/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_c5_2355/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_comb_1735/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_final_1855/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `gate_fix_2224/tree_2c510fb.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_glive_2110/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0353/t49f10b4.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0415/t3d6ff9a.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0504/tc178cb1.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0508/t3cca805.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0518/t3cca805.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0928/t3cca805.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0944/t3cca805.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lg_0951/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `gate_lg_2317/ab436ee.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_lgt_1033/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_nb2_0413/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_nb2r2_1037/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.395 GB | `gate_nb2r3_1149/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gate_nb2t_0502/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `gate_new1_2259/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.393 GB | `gate_new1r2_0048/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.397 GB | `gate_r4_0928/gate_r4.tar` | wrapper tar = exactly {tree_tip.tar, overlay.tar, plan_thor.json}; all three sit beside it with identical sizes |
| tar | 0.395 GB | `gate_r4_0928/tree_tip.tar` | git-archive tar; pax header commit is on origin/agent/arch-inf-20260803 |
| tar | 0.394 GB | `gmo_early_0327/cand_b4a5.tar` | candidate tar; 9 WIP members NOT in git; same 9 files salvaged from gmo_early_0327/code **[salvage]** |
| tar | 0.393 GB | `nb2_gate_4797ffb/cand_4797.tar` | candidate tar; stream-hashed: all 2,809 members are git blobs reachable from origin (0 non-git members) |
| tar | 0.396 GB | `nb2r2_cef9/cand_r2.tar` | candidate tar; stream-hashed: all members are git blobs reachable from origin (0 non-git members) |
| tar | 0.397 GB | `nb2r3_2374/cand_r3.tar` | candidate tar; stream-hashed: all members are git blobs reachable from origin (0 non-git members) |
| tar | 0.397 GB | `nb2r4_1b81/cand_r4.tar` | candidate tar; 2 members (map_hires_overfit.py + its test) are NOT in git; identical files are in nb2r4_1b81/code and are salvaged **[salvage]** |
| tar | 0.399 GB | `nb2r5_2ac0/cand_r5.tar` | candidate tar; stream-hashed: all members are git blobs reachable from origin (0 non-git members) |
| tar | 0.395 GB | `refcv7_run/02ffcebfc3.tar` | git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs) |
| tar | 0.395 GB | `refcv7_run/0fcb1c1cbb.tar` | git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs) |
| tar | 0.395 GB | `refcv7_run/14907a0a5a.tar` | git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs) |
| tar | 0.395 GB | `refcv7_run/53ecf9b806.tar` | git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs) |
| tar | 0.395 GB | `refcv7_run/60f4c068a8.tar` | git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs) |
| tar | 0.395 GB | `refcv7_run/6fa5e8bba4.tar` | git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs) |
| tar | 0.395 GB | `refcv7_run/fec3a0dccf.tar` | git-archive tar of the KEPT launch tree; pax header commit fec3a0dccfda (on origin); its extracted dir is kept |
| codetree | 0.398 GB | `bha_2325/tree` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.119 GB | `bx_0252/tree` | code tree (1681 files); 2 WIP file(s) (0.03 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.402 GB | `bx_0412/tree` | code tree (2815 files); 13 WIP file(s) (1.18 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.402 GB | `bx_0412/tree_r6` | code tree (2817 files); 14 WIP file(s) (1.19 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.416 GB | `bx_0524/tree_main` | code tree (2825 files); 10 WIP file(s) (1.17 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.416 GB | `bx_0524/tree_r6` | code tree (2827 files); 10 WIP file(s) (1.18 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.415 GB | `bx_0524/tree_tip` | code tree (2811 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.418 GB | `bx_1600/tree_a17` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.418 GB | `bx_1600/tree_tip` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.418 GB | `bx_1600/tree_toy` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.403 GB | `bx_anch_1116/tree` | code tree (2829 files); 3 WIP file(s) (0.96 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.403 GB | `bx_cmain_1014/tree` | code tree (2827 files); 8 WIP file(s) (1.16 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.418 GB | `bx_gate_1136/tree_anch` | code tree (2835 files); 3 WIP file(s) (0.96 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.417 GB | `bx_gate_1136/tree_tip` | code tree (2819 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.418 GB | `bx_gate_1251/tree_anch` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.417 GB | `bx_gate_1251/tree_tip` | code tree (2819 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.416 GB | `bx_tests_0934/tree_main` | code tree (2825 files); 8 WIP file(s) (1.14 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.416 GB | `bx_tests_0934/tree_r6` | code tree (2827 files); 9 WIP file(s) (1.17 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.442 GB | `gate_b2_0201/CAND` | code tree (2839 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.441 GB | `gate_b2_0201/TIP` | code tree (2837 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.442 GB | `gate_b3_0258/CAND` | code tree (2840 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.442 GB | `gate_b3_0258/TIP` | code tree (2839 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_bxa17_1626/CAND` | code tree (2884 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_bxa17_1626/TIP` | code tree (2884 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_bxa2_1252/CAND` | code tree (2884 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.445 GB | `gate_bxa2_1252/TIP` | code tree (2868 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_bxa_1156/CAND` | code tree (2884 files); all but 3 overlay file(s) are git blobs; those 3 are kept in gate_bxa_1156/overlay |
| codetree | 0.445 GB | `gate_bxa_1156/TIP` | code tree (2868 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_c3_2200/CAND` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_c3_2200/TIP` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_c4_2245/CAND` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_c4_2245/TIP` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_c5_2355/CAND` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_c5_2355/TIP` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_comb_1735/CAND` | code tree (2887 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_comb_1735/TIP` | code tree (2884 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_final_1855/CAND` | code tree (2890 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.447 GB | `gate_final_1855/TIP` | code tree (2887 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.415 GB | `gate_fix_2224/FIX` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.415 GB | `gate_fix_2224/TIP` | code tree (2827 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.403 GB | `gate_fix_2224/TIPRED` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_glive_2110/CAND` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.446 GB | `gate_glive_2110/TIP` | code tree (2891 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.400 GB | `gate_lg_0353/cab` | code tree (2852 files); all but 6 overlay file(s) are git blobs; those 6 are kept in gate_lg_0353/overlay |
| codetree | 0.399 GB | `gate_lg_0353/tip` | code tree (2845 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.399 GB | `gate_lg_0415/cab` | code tree (2854 files); all but 5 overlay file(s) are git blobs; those 5 are kept in gate_lg_0415/overlay |
| codetree | 0.399 GB | `gate_lg_0415/tip` | code tree (2847 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.399 GB | `gate_lg_0504/cab` | code tree (2855 files); all but 5 overlay file(s) are git blobs; those 5 are kept in gate_lg_0504/overlay |
| codetree | 0.399 GB | `gate_lg_0504/tip` | code tree (2847 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.400 GB | `gate_lg_0508/cab` | code tree (2867 files); all but 4 overlay file(s) are git blobs; those 4 are kept in gate_lg_0508/overlay |
| codetree | 0.399 GB | `gate_lg_0508/tip` | code tree (2859 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.400 GB | `gate_lg_0518/cab` | code tree (2867 files); all but 3 overlay file(s) are git blobs; those 3 are kept in gate_lg_0518/overlay |
| codetree | 0.399 GB | `gate_lg_0518/tip` | code tree (2859 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.400 GB | `gate_lg_0928/cab` | code tree (2867 files); all but 2 overlay file(s) are git blobs; those 2 are kept in gate_lg_0928/overlay |
| codetree | 0.399 GB | `gate_lg_0928/tip` | code tree (2859 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.400 GB | `gate_lg_0944/cab` | code tree (2867 files); all but 1 overlay file(s) are git blobs; those 1 are kept in gate_lg_0944/overlay |
| codetree | 0.399 GB | `gate_lg_0944/tip` | code tree (2859 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.445 GB | `gate_lg_0951/CAND` | code tree (2868 files); all but 1 overlay file(s) are git blobs; those 1 are kept in gate_lg_0951/overlay |
| codetree | 0.443 GB | `gate_lg_0951/TIP` | code tree (2859 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.398 GB | `gate_lg_2317/cab` | code tree (2841 files); all but 4 overlay file(s) are git blobs; those 4 are kept in gate_lg_2317/overlay |
| codetree | 0.398 GB | `gate_lg_2317/tip` | code tree (2836 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `gate_lgt_1033/CAND` | code tree (2869 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.399 GB | `gate_lgt_1033/TIP` | code tree (2860 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.443 GB | `gate_nb2_0413/CAND` | code tree (2857 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.442 GB | `gate_nb2_0413/TIP` | code tree (2845 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.445 GB | `gate_nb2r2_1037/CAND` | code tree (2868 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.445 GB | `gate_nb2r2_1037/TIP` | code tree (2868 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.445 GB | `gate_nb2r3_1149/CAND` | code tree (2868 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.445 GB | `gate_nb2r3_1149/TIP` | code tree (2868 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.404 GB | `gate_nb2t_0502/CAND` | code tree (2859 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.402 GB | `gate_nb2t_0502/TIP` | code tree (2847 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.441 GB | `gate_new1_2259/NEW1` | code tree (2837 files); all but 3 overlay file(s) are git blobs; those 3 are kept in gate_new1_2259/overlay |
| codetree | 0.441 GB | `gate_new1_2259/TIP` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.441 GB | `gate_new1r2_0048/CAND` | code tree (2837 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.441 GB | `gate_new1r2_0048/TIP` | code tree (2835 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.407 GB | `gmo_bind_pf/tree` | code tree (2955 files); 1 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.402 GB | `gmo_early_0327/code` | code tree (2828 files); 9 WIP file(s) (1.35 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.404 GB | `gmo_early_0327/code_t1` | code tree (2828 files); 6 WIP file(s) (1.27 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.408 GB | `nb2_gate_4797ffb/code` | code tree (2841 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.409 GB | `nb2a19_thor/code` | code tree (2956 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.409 GB | `nb2ml2_thor/code` | code tree (2955 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.409 GB | `nb2r2_cef9/code` | code tree (2844 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.409 GB | `nb2r3_2374/code` | code tree (2853 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.409 GB | `nb2r4_1b81/code` | code tree (2854 files); 2 WIP file(s) (0.07 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.407 GB | `nb2r5_2ac0/code` | code tree (2954 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.186 GB | `refcv6_run/284393ccff` | code tree (2200 files) = commit 284393ccff3e; every file byte-identical to a git blob reachable from origin |
| codetree | 0.185 GB | `refcv6_run/287d72ece8` | code tree (2201 files) = commit 287d72ece83b; every file byte-identical to a git blob reachable from origin |
| codetree | 0.187 GB | `refcv6_run/35be81285e` | code tree (2196 files) = commit 35be81285ec8; 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.187 GB | `refcv6_run/4ca6168822` | code tree (2195 files) = commit 4ca616882283; 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.325 GB | `refcv6_run/82c2331a2f` | code tree (2670 files) = commit 82c2331a2fcb; every file byte-identical to a git blob reachable from origin |
| codetree | 0.187 GB | `refcv6_run/pre_C` | code tree (2196 files); 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.187 GB | `refcv6_run/pre_D` | code tree (2196 files); 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.187 GB | `refcv6_run/pre_E` | code tree (2196 files); 3 WIP file(s) (0.61 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.187 GB | `refcv6_run/pre_F` | code tree (2196 files); 6 WIP file(s) (0.96 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.187 GB | `refcv6_run/pre_G` | code tree (2196 files); 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation **[salvage]** |
| codetree | 0.405 GB | `refcv7_bind/tree` | code tree (2884 files); 1 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.404 GB | `refcv7_bind/tree_map` | code tree (2887 files); 1 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block) **[salvage]** |
| codetree | 0.401 GB | `refcv7_bind_r4/tree` | code tree (2899 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `refcv7_dry_thor/tree` | code tree (2884 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.404 GB | `refcv7_probe/tree` | code tree (2884 files); every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `refcv7_run/02ffcebfc3` | code tree (2891 files) = commit 02ffcebfc351; every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `refcv7_run/0fcb1c1cbb` | code tree (2896 files) = commit 0fcb1c1cbb88; every file byte-identical to a git blob reachable from origin |
| codetree | 0.405 GB | `refcv7_run/14907a0a5a` | code tree (2891 files) = commit 14907a0a5add; every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `refcv7_run/53ecf9b806` | code tree (2891 files) = commit 53ecf9b80634; every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `refcv7_run/60f4c068a8` | code tree (2890 files) = commit 60f4c068a8b6; every file byte-identical to a git blob reachable from origin |
| codetree | 0.401 GB | `refcv7_run/6fa5e8bba4` | code tree (2891 files) = commit 6fa5e8bba478; every file byte-identical to a git blob reachable from origin |
| codetree | 0.399 GB | `wgate_0505` | code tree (2847 files); every file byte-identical to a git blob reachable from origin |

## Appendix B -- script anatomy

`rm_rf PATH` = guard + `du` + (`DRYRUN=0` ? `nice ionice rm -rf -- PATH` + post-check that the path is gone). Every call is preceded by a comment line `# <size>  <category>  <reason>  | surviving: <copy>`. `rm_rf_salv PATH TGZ..` additionally requires the salvage tarball(s) to exist in a real run. B adds `rm_if_hash ALGO HASH PATH`, `rm_dir_if_same_file`, `rm_manifest`.
