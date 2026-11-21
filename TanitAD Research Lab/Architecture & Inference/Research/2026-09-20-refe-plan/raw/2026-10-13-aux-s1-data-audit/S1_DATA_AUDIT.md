# S1 input audit (aux drivable-area head): can the 9 scorer fine-tunes run on the dev box?

> Audited 2026-10-13 ~11:00 Berlin (~09:00 UTC). Machine-readable twin: `s1_data_audit.json`. Evidence files: `evidence/`.
> Evidence classes: MEASURED (ours, artifact named) · ESTIMATED · INHERITED (a doc, not re-verified). Every pod probe was read-only
> (`ls`, `stat`, `cat` of small manifests, `grep -c`/`wc -l` of single code files); nothing was run on the pod, nothing was downloaded.
> The "A40 pod" is the P2 pod; its address is deliberately not written here (the repo is public).

## HEADLINE

**S1 cannot run on the dev box as-is, and two argv paths have NO copy anywhere (launch blockers):**

| unresolved | why it matters |
|---|---|
| `$D/train_teacher_lane` | lost with the SFT-4 pod; the queue source of its seeded subset was never banked (`eval/PREREG_SFT4R.md` "NAMED GAP"). **A missing dir does not error**: `train.py:542` globs `teacher_lane_r*.jsonl`, finds nothing, `n_cols` still becomes 7, and the train-side lane loss is silently fully masked (the launch-gate defect class). |
| `$D/heldout_pdm` | lost with the SFT-4 pod; the P2 pod recomputed only `train_pdm`. With `--require-pdm`, `scorer_finetune.py:633` keeps only held-out sets that carry a PDM label, so with the dir absent `ho_idx` is EMPTY and every evaluation reads n = 0. |

Third blocker, resolvable only by a transfer: **`$PIX` (the 4-camera pixels) exists only on the A40 pod.** The dev box holds 6,044 of the bank's 412,148 images (1.47 %).

**Verdict (a), dev box:** infeasible as-is. Feasible only after (1) a pod-side pack and HF relay of the drawn pixels, **21.6 GB for the first seed (all of N, H1, H2 share it), 47.9 GB for all three seeds**, (2) a recompute of `heldout_pdm` (about 19 min of pod CPU), (3) a regeneration of `train_teacher_lane` or a ruling to run without it, and (4) a ruling on the all-rows-must-resolve rule below. Disk allows ONE seed at a time (peak about 38-42 GB of the 47.98 GB free on D:); all nine at once needs about 84 GB and is infeasible. Nothing can start on the pod before P2 ends (P2 is at step 935, GPU 100 %).

**Verdict (b), A40 pod after P2:** holds pixels, targets, calib, on-policy sets and `train_pdm` already. Missing there: u400 (HF fetch), the held-out sets and logs (65 MB tarball on the pod, not extracted), `heldout_lane` and `heldout_teacher_lane` (ship 2.7 MB from the repo), the aux label store (ship 352 MB), `heldout_pdm` (compute), `train_teacher_lane` (blocker), and **the code package (the pod's trainer is the 982-line pre-aux version)**. `after_p2.sh` starts RL-S1 on the same GPU the moment P2 logs `ZZP2R_DONE`, so S1 needs a queue slot decision. Cost per run: A40 1.3-1.6 h against 3.9-5.1 h on the 4060 (both ESTIMATED from `PREREG_BLANKS_PROPOSED.md`'s 11.4-14.4 s/update).

## 1. What ONE S1 run reads (hop by hop)

| hop | file:line | reads |
|---|---|---|
| flags | `scorer_finetune.py:492-497` | `--ckpt --targets --images --calib --onpolicy --heldout --heldout-logs` |
| model | `:578` `config_for_checkpoint(.., a.ckpt)`, `:581` `load_for_inference`, `:937` `save_full` re-reads `a.ckpt` | the u400 checkpoint, 1,288,489,838 B |
| on-policy banks | `:617` `OnPolicyBank(a.onpolicy ..)`, `:619` `OnPolicyBank(a.heldout ..)` | `train.py:395` globs `onpolicy_*.jsonl`; **every line is loaded** into float32 (about 16,896 B per set) |
| label dirs | `:617-620` pass `lane_labels` / `pdm_labels` / `teacher_lane_labels`; `train.py:503` `pdm_*.jsonl`, `:536` `teacher_lane_r*.jsonl`, `:568` `lane_*.jsonl` | the five dirs under `$D` |
| target bank | `:625` `TargetBank(a.targets, a.images ..)`; `train.py:641` `targets_rank*.jsonl` (rank0 103,037 rows + rank1 86,851 = 189,888); `scorer_targets*.jsonl` is **not** read | 652,289,520 B + calib 2,336,659 B |
| **existence gate** | `train.py:692-713` `resolvable()` on **every** bank row, 4 cameras; `SystemExit` above 0.1 % missing | names only, no pixel bytes, but **all 412,148 images (103,037 frames x 4) must resolve** |
| training rows | `:629` `tr_idx` = bank rows whose key is in `op.by`, log not held out, PDM-labelled (**174,912** rows, MEASURED by rebuilding it from the banked keys) | |
| draw | `:672-673` `Generator().manual_seed(seed)`, `shuffle=True`, batch 8, `drop_last`; `:1285-1287` `total = min(ceil(174,912/8/8)=2,733, 400) = 400` | 400 updates x 8 micro x 8 = **25,600 rows, no repeats** |
| pixel reads | `train.py:799-806` `__getitem__` reads the 4 JPEGs of the drawn row only; **one frame per camera, no history**; `CAMERAS` at `train.py:42` = F0, L0, R0, B0 | |
| held-out reads | `:633` `ho_idx` (3,137 rows), `:785` loader, run at `:1290` (step0) and `:1340` (update 400) | 3,137 rows = 1,705 frames = 6,820 images, 24 logs |

**Answers to the brief's questions.** Pixels are read for the drawn rows only (not all TargetBank targets, not history frames, all 4 cameras). The existence gate, however, touches all 412,148 names.

| count (MEASURED by emulating the trainer's own draw on the banked rows, torch 2.11.0+cu128 = the dev venv; `evidence/sim_draws.py`) | frames | images |
|---|---|---|
| one run, train draw, seed 0 / 1 / 2 | 23,936 / 23,954 / 23,972 (25,600 rows; rank-0 and rank-1 twins share a frame) | 95,744 / 95,816 / 95,888 |
| held-out eval (all runs) | 1,705 | 6,820 |
| **one run in total** (N, H1, H2 of a seed share it) | | **102,564 / 102,636 / 102,708** |
| all nine runs (3 seeds + held-out) | 56,969 | **227,876** |
| whole bank (what the existence gate wants) | 103,037 | 412,148 (1,191 logs) |

Caveat: exact for a run in the same torch build; the pod's torch may draw a different, equal-sized subset. The raw `sim_draws.json` carries a placeholder `union3_frame_tuples = 1` (first-run defect, fixed in the evidence copy of the script); the value is 55,264.

## 2. Where every input is today

`E:/` and `D:/` report identical Used/Free: one volume. "Tarball" = present as a `.tar.zst`, not extracted.

| # | input / argv | dev box | A40 pod | HF (listed, nothing downloaded) | bytes | state |
|---|---|---|---|---|---|---|
| I1 | `--ckpt` u400 | **FOUND** `D:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt`, md5 `8e382eebb35b3e155862453617e5786e` (matches RESULT_SFT4) | MISSING | `Sayood/tanitad-refe-rls1` `base/model_sft4_B_u400.pt` | 1,288,489,838 | resolved on dev |
| I2 | `--targets` train_grow | tarball `refe-onpolicy-2026-10-01/refe_train_grow.tar.zst` (1,043,549,966 B) | **FOUND**, byte-equal members | `Sayood/tanitad-refe-navtrain-banks` same file | 652,289,520 needed | extract (0.65 GB) |
| I3 | `--calib` | in the same tarball | FOUND | same | 2,336,659 | extract |
| I4 | `--images` pixels | **MISSING** (probed below) | **FOUND**, 32 shard dirs, 4 cameras | **MISSING** | 21.6 GB per seed, 47.9 GB all | **BLOCKER** |
| I5 | `--onpolicy` sets | tarball `refe-pod-final-2026-10-04/onpolicy_sets_logs_full.tar.zst` (2,070,541,308 B; md5 `4fc97993..` = MD5SUMS = the pod's recorded md5) | **FOUND** carved 11,088,393,052 B + `sets_full` 11,273,447,245 B | banks repo same file | 11.27 GB extracted | extract |
| I6 | `--heldout` sets | **FOUND, extracted** `_tlane_work/ho/refe_heldout/sets` (7 files, 3,137 sets) | tarball only, not extracted | banks repo | 186,272,504 | resolved on dev |
| I7 | `--heldout-logs` | **FOUND** `_tlane_work/ho/refe_heldout/heldout_logs.txt` | in the tarball | in the tarball | 936 | resolved on dev |
| I8 | `--pdm-labels-train` | MISSING | **FOUND** `refe_sft2/train_pdm`, 8 files, 174,912 rows, `pdm_complete.json` COMPLETE | `Sayood/tanitad-refe-p2-joint-pdm` `pdm_labels/pdm_train_labels.tar.gz` | 478,323,318 (51,468,428 as .tar.gz) | 51.5 MB fetch |
| I9 | `--pdm-labels-heldout` | **MISSING** | **MISSING** | **MISSING** | n/a | **BLOCKER** |
| I10 | `--teacher-lane-train` | **MISSING** (24-row smoke only) | **MISSING** | **MISSING** | n/a | **BLOCKER** |
| I11 | `--teacher-lane-heldout` | in the repo as `.xz` (2,687,804 B) | MISSING | MISSING | 2,687,804 | restore (`xz -d`) |
| I12 | `--lane-labels-heldout` | in the repo as `.xz` (54,760 B) | MISSING | MISSING | 54,760 | restore |
| I13 | `--aux-drv-labels` (H1, H2) | **FOUND** `aux_drivable/label_store`, 351,647,342 B, 103,139 keys | MISSING | MISSING | 351,647,342 | pod needs a ship |
| I14 | code package | FOUND (tip) | **STALE**: 982-line `scorer_finetune.py`, 0 `aux_drv` mentions, no `aux_drivable_head.py` (both `refe-plan` and `refe-plan-rls1`) | n/a | n/a | pod needs a ship |

**Second-location probes for pixels (absence at one place is not absence).** `D:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz` (4,496,159,137 B) is 1 of 32 shards and holds 6,044 of the 412,148 bank images (`evidence/shard25.json`). `data/refe_navtest/frames` is 546 per-log/camera zips (10,060,039,745 B) but 138 **navtest** logs with **zero** overlap with the 1,191 bank logs or the 24 held-out logs (`evidence/navtest_frames_overlap.json`). `data/nuplan-camera`, `refe_trunk_cache`, `refe_targets_4cam`, `refe_navtrain*` and `D:/Archive/devbox-C` hold targets or other data, no navtrain pixels. A depth-7 `find` over `D:/Projects`, `D:/Archive`, `C:/Users/Admin` for `navtrain_pixels*`, `train_grow*`, `refe_sft2*`, `*train_pdm*`, `*teacher_lane*`, `heldout_pdm*`, `heldout_lane*` found only the items above.

## 3. Sizes, space and transfer cost

**Pixel bytes (ESTIMATED projection of a MEASURED sample).** 464 JPEGs (40 logs x 3 frames x 4 cameras, drawn from the needed set) were `stat`-ed on the pod: 464 of 464 present, mean **210,312 B**, cluster SE 3,039 B (CAM_F0 213,392 · L0 208,030 · R0 216,927 · B0 202,897). Projection (95 % band in `s1_data_audit.json`):

| set | images | GB |
|---|---|---|
| held-out only | 6,820 | 1.43 |
| **one run / one seed (N, H1, H2 share it)** | 102,564-102,708 | **21.6** (21.0-22.2) |
| all nine runs | 227,876 | **47.9** (46.6-49.3) |
| whole bank | 412,148 | 86.7 (84.2-89.1); the pod manifest's "80 GB" is 80.7 GiB, consistent |

**Layout trap.** Loose JPEGs on D: (exFAT, 1 MiB allocation unit, `train.py:46-49`, INHERITED) would cost about 1 MiB each: 102,564 files = about 107 GB. The pixels must be the per-(log, camera) zip containers `FrameStore` already supports (`train.py:45-76`; `data/refe_navtest/frames` is the same layout on this disk). Stored JPEGs do not compress, so a zip costs about the sum of its members.

**The all-rows-must-resolve rule (needs a ruling).** `TargetBank` raises `SystemExit` unless 99.9 % of all 189,888 bank rows resolve all four cameras, so shipping only the 102,564 drawn images is refused at load. Options: (A) ship all 412,148 images, 86.7 GB, **infeasible** on 47.98 GB free; (B) ship the drawn images plus **zero-length placeholder members** (about 309,584 x about 130 B = about 40 MB, ESTIMATED) so `resolvable()` passes; the bank rows, `tr_idx` and the draw stay bit-identical, and a drawn row with a placeholder fails loud at decode; **untested**; (C) filter the `--targets` dir: changes `len(tr_idx)` and so every draw, the data is no longer the SFT-4 recipe's: not recommended. B is the cheapest; it is a deviation the gate token and the manifest must name.

**Dev-box disk (MEASURED: D: free 47,982,837,760 B at 10:37 and 47,644,147,712 B at 11:05; C: 31,018,676,224 B at 10:37 and 24,129,781,760 B at 11:05, 97 % full, not usable; both shrink while other work runs, so the 6 GB headroom below is a snapshot).**

| item | GB | already on D:? |
|---|---|---|
| on-policy sets extracted | 11.27 | no |
| targets + calib + `train_pdm` + restored lane labels | about 1.2 | no |
| pixels, first seed incl. held-out | 21.6 | no |
| outputs, 3 runs x 2.63 GB (`model_sft_A.pt` + `model_sft_B.pt`, 1.29 GB each, `save_full` at `:937`; dumps 34 MB) | 7.9 | no |
| **first seed total** | **about 41.9** (6.1 GB headroom) | |
| all nine runs without rotation: pixels 47.9 + 12.5 + outputs 23.7 | about 84 | infeasible |

So: one seed at a time, delete or move the previous seed's pixels (seed 1 adds 15.1 GB of new images, seed 2 about the same), and prune `model_sft_A.pt` (the A arm is not an S1 arm) or the outputs fill D:.

**Transfer time (ESTIMATED from two MEASURED numbers).** Dev-box downlink: two 24 MiB ranged GETs against the private banks repo, discarded, not saved: **13.2 and 9.6 MB/s**, time to first byte 0.6-0.7 s (`evidence/dl_probe.json`; a 48 MiB sample, so large transfers may differ). Pod egress to HF: **4.68 MB/s** (the pod's own `pdm_push.log` for a 51.5 MB file, INHERITED, one file). The relay is push-bound:

| payload | pod push at 4.68 MB/s | dev pull at 9.6-13.2 MB/s | pipelined floor |
|---|---|---|---|
| seed-0 pixels 21.6 GB | 77 min | 27-37 min | about 77 min |
| all seeds 47.9 GB | 171 min | 61-83 min | about 171 min |

Plus the pod-side pack (read 21-48 GB of JPEGs into stored zips; minutes of I/O). Small items: `train_pdm` 51.5 MB (about 5 s), the three tarballs are already on D: (extraction only; the cold read of `refe_train_grow.tar.zst` took 194 s, warm 6.9 s, MEASURED).

**RAM (UNVERIFIED).** 31.8 GB total, 10.8 GB free at the time of audit, 24 logical CPUs. `OnPolicyBank` is about 3.0 GB for the carved 174,912 sets (3.2 GB uncarved, ESTIMATED from 16,896 B per set). With the pinned `--workers 3` on Windows, each spawned worker unpickles its own copy of the dataset: worst case about 4 x 3.2 = 12.9 GB resident, above the free RAM. A changed `--workers` is refused by the gate, so this needs a measured preflight, not an assumption.

**Recompute costs of the two blockers.** `heldout_pdm`: 3,137 sets x 0.357 s/set = about 19 min single stream (the pod's relabel log rate; ESTIMATED), needs the 24 logs' nuPlan DBs (pod: yes; dev box: 5 of 24, 209,084,416 B) and the teacher env. `train_teacher_lane`: 13.6 s/set wall on one dev-box worker (24-set smoke, `_tlane_work/smoke/run10.log`): the lost run's 2,109 rows = 8.0 CPU-h, the full 15,000-set subset = 56.7 CPU-h (ESTIMATED), and it needs a `LANE_QUEUE_SRC` (the HF banks' `onpolicy_queue.tar.zst`, 1,817,869,216 B, is the unconfirmed candidate). Whether a regenerated subset equals SFT-4's cannot be known: the original queue was never banked.

## 4. Pod-side picture (read-only probes)

P2 is training (`step 935`, GPU 23.9 GB used at 100 %, 58 s/step; the prereg's 1,209 steps puts the end about 4.4 h away, about 13:05 UTC, ESTIMATED from INHERITED total). `/workspace/data/rls1/after_p2.sh` waits for `ZZP2R_DONE` and then launches the RL-S1 chain on the same GPU. `stat -f /workspace` shows 80,674 MiB free (the MooseFS quota is not visible to it; not trusted). The outputs of nine S1 runs (23.7 GB unpruned) would fit by that reading.

## 5. What I did NOT establish

- Whether option B (placeholder members) passes the launch gate's data fingerprint and G-LIVE. Not tried.
- Whether the 4060 fits ViT-L, 4 cameras, bf16, `--batch 8` with the head resident (`PREREG_BLANKS_PROPOSED.md` says UNVERIFIED; out of scope here).
- The HF copy of u400 was matched by size only, not md5.
- Pod uplink and pod-to-dev direct transfer rates beyond the one log line; no load was put on the training pod to measure them.
- The aux label store's per-frame coverage of the drawn frames (103,139 keys vs 103,037 bank frames; `coverage_vs_navtrain_yaml` 0.99856). Not audited.

## 6. Deliverable manifest

| artifact | where | one place only? |
|---|---|---|
| `S1_DATA_AUDIT.md` (this file) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-13-aux-s1-data-audit/` | yes, working tree, unstaged (Master Mind lands) |
| `s1_data_audit.json` | same dir | yes |
| `evidence/` (scripts and raw outputs: `sim_draws.py/.json`, `byte_sample.json`, `pod_stat_script.sh`, `pod_stat_out.txt`, `byte_stats.json`, `dl_probe.py/.json`, `hf_files.py/.json`, `hf_list.py`, `hf_repos.json`, `shard25.py/.json`, `navtest_frames_overlap.json`, `dbcheck.py`, `tarlist.py`, `tarx.py`) | same dir | yes |
| extracted `targets_rank0/1.jsonl`, `calib_table.json` (654 MB, scratch only, not part of the package) | `C:/Users/Admin/AppData/Local/Temp/claude/.../scratchpad/tg/` | yes, disposable |

## 7. Copy-identity checks (MEASURED this session, `md5sum` on the dev files)

| file | md5 | matches |
|---|---|---|
| `refe_train_grow.tar.zst` (dev) | `e07c7d6a5c0156f31a5de3af0843c325` | the pod's recorded `hf_dl/extracted/refe_train_grow.md5` |
| `onpolicy_sets_logs_full.tar.zst` (dev) | `4fc979939c5f9b5140ba59a952da77fb` | `MD5SUMS.txt` and the pod's recorded `onpolicy_sets_logs_full.md5` |
| `refe_heldout_v5.tar.zst` (dev) | `1f43409ba8ea2a8909e5035236c73676` | `MD5SUMS.txt` |
| extracted held-out sets, 2 of 7 files | `onpolicy_r0_hw0` `e15a00fd..`, `onpolicy_r1_hw6` `e45d88c4..` | the directory's own `MD5SUMS.txt` |
| `model_sft4_B_u400.pt` (dev) | `8e382eebb35b3e155862453617e5786e` | `eval/RESULT_SFT4.md` |

So the dev-box tarballs are the same bytes the pod restored from HF; extraction on the dev box reproduces the pod's inputs.
