# INVENTORY_DEVBOX - disks C: and D: of the Windows dev box (2026-10-04)

**Read-only inventory.** Nothing was deleted, moved, truncated or modified (the only files written are the deliverables in this folder). The scripts were generated but **not run**, not even in dry-run mode. D: and E: are one volume (D: is `subst D: E:\`), inventoried once as D:. Evidence classes: MEASURED = metadata scan or read by this inventory; INHERITED = taken from a document, not re-verified.

**Units.** "GB" = GiB (2^30 B, the Windows convention). Sizes are **on-disk** bytes: 4 KiB clusters on C:, **1 MiB clusters on the exFAT D:** (so a small file costs 1 MiB and every directory costs >= 1 MiB). Where on-disk and real size differ a lot, both are shown.

Deliverables (this folder): `cleanup_devbox_A.ps1`, `cleanup_devbox_B.ps1`, `cleanup_devbox_C.ps1` (all removals commented out), `maintenance_repack_D_repo.ps1` (not a delete), this file, `INVENTORY_DEVBOX_evidence.json`. (The `cleanup_thor_*.sh` / `B_sha256_*` files in the same folder belong to a sibling agent and were not touched.)

## 1. Headline

| drive | capacity | used | free now |
|---|---|---|---|
| C: (NTFS, 4 KiB) | 930.8 GB | 917.5 GB | **13.3 GB** (measured ~09:00 local; the brief said ~14.7) |
| D: = E: (exFAT, 1 MiB clusters, USB) | 3,726 GB | 3,718.9 GB | **7.1 GB** |

Reclaimable, **on-disk GB**, by category and drive (A = safe + re-creatable, B = proven redundant copy, C = probably obsolete / PI decision):

| category | C: | D: | what it is |
|---|---|---|---|
| **A** | 86.9 | 0.0 | package caches 39.0, stale Temp + old Claude scratchpads 14.1, 67 git-archive scratch trees 33.9 (all on C:); D: has no A item |
| **B** proven identical copies (default-on in script B) | 3.7 | 4.3 | REFe pod-final backup on C:; refcv4b_final + refcv3_final copies in devbox-C; duplicate refcv6 step-38000 checkpoint |
| B-derived (script B, **off** by default) | 8.1 | 187.6 | source zips whose every member is extracted next to them: 3 nuPlan zips 187.6 GB on D:, comma2k19 Chunk_1.zip 8.1 GB on C: |
| B-held (commented out in script B) | 4.6 | 0.0 | identical copies, but live qland scripts hard-code the C: path (refcv5v2_final, tanitad-wt/_s2build) |
| **C** needs PI decision (script C, all commented) | 292.0 | 688.8 | old experiment outputs, worktrees, archives, eval kits - see section 6 for risk per line |
| repack of D:\Projects\TanitAD\.git (not a delete) | - | ~36-40 | `maintenance_repack_D_repo.ps1`: 42,842 loose objects x 1 MiB cluster |

**Expected free space** (arithmetic on the on-disk numbers above; the live waiter needs >= 10 GB free on C:, see section 3):

| after | C: free | D: free |
|---|---|---|
| now | 13.3 | 7.1 |
| A | 100.2 | 7.1 |
| A + B (proven copies, default) | 103.9 | 11.4 |
| A + B + B-derived zips (switch on) | 112.1 | 199.0 |
| A + B + B-derived + B-held (after repointing 2 scripts) | 116.6 | 199.0 |
| A + B + B-derived + B-held + **all** of C (upper bound, not a recommendation) | 408.7 | 887.8 |
| (plus git repack on D:) | - | +36..40 |

**The 10 biggest reclaim candidates** (on-disk GB; category in brackets):

| # | path | GB | cat | why |
|---|---|---|---|---|
| 1 | D:\thor_archive | 175.2 | C | D:\thor_archive\2026-09-14 (alpasim 60.3, sam3map 47.2, models 16.6, nurec_scenes 7.9, ... |
| 2 | D:\Projects\TanitAD\_c | 111.0 | C | D:\Projects\TanitAD\_c - 184 tar.zst archives of C: worktrees/scratch trees compacted on ... |
| 3 | D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_val.zip | 90.3 | B/derived | nuPlan val archive (source zip of the extracted dbs) |
| 4 | D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_test.zip | 89.3 | B/derived | nuPlan test archive (source zip of the extracted dbs) |
| 5 | C:\Users\Admin\refav1_trainprobe | 59.7 | C | refav1_trainprobe - fp8 token cache (36.9 GB) + 600 episodes (21.2 GB) + feats (1.3 GB), ... |
| 6 | C:\lgt | 56.4 | C | C:\lgt - lander/gate scratch (2026-09-27/28): rc7cost_runs 51.9 GB + r7ldr_scratch 1.2 + 9 ... |
| 7 | D:\Archive\devbox-C\carla | 53.2 | C | D:\Archive\devbox-C\carla (+ junction C:\Users\Admin\carla) |
| 8 | C:\Users\Admin\tanitad-data\eval | 33.0 | C | tanitad-data\eval - comma2k19-val (14.8 GB), comma_val.tgz (7.8 GB), v1_speedjerk_ckpt.pt ... |
| 9 | D:\Archive\devbox-C\wpd-probe | 31.0 | C | D:\Archive\devbox-C\wpd-probe (+ junction C:\Users\Admin\wpd-probe) |
| 10 | C:\Users\Admin\tanitad-data\tanitdataset | 29.7 | C | tanitad-data\tanitdataset - HF staging copies of the comma2k19 lake (hf_stage_R, ... |

**Surprises**

- **D:\Projects\TanitAD\.git costs 48.1 GB on disk for 7.4 GB of content.** `git count-objects`: 42,842 loose objects (5.37 GB) + 16 packs (2.02 GB); every loose object takes a 1 MiB exFAT cluster. 3,374 loose objects are already inside a pack (`prune-packed` is lossless). Repack = ~36-40 GB back. MEASURED (`git count-objects -vH`, read-only).
- **D:\Archive\devbox-C is not an archive.** 33 of its 35 children are the targets of junctions at `C:\Users\Admin\<same name>` (left by the 2026-09-13 compaction), and the live NavSim run reads `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/...`, `exp/refcv6_navtest416/frame_bank` and `data/openscene/navsim_logs/test`. `devbox-C\tanitad-caches` is exactly the 3 targets of the junctions inside KEEP `C:\Users\Admin\tanitad-caches`. Deleting any child without removing its junction leaves a dangling link.
- **exFAT slack.** D: used = 3,718.9 GB; the sum of the top-level directories on disk is 3665.9 GB; the 53 GB gap is directory clusters (D: holds 54,241 directories x 1 MiB = 53 GB, MEASURED by a metadata walk). Small-file trees are inflated: `devbox-C\navsim\venv` 46.5 GB on disk for 2.2 GB real (47,295 files), `thor_archive\2026-09-14\alpasim` 60.3 GB for 2.8 GB (60,998 files), `TanitAstraResearch` 47.3 GB for 10.8 GB, `TanitAD\.claude\worktrees` 21.0 for 5.9, `TanitAD-wt-1024` 15.7 for 4.7. **Total slack (alloc - real) over the top-level dirs of D: = 426 GB.** Packing such trees into one tar frees the slack without losing a byte.
- **Two NavSim venvs.** `D:\Archive\devbox-C\navsim\venv` (reached through the junction `C:\Users\Admin\navsim\venv`, named as THE venv in 4 briefs/fallback docs) and `C:\Users\Admin\navsim-crun\venv` (used by the live run) share 45,535 of 46,943 files by relative path and size, differ in the launcher exes and pyvenv.cfg. The D: copy costs 46.5 GB on disk. It is inside a KEEP dir, so it is only an observation (section 7).
- **nuPlan source zips sit next to their extraction**: val 90.3 + test 89.3 + mini 8.0 = 187.6 GB on D:. All 1,349 / 1,381 / 64 `.db` members are present extracted with identical name+size (test/val in nuplan-v1.1/splits, mini in dblinks/; zip central directory read, no content read); the extract logs say per-member CRC. Public S3 objects (INHERITED: memory note + pull_splits.log), so re-downloadable - ~35 h for the 201 GB at the 1.6 MB/s dev-box rate (INHERITED from memory).
- **47 `ckpt.pt` files = 51.4 GB in `C:\lgt\rc7cost_runs`**, 26 of them exactly 1,172,814,977 B (MEASURED). Same SIZE only: they are checkpoints of different cost-probe runs, content NOT compared, so I do not call them duplicates.
- **`tanitad-data\tanitdataset` is hardlinked**: hf_stage_R, hf_stage_C and lake are 14.8 GB each by size but share inodes - real 29.7 GB, not 44.5. Deleting only one of the three frees nothing.
- **`C:\Users\Admin\refcv4b_repo` contains a copy of `<token-file>`** (seen as a top-level entry; contents not read). Worth deleting for credential hygiene whatever else you decide (line in script C).
- **`D:\thor_archive` may be the only copy**: the sam3-only-road-map RESULT says the files were md5-verified onto D: and then **removed from Thor** on 2026-09-14.
- **This session's own scratchpad is 5.3 GB** (`Temp\claude\G--...\7c618562-...`), and the REFe session is mid-upload from `E:/Projects/TanitAD-artifacts/refe-pod-final-2026-10-04` and `refe-onpolicy-2026-10-01`; both are KEEP.

**Suggested order (lowest regret first)**

1. `cleanup_devbox_A.ps1` dry-run, then run: C: 13.3 -> ~100 GB, nothing of value lost (everything is cache, stale temp or a tree that is byte-identical to git).
2. `cleanup_devbox_B.ps1` (default switches): D: +4.3 GB (crosses the waiter's 10 GB threshold) and C: +3.7 GB.
3. `maintenance_repack_D_repo.ps1`: D: +36-40 GB, loses no object.
4. Lossless packing instead of deleting: `thor_archive\2026-09-14\alpasim` (60,998 files, 2.8 GB real) into one tar = ~57 GB back; same idea for `devbox-C\navsim\venv`-like small-file trees.
5. B-derived switch (`$IncludeExtractedArchives`): D: +187.6 GB, C: +8.1 GB - the only cost is a ~35 h re-download if the extracted nuPlan dbs are ever lost.
6. Script C is a menu. Lowest regret on C:: `refav1_trainprobe` 59.7 GB (re-creatable from Thor: scp ~16 min + fp8 re-encode), `C:\lgt` 56.4 GB, `rc7restart_work` 10.6 GB, `C:\lgs` 12.4 GB, and `refcv4b_repo` for the <token-file> copy. Highest regret: `thor_archive` and `_c` (possibly the only copies).

## 2. Method and what was NOT done

- **Sizing**: metadata-only walks (`FindFirstFile`-level; no file opened), junction/symlink entries skipped (not followed, so nothing is double counted), 1 MiB / 4 KiB cluster rounding applied. D: scans ran at idle priority. Hardlinks: `os.lstat` link counts checked for tanitad-data, tanitad-caches, refav1_trainprobe, lgt, rc7restart_work, lg0926, tanitad-wt, tanitad-mirror (only `tanitdataset` 28 files, `refav1-eval141` 41 files and `ddv2rl-20260915` 41 files are multi-linked).
- **Reads of D: content** (all sequential, idle priority): md5 of the 5 directory pairs of section 5 (~14 GB read on D:) and of the two refcv6 checkpoints (2.4 GB), the 3 steering registers (3.8 MB), small manifests/logs. No recursive content grep on D:. No file over 2,000,000,000 B was hashed.
- **git-archive trees**: every file whose mtime differs from the tree's modal (extract) mtime was hashed as a git blob and looked up with `git cat-file --batch-check` in `D:/Projects/TanitAD/.git` and `C:/Users/Admin/tanitad-push/.git`; for the A-class trees every file of the tree was hashed (section 4.3). A tree with `.git`, a foreign top-level entry, unique data outputs, or any file written in the last 72 h is category C, not A.
- **Reference check**: text files <= 1.5 MB under the KEEP areas (`qland`, `ev7`, `ev7nav`, `navsim-crun` minus venv/data, `D:\refcv7_eval_kit`, the NavSim package, the memory dir) were searched for each candidate path; the three registers (MODEL_REGISTRY, GOALS_AND_CLAIMS, RETRACTION_LOG) for each directory name. A hit in a finished one-shot script is reported as a note, a hit in a script that is still an input is a hold.
- **Live processes** (command lines) were read to find what is open right now: the NavSim job (`run_navsim_refcv7.py`, `run_bridge7.py`), the waiter (`waiter_final.sh 50400 129600 120 cdisk`), `hf_upload_all.py` / `push_refe_pod_final.py` / `hf_push_media.py`, `adopted_pipeline.py` + `run_adopted_rest.py` (cwd unknown), the TanitADStudio vite server, Hermes, Codex.
- **Not done / not verifiable here**: HF-side copies were not re-checked (no network call); `onpolicy_sets_logs_full.tar.zst` (2,070,541,308 B) and the zips are > 2 GB so only recorded md5 / member-level checks exist; `C:\Windows`, `Program Files`, `ProgramData` (9.6 + 8.0 + 3.0 GB) and the OS files `pagefile.sys` 26.7 GB / `hiberfil.sys` 12.7 GB are out of scope (`powercfg /h off` would return 12.7 GB, a system setting - the PI's call).

## 3. Live constraints that shaped the lists

- **The waiter wants space on C:.** `qland/work/refcv7/launch_waiter_cdisk.sh` -> `waiter_final.sh 50400 129600 120 cdisk` (running, 36 h budget) starts the FINAL-checkpoint chain only when free disk >= 10 GB and free commit >= 6 GB; in `cdisk` mode the 4.5 GB batch cache lives on **C:** because D: has only 7.1 GB (< 10). C: now 13.3 GB free - do not let it fall under 10 GB; script A moves it to ~100 GB. D: needs +3 GB free for `disk` mode.
- **Junctions into D:** `C:\Users\Admin\{carla, dkhead_run, navcomp, navsim, p1gate, refav1_drive, refav1_eval_full, refav1_eval_slice, refav1_probe, refcv3_final, refcv4b_egodrop, refcv4b_final, rlgate2, rl_refcv3_min, rl_rescope, run_paired, run_refcv3_ol, run_refcv3_viz, run_refcv4, run_refcv4v, run_wbank, tanitad-bench, tanitad-ctg, tanitad-egodrop, tanitad-evtool-20260906, tanitad-labelwire, tanitad-lwbase, tanitad-perbatch, tanitad-refcv4b-analyze, tsc2, veto_run, wpa-readout, wpd-probe}` -> `D:\Archive\devbox-C\<same>`; `C:\Users\Admin\tanitad-caches\{encloc-20260818, mm-e19-assets-20260901, sp2-lead130-20260818}` -> `D:\Archive\devbox-C\tanitad-caches\<same>`; `C:\Users\Admin\refcv5cmp\eval` -> `tanitad-data\refav1-eval141\eps`; `tanitad-data\physicalai\{calibration\camera_intrinsics, calibration\sensor_extrinsics, camera\camera_front_wide_120fov}` (self-referential duplicates inside the corpus). The scripts treat a junction as a link (never recurse into it) and refuse any directory that contains one.
- **Memory/commit headroom** (memory note "dev-box commit limit binds"): deleting does not need RAM; but run script B's optional `-VerifyMd5` mode alone, not next to a GPU job.
- **D: letter**: the scripts abort if `D:\Projects\TanitAD\.git` is not visible (`subst D: E:\` is per logon session).

## 4. Category A - safe to delete, re-creatable (script `cleanup_devbox_A.ps1`, C: only)

Total **86.9 GB** on C: (136 items). D: has no category-A item.

### 4.1 Package / model caches (39.0 GB)

| path | GB | reason / evidence | note for the PI |
|---|---|---|---|
| C:\Users\Admin\AppData\Local\uv\cache | 24.7 | package cache, re-downloadable from PyPI. single dir `cache` + uv-receipt.json in AppData\Local\uv; NOT the uv python installs (those are AppData\Roaming\uv\python = KEEP) | offline venv rebuilds (REFe eval venv was rebuilt offline once) lose their wheel source -> need network. Interpreters are untouched. |
| C:\Users\Admin\.cache\huggingface\hub\datasets--Sayood--tanitad-physicalai-w120-256x640cyl | 6.5 | dataset snapshot cache, re-downloadable from the PI private HF (token in <token-file>). named only by stack/scripts/aug120_pipeline.py, pod_pull_b1_epcache.py, refc_v3_train.py (dataset builders / Thor trainer); no live-job script (NavSim package, waiter chain, refcv7 eval) names it; refcv6_loader.py sets HF_HUB_OFFLINE for MODEL init only | any job run with HF_HUB_OFFLINE=1 that expects this dataset would fail; the live refcv7 chain does not read it |
| C:\Users\Admin\AppData\Local\pip\cache | 6.1 | package cache, re-downloadable. 1,952 files |  |
| C:\Users\Admin\AppData\Local\npm-cache | 1.0 | package cache, re-downloadable (TanitADStudio vite dev server keeps its own node_modules).  |  |
| C:\Users\Admin\.cache\huggingface\hub\datasets--Sayood--tanitad-ph0-aug120 | 0.44 | dataset snapshot cache, re-downloadable. default-arg string in stack/scripts/refc_v3_train.py (Thor trainer); not read by a dev-box live job |  |
| C:\Users\Admin\.cache\huggingface\xet | 0.12 | download-dedup cache, re-created on next hf download. 85 files |  |

Kept on purpose: `%APPDATA%\uv\python` (interpreters in use), the HF hub **models--*** entries (sam3 3.2, dinov3 1.4, dinov2 0.3, timm resnet101/34 0.25 GB - repo code loads them with `HF_HUB_OFFLINE=1`, e.g. `refcv6_loader.py:106`), conda: none present. HF **dataset** entries are in A because only builder/Thor-trainer scripts (`aug120_pipeline.py`, `pod_pull_b1_epcache.py`, `refc_v3_train.py`) name them.

### 4.2 Stale Temp and old Claude scratchpads (14.1 GB, 63 items >= 5 MB)

Rule: top-level entry of `%LOCALAPPDATA%\Temp` whose newest file is older than 2026-10-01 09:30 UTC (3 days), not under `Temp\claude` (sessions handled separately), >= 5 MB. Another 278 MB of tiny stale entries (1073 of them, < 5 MB each) are left alone. Families (newest file in brackets):

| family | entries | GB | newest |
|---|---|---|---|
| clipid_ref* | 2 | 6.67 | 2026-09-22 |
| measures_selftest* | 6 | 1.18 | 2026-09-27 |
| v#smoke* | 11 | 0.30 | 2026-09-27 |
| i#probe#* | 2 | 0.29 | 2026-09-27 |
| f#_nav_sigreg_bank* | 2 | 0.27 | 2026-09-27 |
| f#probe_abc_post_F#* | 2 | 0.27 | 2026-09-27 |
| f#_nav_o#* | 2 | 0.27 | 2026-09-27 |
| f#_nav_o#_no_restore* | 2 | 0.27 | 2026-09-27 |
| f#_nav_uniform* | 2 | 0.27 | 2026-09-27 |
| f#probe_nav_sigreg_bank* | 2 | 0.27 | 2026-09-27 |
| f#probe_nav_o#* | 2 | 0.27 | 2026-09-27 |
| f#probe_nav_uniform* | 2 | 0.27 | 2026-09-27 |
| f#probe_abc_pre_F#* | 2 | 0.27 | 2026-09-27 |
| f#_nonav_o#* | 2 | 0.27 | 2026-09-27 |

Claude session scratchpads (not running, > 3 days): `8e7cfa33` 843 MB; `ff19d6ac` 498 MB; `8fc25020` 193 MB; `76eee715` 77 MB. **Held back (inside the 3-day window)**: `bd7d00af` 3.5 GB (newest 2026-10-01 18:28 UTC) and `D--Projects-TanitAD\91effc67` 8.1 GB (newest 2026-10-01 23:08 UTC): they pass the 3-day mark at 2026-10-04 18:30 UTC and 23:10 UTC and then qualify as A if no session resumes them. Live: `7c618562` (this session, 5.3 GB) and `E--Projects-TanitAD` (REFe).

### 4.3 git-archive scratch trees (33.9 GB, 67 trees)

`bxh_*`, `nb2*`, `gt13_*`, `cfix`, `ctree_*`, `tcW8_*`, `tcK/tcL`, `rc7*`, `r4tree`, `libtree`, `ev6_82c2331`, `ev6_postfix*`, `tip_2ac0_clean`, `bxh_tree_e`, `tanitad-a7-run`: no `.git`, no foreign top-level entry, newest file older than 72 h. "in git" = the file's git blob hash was found in `D:/Projects/TanitAD/.git` or `tanitad-push/.git` (MEASURED for EVERY file of every tree below).

| tree | GB | files | not in git (n, MB, ext) | note |
|---|---|---|---|---|
| bxh_tree_e | 6.21 | 18031 | 12, 1.17 MB, .py x12 |  |
| libtree | 3.52 | 559 | 0 | mentioned by qland/work/coord_land/stage_libpdf.py (finished one-shot, 09-26) |
| tcW8_cand | 0.58 | 4588 | 0 | mentioned by qland/work/coord_land/gate_tcW8_*.log (gate logs) |
| tcW8_tip | 0.57 | 4464 | 0 | mentioned by qland/work/coord_land/gate_tcW8_*.log (gate logs) |
| v7f_chain | 0.42 | 2828 | 4, 0.45 MB, .py x4 | CRLF extract (2382 files match git only after CRLF->LF);  |
| rc7ceil | 0.39 | 2842 | 1, 0.17 MB, .py x1 | CRLF extract (2401 files match git only after CRLF->LF);  |
| bxh_tree_a17 | 0.39 | 2835 | 0 |  |
| ctree_cand | 0.39 | 2869 | 0 | mentioned by qland/work/coord_land/make_test_tree.py (generator script, 09-26) |
| bxh_tree_i6 | 0.39 | 2827 | 9, 1.17 MB, .py x9 |  |
| bxh_tree_h6 | 0.39 | 2827 | 10, 1.18 MB, .py x10 |  |
| bxh_tree_i | 0.39 | 2825 | 8, 1.16 MB, .py x8 |  |
| bxh_tree_h | 0.39 | 2825 | 10, 1.17 MB, .py x10 |  |
| nb2r5_tree_2ac0 | 0.38 | 2955 | 1, 0.00 MB, .log x1 |  |
| ev6_postfix | 0.38 | 2867 | 1, 0.00 MB, .txt x1 | CRLF extract (2424 files match git only after CRLF->LF);  |
| ev6_postfix_da8400b_overlay | 0.38 | 2866 | 4, 0.36 MB, .py x3, .txt x1 | CRLF extract (2414 files match git only after CRLF->LF);  |
| rc7combo | 0.38 | 2895 | 0 |  |
| rc7restart | 0.38 | 2891 | 0 |  |
| gt13_cand_212930 | 0.38 | 2949 | 1, 0.20 MB, .xml x1 | mentioned by qland/work/coord_land/gate_b1213.json (gate record) |
| gt13_tip_212930 | 0.38 | 2947 | 1, 0.19 MB, .xml x1 | mentioned by qland/work/coord_land/gate_b1213.json (gate record) |
| nb2r4_tree_1b81 | 0.38 | 2855 | 3, 0.07 MB, .py x2, .log x1 |  |
| nb2r3_tree_2374 | 0.38 | 2854 | 1, 0.00 MB, .log x1 |  |
| bxh_tree_g6 | 0.38 | 2827 | 11, 1.20 MB, .py x11 |  |
| bxh_tree_g | 0.38 | 2825 | 11, 1.20 MB, .py x11 |  |
| nb2r2_tree_cef9 | 0.38 | 2845 | 1, 0.00 MB, .log x1 |  |
| nb2_tree_4797ffb | 0.38 | 2842 | 1, 0.00 MB, .log x1 |  |
| bxh_tip_cef9709 | 0.38 | 2811 | 0 |  |
| bxh_tree_f6 | 0.38 | 2813 | 13, 1.19 MB, .py x13 |  |
| bxh_tree_f | 0.38 | 2811 | 12, 1.18 MB, .py x12 |  |
| bxh_tree_d | 0.38 | 2810 | 12, 1.17 MB, .py x12 |  |
| r4tree | 0.38 | 2896 | 0 |  |
| nb2_tip_2310 | 0.38 | 2835 | 0 | CRLF extract (2395 files match git only after CRLF->LF);  |
| nb2_tip_2203 | 0.38 | 2827 | 0 | CRLF extract (2387 files match git only after CRLF->LF);  |
| nb2_tree_b3prev | 0.38 | 2830 | 1, 0.00 MB, .log x1 |  |
| nb2_tree_b4a5 | 0.38 | 2832 | 9, 1.25 MB, .py x5, .log x4 |  |
| rc7frozen | 0.38 | 2893 | 0 |  |
| bxh_tip_4797ffb | 0.38 | 2797 | 0 |  |
| bxh_tree_b | 0.38 | 2808 | 13, 1.20 MB, .py x13 |  |
| nb2_tree_0638 | 0.38 | 2830 | 14, 1.38 MB, .py x10, .log x4 |  |
| nb2a19_tree | 0.38 | 2956 | 0 |  |
| nb2ml2_tree | 0.38 | 2957 | 1, 0.00 MB, .log x1 |  |
| rc7lever2 | 0.38 | 2891 | 0 |  |
| bxh_tree_m | 0.38 | 2835 | 0 |  |
| bxh_tree_a | 0.38 | 2796 | 11, 1.15 MB, .py x11 |  |
| bxh_tip_68e8ec6 | 0.38 | 2792 | 0 |  |
| bxh_tree_k | 0.38 | 2827 | 3, 0.96 MB, .py x3 |  |
| bxh_tip_49f10b4 | 0.38 | 2796 | 0 |  |
| bxh_tree_toy | 0.37 | 2835 | 0 |  |
| tip_2ac0_clean | 0.37 | 2836 | 1, 0.00 MB, .log x1 |  |
| nb2r5_tree_8796 | 0.37 | 2855 | 2, 0.03 MB, .log x1, .py x1 |  |
| nb2r4_tip_1b81 | 0.37 | 2853 | 0 |  |
| bxh_tree_c | 0.37 | 2811 | 14, 1.19 MB, .py x14 |  |
| nb2r3_tip_2374 | 0.37 | 2852 | 0 |  |
| nb2r2_tip_cef9 | 0.37 | 2843 | 0 |  |
| bxh_tip_2ac0bfb | 0.37 | 2835 | 0 |  |
| nb2_tip_4797ffb | 0.37 | 2829 | 0 |  |
| bxh_tip_28d8365 | 0.37 | 2835 | 0 |  |
| bxh_tip_1b8170e | 0.37 | 2819 | 0 |  |
| bxh_tip_2374cd2 | 0.37 | 2819 | 0 |  |
| nb2_tip_b3prev | 0.37 | 2817 | 0 |  |
| nb2_tip_b4a5 | 0.37 | 2816 | 0 |  |
| nb2_tip_0638 | 0.37 | 2814 | 0 |  |
| ev6_82c2331 | 0.31 | 2743 | 1, 0.00 MB, .txt x1 | CRLF extract (2408 files match git only after CRLF->LF);  |
| cfix | 0.30 | 2710 | 0 | mentioned by qland/work/a16_switch/stage_a16fix.py (finished one-shot, 09-26) |
| tcL | 0.26 | 1111 | 0 | mentioned by qland historic stage records only |
| ctree_tip | 0.21 | 2402 | 0 | mentioned by qland/work/coord_land/make_test_tree.py (generator script, 09-26) |
| tcK | 0.19 | 998 | 0 | mentioned by qland/work/watch6/gate_watch_k.py (finished watch script) |
| tanitad-a7-run | 0.17 | 2152 | 1, 0.00 MB, .txt x1 | CRLF extract (1943 files match git only after CRLF->LF);  |

Trees with a few patched source files not in git (<= 1.4 MB each, listed above with n > 0) are the iterations of the box-head / map-hires candidate patches; their landed form is in the repo. They sit in a separate block (A3b) of the script so you can skip it.

## 5. Category B - redundant copies (script `cleanup_devbox_B.ps1`)

Before each delete the script re-checks that the survivor exists with identical relative paths and sizes (`-VerifyMd5` re-hashes both sides). MEASURED here: md5 of both copies, read on 2026-10-04.

| item (to delete) | GB on disk | cat | proof | survivor | note |
|---|---|---|---|---|---|
| C:\Users\Admin\tanitad-backup\refe-pod-final-2026-10-04 | 3.7 | B-copy | md5 of both copies equal for 6 of 7 files (1.88 GB read-verified); the 1 file > 2 GB (onpolicy_sets_logs_full.tar.zst, 2,070,541,308 B) has equal size and the same recorded md5 4fc97993... in both MD5SUMS.txt (identical manifests) - NOT read-hashed | D:\Projects\TanitAD-artifacts\refe-pod-final-2026-10-04 (the REFe session is uploading it to HF right now: a third copy) | the C: copy was written 2026-10-04 01:19 local (apparently a backup copy made around the HF upload); deleting it leaves one dev-box copy + the HF upload in progress. Wait until the REFe upload finishes if you want 2 independent copies. |
| D:\Archive\devbox-C\refcv4b_final (+ junction C:\Users\Admin\refcv4b_final) | 1.6 | B-copy | md5 equal for 25 of 25 files (1.72 GB read-verified on both sides), same relative paths and sizes | D:\Projects\TanitAD-artifacts\refcv4b_final | same physical disk as the survivor (no failure-domain gain). C:\Users\Admin\refcv4b_final is a junction into the deleted dir; the script removes the link too. |
| D:\Archive\devbox-C\refcv3_final (+ junction C:\Users\Admin\refcv3_final) | 1.6 | B-copy | md5 equal for 7 of 7 files (1.71 GB read-verified on both sides) | D:\Projects\TanitAD-artifacts\refcv3_final | same physical disk as the survivor. Junction C:\Users\Admin\refcv3_final is removed too. |
| D:\refcv6_eval_kit\ckpt\ckpt_step38000_stopped.pt | 1.1 | B-copy | md5 of both = 5a2e7222a9f5f8c7aa7bf38ef4698d8a = the recorded md5 in ckpt\MD5SUMS and STOPPED_BY_PI.json (1,177,897,033 B each); read-hashed 2026-10-04 | D:\refcv6_eval_kit\ckpt_final\ckpt_step38000.pt (+ HF private Sayood/tanitad-refcv6-r101-s0, upload "verified in a follow-up" per MODEL_REGISTRY line 5273 - NOT re-checked by me) | same disk. The MD5SUMS manifest still lists this file name afterwards (a stale manifest line, harmless). |
| D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_val.zip | 90.3 | B-derived | zip central directory read (a few MB, no content read): 1381 of 1381 .db members have an extracted file with identical name+size in nuplan-v1.1/ (test/trainval; the mini members in nuplan-v1.1/splits and dblinks/) (LICENSE is at nuplan/LICENSE); the extract logs (extract_test.log / val_extract.json) record "extracting ... with per-member CRC". Whole-zip md5 NOT computed (> 2 GB). | D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\ (extracted) - and the zips are re-downloadable (public nuPlan S3 objects: INHERITED from memory note nuplan-s3-objects-are-public; pull_splits.log shows test/val were curl-downloaded from it) | frees the most space on D: but the extraction is on the SAME disk; if the dbs are ever damaged the 201 GB of zips must be re-downloaded: ~35 h at the dev-box rate of 1.6 MB/s (INHERITED rate, memory refe-training-data-on-a40-pod). |
| D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_test.zip | 89.3 | B-derived | zip central directory read (a few MB, no content read): 1349 of 1349 .db members have an extracted file with identical name+size in nuplan-v1.1/ (test/trainval; the mini members in nuplan-v1.1/splits and dblinks/) (LICENSE is at nuplan/LICENSE); the extract logs (extract_test.log / val_extract.json) record "extracting ... with per-member CRC". Whole-zip md5 NOT computed (> 2 GB). | D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\ (extracted) - and the zips are re-downloadable (public nuPlan S3 objects: INHERITED from memory note nuplan-s3-objects-are-public; pull_splits.log shows test/val were curl-downloaded from it) | frees the most space on D: but the extraction is on the SAME disk; if the dbs are ever damaged the 201 GB of zips must be re-downloaded: ~35 h at the dev-box rate of 1.6 MB/s (INHERITED rate, memory refe-training-data-on-a40-pod). |
| C:\Users\Admin\tanitad-data\comma2k19\Chunk_1.zip | 8.1 | B-derived | zip central directory vs tanitad-data/comma2k19/extracted/Chunk_1: 7,333 of 7,333 members present with identical size (/ in the name is _ on disk); whole-zip md5 not computed (8.1 GB, > 2 GB) | C:\Users\Admin\tanitad-data\comma2k19\extracted (and the public comma2k19 dataset; HF Sayood/tanitad-comma2k19-episodes) | re-download of Chunk_1 is ~8 GB |
| D:\Projects\TanitAD\data\nuplan\nuplan-v1.1_mini.zip | 8.0 | B-derived | zip central directory read (a few MB, no content read): 64 of 64 .db members have an extracted file with identical name+size in nuplan-v1.1/ (test/trainval; the mini members in nuplan-v1.1/splits and dblinks/) (LICENSE is at nuplan/LICENSE); the extract logs (extract_test.log / val_extract.json) record "extracting ... with per-member CRC". Whole-zip md5 NOT computed (> 2 GB). | D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\ (extracted) - and the zips are re-downloadable (public nuPlan S3 objects: INHERITED from memory note nuplan-s3-objects-are-public; pull_splits.log shows test/val were curl-downloaded from it) | frees the most space on D: but the extraction is on the SAME disk; if the dbs are ever damaged the 201 GB of zips must be re-downloaded: ~35 h at the dev-box rate of 1.6 MB/s (INHERITED rate, memory refe-training-data-on-a40-pod). |
| C:\Users\Admin\tanitad-wt\_s2build | 3.3 | B-held | md5 equal for 962 of 962 files (3.49 GB read-verified on both sides) | D:\Projects\TanitAD-artifacts\_s2build-copy-20260919 | HELD: 238 KEEP-area text files (qland a2_harness_smoke.sh, a3_*, a8_occupancy_5k.sh ...) use LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz as an input. Repoint to the D: copy, then un-comment. |
| C:\Users\Admin\refcv5v2_final | 1.3 | B-held | md5 equal for 19 of 19 files (1.43 GB read-verified on both sides) | D:\Projects\TanitAD-artifacts\refcv5v2_final | HELD: qland/t1_rolls.sh, qland/t1_s1_only.sh and qland/pkgrl/code/t1_rolls.sh hard-code BASE_CK=/c/Users/Admin/refcv5v2_final/ckpt.pt (+config.json) and PREREG_REFCV6.md cites the C: path as MEASURED evidence. Repoint those scripts to the D: copy first, then un-comment the line. |

Also observed, **not** B: `C:\Users\Admin\a40-rescue` vs `TanitAD-artifacts\a40-rescue` (0.34 GB; same sizes, C: copy has one extra meta.json; the B1 join lives there - KEEP); the 12 + 11 `model.pt` files (1.05 GB each) under `devbox-C\tanitad-lwbase` / `labelwire` / `tanitad-mirror` / `tanitad-wt` look like one artifact copied into many worktrees - same size, NOT hashed (13 GB each set), so not claimed; same-size groups elsewhere (ckpt_after.pt x33, ckpt.pt x26, visual_ctx.npy x29) are different runs/shards, not duplicates.

## 6. Category C - probably obsolete, PI decision (script `cleanup_devbox_C.ps1`, every line commented out)

Total **292.0 GB on C:** and **688.8 GB on D:** if everything were approved - treat as a menu. "Registers" = literal mentions of the name in MODEL_REGISTRY / GOALS_AND_CLAIMS / RETRACTION_LOG.

### C1. experiment outputs on C: (157.4 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| C:\Users\Admin\refav1_trainprobe | 59.7 (59.7) | 2026-09-27 | refav1_trainprobe - fp8 token cache (36.9 GB) + 600 episodes (21.2 GB) + feats (1.3 GB), 2026-09-27 - refav1 trunk-probe working set; re-creatable from Thor via refav1_trainprobe/pull_encode.py (scp + fp8 encode); refav1 planner eliminated on the full grid (memory 2026-09-27) | only mention: memory note fp8-encoder-reads-keys-from-g (how it was built) | re-creation = scp the 600 v2ep from Thor (~16 min at the INHERITED 63 MB/s LAN rate) + re-run the fp8 encode on the dev-box GPU (time not measured) |
| C:\lgt | 56.4 (56.3) |  | C:\lgt - lander/gate scratch (2026-09-27/28): rc7cost_runs 51.9 GB + r7ldr_scratch 1.2 + 9 tree/tar scratch dirs ~0.4 GB each - obsolete experiment outputs: rc7cost_runs holds 47 per-run ckpt.pt files (51.4 GB; 26 of them exactly 1,172,814,977 B, the rest 1.17-1.18 GB; same SIZE only - content not compared, different runs) of the refcv7 restart cost study; the rest is gate/dry-run tree extracts and tars | 480 files in rc7cost_runs, 47 of them ckpt.pt; no mention in MODEL_REGISTRY / GOALS / RETRACTION_LOG; only qland docs mention C:/lgt/dry37086(_out) | per-run result JSONs are inside rc7cost_runs; if a cost-study claim cites them, bank them first. Re-creation = re-running the cost probes. |
| C:\lgs | 12.4 (12.3) |  | C:\lgs - lander gate trees (L0fcb1c1cbb, sw) + gate_devbox_* logs (09-27/28) - READINESS_LOOP.md lists "clean C:/lgs gate trees" as a post-launch to-do; launch happened 2026-09-28 | 2 trees of ~6 GB (19k + 16k files) + 8 tiny gate_devbox_<sha> result dirs (73 files each) | gate_devbox_* hold the per-commit gate verdicts (tiny) - keep those 8 dirs (~1 MB) if you delete only the two trees |
| C:\Users\Admin\rc7restart_work | 10.6 (10.6) | 2026-09-28 | rc7restart_work - restart-option A/B trials (ab/ 8.2 GB of ckpt.pt per trial, liveckpt 2.2 GB) - refcv7 restart study 2026-09-28, superseded by the 50,400-step run | 103 files in ab/; 3 files in liveckpt | liveckpt = pulled snapshots of the then-live run (HF holds 5000/20000/30000/50400 only) |
| C:\dzo | 7.8 (7.8) |  | C:\dzo - DriveZero rollouts h0920 + m-* (2026-09-20) - DZ experiment outputs; superseded runs | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 2 mentions; RETRACTION_LOG cites C:/dzo/m-nr-n and m-nr-r1 as rollout dirs of a retracted number | raw rollouts behind a retracted claim - keep m-nr-n / m-nr-r1 if the retraction evidence matters |
| C:\Users\Admin\lg0926 | 7.1 (7.1) | 2026-09-27 | lg0926 - Thor staging tars (thor_stage*/t3cca805.tar x4 ...), gate inputs, g/ _superseded - lander/Thor shipping scratch 2026-09-26/27; referenced only by finished readiness scripts | 12,058 files differ from the tree mtime; unique content not in git; run_dry_devbox.sh names IN=C:/Users/Admin/lg0926/inputs | inputs/ is read by qland/work/refcv7/probe_cost/run_dry_devbox.sh (cost probe, finished) - keep inputs/ if you re-run it |
| C:\Users\Admin\v7f_ckpt_gate | 0.95 (0.95) | 2026-09-27 | v7f_ckpt_gate - v7F gate checkpoints (09-27) - checkpoint copies, not re-creatable from git | files=294 | experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact |
| C:\Users\Admin\refav1_margin | 0.82 (0.82) | 2026-09-06 | refav1_margin - refav1 margin experiment outputs | files=599 | experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact |
| C:\Users\Admin\refav1_r1r6 | 0.78 (0.77) | 2026-09-27 | refav1_r1r6 - refav1 R1-R6 experiment outputs | files=4942 | experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact |
| C:\Users\Admin\v7f_ckpt | 0.40 (0.39) | 2026-09-27 | v7f_ckpt - v7F checkpoint staging (09-27) - checkpoint copies | files=3214 | experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact |
| C:\Users\Admin\tanitad-bench-offrepo | 0.26 (0.26) | 2026-09-26 | tanitad-bench-offrepo - bench files kept off-repo (not scanned) | files=9 | experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact |
| C:\Users\Admin\refcv4b_viz | 0.25 (0.25) | 2026-09-06 | refcv4b_viz - refcv4b visualisation outputs (not scanned file-by-file) | files=41 | experiment outputs; confirm the registry/RESULT docs do not cite them as the only raw artifact |

### C2. old datasets / staging on C: (88.6 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| C:\Users\Admin\tanitad-data\eval | 33.0 (32.9) |  | tanitad-data\eval - comma2k19-val (14.8 GB), comma_val.tgz (7.8 GB), v1_speedjerk_ckpt.pt (3.1 GB), ckpt_full.pt (2.9 GB), sitclf substrates ... - mixed old eval inputs (2026-07-07/08) | v1_speedjerk_ckpt.pt = the deployed flagship v1 (registry: flagship4b-speedjerk-30k) and ckpt_full.pt = the no-speed control: HF copies exist per memory (Sayood/tanitad-flagship-*), NOT re-verified | delete only the two .pt files after confirming HF; comma_val.tgz is probably the compressed twin of comma2k19-val-61c46fca8f7f (not hash-checked) |
| C:\Users\Admin\tanitad-data\tanitdataset | 29.7 (29.7) |  | tanitad-data\tanitdataset - HF staging copies of the comma2k19 lake (hf_stage_R, hf_stage_C, lake) - staging for HF dataset push (2026-07-21..25); the three dirs are HARDLINKED: apparent 44.5 GB, real 29.7 GB on disk | hardlinks.py: tanitdataset logical 44.5 GB vs unique inodes 29.7 GB (28 multi-link files). Deleting only one of the three frees ~0 | confirm the HF dataset it staged is published (Sayood/tanitad-comma2k19-episodes) before deleting all three |
| C:\Users\Admin\tanitad-data\rl-pilot | 15.3 (15.3) |  | tanitad-data\rl-pilot - 33 RL pilot sweep dirs with ckpt_after.pt (417 MB each), 2026-08/09 - RL pilot sweeps, superseded by ddv2rl | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 1 mentions; same-size ckpt_after.pt x33 (different runs) | rl-pilot/obstacle.offline.chunks (0.68 GB) is an input of the obstacle join - keep that subdir |
| C:\Users\Admin\tanitad-data\comma2k19 | 10.7 (10.7) |  | tanitad-data\comma2k19 - extracted Chunk_1 (10.7 GB), without the zip (listed under B-derived) - raw comma2k19 chunk; registry cites comma2k19 13x (it is training data provenance) | registers: MODEL_REGISTRY 13 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 16 mentions | the zip is listed separately as B-derived; the extracted copy is the working data |

### C3. scratch trees with unique content (C:) (40.3 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| C:\Users\Admin\ev6 | 5.8 (5.8) | 2026-09-26 | ev6 - scratch tree with unique content - unique data outputs not in git: .csv x31, .jsonl x23, .json x15, .npz x1 (all unique files incl. sources total 277.2 MB) | files=15726; newest 2026-09-26 23:09 UTC; 779 files differ from the extract mtime, 655 of them are byte-identical blobs in git; 124 NOT in git (277.18 MB: .log x43, .csv x31, .jsonl x23, .json x15, .sh x3); extra entries: .claude, .gitignore, CLAUDE.md, Media | qland/work/evalfw_restart/fallback_*.md, make_fallbacks.py and work/coord_land/watch_navhard30k.sh name C:/Users/Admin/ev6 as the CODE tree of the refcv6 eval fallback (fallback docs of 2026-09-2x). Confirm the fallback is retired before deleting. |
| C:\Users\Admin\tanitad-wt-join3d | 4.8 (4.8) | 2026-09-17 | tanitad-wt-join3d - scratch tree with unique content - top-level entries that are not part of the repo layout: .pytest_cache; unique data outputs not in git: .json x85, .jsonl x9 (all unique files incl. sources total 11.1 MB) | files=14159; newest 2026-09-17 08:26 UTC; 265 files differ from the extract mtime, 21 of them are byte-identical blobs in git; 244 NOT in git (11.12 MB: .py x89, .json x85, .md x25, .log x15, .sh x12); extra entries: .claude, .gitignore, CLAUDE.md, Media |  |
| C:\Users\Admin\tanitad-wt-percep-v6 | 4.8 (4.8) | 2026-09-16 | tanitad-wt-percep-v6 - scratch tree with unique content - unique data outputs not in git: .json x2 (all unique files incl. sources total 1.6 MB) | files=13845; newest 2026-09-16 22:20 UTC; 47 files differ from the extract mtime, 9 of them are byte-identical blobs in git; 38 NOT in git (1.60 MB: .py x27, .md x8, .json x2, .patch x1); extra entries: .claude, .gitignore, CLAUDE.md, Media |  |
| C:\Users\Admin\ygate_fix | 3.2 (3.2) | 2026-09-26 | ygate_fix - scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or tanitad-push) | files=16435; has a .git dir (~1.2 GB); newest file 2026-09-26 12:40 UTC; 3 mtime-anomalous files, 3 of them byte-identical blobs in git; extra entries: .claude, .gitignore, CLAUDE.md, Media; mentioned by qland/work/yawfix/build_ygate.py (generator script) |  |
| C:\Users\Admin\ygate_tip | 3.2 (3.2) | 2026-09-26 | ygate_tip - scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or tanitad-push) | files=16434; has a .git dir (~1.2 GB); newest file 2026-09-26 12:35 UTC; 0 mtime-anomalous files, 0 of them byte-identical blobs in git; extra entries: .claude, .gitignore, CLAUDE.md, Media; mentioned by qland/work/yawfix/build_ygate.py (generator script) |  |
| C:\Users\Admin\cfull_cand | 3.2 (3.2) | 2026-09-26 | cfull_cand - scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or tanitad-push); unique data outputs not in git: .json x1 (all unique files incl. sources total 0.2 MB) | files=17214; has a .git dir (~1.2 GB); newest file 2026-09-26 11:58 UTC; 513 mtime-anomalous files, 508 of them byte-identical blobs in git; 5 NOT in git (0.21 MB: .py x4, .json x1); extra entries: .claude, .gitignore, CLAUDE.md, Media; mentioned by qland/work/coord_land/compare_stack.json (log record) |  |
| C:\Users\Admin\cfull_tip | 3.1 (3.0) | 2026-09-26 | cfull_tip - scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or tanitad-push) | files=16672; has a .git dir (~1.2 GB); newest file 2026-09-26 09:26 UTC; 0 mtime-anomalous files, 0 of them byte-identical blobs in git; extra entries: .claude, .gitignore, CLAUDE.md, Media; mentioned by qland/work/coord_land/compare_*.json (log record) |  |
| C:\Users\Admin\bxh_work | 2.7 (2.7) | 2026-09-27 | bxh_work - scratch tree with unique content - top-level entries that are not part of the repo layout: a17, bench, code, diag, hqs, overlay; unique data outputs not in git: .parquet x42, .tar x27, .md5 x14, .json x12, .list x2, .done x2 (all unique files incl. sources total 2887.4 MB) | files=1885; newest 2026-09-27 15:03 UTC; 559 files differ from the extract mtime, 291 of them are byte-identical blobs in git; 268 NOT in git (2887.43 MB: .py x109, .parquet x42, .txt x29, .tar x27, .log x18) |  |
| C:\Users\Admin\v7f_gate | 2.7 (2.7) | 2026-09-27 | v7f_gate - scratch tree with unique content - top-level entries that are not part of the repo layout: argv, gate_runs, raw, scratch, tmp_keys, tree; unique data outputs not in git: .json x1014, .pt x105, .jsonl x67, .csv x29, .svg x19 (all unique files incl. sources total 2172.4 MB) | files=6385; newest 2026-09-27 15:35 UTC; 3470 files differ from the extract mtime, 459 of them are byte-identical blobs in git; 3011 NOT in git (2172.43 MB: .py x1504, .json x1014, .pt x105, .sh x82, .jsonl x67) |  |
| C:\Users\Admin\v7f_r1r4 | 0.89 (0.88) | 2026-09-27 | v7f_r1r4 - scratch tree with unique content - top-level entries that are not part of the repo layout: baseline, deliv, mut, raw, smoke; unique data outputs not in git: .json x690, .jsonl x8 (all unique files incl. sources total 71.9 MB) | files=8156; newest 2026-09-27 13:07 UTC; 3496 files differ from the extract mtime, 285 of them are byte-identical blobs in git; 3211 NOT in git (71.88 MB: .py x2236, .json x690, .sh x146, .md x56, .log x21) |  |
| C:\Users\Admin\v7f_r3 | 0.81 (0.81) | 2026-09-27 | v7f_r3 - scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or tanitad-push); top-level entries that are not part of the repo layout: deliverable, env.sh, proof, raw; unique data outputs not in git: .json x55, .sample x14, .pt x11, .jsonl x3 (all unique files incl. sources total 456.4 MB) | files=3252; has a .git dir (~1.2 GB); newest file 2026-09-27 13:41 UTC; 161 mtime-anomalous files, 20 of them byte-identical blobs in git; 141 NOT in git (456.40 MB: .json x55, .log x23, .txt x22, .sample x14, .pt x11) |  |
| C:\Users\Admin\tipsnap | 0.65 (0.64) | 2026-09-27 | tipsnap - scratch tree with unique content - top-level entries that are not part of the repo layout: b3f7ea6f, c36b6ddd; unique data outputs not in git: .json x645, .csv x29, .jsonl x20, .svg x19 (all unique files incl. sources total 121.4 MB) | files=6727; newest 2026-09-27 12:33 UTC; 2869 files differ from the extract mtime, 440 of them are byte-identical blobs in git; 2429 NOT in git (121.44 MB: .py x1482, .json x645, .sh x82, .md x51, .log x49) |  |
| C:\Users\Admin\v7f_merge | 0.42 (0.41) | 2026-09-27 | v7f_merge - scratch tree with unique content - top-level entries that are not part of the repo layout: _tip_eff.py, apply_f2_gate_test.py, apply_f2_strict.py, apply_f7_f8.py, apply_navcond_map.py, bank_merge.py; unique data outputs not in git: .json x328, .csv x29, .svg x19, .jsonl x15, .html x7 (all unique files incl. sources total 89.2 MB) | files=3785; newest 2026-09-27 15:52 UTC; 1171 files differ from the extract mtime, 328 of them are byte-identical blobs in git; 843 NOT in git (89.21 MB: .py x342, .json x328, .log x42, .csv x29, .txt x26) |  |
| C:\Users\Admin\ev6_battery | 0.40 (0.39) | 2026-09-27 | ev6_battery - scratch tree with unique content - top-level entries that are not part of the repo layout: RESULT.md, RESULT_DRAFT_METHOD.md, SPEC.md, code, raw; not a git-archive extract (6195 mtime-anomalous files) | files=6528; newest 2026-09-27 00:49 UTC; 6195 files differ from the extract mtime, 0 of them are byte-identical blobs in git |  |
| C:\Users\Admin\nb2_tree_2301 | 0.39 (0.39) | 2026-09-27 | nb2_tree_2301 - scratch tree with unique content - top-level entries that are not part of the repo layout: _pkg | files=3324; newest 2026-09-27 16:40 UTC; 232 files differ from the extract mtime, 188 of them are byte-identical blobs in git; 44 NOT in git (2.97 MB: .py x27, .log x14, .txt x2, .diff x1); extra entries: _regress_final_cand.log, _t_all_a6.log, _t_bt.log, _t_f_a6.log |  |
| C:\Users\Admin\nb2_tree_2128 | 0.39 (0.38) | 2026-09-26 | nb2_tree_2128 - scratch tree with unique content - top-level entries that are not part of the repo layout: _out, _pkg; unique data outputs not in git: .json x3, .npz x1 (all unique files incl. sources total 5.8 MB) | files=3135; newest 2026-09-26 21:20 UTC; 64 files differ from the extract mtime, 12 of them are byte-identical blobs in git; 52 NOT in git (5.76 MB: .py x39, .json x3, .log x3, .txt x3, .npz x1) |  |
| C:\Users\Admin\v7f_r3_base | 0.39 (0.38) | 2026-09-27 | v7f_r3_base - scratch tree with unique content - unique data outputs not in git: .json x1 (all unique files incl. sources total 0.1 MB) | files=3093; newest 2026-09-27 13:00 UTC; 2 files differ from the extract mtime, 0 of them are byte-identical blobs in git; 2 NOT in git (0.07 MB: .md x1, .json x1) |  |
| C:\Users\Admin\nb2ml_tree_3708 | 0.38 (0.37) | 2026-09-27 | nb2ml_tree_3708 - scratch tree with unique content - unique data outputs not in git: .json x1 (all unique files incl. sources total 0.5 MB) | files=2998; newest 2026-09-27 14:45 UTC; 9 files differ from the extract mtime, 4 of them are byte-identical blobs in git; 5 NOT in git (0.51 MB: .py x3, .log x1, .json x1); extra entries: _t_gate.log |  |
| C:\Users\Admin\fixtree | 0.38 (0.37) | 2026-10-04 | fixtree - scratch tree with unique content - modified within the last 72 h (newest file 2026-10-04 01:44 UTC): may still be in use by the lander | files=2909; newest 2026-10-04 01:44 UTC; 16 files differ from the extract mtime, 14 of them are byte-identical blobs in git; 2 NOT in git (0.01 MB: .txt x2); extra entries: relevant_tests.txt, relevant_tests_edited.txt |  |
| C:\Users\Admin\ygate_fixes | 0.38 (0.37) | 2026-10-04 | ygate_fixes - scratch tree with unique content - modified within the last 72 h (newest file 2026-10-04 02:38 UTC): may still be in use by the lander | files=2907; newest 2026-10-04 02:38 UTC; 14 files differ from the extract mtime, 14 of them are byte-identical blobs in git |  |
| C:\Users\Admin\fixtree_tip | 0.37 (0.37) | 2026-10-04 | fixtree_tip - scratch tree with unique content - modified within the last 72 h (newest file 2026-10-04 01:02 UTC): may still be in use by the lander | files=2902; newest 2026-10-04 01:02 UTC; 3 files differ from the extract mtime, 0 of them are byte-identical blobs in git; 3 NOT in git (0.00 MB: .log x1, .txt x1, .sh x1); extra entries: baseline_stack.log, relevant_tests.txt, run_baseline.sh |  |
| C:\Users\Admin\refcv4b_repo | 0.34 (0.32) | 2026-09-06 | refcv4b_repo - repo snapshot of 2026-09-04 (refcv4b landing) with result files and a <token-file> copy | files=8283; newest 2026-09-06 11:35 UTC; 7614 files differ from the extract mtime, 0 of them are byte-identical blobs in git; extra entries: CLAUDE.md, Media, README.md | CONTAINS A COPY OF <token-file> (secrets file, seen as a top-level entry; contents NOT read). Delete this dir for credential hygiene regardless of the rest; never commit/zip it. Not in git (7,614 mtime-anomalous files). |
| C:\Users\Admin\ygate_video | 0.27 (0.27) | 2026-10-03 | ygate_video - scratch tree with unique content - modified within the last 72 h (newest file 2026-10-03 23:59 UTC): may still be in use by the lander | files=1417; newest 2026-10-03 23:59 UTC; 3 files differ from the extract mtime, 3 of them are byte-identical blobs in git |  |
| C:\Users\Admin\dz | 0.20 (0.20) | 2026-09-26 | dz - scratch tree with unique content - .git present (scratch `git init` repo, single synthetic commit not found in D:/Projects/TanitAD/.git or tanitad-push); top-level entries that are not part of the repo layout: DriveZero, dblinks, out; unique data outputs not in git: .tsv x11, .temp x5, .xz x4, .nuboard x3, .yaml x3 (all unique files incl. sources total 96.7 MB) | files=1006; has a .git dir (~1.2 GB); newest file 2026-09-26 09:56 UTC; 131 mtime-anomalous files, 17 of them byte-identical blobs in git; 114 NOT in git (96.70 MB: .log x77, .tsv x11, .txt x8, .temp x5, .xz x4) |  |

### C4. old worktrees / mirrors (42.5 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| D:\Projects\TanitAD\.claude\worktrees | 21.0 (5.9) | 2026-09-20 | D:\Projects\TanitAD\.claude\worktrees - agent worktrees admiring-hamilton-6a722d, upbeat-lamport-d8e99a (2026-09-20) - git worktrees registered in the repo (`git worktree list`); 16,342 files; stale since 2026-09-20 | upbeat-lamport is at the same commit as the repo tip 37645fcc | use `git worktree remove` (not Remove-Item) so .git/worktrees is cleaned; check no session is attached |
| D:\Projects\TanitAD-wt-1024 | 15.7 (4.7) |  | D:\Projects\TanitAD-wt-1024 - agent worktree agent/dataeng-256x1024-20260916 - registered git worktree, stale since 2026-09-16 |  | use `git worktree remove`; confirm branch agent/dataeng-256x1024-20260916 is merged/pushed |
| C:\Users\Admin\tanitad-wt | 3.1 (3.1) | 2026-09-10 | tanitad-wt (without _s2build, which is listed under B-held) - old worktree (2026-07/08) incl. stack/experiments/*/model.pt, _s2build (B-held), v3b1smoke | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 4 mentions; 3,602 mtime-anomalous files, 1,941 byte-identical to git, 1,661 not (6.5 GB) | _s2build inside is the input of live qland harness scripts (see B-held); delete tanitad-wt only after repointing them |
| C:\Users\Admin\tanitad-mirror | 2.7 (2.7) | 2026-08-29 | tanitad-mirror - old repo mirror with stack/experiments model.pt files (2026-08-27) - mirror; not synced since 2026-08-29 | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 8 mentions; 614 files / 2.1 GB not in git (model.pt p0-sA01/sB00 ... same 1.05 GB files also exist x12 in devbox-C\tanitad-lwbase etc.) | RETRACTION_LOG cites it 8x as a comparison tree (v6.py md5 table) - a retraction record, not live code |

### C6. D:\Archive\devbox-C - junction-backed dirs moved from C: on 2026-09-13 (264.1 GB)

Purpose is inferred from the directory name and file types unless a register mentions it. Register mentions are literal-name counts, so a 0 means the name is not cited, not that the numbers it produced are unused.

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| D:\Archive\devbox-C\carla | 53.2 (25.3) | 2026-07-08 | D:\Archive\devbox-C\carla (+ junction C:\Users\Admin\carla) - CARLA 0.9 sim install + data (2025-06..2026-07) | newest file 2026-07-08; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\wpd-probe | 31.0 (30.1) | 2026-09-10 | D:\Archive\devbox-C\wpd-probe (+ junction C:\Users\Admin\wpd-probe) - wpd probe banks (tok_D0/D1/D2.npy 2.98 GB each x2 copies) | newest file 2026-09-10; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\dkhead_run | 18.8 (15.9) | 2026-09-06 | D:\Archive\devbox-C\dkhead_run (+ junction C:\Users\Admin\dkhead_run) - dkhead run outputs | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\refav1_eval_full | 14.0 (13.8) | 2026-09-06 | D:\Archive\devbox-C\refav1_eval_full (+ junction C:\Users\Admin\refav1_eval_full) - refav1 full-eval episodes (13.8 GB); same size as C:\tanitad-data\refav1-eval141 (13.8 GB) - not hash-compared | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\refav1_probe | 11.8 (5.9) | 2026-09-27 | D:\Archive\devbox-C\refav1_probe (+ junction C:\Users\Admin\refav1_probe) - refav1 probe runs | newest file 2026-09-27; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\run_refcv4 | 11.0 (5.8) | 2026-09-04 | D:\Archive\devbox-C\run_refcv4 (+ junction C:\Users\Admin\run_refcv4) - refcv4 run (2026-09-04) | newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\wpa-readout | 10.5 (8.2) | 2026-09-08 | D:\Archive\devbox-C\wpa-readout (+ junction C:\Users\Admin\wpa-readout) - wpa readout banks (tok.npy 2.2 GB x2) | newest file 2026-09-08; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\refav1_eval_slice | 8.3 (7.8) | 2026-09-04 | D:\Archive\devbox-C\refav1_eval_slice (+ junction C:\Users\Admin\refav1_eval_slice) - refav1 eval slice | newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\navcomp | 7.9 (6.2) | 2026-09-05 | D:\Archive\devbox-C\navcomp (+ junction C:\Users\Admin\navcomp) - NavSim comparison runs | newest file 2026-09-05; registers: MODEL_REGISTRY 2 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\p1gate | 7.1 (7.0) | 2026-09-06 | D:\Archive\devbox-C\p1gate (+ junction C:\Users\Admin\p1gate) - P1 gate runs | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\rlgate2 | 7.1 (7.0) | 2026-09-06 | D:\Archive\devbox-C\rlgate2 (+ junction C:\Users\Admin\rlgate2) - RL gate 2 runs | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\run_refcv3_ol | 6.7 (5.4) | 2026-09-03 | D:\Archive\devbox-C\run_refcv3_ol (+ junction C:\Users\Admin\run_refcv3_ol) - refcv3 open-loop run | newest file 2026-09-03; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\run_refcv4v | 6.0 (4.0) | 2026-09-04 | D:\Archive\devbox-C\run_refcv4v (+ junction C:\Users\Admin\run_refcv4v) - refcv4 variant run | newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-bench | 5.6 (3.4) | 2026-09-04 | D:\Archive\devbox-C\tanitad-bench (+ junction C:\Users\Admin\tanitad-bench) - bench worktree (refav1_eval_21109 ckpts) | newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\rl_refcv3_min | 5.2 (5.0) | 2026-09-05 | D:\Archive\devbox-C\rl_refcv3_min (+ junction C:\Users\Admin\rl_refcv3_min) - RL refcv3-min | newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 9 / RETRACTION_LOG 3 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-lwbase | 5.0 (2.3) | 2026-09-06 | D:\Archive\devbox-C\tanitad-lwbase (+ junction C:\Users\Admin\tanitad-lwbase) - label-wire base worktree (12 x 1.05 GB model.pt copies) | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-labelwire | 5.0 (2.3) | 2026-09-06 | D:\Archive\devbox-C\tanitad-labelwire (+ junction C:\Users\Admin\tanitad-labelwire) - label-wire worktree (same model.pt copies) | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-refcv4b-analyze | 4.8 (2.2) | 2026-09-06 | D:\Archive\devbox-C\tanitad-refcv4b-analyze (+ junction C:\Users\Admin\tanitad-refcv4b-analyze) - refcv4b analysis worktree | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-evtool-20260906 | 4.7 (2.1) | 2026-09-06 | D:\Archive\devbox-C\tanitad-evtool-20260906 (+ junction C:\Users\Admin\tanitad-evtool-20260906) - evtool worktree | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-perbatch | 4.5 (2.1) | 2026-09-07 | D:\Archive\devbox-C\tanitad-perbatch (+ junction C:\Users\Admin\tanitad-perbatch) - per-batch worktree | newest file 2026-09-07; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\run_paired | 4.4 (2.1) | 2026-09-03 | D:\Archive\devbox-C\run_paired (+ junction C:\Users\Admin\run_paired) - paired runs | newest file 2026-09-03; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-ctg | 4.3 (2.1) | 2026-09-06 | D:\Archive\devbox-C\tanitad-ctg (+ junction C:\Users\Admin\tanitad-ctg) - ctg worktree | newest file 2026-09-06; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\veto_run | 4.3 (3.7) | 2026-09-05 | D:\Archive\devbox-C\veto_run (+ junction C:\Users\Admin\veto_run) - veto runs | newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 2 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\run_refcv3_viz | 4.2 (1.3) | 2026-09-04 | D:\Archive\devbox-C\run_refcv3_viz (+ junction C:\Users\Admin\run_refcv3_viz) - refcv3 viz run | newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tsc2 | 4.1 (3.2) | 2026-09-02 | D:\Archive\devbox-C\tsc2 (+ junction C:\Users\Admin\tsc2) - tsc2 runs | newest file 2026-09-02; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\run_wbank | 3.9 (2.5) | 2026-09-05 | D:\Archive\devbox-C\run_wbank (+ junction C:\Users\Admin\run_wbank) - weight-bank runs | newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\nuscenes | 2.9 (2.9) | 2026-09-26 | D:\Archive\devbox-C\nuscenes (nuScenes archives, 2.9 GB) - off-repo, one copy, re-fetchable (qland r_d.md: "accepted_terms_by: Sayed") | no junction | re-fetch needs the nuScenes licence acceptance again |
| D:\Archive\devbox-C\refcv4b_egodrop | 2.7 (1.6) | 2026-09-05 | D:\Archive\devbox-C\refcv4b_egodrop (+ junction C:\Users\Admin\refcv4b_egodrop) - refcv4b ego-drop runs | newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\rl_rescope | 2.0 (1.6) | 2026-09-05 | D:\Archive\devbox-C\rl_rescope (+ junction C:\Users\Admin\rl_rescope) - RL rescope | newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\refav1_drive | 1.6 (1.4) | 2026-09-05 | D:\Archive\devbox-C\refav1_drive (+ junction C:\Users\Admin\refav1_drive) - refav1 drive outputs | newest file 2026-09-05; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 3 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |
| D:\Archive\devbox-C\tanitad-egodrop | 1.2 (1.2) | 2026-09-04 | D:\Archive\devbox-C\tanitad-egodrop (+ junction C:\Users\Admin\tanitad-egodrop) - ego-drop checkpoint | newest file 2026-09-04; registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions | finished-experiment raw artifacts (refav1/refcv3/refcv4 era); the junction on C: must go first (the script line does it) |

### C7. archives of removed originals (thor_archive, _c) (286.3 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| D:\thor_archive | 175.2 (107.9) |  | D:\thor_archive\2026-09-14 (alpasim 60.3, sam3map 47.2, models 16.6, nurec_scenes 7.9, ckpt_snaps 6.9, experiments 6.6, backup 6.2, valdata 5.6, trt_* ~14 GB) - Thor content archived to the dev box and then REMOVED FROM THOR on 2026-09-14 (sam3-only-road-map RESULT.md: md5-verified, then deleted on Thor) | this may be the ONLY copy of those Thor files; alpasim = 60,998 files / 2.8 GB real (57 GB is 1 MiB-cluster slack) | irreversible if Thor no longer has them. Cheaper first step: pack alpasim/ into one tar (frees ~57 GB, loses nothing) |
| D:\Projects\TanitAD\_c | 111.0 (110.9) |  | D:\Projects\TanitAD\_c - 184 tar.zst archives of C: worktrees/scratch trees compacted on 2026-09-27 (22 of them 4.3-5.0 GB) - archives of dirs that were deleted from C: after sha256+CRC verification (MANIFEST.tsv, MOVE_LOG.txt) | each archive has a sha256 in _c\MANIFEST.tsv; the originals no longer exist | these ARE the only copies of those old worktrees (git history is in .git, so tracked content is recoverable; untracked outputs inside are not) |

### C8. D:\Projects\TanitAD-artifacts (71.3 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl | 10.8 (10.7) |  | TanitAD-artifacts\v2ep-eval139-416x1024cyl - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); eval episode cache 416x1024 cylindrical; re-creatable from the Thor v2ep + build scripts | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 1 / RETRACTION_LOG 0 mentions; named as an input by qland/a3_halfA_maphead.sh, a3_heldout_read.sh, a8_occupancy_5k.sh (one-shots of 2026-09-19) | KEEP-area one-shot scripts name it - delete only if the A-series (A1-A9) panels will not be re-run |
| D:\Projects\TanitAD-artifacts\v2ep-eval139-408x1024cyl | 10.7 (10.6) |  | TanitAD-artifacts\v2ep-eval139-408x1024cyl - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); eval episode cache 408x1024; re-creatable from the Thor v2ep + build scripts | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions; not named by any KEEP-area script | low |
| D:\Projects\TanitAD-artifacts\refcv5cmp | 8.3 (5.6) |  | TanitAD-artifacts\refcv5cmp (out/dumps/npz comparison outputs, 8.3 GB on disk / 5.6 GB real) - refcv5 comparison outputs (2026-09-07); 2,971 files | the live NavSim package mentions refcv5cmp (C:\Users\Admin\refcv5cmp is the working copy) | separate from C:\Users\Admin\refcv5cmp (KEEP, referenced) |
| D:\Projects\TanitAD-artifacts\v2ep-eval139-256x1024cyl | 7.6 (7.5) |  | TanitAD-artifacts\v2ep-eval139-256x1024cyl - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); eval episode cache 256x1024; re-creatable from the Thor v2ep + build scripts | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions; not named by any KEEP-area script | low |
| D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl-splitA | 5.6 (5.5) |  | TanitAD-artifacts\v2ep-eval139-416x1024cyl-splitA - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); split A of the 416x1024 cache; re-creatable from the Thor v2ep + build scripts | no register mention; named only by qland/a0_clean124_split.py (the generator of the halves) | low |
| D:\Projects\TanitAD-artifacts\hf-corpus-aug-20260915 | 5.5 (4.9) |  | TanitAD-artifacts\hf-corpus-aug-20260915 (HF corpus augmentation staging, 5.5 GB) - staging dir of the 2026-09-15 augmented-corpus HF push (1,128 files) | HF push assumed done (not re-verified) | verify the HF dataset before deleting |
| D:\Projects\TanitAD-artifacts\v2ep-eval139-416x1024cyl-splitB | 5.2 (5.2) |  | TanitAD-artifacts\v2ep-eval139-416x1024cyl-splitB - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); split B of the 416x1024 cache; re-creatable from the Thor v2ep + build scripts | no register mention; named only by qland/a0_clean124_split.py (the generator of the halves) | low |
| D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB | 4.9 (4.8) |  | TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); clean-124 half B; re-creatable from the Thor v2ep + build scripts | no register mention; named as an input by qland/a1_per_half_floor.py, a2_harness_smoke.sh, a3_*.sh, a8_occupancy_5k.sh (one-shots of 2026-09-19) | KEEP-area one-shot scripts name it - delete only if the A-series (A1-A9) panels will not be re-run |
| D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfA | 4.8 (4.8) |  | TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfA - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); clean-124 half A; re-creatable from the Thor v2ep + build scripts | no register mention; named as an input by qland/a1_per_half_floor.py, a2_harness_smoke.sh, a3_*.sh, a8_occupancy_5k.sh (one-shots of 2026-09-19) | KEEP-area one-shot scripts name it - delete only if the A-series (A1-A9) panels will not be re-run |
| D:\Projects\TanitAD-artifacts\v2ep-eval139-build | 4.0 (3.8) |  | TanitAD-artifacts\v2ep-eval139-build - eval episode cache (`v2ep`) for the map-head/occupancy panels (2026-09-15..19); build workspace of the eval139 caches; re-creatable from the Thor v2ep + build scripts | no register mention; not named by any KEEP-area script | low |
| D:\Projects\TanitAD-artifacts\refe-final-2026-10-01 | 2.7 (2.4) |  | TanitAD-artifacts\refe-final-2026-10-01 (2.7 GB) - REFe final model bundle of 2026-10-01 (model_final.pt 1.29 GB also at data\refe_runs_eval\model_final.pt) | not named in the live upload script (it names refe-pod-final-2026-10-04 and refe-onpolicy-2026-10-01) | REFe session is active - ask it before touching anything named refe-* |
| D:\Projects\TanitAD-artifacts\hf-refcv5v2 | 1.2 (1.2) |  | TanitAD-artifacts\hf-refcv5v2 (HF staging of refcv5-v2, 1.2 GB) - HF push staging (also packed in _c\hf-refcv5v2.tar.zst) | overlaps refcv5v2_final/ckpt.pt (1.3 GB) |  |

### C9. eval kits (30.3 GB)

| path | GB (real) | newest | what / reason | evidence | risk of deleting |
|---|---|---|---|---|---|
| D:\refcv6_eval_kit | 28.5 (28.3) |  | D:\refcv6_eval_kit (data 23.7, ckpt 4.8 -> 3.6 after the B duplicate, ckpt_final 1.1 GB) - refcv6 stopped 2026-09-26; eval kit data (refcv6-b1-416x1024-eval139 caches) + step ckpts | registers: MODEL_REGISTRY 0 / GOALS_AND_CLAIMS 0 / RETRACTION_LOG 0 mentions; qland hfpush_refcv6.py + a16_switch meta name it; ckpt_step38000 md5 5a2e7222... verified = recorded; HF private Sayood/tanitad-refcv6-r101-s0 (not re-checked) | delete data/ + ckpt/ first; keep ckpt_final/ (the stopped checkpoint) until the HF copy is verified |
| D:\sam3map_lidar | 1.8 (1.8) |  | D:\sam3map_lidar (2 dirs e38030ef, 54330db8; 192 files, 1.8 GB) - SAM3 map LiDAR spin decodes (decode_sweeps.py output, 2026-09-13) | only mention: sam3-only-road-map decode_sweeps.py | SAM3 map GT is a PI-led direction (memory 2026-09-13); small |

## 7. Category D - keep (live, needed, or not ours)

### 7.1 Live / needed

| path | GB on disk | why it is KEEP |
|---|---|---|
| C:\Users\Admin\tanitad-caches | 116.4 | KEEP (brief): live caches; 3 junctions to D:\Archive\devbox-C\tanitad-caches; navsim package names refcv6-navsim-20260923 |
| C:\Users\Admin\qland | 22.7 | KEEP (brief): the lander + work (waiter_final.sh / launch_waiter_cdisk.sh are running from qland/work/refcv7) |
| C:\Users\Admin\navsim-crun | 21.6 | KEEP (brief): NavSim devkit + metric caches + data; the live run uses navsim-crun/venv/Scripts/python.exe |
| C:\Users\Admin\venvs | 17.3 | KEEP (brief): venvs\tanitad (all live jobs), colab (MCP), driverl-eval 6.4 GB, carla312 4.6 GB, sam3run (the last three are candidates for the PI, not touched) |
| C:\Users\Admin\tanitad-push | 6.4 | KEEP (brief): git dir of the lander (49k loose objects, 6.4 GB) |
| C:\Users\Admin\ev7 | 0.4 | KEEP (brief): ev7/stack + ev7/taniteval on PYTHONPATH of the waiter chain |
| C:\Users\Admin\ev7nav | 0.6 | KEEP (brief): NavSim package PYTHONPATH |
| C:\Users\Admin\tanitad-media-backup | 0.8 | KEEP (brief): the Media backup (manifest relies on it) |
| C:\Users\Admin\tanitad-data\physicalai | 140.2 | canonical corpus (camera 65 GB, _epcache 54.7 GB, labels 14 GB, r0 6.2 GB): parity corpus; named by qland scripts (labels/obstacle_offline_b1eval, camera_front_wide_120fov) |
| C:\Users\Admin\tanitad-data\refav1-eval141 | 13.8 | 141 B1 v7.2 EVAL clips: "the only dev-box v2ep set with poses..." (PREREG_DDV2_RL_VALIDATION); junction target of C:\Users\Admin\refcv5cmp\eval |
| C:\Users\Admin\tanitad-data\joins | 1.7 | agent joins (train2400_agents.jsonl.xz) - parity-line join, cited in memory agent-joins-two-corpus-lines |
| C:\Users\Admin\tanitad-data\models | 1.2 | refc-base-30k/ckpt.pt is the file p_rc21_chain.sh waits on (md5 8f10d6f..) |
| C:\Users\Admin\a40-rescue | 0.3 | the B1 join (b1_train_plus_eval_agents.jsonl.xz) lives here; a near-identical mirror is in TanitAD-artifacts\a40-rescue (differs by 1 meta.json) |
| C:\Users\Admin\refcv5cmp | 0.5 | working copy named by qland keep_realdata.py / t1_rolls.sh; the live NavSim package mentions it; contains junction eval -> tanitad-data\refav1-eval141\eps |
| C:\Users\Admin\.claude | 5.3 | Claude Code session transcripts/projects (5.3 GB) - needed to resume sessions; not ours to prune blindly |
| C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\7c618562-39eb-4a61-af81-755048086a13 | 5.3 | THIS session (the Master Mind): scratchpad 5.3 GB - prune when the session ends |
| C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD | 0.0 | REFe session scratch (live uploads running from here) - brief says do not touch |
| C:\Users\Admin\AppData\Local\Temp\claude\D--Projects-TanitAD\91effc67-8c1e-4b66-9a63-341a3109dfc1 | 8.1 | recent (newest file 2026-10-01 23:08 UTC, 2.4 days): session scratchpad of an older D:-rooted session - looks stale but inside the 3-day window; re-check after 2026-10-04 23:10 UTC |
| C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\bd7d00af-b98e-42f1-a53c-cb4113059b0f | 3.5 | recent (newest file 2026-10-01 18:28 UTC, 2.6 days): session scratchpad (REFe data ruling session) - re-check after 2026-10-04 18:30 UTC |
| C:\Users\Admin\.cache\huggingface\hub | 5.5 | hub: models--facebook--sam3 3.2, dinov3-vitl16 1.1, dinov3-vitb16 0.3, dinov2-base 0.3, timm resnet101/34 - USED by repo code with HF_HUB_OFFLINE=1 (refcv6_loader.py ..); the two dataset entries are listed under A |
| D:\Archive\devbox-C\navsim | 273.9 | live NavSim run reads D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs, exp/refcv6_navtest416/frame_bank, data/openscene/navsim_logs/test; C:\Users\Admin\navsim is a junction to it |
| D:\Archive\devbox-C\tanitad-caches | 26.9 | exactly the 3 junction targets of C:\Users\Admin\tanitad-caches\{encloc-20260818, mm-e19-assets-20260901, sp2-lead130-20260818} |
| D:\refcv7_eval_kit | 5.9 | KEEP (brief): live (ckpt_50400.pt in the running NavSim job; ckpt_5000/30000 named by the package; HF holds 5000/20000/30000/50400) |
| D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-09-28-refcv7-standard-tests | 1.8 | KEEP (brief): live NavSim package (1.8 GB) |
| D:\Projects\TanitAD\data\nuplan | 351.7 | DZ-11 nuPlan val/test dbs (extracted nuplan-v1.1 338 GB + dblinks 13.4 GB + 3 zips 187.6 GB listed under B-derived): needed for the headline reproduction on the 4060 (memory refe-training-data-on-a40-pod) |
| D:\Projects\TanitAD\data\nuplan-camera | 100.5 | nuPlan camera chunks for DZ-11 (mini 51 GB zips, scenarios 25 GB on disk / 5 GB real, scenarios4 18 GB, probe 5.3 GB on disk / 1.1 real) |
| D:\Projects\TanitAD\data\refe_proxy | 39.6 | REFe proxy data (cache_train_ep015 13.3, pixels 8.5, cache_eval 8.3, pod_extract_part0-3.tar 7.8 GB) - REFe session active; the 4 tars are probably extract leftovers (ask REFe) |
| D:\Projects\TanitAD\data\refe_navtest | 21.1 | REFe navtest (modified 2026-10-04 04:18 UTC) - live |
| D:\Projects\TanitAD\data\refe_trunk_cache | 9.0 | REFe trunk cache |
| D:\Projects\TanitAD\data\refe_m5 | 8.9 | REFe m5 (8,116 small files: 8.9 GB on disk / 2.5 GB real) |
| D:\Projects\TanitAD\data\refe_runs_eval | 2.3 | REFe eval runs (model_final.pt 1.29 GB duplicates TanitAD-artifacts\refe-final-2026-10-01\model_final.pt) |
| D:\Projects\TanitAD\data\refe_navtrain10 | 2.2 | REFe navtrain10 |
| D:\Projects\TanitAD\data\openscene_probe | 4.2 | OpenScene probe 4.2 GB (2 files, 2026-09-22) - probably obsolete, ask REFe/PI |
| D:\Projects\TanitAD\data\nuplan-maps | 2.2 | nuPlan maps (2.2 GB) |
| D:\Projects\TanitAD\data\backbones | 1.5 | backbone weights (1.5 GB) |
| D:\Projects\TanitAD-artifacts\refe-pod-final-2026-10-04 | 6.9 | live: source of the REFe HF upload (push_refe_pod_final.py reads E:/Projects/TanitAD-artifacts/refe-pod-final-2026-10-04) |
| D:\Projects\TanitAD-artifacts\refe-onpolicy-2026-10-01 | 3.0 | live: source of the REFe HF upload (push_refe_pod_final.py reads .../refe-onpolicy-2026-10-01) |
| D:\Projects\TanitAD-artifacts\refcv4b_final | 1.6 | surviving copy of the B pair (md5-verified) |
| D:\Projects\TanitAD-artifacts\refcv3_final | 1.6 | surviving copy of the B pair (md5-verified) |
| D:\Projects\TanitAD-artifacts\refcv5v2_final | 1.3 | surviving copy of the B-held pair (md5-verified) |
| D:\Projects\TanitAD-artifacts\_s2build-copy-20260919 | 3.9 | surviving copy of the B-held pair (md5-verified) |
| D:\venvs | 3.4 | PI-approved opencv-ref-454 venv (PI_DECISION_QUEUE 23a APPROVED and EXECUTED); 3.4 GB on disk for 0.17 GB real (3,364 small files) |
| D:\Projects\TanitAD\.git | 48.6 | the repo object store: 48.1 GB on disk but only 7.4 GB of content - see "repack" in the doc (not a delete) |

### 7.2 Observations INSIDE the KEEP set (owner decision, not proposed as deletions)

| path | GB on disk | observation |
|---|---|---|
| D:\Archive\devbox-C\navsim\venv | 46.5 | second NavSim venv; 46.5 GB on disk for 2.2 GB real (see Surprises); the live run uses `navsim-crun\venv`, but `C:\Users\Admin\navsim\venv` is still named in 4 briefs/fallbacks - repoint them, then it can go (or tar it) |
| D:\Archive\devbox-C\navsim\data | 173.3 | NavSim data (openscene logs/maps, 173 GB on disk); `C:\Users\Admin\navsim-crun\data` (14.2 GB) is the live run's own copy; maps (map.gpkg 524 MB, nuplan-maps zip 971 MB) also exist in `D:\Projects\TanitAD\data\nuplan-maps` |
| D:\Archive\devbox-C\navsim\exp | 52.4 | 52.4 GB of NavSim experiment outputs; newest file 2026-10-04 05:49 UTC (live writer) |
| C:\Users\Admin\tanitad-caches\bevhead-20260913 | 63.0 | BEV-head token/pixel caches (tokens_s32_fp16.npy 6.2 GB x4, pix64_u8.npy 2.5 GB x2) |
| C:\Users\Admin\tanitad-caches\navsim-frames-navhard-20260920 | 10.9 | NavHard frame cache (10.9 GB) |
| C:\Users\Admin\tanitad-caches\refcv6-navsim-20260923 | 10.2 | refcv6 NavSim run (live-referenced by the NavSim package) |
| C:\Users\Admin\tanitad-caches\ddv2rl-20260915 | 7.6 | ddv2rl heldout dumps + ckpts (8.2 GB; 41 hardlinked files) |
| C:\Users\Admin\tanitad-caches\refcv6-pipeval-20260917 | 6.7 | refcv6 pipeline-validation run |
| C:\Users\Admin\tanitad-caches\ddv2rl-l1-20260917 | 3.6 | ddv2rl L1 run (3.6 GB) |
| C:\Users\Admin\venvs\driverl-eval / carla312 | 6.4 / 4.6 (C:) | venvs of other work (DriveRL eval, CARLA); inside KEEP `venvs`, not touched |

### 7.3 Not ours - the PI decides, sizes only (nothing proposed)

| path | GB on disk | real GB | files | what |
|---|---|---|---|---|
| D:\euw-audi-fashaf-l3-prod | 1494.5 | 1493.0 | 2,817 | not TanitAD (PI) |
| D:\Daten_Paris | 236.4 | 236.3 | 2,024 | not TanitAD (PI) |
| D:\TanitAstra | 0.2 | 0.0 | 225 | not TanitAD (PI) |
| D:\TanitAstraResearch | 47.3 | 10.8 | 39,243 | not TanitAD (PI); 39,243 files = 47 GB on disk for 10.8 GB real |
| D:\Install Security for Mac.zip | 0.0 | 0.0 | 1 | installer |
| D:\Install Security for Windows.exe | 0.0 | 0.0 | 1 | installer |
| D:\Projects\HierarchicalWorldModeling | 34.3 | 22.0 | 15,136 | other project of the PI (total only) |
| D:\Projects\JevRLsimulation | 15.4 | 6.3 | 10,611 | other project of the PI (total only) |
| C:\Users\Admin\.cache\jevrl | 15.6 | 15.5 | 55,960 | cache of the PI other project JevRLsimulation (total only; venvs + results) |
| C:\Users\Admin\.cache\world-modeling | 4.1 | 4.0 | 35,520 | cache of the PI other project HierarchicalWorldModeling (total only) |
| C:\Users\Admin\.cache\codex-runtimes | 1.3 | 1.3 | 21,783 | OpenAI Codex runtimes (a Codex node runtime is running) |
| C:\Users\Admin\EarningsSwingResearch | 9.0 | 9.0 | 20,469 | other project of the PI (total only) |
| C:\Users\Admin\Documents | 8.4 | 8.2 | 85,870 | personal folder - total only |
| C:\Users\Admin\Downloads | 1.4 | 1.4 | 38 | personal folder - total only |
| C:\Dev | 2.9 | 2.8 | 44,462 | older projects from 2025 (total only) |
| C:\Users\Admin\.codex | 2.3 | 2.3 | 7,389 | OpenAI Codex state |
| C:\Users\Admin\AppData\Local\Packages | 9.2 | 9.1 | 26,105 | Windows Store app data |
| C:\Users\Admin\AppData\Local\Google | 6.7 | 6.7 | 18,210 | Google Drive FS cache 4.3 GB (the G: mount) + Chrome 1.8 GB - never touch |
| C:\Users\Admin\AppData\Local\hermes | 3.4 | 3.1 | 129,011 | Hermes agent app (running) |
| C:\Users\Admin\AppData\Roaming\Code | 2.4 | 2.4 | 8,048 | VS Code |
| C:\Users\Admin\AppData\Roaming\Claude | 1.3 | 1.2 | 15,444 | Claude desktop app |
| D:\$RECYCLE.BIN | 0.0 | 0.0 | 1 | empty (only desktop.ini, 129 B) - nothing to empty |
| D:\System Volume Information | 0.0 | 0.0 | 2 | WPSettings.dat, IndexerVolumeGuid |
| C:\pagefile.sys / hiberfil.sys | 26.7 / 12.7 |  | 2 | OS; `powercfg /h off` returns 12.7 GB (system setting, not done) |
| C:\Windows, Program Files, Program Files (x86), ProgramData | ~21 |  |  | out of scope |

## 8. Appendix

### 8.1 How to run

1. Open a normal PowerShell 5.1 window. Check `subst` shows `D:\ => E:\` (script B/C abort otherwise).
2. `powershell -NoProfile -File "...\cleanup_devbox_A.ps1"` with `$DryRun = $true` prints every action and a total. Read it, then set `$DryRun = $false` and run again. Same for B. For C, un-comment the lines you approve first.
3. Each script's `Remove-Item` lines are routed through a guarded function (KEEP set, junction refusal, idle check, size accounting) - read the header (first ~190 lines) once.
4. B switches: `$IncludeExtractedArchives` (nuPlan + comma2k19 zips), `$VerifyMd5` (re-hash both copies; ~7 GB of D: reads).
5. `maintenance_repack_D_repo.ps1` is separate and not a delete; run it only when nothing is committing to the repo.

### 8.2 exFAT slack by top-level directory of D: (on disk vs real)

| dir | on disk GB | real GB | slack GB | dirs | files |
|---|---|---|---|---|---|
| euw-audi-fashaf-l3-prod | 1494.5 | 1493.0 | 1.5 | 133 | 2,817 |
| Projects | 1103.2 | 940.4 | 162.9 | 13,911 | 189,285 |
| Archive | 568.0 | 414.1 | 153.9 | 29,320 | 172,870 |
| Daten_Paris | 236.4 | 236.3 | 0.1 | 4 | 2,024 |
| thor_archive | 175.2 | 107.9 | 67.3 | 7,278 | 82,695 |
| TanitAstraResearch | 47.3 | 10.8 | 36.5 | 3,078 | 39,243 |
| refcv6_eval_kit | 29.6 | 29.4 | 0.3 | 16 | 595 |
| refcv7_eval_kit | 5.9 | 5.5 | 0.4 | 53 | 601 |
| venvs | 3.4 | 0.2 | 3.2 | 396 | 3,367 |
| **all** | 3665.9 | 3239.5 | 426.4 | 54,241 |  |

### 8.3 Coverage: every top-level item >= 200 MB, with its category

Category letters per child item: A/B/C = the item lists above; D = keep (section 7). A directory that is split shows the parts. "rest" = remainder (not individually listed: app data, small subdirs, hardlink double counts) - category D unless the line says otherwise.

| path | GB on disk | classification |
|---|---|---|
| D:\euw-audi-fashaf-l3-prod | 1494.5 | D-not-ours 1494.5 |
| D:\Projects | 1103.2 | B-derived 187.6, C 219.1, D-keep 612.0, D-not-ours 49.7, rest 34.9 |
| D:\Projects\TanitAD | 946.0 | B-derived 187.6, C 132.0, D-keep 593.6, rest 32.8 |
| D:\Projects\TanitAD\data | 732.2 | B-derived 187.6, D-keep 543.2, rest 1.4 |
| D:\Archive | 568.0 | B-copy 3.2, C 264.1, D-keep 300.7 |
| D:\Archive\devbox-C\navsim | 273.9 | D-keep 273.9 |
| C:\Users\Admin\tanitad-data | 271.1 | B-derived 8.1, C 88.6, D-keep 156.7, rest 17.6 |
| D:\Daten_Paris | 236.4 | D-not-ours 236.4 |
| D:\thor_archive | 175.2 | C 175.2 |
| C:\Users\Admin\tanitad-caches | 116.4 | D-keep 116.4 |
| D:\Projects\TanitAD\_c | 111.0 | C 111.0 |
| C:\Users\Admin\AppData | 98.8 | A 46.0, D-keep 16.9, D-not-ours 22.9, rest 13.0 |
| D:\Projects\TanitAD-artifacts | 91.8 | C 71.3, D-keep 18.3, rest 2.1 |
| C:\Users\Admin\refav1_trainprobe | 59.7 | C 59.7 |
| C:\lgt | 56.4 | C 56.4 |
| D:\Archive\devbox-C\carla | 53.2 | C 53.2 |
| D:\Projects\TanitAD\.git | 48.6 | D-keep 48.6 |
| D:\TanitAstraResearch | 47.3 | D-not-ours 47.3 |
| D:\Projects\HierarchicalWorldModeling | 34.3 | D-not-ours 34.3 |
| C:\Users\Admin\.cache | 33.6 | A 7.1, D-keep 5.5, D-not-ours 21.0 |
| D:\Archive\devbox-C\wpd-probe | 31.0 | C 31.0 |
| D:\refcv6_eval_kit | 29.6 | B-copy 1.1, C 28.5 |
| D:\Archive\devbox-C\tanitad-caches | 26.9 | D-keep 26.9 |
| C:\pagefile.sys | 26.7 | rest 26.7 |
| C:\Users\Admin\qland | 22.7 | D-keep 22.7 |
| D:\Projects\TanitAD\TanitAD Research Lab | 21.9 | rest 21.9 |
| C:\Users\Admin\navsim-crun | 21.6 | D-keep 21.6 |
| D:\Projects\TanitAD\.claude | 21.0 | C 21.0 |
| D:\Archive\devbox-C\dkhead_run | 18.8 | C 18.8 |
| C:\Users\Admin\venvs | 17.3 | D-keep 17.3 |
| D:\Projects\TanitAD-wt-1024 | 15.7 | C 15.7 |
| D:\Projects\JevRLsimulation | 15.4 | D-not-ours 15.4 |
| D:\Archive\devbox-C\refav1_eval_full | 14.0 | C 14.0 |
| C:\hiberfil.sys | 12.7 | rest 12.7 |
| C:\lgs | 12.4 | C 12.4 |
| D:\Archive\devbox-C\refav1_probe | 11.8 | C 11.8 |
| D:\Archive\devbox-C\run_refcv4 | 11.0 | C 11.0 |
| C:\Users\Admin\rc7restart_work | 10.6 | C 10.6 |
| D:\Archive\devbox-C\wpa-readout | 10.5 | C 10.5 |
| C:\Program Files | 9.6 | rest 9.6 |
| C:\Users\Admin\EarningsSwingResearch | 9.0 | D-not-ours 9.0 |
| C:\Users\Admin\Documents | 8.4 | D-not-ours 8.4 |
| D:\Archive\devbox-C\refav1_eval_slice | 8.3 | C 8.3 |
| C:\Program Files (x86) | 8.0 | rest 8.0 |
| D:\Archive\devbox-C\navcomp | 7.9 | C 7.9 |
| C:\dzo | 7.8 | C 7.8 |
| D:\Archive\devbox-C\p1gate | 7.1 | C 7.1 |
| D:\Archive\devbox-C\rlgate2 | 7.1 | C 7.1 |
| C:\Users\Admin\lg0926 | 7.1 | C 7.1 |
| D:\Archive\devbox-C\run_refcv3_ol | 6.7 | C 6.7 |
| C:\Users\Admin\tanitad-push | 6.4 | D-keep 6.4 |
| C:\Users\Admin\tanitad-wt | 6.3 | B-held 3.3, C 3.1 |
| C:\Users\Admin\bxh_tree_e | 6.2 | A 6.2 |
| D:\Archive\devbox-C\run_refcv4v | 6.0 | C 6.0 |
| D:\refcv7_eval_kit | 5.9 | D-keep 5.9 |
| C:\Users\Admin\ev6 | 5.8 | C 5.8 |
| D:\Archive\devbox-C\tanitad-bench | 5.6 | C 5.6 |
| C:\Users\Admin\.claude | 5.3 | D-keep 5.3 |
| D:\Archive\devbox-C\rl_refcv3_min | 5.2 | C 5.2 |
| D:\Archive\devbox-C\tanitad-labelwire | 5.0 | C 5.0 |
| D:\Archive\devbox-C\tanitad-lwbase | 5.0 | C 5.0 |
| D:\Projects\TanitAD\FlyWheels | 4.9 | D-keep 1.8, rest 3.1 |
| D:\Archive\devbox-C\tanitad-refcv4b-analyze | 4.8 | C 4.8 |
| C:\Users\Admin\tanitad-wt-join3d | 4.8 | C 4.8 |
| C:\Users\Admin\tanitad-wt-percep-v6 | 4.8 | C 4.8 |
| D:\Archive\devbox-C\tanitad-evtool-20260906 | 4.7 | C 4.7 |
| D:\Archive\devbox-C\tanitad-perbatch | 4.5 | C 4.5 |
| D:\Archive\devbox-C\run_paired | 4.4 | C 4.4 |
| D:\Archive\devbox-C\tanitad-ctg | 4.3 | C 4.3 |
| D:\Archive\devbox-C\veto_run | 4.3 | C 4.3 |
| D:\Archive\devbox-C\run_refcv3_viz | 4.2 | C 4.2 |
| D:\Archive\devbox-C\tsc2 | 4.1 | C 4.1 |
| D:\Archive\devbox-C\run_wbank | 3.9 | C 3.9 |
| C:\Users\Admin\tanitad-backup | 3.7 | B-copy 3.7 |
| C:\Users\Admin\libtree | 3.5 | A 3.5 |
| D:\venvs | 3.4 | D-keep 3.4 |
| C:\Users\Admin\ygate_fix | 3.2 | C 3.2 |
| C:\Users\Admin\ygate_tip | 3.2 | C 3.2 |
| C:\Users\Admin\cfull_cand | 3.2 | C 3.2 |
| C:\Users\Admin\cfull_tip | 3.1 | C 3.1 |
| C:\ProgramData | 3.0 | rest 3.0 |
| C:\Dev | 2.9 | D-not-ours 2.9 |
| D:\Archive\devbox-C\nuscenes | 2.9 | C 2.9 |
| C:\Users\Admin\bxh_work | 2.7 | C 2.7 |
| D:\Archive\devbox-C\refcv4b_egodrop | 2.7 | C 2.7 |
| C:\Users\Admin\tanitad-mirror | 2.7 | C 2.7 |
| C:\Users\Admin\v7f_gate | 2.7 | C 2.7 |
| D:\Projects\TanitAD\stack | 2.5 | rest 2.5 |
| C:\Users\Admin\.codex | 2.3 | D-not-ours 2.3 |
| D:\Projects\TanitAD\taniteval | 2.2 | rest 2.2 |
| D:\Archive\devbox-C\rl_rescope | 2.0 | C 2.0 |
| D:\sam3map_lidar | 1.8 | C 1.8 |
| D:\Archive\devbox-C\refcv4b_final | 1.6 | B-copy 1.6 |
| D:\Archive\devbox-C\refcv3_final | 1.6 | B-copy 1.6 |
| D:\Archive\devbox-C\refav1_drive | 1.6 | C 1.6 |
| C:\Users\Admin\Downloads | 1.4 | D-not-ours 1.4 |
| C:\Users\Admin\refcv5v2_final | 1.3 | B-held 1.3 |
| D:\Archive\devbox-C\tanitad-egodrop | 1.2 | C 1.2 |
| C:\Users\Admin\v7f_ckpt_gate | 1.0 | C 1.0 |
| C:\Users\Admin\v7f_r1r4 | 0.9 | C 0.9 |
| C:\Users\Admin\refav1_margin | 0.8 | C 0.8 |
| C:\Users\Admin\v7f_r3 | 0.8 | C 0.8 |
| C:\Users\Admin\refav1_r1r6 | 0.8 | C 0.8 |
| C:\Users\Admin\tanitad-media-backup | 0.8 | D-keep 0.8 |
| C:\Users\Admin\tipsnap | 0.7 | C 0.7 |
| C:\Users\Admin\.venvs | 0.6 | rest 0.6 |
| C:\Users\Admin\ev7nav | 0.6 | D-keep 0.6 |
| C:\Users\Admin\tcW8_cand | 0.6 | A 0.6 |
| C:\Users\Admin\tcW8_tip | 0.6 | A 0.6 |
| D:\Projects\TanitAD\colab | 0.6 | rest 0.6 |
| D:\Projects\TanitAD\products | 0.5 | rest 0.5 |
| C:\Users\Admin\refcv5cmp | 0.5 | D-keep 0.5 |
| C:\Users\Admin\.openhuman | 0.4 | rest 0.4 |
| C:\Users\Admin\v7f_merge | 0.4 | C 0.4 |
| C:\Users\Admin\v7f_chain | 0.4 | A 0.4 |
| C:\Users\Admin\ev6_battery | 0.4 | C 0.4 |
| C:\Users\Admin\v7f_ckpt | 0.4 | C 0.4 |
| C:\Users\Admin\ev7 | 0.4 | D-keep 0.4 |
| C:\Users\Admin\nb2_tree_2301 | 0.4 | C 0.4 |
| C:\Users\Admin\rc7ceil | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tree_2128 | 0.4 | C 0.4 |
| C:\Users\Admin\bxh_tree_a17 | 0.4 | A 0.4 |
| C:\Users\Admin\ctree_cand | 0.4 | A 0.4 |
| C:\Users\Admin\v7f_r3_base | 0.4 | C 0.4 |
| C:\Users\Admin\bxh_tree_i6 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_h6 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_i | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_h | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r5_tree_2ac0 | 0.4 | A 0.4 |
| C:\Users\Admin\ev6_postfix | 0.4 | A 0.4 |
| C:\Users\Admin\ev6_postfix_da8400b_overlay | 0.4 | A 0.4 |
| C:\Users\Admin\rc7combo | 0.4 | A 0.4 |
| C:\Users\Admin\rc7restart | 0.4 | A 0.4 |
| C:\Users\Admin\gt13_cand_212930 | 0.4 | A 0.4 |
| C:\Users\Admin\gt13_tip_212930 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r4_tree_1b81 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r3_tree_2374 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_g6 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_g | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r2_tree_cef9 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tree_4797ffb | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_cef9709 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_f6 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_f | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_d | 0.4 | A 0.4 |
| C:\Users\Admin\r4tree | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tip_2310 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tip_2203 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tree_b3prev | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tree_b4a5 | 0.4 | A 0.4 |
| C:\Users\Admin\rc7frozen | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_4797ffb | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_b | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tree_0638 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2a19_tree | 0.4 | A 0.4 |
| C:\Users\Admin\nb2ml2_tree | 0.4 | A 0.4 |
| C:\Users\Admin\rc7lever2 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2ml_tree_3708 | 0.4 | C 0.4 |
| C:\Users\Admin\fixtree | 0.4 | C 0.4 |
| C:\Users\Admin\bxh_tree_m | 0.4 | A 0.4 |
| C:\Users\Admin\ygate_fixes | 0.4 | C 0.4 |
| C:\Users\Admin\bxh_tree_a | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_68e8ec6 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_k | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_49f10b4 | 0.4 | A 0.4 |
| C:\Users\Admin\fixtree_tip | 0.4 | C 0.4 |
| C:\Users\Admin\bxh_tree_toy | 0.4 | A 0.4 |
| C:\Users\Admin\tip_2ac0_clean | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r5_tree_8796 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r4_tip_1b81 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tree_c | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r3_tip_2374 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2r2_tip_cef9 | 0.4 | A 0.4 |
| C:\Users\Admin\.vscode | 0.4 | rest 0.4 |
| C:\Users\Admin\bxh_tip_2ac0bfb | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tip_4797ffb | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_28d8365 | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_1b8170e | 0.4 | A 0.4 |
| C:\Users\Admin\bxh_tip_2374cd2 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tip_b3prev | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tip_b4a5 | 0.4 | A 0.4 |
| C:\Users\Admin\nb2_tip_0638 | 0.4 | A 0.4 |
| C:\Users\Admin\refcv4b_repo | 0.3 | C 0.3 |
| C:\Users\Admin\a40-rescue | 0.3 | D-keep 0.3 |
| C:\Users\Admin\ev6_82c2331 | 0.3 | A 0.3 |
| C:\Users\Admin\cfix | 0.3 | A 0.3 |
| C:\Users\Admin\ygate_video | 0.3 | C 0.3 |
| C:\Users\Admin\tcL | 0.3 | A 0.3 |
| C:\Users\Admin\tanitad-bench-offrepo | 0.3 | C 0.3 |
| D:\refcv7_diag_bin | 0.2 | rest 0.2 |
| C:\Users\Admin\refcv4b_viz | 0.2 | C 0.2 |
| D:\Projects\TanitAD\Project Steering | 0.2 | rest 0.2 |
| D:\TanitAstra | 0.2 | D-not-ours 0.2 |
| C:\Users\Admin\ctree_tip | 0.2 | A 0.2 |
| C:\Users\Admin\dz | 0.2 | C 0.2 |

Not itemised: the 785 top-level entries of C:\Users\Admin below 200 MB (2.05 GB in total) and every file/dir below 200 MB inside the listed dirs.

### 8.4 Raw numbers

`INVENTORY_DEVBOX_evidence.json` bundles the machine-readable results behind the tables: per-tree blob check, B-pair md5 verification, zip member checks, hardlink check, register citation counts, totals.

