# E: working-tree stranded-work audit (2026-10-18)

Tip audited against: `780a025558d29102bf5f15ba695e5b376c64e301` (branch `agent/arch-inf-20260803`, mirror `tanitad-push`). E: checkout HEAD `37645fcc` (2026-09-19), ancestor of the tip.
Evidence class for every number below: **MEASURED** (ours; `classes.json` and `s4_list.csv` in this package, produced by `scripts/`). Content comparison is by git blob id of the file BYTES (`hash-object --no-filters` equivalent), with a CRLF<->LF variant tried; mtime is used only as a secondary hint and never alone.

## Headline

* **2279 of 15822 changed/untracked paths are STRANDED candidates (S4: absent from the tip AND from all 3272-commit history, content absent from the tip under any other path)** - 4422.1 MB total, mostly raw run outputs and package-local code of EvalFlyWheel / Research-Lab / Data-Audit packages (13 of the 74 `LANDING_READY*` manifests are themselves not on the tip, section 7).
* **Recommendation split of the S4 set (+ gitignored extras):** LAND (P1+P2, in paths.json) = 1924 files / 523.8 MB; LAND-LARGE (5-20 MB, optional separate commit) = 81 files / 847.9 MB; LAND-HOLD-SCRUB (secret/IP/token-path flag) = 11; BULK-ONLY = 2171 files / 3132.3 MB; DROP-scratch = 151; GITIGNORED-DERIVED = 105.
* **47 CANDIDATE-EDITs** (S2b: path is on the tip, E content differs and is not any historical version of the path) - list in section 4; **none landed**.
* **Most differences are stale or already landed:** 12552 paths (79.3 %) are byte-identical to the tip blob (S1), 106 are stale (S2a: E blob is a historical version of the path).
* **E: and D: are the same volume** (`subst D: => E:\`, volume serial identical), i.e. E:\Projects\TanitAD *is* the canonical D:\Projects\TanitAD working tree where agents write; its git HEAD simply stopped moving at 2026-09-19 because the Master Mind lands through the mirror. That is why the tree holds three weeks of agent output that git on E: has never seen.

## 1. Scope and method

* Population (MEASURED): staged paths `git diff --cached` = 6,654 (6,590 A + 64 M); unstaged-modified `ls-files -m` = 133 (+1 deleted); untracked `ls-files -o --exclude-standard` = 9,147; union = **15822 distinct paths**. All read through plumbing with `--no-optional-locks`; E:'s index and files were not modified.
* Not in the population: tracked files unchanged since E-HEAD (they only lag the tip) and files ignored by E:'s `.gitignore` - the latter were enumerated separately in section 6 (the ignore rule `data/` hides package `code/fix/**/stack/tanitad/data/*.py`, `taniteval/**/report/` and all mp4/Media replay frames).
* Tip side: 31754 blobs from `ls-tree -r`; history side: `git log --all --raw --no-renames -m` over 3,272 commit entries -> path -> set of blob ids ever held (37804 paths, 37641 distinct blobs).
* 75 untracked files exceed Windows MAX_PATH (> 240 chars) and were first reported unreadable by the naive probe - the same "a zero is a claim about the probe" trap as in CLAUDE.md; they were re-read with the `\\?\` prefix and classified (all S4). Paths longer than 259 chars need the same prefix at landing time.

## 2. Counts per class

| class | meaning | n | share |
|---|---|---:|---:|
| S1 | identical to tip blob (raw 10141 / after CRLF<->LF 2411) | 12552 | 79.33 % |
| S2a | on tip, different; E blob IS a historical version of the path -> STALE | 106 | 0.67 % |
| S2b | on tip, different; E blob is NOT any historical version -> CANDIDATE-EDIT | 47 | 0.30 % |
| S3 | absent from tip, path in history (deleted/renamed later) | 0 | 0.00 % |
| S4 | absent from tip AND all history, content nowhere on tip -> STRANDED candidate | 2279 | 14.40 % |
| S4m | absent path, same content on tip under another path (MOVED) - strong 364 (size>=256 B and same basename), weak 370 | 734 | 4.64 % |
| S4h | absent path, same content in history under another path - strong 101, weak 2 | 103 | 0.65 % |
| MISSING-WT | in E index, file gone from worktree | 1 | 0.01 % |
| **total** | | **15822** | |

S3 = 0: none of the 15,822 paths is a path the tip later deleted or renamed. The 103 S4h and 734 S4m paths are the only "elsewhere" cases; the weak ones (tiny or different-basename content-identical files, e.g. identical per-run score logs) mean the **content** is preserved on the tip while the **path** is not - they are listed in `companions.csv` when they sit in a directory where S4 files are being landed, so a lander can keep the package layout complete (optional).

### By origin (staged / unstaged / untracked)

| origin | S1 | S2a | S2b | S3 | S4 | S4m | S4h | MISSING-WT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| staged-A | 6295 | 23 | 14 | 0 | 81 | 75 | 0 | 0 |
| staged-A+unstaged-D+unstaged-M | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| staged-A+unstaged-M | 78 | 18 | 5 | 0 | 0 | 0 | 0 | 0 |
| staged-M | 36 | 13 | 5 | 0 | 0 | 0 | 0 | 0 |
| staged-M+unstaged-M | 1 | 5 | 4 | 0 | 0 | 0 | 0 | 0 |
| unstaged-M | 17 | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| untracked | 6125 | 44 | 18 | 0 | 2198 | 659 | 103 | 0 |

### By top-level directory

| top-level dir | S1 | S2a | S2b | S3 | S4 | S4m | S4h | MISSING-WT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TanitAD Research Lab | 7338 | 59 | 16 | 0 | 463 | 184 | 100 | 0 |
| FlyWheels | 3613 | 1 | 11 | 0 | 1575 | 236 | 3 | 1 |
| taniteval | 578 | 8 | 8 | 0 | 134 | 221 | 0 | 0 |
| colab | 524 | 0 | 1 | 0 | 27 | 82 | 0 | 0 |
| products | 368 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| stack | 90 | 27 | 2 | 0 | 0 | 0 | 0 | 0 |
| Media | 11 | 0 | 0 | 0 | 58 | 3 | 0 | 0 |
| Project Steering | 14 | 9 | 7 | 0 | 16 | 0 | 0 | 0 |
| tools | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| .claude | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| Benchmarks & Eval | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| CLAUDE.md | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| c3_run_p2full_vs_p2.diff | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| data | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| p2full_driver_vs_p2tab.diff | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

### The 157 staged-`A` paths absent from the tip (brief list: 157 lines)

Classification of the paths in `e_staged_absent.txt`: MISSING-WT 1, S4 81, S4m 75.
So 81 of them are genuinely stranded (S4) and 75 are staged copies whose bytes the tip already holds under another path (typically package `code/fix/stack/...` copies landed at `stack/...`).

One more staged path is `MISSING-WT` (index entry, worktree file deleted): see section 5.

## 3. S4 - stranded candidates: recommendation per class

| recommendation | files | MB | rule |
|---|---:|---:|---|
| LAND P1 | 1543 | 101.2 | text/code/docs/json/csv/log <= 1 MB |
| LAND P2 | 381 | 422.7 | other landable files <= 5 MB |
| LAND-LARGE (P3) | 81 | 847.9 | landable files 5-20 MB: raw data, optional separate commit; NOT in paths.json (`paths_P3_optional.json`) |
| LAND-HOLD-SCRUB | 11 | 0.6 | scan flag (public IP, ssh port, token-file path); scrub then land - NOT in paths.json |
| BULK-ONLY | 2171 | 3132.3 | > 20 MB, or .pt/.npy/.bin/video/media (list only; never land) |
| DROP-scratch | 151 | 16.2 | `_scratch/` trees, caches, pid/lock files, `.done` markers of non-landable media |
| GITIGNORED-DERIVED | 105 | 158.8 | ignored by E:'s own `.gitignore`; regenerable reports / replay frames |

`paths.json` contains exactly the LAND set (P1+P2) plus this audit package and the recovered index-only blob; `paths_by_tier.json` splits it so the Master Mind can take P1 only; `paths_P3_optional.json` carries the 5-20 MB LAND-LARGE files. Full per-file list with one-line descriptions: `s4_list.csv`.

### Rollup per package (S4 files only)

| package | S4 | LAND | MB landed | LAND-LARGE | MB large | BULK-ONLY | MB bulk | DROP | HOLD | files of this package already on tip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit | 123 | 98 | 76.5 | 12 | 122.5 | 10 | 1675.4 | 3 | 0 | 323 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests | 844 | 795 | 275.9 | 34 | 271.3 | 14 | 579.4 | 1 | 0 | 1044 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan | 246 | 94 | 30.1 | 23 | 356.5 | 31 | 739.5 | 94 | 4 | 6936 |
| Media/2026-10-04-refcv7-replay/data | 2072 | 0 | 0.0 | 0 | 0.0 | 2059 | 98.9 | 13 | 0 | 0 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend | 56 | 52 | 34.9 | 4 | 38.4 | 0 | 0.0 | 0 | 0 | 380 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest | 438 | 432 | 31.6 | 6 | 35.8 | 0 | 0.0 | 0 | 0 | 211 |
| taniteval/results/bench/navsim_v2 | 149 | 74 | 25.0 | 1 | 13.5 | 0 | 0.0 | 74 | 0 | 362 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests | 67 | 66 | 27.0 | 1 | 9.9 | 0 | 0.0 | 0 | 0 | 951 |
| Media/2026-10-04-refcv7-replay/keyframes | 50 | 0 | 0.0 | 0 | 0.0 | 50 | 33.2 | 0 | 0 | 0 |
| taniteval/results/bench/_scratch | 53 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 53 | 0 | 0 |
| colab/raw/2026-09-27-navtest-p0 | 12 | 12 | 5.7 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 368 |
| Media/2026-10-04-refcv7-replay/verify_frames | 6 | 0 | 0.0 | 0 | 0.0 | 6 | 5.1 | 0 | 0 | 0 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage | 123 | 123 | 4.6 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 121 |
| Project Steering/Ops | 14 | 9 | 2.1 | 0 | 0.0 | 0 | 0.0 | 0 | 5 | 0 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning | 13 | 13 | 1.8 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 508 |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state | 6 | 6 | 1.7 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 365 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head | 14 | 14 | 1.7 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 208 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only | 30 | 30 | 1.5 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 7 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps | 12 | 12 | 1.3 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 323 |
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing | 5 | 5 | 0.8 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 20 |
| Media/2026-10-04-refcv7-replay/tool_screenshot_clip03_w109.png | 1 | 0 | 0.0 | 0 | 0.0 | 1 | 0.8 | 0 | 0 | 0 |
| taniteval/results | 1 | 1 | 0.6 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 781 |
| taniteval/results/bench/internal_t1 | 1 | 1 | 0.3 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 9 |
| taniteval/results/bench/nuscenes_ol | 18 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 18 | 0 | 98 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navhard-download | 5 | 5 | 0.2 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 10 |
| Project Steering/Handover | 1 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 1 | 5 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture | 16 | 16 | 0.1 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 22 |
| colab/raw/2026-09-27-navtest-plan | 15 | 15 | 0.1 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 80 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-leaderboard-currency | 1 | 1 | 0.1 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 12 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-step-cost | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 240 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 52 |
| taniteval/tests | 3 | 3 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 127 |
| FlyWheels/TanitAD_DataFlyWheel | 1 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 1 | 81 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-stage1-extract | 34 | 34 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 7 |
| taniteval/tools | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 62 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-fixes | 2 | 2 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 18 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 62 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-launch | 3 | 3 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 11 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 146 |
| Project Steering | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 265 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-gt-validation | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 29 |
| Media/2026-10-04-refcv7-final-reel/LANDING_READY.txt | 1 | 1 | 0.0 | 0 | 0.0 | 0 | 0.0 | 0 | 0 | 0 |

### BULK-ONLY files (list; never land)

| path | MB | reason |
|---|---:|---|
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p4_mask/mask_R7_A1.bin | 1132.8 | media/checkpoint/array format .bin |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p1x_fan/fan_R7_A1.jsonl | 152.8 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p4_fan/fan_R7_A1.jsonl | 152.8 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p1_fan/fan_R7_A1.jsonl | 62.1 | size>20MB |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/lane_census.jsonl | 56.2 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/bridge_navtest/rows_R7_A1_s1.jsonl | 55.9 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step5000/bridge_navtest/rows_R7_A1_s1.jsonl | 55.9 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/bridge_navtest/rows_R7_A1.jsonl | 55.9 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step5000/bridge_navtest/rows_R7_A1.jsonl | 55.8 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/bridge_navtest/rows_R7_A1_s1.jsonl | 55.7 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/bridge_navtest/rows_R7_A1.jsonl | 55.6 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/vmaxoff_legal/bridge_navtest/rows_R7_VMAXOFF.jsonl | 55.6 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/raw/refcv7_run_records/metrics.jsonl | 36.8 | size>20MB |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W0e2_stopped_233806/partial.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W1e2_stopped_234250/partial.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/P0e2/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/P1e2/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W0e2/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W1e2/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/G3_zero_lr/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/T0/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/T1/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/T2/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/G3_zero_lr/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/Wt0/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/Wt1/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/arms/Wt2/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/P0/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/P1/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W0/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W1/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-27-m6-proxy/arms/W1_cachepath_222349/final.pt | 33.2 | media/checkpoint/array format .pt |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/tables/label_truth_train.npz | 30.9 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/raw/thor_train/windows_train.npz | 29.9 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step5000/bridge_navhard/rows_R7_A1_s1.jsonl | 27.1 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step5000/bridge_navhard/rows_R7_A1.jsonl | 27.1 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/bridge_navhard/rows_R7_A1_s1.jsonl | 27.0 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p1x_bridge/rows_R7_A1.jsonl | 27.0 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/p4_bridge/rows_R7_A1.jsonl | 27.0 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/bridge_navhard/rows_R7_A1.jsonl | 27.0 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/bridge_navhard/rows_R7_A1_s1.jsonl | 27.0 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/bridge_navhard/rows_R7_A1.jsonl | 26.9 | size>20MB |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step50400/vmaxoff_legal/bridge_navhard/rows_R7_VMAXOFF.jsonl | 26.8 | size>20MB |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/raw/geom_all.jsonl | 23.4 | size>20MB |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/lane_census2.jsonl | 20.9 | size>20MB |

Plus 2126 smaller BULK-ONLY files (170.1 MB; .npz/.pkl/.pt/.bin/media-format or Media keyframes) - see `s4_list.csv`.

### Secret / PII flags (never printed - file names and flag classes only)

Patterns scanned: HF/GitHub/AWS/Slack/OpenAI-style tokens, PEM private keys, `Bearer` values, key=value secret assignments, a token-file path/name, runpod ssh host names, `ssh -p <port>`, public IPv4 (with port, or in ssh/host context). 0 token-class hits anywhere. The 350 first-pass `secret_assign` hits were NavSim scene `token` ids (16-hex), a measured false positive, and are not flagged. Remaining flags (files only):

| class | file | flags | scrubbed copy (OUT/scrubbed/, listed in paths_scrubbed.json) |
|---|---|---|---|
| S4 | FlyWheels/TanitAD_DataFlyWheel/HANDOFF_DATAFLYWHEEL_2026-10-11.md | tokenfile_path | yes: token-file x1 |
| S4 | Project Steering/Handover/TanitAD_HANDOVER_2026-10-15_refa_refd_v8f.md | public_ip_ctx,ssh_port,tokenfile_path | yes: ip x1, ssh-port x1, token-file x1 |
| S4 | Project Steering/Ops/disk_cleanup_2026-10-04/INVENTORY_DEVBOX.md | tokenfile_path | yes: token-file x5 |
| S4 | Project Steering/Ops/disk_cleanup_2026-10-04/INVENTORY_DEVBOX_evidence.json | tokenfile_path | yes: token-file x1 |
| S4 | Project Steering/Ops/disk_cleanup_2026-10-04/cleanup_devbox_A.EXECUTE.ps1 | tokenfile_path | yes: token-file x1 |
| S4 | Project Steering/Ops/disk_cleanup_2026-10-04/cleanup_devbox_A.ps1 | tokenfile_path | yes: token-file x1 |
| S4 | Project Steering/Ops/disk_cleanup_2026-10-04/cleanup_devbox_C.ps1 | tokenfile_path | yes: token-file x2 |
| S4 | TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/eval_snapshot.sh | public_ip_ctx,ssh_port | yes: ip x4, ssh-port x4 |
| S4 | TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/m5_transfer.py | public_ip_ctx | yes: ip x1 |
| S4 | TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/wait_and_eval.sh | public_ip_ctx,ssh_port | yes: ip x1, ssh-port x1 |
| S4 | TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/report/rebuild_report.sh | public_ip_ctx,ssh_port | yes: ip x4, ssh-port x3 |
| S2b | FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/RESULT.md | public_ip_ctx | n/a (tip file; candidate edit) |
| S2b | Project Steering/MODEL_REGISTRY.md | public_ip_ctx,public_ip_port | n/a (tip file; candidate edit) |
| S2b | Project Steering/PI_DECISION_QUEUE.md | tokenfile_path | n/a (tip file; candidate edit) |
| S2b | Project Steering/RETRACTION_LOG.md | public_ip_ctx,public_ip_port,tokenfile_path | n/a (tip file; candidate edit) |
| S2b | Project Steering/TANITAD_PROGRAMME.md | tokenfile_path | n/a (tip file; candidate edit) |

S2b rows flagged above: scanning the TIP blob of the same path with the same patterns gives -> RESULT.md: public_ip_ctx; MODEL_REGISTRY.md: clean; PI_DECISION_QUEUE.md: tokenfile_path; RETRACTION_LOG.md: tokenfile_path; TANITAD_PROGRAMME.md: tokenfile_path. (E-side flags that are absent on the tip are E-only additions and are withheld from `candidate_edits/` when they fall inside the exported lines.)

Files modified in the last 24 h (possibly still being written): 0.

### S4 package-local code that targets the repo (INTEGRATION DECISION NEEDED)

S4 `.py/.json` files under a package `code/<variant>/stack|taniteval|tools/...` tree whose target repo path exists on the tip with DIFFERENT content (so landing the package copy does not change the repo code): 27 of 29 target paths exist on the tip.

| package-local copy (S4) | target path | target on tip |
|---|---|---|
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/code/fix/stack/scripts/refc_v3_train.py | stack/scripts/refc_v3_train.py | yes (content differs) |
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/code/fix/stack/tests/test_declared_vs_built.py | stack/tests/test_declared_vs_built.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/scripts/refc_v3_train.py | stack/scripts/refc_v3_train.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tanitad/models/refcv6_perception_branch.py | stack/tanitad/models/refcv6_perception_branch.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tanitad/train/declared_vs_built.py | stack/tanitad/train/declared_vs_built.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tests/test_declared_vs_built.py | stack/tests/test_declared_vs_built.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tests/test_occ_knob_is_stamped.py | stack/tests/test_occ_knob_is_stamped.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/scripts/refc_v3_train.py | stack/scripts/refc_v3_train.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tanitad/models/refcv6_perception_branch.py | stack/tanitad/models/refcv6_perception_branch.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tanitad/train/declared_vs_built.py | stack/tanitad/train/declared_vs_built.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tests/test_cls_weight_stamp.py | stack/tests/test_cls_weight_stamp.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tests/test_declared_vs_built.py | stack/tests/test_declared_vs_built.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tests/test_occ_knob_is_stamped.py | stack/tests/test_occ_knob_is_stamped.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/ops/runs.d/refcv8-wpb-smoke.argv.json | stack/ops/runs.d/refcv8-wpb-smoke.argv.json | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/scripts/launch_gate.py | stack/scripts/launch_gate.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/scripts/refc_v3_train.py | stack/scripts/refc_v3_train.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/eval/refcv7_loader.py | stack/tanitad/eval/refcv7_loader.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/eval/speed_leak.py | stack/tanitad/eval/speed_leak.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/refs/refc.py | stack/tanitad/refs/refc.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/refs/refc_v3.py | stack/tanitad/refs/refc_v3.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/refs/refcv8_conditioning.py | stack/tanitad/refs/refcv8_conditioning.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/train/refcv8_train.py | stack/tanitad/train/refcv8_train.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tests/test_declared_vs_built.py | stack/tests/test_declared_vs_built.py | yes (content differs) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tests/test_refcv8_speed_input.py | stack/tests/test_refcv8_speed_input.py | yes (content differs) |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/code/fix/stack/scripts/train_v6_staged.py | stack/scripts/train_v6_staged.py | yes (content differs) |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/code/fix/stack/tanitad/models/v6.py | stack/tanitad/models/v6.py | yes (content differs) |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/code/fix/stack/tanitad/train/declared_vs_built_v6.py | stack/tanitad/train/declared_vs_built_v6.py | NO (new file) |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r3/code/fix/stack/scripts/train_v6_staged.py | stack/scripts/train_v6_staged.py | yes (content differs) |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r3/code/fix/stack/tanitad/train/declared_vs_built_v6.py | stack/tanitad/train/declared_vs_built_v6.py | NO (new file) |

### S4 decision-grade documents and code outside raw/

| path | KB | rec | description |
|---|---:|---|---|
| FlyWheels/TanitAD_DataFlyWheel/HANDOFF_DATAFLYWHEEL_2026-10-11.md | 17.9 | LAND-HOLD-SCRUB | Data FlyWheel handoff — 2026-10-11 |
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/code/fix/stack/scripts/refc_v3_train.py | 744.0 | LAND | py: REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm. |
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/code/fix/stack/tests/test_declared_vs_built.py | 28.9 | LAND | py: ⛔⛔ DECLARED vs BUILT — a lever named in argv must be a lever the BUILT model has. |
| Project Steering/Handover/TanitAD_HANDOVER_2026-10-15_refa_refd_v8f.md | 183.2 | LAND-HOLD-SCRUB | HANDOVER — REF-A (refav1), REF-D and the flagship (v7F → v8F) |
| Project Steering/Ops/disk_cleanup_2026-10-04/INVENTORY_DEVBOX.md | 90.0 | LAND-HOLD-SCRUB | INVENTORY_DEVBOX - disks C: and D: of the Windows dev box (2026-10-04) |
| Project Steering/Ops/disk_cleanup_2026-10-04/INVENTORY_THOR.md | 62.9 | LAND | INVENTORY_THOR -- disk inventory and safe cleanup plan, 2026-10-04 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/m5_transfer.py | 16.7 | LAND-HOLD-SCRUB | py: Measure 5 effectiveness run (eval/PREREG_MEASURE5.md §3a): the pod gate and the frame pull. POD SIDE IS RE |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/scripts/refc_v3_train.py | 694.2 | LAND | py: REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tanitad/models/refcv6_perception_branch.py | 49.2 | LAND | py: refcv6 §2/§6 -- THE PERCEPTION BRANCH, as ONE module a trainer can own. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tanitad/train/declared_vs_built.py | 69.6 | LAND | py: G-DVB — every argv lever equals the BUILT model's value, or the run refuses (SPEC_REFCV7 §2). |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tests/test_declared_vs_built.py | 27.3 | LAND | py: ⛔⛔ DECLARED vs BUILT — a lever named in argv must be a lever the BUILT model has. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix/stack/tests/test_occ_knob_is_stamped.py | 7.3 | LAND | py: A behaviour knob that leaves no trace in the run record — pinned by MUTATION. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/scripts/refc_v3_train.py | 696.6 | LAND | py: REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tanitad/models/refcv6_perception_branch.py | 50.4 | LAND | py: refcv6 §2/§6 -- THE PERCEPTION BRANCH, as ONE module a trainer can own. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tanitad/train/declared_vs_built.py | 70.1 | LAND | py: G-DVB — every argv lever equals the BUILT model's value, or the run refuses (SPEC_REFCV7 §2). |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tests/test_cls_weight_stamp.py | 21.7 | LAND | py: H-BOXCLS-1 — the class weight REACHES the loss, and the RECORD states the tensor. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tests/test_declared_vs_built.py | 27.3 | LAND | py: ⛔⛔ DECLARED vs BUILT — a lever named in argv must be a lever the BUILT model has. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-box-head/code/fix_r6/stack/tests/test_occ_knob_is_stamped.py | 7.3 | LAND | py: A behaviour knob that leaves no trace in the run record — pinned by MUTATION. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/BUNDLE_PMAP.md | 6.4 | LAND | BUNDLE_PMAP — running the registered P-MAP arms on any CUDA box |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pbox_analyze.py | 13.1 | LAND | py: WP-D P-BOX analysis (CPU): every registered metric, bar and must-fail of PREREG_WPD_PROBES.md + PREREG_WPD |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pbox_arms.py | 28.5 | LAND | py: WP-D P-BOX arms (PREREG_WPD_PROBES.md, registered sha256 c054190b…): the refcv7 box decoder retrained on t |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pbox_capture.py | 11.3 | LAND | py: WP-D P-BOX capture (PREREG_WPD_PROBES.md, registered sha256 c054190b…): the FROZEN box memory of refcv7-50 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pbox_prep.py | 3.1 | LAND | py: WP-D P-BOX prep (CPU, no GPU): the registered TRAIN clip set as a symlink subset cache + its loader remap. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pbox_table.py | 3.1 | LAND | py: Print the P-BOX result tables (markdown) from raw/pbox_result.json: per arm the registered metrics, the pa |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pmap_analyze.py | 7.5 | LAND | py: WP-D P-MAP analysis (CPU): the registered metrics, bars and must-fails of PREREG_WPD_PROBES.md §P-MAP from |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pmap_arms.py | 24.0 | LAND | py: WP-D P-MAP arms (PREREG_WPD_PROBES.md, registered sha256 c054190b…): the refcv7 10 cm map branch retrained |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/RESULT_R1.md | 11.2 | LAND | RESULT R1 — head-only retraining on a frozen refcv7 trunk: the cut, the cache, the identity controls (arms: §4 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/build_nav_ann_table.py | 4.9 | LAND | py: H3's nav input source: per record, the nav_30s entries + A6's announced() flags, for train AND eval. CPU,  |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/prep_train_r1.py | 4.6 | LAND | py: R1 TRAIN sample (DESIGN.md sec. 4): choose the train clips, build a symlink subset cache + the remap the e |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_a2_choose_w.py | 2.0 | LAND | py: SPEC_R1_A2-2: choose the guidance scale w on the TRAIN-FIT windows (never on eval). |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_arms.py | 26.0 | LAND | py: refcv8 R1 arms -- train the planner-side heads of refcv7-r101-s0 @ 50,400 from the frozen-trunk cache, the |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_capture.py | 27.7 | LAND | py: refcv8 R1 -- cache the frozen trunk of refcv7-r101-s0 @ 50,400 and run the IDENTITY controls (DESIGN.md se |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_codec.py | 9.1 | LAND | py: R1 codec study (DESIGN.md sec. 5 rule: "if only I-S fails, the codec is changed before any arm"). |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_heads.py | 8.6 | LAND | py: refcv8 R1 -- the head-only path: refcv7's OWN forward with the frozen trunk replaced by cached tensors. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_identity_report.py | 4.8 | LAND | py: R1 identity report (CPU, Thor): I-0 (capture FULL forward vs the route package's banked eval_s0.npz on the |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_lib.py | 11.3 | LAND | py: refcv8 R1 (head-only, frozen trunk) -- pure numpy helpers shared by the capture, the arms and the scoring. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_score.py | 15.6 | LAND | py: R1 scoring (CPU; Thor or dev box) -- SPEC_R1 sec. 5-7 on the arms' eval outputs. Writes arms/score.json. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/r1_trainfit_score.py | 3.5 | LAND | py: SPEC_R1 sec. 7 TRAIN-FIT diagnostic (secondary, never a bar): each arm scored on 1,000 of its OWN training |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/code/test_r1_lib.py | 6.8 | LAND | py: Analytic / known-value controls for r1_lib (written with literal expectations; run: python test_r1_lib.py) |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/scripts/launch_gate.py | 348.6 | LAND | py: THE LAUNCH GATE -- no training step runs until this writes a PASS token for the EXACT |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/scripts/refc_v3_train.py | 751.2 | LAND | py: REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/eval/refcv7_loader.py | 55.1 | LAND | py: refcv7 EVAL LOADER -- the model and the held-out eval dataset, built EXACTLY as refcv7's |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/eval/speed_leak.py | 9.7 | LAND | py: X3's LEAK instrument (SPEC_REFCV8 sec. 8.3 (a)(1)): how much FUTURE-speed information a speed INPUT carrie |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/refs/refc.py | 325.6 | LAND | py source |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/refs/refc_v3.py | 182.5 | LAND | py: REF-C v3 — the goal-mediated strategic/tactical/operative hierarchy on the |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/refs/refcv8_conditioning.py | 36.4 | LAND | py: refcv8 WP-B -- the tactical layer CONDITIONS fan generation and selection (DESIGN.md sec. 3, the WP-B pack |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tanitad/train/refcv8_train.py | 53.1 | LAND | py: refcv8 WP-B -- the TRAINER side of the tactical conditioning (DESIGN.md sec. 3; the model side is |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tests/test_declared_vs_built.py | 29.6 | LAND | py: ⛔⛔ DECLARED vs BUILT — a lever named in argv must be a lever the BUILT model has. |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/code/fix_P7/stack/tests/test_refcv8_speed_input.py | 21.9 | LAND | py: refcv8 X3 (SPEC_REFCV8 sec. 8.3; MM ruling Q5; MM item 1, 2026-10-04) -- the PAST-ONLY speed input. |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/code/fix/stack/scripts/train_v6_staged.py | 648.1 | LAND | py: v6 — the STAGED trainer (S-W → S-T → S-S → optional S-J). |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/code/fix/stack/tanitad/models/v6.py | 361.4 | LAND | py: v6 — the STAGED hierarchical world-model composition module. |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r1r4/code/fix/stack/tanitad/train/declared_vs_built_v6.py | 8.3 | LAND | py: G-DVB for ``train_v6_staged.py`` -- the R1 / R4 levers (PI directive 2026-09-27, v7F). |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r3/code/fix/stack/scripts/train_v6_staged.py | 677.9 | LAND | py: v6 — the STAGED trainer (S-W → S-T → S-S → optional S-J). |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_r3/code/fix/stack/tanitad/train/declared_vs_built_v6.py | 12.3 | LAND | py: G-DVB for the v6-STAGED trainer (v7F) — argv levers vs the BUILT ``V6Stack``. |
| taniteval/tests/test_leaderboard_hand_edit_guard.py | 5.5 | LAND | py: A hand edit INSIDE the generated section is REFUSED, never overwritten silently. |
| taniteval/tests/test_leaderboard_navtest_single_stage_ours.py | 8.6 | LAND | py: The leaderboard SHOWS every run it reads — in the right place — and never shows a withdrawn one as a resul |
| taniteval/tests/test_leaderboard_yawmask_input.py | 10.7 | LAND | py: RETR-2026-09-26-YAWMASK lives in the leaderboard's INPUTS, not in a hand edit (Master Mind decision, batch |
| taniteval/tools/derive_yawmask_record.py | 7.5 | LAND | py: Derive the leaderboard INPUT record for refcv3 @40,284 with the RETR-2026-09-26-YAWMASK correction applied |

## 4. S2b - CANDIDATE-EDITs (47). Listed, not landed.

Columns: line deltas CR-normalised. `vs tip` = what E has/lacks relative to the tip blob; `vs E-HEAD` = E edits on top of its own HEAD (only for tracked paths); `tip vs E-HEAD` = how far the tip moved off E's base. `E-only lines` = the exported file with E lines absent from the tip (the part that would be lost).

| path | origin | verdict | mtime hint | vs tip (+/-) | vs E-HEAD (+/-) | tip vs E-HEAD (+/-) | E mtime | tip last change | E-only lines file |
|---|---|---|---|---|---|---|---|---|---|
| .claude/settings.local.json | unstaged-M | MACHINE-LOCAL (do not land) | E NEWER than tip last change | +3/-0 | +3/-0 | +0/-0 | 2026-09-27 17:44 | 2026-08-03 00:35 a96df2a |  |
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/COMMS.md | untracked | LIKELY-STALE (E is a strict line-subset of the tip: tip only adds) | E OLDER than tip last change | +0/-24 |  |  | 2026-10-04 15:40 | 2026-10-04 16:18 c367e7b |  |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navhard-download/raw/navhard_reference_chain.log | staged-A+unstaged-M | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +4/-0 |  |  | 2026-09-19 23:47 | 2026-09-20 03:23 3b02a41 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/24687c33_navhard_reference_chain.log.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw/K7_packer_vs_trainer.json | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +6/-6 |  |  | 2026-09-19 11:56 | 2026-09-20 03:23 3b02a41 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/82d90124_K7_packer_vs_trainer.json.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/raw/navhard/mirror/data_mirror_verify_navhard_logs.json | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +1/-1 |  |  | 2026-09-19 13:33 | 2026-09-20 03:23 3b02a41 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/b89af0ca_data_mirror_verify_navhard_logs.json.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-stage1-extract/raw/receipts/shard_6.json | staged-A+unstaged-M | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +3/-3 |  |  | 2026-09-21 14:55 | 2026-09-26 10:08 da27b38 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/4ef312e3_shard_6.json.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/SPEC.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +44/-0 |  |  | 2026-09-26 14:26 | 2026-09-26 14:17 51a4443 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/dbb0a6d0_SPEC.md.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/RESULT.md | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +5/-0 |  |  | 2026-09-26 22:36 | 2026-09-26 21:36 da8400b | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/d9047764_RESULT.md.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/LANDING_READY.txt | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +45/-0 |  |  | 2026-09-26 23:00 | 2026-09-26 21:36 da8400b | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/cbb29d01_LANDING_READY.txt.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/RESULT.md | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +109/-0 |  |  | 2026-09-26 22:59 | 2026-09-26 21:36 da8400b | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/d8ac0643_RESULT.md.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/LANDING_READY.txt | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +19/-0 |  |  | 2026-10-04 14:49 | 2026-10-04 08:34 a9b981d | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/e56a5560_LANDING_READY.txt.E_only_lines.txt |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/controls/KD_exact_dedup_cuda.json | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +2/-2 |  |  | 2026-10-05 00:27 | 2026-09-28 13:48 fd73a28 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/1558f147_KD_exact_dedup_cuda.json.E_only_lines.txt |
| Project Steering/BACKLOG.md | staged-M | CONFLICT (tip and E both moved off the E-HEAD base) | E OLDER than tip last change | +1/-198 | +1/-0 | +199/-1 | 2026-09-21 00:25 | 2026-10-18 11:14 780a025 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/ff43f824_BACKLOG.md.E_only_lines.txt |
| Project Steering/GOALS_AND_CLAIMS.md | staged-M+unstaged-M | CONFLICT (tip and E both moved off the E-HEAD base) | E OLDER than tip last change | +24/-4529 | +174/-22 | +4659/-2 | 2026-10-11 12:53 | 2026-10-18 10:04 3213b4f | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/c2888fa7_GOALS_AND_CLAIMS.md.E_only_lines.txt |
| Project Steering/MODEL_REGISTRY.md | staged-M+unstaged-M | CONFLICT (tip and E both moved off the E-HEAD base)  [E-only lines FLAGGED public_ip_ctx,public_ip_port: not exported] | E OLDER than tip last change | +10/-477 | +65/-0 | +536/-4 | 2026-10-11 12:54 | 2026-10-18 08:22 5448a05 |  |
| Project Steering/PI_DECISION_QUEUE.md | staged-M | LIKELY-STALE (E is a strict line-subset of the tip: tip only adds) | E OLDER than tip last change | +0/-567 | +216/-0 | +783/-0 | 2026-09-27 00:41 | 2026-10-18 10:22 336b708 |  |
| Project Steering/RETRACTION_LOG.md | staged-M | CONFLICT (tip and E both moved off the E-HEAD base)  [E-only lines FLAGGED public_ip_ctx,public_ip_port: not exported] | E OLDER than tip last change | +3/-1248 | +1425/-0 | +2673/-3 | 2026-09-28 13:51 | 2026-10-18 07:55 d6a4ce9 |  |
| Project Steering/SPEC_REFCV8_DRAFT.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +206/-773 |  |  | 2026-10-04 23:11 | 2026-10-12 01:57 d014ff2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/aa59f302_SPEC_REFCV8_DRAFT.md.E_only_lines.txt |
| Project Steering/TANITAD_PROGRAMME.md | staged-M | REAL-LOCAL-EDIT (tip unchanged since E-HEAD base; E has uncommitted edits) | E NEWER than tip last change | +9/-0 | +9/-0 | +0/-0 | 2026-09-27 17:20 | 2026-08-23 14:26 0e53d7e | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/40b3fb1e_TANITAD_PROGRAMME.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-a7-imagenet-knockout/RESULT.md | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +9/-3 |  |  | 2026-09-26 22:35 | 2026-09-19 11:59 d5d9bfc | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/3e2685de_RESULT.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-s1-collision-gate/RESULT.md | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +8/-3 |  |  | 2026-09-26 22:35 | 2026-09-26 11:40 1ee1c24 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/5e332fa0_RESULT.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-s1-collision-gate/raw/mutation_proof_p_runner.json | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +46/-30 |  |  | 2026-09-20 12:22 | 2026-09-20 10:45 f1bb51f | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/6119d62c_mutation_proof_p_runner.json.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/PREREG_AUX_DRIVABLE_S1.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +1/-143 |  |  | 2026-10-15 06:29 | 2026-10-15 18:12 421ffc3 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/e143c97f_PREREG_AUX_DRIVABLE_S1.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/S1_POD_RUNBOOK.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +1/-1 |  |  | 2026-10-15 06:22 | 2026-10-15 06:55 d257647 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/cec0f1f1_S1_POD_RUNBOOK.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/scorer_finetune.py | staged-A+unstaged-M | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +11/-363 |  |  | 2026-10-15 06:22 | 2026-10-18 09:11 c2d4042 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/46b7c0b1_scorer_finetune.py.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/test_aux_g12_bf16.py | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +4/-10 |  |  | 2026-10-15 06:19 | 2026-10-18 06:20 c968f99 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/8742223a_test_aux_g12_bf16.py.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/test_aux_s1_wiring.py | staged-A+unstaged-M | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +2/-7 |  |  | 2026-10-15 06:14 | 2026-10-17 13:51 e2d4422 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/3999a995_test_aux_s1_wiring.py.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/PERCEPTION_DESIGN.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +25/-0 |  |  | 2026-10-04 21:00 | 2026-10-04 14:46 49a0655 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/8246edd0_PERCEPTION_DESIGN.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/RESULT.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +159/-0 |  |  | 2026-10-05 05:21 | 2026-10-04 14:46 49a0655 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/4f76f2ba_RESULT.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pgrad.py | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +42/-4 |  |  | 2026-10-04 16:22 | 2026-10-04 14:46 49a0655 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/78829aff_pgrad.py.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/DESIGN.md | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +29/-6 |  |  | 2026-10-04 14:36 | 2026-10-04 13:17 2081f5d | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/54b418f8_DESIGN.md.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/LANDING_READY_WPB.txt | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +124/-19 |  |  | 2026-10-05 11:01 | 2026-10-04 23:56 0efda5f | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/d478bce3_LANDING_READY_WPB.txt.E_only_lines.txt |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/raw/d0_dose_response.json | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +1/-1 |  |  | 2026-10-04 14:35 | 2026-10-04 13:38 2b8216e | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/08ae32d5_d0_dose_response.json.E_only_lines.txt |
| TanitAD Research Lab/Library/LIBRARY.md | staged-M+unstaged-M | CONFLICT (tip and E both moved off the E-HEAD base) | E OLDER than tip last change | +115/-273 | +1064/-187 | +1247/-212 | 2026-10-17 14:38 | 2026-10-18 07:06 0a5958a | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/81668648_LIBRARY.md.E_only_lines.txt |
| TanitAD Research Lab/Library/library.json | staged-M+unstaged-M | CONFLICT (tip and E both moved off the E-HEAD base) | E OLDER than tip last change | +68/-535 | +4064/-111 | +4561/-141 | 2026-10-17 14:38 | 2026-10-18 07:06 0a5958a | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/2f3bd6fb_library.json.E_only_lines.txt |
| colab/MCP_SETUP.md | staged-M | REAL-LOCAL-EDIT (tip unchanged since E-HEAD base; E has uncommitted edits) | E NEWER than tip last change | +6/-0 | +6/-0 | +0/-0 | 2026-09-27 17:34 | 2026-08-16 15:34 05c37c0 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/23b81a0b_MCP_SETUP.md.E_only_lines.txt |
| data/nuplan_cam_calib.json | staged-A+unstaged-M | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +65/-4 |  |  | 2026-09-21 00:10 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/03f7d4f9_nuplan_cam_calib.json.E_only_lines.txt |
| stack/tests/test_map_head_hires.py | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +9/-365 |  |  | 2026-09-27 02:29 | 2026-09-27 12:42 1b8170e | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/a1b16bdc_test_map_head_hires.py.E_only_lines.txt |
| stack/tests/test_semantic_map_gt_fine.py | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +2/-3 |  |  | 2026-09-27 02:29 | 2026-09-27 05:06 cef9709 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/0db877cf_test_semantic_map_gt_fine.py.E_only_lines.txt |
| taniteval/taniteval/leaderboard/__main__.py | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +3/-1 |  |  | 2026-09-26 22:26 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/af3a509f___main__.py.E_only_lines.txt |
| taniteval/taniteval/leaderboard/build.py | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +60/-5 |  |  | 2026-09-26 22:27 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/d3f1508d_build.py.E_only_lines.txt |
| taniteval/taniteval/leaderboard/leaderboard_sources.json | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +1/-1 |  |  | 2026-09-26 22:48 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/ac4c1ca7_leaderboard_sources.json.E_only_lines.txt |
| taniteval/taniteval/leaderboard/render.py | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +119/-14 |  |  | 2026-09-26 22:48 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/caa9d0d6_render.py.E_only_lines.txt |
| taniteval/taniteval/leaderboard/sources.py | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +35/-4 |  |  | 2026-09-26 22:48 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/221ad561_sources.py.E_only_lines.txt |
| taniteval/taniteval/map_hires_metrics.py | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +12/-36 |  |  | 2026-09-27 02:29 | 2026-09-27 05:06 cef9709 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/4c9aa8db_map_hires_metrics.py.E_only_lines.txt |
| taniteval/tests/test_leaderboard.py | staged-A | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E NEWER than tip last change | +6/-6 |  |  | 2026-09-26 22:26 | 2026-09-26 11:32 24d5ce2 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/7620f64a_test_leaderboard.py.E_only_lines.txt |
| taniteval/tests/test_map_hires_metrics.py | untracked | NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) | E OLDER than tip last change | +2/-46 |  |  | 2026-09-27 02:29 | 2026-09-27 05:06 cef9709 | Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/526ae816_test_map_hires_metrics.py.E_only_lines.txt |

Verdict counts: NEW-DRAFT-vs-TIP (no E-HEAD base; E has lines the tip lacks) = 36; CONFLICT (tip and E both moved off the E-HEAD base) = 6; LIKELY-STALE (E is a strict line-subset of the tip: tip only adds) = 2; REAL-LOCAL-EDIT (tip unchanged since E-HEAD base; E has uncommitted edits) = 2; MACHINE-LOCAL (do not land) = 1.

How to read the verdicts: **REAL-LOCAL-EDIT** = the tip did not move off E-HEAD for this path and E carries uncommitted lines (strongest "newer" evidence; 3 small edits incl. one machine-local settings file). **CONFLICT** = both moved; the four `Project Steering/*` registers carry E-only additions (GOALS_AND_CLAIMS +24 lines, MODEL_REGISTRY +10, RETRACTION_LOG +3, BACKLOG +1 vs the tip) that the memory note says to re-apply from `git show <tip>:path`, never from E:. **NEW-DRAFT-vs-TIP** = no E-HEAD base (untracked or staged-A); E and tip diverge in both directions - compare `E mtime` with `tip last change`: where E is OLDER than the tip's last change the tip is almost certainly the landed, later version (pre-landing draft); where E is NEWER, the E lines are candidate unlanded edits. **LIKELY-STALE** = E is a strict line-subset of the tip. mtimes on a live working tree are a hint only.

## 5. MISSING-WT and index-only blobs

* `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/families_warmup_s1000_BLIND_stage1.json` (origin staged-A+unstaged-D+unstaged-M): index blob `a8904c369651` - on tip: False, in history: False, recovered from E:'s object store to `Project Steering/audits/2026-10-18-e-tree-stranded/recovered_index_blobs/a8904c3696516fc3bcbca2e5f49858ff6a978dd1.blob` (89581 B).

Staged entries whose worktree bytes differ from the staged blob (edited after `git add`): 109 (index blob on tip or in history for 48 of them). The other 61 staged blobs exist only in E:'s object store as EARLIER drafts; the worktree bytes that superseded them are classified above, so no separate action is recommended (list in `classes.json` -> `index_blob_mismatch`).

## 6. Files ignored by E:'s .gitignore (outside the population above)

Enumerated `ls-files -o -i --exclude-standard` excluding root `data/` (datasets, never committed), `_c/`, `.claude/`, the token file, caches and mp4: 2213 files -> {'MISSING-WT': 2, 'S1': 6, 'S4m': 41, 'S4': 2164}. Of the 2164 S4: 2,059 jpg + 13 js are `Media/2026-10-04-refcv7-replay/data/` replay assets (248.0 MB; media, regenerable); 92 are `taniteval/**/report/` derived outputs. **No ignored source file is stranded:** the package `code/fix/**/stack/tanitad/data/*.py` files hidden by the `data/` rule are 6 S1 + 7 S4m (content already on the tip at `stack/...`), including X10 `pose_sync.py`.
119 mp4 files (refcv6/refcv7 videos, u400 segments, teacher mini clips, Qwen review clips) are ignored by the `*.mp4` rule: bulk media, list only; the token file at the repo root exists on E: and was deliberately neither hashed nor read.

## 7. Agent-declared landing lists (`LANDING_READY*.txt`) vs reality

74 manifests are present in the E: tree. 13 of them are themselves **not on the tip** (S4/S2b). For manifests that list `md5/blob + path`, entries were compared with the tip. Manifests whose entries are on E: but absent from the tip path-wise (the lists the Master Mind never finished landing):

| manifest | class | entries | tip-exact | tip-differs | path-only (on tip) | NOT on tip | of which present on E: |
|---|---|---:|---:|---:|---:|---:|---:|
| FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/LANDING_READY_X10.txt | S4 | 28 | 22 | 4 | 0 | 2 | 2 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/LANDING_READY.txt | S1 | 437 | 0 | 0 | 434 | 3 | 3 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/LANDING_READY.txt | S1 | 659 | 0 | 0 | 658 | 1 | 1 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/LANDING_READY.txt | S2b | 180 | 0 | 0 | 175 | 5 | 5 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/LANDING_READY.txt | S2b | 164 | 0 | 0 | 164 | 0 | 0 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/LANDING_READY_A7IMPL.txt | S4 | 8 | 8 | 0 | 0 | 0 | 0 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/LANDING_READY_G0DIAG50400.txt | S4 | 23 | 0 | 0 | 23 | 0 | 0 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/LANDING_READY_X8.txt | S4 | 68 | 0 | 0 | 68 | 0 | 0 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/LANDING_READY_NAVSIM50K_LEGAL.txt | S4 | 186 | 13 | 0 | 0 | 173 | 173 |
| FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend/LANDING_READY_NAVSIM_THOR.txt | S4 | 688 | 371 | 0 | 0 | 317 | 317 |
| Media/2026-10-04-refcv7-final-reel/LANDING_READY.txt | S4 | 6 | 0 | 0 | 6 | 0 | 0 |
| Project Steering/LANDING_READY_SPECR8.txt | S4 | 2 | 0 | 2 | 0 | 0 | 0 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/LANDING_READY.txt | S1 | 453 | 0 | 0 | 430 | 23 | 10 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/LANDING_READY.txt | S1 | 114 | 0 | 0 | 113 | 1 | 1 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/LANDING_READY.txt | S1 | 50 | 0 | 0 | 48 | 2 | 2 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-launch-gate/LANDING_READY.txt | S1 | 225 | 0 | 0 | 215 | 10 | 1 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-step-cost/LANDING_READY.txt | S4 | 240 | 0 | 0 | 240 | 0 | 0 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following/LANDING_READY_A5.txt | S4 | 0 | 0 | 0 | 0 | 0 | 0 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/LANDING_READY_WPD.txt | S4 | 15 | 0 | 0 | 0 | 15 | 0 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-fixes/LANDING_READY_WPC.txt | S4 | 17 | 0 | 0 | 0 | 17 | 0 |
| TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/LANDING_READY_WPB.txt | S2b | 125 | 20 | 35 | 0 | 70 | 0 |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/LANDING_READY_P4.txt | S4 | 46 | 43 | 0 | 0 | 3 | 3 |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/LANDING_READY_D3.txt | S4 | 47 | 47 | 0 | 0 | 0 | 0 |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/LANDING_READY_D4.txt | S4 | 15 | 0 | 0 | 0 | 15 | 0 |
| TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/LANDING_READY_D5.txt | S4 | 3 | 3 | 0 | 0 | 0 | 0 |
| products/P4-training-pipelines/2026-09-27-v7f-refav1-state/LANDING_READY.txt | S1 | 1548 | 0 | 0 | 1503 | 45 | 19 |

The two large unlanded slices are `2026-09-28-refcv7-standard-tests/navsim` (173 listed files absent from the tip) and `2026-10-04-navsim-thor-backend` (317 absent), plus three smaller ones (X10 pose-timing 2, P4 3, suite-runnability-audit 5). Counts of manifests whose entry format could not be parsed are shown as 0 entries.

## 8. Reproduction

`scripts/classify.py` (S1-S4 by blob id; `--redo-missing`, `--ignored` modes), `scripts/stage2.py` (scan + describe), `scripts/rescan.py` (secret-pattern refinement), `scripts/stage3a.py` (landing-manifest cross-reference), `scripts/stage3b.py` (S2b diffs), `scripts/scanlib.py`, `scripts/report.py` (this document). Run with `python -X utf8 -I` (isolated mode ignores `PYTHONIOENCODING`; stdout is cp1252 otherwise).

## 9. Landing notes for the Master Mind

1. **Order:** take `paths_by_tier.json` P1 first (1543 files, 101.2 MB: code, docs, small json/csv/log), then P2 (381 files, 422.7 MB). P3 (`paths_P3_optional.json`, 81 files, 847.9 MB, 5-20 MB raw data) only as a separate commit if the raw evidence is wanted in git; the >20 MB set and all checkpoints/media stay on D:/E: (list: section 3 and `s4_list.csv`) - they need a durable second copy outside this one drive (E: is an external disk; CLAUDE.md: GitHub is the durable copy).
2. **Scrubbed copies:** the 11 LAND-HOLD-SCRUB files were redacted mechanically (public IPs, ssh ports, token-file name; 1-5 lines each, counts in the table above), re-scanned clean, and are listed in `paths_scrubbed.json` with source `OUT/scrubbed/<repo path>`. They are NOT byte-copies of E:; the cleanup `.ps1/.sh` scripts and the pod `.sh` helpers are archival records after redaction, not runnable.
3. **Long paths:** 75 LAND files have an absolute E: path longer than 259 characters (FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim: 75); the lander needs `core.longpaths=true` and the extended-length path prefix, or these should be deferred.
4. **Integration (not solved by landing):** the 29 package-local code variants listed under section 3 (`code/fix*/stack/...`) are *not* applied to `stack/` on the tip; landing them only archives them next to the report. Applying them is a code-integration decision (base = tip, never E:), notably `refcv8-tactical-conditioning/code/fix_P7/` (launch_gate, refc_v3_train, refcv8_conditioning, refcv8_train ...), `refcv7-box-head/code/fix_r6/` and `v7f-refav1-state/v7f_r1r4|v7f_r3/`.
5. **Candidate edits:** nothing in section 4 was landed. Rows whose E file is NEWER than the tip's last change to the same path AND carry >= 20 E-only lines (mtime is a secondary hint; the E-only lines are in `candidate_edits/`): `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/RESULT.md` (+159); `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/LANDING_READY_WPB.txt` (+124); `taniteval/taniteval/leaderboard/render.py` (+119); `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/RESULT.md` (+109); `taniteval/taniteval/leaderboard/build.py` (+60); `TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-s1-collision-gate/raw/mutation_proof_p_runner.json` (+46); `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/LANDING_READY.txt` (+45); `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/SPEC.md` (+44); `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/code/pgrad.py` (+42); `taniteval/taniteval/leaderboard/sources.py` (+35); `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/DESIGN.md` (+29); `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/PERCEPTION_DESIGN.md` (+25). The leaderboard code (`taniteval/taniteval/leaderboard/*`, staged-A on 2026-09-26 22:26-22:48, tip commit 11:32 the same day) is the clearest PROBABLE case of a later E-side edit that is not on the tip (UNVERIFIED until a human diffs it). The `Project Steering/*` registers must be re-edited from `git show <tip>:<path>`, never from E:.
6. **Provenance caveat:** E: is a live working tree; `recent24h` flagged 0 files. Re-run `scripts/classify.py` immediately before landing if more than a few hours pass.

## 10. Deliverable manifest

| artifact | location | exists in |
|---|---|---|
| this audit | `Project Steering/audits/2026-10-18-e-tree-stranded/AUDIT.md` | OUT dir only (to be landed) |
| per-path classes (non-S1) | `Project Steering/audits/2026-10-18-e-tree-stranded/classes.json` | OUT dir only |
| per-file S4 list with recommendation + one-line description | `Project Steering/audits/2026-10-18-e-tree-stranded/s4_list.csv` | OUT dir only |
| optional companions (content already on tip) | `Project Steering/audits/2026-10-18-e-tree-stranded/companions.csv` | OUT dir only |
| E-only lines of the candidate edits | `Project Steering/audits/2026-10-18-e-tree-stranded/candidate_edits/*.E_only_lines.txt` | OUT dir only |
| recovered index-only blob | `Project Steering/audits/2026-10-18-e-tree-stranded/recovered_index_blobs/` | OUT dir only (original is in E:'s object store) |
| scripts | `Project Steering/audits/2026-10-18-e-tree-stranded/scripts/` | OUT dir only |
| scrubbed copies of the HOLD files | `OUT/scrubbed/` + `paths_scrubbed.json` | OUT dir only (originals on E:) |
| LAND list | `OUT/paths.json` (P1+P2+recovered blob+audit package), `paths_by_tier.json`, `paths_P3_optional.json` | OUT dir only |
| the stranded files themselves | E:\Projects\TanitAD (= D:) working tree | **ONE place only until landed** (E: is an external disk) |

