# S0 search-bank inputs: where each input of the 117 P2-S0 logs lives (read-only probe, 2026-10-20)

Mirror tip pinned once: `901850561cc741695b6893fe1036791e94ca8206` (`agent/arch-inf-20260803`). CPU only, no ssh, no GPU, nothing written outside `C:/Users/Admin/par18/s0probe/`. Machine-readable twin: `s0_inputs.json` (per-log rows, probes, gap).
Evidence classes: MEASURED = read on this box today (artifact named); INHERITED = from another agent or doc, not re-run; ESTIMATED = a stated extrapolation; UNVERIFIED = not checked.

## 1. Headline

The builder's "22 of 117" is a **stale pre-S0 snapshot** (`srttc_s0/run/plan/NOTES.md` and `plan/availability.json`, 2026-10-18: 25 logs with a DB, 38 with four zips, 22 with both). Since then S0 ran (it staged 76 DBs and frames from the pod, banked, and deleted them), and the pod-stage stream (`C:/Users/Admin/par18/podstage`, HANDOFF 2026-10-19) re-staged the same 76 DBs into `splits/` and 7,864 frames for SH5.

| input | covered of the 117 logs / 8,922 tokens | verdict |
|---|---|---|
| (b) nuPlan DB | **117 / 117 logs, 8,922 / 8,922 tokens found in `lidar_pc`** (15.7 GB) | builder claim FALSE. MEASURED |
| (b) maps for PDM labelling | all 4 cities local (`nuplan-maps-v1.0`); S0 logs use sg-one-north 11, us-nv-las-vegas-strip 77, us-ma-boston 19, us-pa-pittsburgh-hazelwood 10 | complete. MEASURED |
| (b) NAVSIM metric cache for navtrain | none local; not needed by the repo labeller (it rebuilds the cache from DB + maps) | pod has one (text only). MEASURED / INHERITED |
| (a) camera frames, 4 cams | **41 logs / 2,782 tokens complete (31.2 %)**; **76 logs / 6,140 tokens have none** (0 partial) | builder claim wrong in count, right in direction. MEASURED |
| (c) P2 S0 bank | **complete, hash-verified**: 69 files, 23 units, 8,922 own-fan tokens, 15,808 stored-fan sets | EXISTS, but see "what it does not hold" |
| (c) P2 own-fan trajectories and labels on S0 tokens | **0 of 8,922** (bank dumped `--lean`; `proposals` arrays empty in all 23 `fan.npz`) | builder claim TRUE for this object. MEASURED |
| (d) per-token P2 visual ctx cache | **816 tokens / 38 logs** (3.2 GB, C-FID pass); none for the other 8,106 | partial. MEASURED |
| plan json | `master_plan.json` + 23 per-unit `batch_tokens.json` (union 8,922 tokens) + sh5 `plan_full` (2,782) | builder claim FALSE (token_root/db inside the per-unit plans are stale, see Plans in section 2). MEASURED |

**The gap that blocks a full 117-log search bank is frames only: 76 logs, 6,140 tokens, 24,560 frames, 4.91 GB to fetch from the pod (MEASURED sizes), 137 of those tokens recoverable locally. Everything else (DB, maps, plans, P2 checkpoint, logits and labels on the stored fans) is present.** Labelled P2 own-fan data on S0 tokens does not exist anywhere and must be created by a P2 forward plus PDM labelling.

## 2. Where each input lives

### (a) Camera frames (CAM_F0, CAM_L0, CAM_R0, CAM_B0; one nearest-timestamp frame per camera, within 120 ms, 4 frames per token)

| root | S0 logs | note |
|---|---|---|
| `D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/frames_full` | **41 / 41** | 7,864 jpgs, 1,646,644,900 B, `<log>/<CAM>/<hash>.jpg`; staged from the pod on 2026-10-19 by the pod-stage stream for SH5; all present, sizes equal, 400 random sha256 equal (0 bad). Listing: `pod_stage_2026-10-19/frames_full_sha256.txt` |
| `D:/Projects/TanitAD/data/refe_proxy/pixels` | 38 (of 407 navtrain logs) | zip per (log, camera), a stratified frame sample |
| `D:/Projects/TanitAD/data/refe_m5/pixels` | 15 (of 206) | loose jpgs |
| `D:/Projects/TanitAD/data/nuplan-camera/mini` | 3 (of 64) | CAM_F0 only |
| `D:/Projects/TanitAD/data/nuplan-camera/scenarios4` | 1 (of 7) | |
| `D:/TanitAstraResearch/data/e024_anchors_20261010/sensors`, `e024_sequence_images_v1/stage_*`, `e024_report_images_v1`, `e023_sequences_20261010` | 12 to 15 | fragments only: 573 of the 24,560 frames the 76 gap logs need, no complete token; anchors are byte-identical to the proxy zips (66/66) |
| `refe_navtrain`, `refe_navtrain10`, `refe_navtest`, `refe_m5r`, `refe_consistent`, `nuplan-camera/scenarios`, `nuplan-camera/probe`, `videos`, `heldout_frames` | 0 | name-scanned to depth 3 (`roots13.json`) |
| `_rlfan_work/atol_cal{,2}/pix_{jpeg,hflip,bgr_rgb}` | 2 to 4 | calibration variants (re-encoded, flipped, BGR), NOT original frames; excluded |
| `srttc_s0_stage/frames` | 0 | empty: S0 deleted its staged copies after banking each unit |
| `D:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz` (4,496,159,137 B) | 4 (2 are gap logs) | gzip tar of pod shard 25; `members.txt` lists every file. The two gap logs `2021.06.14.14.25.15_veh-26_05108_05312`, `2021.06.14.16.48.02_veh-12_04978_05337` need 548 frames and all 548 are in it: **137 tokens recoverable with no network** (119.7 MB), sequential extract only |

Validation: my read-only re-implementation of `planner.FrameResolver` (DB `image` rows, nearest per camera within 120 ms, basename present in a root) gives the same 41 logs and 2,782 tokens as SH5's independent plan (`sh5_ctx_2026-10-19/plan_full/coverage.json`, built with the real resolver). MEASURED.

### (b) nuPlan DB, maps, metric cache

- DBs: `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/<log>.db` (101 logs) and `.../splits/trainval/<log>.db` (16 logs); 3 of them also under `.../nuplan/dblinks/driverl_val14` (link copies). `find` over D:/Projects, D:/tanitad, D:/hold_temporal, D:/TanitAstra*, D:/Archive, D:/thor_archive, D:/qland_work_archive_sept, D:/tmp_fixtree, D:/lgs, D:/t6, D:/refcv*_kit and C:/Users/Admin (3,130 DBs) found all 117; each opened read-only with all its S0 tokens present in `lidar_pc` (8,922 / 8,922). MEASURED.
- 76 DBs (11,324,555,264 B) staged from the pod on 2026-10-19 by the pod-stage stream now sit in `splits/` (`pod_stage_2026-10-19/dbs_76_sha256.txt`; S0 itself had deleted its staged copies): all 76 size-equal, the 12 smallest sha256-equal, and all 76 equal (size + sha256) to the S0 stage manifests. MEASURED.
- Maps: `D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0/{sg-one-north,us-ma-boston,us-nv-las-vegas-strip,us-pa-pittsburgh-hazelwood}/*/map.gpkg` (186 to 524 MB each) and `nuplan-maps-v1.1.zip`.
- NAVSIM metric caches found by name on D: and C: (second location): Astra `reusable_native_cache` (256 entries), `eval_fixed10` (512), `teacher_v2` (512), `teacher_v1` (469), `e031` (125), the `official_metric_cache.tar` (257 members), `D:/Archive/devbox-C/navsim/exp/*`, `C:/Users/Admin/navsim-crun/exp/*`. S0-token overlap: 0 / 22 / 3 / 4 / 0 tokens (29 at most), 0 S0 logs elsewhere. No `*.pkl.xz` navtrain cache. The labeller used by the search chain (`label_stepB.py` -> `label_perturbations.label_plans` -> `navsim_pdm_targets.metric_cache_parts`) rebuilds the cache from DB + maps, so DB + maps are what labelling needs (read from the staged code). MEASURED / code read.

### (c) What P2's S0 bank is, and which `bank_dir` the packages used

`bank_dir = C:/Users/Admin/srttc_s0/work/bank` (stated in the repo text of P6-READ-PREP `READY_P6_READ.md`, PICKRULE-AUDIT-NC, HEADROW-PROBE, EP-R1UB, `sel-rank/S0_HANDOFF.md`; `ep_admission.py selftest-real --p2-bank` points here). `ranked_fit_logs.tsv` = `C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv` (sha256 `a61275037a3d...`, equal to the pin in `ep_admission.py`). 69 files sha256-equal to `MANIFEST.sha256.json`, 0 bad. Single copy, on C:.

| artifact | holds | does NOT hold |
|---|---|---|
| `<unit>.sets.npz` x 23 (what `ep_admission.py` loads) | 15,808 sets (key log, token, step, rank): `ckpt_step`, `feat` [S,64,256] fp16 (score_head input), `logits` [S,64,6] (P2's scorer on the STORED fan), `top8`, `labels6` [S,64,6] (NC, DAC, EP, TTC, C, DDC) | trajectories |
| `<unit>.fan.npz` x 23 | P2's OWN fan for 8,922 tokens: `logits` [n,64,6], `pick` [n], `fingerprint` | `proposals`, `proposals20` (empty), labels |
| `C:/Users/Admin/srttc_s0/work/sets/<log>.npz` x 117 | `traj3` [S,64,20,3], `labels6` for the 15,808 stored sets; all 15,808 join the bank by key | these are on-policy fans of EARLIER checkpoints (1,281 step/rank/ckpt_step combos), not P2's own fan; single copy on C: |
| `work/sets_pod/{sets/onpolicy_selected.jsonl, pdm/pdm_selected.jsonl}` (997 MB) | pod text copies of the stored sets (16,946 lines) and their PDM labels (15,808 lines) | |
| `D:/Projects/TanitAD-artifacts/headrow_probe_2026-10-19/oof_proposals_f10.npz` | derived copy over the 15,808 sets: P2 logits, labels6, top8, pick, OOF head rows | trajectories (despite the name) |

Searched for labelled P2-own-fan trajectories on S0 tokens and found none: `p2_full_table` (12,146 tokens, 0 in S0), `p2_heldout_table_full` (0), `p2_zero_tail_table` (0), `srttc_s0/work/dgdump` (4 tokens, none in S0), `l2_offprop/feats.npz` (400 Step-A tokens, 0). The P2 checkpoint `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` (1,288,484,600 B, sha256 `ea31db22...`) is local and equals the S0 pin. MEASURED.

### (d) Per-token ctx or visual-feature cache

`D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19/ctx_local/ctx.npy`: P2 `visual_ctx` as bf16 bits, shape (816, 7680, 256), 3,208,642,688 B, 816 tokens in 38 logs (all in S0, all done, `bad` empty), checkpoint `p2/model_final.pt`, C-FID: 1,444 of 1,444 sets reproduce the stored S0 logits within 1e-3 (max abs 0.0). `ctx_manifest.json` has the token list. Not found for any other S0 token. `refe_trunk_cache/d7c59f4f` is the epoch-15 snapshot (wrong checkpoint) and `l2_offprop/feats.npz` is 400 Step-A tokens: neither is usable. MEASURED.

### Plans (per-unit plans and the master plan)

`srttc_s0/run/plan/master_plan.json` (117 logs, 8,922 tokens, 15,808 sets); `srttc_s0/run/batches/<unit>/batch_tokens.json` for 23 units (no token repeated, union = 8,922, format `{tokens, token_log, token_root, token_ts_us, db}`); SH5 `plan_full/plan.json` (2,782 tokens, current roots). In the per-unit plans `token_root` and `db` point at the deleted S0 staging dirs, so re-resolve them. In the repo at the mirror tip (`raw/2026-10-18-sel-rank/raw/{s0,a2_subsample}`): `master_plan.json` and `ranked_fit_logs.tsv` (both sha256-equal to the local files), `S0_SUMMARY.json`, 23 batch `audit.json`, 19 stage manifests, `availability.json` and `NOTES.md` (the pre-S0 snapshot behind the "22"). NOT in the repo: the bank npz files, `work/sets/*.npz`, `batch_tokens.json`, the ctx cache, the frames. The bank is a single copy on C: (name search for three unit ids on D:/Projects, D:/tanitad, D:/Archive and C:/Users/Admin found no second copy). MEASURED.

## 3. The gap, and what closing it costs

| missing | scope | size | basis |
|---|---|---|---|
| camera frames | **76 logs, 6,140 tokens (68.8 %), 24,560 frames needed, 23,987 absent**; they carry 10,867 of the 15,808 sets | **4.911 GB to fetch** (5.024 GB for the whole logs) | MEASURED pod file sizes recorded in the S0 stage manifests (2026-10-18); cross-check ESTIMATED from the 41 present logs, 837,562 B per token x 6,140 = 5.143 GB |
| of which free locally | 2 logs, 137 tokens, 548 frames in `navtrain_current_25.tgz` | 119.7 MB | `members.txt` lists them (not extracted: read-only) |
| DB | 0 logs | 0 | if the 76 re-staged DBs were lost: 11.32 GB to re-stage |
| maps | none | 0 | |
| P2 own-fan trajectories and labels | 8,922 of 8,922 tokens | created, not fetched | one P2 forward per token plus PDM labelling of 64 plans per token. S0 dump MEASURED 14,173 s for 8,922 tokens (3.94 h GPU, `build_candidate_bank.md`); scaling gives about 1.2 h for the 2,782 tokens that have frames (ESTIMATED) |
| ctx cache | 8,106 tokens | about 29.7 GiB if built (3.75 MiB per token, ESTIMATED), derived by GPU, not fetchable | |

**Disk (MEASURED today):** C: 4.9 GB free, D: 0 GB free. The 4.9 GB frame fetch fits on neither comfortably; free space on D: first (PI-only delete per memory) or land it on C: in per-log zip containers.

**Pod paths, from repo text only (no ssh; existence on the pod NOT re-checked, INHERITED from the S0 stage manifests):** frames `/workspace/data/navtrain_pixels/navtrain_current_N/<log>/<CAM>/<hash>.jpg` (the 76 gap logs sit in 28 of the 32 shards; N per log in `s0_inputs.json` `per_log[].pod_pixel_shard`); DBs `/workspace/data/navtrain_dbs/<log>.db`; stored sets `/workspace/data/refe_navtrain/onpolicy/sets`; their PDM labels `/workspace/data/refe_sft2/train_pdm`; NAVSIM metric cache `/workspace/data/refe_navtrain/metric_cache/<log>/<token>.pkl.xz`. Sources in repo: `raw/2026-10-18-sel-rank/S0_HANDOFF.md`, `raw/2026-10-16-p6-d7-pdm-route-rank/{HANDOFF.md,code/refe/p6_pod.sh:32}`, `srttc_s0_stage.py`, `srttc_s0_podsets.py`.

**Consequences for the fallback bank (named, not decided):** (1) a search bank on the 41 complete logs (2,782 tokens, 4,941 sets, SH5 `plan_full`) needs no fetch, but it is not the gate's S0 geometry (the epadm-searchbank brief says `ep_admission.py` C-KEYS wants a bijection onto P2's 117-log, 15,808-set bank, INHERITED). (2) The full S0-geometry bank needs the 4.91 GB frame fetch (pod reads need coordinator approval under the P6 overlap rule, INHERITED from the S0 stager docstring and the pod-stage HANDOFF) plus disk. (3) Either way P2-own-fan labels must be generated first; none exist.

## 4. Probes run (including those that found nothing)

1. P01 srttc_s0/run/plan/availability.json + NOTES.md (source of the "22"): pre-S0 snapshot
2. P02 find *.db named like nuPlan logs under D:/Projects, D:/tanitad, D:/hold_temporal, D:/TanitAstra*, D:/Archive, D:/thor_archive, D:/qland_work_archive_sept, D:/tmp_fixtree, D:/lgs, D:/t6, D:/refcv*_kit and C:/Users/Admin: 3,130 DBs, all 117 S0 logs found
3. P03 sqlite open (read-only) of each S0 DB, 8,922/8,922 tokens found in lidar_pc; camera/image tables read; 76 re-staged DBs size-equal, 12 smallest sha256-equal, all 76 size+sha256-equal to the S0 stage manifests
4. P04 13 local frame roots (READY.md sec I list) name-scanned to depth 3 for S0 log names (roots13.json): hits only in refe_proxy/pixels (38 logs), refe_m5/pixels (15), nuplan-camera/mini (3), scenarios4 (1); 0 in scenarios, probe, refe_navtrain, refe_navtrain10, refe_navtest, refe_m5r, refe_consistent, heldout_frames, videos
5. P05 zip/dir membership resolution of every S0 token x 4 cameras against 18 roots incl. sh5 frames_full and Astra image dirs
6. P06 find log-named dirs and *_CAM_*.zip on D: roots and C:/Users/Admin (5,080 hits) grouped by parent: found sh5 frames_full (41/41 S0), Astra e023/e024 dirs (fragments), _rlfan_work calibration copies
7. P07 sha256: 400 random frames_full jpgs vs frames_full_sha256.txt (0 mismatch); all 7,864 present with equal size
8. P08 openscene_probe/members.txt: shard 25 holds 4 S0 logs, 2 of them gap logs, every needed frame listed
9. P09 metric cache search by name on D: and C:/Users/Admin: Astra caches (5 dirs), devbox-C warmup, navsim-crun navtest/navhard: S0 token overlap 22/3/4/0/0 (Astra e024/e025/e026/e023/e031), 0 S0 logs elsewhere; no *.pkl.xz navtrain cache
10. P10 bank: 69 files sha256 vs MANIFEST (0 bad); 8,922 fan tokens, 15,808 sets, 117 logs; proposals arrays empty in all 23 fan.npz; work/sets 117 stores / 15,808 sets all joined by key
11. P11 token scan of p2_full_table, p2_heldout_table_full, p2_zero_tail_table, dgdump, headrow oof, sh5 inputs, l2_offprop feats for S0 tokens
12. P12 sh5 ctx_local: 816 tokens (38 logs) all done, C-FID pass; refe_trunk_cache is the epoch-15 snapshot (wrong ckpt)
13. P13 repo text at the mirror tip: pod paths (git grep navtrain_pixels / navtrain_dbs / onpolicy/sets / train_pdm / metric_cache)
14. NOT DONE: ssh to any pod (forbidden), GPU use, any download; pod-side existence of the 76 logs was NOT re-checked (INHERITED from the S0 stage manifests, 2026-10-18)

## 5. Per-log table

`tok` = S0 tokens, `4cam` = tokens with all four cameras local now, `ctx` = tokens with a cached P2 ctx, `shard` = pod `navtrain_current_N`. Complete logs (41):

| log | tok | sets | 4cam | ctx | DB MB |
|---|---|---|---|---|---|
| 2021.06.14.18.33.41_veh-35_04905_05090 | 53 | 99 | 53 | 0 | 111 |
| 2021.06.14.16.32.09_veh-35_03129_03220 | 49 | 79 | 49 | 15 | 36 |
| 2021.10.05.07.38.12_veh-50_00245_00433 | 93 | 159 | 93 | 35 | 23 |
| 2021.07.24.20.37.45_veh-17_00015_00375 | 127 | 235 | 127 | 49 | 247 |
| 2021.06.14.13.27.42_veh-35_01854_01994 | 53 | 101 | 53 | 16 | 67 |
| 2021.08.17.13.15.12_veh-45_02025_02103 | 82 | 153 | 82 | 24 | 5 |
| 2021.06.23.22.05.48_veh-16_00602_00800 | 117 | 207 | 117 | 35 | 102 |
| 2021.09.14.18.43.41_veh-45_02296_02477 | 78 | 130 | 78 | 28 | 32 |
| 2021.07.16.00.24.14_veh-38_00367_01154 | 126 | 224 | 126 | 38 | 567 |
| 2021.06.23.14.06.20_veh-26_02505_02775 | 64 | 120 | 64 | 0 | 141 |
| 2021.06.09.17.23.18_veh-38_03302_03414 | 5 | 10 | 5 | 1 | 67 |
| 2021.06.14.13.27.42_veh-35_01160_01331 | 32 | 59 | 32 | 9 | 95 |
| 2021.08.31.17.42.52_veh-40_00833_00953 | 17 | 34 | 17 | 6 | 26 |
| 2021.08.24.18.30.46_veh-08_02605_02732 | 48 | 88 | 48 | 19 | 10 |
| 2021.09.14.20.42.30_veh-45_00464_00579 | 59 | 107 | 59 | 20 | 22 |
| 2021.06.09.17.23.18_veh-38_05423_05550 | 35 | 66 | 35 | 10 | 58 |
| 2021.06.07.12.42.11_veh-38_02445_02843 | 42 | 73 | 42 | 16 | 216 |
| 2021.10.06.17.43.07_veh-28_01587_01694 | 36 | 62 | 36 | 11 | 26 |
| 2021.06.14.14.03.45_veh-38_01155_01358 | 22 | 40 | 22 | 7 | 107 |
| 2021.09.15.13.16.40_veh-28_01473_01612 | 16 | 28 | 16 | 5 | 39 |
| 2021.06.23.16.52.00_veh-26_00828_01032 | 11 | 22 | 11 | 3 | 126 |
| 2021.06.14.13.27.42_veh-35_02283_02603 | 103 | 180 | 103 | 30 | 222 |
| 2021.10.05.08.11.15_veh-50_00970_01211 | 169 | 292 | 169 | 65 | 30 |
| 2021.10.05.07.10.04_veh-52_00596_00663 | 31 | 53 | 31 | 9 | 5 |
| 2021.08.31.14.01.15_veh-40_01284_01345 | 4 | 8 | 4 | 1 | 14 |
| 2021.10.11.05.34.05_veh-50_01281_01692 | 203 | 339 | 203 | 59 | 52 |
| 2021.06.23.21.56.29_veh-35_00220_00936 | 229 | 423 | 229 | 0 | 488 |
| 2021.06.08.14.35.24_veh-26_01989_02235 | 79 | 137 | 79 | 30 | 197 |
| 2021.10.05.06.24.06_veh-50_00717_01300 | 244 | 443 | 244 | 89 | 56 |
| 2021.08.17.17.17.01_veh-45_02098_02251 | 46 | 81 | 46 | 13 | 24 |
| 2021.08.24.18.30.46_veh-08_02327_02583 | 122 | 211 | 122 | 42 | 20 |
| 2021.06.09.12.51.31_veh-35_05024_05275 | 25 | 46 | 25 | 7 | 247 |
| 2021.06.09.17.23.18_veh-38_00016_00120 | 6 | 11 | 6 | 2 | 73 |
| 2021.10.05.06.24.06_veh-50_01311_01409 | 82 | 136 | 82 | 29 | 8 |
| 2021.07.16.21.17.55_veh-26_01392_01488 | 15 | 24 | 15 | 4 | 74 |
| 2021.10.05.06.24.06_veh-50_00431_00527 | 77 | 137 | 77 | 27 | 9 |
| 2021.06.14.16.32.09_veh-35_03635_03792 | 2 | 4 | 2 | 1 | 189 |
| 2021.06.09.11.54.15_veh-12_05108_05331 | 89 | 166 | 89 | 26 | 138 |
| 2021.06.09.14.50.36_veh-26_02081_02143 | 7 | 9 | 7 | 2 | 76 |
| 2021.10.05.04.38.41_veh-50_00576_00721 | 67 | 116 | 67 | 28 | 13 |
| 2021.06.09.14.58.55_veh-35_01095_01484 | 17 | 29 | 17 | 5 | 329 |

Gap logs (76): frames absent, DB present, no ctx. `MB` = pod frame bytes recorded by S0 for the log.

| log | tok | sets | shard | frames MB | note |
|---|---|---|---|---|---|
| 2021.06.23.15.56.12_veh-16_01308_04289 | 819 | 1495 | 13 | 665 |  |
| 2021.07.09.02.42.50_veh-35_00038_02629 | 651 | 1159 | 27 | 503 |  |
| 2021.07.16.00.51.05_veh-17_00023_01331 | 434 | 785 | 19 | 334 |  |
| 2021.08.17.16.48.45_veh-43_02070_02652 | 245 | 408 | 29 | 175 |  |
| 2021.06.14.14.03.45_veh-38_04499_05170 | 216 | 369 | 18 | 190 |  |
| 2021.06.23.21.56.29_veh-35_00947_01581 | 207 | 386 | 27 | 150 |  |
| 2021.06.14.16.48.02_veh-12_03472_03779 | 156 | 277 | 6 | 135 |  |
| 2021.10.11.08.31.07_veh-50_01184_01318 | 133 | 230 | 16 | 134 |  |
| 2021.08.17.16.48.45_veh-43_01060_01405 | 122 | 191 | 7 | 90 |  |
| 2021.07.16.16.27.22_veh-26_05058_05383 | 120 | 209 | 20 | 109 |  |
| 2021.06.14.17.26.26_veh-38_01293_01488 | 112 | 199 | 6 | 92 |  |
| 2021.06.23.16.54.19_veh-35_03705_04009 | 111 | 193 | 26 | 82 |  |
| 2021.06.23.20.00.35_veh-35_03660_04140 | 109 | 187 | 18 | 80 |  |
| 2021.08.17.13.10.50_veh-08_00726_01027 | 99 | 171 | 28 | 75 |  |
| 2021.06.09.17.23.18_veh-38_01151_01532 | 97 | 178 | 4 | 90 |  |
| 2021.06.09.14.58.55_veh-35_02861_03037 | 91 | 162 | 10 | 72 |  |
| 2021.09.15.15.34.53_veh-28_01639_01805 | 89 | 153 | 31 | 77 |  |
| 2021.06.09.20.26.11_veh-35_00247_00529 | 87 | 153 | 11 | 73 |  |
| 2021.07.16.18.06.21_veh-38_04311_04460 | 85 | 150 | 20 | 70 |  |
| 2021.08.17.17.17.01_veh-45_01796_02069 | 83 | 145 | 29 | 71 |  |
| 2021.06.14.18.42.45_veh-12_02737_02967 | 81 | 151 | 13 | 70 |  |
| 2021.10.06.17.08.46_veh-28_01298_01548 | 81 | 118 | 24 | 69 |  |
| 2021.05.12.19.36.12_veh-35_02079_02176 | 80 | 151 | 1 | 72 |  |
| 2021.10.11.02.57.41_veh-50_01028_01289 | 79 | 133 | 24 | 63 |  |
| 2021.06.14.14.25.15_veh-26_05108_05312 | 77 | 140 | 25 | 65 | in tgz25 (free) |
| 2021.10.01.19.16.42_veh-28_03887_04040 | 75 | 135 | 23 | 59 |  |
| 2021.10.01.17.52.06_veh-28_00748_00952 | 72 | 117 | 15 | 63 |  |
| 2021.07.16.21.17.55_veh-26_03254_03336 | 71 | 130 | 20 | 64 |  |
| 2021.06.14.16.48.02_veh-12_04783_04967 | 65 | 110 | 6 | 54 |  |
| 2021.07.09.15.53.28_veh-38_04778_04886 | 62 | 106 | 27 | 55 |  |
| 2021.06.14.16.48.02_veh-12_04978_05337 | 60 | 109 | 25 | 55 | in tgz25 (free) |
| 2021.09.15.12.32.43_veh-28_01513_01697 | 55 | 101 | 30 | 40 |  |
| 2021.06.09.14.50.36_veh-26_01124_01198 | 53 | 102 | 4 | 50 |  |
| 2021.06.09.14.50.36_veh-26_03208_03299 | 51 | 94 | 4 | 45 |  |
| 2021.06.14.11.44.56_veh-35_02696_02932 | 51 | 93 | 5 | 44 |  |
| 2021.06.09.12.51.31_veh-35_04882_05013 | 51 | 95 | 3 | 46 |  |
| 2021.09.15.13.16.40_veh-28_01343_01432 | 49 | 85 | 30 | 41 |  |
| 2021.06.09.17.37.09_veh-12_03420_03578 | 49 | 90 | 11 | 45 |  |
| 2021.10.06.18.52.07_veh-28_00442_00578 | 47 | 83 | 32 | 39 |  |
| 2021.06.09.14.58.55_veh-35_00016_00182 | 44 | 77 | 4 | 35 |  |
| 2021.10.06.14.31.13_veh-28_00223_00350 | 41 | 68 | 32 | 30 |  |
| 2021.06.23.14.54.32_veh-16_01011_01166 | 41 | 70 | 13 | 34 |  |
| 2021.06.14.13.27.42_veh-35_01025_01086 | 40 | 74 | 18 | 31 |  |
| 2021.10.01.17.52.06_veh-28_00450_00599 | 40 | 65 | 31 | 34 |  |
| 2021.06.23.18.23.38_veh-26_01438_01758 | 39 | 70 | 26 | 32 |  |
| 2021.10.01.19.16.42_veh-28_01511_01624 | 36 | 59 | 15 | 32 |  |
| 2021.06.09.14.50.36_veh-26_03874_04112 | 33 | 55 | 4 | 30 |  |
| 2021.06.14.11.44.56_veh-35_01145_01297 | 32 | 55 | 17 | 23 |  |
| 2021.05.12.23.36.44_veh-35_00712_00774 | 30 | 57 | 1 | 24 |  |
| 2021.06.09.17.23.18_veh-38_02094_02305 | 30 | 50 | 17 | 28 |  |
| 2021.09.15.14.00.15_veh-28_00895_00981 | 30 | 58 | 31 | 26 |  |
| 2021.06.09.14.03.17_veh-12_01883_01955 | 29 | 49 | 4 | 26 |  |
| 2021.06.09.14.58.55_veh-35_02778_02850 | 29 | 51 | 10 | 24 |  |
| 2021.10.11.08.31.07_veh-50_01001_01076 | 28 | 48 | 16 | 23 |  |
| 2021.06.14.18.42.45_veh-12_01065_01152 | 28 | 49 | 13 | 22 |  |
| 2021.09.15.12.32.43_veh-28_01070_01157 | 28 | 54 | 22 | 22 |  |
| 2021.07.09.16.12.19_veh-26_07208_07271 | 28 | 48 | 27 | 24 |  |
| 2021.06.23.20.00.35_veh-35_04162_04257 | 27 | 44 | 18 | 19 |  |
| 2021.06.09.12.06.35_veh-35_00149_00262 | 26 | 44 | 3 | 18 |  |
| 2021.06.14.18.33.41_veh-35_00488_00562 | 23 | 44 | 12 | 19 |  |
| 2021.10.06.13.21.47_veh-28_00262_00334 | 21 | 41 | 32 | 16 |  |
| 2021.06.14.13.27.42_veh-35_00842_00940 | 20 | 39 | 5 | 16 |  |
| 2021.06.09.17.23.18_veh-38_00131_00294 | 20 | 35 | 17 | 18 |  |
| 2021.06.14.14.25.15_veh-26_02841_02921 | 18 | 34 | 18 | 16 |  |
| 2021.06.09.17.23.18_veh-38_02450_02515 | 17 | 28 | 17 | 16 |  |
| 2021.09.15.14.50.05_veh-28_00182_00253 | 16 | 31 | 14 | 14 |  |
| 2021.06.14.18.33.41_veh-35_01193_01304 | 16 | 29 | 7 | 15 |  |
| 2021.06.14.11.44.56_veh-35_02399_02672 | 15 | 27 | 17 | 13 |  |
| 2021.06.14.16.48.02_veh-12_02517_02590 | 13 | 26 | 12 | 12 |  |
| 2021.06.14.16.48.02_veh-12_02679_02850 | 7 | 13 | 6 | 6 |  |
| 2021.09.15.13.52.55_veh-39_01385_01446 | 6 | 10 | 31 | 4 |  |
| 2021.06.14.17.26.26_veh-38_04030_04274 | 5 | 8 | 12 | 4 |  |
| 2021.06.09.14.50.36_veh-26_04605_04729 | 3 | 4 | 10 | 3 |  |
| 2021.06.14.16.32.09_veh-35_02537_02597 | 3 | 5 | 6 | 3 |  |
| 2021.06.09.12.51.31_veh-35_01587_01718 | 2 | 3 | 3 | 2 |  |
| 2021.06.14.18.33.41_veh-35_00016_00213 | 1 | 2 | 7 | 1 |  |

## 6. Deliverable manifest

| artifact | where | note |
|---|---|---|
| `S0_INPUTS.md`, `s0_inputs.json` | `C:/Users/Admin/par18/s0probe/` | only copy; small; not staged to the repo (the task confines writes to the work dir; Master Mind lands if wanted) |
| probe scripts and raw outputs | `C:/Users/Admin/par18/s0probe/` (`probe_s0.py`, `probe_s0_raw.json`, `verify_*.py/json`, `scan_*.py`, `roots13.json`, `dbfind_*.txt`, `logdirs_find.txt`, ...) | only copy |

Nothing was committed, pushed, downloaded, extracted or modified; no GPU, no ssh, no lock touched.
