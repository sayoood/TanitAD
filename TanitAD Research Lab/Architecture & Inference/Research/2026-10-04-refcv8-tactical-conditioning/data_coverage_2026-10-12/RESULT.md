# refcv8 training-data coverage audit (2026-10-12) -- read-only, 0 GPU, nothing built

Layer key: PUB = published PhysicalAI-AV corpus; SEL = our B1 clip selection; CACHE = built 416x1024 v2 cache; LAB = v8 label release; VIEW = refcv6 train view.
Evidence: MEASURED unless tagged. Artifacts under `raw/` (copied read-only from Thor) and Thor `/home/nvidia/data/`.

## 1. Clip accounting (the "26 h")
| set | n clips | hours (x20 s) | source |
|---|---|---|---|
| PUB corpus (`clip_index.parquet`, index `clip_id`; cols `clip_is_valid, chunk, split`) | 306,152 (official train 153,625 / val 90,928 / test 61,599) | 1,701 h | local parquet |
| SEL = CACHE `physicalai-b1-w120-416x1024cyl` | **4,713** | **26.2 h** (= the PI's "26 h") | Thor listing, `MANIFEST.json` |
| of which eval-labelled (LAB `s2_labels_v8_eval`) | 141 in CACHE (147 labelled; 6 not in CACHE) | 0.78 h | raw/s2_labels_v8_eval.jsonl.gz |
| train-labelled (LAB `s2_labels_v8_train`) | **4,572** (speed census 1741/1593/1014/224) | 25.4 h | raw/ |
| VIEW `refcv6-b1-416x1024-train` (what refcv7 and refcv8 read) | **4,369** (746,946 windows) | 24.3 h | `_VIEW_RECORD.json`, `refcv6_train_clips.txt` |
| **excluded 203** = 145 `no_agent_join` + 58 `no_validated_map` | 203 | 1.13 h | `_VIEW_RECORD.json` removed_sha12 (sha12 = sha256(clip_id)[:12]; 203/203 reproduce) |
| eval view `refcv6-b1-416x1024-eval139` | 139 (2 more eval-labelled clips sit in CACHE unused; 6 absent) | | `_VIEW_RECORD.json` |

Chain (`_VIEW_RECORD.json`): cache 4,713 -> v8-train labels 4,572 (-141 eval) -> agent join 4,427 (-145) -> validated SAM3 map 4,369 (-58). 4,369 + 139 = 4,508 = extrinsics/pose-sync coverage. The 4,566 is `agent_cls_weight.provenance.n_clips` (INHERITED, not re-derived; equals no set above).
Reasons for the 203 (MEASURED, `feature_presence.parquet`): of the 145, only **13** have `obstacle.offline` in the PUB corpus (local `labels/obstacle_offline_b1train` holds 4,440 = 4,427 + 13 files, so those 13 were present but dropped from the join -- cause UNVERIFIED); **132 have no published obstacle data at all**. The 58 all have obstacle data but no validated SAM3 map (stationary/intersection clips; Q7 audit `2026-09-26-refcv6-frozen-trunk-audit/RESULT.md:317-318`: HOLD 39.7 % vs 2.3 %, CREEP 20.7 % vs 2.7 %, p<=0.0025). **None of the 203 has a file in `sam3_gt_v3`** (0/145, 0/58; the dir holds 4,506 = 4,369 + 137 eval).
More data exists: PUB is 306,152 clips; B1 is 1.5 % of it. "All 26 h" = the whole CACHE; 0.78 h of it is eval and must stay out. The programme split is not the official split: B1 train labels include 1,213 official-`val` and 671 official-`test` clips.

## 2. Sampler (`stack/scripts/refc_v3_train.py`)
- `ResumableEpochSampler` (:7233): per-epoch `torch.randperm(n, Generator.manual_seed(seed*1_000_003+epoch))`, **without replacement within an epoch, reshuffled each epoch, no weighting/stratification**; DataLoader `drop_last=True` (:9031-9035); windows are every frame (stride 1, `tanitad/data/_contract.py:114-117`): 170.96 windows/clip (746,946/4,369). One epoch = 746,946 windows = **46,684 batches** at b16 (2 windows dropped per epoch).
- `--steps 30000` = 480,000 samples = **0.643 epoch**: exactly 64.26 % of windows seen once (a permutation prefix, 0 repeats), 109.9 samples/clip of 171; every clip is seen. Windows overlap ~95 %, so clip-content coverage is complete; window-level coverage is 64 %.
- **FINDING: refcv8 replays refcv7's data order.** refcv7 config: same `--v2-cache`, `--seed 0`, `--batch 16`, n_windows 746,946, 50,400 steps (1.08 epoch); refcv8 smoke config: seed 0, n 746,946, optimizer fresh, no `data_pos` carried. Same (seed, epoch 0, n) gives an identical permutation, so refcv8's 30k steps re-present refcv7's first 480,000 positions and never present the other 266,946 windows (35.7 %), which refcv7 saw at its steps 30,000-46,683. A different n (ADD) or seed breaks the replay. (Inferred from the two configs + code; the permutation itself was not replayed.)

## 3. Steps and wall-clock (9.212 s/step; 4 gs readings x 838.9 s = 0.93 h; eval every 500 steps)
In-run eval cost is NOT separable from the smoke (elapsed_s 1386.0 -> 1846.6 over steps 50 -> 100; 460.6 s = 50 x 9.212, so the ~86 s residual in the first interval covers startup + eval): evals bounded **<= ~86 s each** (UNVERIFIED upper figure).
| corpus | windows | 30k steps | 1.0 epoch | 1.5 epoch |
|---|---|---|---|---|
| VIEW 4,369 | 746,946 | 0.643 ep; 76.8 h = 3.20 d | **46,685 steps** (batches/epoch 46,684) = 119.5 h = 4.98 d (+0.93 h gs, +<=2.2 h evals) | **70,027 steps** = 179.2 h = 7.47 d (+0.93, +<=3.3 h) |
| VIEW + 203 ADD (ESTIMATED 781,652 windows; ADD windows not measured, 170.96/clip assumed) | ~781,652 | 0.614 ep | **48,854 steps** = 125.0 h = 5.21 d (+0.93, +<=2.3 h) | **73,280 steps** = 187.5 h = 7.81 d (+0.93, +<=3.5 h) |
refcv8 inits from refcv7 step 50,400, so these epochs come on top of 1.08 prior epochs.

## 4. PI ruling: add the missing clips, never the val split
**ADD list = `raw/ADD_203_clip_ids.txt` (203 ids) = LAB train ids minus VIEW ids; groups in `raw/add_groups.json` (145 / 58).** Positive disjointness (MEASURED): ADD n eval labels (147) = 0; ADD n eval view listing (139 symlinks, equal to `eval139_ids.txt`) = 0; ADD n eval speed sidecar (147) = 0; ADD n eval extrinsics keys = 0; `parity.clips_in_v72_eval(ADD)` = 0 (control: the same call on eval139 returns 139; on all 4,572 train ids returns 0); train labels n eval labels = 0. The 8 eval-labelled clips outside eval139 (2 in CACHE, 6 absent) are not in ADD.
**Drive/session level: NOT PROVABLE.** `clip_index.parquet` has only `clip_is_valid, chunk, split`; `data_collection.parquet` only `country, month, hour_of_day, platform_class, radar_config`; egomotion is clip-local metres with no GNSS. No session/drive id is exposed, so the existing train/val split's drive-level disjointness is UNKNOWN too (storage `chunk` overlap: ADD 23 of 171 chunks shared with eval, VIEW 93 of 1,338 -- `chunk` is a shard, not a drive). One untested lead: absolute camera timestamps in `*.timestamps.parquet` may expose temporal adjacency.

## 5. Build plan for the ADD list (nothing built)
| artifact | status for the 203 | action / cost |
|---|---|---|
| 416x1024 v2 cache entries | **already exist** (203/203 in the parent CACHE; ~81 MB for the one file sized) | none: a new symlink view only (Q7 P5). No `build_compressed` run. |
| v8 labels + speed sidecar | present (4,572; sidecar 203/203) | none |
| agent join (`/home/nvidia/percprobe/raw/b1train_agents.jsonl.xz`, 317 MB) | 0/203 | 13 clips can be joined (dev box, `build_b1_agent_join.py`, minutes); 190 have no source data -> trainer emits NO_LABEL (`_agent_item` :3303; `enable_agent_join` refuses only zero coverage) |
| SAM3 map GT (`sam3_gt_v3`) | 0/203 | no GT exists; trainer emits `map_label` False (`_map_item` :3480) but `require_map_coverage` floor is **0.90** (`tests/test_refcv6_perception.py:532`): +34.7k unlabelled windows = 4.4 % of ~781.7k lowers train frac_ok by ~4.4 pts (current train frac_ok UNVERIFIED; eval is 0.9712). Re-census needed. Real GT would need SAM3 on Thor (`sam3_corpus` holds a 4,719-clip run; whether it covers the 145 is UNVERIFIED) |
| extrinsics (`refcv6_train_extrinsics.json`, 4,369 keys) | 0/203 | rebuild with `build_rig_extrinsics_table.py` / `pai_extrinsics_table.py` from `sensor_extrinsics` (203/203 present in PUB); dev box, minutes |
| pose-sync sidecar (4,508 rows; `build_pose_sync_sidecar.py`, needs `<clip>.timestamps.parquet` + manifest) | 0/203 | rebuild; reads timestamps + manifest only, no frames; dev box |
| clip-clock sidecar (4,483 rows; covers 4,347/4,369 of the view) | 0/203 | rebuild via `build_clip_clock_sidecar.py` (manifest-only). **Guard:** `--label-clock-max-unverified` is 0.01 (:337); unextended, (22+203)/4,572 = 4.9 % REFUSES |
| view `_v2manifest.pt` (md5 3c9f8bc8...) | changes | `build_v2_providers(rebuild=True)` opens every file of the new view (~72 MB/clip avg, ~330 GB for 4,572) -- Thor IO; run when Thor is idle or from a copy |
| launch-gate pins | change | `launch_gate.py:572` pins `vis1_sidecar_refcv6b1_train4369_eval139.npz` (needs the 203 or a new file); the `--r8-v9-md5`, train-manifest md5 and pose-sync `inputs_md5` pins must be re-registered; `agent_cls_weight` (provenance n_clips 4,566) per-class frequencies change and `_cls_weight_expectation` (:6354) refuses until regenerated |
Parity note: the B1 view is already a non-parity corpus, so ADD does not touch the e438721ae894 invariant; it does make refcv6/refcv7 training data differ from refcv8's (same eval139).
Caveat for the decision: the 58 stationary clips are the HOLD/CREEP-rich clips the tactical head lacks, but have no map GT; the 145 are highway/fast clips mostly with no agent GT. Those are label gaps, not build gaps; the trainer tolerates both.
