# GEO10K: the per-clip label inputs of the v8f-mini trainer, and what can be built for the 10,000 new clips

Base: commit `cdbff971` (branch `origin/agent/v8f-otband-20261109`). Evidence class of every line: MEASURED (read from that commit, file:line given) unless marked.
Trainer = `stack/scripts/v8f_mini_train.py` (T), data = `stack/tanitad/data/v8f_mini_data.py` (D), v9 loader = `stack/tanitad/data/v9_labels.py`.

## 1. What the trainer reads per clip

| # | input (flag, default) | read at | what it carries | builder | needs a VLM / Alpamayo record? | 10k status |
|---|---|---|---|---|---|---|
| 1 | cache clips `<cache>/train/clips/<clip>.pt` + `manifest.jsonl` (`--cache-dir`) | T:261, D:120 `read_manifest`, D:530 `ClipStore` | pooled DINOv3-L tokens, v/a/steer/yaw_rate, pose_grid, kin3, p10 | `cache_pod.py` (the 10k cache) | no | DONE, banked (HF `Sayood/tanitad-datascale-10k`, 42 shards, 10,000 clips) |
| 2 | v9 release npz (`--v9-train`, md5 pinned by `--v9-train-md5`; accepted md5s in D:67) | T:3084-3088, D:166 `build_label_join` | per frame (sid, k): nav token + d_next (D:216-226), RC-A 30/50/80 x,y,psi,valid (D:151 `_rc_arrays`, D:227), goal_y / goal_w 22-bit goal sets (D:229-230), lat_cls / lat_allowed (a), lon_cls / lon_allowed, 26 constraint cells (D:235-236), lead_valid (D:237), now_s | `build_v9_labels.py cmd_build` (`stack/scripts`, md5 e6b2a7d4 at cdbff971) over `s2_geom_emit_v7` turn table | PART: the geometry fields no; goal bits of the VLM classes (TRAFFIC_LIGHT_*, STOP sign, pedestrian etc., provenance `vlm-cot`) yes (`cmd_build` loads the records at build_v9_labels.py:1186-1191 and reads `recs.get(cid)` at :1216; `rec=None` is a supported path) | geometry side BUILT (section 3); VLM goal bits NOT BUILDABLE |
| 3 | lead file (`--lead-train`) | T:3087, D:192 (`lead_gap`), D:277 `lead_label_arrays` (lead head) | per (sha12, k): lead present, gap, time-gap | `build_v9_labels.py cmd_lead_pass` over the obstacle join | no (needs `obstacle.offline` join, present for all 10,000: banked in keep10k per shard) | BUILT |
| 4 | speed-limit sidecar (`--speed-train`, `refcv6_speed_max_v8_train.jsonl`) | T:3088, D:109 `load_speed_sidecar` | per clip: limit class one-hot [4] (an INPUT) | `build_refcv6_speed_max_window.py` from the v8 label record's `SPEED_BAND.v_hi_ms` | the record is made by `emit_one`, but `SPEED_BAND` is unconditional geometry (s2_geom_emit_v7.py:290-314, called with `cot=None`) | BUILT from geometry (section 3) |
| 5 | native pose tracks (`--tracks`, `native_tracks.pt`; required by `--kin-mode yaw`, the default) | T:3118-3120, D:366 `load_native_tracks` | raw 10 Hz x,y,yaw,v, row i = raw frame i+2 | `build_native_tracks.py` from the v2 manifest poses | no | BUILT |
| 6 | clip clock sidecar (consumed inside the v9 build, `clip_clock`) | build_v9_labels.py:494 | grid_start_s, dt_s per clip | `build_clip_clock_sidecar.py` | no | BUILT (364 of 10,000 refused by the fit's own gates; they take the builder's own `pose_dt` fallback, as in the existing corpus) |
| 7 | lane-change SAM3 codes (`lc_train.npz`, `--lc`, builder flag `--lc-labels off` for the release) | build_v9_labels.py:1198-1203 | diagnostics only (`lc_*` columns); the lane-change classes ship never-positive | `build_v9_labels.py lc-pass` on Thor (SAM3 map GT) | no, but needs frames + SAM3 | NOT BUILT: `lc_*` diagnostic columns read their no-input value (`lc_measurable_raw` = -1) |
| 8 | v10 band release (`--v10-md5`, loaded through `--v9-train`; md5 must be in `ACCEPTED_V10_TRAIN_MD5`, D:70-86, EMPTY at cdbff971) | T:363 | the 2-6 s tactical labels | `band_labels/build_v10_labels.py` (arch-inf tip, not at cdbff971) | same VLM goal-bit dependency as row 2 | NOT BUILT here (see section 6) |
| 9 | per-clip strategic tokens (`--t-stok-dir`, default off) | T:355, D:473 `StokStore` | `<clip>.npz` stok [T,D] | stream C (a VLM/encoder dump) | yes | NOT BUILDABLE here; off by default |
| 10 | frozen P10 prior (`--p10`) and `--wm-zcache` | T:269, T:346 | one global file / a z cache, not per clip | not per clip | no | not part of the per-clip set |

The trainer's nav block feeds only INPUT fields (D:175, `NAV_FORBIDDEN_AS_INPUT` are never read).

## 2. What needs a VLM / Alpamayo record, with n

* `emit_one` (s2_geom_emit_v7.py:863) calls `AF.cot_text(clip)`, `_alpamayo_layer`, `AF.ground_tokens` and `AF.corroborate`: all read `alpamayo_records`. For a clip with no record the pilot probe raised `KeyError: no egomotion` first (a lookup problem, solved in section 3) and `AlpamayoRecordsUnavailable` for the records. Alpamayo CoT exists for the 4,729 clips of `Sayood/tanitad-alpamayo2-augmentation` only (SCOPE_DATASCALE.md section 6, MEASURED there). NVIDIA event CoC covers 1,740 clips, 1,364 of them in the pool (SCOPE, MEASURED); the overlap with the 10,000 was not measured here.
* Consequence, MEASURED on 20 existing-corpus clips (control ctlC vs the release, `compare_ctlC.json`; the path with records absent, lead present): 8 of 115 row fields differ materially. Record-dependent: `goal_y`, `goal_w`, `goal_lc_prov`, `vlm_has_cot`, `vlm_tl_probe_negative`. lc-input-dependent: `lc_frac`, `lc_measurable`, `lc_measurable_raw`. Every other field is within 1e-9 of the release. The `lon_*` classes and `lead_*` aggregates depend on the lead input (control ctlB, lead absent, moves 12 more fields), which exists for the new clips (row 3).
* n of the 10k clips with any VLM tactical goal: 0 (no record exists). A VLM pass (compute plus a PI decision) would be needed; frames are already encoded but the VLM reads images, so frames must be re-downloaded.
* Trainer behaviour: a goal bit with `goal_w` = 0 is masked (D:229-230 reads `goal_w` as the weight), so the geometry goal bits train and the VLM ones are IGNORE, as the SCOPE note states.

## 3. How the geometry side was built (all on the pod CPU, tree = clean export of `stack/` at cdbff971)

1. Camera timestamps (first and last frame time) re-fetched per clip from the NVIDIA camera chunk zip (member range reads, no video read): `g1_fetch.py`.
2. v2-manifest poses rebuilt exactly as `physicalai.build_episode` does (`signals_at` on `linspace(t_first, t_last, n)`): `g2_poses.py`. Native tracks from them (`D.full_from_manifest`).
3. Clip clock: `build_clip_clock_sidecar.fit_clock`, unedited: `g3a_clock.py`.
4. Lead pass: the builder's own `lead_for_frame` over the per-shard obstacle joins: `g3b_lead.py`. The join key was MEASURED, not assumed: the join's `frame_idx` is the RAW frame index (median residual of `t_s` against `g0 + frame_idx*dt`: 0.0065 s; against `g0 + (frame_idx+2)*dt`: 0.195 s), so no offset is applied, matching the existing lead file's k range (0..199).
5. v9 geometry labels: `build_v9_labels.cmd_build`, unedited, run in 20 parts and merged: `g3c_build.py` (records: none, lc: none, lead: step 4).
6. Speed sidecar rows: `g6_speed.py`.
7. Context metadata: `g5_context.py`.
8. Fill of the three lane-change columns the trainer reads and the builder omits without an lc input: `g3d_fill.py`. Reachability through the trainer's own loaders: `g7_check.py`.
* Egomotion lookup: configuration only. `egomotion_source.EGO_FLAT` / `EGO_GLOB` (module constants) are pointed at a flat directory of symlinks to the banked keep10k parquets (`geo_common.py`), and `build_v9_labels --ego` takes the same directory as its format string. No library file is edited.

## 4. Gaps that remain for training on the 10k (escalation)

* `ACCEPTED_TRAIN_MD5` (D:67-69) names two v9 releases. A new 10k release has a new md5, so the trainer refuses it until the Master Mind registers that md5 (or merges the 10k rows into a release and registers the merged md5). The release loader needs `clips` keyed by sid with unique sids; the 10k sids are disjoint from the 4,572 (MEASURED: clip overlap 0, `selection` vs cache manifest).
* The cache manifest the trainer reads (`--cache-dir/train/manifest.jsonl`) must list the 10k clips; the banked `manifest/` in the HF repo holds the corpus manifest.
* Fields that read their no-input value on the 10k: VLM goal bits (`goal_w` zero for those classes), `lc_*` diagnostics. `lc_t_cross_s`, `lc_d_cross_m`, `lc_side` are filled with the builder's no-crossing value (`g3d_fill.py`), because the trainer reads the first two as constraint cells (found by the reachability check, `KeyError: lc_t_cross_s`).
* The lead file has no known-value control (no obstacle join exists for the existing-corpus controls); the lead-derived fields are UNVERIFIED against a known value. 441 of the 10,000 clips have no lead rows (205 without `obstacle.offline`, 236 with a parquet but no join row).
* The new v9 file's md5 is in `labels/v9_labels_datascale10k.manifest.json` (`npz_md5`); it is not in `ACCEPTED_TRAIN_MD5`.
