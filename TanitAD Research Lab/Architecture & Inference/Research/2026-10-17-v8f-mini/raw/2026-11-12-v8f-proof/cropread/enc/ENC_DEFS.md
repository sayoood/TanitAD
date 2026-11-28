# CROPREAD data-side kit: definitions, assumptions, commands, measured rates

Kit for the card `PREREG_V8F_PROOF_ADD99.md` (sections 2, 3, 4, 7). Dev-box folder `F:/TanitAD-scratch/v8proof/cropread/enc/`; pod copy `/workspace/v8proof/cropread/kit/enc/` (pushed by `push_enc.sh`, sha256 compared per file). Evidence class of every number below: MEASURED on the smoke (2 evaluation + 2 train clips, pod folders `smoke1`, `smoke2`, `smoke3`) unless it says ESTIMATED.

## 0. FIRST: controls that could fail by construction, and what the code cannot do as written

1. **c1 cannot fail for a defect in the window builder, as the card words it.** c1 compares the full 256x640 frame, built by the cache's own route and encoded by the same weights, with the cache. That is the same code, weights and pod that made the cache: on the smoke the fp8 bytes are identical (fraction 1.0), cosine 0.99965 (the remaining 3.5e-4 is the fp8 rounding of the cache; rel-L2 0.0265). It tests the pod's reproducibility, not the windows. So the kit adds a second check inside c1 (`kit_route_pass`, field `window_route_frames_identical`): the full frame rebuilt through the window builder's own route (`decode_selected` + `window_grid` + `window_sample` with the full frame's parameters) must equal the builder's uint8 frames exactly. A frame shift of one in the window route makes it differ. `gate_pass` = card clauses AND this check; the card's `pass` is reported unchanged beside it. If you want c1 to stay exactly as the card words it, ignore `kit_route_pass`.
2. **p1 (ii) is not guaranteed by the geometry.** At the card's thresholds a path curvature of 1/300 per metre moves a lead at 15 m by only 0.4 m (1.4 degrees), while the clause needs an azimuth of at least 3 degrees: most of that azimuth comes from the lead's lane position, not from curvature, and its sign is then not tied to the curvature. The smoke gave 28 of 28 agreeing over 28 frames (4 clips); the clause needs at least 200 frames and 75 percent. Frames: ESTIMATED about 4,500 for 641 clips (28 per 4 clips x 641/4). The agreement over the full set is unknown until the run. Also, the card gives no definition of "path curvature of the ego from its own track": the kit uses the heading change over the next 30 m of the ego's own track divided by 30 m, left positive (`lead_box.kappa_ahead`; sensitivity to the 30 m is not measured).
3. **The card says "elevation 0 its horizon row"; the builder's rectified frame has no horizon row.** The cylindrical build centres on the per-clip principal point and applies no mount pitch (calib.py:906-924; `v2_compressed._decode_cropped_selected` passes no extrinsics). The kit takes elevation 0 as the optical axis (the centre row of the 256x640 frame). The front-wide optical axis is pitched down 0.07 to 1.7 degrees below the vehicle horizontal in the 4 smoke clips (field `optical_axis_pitch_deg_down` in `box/ledger.jsonl`), so arm D's fixed windows (-6, -2 degrees) sit that much lower relative to the true horizon than the card text suggests. A switch to a true-horizon zero needs one extra rotation and is a lead decision.
4. **"The bottom of the box"** has no field in the label (a cuboid has centre, size, orientation). The kit takes the bottom-face centre of the end of the long axis nearest the camera (the face the image shows). "The direction of the centre of the lead's box" (arm B) is the cuboid centre (centre z included).
5. **"Frames with no lead in the label"** (arm B falls back to arm D) is taken as: the join's own lead rule finds no lead on that frame from the obstacle labels (range up to 100 m). That is wider than the join's valid-lead mask used for training targets (0 to 60 m, `lead_valid == 1`). Frames outside the label span, and the clips with no obstacle source file, have no lead and use arm D's windows; their arm B tokens are arm D's tokens (same pixels, same encoder, copied).
6. **The card's replacement rule is read as including the obstacle source:** a train clip is "staged" only when video (sha256 verified), timestamps, egomotion AND its obstacle rows exist (the card needs the box for p1 and arm B). A train clip with no obstacle source in `agents/obstacle_offline/index.parquet` is replaced by the next of the order. Evaluation clips are kept without it (flag `has_obs`); frames with a valid join lead but no box would count against p1 (i).
7. Frame time: the kit uses the builder's own frame times (`t_query` of `_resampled`) as the clock of the label frames. The join was built with a registered / sidecar clock; on the smoke the recomputed gap equals the join's within 2.0e-4 m (median 1.1e-4 m) on all 304 valid-lead frames, so the two clocks agree there. Not known for the other clips until the run.
8. Everything the card asks can be produced as written. Nothing in c1, c7, p1 was changed to make it pass. c2 to c5 are the other helper's.

## 1. Files (all in `cropread/enc/`)

| file | job |
|---|---|
| `make_lists.py` | lists + `LISTS_RECORD.json` (eligible train 1,679 = expected; overlaps 0) |
| `stage_clips.py` | stage a list (pod): mp4 + sha, timestamps, egomotion, obstacle rows, intrinsics + extrinsics tables, ledger |
| `make_final.py` | `train_final.txt` by the replacement rule (500 lines, 450 `fit`, 50 `sel`) |
| `enc_common.py` | frame plan, decode, window geometry, encoder path, pooling, stage layout |
| `lead_box.py` | `boxes` (per clip `box/<clip>.npz`) and `p1` (`ctl/P1.json`) |
| `window_encode.py` | windows + tokens, both arms, manifests |
| `ctl_c1.py`, `ctl_c7.py` | controls c1, c7 |
| `contact_sheet.py` | smoke only, private sheet on the pod |
| `pod_cropread_enc_queue.sh` | the queue (markers START / DONE / STOP, resumable, `SMOKE=1`, `WORKERS`) |
| `push_enc.sh` | push the kit, sha256 compare per file |
| `test_enc.py` | 14 local tests (synthetic) |
| `lists/` | `eval141.txt`, `train_order.txt` (first 520 with 1-based position), `train_eligible.txt` (all 1,679 in order), `LISTS_RECORD.json`; `train_final.txt` comes from the full run |
| `ctl/` | `smoke2_*`, `smoke3_*` control records of the smoke (the full run writes `C1.json`, `P1.json`, `C7.json` on the pod) |
| `probe_*.py` | read-only probes used while building (listing, sizes, window geometry, rates) |

## 2. Definitions with sources

Line numbers are those of the dev-box stack copy `E:/Projects/TanitAD/stack` (read 2026-11-21); the pod tree is byte-identical to it for `v2_compressed.py`, `calib.py`, `physicalai.py`, `build_obstacle_join.py`, `bev_raster.py`, `dinov3_fp8_encode_ship.py`; `build_v9_labels.py` on the pod has 6 more lines (`lead_for_frame` at 507-538, same literals), the pod copy is what runs.

* **Cache frame j** = episode frame 2j; video frame `frame_idx[2j]` with `frame_idx = searchsorted(t_frames, linspace(t_frames[0], t_frames[-1], n_target))`, `n_target = max(int(span/unit*10), 4)` (`v2_compressed.py:111-121`; `pull_encode_pooled.py:82`; `dinov3_fp8_encode_ship.py:119`). `n_cache = ceil(n_target/2)` equals the join's `n` (c7 checks, smoke 101 = 101). Implemented in `enc_common.frame_plan`.
* **Decode**: PyAV, `thread_type AUTO`, rgb24 -> [3,H,W] uint8, the loop of `v2_compressed.py:83-96` (`enc_common.decode_selected`).
* **Window** (`enc_common.window_grid`, `window_sample`): the builder's chain `cylindrical_rays` (calib.py:906) -> `ftheta_project_rays` (927) -> native px scaled to the decoded frame -> `grid_sample` bilinear, zeros padding, mask multiply, clamp, uint8 (937-998). Window frame = square 256x256 cylindrical `CanonicalFrame.from_hfov(size, 256, 256)`: horizontal angular size exactly 32 or 16 degrees, azimuth linear in the column; vertical axis the builder's per-column pinhole (half-angle 15.6 / 7.9 degrees). The centre direction enters as a rotation of the rays, `R = Ry(-az) Rx(el)`; (az, el) are the rectified frame's own coordinates: azimuth left positive and linear in the column, row offset = f_ref tan(el), up positive. With az = el = 0 no rotation is applied and a window with the 256x640 frame's parameters is the builder's frame exactly (test `test_window_with_full_frame_parameters_reproduces_the_builders_frame`, and on the pod `window_route_frames_identical`). Pixels are cut from the RAW decoded frame (1080x1920), not from the 256x640 frame.
* **Arm D**: W32 at (0, -6) degrees, W16 at (0, -2), every frame. **Arm B**: both windows on the box-centre direction when the label has a lead on the frame, else arm D's.
* **Tokens**: `Encoder.window_tokens`: bf16 weights, batch 8, no autocast, `(x/255 - mean)/std`, `last_hidden_state` (the arithmetic of `dinov3_fp8_encode_ship.py:108-129`, weights from `pull_encode_pooled.build_model`, sha check in `hierrep/dinov3_sha_check.json`); token 0 = class token; the 5 special tokens (class + 4 registers) are dropped; the 16x16 patch tokens are mean-pooled 2x2 (fp32) to 8x8 row-major. File `tok/<arm>/<clip>.npy` float16 `[n_frames, 2, 65, 1024]`, window order W32 then W16. Frame i of the file is frame i of the join. `tok/<arm>/MANIFEST.json`: `n_frames`, `sha256`, `bytes` per clip (from `ledger.jsonl`, the resume record). Output files are written whole under a `.part` name and renamed once.
* **Lead boxes** (`lead_box.compute_clip`): obstacle rows (HF `agents/obstacle_offline`, rig frame at their own timestamp) -> agents of the frame by the join builder's own composition (`build_obstacle_join.py:253-282` `rig_to_world`, `406-432` `world_agents_at`, `bev_raster.ego_frame_agents`; poses = `physicalai.signals_at` at `t_query`, as `join_clip` 438-493 with the registered time replaced by the builder's frame time, see 0.7) with `center_z` and `size_z` carried -> the join's own lead rule `build_v9_labels.lead_for_frame` (501-532, called unchanged; `lead_index` restates it only to learn which agent it chose, and the two gaps are asserted equal: 0 mismatches on 404 frames) -> box to the camera frame with the front-wide mount (`calibration/sensor_extrinsics`, `physicalai.FrontWideExtrinsics`, physicalai.py:245; camera axes +x right, +y down, +z boresight; camera at about (2.0, 0, 1.35) m in the rig). Angles: azimuth = atan2(-x, z) (left positive), elevation = atan2(-y, hypot(x, z)) (up positive).
* **Valid lead frame** (card section 2 and p1): `lead_valid == 1`, finite `lead_gap`, 0 <= gap <= 60 (`make_lists.valid_lead_mask`).
* **p1 clauses** (`lead_box.p1_from_arrays`): (i) |recomputed gap - join gap| <= 0.5 m on >= 98 percent of valid-lead frames, a clip with no box file or a frame-count mismatch counts as failed frames; (ii) frames with recomputed gap >= 15 m, |centre azimuth| >= 3 degrees, |curvature| >= 1/300: sign(azimuth) == sign(curvature) on >= 75 percent and at least 200 such frames; (iii) valid-lead frames with join gap < 20 m: bottom elevation < 0 on >= 98 percent. All three are over the 141 + final train clips; the record has counts, fractions, a by-gap-bin table for (ii) and the median / max gap difference for (i).
* **c1** (`ctl_c1.py`): first 6 evaluation clips by ascending sha256 hex of the clip id; frames by the builder's own call; encoder bs 8; pooled 8x20 fp32; cosine of the flattened [160 x 1024] tokens per frame against the cached fp8 tokens (decoded); clause 1 median over all frames >= 0.98; clause 2 on frames with the cache's own `v` >= 2 m/s, the cached frame with the largest cosine is the same index on >= 95 percent.
* **c7** (`ctl_c7.py`): 141 evaluation and >= 490 train clips in both manifests, files present with the manifest's byte size, `n_frames` equal to the join's `n`; `--verify-sha` re-hashes every file; the clips left out are listed with the reason.

## 3. Commands (lead's full run)

```
# 1. push (dev box)
bash /f/TanitAD-scratch/v8proof/cropread/enc/push_enc.sh
# 2. full run on the pod, detached (read memory.max and anon first: keep 28 GB free; no other GPU process)
nohup setsid env WORKERS=5 bash /workspace/v8proof/cropread/kit/enc/pod_cropread_enc_queue.sh > /dev/null 2>&1 < /dev/null &
#    outputs: /workspace/v8proof/cropread/enc_out/{lists,ctl,box,tok}, log queue.log, markers START.txt DONE.txt STOP.txt
#    stage root (re-usable): /workspace/v8proof/cropread/stage ; shared pulls: /workspace/v8proof/cropread/hfdl
# 3. wait (one pssh call, remote loop, <= 540 s per call)
#    until [ -e .../enc_out/DONE.txt ] || [ -e .../enc_out/STOP.txt ]; do sleep 20; done
# smoke again (new folder smokeN each time):  nohup setsid env SMOKE=1 WORKERS=5 bash .../pod_cropread_enc_queue.sh ...
```
Queue order: lists (reused if present) -> stage 141 eval + first 500 train positions (then 510, 520 if the list is short) -> `train_final.txt` -> **c1 (gate: the queue stops with `STOP.txt` when c1 fails; `C1_GATE=0` continues)** -> boxes -> windows + tokens -> p1 -> c7 (with sha re-hash) -> `DONE.txt`. Every step is idempotent: a re-run skips staged clips, finished boxes and encoded clips. The kit never deletes or moves anything (its only rename is the `.part` -> final of its own output file).

## 4. Measured rates (smoke, A40 pod, CPU quota 7.65 cores)

| step | per clip (101 cache frames) | note |
|---|---|---|
| stage: sha-verified mp4 download | 0.6 to 0.9 s (5 to 27 MB) | one-off: timestamps.tar 2 s, egomotion tar (2.0 GB) 8 s, each obstacle shard about 1 s |
| video decode, 1 decode thread | 9.3 to 11.0 s | about 600 raw frames decoded, 101 kept |
| window cuts (4 windows x 101 frames), 2 torch threads | about 1.7 s | decode dominates |
| decode + windows inside a worker thread | 7.5 to 13.4 s | 2 to 5 threads, same range |
| encode (GPU, D: 202 windows, B: 81 to 99 lead frames x 2) | 2.0 to 2.6 s | arm B reuses arm D's tokens on no-lead frames |
| file write + sha256 | 0.1 s | |
| boxes | 0.07 to 2.0 s | first clip includes imports |
| c1 | 22 to 25 s per clip | decode twice + full-frame encode |

Whole smoke chain (4 clips): 21 s with `WORKERS=5` for windows + encode, whole queue about 90 s. Full run (ESTIMATED from the rates): windows + encode bound by decode at 5 threads, about 641 x 10 s / 5 = 21 min, GPU encode 641 x 2.3 s = 25 min overlapped, so about 30 to 40 min; stage about 10 min (one-off tars 10 s, shards up to 48 x 52 MB); c1 about 2.5 min; boxes about 5 min. Tokens of the smoke are identical across runs and worker counts (sha256 of the 4 clips x 2 arms equal in smoke1 and smoke2).

## 5. Disk

Tokens 26.9 MB per clip and arm (MEASURED: 26,890,368 bytes) = 641 x 2 x 26.9 MB = 34.5 GB. Staged mp4s about 641 x 15 MB = 10 GB (re-creatable from HF; left in place). Shared pulls `hfdl` 2.1 GB now, up to about 4.5 GB with all obstacle shards. Boxes under 1 MB per 10 clips. Smoke keeps `win/` (private camera windows, 79 MB per clip) and `sheet_private.png` on the pod only. Pod disk free 392 GB at the end of the smoke.

## 6. Temporary files left in place (no delete was issued)

* dev box: `cropread/enc/tmp_pt/` (pytest basetemp t1, t2, t3), `probe_*.py`, `ctl/smoke2_*.json`, `ctl/smoke3_*.json`.
* pod: `/workspace/v8proof/cropread/probe/` (probe scripts, `hf/` with one obstacle shard and the extrinsics / index tables, about 60 MB); `/workspace/v8proof/cropread/smoke1`, `smoke2`, `smoke3` (each: stage 75 MB, tok 206 MB, win 304 MB PRIVATE, sheet_private.png); `/workspace/v8proof/cropread/hfdl` (2.1 GB shared pulls, reused by the full run); `/workspace/v8proof/cropread/enc_out/lists` (the lists, reused by the full run); `/tmp/s1_D.txt s1_B.txt s2_D.txt s2_B.txt` (small comparison files outside the work folder). The smoke `win/` folders are the only camera pixels the kit ever keeps; they are re-creatable and are the first thing to sweep.
