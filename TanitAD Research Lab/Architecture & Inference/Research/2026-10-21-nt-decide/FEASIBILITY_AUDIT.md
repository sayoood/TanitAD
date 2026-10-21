# NT-DECIDE deliverable 1: can refc be fine-tuned on NAVSIM navtrain? Feasibility audit from source and data

Research Lab (Architecture & Inference), 2026-10-21, 19:55 to 20:20 Berlin. Design and audit only: no GPU was used, nothing was trained, nothing was committed or staged. The only writes went to `C:/Users/Admin/refcv8_pf/nt_decide/`.

**Evidence classes:** MEASURED (this session, artifact named) · INHERITED (another record, not re-measured) · PUBLISHED (banked primary in the Library) · ESTIMATED.

**Source pins:**
* Trainer and model code are read at the refcv8 launch commit **`fb47a58`**.
  * `src_fb47a58/` in this folder holds `git show fb47a58:<path>` extracts.
  * The Thor code bundle `C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/tree/stack/scripts/refc_v3_train.py` is byte-identical to `git show fb47a58:stack/scripts/refc_v3_train.py`: md5 `ecfb366e…` on both, MEASURED.
* The NavSim agent code is read from the dev-box mirror of Thor's `/home/nvidia/r8ns/code/`:
  * `[PKG]` = `…/navsim_ckpt/thor_bundle/stage/pkg/code/`. This is repo `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/`.
  * `[R6]` / `[R7]` = the refcv6 / refcv7 NavSim suites under `…/thor_bundle/stage/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-2{3,8}-…/navsim/code/`.

---

## 1. How refc is fed NAVSIM today (the r8ns agent)

| input | what the agent feeds | source |
|---|---|---|
| **cameras** | ⚠️ **Not CAM_F0 alone.** The agent builds one **virtual 120° cylindrical camera** at 416×1024 by re-stitching **CAM_L0 + CAM_F0 + CAM_R0**.<br>• Boresight is CAM_F0's, rolled level.<br>• Each output ray takes the camera where it lands with the largest cos (bilinear sampling); unseen rays stay black.<br>• It is a pure-rotation reprojection: camera centres differ by about 0.3 m and this is ignored.<br>• `f_ref` = 488.924 px. The column is linear in azimuth: hfov exactly 120°, vfov 46.09°. | `frames416.py:4-14, 30-37` [R6]; `refcv8_bridge.py:6-8` [PKG] |
| **frame geometry guard** | `assert_frame` refuses any frame other than `trunk_shapes.FRAME_416x1024` (hfov 120). The loader asserts `image_hw == (416, 1024)`. | `frames416.py:106-111`; `refcv8_bridge.py:261` |
| **time slots** | NavSim stores 4 history frames at 2 Hz. refc reads **10 raw slots at 10 Hz** (8 window rows × a 3-frame D-015 stack), packed as `[8, 9, 416, 1024]` u8.<br>The bar arms (R8_LEGAL, `_s1`) use **ST: every slot ← the t0 frame**. R8_NT (nearest-time) is a declared sensitivity only. | `frames416.py:39-42`; `refcv6_bridge.py:264-280`; `refcv8_bridge.py:70-79` |
| **trunk dedup under ST** | The forward recorded **1 trunk frame computed for 24 slots**: `last_dedup = (24, 1)` on **30/30** smoke rows. MEASURED, `…/navsim_ckpt/raw_thor_smoke/thor_rows_*.jsonl`.<br>⚠️ A plain reading of `_backbone_dedup` predicts at least 3 frames for 8 rows × K 3, so the mechanism is UNVERIFIED. | `timm_trunk.py:872-906` |
| **nav command** | `driving_command[3]`, the t0 one-hot (the only command field the arm declares), is mapped through `NAVSIM_LEGAL_NAV_CMD`:<br>• left → TURN_L (1)<br>• forward → FOLLOW (0, never "straight" 3)<br>• right → TURN_R (2)<br>• unknown → FOLLOW<br>A non-one-hot vector is refused. | `refcv8_bridge.py:154-180` |
| **nav args** (`r8_nav`) | Always the **UNKNOWN row**: `scale_nav(zeros, known=False)` = zeros `[1, 6]`. NavSim supplies no route geometry. | `refcv8_bridge.py:191-207` |
| **route checkpoint** | Not fed (`--r8-no-rc`). | `refcv8_bridge.py:15` |
| **speed ceiling** | R8_LEGAL feeds the trained UNKNOWN row `(0, 0)`. R8_PRIV (diagnostic only) feeds the nuPlan map limit snapped to its road-type bin. | `refcv8_bridge.py:213-239` |
| **ego state** | `v0` = \|velocity\| at t0.<br>The ego window is 8 × (x, y, yaw, v) at 10 Hz, **linearly interpolated** between the 2 Hz states t0−1.0, −0.5, 0 s.<br>a0 and yaw rate are derived inside the forward (`ha0_ext_pose`); **curvature is not fed**.<br>`set_ego_window(poses, 8, actions=None)` is called as the trainer calls it for a pose-only prior. | `refcv6_bridge.py:120-160`; `refcv8_bridge.py:9-11, 343` |
| **BEV lift** | The stitch's virtual camera goes through the model's own 0.25 m bank: rotation = the stitch's M, centre = CAM_F0's mount, height = F0.z − road plane. | `refcv7_bridge.py:178-200` |
| **forward keywords** | Exactly the trainer's 12-keyword set (`FORWARD_KWARGS`).<br>`lan` / `ego_state` / `nav_args` / `agent_gt` = None. bf16 (`as_trained`) on CUDA. | `refcv8_bridge.py:58-61, 311-332` |
| **output** | `out["traj"]` `[8, 2]` at 0.5, 1, 1.5, 2, 3, 4, 5, 6 s.<br>→ a not-a-knot cubic spline through (0, 0) and the knots<br>→ 8 NavSim poses at 0.5–4.0 s (heading = spline tangent). | `refcv8_bridge.py:18-20` |

**Camera geometry** (MEASURED from one navtest log pickle, `_probe_rig.py`):
* Every nuPlan camera has fx 1545 px at 1920 px width, i.e. a 63.7° horizontal field.
* Camera yaws: F0 0.2°, L0 +55.2°, R0 −55.5°, L1/R1 ±112°, B0 179.7°.
* L0+F0+R0 therefore cover about ±87°. refc keeps the middle **±60°**.
* REFe takes **F0, L0, R0 and B0** at 512×960 (registry §14.1; GROUNDWORK inventory).

So against REFe, refc's camera deficit has three parts:
* (a) the outer ~27° of L0 and R0 on each side;
* (b) the rear camera;
* (c) angular resolution of 488.9 px/rad vs about 772 px/rad (1545 × 960/1920). That is **0.63×** (ESTIMATED from the intrinsics; REFe's resize is assumed to be ×0.5).

## 2. What refc's losses need, and what NAVSIM can supply

| refc loss / input (refcv8 argv, `stack/ops/runs.d/refcv8-r101-s0.argv.json` @ fb47a58) | NAVSIM source | verdict |
|---|---|---|
| **Trajectory / diffusion / selector critics.** The targets are `future_poses_ext` `[60, 4]` at 10 Hz plus `future_valid_ext` (`refc_v3_train.py:4515-4521`, `MAX_H_EXT = 60` at `:233`). | OpenScene log pickle `ego2global` per 2 Hz frame (MEASURED: dt 0.500 s; one log has 804 frames). The knots at 0.5–6 s are exact 2 Hz samples. The 10 Hz ticks need interpolation (spline, as `knots_to_navsim` does). The 6 s knot falls beyond a short log's end, where `future_valid_ext` masks it. | **YES**, with a declared 2 Hz → 10 Hz interpolation |
| **Tactical v6 decoder** (`--w-tac-v6 1.0`): `lat_v7` / `lon_v7` from `--v7-labels` (`:4542-4560`), plus v9 constraint supervision (`--r8-v9-labels`, `--w-r8-v9-cons 0.05`) | The geometric parts of both builders read only the ego track:<br>• `build_v9_labels.py:1-60`: band [2, 8] s, 10 Hz grid, admissible log gap ≤ 1.0 s, so 0.5 s steps are admissible.<br>• The v7 `s2_geom` geometry follows the same rule.<br>The **VLM-CoT goals** (traffic-light colour etc.) do not exist on NAVSIM; they become IGNORE. Per-frame `traffic_lights` is in the pickle (MEASURED key), so a TL label is buildable later. | **YES** for lateral/longitudinal classes and the continuous constraints (labels may use ego, PI 2026-08-03). VLM goals: IGNORE. |
| **nav INPUT** (`--nav-from-v7`, `--r8-nav-from-v9`: a per-frame token derived from the ego future) | NAVSIM `driving_command` at t0. This is the **exact inference feed**: `refcv8_bridge.py:163` mapping. | **YES.** Use `driving_command`, **not** the future-derived token, so that training equals inference. The v9 nav args stay future-derived with the recipe's 0.5 dropout; at inference the args are always unknown, as on PhysicalAI. |
| **agent head** (`--agents head --w-agent 1.0`, `--agent-join`) and **box3d** (`--w-box3d 1.0`, `--join3d`) | **OpenScene per-frame 3D annotations exist** (MEASURED, `_probe_pkl.py` on a navtest pickle): `anns.gt_boxes` `[N, 7]`, `gt_names`, `gt_velocity_3d`, `track_tokens`, `instance_tokens`. 46–297 boxes per frame, mean 124.9.<br>Classes: vehicle, pedestrian, bicycle, generic_object, traffic_cone, barrier, czone_sign. | **YES, with a declared class map** onto refc's 10-class enum (`bev_raster.py:93-94`):<br>• vehicle → automobile; nuPlan does not split truck/bus/trailer, so those three classes get no positives;<br>• pedestrian → person;<br>• bicycle → rider;<br>• cones, barriers, signs, generic → ignore.<br>`--vis1-sidecar` and `--join-defect-masks` must be rebuilt or dropped. |
| **map head** (`--map-hires on --w-map-hires 1.0 --map-gt-root sam3_gt_v3`) | There is no SAM3 GT on NAVSIM. ⛔ `--map-hires on` **refuses without `--map-gt-root`** (`refc_v3_train.py:2161-2165`), and `map-hires on` is load-bearing: the planner reads `--bev-source map_hires_pool`. | **OFF**: `w_map_hires 0` plus either an all-IGNORE GT store or a guard relaxation for w = 0. That is a trainer-input change, not an architecture change.<br>Option for later: rasterise the nuPlan HD map into the SAM3 schema (maps as training labels are admissible, PI 2026-10-11). |
| **LAW aux** (`LAW_WEIGHT`; the target is the pooled latent of `future_frames[:, LAW_AHEAD-1]` = t0 + 0.5 s, `:5581-5585`) | The t0 + 0.5 s frame is the next 2 Hz frame. navtrain has `frame_interval 1` (MEASURED from the filter header), so it is usually the **next token's current frame**. | **YES.** At a log end or a filter gap, fall back to the t0 frame and count the fallbacks. Every other future slot is never read; a control must prove that. |
| **set-speed channel** (`--r8-speed-input label`, `--max-speed-input-v6`, `--speed-max-sidecar-v6`) | Possible source: the nuPlan map speed limit. Thor holds the Las Vegas map only (GROUNDWORK); the dev box has all four maps. | **Feed the UNKNOWN row always.** This equals the R8_LEGAL inference feed. It is a declared departure from p_unknown 0.45. |
| **per-clip camera rig** (`--agent-rig-extrinsics`; `--map-hires on` refuses a single-camera file, `:8985-8994`; schema `FrontWideExtrinsics` qx qy qz qw x y z, `:2287-2345`) | Per token: the stitch's virtual camera (`frames416.rig_record`, `:144-154`) plus the road-plane height. | **YES.** A converter is needed, with a lift-parity control against `hires_lift_for_rig`. |
| **warm start** (`--init-from`) | `refcv8_train.warm_start_from` (`refcv8_train.py:914-948`) accepts a refcv8 checkpoint: strict, and only refcv8 seam keys may be missing. θ_DEC is a refcv8-shaped file. | **YES.** `--r8-alloc-emit-start` must be ≥ 1 for a warm-started run (`refc_v3_train.py:1041-1044`); the recipe has 1000. |

## 3. Where navtrain data lives (probed 2026-10-21, 20:02 Berlin)

| location | what is there | evidence |
|---|---|---|
| **Thor** `/home/nvidia/r8ns/navtrain_dl/` | `navtrain_current_1.tgz` (9,469,285,958 B), `navtrain_history_1.tgz` (3,831,962,664 B), `openscene_metadata_trainval.tgz` (7,052,210,642 B) | MEASURED (`ls -la`) |
| **Thor** `/home/nvidia/r8ns/navtrain/` | `navsim_logs/trainval`: **1,310 log pickles** (14 GB). That is ≥ the 1,192 navtrain logs, so all of navtrain's metadata is already on Thor.<br>`frame_bank`: the 2,089-token selection bank, 3.6 GB, 2,960 frames, raw u8 at **1,277,952 B per frame**.<br>`nsexp`: the selection metric cache (916 MB). | MEASURED (`ls \| wc -l`, `du -sh`, `BUILD_REPORT.json`) |
| **Thor** θ_DEC | `/home/nvidia/r8ns/navtrain_sel/wiseft/ckpt_mod2/ckpt_wiseft_mod2_DEC.pt`, 403,204,417 B, md5 `dad4cb5b5fb316922b4e9a7e85cee8f8`, read back bit-exact | MEASURED (`WISEFT_MOD2_BUILD.json` on Thor) |
| **Thor** capacity | 169 GB free of 937 GB (82 % used). GPU at 95 % with other agents' jobs at probe time. MemAvailable 104 GB. | MEASURED |
| **A40 pod** `growing_orange_anaconda` | `ssh` → **Connection refused** (2026-10-21 20:02 Berlin). The pod had **no persistent volume** (registry §14.1 "Location"), so its `navtrain_pixels` (413,152 frames × 4 cams) must be assumed gone. REFe's training banks (targets, not pixels) survive on HF `Sayood/tanitad-refe-navtrain-banks`, 8.1 GB. | MEASURED (probe); INHERITED (`STOP_CONSEQUENCES_2026-10-11.md`) |
| **dev box** | `E:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz` (4.5 GB, 38 logs), the REFe m5 pixel sample (1,900 random tokens, no metadata), and the navtrain filter. **No trainval metadata.** The uplink is about 1.2 MB/s. | INHERITED (GROUNDWORK §1) |
| **HF** `OpenDriveLab/OpenScene` (public, cc-by-nc-sa-4.0) | `navtrain_current` 32 shards, **304,548,267,046 B**; `navtrain_history` 32 shards, 144,459,143,360 B | INHERITED (GROUNDWORK §1, HF metadata API) |

⇒ **The ST design needs only the 31 remaining `navtrain_current` shards (about 295 GB) on Thor.**
* The trainval metadata, θ_DEC, the selection sample, its bank and its metric cache are already there.
* The history shards are not needed: ST reads t0 only, and the LAW target is the next token's current frame.
* At the MEASURED Thor HF rate the download takes **6.4 h**: S1 moved 20,353,459,264 B in 1,593 s = 12.8 MB/s (`RUN_LEDGER.json`). At the INHERITED pilot rate of 31.2 MB/s it takes 2.6 h.
* Disk: the shards must be **streamed** (download, extract L0/F0/R0, stitch, encode, delete). A raw-u8 bank is 99,824 × 1,277,952 B ≈ **128 GB**, which leaves too little headroom in 169 GB. A **PNG** v2ep cache is about **53.8 GB**, and it is lossless, so the training pixels equal the eval bank's pixels bit-for-bit.
  * MEASURED on 40 real navtest stitch frames (`PNG_PROBE.json`, `_png_probe.py`): `torchvision.io.encode_png`, the trainer's own PNG codec at `v2_compressed.py:143`, gives **538,884 B per frame** (457k–619k), 2.37× smaller than raw.
  * The round trip was bit-exact on 40/40 frames.
  * Encode time was 0.115 s/frame on one dev-box thread; Thor's rate is unmeasured.

## 4. What in `stack/scripts/refc_v3_train.py` @ fb47a58 assumes PhysicalAI episodes

| assumption | lines | NAVSIM adaptation (no architecture change) |
|---|---|---|
| Episodes are `<clip_id>.v2ep.pt` clips at 10 Hz with a D-015 3-frame stack, windowed at every frame by the unchanged `FlagshipWindowDataset` / `FailLoudWindowDataset`. | `:11073` (`--v2-cache`), `:9292-9303`; `v2_dataset.py:1-45`; `refb_train.py:122+` | Write **one short v2ep "token-episode" per navtrain token** in the trainer's own format. T = 2 (stack) + 8 (window) + 60 (horizon) ticks at 10 Hz:<br>• ST frames (t0 repeated);<br>• poses/actions interpolated from 2 Hz;<br>• the LAW slot filled with the t0 + 0.5 s frame. |
| Joins key on `stable_episode_id(clip_id)` and on the RAW v2ep frame index (= episode index + n_stack − 1). | `:4542-4548, 9360, 9672-9696, 7856-7880` | Use a synthetic `clip_id` per token (`navtrain:<token>`). Every sidecar is emitted under it, so the joins work unchanged. |
| The eval-clip refusal reads v7.2 EVAL clip ids from filenames. | `:7186-7220, 9302` | It passes trivially on NAVSIM ids. A **NEW guard** must refuse the 38 selection logs and every navtest log (control K3). |
| The v9 release is **md5-pinned** to the PhysicalAI build. | `:12272-12279` | A NavSim v9 release is passed with its own md5 (`--r8-v9-md5`). |
| Per-clip `FrontWideExtrinsics` come from PhysicalAI `calibration/sensor_extrinsics`; map-hires refuses a single camera. | `:2287-2345, 8985-8994` | Use a per-token table converted from the stitch rig. |
| SAM3 map GT is required under `--map-hires on`. | `:2161-2165` | Map off (see §2). |
| Pose-sync (X10) and clip-clock sidecars correct PhysicalAI's camera/egomotion timing. | `:11119-11131`, `:4485-4515` | Omit them. OpenScene frames and poses share keyframe timestamps. |
| Speed-max road-type sidecar and obstacle.offline joins (`--agent-join`, `--join3d`, `--vis1-sidecar`, `--join-defect-masks`). | `:12419, 11563-11660, 12119` | Build each from OpenScene `anns`, or feed the unknown row (§2). |
| Anchors `refc_anchors_6s_v0cond_alat_117.pt` are k-means over PhysicalAI controls. | `:11289` | **Keep.** They are part of the architecture (117 anchors, alat units). |
| In-run eval and B1 guard on PhysicalAI eval139 (`--eval-cache`, `--b1-guard-ref-ap2m 0.21211`). | `:11098-11108` | **Keep** as a **forgetting monitor**. ⚠️ The B1 guard reference is the known-blind RAW minimum (registry §16.13), so drop the guard or re-reference it. |
| Horizon parity `max_horizon = 20` enumeration. | `:9331` | Unchanged. The token-episode supplies ≥ 20 future ticks. |

## 5. Can refc take multi-camera input without an architecture change? **NO.**

From source:
1. The encoder input is `[B, 3K, H, W]` per window row (`timm_trunk.py:908-944`). There is no camera axis.
2. One `CanonicalFrame` (`FRAME_416x1024`) is asserted at build and at the NavSim seam (`refcv8_bridge.py:261`; `frames416.py:106-111`).
3. The BEV lift bank is built from **one** camera pose per clip (`refc_v3_train.py:2287-2345, 8985-8994`).

A second view needs a camera axis in the trunk call, a lift per camera, and fusion. That is new structure.

**The only parameter-neutral proxy** is a **wider cylindrical stitch into the same 416×1024 tensor**, for example about 170° from the same L0/F0/R0 jpgs (`f_ref` ≈ 345 px).
* It needs no new download and no new weights.
* It is a **declared geometry change**: new frame constants, a re-derived lift bank, and the `assert_frame` / `image_hw` guards re-pointed.
* It halves angular resolution against REFe (0.45×) and **cannot include the rear camera**.

It is therefore a confounded "camera" arm. It is offered as optional A2 in the design, not as a clean isolation.

## 6. Things that would silently break the experiment (traps found while auditing)

* **Train/inference feed skew.** On PhysicalAI the nav token is future-derived; at NavSim inference it is `driving_command`. Training must use the inference function, pinned by a bit-exact parity control (K1).
* **Selection-sample leakage.** The 2,089 selection tokens sit in navtrain. All **38** scanned logs (3,464 navtrain tokens, 2,089 readable; MEASURED, `run/sample/NAVTRAIN_SAMPLE_TOKENS.json` `per_log`) must leave the training set.
  * Training pool = **103,288 − 3,464 = 99,824 tokens / 1,154 logs**.
  * Consecutive 2 Hz tokens share 13 of 14 frames, so excluding by token is not enough; exclusion is by log.
* **θ_DEC was chosen ON the selection sample** (DECLAM / MOD2 picked λ and the module group there). Its 76.36 / 75.68 carry a winner's-curse bias. The **decision comparison must be navtest**, never the navtrain sample.
* **The EMA at 0.9998 has a 5,000-step time constant.** At 0.5 epoch the EMA is still 53 % θ_DEC; at 2 epochs it is 8 %. Read EMA and RAW at every snapshot.
* **Shared Thor GPU.** Three GPU-bound jobs ran 3.9× slower (memory note, INHERITED). The fine-tune needs the GPU exclusively, and no eval may run on the box while it trains.
* **Launch gate is binding** (PI 2026-09-26): a profile and a PASS token are needed before step 1.
