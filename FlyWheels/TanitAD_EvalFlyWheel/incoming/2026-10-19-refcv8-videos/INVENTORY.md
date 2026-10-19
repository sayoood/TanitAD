# INVENTORY -- what the refcv8 eval banks carry for judgement videos (EvalFlyWheel, 2026-10-19, CPU only, read-only)

Evidence class on every line: MEASURED = read from the artifact named, today.  UNVERIFIED = stated as such.  Tier of everything below:
**T1 self-action OPEN LOOP on the logged eval139 frames (not driving performance).**  Clip UUIDs are never printed; episodes are SHORT clip indices (0..138).

## 1. Banked per-window dumps (refcv8)  -- MEASURED, `C:/Users/Admin/refcv8_eval_archive/battery/<tree>/`
Trees present: `step5000_unc`, `step10500[ema]_unc`, `step15000[ema]_unc`, `step20000[ema]_unc`, `step30000[ema]_unc`, **`step40000ema_unc` (used here)**,
`step5000_corr`; refcv7-50,400 re-rolls `r7on_step50400_unc`, `r7step{15000,20000,30000}_unc` (+ `r7ctl*` smokes).  Windows are the battery's S2 set:
**4,754 windows / 139 episodes**, origin rows `ws` every **5th row** (ws 7..11 .. 173..177, 34-35 per episode; clip = 201 raw frames, 10 Hz, 199 stacked rows).
Inference seed 0 carries every arm; seed 1 carries `os` + model-free controls only.  The sampler is stochastic: a re-run does NOT reproduce the plan (CPU re-run of one
window: selected plan 10.3 m away from the banked GPU plan, `sel_idx` 49 vs 125), while the tactical logits agree to <= 0.16 (argmax equal, 10/10 checks).

| layer | where (per episode `ep###.npz` unless noted) | shape / content |
|---|---|---|
| **plans (os)** | `panel_s0` `os`; `dump_s0/decisions` `plan_full_nav_true` | S2: 4 slots (0.5/1/1.5/2 s); **full 8 slots 0.5,1,1.5,2,3,4,5,6 s** (`plan_full`, ego frame at t0); `plan_full == extras nav_true.traj` bit-exact (control) |
| S6 plans | `s6_s0` (`os`, 27 windows/ep, ws <= 137) | 6 slots 1..6 s; arms `os`, `os_nav*`, `ha*`, `stop`, `b_refcv7_50400_s0/1`, `b_refcv5v2_s0/1`, `b_refcv4b` |
| GT | `panel_s0` `g` (ego frame, 4 slots); `decisions` `gt_future_ext` [n,60,4] world x,y,yaw,v at t0+0.1..6.0 s, `ep_poses` [199,4] | recomputed GT == banked `g` to 4.6e-6 m (control, ep114) |
| **refcv7-50,400** | `b_refcv7_50400_s{0,1}` inside every `panel_s0` / `s6_s0`; full re-roll in `r7on_step50400_unc` | 4 slots (S2) / 6 slots (S6).  NOTE `r7on` traj != the panel baseline (ep000 max 3.7 m): the panel arm is the one the paired reads use -> used here |
| **shipped re-timed plan** | `.../2026-10-15-refcv8-milestone-battery/raw/retime_deploy/DEPLOY8_POS_step30000{,ema}_unc_s{0,1}.npz` (`S2_pos8` [4754,8,2], `S2_plan8`, `S2_prior8`) | **30k ONLY** (no 40k deploy table exists) |
| **causal prior** | `DEPLOY8_POS_*` `S2_prior8` = `ha0_ext_pose` rolled at the 8 slots (`deploy8_forward_r8.py:262-264`) | model-free, so valid for any checkpoint's windows; same (clip, ws) order verified.  NOT equal to the panel `ha0_ext` (max 0.29 m) -- different object, prior8 used |
| **tactical lat / lon posteriors** | `dump_s0/refcv6_extras.npz` `nav_true.tacv6_lat_logits` / `tacv6_lon_logits` [4754,8] (v6 behaviour decoder), `lat_logits_tac` / `lon_logits_tac` [4754,8] (z_tac v7 heads), `tacv6_goal_logits` [4754,22] | vocab `vocab_v7.TACTICAL_LAT/LON_ACTIONS_V7`; also per nav condition (`nav_shuffled`, `nav_flipped`, `nav_zero`), `filter_off`, `vmax_zero`, `obey30` |
| GT tactical labels | extras `lat_v7`, `lon_v7` (-100 = unlabelled); `decisions` `lat_label`, `lon_label`, `route_label` | |
| **selected index / fan terms** | extras `nav_true.sel_idx`; `decisions` `sel_idx`, `sel_bank_nav_true`, `fan_term_heading_*` [n,149], `reach_keep_*` [n,149], `oracle_sel`, `sel_agrees_oracle` | the FAN ITSELF (149 candidate trajectories) is NOT banked |
| **ceiling / speed** | extras `v_max_ms` (4-way road-type label 8.33/13.89/27.78/33.33 m/s), `v_max_valid`, `gt_max_speed_6s`, `v0` | |
| **nav command** | `decisions` `nav_cmd` (0 FOLLOW_ROAD, 1 TURN_L, 2 TURN_R), `nav_valid` | **CONSTANT PER EPISODE in 139 of 139 episodes** (`raw/NAV_CONSTANCY_step40000ema.json`): it is a clip-level route command, not a time-local one.  Window totals F/L/R = 3,011 / 445 / 1,298.  Consistent with GT in aggregate (TURN_L windows: GT lateral at 6 s mean +2.6 m; TURN_R: -3.2 m) but NOT within a clip (ep13: TURN_R the whole clip, first manoeuvre is a left turn) |
| strategic route | `route_logits` exists in the model output; refcv8 is `--no-strategic` | shown as "off (n/a)" |

## 2. NOT banked  -- needs a dump (CPU or GPU)
* refcv8's **MAP-hires head** output (the 10 cm logits ride out on `out["perception"]["map_hires_logits"]`, `refc_v3.py:2192`), the **BOX head** slots (300 slots: presence, class, box3d,
  yaw, rates), the **allocation** (`r8_hyp_alloc`, `r8_alloc`, `r8_lat3`, `r8_lon6`), the **fan** (`anchor_traj` [149,8,2]) and any per-frame (non-window) output.
* -> `code/dump_heads_r8.py` (written, smoked, below).  Verified on CPU, 2 windows then all 34 windows of ep114: it captures every one of the above (keys listed in `keys.json`),
  the captured window is the banked window (v0 equal to 0.0), CPU-vs-banked tactical logits <= 0.16.  **The map/box heads are deterministic functions of the frames, so a CPU dump is a valid
  head dump**; only the plan / fan / selection are sampler draws (so the dump's fan is never drawn next to the banked plan).
* MEASURED cost of the CPU dump (ep114, 34 windows, 4 threads, fp32): **289.7 s = 8.5 s per window**, **19.3 MB per 34 windows (0.57 MB per window)**.  Extrapolated (ESTIMATED, linear) to all 4,754 windows:
  **~11.2 h CPU** (not practical) / **~2.7 GB** on disk; the GPU run is ~41 min (HANDOFF.md ii).  The CPU path is therefore for showcase episodes only; the three rule picks (13 / 103 / 114) are dumped on CPU (68 + 34 windows, 685 s / 68 windows with other jobs running, v0 control 0.0) and rendered as the `_HEADS` videos.
* MEASURED from those dumps: the box head's presence probability **never reaches 0.5** (max 0.399 / 0.289 / 0.324 for ep13 / 103 / 114), i.e. the box head is uncalibrated at EMA-40k; the videos draw p >= 0.20 (see VIDEO_MANIFEST.md finding 2).

## 3. Camera frames -- MEASURED
* `E:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/<clip>.v2ep.pt` x 139 (the `--eval-cache` of the refcv8 argv, re-pointed by `r8_repoint.py` / `refcv8_eval_remap_devbox.json`,
  kit root default `R8_EVAL_KIT=E:/refcv6_eval_kit`).  Payload: **PNG, 416x1024, `projection: cylindrical`, `f_ref 488.924` (= 305.577 x 1.6, same 120 deg HFOV; the 256x640 / 305.577 of the brief
  is the same camera at 1/1.6 scale)**, `n_stack 3`, 201 raw frames, `poses` [201,4] x,y,yaw,v, `actions` [201,2].
* Loader: `stack/tanitad/data/v2_dataset.py` (`_decode_stacked`; stacked row j = raw frames j..j+2, newest last; `poses[2:]` = the 199 stacked rows).  **MEASURED index convention:** `ws` = the stacked row of
  the window origin, `ep_poses[ws] == pose_last`, the camera frame at row j is RAW frame j+2.
* Per-clip extrinsics (quaternion cam->vehicle + mount xyz): `E:/refcv8_eval_kit/data/refcv8_train4572_eval139_extrinsics.json` (139/139).
* Cost: one episode decodes to 257 MB uint8; render of one 201-frame episode 18 s wall (CPU), so the full 139-episode set is ~45 min of CPU.

## 4. GT agent boxes and GT map -- MEASURED
* **Agent boxes: YES, 139/139 episodes.**  B1 3-D agent join `E:/refcv6_eval_kit/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz`: 26,394 labelled raw frames, 905,512 agents
  (cx, cy, yaw, l, w, occ, cz, h; per-frame EGO frame, +x fwd +y left; keyed on the RAW frame index).  Extracted once to `E:/refcv8_eval_kit/videos/_cache/eval139_agents3d.pkl` (41 MB; `code/extract_eval_agents_r8.py`).
  Caveat (UNVERIFIED): `cz - h/2` is about -0.6 m in the sampled lines, so footprints are drawn on the z = 0 plane (labelled) rather than as 3-D boxes.
* **GT map: YES, 137/139 episodes** (missing: episodes 92 and 136).  SAM3 v3 `E:/refcv6_eval_kit/data/sam3_gt_v3_eval/<sha12>.sam3mapgt.npz`: `cart_frac` [T,9,200,120] uint8 at 0.5 m (x 0..100 m, y +-30 m; col 0 = RIGHT),
  also `fine_codes` [T,1000,600] at 0.1 m (the head's own grid); RAW-frame indexed; pose alignment checked on ep0 (displacement equal to <= 0.005 m at raw 0/50/100/150/199).  Label-only (non-causal): never an input.

## 5. Brief vs measurement
* "eval139 set (138 episodes)": every banked dump holds **139** episodes; 138 appears only in cells that need a cruise window (e.g. `false_decel_ep` 138).
* The brief's 256x640 frame: the eval cache is 416x1024 (cylindrical, same optics); stated on every video's camera strip.
