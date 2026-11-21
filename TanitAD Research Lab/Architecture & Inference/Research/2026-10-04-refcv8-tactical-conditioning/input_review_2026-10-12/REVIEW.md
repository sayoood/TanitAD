# refcv8 INPUT REVIEW — data, resolution, cycle time / past frames, extrinsics, REFe learnings (2026-10-12)

**Asked by the PI, 2026-10-12:** *"Can you review that we are using the right data with the right resolution, cycle time
including past camera frames, considering extrinsics and learnings from REFe."*
**Scope:** the refcv8 launch argv as smoked on Thor (`/home/nvidia/refcv8_smoke/smoke.argv.json`, run dir
`/home/nvidia/refcv8_smoke/run/config.json`, smoke 2026-10-11, 9.212 s/step) and the running R2 ladder
(`/home/nvidia/refcv8_ladder/W/argv/V-R8.json`: same cache, same 416x1024 frame, same window, resnet34 tiny rig, b8).
**Tip read:** `origin/agent/arch-inf-20260803` @ `dfaf297`. Zero GPU. Read-only on Thor (config/manifest heads only).
Two zero-GPU probes were run on the dev-box CPU against the refcv7 checkpoints refcv8 warm-starts from
(`D:/refcv7_eval_kit/ckpt/ckpt_{5000,50400}.pt`, md5 of 50400 = `b418d0fc…`, matches `MD5SUMS`).

## Bottom line

* **Nothing in the input pipeline blocks the refcv8 launch, and nothing requires stopping the ladder.** The data are the
  right data (per-clip extrinsics everywhere, both rigs handled, pose-synced labels), the geometry is the one refcv7 was
  trained on and refcv8 warm-starts from, and every finding below is common to all ladder arms.
* **The one structural problem is TEMPORAL, and it is a refcv9 question, not a refcv8 fix.** The model is trained on
  10 camera frames at 10 Hz (t-0.9 ... t s). The tactical brain that consumes them (`PhiTac`) was *designed* for
  1 Hz samples over 3 s, and **NAVSIM cannot deliver 10 Hz frames at all** (2 Hz, 4 frames). At NAVSIM every 10 Hz slot
  is synthesised. The cost of that mismatch on our own eval has **never been measured**; one zero-training eval measures
  it (section 2.5). It is the next lever, blocked only on a GPU slot.
* Changing resolution, frame stride/count, rectification or masks **now** would change the trained input: it voids the
  refcv7 warm start's geometry and the ladder's transfer to refcv8. None is recommended before launch.

## 1. Resolution and projection

**What we use (MEASURED, `/home/nvidia/data/refcv6-b1-416x1024-train/MANIFEST.json` `frame` block, and smoke
`config.json`):** projection **CYLINDRICAL**, 416 x 1024, `f_ref` 488.924 px/rad, column linear in azimuth:
HFOV = 2·(W/2)/f_ref = **120.00°**, **8.533 px/deg** (`deg_per_col` 0.1171875). Rows are pinhole per column:
VFOV = 2·atan((H/2)/f_ref) = **46.09°**, i.e. ±23.05° about the boresight. PNG (lossless), `n_stack` 3. Builder
`build_v2ep_wide.py` -> `v2_compressed.build_compressed` -> `calib.cylindrical_rectify` (UNMODIFIED). refcv7 used the
same cache and `image_hw [416, 1024]` (MEASURED, `/home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json`).

**Native vs cache.** Native front-wide is a 1920x1080 f-theta fisheye; corpus-median paraxial focal **927.5 px/rad =
16.2 px/deg** (`stack/tanitad/data/calib.py:376-391`, MEASURED in that file over 500 clips). The cache keeps
**0.53x the native linear angular resolution** at the centre (488.9/927.5) and ~0.57x at ±60° (f-theta local slope
~859 px/rad, ESTIMATED from the median polynomial).

**What the trunk sees (MEASURED, smoke config):** ResNet-101 stride-8 tap 52x128 (newest frame only, `s8_frame:
newest`), stride-16 26x64, stride-32 13x32; BEV tokens 30x16; map branch 0.25 m lift grid 400x240 -> 10 cm
1000x600 over 100 m x ±30 m; planner BEV crop 60 m x ±16 m.

**Rigs.** Per-clip principal point is enforced: `cylindrical_rectify` REFUSES the corpus-median intrinsic
(`calib.py:960-985`). Rig B's shallower below-boresight coverage leaves a black bottom strip: **2,721 / 4,713 clips
carry 26–43 black rows** (MEASURED, `/home/nvidia/data/_b1_416_strip_census_full.json`), equalised for ALL clips by
`--equalize-bottom-rows 43` in the trunk's normalisation (train and eval alike) and in the lift's valid mask
(`refc_v3_train.py:8993-8995`).

**What is lost (ESTIMATED from f_ref and the measured median mount height 1.312 m, pitch 0):**
* *Near field:* the 43 zeroed rows are 18.7°–23.05° below the boresight = ground nearer than **~3.1–3.9 m** (largely
  hood). Negligible.
* *High elevations:* a signal head ~5 m above the road leaves the frame (>23.05°) closer than **~8.7 m**.
* *Far field is row-starved:* a ground point at distance d sits f·h/d px below the horizon -> **20–40 m spans 16 px
  (2 s8 rows), 40–60 m 5.3 px, 60–100 m 4.3 px (<1 s8 row)**. MEASURED consequence, refcv7 step 50,400 eval
  (`thor_reads_2026-10-12.txt`): lane IoU **0.237 / 0.048 / 0.009 / 0 / 0** over the 0–20 … 80–100 m bands; arrow,
  crosswalk, edge **0 beyond 20 m**; drivable degrades gently **0.735 -> 0.509**. Whether thin classes are
  resolution-limited or SAM3-label-limited at range is **not separated** (HYPOTHESIS). The planner crop stops at 60 m.
* *Horizon jitter:* the rectification is about the camera **optical axis**, not gravity-levelled
  (`calib.py:906-924`); per-clip mount pitch spans **−0.0755 … +0.0683 rad** (MEASURED, smoke config
  `agent_rig_camera`), so the horizon row moves **~±35 px** across clips (ESTIMATED, f·tan pitch). The lift
  absorbs it geometrically (per-clip pose); the trunk must absorb it as nuisance variation.

**Against SOTA:** DriveZero feeds 4 cameras at 960x512 (PUBLISHED, `2609.06055` §3.2, quoted in
`2026-09-20-refe-plan/REVIEW_7_GAP_TO_PAPER.md:133`); on nuPlan F0 (fx 1545, MEASURED in `REVIEW_5_CAMERAS.md:76`)
that is ~13.5 px/deg vs our 8.53. TransFuser / DiffusionDrive on NAVSIM stitch L0/F0/R0 to 1024x256 (INHERITED, library
`2205.15997`, `2411.15139`, not re-read this session) — coarser vertically than ours.

**Verdict:** right for refcv8 (it is refcv7's trained geometry; s/step 9.212 against the 10.5 ceiling leaves no room
for more pixels). **Later (refcv9):** a gravity-levelled rectification and more vertical pixels for the 20–60 m band.
Cost: a full re-rectification from source (a PNG cache can only be sliced to a centred sub-frame at the SAME f_ref,
`calib.py` "RIG-CLEAN FIELD" block), ~4.3 h wall at the MANIFEST's 39.4 s/clip/12 workers (ESTIMATED), plus trunk
time scaling with pixels. **Ladder impact: none now; it would be a new arm, not a restart.**

## 2. Cycle time and past camera frames

### 2.1 What is built (MEASURED, code at the tip + smoke config)
| item | value | where |
|---|---|---|
| source camera | ~30 Hz (pose-sync self-test `delta_max` 33.17 ms = one 30 Hz period) | smoke `config.json` `pose_sync` |
| cache grid | 10 Hz pose grid, dt 0.1005 s; `t_image` = first camera frame >= `t_pose` | sidecar meta rule (`v2_compressed.py:119-121`) |
| window | **W = 8 rows** | `refc.py:754` |
| per-row stack | **K = 3** consecutive raw frames (t-0.2, t-0.1, t), 9 channels, oldest first | MANIFEST `n_stack` 3; smoke `seams.trunk_frames` 3 |
| frames through the trunk | **10 distinct frames per sample, t-0.9 … t** (dedup on) | `--trunk-dedup-frames` help, `timm_trunk.py:872-902` |
| K-fusion | `concat1x1`, identity-initialised on the newest frame, at s16 and s32 | `timm_trunk.py:930-940`, `--trunk-fuse` help `refc_v3_train.py:11944` |
| planner / map / boxes | **last row only** (`fmap`, `fmap_s16`, s8 tap on the newest frame) | `refc.py:4695-4714` |
| tactical brain | **all 8 rows**: pooled s32 -> `PhiTac` causal TCN (W 8, dilations 1/2/4) | `refc_v3.py:1140-1141, 1608-1610, 2533-2536` |
| strategic | GRU over the 8 rows, output bypassed (`--no-strategic`) | `refc.py:4715`, `:5074` |
| ego history | GRU, 8 steps x 0.1 s, 3 channels (v, a_long, yaw rate), past only | smoke `seams.ego_history`; `--ego-history` help `:12435` |
| v0 | measured at t0 (admissible, PI 2026-09-02) | SPEC_REFCV8 §2.4 (`:136-142`) |
| plan | 6.0 s at dt 0.1, horizons 0.5…6.0 s | smoke `horizons`, `anchors.artifact_declared` |
| NOW | newest raw frame of the last row: `grid_start + (t+w-1+n_stack-1)·dt` | `refc_v3_train.py:4244, 4420` |
| LAW aux | pooled latent of ONE future row (index 4, frames t+0.3…t+0.5), no-grad target | `refc_v3_train.py:5581-5585` |

⚠️ **Correction made during this review, recorded so nobody repeats it.** I first read rows 0–6 as dead (the trainer
passes no `hierarchy_hook=`, and `ctx` is bypassed). That was an absence found at ONE location: the hook is supplied by
`refc_v3.py:2533-2536` and feeds every row to `PhiTac`. The trunk's 10-frame window **is** consumed.

### 2.2 Finding T1 — the tactical brain is fed a different clock than it was designed for
`PhiTac`'s own contract (`stack/tanitad/models/tactical.py:99-111`): *the W inputs are **1 Hz samples** of the operative
latent (t-3 s … t for W=4), not consecutive 10 Hz frames — down-sampling in time IS the abstraction mechanism.* Built:
**8 consecutive 10 Hz rows = 0.7 s of context** (`refc_v3.py:1140-1141`, window = `cfg.core.window` = 8). Tactical
labels span NOW+2…+8 s (PI R8-1). Evidence class: MEASURED (code). Effect on metrics: unknown.
Control that read its known value: in refcv7, the dilation-4 block's oldest tap (t-8 rows, which only ever sees the
causal zero-padding at W=8) is **5.334 at step 5,000 and at step 50,400** — never updated — while every live tap moved
(`raw/phitac_taps.txt`). The probe therefore sees training updates, and the older rows' taps are trained.

### 2.3 Finding T2 — NAVSIM cannot supply our temporal input
NAVSIM hands **4 frames at 2 Hz (t0-1.5, -1.0, -0.5, 0)** (INHERITED, `frames416.py:39-42`; PUBLISHED NAVSIM `2406.15349`).
The bridge fills our 10 slots by **ST** (every slot <- t0, the primary arm) or **NT** (nearest 2 Hz frame)
(`refcv6_bridge.py:50-58`, `slot_sources6`). Consequences:
* the **K=3 fusion always sees a static stack** at NAVSIM (slots -0.2/-0.1/0 map to t0 under ST **and** NT);
* `PhiTac` sees either 8 copies of t0 (ST) or two 2 Hz frames each held 3–5 slots (NT); training never had either;
* ego history is linearly interpolated from 2 Hz states (`refcv6_bridge.py:127-131`).

How much the trained model leans on history (MEASURED weight statistics, `raw/fuse_cos.txt`, refcv7):
|past-frame blocks summed| / |newest block| of the 1x1 fusion = **0.37 (s16) and 0.56 (s32) at step 50,400**, up from
0.18 / 0.19 at step 5,000; cos(oldest, middle block) 0.71–0.82 (history acts mostly as a smoothed sum, not a pure
difference). Weight-level only, not a functional ablation.
What NAVSIM already measured (MEASURED, `…/2026-09-28-refcv7-standard-tests/navsim/RESULT.md:112, 210`): A1 (ST) − A1NT
(NT) = **−0.0008** and **+0.0013** PDMS, plan endpoints 0.006 m apart — inside the inference seed floor (|Δ| 0.0136 /
0.0069). ⚠️ That arm varies only the **PhiTac rows**; the fusion's static stack is identical in both arms, so A1−A1NT
**cannot see** the fusion's sensitivity to missing 0.1/0.2 s motion. It answers "ST vs NT", not "NAVSIM history vs real
history".

### 2.4 Against the literature
Current-frame-only is the strong NAVSIM contract: DriveZero (PUBLISHED, `2609.06055`, via REVIEW_7:133), DrivoR
(library note: "DrivoR's no-history gap"), TransFuser / DiffusionDrive (INHERITED, `2205.15997`, `2411.15139`).
Temporal methods on nuScenes (BEVFormer/UniAD/VAD) use ~2 Hz history queues (INHERITED, not banked here). Our 10 Hz,
0.9 s camera window matches neither, and it is the one NAVSIM cannot feed.

### 2.5 Verdict and the next lever (not run: blocked on a GPU slot)
**refcv8: keep the built contract** — it is refcv7's, the warm start carries trained history weights, and the ladder runs
it. **The cheapest discriminating experiment, zero training:** refcv7 `ckpt_50400.pt` on eval139, four arms on the same
windows: REAL (as trained) · STACK-HOLD (each row's K=3 stack <- 3 copies of its newest frame; isolates the fusion) ·
NAVSIM-NT (the bridge's 2 Hz construction applied to PhysicalAI frames) · NAVSIM-ST (all slots <- t0). Four metric
families, paired episode-cluster bootstrap, **plus an inference-seed replicate of REAL** as the floor (CLAUDE.md
third-variance rule). Both outcomes committed now:
* NAVSIM-ST/NT within the replicate floor of REAL -> history is not load-bearing; **refcv9 adopts the NAVSIM-native
  contract: K=1, rows at 0.5 s, W=4 (t-1.5…t)** — exactly NAVSIM's inference contract, 2x the tactical context, and
  **10 -> 4 trunk frames per sample**. The trunk is **92.8 % of per-sample FLOPs** (MEASURED, refcv7 opcount
  `…/2026-09-27-refcv7-step-cost/raw/opcount_devbox/R7_opcount_b1/opcount_steps.json`: encoder 6.32 of 6.81 TF), so
  FLOPs drop ~56 % (ESTIMATED); wall-clock less, because the map branch is bandwidth-shaped. No cache rebuild: raw
  10 Hz frames are stored individually and stacked at load (`v2_dataset.py:109-149`) — a loader stride change.
* REAL clearly better -> NAVSIM numbers carry a history-mismatch penalty; fix by training with a 2 Hz-hold augmentation
  (rows built the NT way with p≈0.3) so the model learns NAVSIM's input.
Cost ~1–2 GPU-h; needs a Thor lock gap or dev-box GPU authorisation. **Ladder impact: none** (an eval of a finished
checkpoint).

## 3. Extrinsics and intrinsics

* **BEV lift (map branch, planner BEV, coupling):** PER-CLIP full pose — quaternion + translation in metres — through
  `build_lift_geometry` on the cylindrical frame (`stack/tanitad/models/bev_lift.py:146-185`); the trainer REFUSES a
  single-camera table under `--map-hires on` (`refc_v3_train.py:8985-8995`); a missing clip REFUSES
  (`refcv6_perception_branch.py:406-420`). MEASURED.
* **Agent heads:** `agent_rig_camera` source `extrinsics`, mount pose `from-sensor_extrinsics`, **PER-CLIP**, table of
  **4,508 clips**, coverage **4,369 / 4,369** train (frac 1.0, no default), height 1.203–1.687 m (median 1.312 m), pitch
  −0.0755…+0.0683 rad (MEASURED, smoke config). Units: metres, radians.
* **Intrinsics:** per-clip f-theta via 1,363 calibration chunks, 0 missing (VIS-1 sidecar sources, smoke config);
  the frame itself is canonical (one `f_ref`), so intrinsics act once, at rectification.
* **Pose sync (X10) is ON:** Δ(image − pose) **0–35.4 ms, mean 16.5 ms**, 4,369/4,369 clips; shifts `pose_last`,
  `future_poses(_ext)`, `pose_hist`, `goal_tac`, `actions`, `future_actions` (MEASURED, smoke config `pose_sync`).
  refcv7 trained WITHOUT it (no `pose_sync` key in its config.json), so refcv8 corrects label timing.
* **Finding E1 (later):** the 3-D agent boxes are joined at the **pose-grid time** (`build_b1_agent_join_3d.py:35,
  220-254`, `register_poses_to_time`) and are **not** in X10's shifted fields, so box-vs-image offset is up to the
  sync delta (≤35 ms) plus the track-sample offset. ESTIMATED ≤0.7 m at 20 m/s ego, more for oncoming traffic. Cheap
  check: histogram of (t_track_sample − t_image) from the join's `dt_s` plus the sidecar. Aux heads only. Ladder: none.
* **NAVSIM:** a virtual cylindrical camera at CAM_F0's boresight, **rolled level**, stitched from L0/F0/R0 (pure
  rotation; 0.3 m centre offsets ignored) and a per-rig lift table from CAM_F0's `sensor2lidar` pose (INHERITED,
  `frames416.py` docstring). Extrinsics are handled. **Finding E2 (later):** nuPlan's 38.5° VFOV cannot fill our 46.1°,
  so NAVSIM frames carry a scalloped black band at **top and bottom** (10.7 % of the 256x640 frame, INHERITED
  `…/2026-08-28-navsim-corpus-adaptation/FRAME_DECISION.md:10`; somewhat more at 46.1°). Training equalises only the
  bottom 43 rows; **no top-band handling exists** (two probes over `stack/` and the docs, with a control that reads 3
  hits for `equalize_bottom_rows`). Fix: random top-band masking during training. Ladder: new arm, not a restart.

## 4. Learnings from REFe — carried or not
| REFe learning | evidence | refcv8 |
|---|---|---|
| Declare and match the rate; a 10 vs 20 Hz sim moved targets ADE 0.23 m / FDE 0.60 m | MEASURED, memory `refe-sim-rate-10-vs-20hz` (`refe/diag_sim_rate.py`) | bridge declares 10 Hz slots <- 2 Hz frames and interpolates ego history: **carried at eval**; **training never sees the 2 Hz construction** (T2) |
| Measure camera–pose pairing (REFe: mean 24 ms, max 38 ms) | MEASURED, `REFE_MODEL.md:147` | **carried** for poses (X10); **not** for agent boxes (E1) |
| Current frame only is the SOTA NAVSIM input | PUBLISHED `2609.06055` via `REVIEW_7_GAP_TO_PAPER.md:133` | **not carried** — 10 frames; refcv9 candidate after §2.5 |
| Camera geometry analytic from K + extrinsics, not a learned table | `MODULE_SIZING_STUDY.md:23-36` | **carried** (per-clip analytic lift) |
| Per-camera resolution 960x512 (~13.5 px/deg on F0) | PUBLISHED, as above | **not carried** (8.53 px/deg; budget-bound) |
| STOP is a strong NAVSIM baseline (A1 − STOP +0.0011 at refcv7 warmup) | MEASURED, refcv7 navsim `RESULT.md:210` | the BAR-R8-N1 "> STOP" bar exists (SPEC_REFCV8:680) — carried |

## 5. Verdict table
| # | finding | severity | change | cost | ladder |
|---|---|---|---|---|---|
| T2 | NAVSIM cannot feed the 10 Hz, 10-frame input; fusion always static there; mismatch unmeasured | **should run before NAVSIM numbers are quoted** (eval only) | the §2.5 four-arm eval139 discriminator | ~1–2 GPU-h, 0 training | none |
| T1 | PhiTac fed 0.7 s @10 Hz vs its 3 s @1 Hz design | later (refcv9) | NAVSIM-native K=1, 0.5 s rows, W=4 if §2.5 allows | loader change, no rebuild; −56 % FLOPs (EST) | new arm, not a restart |
| R1 | far-field rows: lane IoU ~0 beyond 40 m; resolution vs label limit unseparated | later | per-band label-density check first; then levelled rectification / more rows | rebuild ~4.3 h + trunk time | new arm |
| R2 | horizon jitter ±35 px from un-levelled rectification | later | gravity-levelled rectification | rebuild | new arm |
| E1 | agent boxes not pose-synced | later | apply X10 to box times | small; loader | none for transfer (aux) |
| E2 | NAVSIM black top band never seen in training | later | random top-band masking | training change | new arm |
| D1 | loader decodes and ships 20 future rows; only row 4 is read; all 20 go to fp32 on device | later (efficiency) | ship only the LAW row | small; function-identical | none |
| — | everything else (per-clip extrinsics, both rigs, pose sync, v0/ego-history admissibility, per-frame nav at NOW) | correct | — | — | — |

**Nothing blocks launch. Nothing requires stopping the ladder now.** Any input change before launch would alter the
trained input of every arm after the ladder started, and of the warm start — not second-order — so none is recommended.

## 6. Data line D1, in numbers
`frames+future_frames = 1.718 GB per collated batch (uint8), <= 6 in flight -> ~10.31 GB` (MEASURED,
`/home/nvidia/refcv8_smoke/train.log:40`); `max_horizon=20` (`refc_v3_train.py:9331`); only
`fut_frames[:, LAW_AHEAD-1]` is read (`:5583`, `LAW_AHEAD = 5` in `refc_train.py:114`), while
`frames_to_device(batch["future_frames"])` converts all 20 rows (`:5241`): ~4.9 GB fp32 per step at b16 (ESTIMATED
16·20·9·416·1024·4 B) on Thor's unified memory.

## Deliverable manifest
| artifact | where |
|---|---|
| this review | `C:/Users/Admin/qland/hold_inputreview/RESULT.md` and `…/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/input_review_2026-10-12/REVIEW.md` (same content) |
| probe code | `…/input_review_2026-10-12/code/{fuse_probe.py,fuse_cos.py,phitac_probe.py}` |
| raw outputs | `…/input_review_2026-10-12/raw/{fuse_cos.txt,phitac_taps.txt,thor_reads_2026-10-12.txt}` |
| commit message | `C:/Users/Admin/qland/hold_inputreview/msg.txt` |
Nothing is staged or committed (the brief says stage nothing). Nothing was written to G: or to Thor.
