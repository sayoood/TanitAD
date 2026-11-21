# VIDEO_MANIFEST -- refcv8 judgement videos (EvalFlyWheel, 2026-10-19, CPU only).  NOT staged in git: qland hold `hold_r8video` + the E: drive.

**Tier of everything shown: T1 self-action OPEN LOOP on the logged eval139 frames -- a diagnostic, not driving performance.**  One training seed, inference seed 0.
Checkpoint: refcv8-r101-s0 **EMA step 40,000** (`ckpt_ema_40000.pt`, md5 `57807456bd7169800be159ff95871d22`).  The final 48,852 set is NOT rendered yet (HANDOFF.md section i).
Episodes are the short eval139 clip indices (no clip UUIDs anywhere).  All videos: **1920x1080, H.264 (libx264, CRF 18, yuv420p), 10 fps, 201 frames = 20.1 s** (one full 20 s clip); reels 803 frames = 80.3 s.

## What is on every frame
| panel | content |
|---|---|
| title bar | arm, checkpoint md5, episode, time; tier stamp |
| camera (1280x520) | the logged front camera (cylindrical 416x1024, f_ref 488.9, column linear in azimuth -- the cylindrical model is used, not a pinhole); GT path (green), refcv8 plan (orange, 2 / 4 / 6 s marks), shipped re-timed plan (cyan; **30k EMA layer**, labelled -- no 40k deploy table exists), refcv7-50,400 (violet dashed), causal prior (grey), GT agents (blue, z=0) |
| BEV (340x508) | metric 0-60 m ahead +-20 m: GT path + GT agent footprints (blue) vs refcv8 plan (orange) and the same arms; GT green / model orange as in the camera |
| MAP (340x508) | GT SAM3 v3 map (GT-only videos); `_HEADS` videos split GT (left) | refcv8 MAP-hires head (right) with legend |
| tactical (640x520) | per-frame refcv8 lat / lon behaviour-decoder softmax with GT label + MISS/OK, NAV in, STRATEGIC = "off (n/a, --no-strategic)", speed, fed road-type ceiling, plan end-speed, the false-brake banner, head allocation (HEADS) |
| metrics (1240x508) | per-frame ADE 2 s / 6 s, this window and the clip mean so far, for refcv8 / refcv7-50,400 / shipped 30k layer / causal prior; speed-vs-time plot (plan vs logged human speed, ceiling) |

Held plan: plan snapshots refresh at 2 Hz (every 5th frame) and are held and re-expressed in the current ego frame with the logged poses; the age is printed.
ADE = mean over plan slots of the Euclidean error (m) vs the logged GT path (2 s: slots 0.5-2 s; 6 s: slots 1-6 s, only where the 6 s GT lies inside the clip).

## Picks -- BY RULE, never by how the plan looks (`code/pick_episodes_r8.py`, `raw/PICKS_step40000ema.json`)
| rule | episode | stat that selected it (MEASURED from the banked EMA-40k trees) |
|---|---|---|
| R1 cruise false-braking: most windows with GT 'cruising' (|v6-v0| < 1.5 m/s) AND plan class 'decel' (last-second plan speed - v0 <= -1.5), among episodes with >= 8 GT-cruising S6 windows | **114** | 24 of 24 GT-cruising S6 windows flagged (flag fraction 1.00); runners-up 102, 105, 51 |
| R2 sharp turn: largest |heading change| over any 3 s of the logged path, the 3 s span inside the plan-window coverage | **13** | 1.29 rad in 3 s; runners-up 12, 16, 64 |
| R3 stop / slow-down: speed >= 5 then < 1 m/s, largest drop | **103** | 15.6 m/s drop to standstill; runners-up 84, 39, 29 |
**R2/R3 amendment (stated, not hidden):** the first R2 pick (episode 31) turned out to have its turn in rows 2-32, *before its first plan window*, i.e. a showcase with no plan on it.  Coverage ("the span lies inside the plan windows") was added to R2 and R3; no plan-quality number entered the amendment.  Episode 31's video was deleted.

## Videos (all on E:, none in git) -- `E:/refcv8_eval_kit/videos/`
| file | dur | size (B) | md5 (12) | what it shows |
|---|---|---|---|---|
| `refcv8_ema40k_ep114_cruise_false_brake.mp4` | 20.1 s | 7,581,345 | 96b5750bdffd | R1.  Night road, ~21.5 m/s, GT cruising; the plan decelerates (end-speed 17.4 m/s, dv -4.1) and the red FALSE BRAKE banner is up for the whole clip |
| `refcv8_ema40k_ep13_sharp_turn.mp4` | 20.1 s | 13,746,707 | 8c3729f3e024 | R2.  Day, urban left turn at t = 1-4 s.  NAV input is TURN_R for the whole clip (see finding 1) |
| `refcv8_ema40k_ep103_stop_slowdown.mp4` | 20.1 s | 9,187,543 | cdb9717c7422 | R3.  Day, 16 m/s down to a standstill at a signalised junction (red light, a "right lane closed" sign); the plan's end-speed rises again after the stop |
| `refcv8_ema40k_ep12_sharp_turn_runnerup.mp4` | 20.1 s | 8,767,257 | 38ed90d55684 | R2 runner-up (extra) |
| `..._ep114_cruise_false_brake_HEADS.mp4` | 20.1 s | 7,648,190 | 3379be38270e | R1 with the refcv8 head layers (below) |
| `..._ep13_sharp_turn_HEADS.mp4` | 20.1 s | 13,964,861 | bb261351ce01 | R2 with the head layers |
| `..._ep103_stop_slowdown_HEADS.mp4` | 20.1 s | 9,236,218 | 4f9cd55bc0e9 | R3 with the head layers |
| `refcv8_ema40k_reel.mp4` | 80.3 s | 30,715,509 | c5c56f7845e2 | reel: intro (5 s) + per-episode title card (4 s) x 3 + the three GT-only videos + end card (3 s) |
| `refcv8_ema40k_reel_HEADS.mp4` | 80.3 s | 31,062,988 | 64dd04cbcdc9 | the same reel built from the three `_HEADS` videos |
(`_HEADS` = the map-hires head (GT | head split), the box-head footprints (orange, camera + BEV, presence >= 0.20) and the head allocation line from a **CPU head dump** of the same checkpoint,
`E:/refcv8_eval_kit/videos/heads_cpu_picks/` -- 68 windows, 685 s, v0 control vs the banked dump = 0.0 over 68 windows.  The head outputs are deterministic functions of the frames; the plan is the BANKED plan, not the dump's sampler draw.)

## Findings that surfaced while building them (MEASURED, per the artifacts named)
1. **nav_cmd is constant per episode in 139 of 139 episodes** (`raw/NAV_CONSTANCY_step40000ema.json`): a clip-level route command, not time-local.  In the sharp-turn pick (13) it is TURN_R for the whole clip while the first manoeuvre is a LEFT turn.  Needs a PI / Master Mind decision: is a time-local nav intended?
2. **Box-head presence is uncalibrated on this checkpoint**: the maximum presence probability over every captured window is 0.399 (ep13), 0.289 (ep103), 0.324 (ep114) -- it never reaches 0.5, so a standard threshold draws nothing.  Mean boxes per window at p >= 0.10 / 0.20 / 0.30: 26.9 / 3.7 / 0.2 (ep13), 41.0 / 1.7 / 0.0 (ep103), 6.1 / 0.6 / 0.1 (ep114).  The 0.20 display threshold is a display choice, not a detector threshold (`--box-p-thr`).  At 0.10 the day clip is cluttered with false footprints.
3. **Ep103 (R3): at the stop the plan accelerates** -- LON pred ACCELERATE (66 %) vs GT label HOLD at t = 12 s, speed 3.0 m/s, GT 6 s speed 0.0; the plan end-speed climbs to ~5 m/s while the human stands still (the spike is visible in the speed plot).
4. The shipped re-timed layer exists for 30k only; no 40k / 48,852 deploy table is built (nobody owns it).
5. GT map missing for episodes 92 and 136 (rendered, panel says "no map GT").  The brief's "138 episodes" is 139 in every banked dump.
6. Caveat (UNVERIFIED): agent footprints are drawn on the z = 0 plane because `cz - h/2` is about -0.6 m in the sampled join lines.

## Code integrity note
The renderer source was corrupted once during this work by a bad one-off patch (an empty-string `str.replace` inserted the patch text between every character, 40 MB) and **reconstructed by removing the inserted copies**.  Verified by re-rendering ep114 `_HEADS` with the reconstructed source and comparing all 201 decoded frame hashes (`ffmpeg -f framemd5`) against the previously rendered video: **identical**.  Both test suites (13 tests) pass.  The corrupted copy is not kept in the package.

## Deliverable manifest  (single-location items are the risk: nothing below is in git)
| artifact | location | other copy? |
|---|---|---|
| render / data / geometry / pick / reel / dump code (`code/*.py`, 10 files) | `C:/Users/Admin/qland/hold_r8video/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-videos/code/` | **SINGLE LOCATION (qland hold, dev box)** -- needs the integrating agent to land it in the repo |
| `INVENTORY.md`, `HANDOFF.md`, `VIDEO_MANIFEST.md`, `raw/PICKS_*.json`, `raw/NAV_CONSTANCY_*.json` | same package dir | **SINGLE LOCATION** |
| 9 videos listed above (~124 MB) | `E:/refcv8_eval_kit/videos/` (dev box, E: drive) | **SINGLE LOCATION** (binaries, not for git; copy off-box if they must be kept) |
| head dump of the 3 picks (`heads_ep013/103/114.npz`, 18-19 MB each + `keys.json`) | `E:/refcv8_eval_kit/videos/heads_cpu_picks/` | **SINGLE LOCATION**; regenerable (`dump_heads_r8.py`, 10 s per window on CPU) |
| 3D agent join cache (41 MB) | `E:/refcv8_eval_kit/videos/_cache/eval139_agents3d.pkl` | regenerable from the B1 join (`extract_eval_agents_r8.py`) |
| check stills | `E:/refcv8_eval_kit/videos/_stills/` | scratch |
| smoke / scratch dirs (`heads_smoke`, `heads_smoke2`, `heads_cpu_ep114`, `_cache/*`) | `E:/refcv8_eval_kit/videos/` | scratch (`rm -rf` was denied for the glob; left in place) |
Staged in git: **NOTHING** (no git writes were allowed).  Integration needed: land `code/` + the three docs into `FlyWheels/TanitAD_EvalFlyWheel/incoming/` of the repo (the package dir mirrors that path) and bank the videos by path in the registry / leaderboard note.
