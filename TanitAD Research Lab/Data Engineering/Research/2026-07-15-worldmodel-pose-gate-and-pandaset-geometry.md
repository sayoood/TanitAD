# Data Engineering — 2026-07-15 (Tuesday agent)

**Focus.** Advance the OWN_DATASET_PLAN (2026-07-13) toward a license-clean "one dataset for all":
(1) close the WorldModel-Synthetic-Scenarios **pose-field gate** (BACKLOG P0.1) on real bytes;
(2) ship the **PandaSet loader** (plan §7 ingest #2) as an intake package with an episode-contract test;
(3) HF landscape sweep (D-012 standing duty) + literature (D-013/D-028).

**Compute.** Local dev box only (pod busy — `refb-speed-30k` training, off-limits per protocol). HF tree API
+ `truststore` TLS. RTX-4060 not needed (loaders unit-tested; geometry is analytic). Wall-clock ≈ session.
Cost **$0**. All numbers below are measured this run.

---

## 1. MEASURED — WorldModel-Synthetic-Scenarios is POSE-LESS (BACKLOG P0.1 gate → CLOSED)

The 2026-07-09 note flagged the "near-zero cosmos-mirror" assumption *at risk* (card lists RGB+captions,
no pose). **Confirmed on real bytes this run.** HF tree walk of
`nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios` + a fetch of one real clip:

- **Layout:** `<family>/<clip_id>/{description, video}`. `video/` holds **7 camera mp4s**
  (`front_wide`, `front_tele`, `left_fisheye`, `right_fisheye`, `rear_fisheye`, `rear_left`, `rear_right`),
  24 fps, ~462 frames. `description/<cam>.json` is the only metadata.
- **The "description" is a language caption, NOT a pose.** Real `front_wide.json` (verbatim keys):
  `{"framerate":24.0, "nb_frames":462, "t2w_windows":[{"start_frame","end_frame","qwen2p5_7b_caption": "<VLM text>"}], "metadata":{"weather","time_of_day","surface_type","region"}}`.
  Example caption: *"At night, the ego vehicle stops at a stop sign … waits for a police vehicle to pass …"*
- **No `vehicle_pose`, no CAN, no trajectory, no action anywhere in a clip.**
- Families present: `emergency, lanechange, nudging, pedestrian, weather_degradation` (matches the card).

**Verdict (falsifier resolved).** The corpus **cannot back an action-conditioned loader now** — the episode
contract requires finite `actions[T,2]`/`poses[T,4]` (`_contract.assert_contract`), which pose-less video
cannot supply without fabrication. The three real options, in order:
1. **Video-only representation** (no action head) — a *separate* contract, out of the D-010 action-mix.
2. **IDM / H7 pseudo-labelling** — infer actions with a trained inverse-dynamics head (the labelled bridge =
   comma2k19/ZOD real CAN). **This is the unlock, and the literature is now dense** (§4): jointly train an
   IDM + world-model on a frozen encoder over unlabelled video (our frozen-DINO REF-A lineage fits exactly).
   Gate: a trained IDM head → Phase-1, not this week.
3. **Semantic-label mining (usable TODAY, $0):** the `qwen2p5_7b_caption` + `{weather, time_of_day,
   surface_type, region}` per clip are a **strategic/behaviour-label source** — directly serving BACKLOG P1
   item **2d** (semantic-label survey: comma2k19 is highway-`follow`-starved) and **SCENARIO_DATABASE** data
   sourcing (emergency→SC-06, pedestrian→SC-02, weather_degradation→SC-05). 264k clips of captioned
   safety-critical long-tail is a large, ungated (OpenMDW-1.1) semantic index even before actions exist.

**Actionable:** retire "cosmos-mirror loader" for WorldModel-Synth; re-file it as (a) a Phase-1 IDM target and
(b) a **now-usable semantic index** (new BACKLOG P0 item). Do not download the 8.3 TB pixels until (a) or (b)
is scheduled — the captions/metadata are tiny and are what we need first.

## 2. MEASURED — PandaSet loader shipped + a grounded D-016 GEOMETRY BLOCKER (plan §7 #2)

Built `pandaset.py` (intake pkg `2026-07-15-pandaset-loader/`, **16 tests green** standalone) — a CC-BY-4.0
real-urban adapter reusing the Cosmos geometry: `poses.json`→signals via a **motion-heading** 4×4 (offset-free,
avoids the unknown camera→vehicle extrinsic) → the tested `cosmos_drive.poses_to_signals`; D-015 9-ch stack;
`CORPUS_META` byte-identical to comma2k19 (I7 → admissible in the mix); SEQUENCE-level split (I3). Schema
**grounded from the pandaset-devkit source** (not guessed): `poses.json` = list of
`{position:{x,y,z}, heading:{w,x,y,z}}` world-frame; `intrinsics.json` = `{fx,fy,cx,cy}`.

**But the loader is BLOCKED-by-design on a real geometry defect it now fails loud on.** Grounded on the
**real** front-camera calibration (arXiv 2112.12610 / devkit: `fx=1970.01`, `1920×1080`, distortion
`k1=−0.589`):

| Camera | ideal square crop | frame min-dim | used crop | **achieved f_eff** | drop-in? |
|---|---|---|---|---|---|
| PandaSet front (real fx=1970) | 1896 px | **1080** | 1080 (clamped) | **467 px** | **NO** |
| a wide pinhole (fx=1000) | 962 px | 1080 | 962 | 266.1 px | yes |

- **Root cause (vertical-FOV bound):** D-016 canonicalizes by a *centered square* crop of side `fx·size/F_REF`.
  On a 16:9 frame that square is bounded by the **1080 height**. PandaSet's front camera needs a 1896-px square
  but the frame is 1080 tall → clamps to 1080 → lands **f_eff≈467 px ≈ 1.75× the canonical 266** — a real
  **cross-corpus action→pixel SCALE mismatch** (PandaSet objects render 1.75× larger than comma's). Closed
  form: on a 1080-tall frame, **any front camera with `fx > 1122 px` is height-bound** and cannot square-crop
  to 266; PandaSet's 1970 fails it by a wide margin.
- **Second defect (ignored distortion):** the front camera has real barrel distortion (`k1=−0.589`); the
  pinhole `focal_crop_resize` ignores it — the same class of silent error as the pre-D-016 physicalai bug
  (fisheye fed through the pinhole path) and the two-rig `cy` bug now being fixed in `stack/` (validate_data.py
  + test_physicalai_rig.py, in flight).
- **The loader refuses to emit silently mis-scaled frames** (`_canonicalize(..., strict=True)` raises
  `GeometryError` naming both issues; `verify_real_clip` measures the residual under `strict=False`). So a naive
  PandaSet ingest **cannot pollute the owned mix** — exactly the review discipline that caught chunk-pairing,
  the fisheye, and the two rigs.

**Readiness (D-029): pose/signal + contract path = VALIDATED (16 tests); geometry = BLOCKED pending D-016 R1.**
The fix is the same mechanism already landing for rig-B: a **pad/letterbox-aware crop** (replicate-pad the
below-frame overflow so the square can exceed the frame height and reach f_eff=266) **+ undistort** using the
shipped `k1`. That belongs in `stack/tanitad/data/calib.py` (D-016 R1) — proposed in the INTAKE, not built here
(boundary: no `stack/` edits). PandaSet is then a one-line switch from BLOCKED to drop-in.

**Why this matters beyond PandaSet:** the height-bound rule (`fx>1122 → not square-croppable to 266 on a 16:9
frame`) is a **general OWN_DATASET ingest gate**. ZOD (plan §7 #1, fisheye), Udacity, and any narrow-FOV real
camera hit it too. The D-016 R1 pad-crop is therefore a **prerequisite for the whole owned real-urban tier**,
not a PandaSet detail — this run promotes it from "deferred R1 nicety" to "blocking dependency", with numbers.

## 3. HF landscape sweep (D-012 standing duty)

Live HF sweep (`list_datasets` lastModified + tree probes). Deltas since last run:
- **NEW: `nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic`** — a sibling to Cosmos-Drive-Dreams, card-only
  today (README + `.gitattributes`, no data payload in the tree yet). Watch for population; if CC-BY like
  Cosmos-DD it is another publicly-claimable synthetic shard. Added to DATASET_LANDSCAPE as *tracked, empty*.
- `Newsflare/newsflare-autonomous-driving-videos` (2026-05-18) — commercial **stock-video** corpus → same
  copyright-barrier class as OpenDV-YouTube (per-clip © / ToS); **not owned-redistributable.** Logged, excluded.
- `roboticslaburjc-org/CARLA_e2e_autonomous_driving` (2026-07-12) — a CARLA E2E set; CARLA self-gen is already
  our owned off-expert arm (plan §5.3). No action.
- No new ungated real-AV video corpus surfaced. The owned real-urban gap stays **ZOD-shaped** (plan §7 #1).

## 4. Literature (D-013/D-028) — IDM/latent-action is the WorldModel-Synth unlock

The pose-less verdict (§1) makes the H7 inverse-dynamics head the gate for 264k clips of long-tail. The area is
now dense and directly on-lineage:
- **Learning Latent Action World Models In The Wild** (arXiv 2601.05230) — latent-action WM on in-the-wild
  video, beyond sim/manipulation → the pattern for WorldModel-Synth's captioned pixels.
- **Factored Latent Action World Models** (2602.16229), **HiLAM** (2603.05815, ICLR-26 wkshp, already tracked)
  — factored/hierarchical latent actions.
- **LatentVLA** (driving-specific) + **FLAM** — IDM over two frames → continuous action; train policies on the
  pseudo-labels. Confirms: **frozen encoder + jointly-trained IDM+WM on unlabelled video** (our frozen-DINO
  REF-A setup) is the standard recipe. Support only, no ledger status change (P8).
- **Implication:** the comma2k19/ZOD **real-CAN** corpora are the labelled bridge that makes every pose-less
  corpus (WorldModel-Synth, YouTube dashcam) trainable — reinforcing ZOD's priority (real CAN, plan §7 #1).

---

## 5. Deliverables & gate self-check

| Gate | Status | Evidence |
|---|---|---|
| G-A (sourced claims) | ✅ | HF tree/fetch, devkit source, arXiv 2112.12610, repo paths cited inline |
| G-B (actionable rec) | ✅ | promote D-016 R1 pad-crop to blocking; WorldModel-Synth → semantic index + IDM (Phase-1) |
| G-C (KB updated) | ✅ | 3 deltas, newest-first |
| G-D (ledger) | ✅ | H7 data-availability row (no status change, P8) |
| G-E / G-D2 (loader + contract test) | ✅ | `pandaset.py` + 16 standalone tests (incl. `assert_contract` channels=9) |
| G-H (measured experiment) | ✅ | **two** measured results: WorldModel pose gate (real bytes) + PandaSet f_eff=467 (real fx) |
| G-D1 (dataset entry: license/size/actions/cost) | ✅ | DATASET_LANDSCAPE PandaSet row + WorldModel pose verdict |

**QUALITY: full.**

### Provenance
Loader: `Implementation/incoming/2026-07-15-pandaset-loader/{pandaset.py,tests/test_pandaset.py,hf_probe.py,INTAKE.md}`.
Reuses `stack/tanitad/data/{_contract,calib,comma2k19,cosmos_drive,toy_driving,mixing}.py`. Plan:
`Data Engineering/OWN_DATASET_PLAN.md` §7. Sources: PandaSet [arXiv 2112.12610](https://arxiv.org/pdf/2112.12610),
[pandaset-devkit](https://github.com/scaleapi/pandaset-devkit); IDM [2601.05230](https://arxiv.org/abs/2601.05230),
[2602.16229](https://arxiv.org/abs/2602.16229). This is an engineer's read, not legal advice.
