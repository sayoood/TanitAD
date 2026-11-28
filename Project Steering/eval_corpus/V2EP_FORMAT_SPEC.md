# The exact format the clean-val corpus must reproduce

**MEASURED 2026-08-05** by reading a real training episode out of v1arch's own cache
(`epcache-physicalai-v2bal-4b7eeeac222d/00097de1-….v2ep.pt`, 2 569 912 B). Taken from the DATA,
not from the build code — the cache records no geometry block, so the code path is an inference and
the file is the fact.

## Per-episode `.v2ep.pt` — a plain dict

| key | shape / type | note |
|---|---|---|
| `jpeg_buf` | `(2560968,) uint8` | ALL frames concatenated as JPEG bytes |
| `jpeg_len` | `(201,) int64` | per-frame byte lengths — the index into `jpeg_buf` |
| `actions` | `(201, 2) float32` | |
| `poses` | `(201, 4) float32` | x, y, yaw, v |
| `n_stack` | `int = 3` | |
| `image_size` | `int = 256` | square |
| `episode_id` | `int = 808464441` | ⚠️ see below |
| `clip_id` | `str` | full UUID |
| `quality` | `int = 90` | JPEG quality |

`T` varies per clip (manifest `T_out` spans 189–198 across the pool); 201 here.

## ⚠️ `episode_id` IS THE PACKED-ASCII DEFECT, AT ITS SOURCE

`808464441` → big-endian bytes → `"0009"` → the first **4 hex chars of `clip_id`**
(`00097de1-…`), packed as an int32. This is the ORIGIN of the packed `eid` found in
`windows_flagship-v4.1-10k.pt`, `-v4.2-step4000.pt` and `-v16-ab-ft.pt` and repaired in
`taniteval/rollout.py` earlier today — the defect is not in the eval writer, it is **inherited from
the episode cache format**.

It is also **lossy by construction**: 4 hex chars is 16 bits, so ids collide. `eval_flagship_v4.py`
says so out loud on this very corpus — *"episode ids from 'episode_uid' [collision-free; the
as-built 16-bit ids would give only **21 distinct for 23 clips**]"*. ⇒ Any new build must carry
`clip_id` and key on it; `episode_id` must never be a join key.

## Geometry — square 256/266 f-theta crop, NOT 120°

The camera is `camera_front_wide_120fov`, an **f-theta FISHEYE**. The deployed canonical frame
CROPS it: `_decode_mp4` resolves the clip's real per-clip intrinsics and crops against the true
radial map so `f_eff == F_REF ≈ 266 px` (comma-matched) — *"instead of the old nominal-120-deg
PINHOLE focal (554 px) … retaining only ~16.4° → 1.6× over-zoomed vs comma"*. Effective HFOV
≈ **51.4°**, centred on the per-clip principal point (the two-rig fix, D-016 R1).

⇒ **v1arch consumes ~51.4°, not 120°.** The 120°-wide path is the CYLINDRICAL 256×640 frame
(`--frame-hfov 120 --projection cylindrical`) used by v5f. v1arch's launch command carries none of
those flags and its cache is 256×256. **Building the clean val at 120° would not match it.**

⚠️ **VERIFY BEFORE BUILDING 19 EPISODES.** The manifest stores no projection, so the above is an
inference from (a) the square 256×256 shape and (b) the absence of geometry flags at launch. The
decisive check is a **one-episode round trip**: rebuild a clip that IS in the pool from HF with
`ftheta_crop` at 256/266 and compare pixels to its cached `.v2ep.pt`. Contamination is irrelevant
for a pixel check.

## ⛔ WHAT THE TRAINING FORMAT DOES **NOT** CONTAIN

Images, actions, poses. That is all. Specifically absent:

* **no manoeuvre / tactical labels** — derived at train and eval time from `poses` via
  `refb_labels.classify_maneuver` (a pure function of future ego kinematics);
* **no lead-agent / obstacle tracks** — so LONGITUDINAL **distance-keeping cannot be computed from
  this cache alone**. It needs `obstacle.offline` (3D agent tracks, 97.44 % of the corpus, 10
  dynamic-agent classes) ingested alongside;
* **no route / nav / goal signal, and no map**.

⇒ Reproducing the format exactly gives LONGITUDINAL (speed, along-track), LATERAL and TACTICAL.
It does **not** give distance-keeping (needs `obstacle.offline`) and cannot give **STRATEGIC** at
all: PhysicalAI-AV has no map, lane graph, junction, roundabout, traffic-light or route signal —
the card says verbatim *"we do not include open maps data"* — and `egomotion` carries no lat/lon, so
OSM map-matching is impossible. A strategic vocabulary must come from AlpaSim or an external corpus.
