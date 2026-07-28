# YouTube → GeoCalib → IDM: the block is lifted, the pipeline works end-to-end, and the geometry assumption is confirmed wrong

**MEASURED 2026-07-28**, pod3. PI instruction: *"try to download a few youtube videos to run the idm
model."* Done end-to-end. Evidence class **MEASURED (ours)** throughout.

## 1. D-B status — the block has lifted

**Zero bot-block messages** across both of today's runs (`grep -ci "sign in to confirm|not a bot|
HTTP Error 429"` = 0). The retry was made from **pod3 — the same egress that was blocked on
2026-07-26 16:11 UTC** — after a ~37 h cooldown. **Not an egress rotation**; switching hosts to dodge
a block stays out of bounds regardless of convenience, and idle `tanitad-eval` was deliberately not
used for that reason.

Harvest: **20 clips from 3 accepted videos**, 250 frames/clip. Rejects: `duration` 25,
`not_cc_kept` 27, `dl_fail` 0, `decode_fail` 0.

## 2. ⭐ GeoCalib confirms the fixed-HFOV assumption is wrong — independently

`geometry: geocalib_per_video`, `geocalib_enabled: true`, fallback 100.0°, confidence
`{low: 1, medium: 2}`. The two logged videos resolved to **hfov 53°** and **hfov 58°**.

⇒ **Against a 100° fixed fallback, both are off by ~2×.** This independently reproduces the standing
D-B finding that the fixed 100° HFOV was wrong for 11 of 12 real clips, on a fresh sample — and it is
exactly why an earlier `--no-geocalib` probe was declared **unquotable** rather than banked.

## 3. Privacy — verified, not assumed

Manifest records: *faces+plates+bodies Haar-blurred at full-res **before** the 256 downscale; no raw
video or full-res frames persisted; clip frames transient.*
**Verified by probe, not by trusting the manifest:** `find` for `*.mp4 / *.webm / *.mkv / *.part`
across the work dir returns **nothing**. Per-video blur counts were non-zero
(e.g. `f/p/b = 13/58/21`), so the anonymiser actually fired rather than silently no-opping.
*(This check exists because C53's kill left a raw 137 MB video on disk — the deferred cleanup never
ran. Here the pipeline exited normally and cleaned up.)*

## 4. IDM pseudo-labels — 20 clips, 2,240 windows

`ckpt_md5 b5f07d9e3dd2ca643949bc86832e6585`, `ckpt_step 29999`, `state_dim 2048`,
labeler `v1 + IDMHead{parity rigA[:60]+rigB[:60]+comma[:40]}`, built on **16,063 parity windows**.

**Channel honesty carried by the artifact itself:**
| status | channels |
|---|---|
| primary | `speed`, `long_traj` |
| caveated | `yaw_rate` |
| **dropped** | `long_accel`, `steer` |

That matches the standing IDM record — the seed-0 steer file failed its own ship bar and `long_accel`
remains negative — so the pipeline is **not** using channels the program has already refused.

**Speed sanity (GT-free):**
`n_windows 2240 · mean 14.212 m/s (≈51 km/h) · std 5.815 · min −0.19 · max 29.443 ·
frac_in_plausible_0_45_mps = **1.0**`

100 % of windows sit in the road-plausible band and the distribution has **not collapsed to a
constant** — the two failure modes this check exists to catch.

## 5. 🔴 What this does NOT establish

- ⛔ **There is no ground truth here.** The artifact says so itself: *"GT-free check … a real
  speedometer-overlay clip is the only direct GT."* These are **plausible, unvalidated** pseudo-labels.
  **No accuracy claim is made and none may be quoted.**
- ⚠️ **`speed_min = −0.19 m/s` is physically impossible** for forward speed. Small, and consistent
  with a stopped-vehicle window plus noise — but recorded rather than rounded away, because "100 % in
  band" and "contains a negative speed" are both true and only one of them flatters.
- ⚠️ **Licensing:** `license_distribution = {None: 27, CC-BY: 1}` with `not_cc_kept: 27`. The harvest
  ran with `allow_noncc` (its default), so the accepted videos are almost certainly **not** CC.
  Pointers-and-latents rather than bytes mitigates this, but it is a **posture for the PI to confirm**,
  not one to adopt silently.
- **n = 3 videos.** Nothing here supports a scaling claim; it demonstrates the pipeline runs and the
  egress is open.

## 6. Artifacts

`pod3:/workspace/tmp/yt_geo/` — `clips/` (20 × `clip_*.pt`, 2.8 GB), `pointers.jsonl` (20),
`latents/`, `results/pseudo_labels.json`, `manifest.json`.
De-stranded here: `harvest_manifest.json`, `pseudo_labels_summary.json`.
**Nothing pushed to HF** — publishing beyond what is already decided is not authorized.
