# INTAKE — D-016 R1 pinhole rectify-to-canvas (undistort + pad)

- **Date:** 2026-07-17
- **Author:** Data Engineering agent (Tuesday)
- **Package:** `Data Engineering/Implementation/incoming/2026-07-17-d016-r1-pinhole-rectify/`
- **Depends on / continues:** `2026-07-15-pandaset-loader/` (which shipped the fail-loud
  `GeometryError` this package resolves).

## What

A new pure-geometry primitive `pinhole_rectify` (+ `pinhole_rectify_grid`,
`brown_conrady_distort`, `PinholeIntrinsics`, `pinhole_geometry_report`) that
canonicalizes a **pinhole** camera to the shared `F_REF = 266 px` effective focal
using a `grid_sample` rectify-to-canvas — mirroring the existing fisheye
`ftheta_undistort` design in `stack/tanitad/data/calib.py`. It:

1. lands `f_eff == 266` **exactly, by construction** (an ideal pinhole canvas of
   focal `F_REF` is forward-mapped onto the native sensor), where the current
   `focal_crop_resize` centered-square crop is **height-bound** and lands the wrong
   scale on any `fx > 1122 px` @ 1080-tall frame;
2. removes **Brown-Conrady barrel distortion** (radial k1,k2,k3 + tangential p1,p2)
   — the current pinhole path ignores it entirely;
3. replaces the silent zoom-in with an **explicit, measured unobserved mask**
   (`last_observed_frac`, `last_mask`) for the periphery a narrow/short frame can't
   cover — the H17 masked-periphery philosophy, and a free H15 imagination target.

With zero distortion coeffs the same function degrades to a pure **pad-crop**
("letterbox"), so one primitive covers both halves of the D-016 R1 request the 0715
INTAKE filed.

## Why

The 2026-07-15 run proved a HARD blocker gating the **entire owned real-urban tier**
(OWN_DATASET_PLAN §7): PandaSet's real front camera (fx=1970.01 @ 1920×1080,
k1=−0.589) cannot be square-cropped to 266 — the ideal crop (1896 px) exceeds the
1080-px frame height, clamps, and lands `f_eff ≈ 467 px` (~1.75× zoom), a real
cross-corpus action→pixel scale mismatch. Udacity (narrow FOV) hits the same wall.
The PandaSet loader therefore raises `GeometryError` and stays blocked. This
primitive is the promised `stack/tanitad/data/calib.py` fix that unblocks it.

## Evidence (measured, this run — `report_r1_geometry.py`, CPU ~1 s, $0)

| Camera | naive square-crop f_eff | rectify f_eff | observed_frac | k1 edge-displacement corrected |
|---|---|---|---|---|
| comma2k19 (F_REF reference) | 266.54 (drop-in) | **266.0** | **0.9961** | 0 (no dist) |
| **PandaSet front** (fx=1970, k1=−0.589) | **466.97 (BLOCKED)** | **266.0** | **0.6233** | **109.07 px** |
| Udacity-like (fx=1590 @ 640×480) | 848.0 (blocked) | 266.0 | 0.1306 | 0 |

- **PandaSet is unblocked**: 467 → 266.0 exact, at a measured cost of **37.7 %
  masked periphery** (its native VFOV 30.7° < canonical 51.4°, so the vertical
  band — predominantly sky/near-hood — is genuinely uncaptured; the horizontal
  road band is fully retained), and **109 px of barrel distortion removed**.
- **comma2k19 regression**: the reference corpus is untouched (266.0, 99.6 % observed).
- **Falsifier surfaced**: Udacity at f_eff=266 is **87 % mask** — a narrow-FOV camera
  is *geometrically* canonicalizable but mostly unobserved; ingest decision must
  gate on `observed_frac` (proposed floor ≥ ~0.5), or defer to the H17 unified canvas.

## Tests run (standalone, 9/9 ✓)

```
cd 2026-07-17-d016-r1-pinhole-rectify && python -m pytest tests/test_calib_r1.py -q
9 passed
```

Coverage: f_eff-exact-by-construction; center-ray→principal-point; PandaSet naive
height-bound (regression anchor) vs rectify canonical; comma near-full observation;
**Brown-Conrady forward↔independent-iterative-inverse round-trip < 1e-4**; end-to-end
**checkerboard recovery** (rectify corr > 0.9 and > undistort-skipped baseline);
zeros-padding blanks the unobserved band; and **episode-contract (G-D2)**: rectified
frames stack to a drop-in `[T,9,256,256]` u8 episode that passes `assert_contract`.

## Proposed target location in `stack/`

1. Append `PinholeIntrinsics`, `brown_conrady_distort`, `pinhole_rectify_grid`,
   `pinhole_rectify`, `square_crop_feff`, `pinhole_geometry_report` to
   `stack/tanitad/data/calib.py` (no changes to existing symbols; net-new).
2. In `stack/tanitad/data/pandaset.py` (when the 0715 loader integrates), flip
   `_canonicalize` to call `pinhole_rectify(vid, PANDASET_FRONT_INTR)` and carry
   `observed_frac` into the data card; the `GeometryError` becomes a
   `observed_frac < floor` guard instead of a hard block.
3. Udacity loader (BACKLOG) uses the same path with an `observed_frac` gate.

## Coverage map (which primitive each owned source uses)

| Model | Sources | Primitive |
|---|---|---|
| Pinhole + Brown-Conrady | **PandaSet, Udacity, comma2k19** | **`pinhole_rectify`** (this) |
| f-theta / Kannala-Brandt fisheye | PhysicalAI, Cosmos-DD, **ZOD** | existing `ftheta_undistort` / `ftheta_crop_resize` |

→ the whole owned real-urban tier now has a rectify path. ZOD's Kannala-Brandt is
the f-theta θ-polynomial the existing fisheye path already models (its coefficients
land at ZOD ingest, BACKLOG).

## Risk / rollback

- **Risk (low):** grid normalization matches `calib.ftheta_undistort_grid`'s
  `u/(w-1)*2−1` / `align_corners=False` convention (sub-px, codebase-consistent).
  `observed_frac < 1` means real information loss at the periphery — that is the
  honest, intended behavior (vs the old silent zoom that corrupted scale). The
  ingest gate must consume `observed_frac`.
- **No new deps** (torch only); no `stack/` file touched by this package.
- **Rollback:** delete the appended symbols and keep `focal_crop_resize`; PandaSet
  reverts to `GeometryError`-blocked. No effect on any existing corpus (comma/Cosmos/
  PhysicalAI paths unchanged).
