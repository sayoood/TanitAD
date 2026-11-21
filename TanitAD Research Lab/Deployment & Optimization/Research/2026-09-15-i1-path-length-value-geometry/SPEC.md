# SPEC / PRE-REGISTRATION — `E-DEP-VGEO-3`: does the frozen trunk carry goal geometry that PIXELS cannot fake by smoothness?

**Date** 2026-09-15 (Europe/Berlin) · **Author** Research Lab (daily pass, LAB-RUN-013) ·
**Branch** `agent/arch-inf-20260803` · **Serves** INJECTED row **I-1** (top of lane) via proposed row **DP13-1**.
**Status:** written **BEFORE any statistic below was computed.** Known facts at writing time: the token
bank's shape (inherited from `2026-09-13-i1-value-geometry-screen/SPEC.md`), and the `bev_gt/*.bevgt.npz`
schema (keys `raw_frame`, `ego_v_ms`, `label_valid`; `ego_v_ms` in m/s on the SAME raw v2ep frame index as the
bank's `raw_frame` per its `meta_json.time_grid`) — read from ONE file's metadata; no speed statistic was looked at.

## 0. Why

`E-DEP-VGEO-2` (2026-09-13) ended `HEAD-CLEARS-BAR-BUT-PIXELS-TIE`: a learned metric on frozen tokens ranks
time-to-goal at ρ 0.5637, and a learned metric on 1,440-d pixels ties it (+0.0146 [−0.0269, +0.0534]).
Its own risk (a): *time*-to-goal rewards visual smoothness, which pixels have for free. DP13-1 committed the
fix: **rank by ego path-length-to-goal** (stationary spans stop counting) and **add cross-clip distractor
goals at matched pixel distance.** Blocker was the pose join and a live MM arm; at 2026-09-15 `tasklist` shows
**no python.exe** and the 4060 at 9 % / 1.2 GB — the dev box is free. The join is `ego_v_ms` from the A&I
LiDAR-BEV GT artifacts (no new extraction needed).

## 1. Data

* **Bank:** `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/` (refcv5-v2 s32 tokens [27,664, 704, 8, 20], pix64 floor, 139 clips) — identical to VGEO-1/2.
* **Speed:** `C:/Users/Admin/tanitad-caches/bevhead-20260913/bev_gt/<clip_sha12>.bevgt.npz` → `ego_v_ms[raw_frame]`. **Clips without a bev_gt file are EXCLUDED and counted** (a file-existence probe, not a statistic, found 134/139 present). Frames with `label_valid == False` get speed by linear interpolation over valid frames of the same clip; clips whose `meta.stats.deskew_source` is not `egomotion_alpamayo` are excluded.
* **Path length** `L(s,g) = Σ_{k=s}^{g-1} v_k · 0.1 s` over raw frames (cumulative sum on the full 10 Hz series, then read at subsampled rows).
* **Rows:** every 4th row per clip (as VGEO-1/2), CPU only, below-normal priority.
* **Split:** the SAME seeded clip permutation rule as VGEO-2 (seed 20260913) applied to the clips that survive the speed join ⇒ fit ≈ 70 % / scored ≈ 30 %, clip-disjoint, asserted.

## 2. Statistics (all on SCORED clips; learned metrics fit on FIT clips only)

* **Representations:** `latent_704` (mean pool), `latent_7040` (2×5 pool), `pixel_1440` — each as `raw` L2 and `diag` (NNLS diagonal metric, VGEO-2's fitter, **target = log L** over fit pairs Δ ∈ [4, 60] rows with L ≥ 0.5 m).
* ⭐ **P1 — within-Δ path ordering (PRIMARY).** For each scored clip and each fixed gap Δ ∈ {12, 16, …, 60} rows, take all pairs with exactly that Δ; Spearman ρ between metric distance and L. Pairs at one Δ differ ONLY in how far the ego travelled, so frame-gap smoothness gives no advantage. ρ undefined (NaN) when fewer than 8 pairs or when `std(L)/mean(L) < 0.05` at that Δ. Per clip = nanmean over Δ; clip excluded from P1 if all NaN.
* **P2 — cross-clip distractor accuracy (SECONDARY).** Per scored clip, 200 seeded (s, g) pairs with Δ ∈ [12, 60]; distractor d = the row from a DIFFERENT scored clip whose raw pixel_1440 distance to s is nearest to ‖x_s − x_g‖ (from a seeded pool of 3,000 rows). Accuracy = mean 1[dist(s,g) < dist(s,d)], ties 0.5.
* **P3 — path-length ordering, all Δ ∈ [4, 60]** (VGEO-2's statistic with target L instead of Δ) — reported for continuity, decides nothing.
* **Estimator:** clip-cluster bootstrap over scored clips, 2,000 resamples, 95 % percentile; paired differences on the same draws. Answers *another draw of clips* only — one checkpoint, blind to training variance.

## 3. Controls (each must read its known value, or the run is VOID)

| control | construction | must read |
|---|---|---|
| C-const | constant distance | P1 = 0 exactly; P2 = 0.5 exactly |
| C-vshift | L computed from the clip's speed series **circularly shifted** by a seeded offset ∈ [T/3, 2T/3] (keeps the speed distribution, breaks alignment) | P1 on `latent_7040|diag`: \|point\| ≤ 0.03 |
| C-mut | within-clip row permutation of features (deliberate regression) | P1 \|point\| ≤ 0.03 |
| P2 raw-pixel arm | matched by construction | P2 ∈ [0.45, 0.55] |

## 4. ⛔ Committed verdicts (primary contrast = `latent_7040|diag − pixel_1440|diag` on P1; 7040|diag named because it was VGEO-2's best arm — a selection declared here)

| verdict | rule |
|---|---|
| **TRUNK-CARRIES-PATH-GEOMETRY** | paired P1 CI lower > 0 **and** latent P1 CI lower > 0 **and** controls pass ⇒ I-1 proceeds on the frozen trunk (DP13-1 branch 1) |
| **TIE-AGAIN** | paired P1 CI contains 0 ⇒ DP13-1 branch 2: **a distance-shaped value should not read the frozen trunk**; next lever is an unfrozen or separately-trained value encoder (**MM decision**) |
| **PIXELS-WIN** | paired P1 CI upper < 0 ⇒ as TIE-AGAIN, stronger |
| **SCENE-IDENTITY-ONLY** (modifier) | P2 paired latent−pixel CI lower > 0 while P1 is TIE/PIXELS-WIN ⇒ the trunk separates *places*, not *distances*; **no geometry credit** |
| **UNDERPOWERED** | < 20 scored clips with defined P1 ⇒ no verdict, say so |

The raw arms and `latent_704` contrasts are reported and **do not** override the primary.
