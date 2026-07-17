# D-016 R1 pinhole rectify — the owned real-urban unblock (measured)

**Date:** 2026-07-17 · **Agent:** Data Engineering (Tuesday) · **Branch:** `agent/data-engineering-20260717`
**Hardware:** local CPU (RTX 4060 box), wall-clock ~2 min total, **cost $0** (no download, no GPU).
**Readiness:** **validated** (standalone package, 9/9 tests, measured on grounded real intrinsics + real
comma bytes). Gap to *production*: fold into `stack/tanitad/data/calib.py` + real-bytes PandaSet decode on
the HF mirror (pod/local, `observed_frac` gate) — orchestrator integration + one real sequence.

## 0. Session context

Continues the 2026-07-15 run, which shipped the PandaSet loader **blocked** on a measured geometry wall and
named the fix as the #1 next item: *"D-016 R1 pad-crop + undistort — the owned real-urban BLOCKER; unblocks
PandaSet + ZOD."* This run builds and validates that primitive. Pod OFF-LIMITS (3-arm bake-off training:
flagship pod2 / REF-A done / REF-B pod1) — all work local/$0, per resource protocol.

Monday (tools-devenv) produced no new Research note this week (folder empty at read time) — no loader-affecting
tooling delta to consume.

## 1. The blocker, precisely (recap + grounded numbers)

D-016 canonicalizes every camera to a shared effective focal `F_REF = 266 px` (comma2k19's scale) so the
action→pixel geometry is identical across corpora — the prerequisite for any metric probe or mixed-corpus
world model. The current mechanism (`calib.focal_crop_resize`) crops a **centered square** of side
`c = fx·size/F_REF` then resizes. On a 16:9 frame the square is bounded by the **height**, so:

- **PandaSet front** (real calib, arXiv 2112.12610: fx=1970.01 @ 1920×1080, k1=−0.589): ideal crop
  `1896 px > 1080` → clamps to 1080 → **f_eff ≈ 467 px (~1.75× zoom)**, a real cross-corpus scale mismatch,
  **and** the pinhole crop ignores the barrel distortion entirely.
- **General rule (0715):** any pinhole with `fx > 1122 px` on a 1080-tall frame is not square-croppable to 266.
  Udacity (narrow FOV) hits the same wall. → the blocker gates the **entire owned real-urban tier**
  (OWN_DATASET_PLAN §7 #1–#4), not just PandaSet.

## 2. The fix — rectify-to-canvas (`calib_r1.pinhole_rectify`)

Mirror the *existing* fisheye rectify (`calib.ftheta_undistort`, built on `grid_sample`) for the **pinhole +
Brown-Conrady** case: build an ideal pinhole canvas of focal `F_REF` at the output size, forward-map each
ideal ray through the distortion model onto the native (distorted) sensor, and `grid_sample` it back.

- **`f_eff == 266` holds by construction** — the output canvas is *defined* at F_REF (output pixel at `d` px
  from center ≡ ray `atan(d/F_REF)`); no crop-clamp can corrupt it.
- **Barrel distortion removed** — radial (k1,k2,k3) + tangential (p1,p2), OpenCV order.
- **Out-of-frame periphery → explicit measured mask** (`last_observed_frac`, `last_mask`), not a silent zoom.
  This is exactly H17's masked-unobserved periphery (`Architecture & Inference/…/UNIFIED_FOV_FOVEATED_PATCHING.md`)
  and a free H15 imagination target.
- **One primitive, both halves of the request:** with zero distortion coeffs it degrades to a pure pad-crop
  ("letterbox"), covering the "pad-crop" and the "undistort" the 0715 INTAKE asked for.

Package: `Implementation/incoming/2026-07-17-d016-r1-pinhole-rectify/` (`calib_r1.py`, `tests/`,
`report_r1_geometry.py`, `INTAKE.md`). No `stack/` file touched; no new deps (torch only).

## 3. Measured results (`report_r1_geometry.py`, CPU ~1 s)

| Camera | naive square-crop f_eff | **rectify f_eff** | **observed_frac** | k1 edge-displacement corrected |
|---|---|---|---|---|
| comma2k19 (F_REF reference, ~pinhole) | 266.54 (drop-in) | **266.0** | **0.9961** | 0 (no distortion) |
| **PandaSet front** (fx=1970, k1=−0.589) | **466.97 — BLOCKED** | **266.0 exact** | **0.6233** | **109.07 px** |
| Udacity-like (fx=1590 @ 640×480) | 848.0 — blocked | 266.0 | 0.1306 | 0 |

**Reading the numbers:**
- **PandaSet is UNBLOCKED**: 467 → 266.0 exact, at a *measured, honest* cost of **37.7 % masked periphery**
  — its native VFOV is 30.7° vs the canonical 51.4°, so the vertical band (predominantly **sky / near-hood**,
  the low-information region) is genuinely uncaptured; the **horizontal road band is fully retained**. 109 px
  of barrel distortion is corrected.
- **comma2k19 regression clean**: the reference corpus is untouched (266.0, 99.6 % observed).
- **Falsifier surfaced (Udacity):** at f_eff=266 it is **87 % mask** — a narrow-FOV camera is *geometrically*
  canonicalizable but almost entirely unobserved. → **new ingest rule: gate every source on `observed_frac`**
  (proposed floor ≥ ~0.5); below it, either drop the source or wait for the H17 unified-canvas variant that
  legitimately pads narrow cameras. This is a concrete, measured guardrail the plan lacked.

**Correctness (falsifiers passed):**
- Brown-Conrady forward map ↔ an **independent iterative inverse** round-trips to **< 1e-4** over a ±0.4-rad
  ray grid (the model is self-consistent, not just "nonzero").
- End-to-end **checkerboard recovery**: warp an ideal checkerboard onto a k1=−0.589 native sensor, then
  rectify → correlation **> 0.9** with the original, and **> the undistort-skipped baseline** (undistortion
  demonstrably matters, not decorative).
- **Episode contract (G-D2):** rectified frames stack to a drop-in `[T,9,256,256]` u8 episode passing
  `assert_contract(channels=9)`.

## 4. Secondary measured experiment — A8 on real comma-val bytes (+ a harness pitfall)

Ran the consequence-dominance harness (`stats.consequence_dominance_stats`) on **12 real comma2k19-val
episodes (3,600 frames)** from the local epcache, current-frame slice (ch 6:9 of the 9-ch stack):

| threshold | mean | median | p10 | p90 |
|---|---|---|---|---|
| A8@0.05 | **0.0596** | 0.0654 | 0.0361 | 0.0744 |
| A8@0.10 | **0.0240** | 0.0243 | 0.0102 | 0.0358 |

Consistent with the 2026-07-07 baseline (~0.053@0.05 / 0.012@0.10) → the low-consequence highway regime
(justifying change-weighted loss, H3/A8) reproduces on the held-out val set.

**Pitfall found (worth a harness note):** `frame_change_fraction` assumes **float [0,1]** frames, but the
epcache stores **uint8 [0,255]**. Feeding uint8 directly gives a meaningless ~0.74 (threshold 0.05 on a 0–255
scale ≈ "any change", and uint8 subtraction wraps). The window datasets convert via `to_float_frames` before
scoring; a direct `stats` caller must do the same. → BACKLOG item to make `stats` uint8-safe (auto-convert or
assert dtype) so no future run mis-measures.

## 5. What this changes for TanitAD

- **Owned-dataset thesis advances (GOAL G1):** the geometry blocker on the owned real-urban tier is
  **resolved for the pinhole family** (PandaSet, Udacity, comma) and **mapped for the fisheye family** (ZOD's
  Kannala-Brandt ≡ the existing f-theta θ-poly path). Every owned real-urban source now has a rectify path →
  the plan's "one owned dataset, shippable as real episodes" (§0) is geometrically unblocked.
- **Top program risk (single-camera driving-capability gap):** the strongest data-side lever on D1 is more
  *real urban* action-grounded data at the *consistent* canonical scale. PandaSet (SF/El-Camino real urban,
  CC-BY, commercial-OK) was scale-blocked; it is now ingestable at f_eff=266, adding real-urban diversity
  comma2k19 (CA-280 highway) lacks — directly feeding the D1 denominator.
- **Honesty upgrade (P8):** the old path silently *zoomed* mis-scaled corpora into the mix (worst kind of
  corruption — invisible). The rectify path makes the cost **explicit and measured** (`observed_frac`), and
  turns an unavoidable FOV shortfall into a labelled H15/H17 target rather than a lie.

## 6. Falsifier ledger (this run)

| Claim | Falsifier | Verdict |
|---|---|---|
| Rectify lands the canonical scale | achieved f_eff ≠ 266 | **PASS** (266.0 exact, 3 cameras) |
| Reference corpus untouched | comma observed_frac drops / scale shifts | **PASS** (266.0, 0.9961) |
| Undistort is correct | fwd↔inverse round-trip, checkerboard recovery | **PASS** (<1e-4; corr>0.9 & > baseline) |
| Output is contract-drop-in | `assert_contract` fails | **PASS** (9-ch, u8, finite) |
| Every source is worth canonicalizing | observed_frac too low | **FALSIFIED for Udacity** (0.13) → observed_frac gate added |

## 7. Provenance

Code: `Implementation/incoming/2026-07-17-d016-r1-pinhole-rectify/` (this run). Blocker origin:
`Research/2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md`, `Implementation/incoming/2026-07-15-pandaset-loader/`.
Contract/geometry: `stack/tanitad/data/{calib,_contract,comma2k19,stats}.py`. Plan: `OWN_DATASET_PLAN.md`
§1/§5.4/§7. Real intrinsics: arXiv 2112.12610 (PandaSet). H17: `Architecture & Inference/Research/
UNIFIED_FOV_FOVEATED_PATCHING.md`. Decisions: D-016, D-017. Real A8 bytes: local comma2k19-val epcache
`61c46fca8f7f` (12 eps / 3,600 frames).
