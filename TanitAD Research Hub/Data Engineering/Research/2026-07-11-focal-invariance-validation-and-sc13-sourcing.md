# Focal-invariance validation of D-016 + SC-13 real-data sourcing

**Agent:** Data Engineering (Tuesday) · **Date:** 2026-07-11 · **Run branch:** `agent/data-engineering-20260711`
**Quality:** full (G-A…G-H + G-D1/G-D2). Loop: 1 iteration, 2 web searches, ~1.2 h wall-clock.

## 0. Context / why this run

The pod trainer (`p0-sB01-realmix`, PID 200748) is **still running the 30k record run** → the pod is
off-limits (training priority), so backlog **P0.0 (R1 top-up on pod) is blocked**. Backlog **P0.1
(WorldModel-Synthetic pose probe) is already DONE** on the unmerged branch
`agent/data-engineering-20260710` (`96d85eb`, verdict NO-POSE → video-only loader) — not yet swept into
main by the orchestrator; **not re-done here** to avoid duplication. So this run took the two locally
runnable, high-value items: **validate the D-016 focal canonicalization** (agent-file focus item 3, the
Y1 enabler for the banked Y-pilot-50 dashcam directive) and the **mandatory joint SCENARIO duty**.

## 1. Finding: `calib.py` focal canonicalization was implemented but never validated

`stack/tanitad/data/calib.py` (`focal_crop_resize`, D-016) is wired into **all three** camera loaders
(comma2k19, cosmos_drive, physicalai) and crops each camera to a shared effective focal
`F_REF = 266 px @ 256`. It is unit-tested for **arithmetic** (`test_calib.py`) but there was **no
evidence** it makes the *trained encoder's representation* invariant to focal length on real frames —
the whole point of the VLM3 principle for H7. If it silently fails, the D-010 real+sim mix feeds the
world model inconsistent action→pixel-motion geometry across corpora, and the I7 fingerprint's
`f_eff_px = 266` is only nominal. This run closes that gap.

## 2. Measured experiment (G-H) — controlled correct-vs-wrong intrinsics

**Design (single scene → no cross-corpus content confound).** A real 120° clip frame at focal `f0` is
re-imaged as the SAME scene at a longer focal `f1 = z·f0` (a narrower camera) via central-crop-by-1/z +
resize (`virtual_focal_zoom`, pinhole-exact). Both captures are canonicalized to `F_REF` with the
**correct** intrinsics (`f0`, `z·f0`) vs the **wrong** ones (both = `f0`), encoded by the trained
`WorldModel`, and the relative latent drift `‖z_a−z_b‖/‖z_a‖` is compared. The only difference between
the policies is whether the perturbed camera's TRUE focal is used. D-016 works iff
`drift_correct ≪ drift_wrong`. Base corpus = **Cosmos-Drive-Dreams** front_wide_120fov (local extracted,
704×1280) — it has the FOV headroom comma2k19 lacks (comma sits right at `F_REF`, no headroom).

**Setup.** Local RTX 4060, `ckpt_full.pt` (partial ~step-6500 ckpt — the encoder we deploy; P8 it is
still moving), 12 clips → 48 scenes, **7.9 s, $0**. Base f_eff measured **265.7 ≈ F_REF 266** ✓.

| zoom z (focal change) | drift correct | drift wrong | reduction | cos correct | cos wrong |
|---|---|---|---|---|---|
| 1.25 (+25 %) | 0.041 | 0.402 | **9.8×** | 0.999 | 0.919 |
| 1.50 (+50 %) | 0.049 | 0.711 | **14.5×** | 0.999 | 0.753 |
| 2.00 (+100 %) | 0.063 | 0.934 | **14.9×** | 0.997 | 0.600 |

**Verdict — D-016 VALIDATED.** Falsifier (`drift_correct ≥ drift_wrong`) fails at every zoom. With
correct per-camera intrinsics the same scene's encoding stays near-identical (cosine ≥ 0.997) across a
25–100 % focal change; ignoring intrinsics drops cosine to **0.60** and pushes relative drift to ~0.93
(≈ an unrelated state). Note **both** policies produce a *nominally* 266-px image — the wrong one just
carries the wrong physical scale, which is exactly the cross-corpus nuisance canonicalization removes.
The small residual under correct intrinsics (0.04→0.06, rising gently with z) is double-resampling +
minor content loss, not a geometry error.

**Actionable implications.**
1. **Getting intrinsics wrong costs ~10–15× encoder drift** → landing **per-clip intrinsics** (the
   pending DataEng task `calib.py` flags — PhysicalAI `calibration/`; per-video focal for YouTube) is a
   *high-value*, not cosmetic, task. Promoted on the backlog.
2. **Y-pilot-50 (dashcam) Y1 = focal self-calibration is de-risked**: if per-video focal recovery lands
   within ~±25 % it already buys cosine ≥ 0.92; within ~±10 % → ≥ 0.99. Sets the Y1 acceptance bar.
3. **Hardening shipped:** `assert_effective_focal()` — a fail-loud data-card guard. `focal_crop_resize`
   silently CLAMPS the crop to the frame, so a camera *narrower* than the reference (no FOV headroom)
   lands ABOVE `F_REF` with an `f_eff` inconsistent with the `CORPUS_META`/I7 `f_eff_px = 266` claim.
   The guard raises at ingest instead of corrupting geometry in training (tested: passes comma/120°,
   raises for a 20° telephoto). Proposed for the loaders' data-card path.

## 3. Joint SCENARIO duty (D-020 §5) — SC-13 sourced + measured

SC-13 (stationary-object / slow-lead response, from the Avride NHTSA ODI) claimed "comma2k19 has
abundant real lead-vehicle following." **Measured on local comma2k19 Chunk_1** (all 188 segments, CAN
speed mine for creep-to-stop-then-resume, v<2→>5 m/s): **13 stop-go events across 12/188 segments (6 %)**,
all low-speed (pre-stop speed mean 3.9 / p90 4.7 m/s; **0 events braking-to-stop from >8 m/s**). Honest
read (P8): comma gives plenty of *car-following* but the **hard stopped-lead subset is Chunk_1-sparse** —
the matched segments the SC-13 falsifier needs are not there at Chunk_1 scale. → SC-13 advanced
**catalogued → data-sourced (partial)**; next step is the **full 10-chunk mine** (~130 low-speed events
extrapolated) or **CARLA/synthetic hard-braking recipes** (blocked_route family; WorldModel-Synthetic
cut-in 32.9 %). Row updated in `SCENARIO_DATABASE.md` (also in STATE HANDOFF — shared-file race).

## 4. Literature sweep (since 2026-07-09; 2 searches — external support, no status change, P8)

- **VLM3** confirms the principle our `F_REF` encodes: rescale each image so focal = a fixed target
  (VLM3 uses 1000 px at their input; our 266 @256 is the same idea at model resolution). Direct backing
  for D-016.  — https://www.emergentmind.com/topics/vlm3
- **DeltaCam** (2605.25266, differential intrinsic camera modeling) & **Infinite-Homography conditioning**
  (2512.17040) — intrinsics as an explicit, device-invariant conditioning signal for video generation;
  adjacent to our per-clip-intrinsics plan (H2 side-view re-introduction of the sacrificed periphery).
- H7 latent-action / inverse-dynamics surge continues: **Latent-WAM** (2603.24581), **IDOL** (2605.31476,
  inverse-dynamics-guided future prediction), **DriveWAM** (2605.28544), **DynVLA** (2603.11041) — all
  factor driving into world-modeling + inverse-dynamics action recovery, i.e. the labeled-bridge our
  comma2k19 IDM serves. "World model + IDM from unlabeled video" is now table stakes → the moat stays
  hierarchy + efficiency (CNCE) + imagination + self-monitoring. No hypothesis upgrade.

## 5. Artifacts

- Intake pkg `Implementation/incoming/2026-07-11-focal-invariance-validation/`: `focal_invariance.py`,
  `run_focal_invariance.py`, `tests/test_focal_invariance.py` (8✓), `focal_invariance_result.json`,
  `INTAKE.md`. Proposed target: `assert_effective_focal`/`achieved_focal` → `stack/tanitad/data/calib.py`.
- Measured numbers: focal-invariance table (§2); SC-13 mine (§3). Hardware RTX 4060, $0.

## 6. Ressources inbox note

`Ressources/` files (`2606.27014v1.pdf`, `2507.00028v2.pdf`, `End2End Driving based on DinoVx…md`,
`AD_TRANSFER_RESEARCH.md`, WP.29 GTR PDF, `Deep Think Analysis/`) are untracked and carry checkout-reset
mtimes (not reliable "new since last run" signals); `2606.27014` is already the arch spectral-sizing
anchor. None re-analyzed deeply this run (time-boxed to the experiment); flagged for the next run's §2c
inbox pass with a git-independent mtime source.
