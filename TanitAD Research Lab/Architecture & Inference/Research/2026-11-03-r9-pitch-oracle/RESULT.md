# R10-A — per-frame pitch ORACLE on the BEV lift: RESULT (2026-11-03)

**Verdict (committed rule, read as written): R10 is OUT for refcv9.** No decision cell has delta > +0.010 with a CI that
excludes 0. Per the committed outcome, the next pitch-free lever is the depth-distribution lift (refcv10). Inside refcv9 the
depth lever already queued is B1 (C3, the box-derived depth aux head), an E1 chain-2 arm.

Evidence class: **MEASURED (ours)**. Tier: **T0 perception instrument** (the map head against the SAM3 GT raster) with an
ORACLE pitch. This is not a deployable arm and not a driving number. Same checkpoint for every arm (refcv8 EMA-48,852), DEV
split (70 episodes, `DEV_HOLD_SPLIT.json`), pose-sync OFF.

## Decision cells (pre-registered in `HANDOFF.md` before any oracle raster was scored)

Pooled 20–60 m (bands 20–40 + 40–60, each at its own cross-fit threshold), arm minus static. Estimator: PAIRED episode-cluster
bootstrap (`taniteval/ci.py`), B 2000, seed 0, cross-fit 34 / 34 DEV episodes (1,164 fit / 1,165 score windows, 0 shared).

| cell | H (pitch − 3 s centred MA) | C (pitch − clip mean) |
|---|---|---|
| drivable IoU | +0.0006 [−0.0006, +0.0019] | −0.0022 [−0.0076, +0.0026] |
| boundary recall @0.3 m | +0.0033 [−0.0023, +0.0090] | **−0.0142 [−0.0274, −0.0031]** (separated, WORSE) |

Bar: +0.010 with a separated CI. H: both cells are within ±0.004 and their CIs include 0. C: one cell is separated, and it is
negative.

## Controls

* **Static reproduction (VOID gate): PASS.** The static arm's MAP-4 / MAP-3a / MAP-3b, in all four bands, equal E0D's
  `E0D_RESULT.json`: worst |diff| = 0.0 (tolerance ±0.002).
* **Scorer null control (MM):** `r10_score.py read` with static = H = C = E0D's own bank gives every delta +0.0000 [+0.0000,
  +0.0000] and static reproduction diff 0.0.
* **Instrument controls (R10-A, `test_r10_pitch.py`, 15 pass):**
  * Hook OFF is bit-identical to the tip (`torch.equal` 16 / 16 inside the real model).
  * Analytic row shift equals the independent cylindrical formula to 1e-8 px.
  * Sign: a synthetic braking dip reads +1.0° and moves the scene up; a sign-flipped hook FAILS.
  * A zero-delta run reproduces the static rasters, and those equal the banked MAPSEL `rast_s0` exactly (12 / 12).
  * Future-frame control PASS.
* **Data reads (`raw/pitch_table_eval139.meta.json`):**
  * 3 s-low-passed egomotion pitch against road grade: slope −1.03, corr −0.992, so **C carries the road grade**. That fits C
    hurting here: the lift is already fitted to the grade-blind static geometry, and C injects the grade as if it were body pitch.
  * H shows **no braking dive**: mean H at ax < −2 m/s² = +0.02° ± 0.06°. So H is probably not suspension pitch, and E0c's
    UNVERIFIED stands.

## Beside (not decision)

Per-band deltas (`raw/R10_TABLE.md`): H is within ±0.006 everywhere. Its only separated cells are all lane tol-IoU, and all
negative: 0–20 m −0.0016, 20–40 m −0.0017, pooled 20–60 m −0.0009. C is separated, and negative, on 10 of 15 cells.

Box AP@2 m: gate fitted ONCE on the static arm's DEV scores (0.2654, `--gate-from-scored`, declared) and applied unchanged to all three arms, so it can only favour static. Paired against static: all ranges H +0.0002 [−0.0001, +0.0004], C +0.0002 [−0.0007, +0.0009]. By range, |Δ| ≤ 0.0023 everywhere; separated only H 0–20 m +0.0004 [+0.0000, +0.0009] and C 40–60 m +0.0023 [+0.0005, +0.0040]. No material box effect either way.

## Caveat (stated in advance)

The checkpoint was trained with the static lift, so this null bounds the **free oracle gain without retraining**. It does not
bound a retrained, pitch-aware model. The oracle uses the NEWEST frame's pitch only.

## Manifest

| artifact | where |
|---|---|
| decision read | `raw/R10_RESULT.json`, `raw/R10_TABLE.md` (this package) |
| per-arm rasters + detections | `E:/r10_pitch_out/{H,C,static}/` (dev box only; ~32 MB detections + rasters per arm) |
| per-arm extraction stats | `E:/r10_pitch_out/stats_{static,H,C}.npz` (the E0D instrument's `extract`, unchanged) |
| box AP | `E:/r10_pitch_out/boxap_{static_gate,H_vs_static,C_vs_static}.json` |
| pitch table | `raw/pitch_table_eval139.{npz,meta.json}` |
| hook | `stack/tanitad/models/bev_lift.py` (`compose_pitch_delta`, `pitch_delta_rad=0.0`, dormant: OFF bit-identical) |
