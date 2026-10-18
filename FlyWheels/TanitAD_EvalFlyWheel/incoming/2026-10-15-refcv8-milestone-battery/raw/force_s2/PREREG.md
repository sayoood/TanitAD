# PREREG -- forced tactical-lon re-roll on the S2 windows where BAR-R8-E1 is lost

**Registered 2026-10-18, 21:30 +0200** (Master Mind brief to the implementer, quoted verbatim below). The `decide()` of `code/force_s2_reroll_r8.py` was in that file from its creation (21:16:56) and pinned by `code/test_force_s2_r8.py` (created 21:19:01), **before any real number of this roll existed**. The only numbers consulted while writing it are the banked E1 decomposition ones (`raw/e1_decomp/E1_DECOMP.md`, `VERDICT_ANALYST.md`): S2A 434 windows +0.0135, S2B 424 windows +0.0066, E1 = -0.0085 [-0.0229, +0.0065]. No forced plan has been computed.

Evidence class of everything below: **HYPOTHESIS under test**. Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Model: refcv8-r101-s0 EMA step 30,000, CPU fp32 trunk. Estimator: paired episode-cluster bootstrap over the episodes present (models FIXED, B 2000, draws identical to `taniteval.ci._draws`) -- it answers "another draw of EPISODES" only; NOT another training run (single training seed), NOT another inference run (N2 is the draw-to-draw floor).

## The pre-registration (verbatim from the brief)

Context (MEASURED, PKG/raw/e1_decomp/E1_DECOMP.md): refcv8 EMA-30k, capped, ceiling-compliant S2 set (4,159 windows / 138 ep):
E1 = os - ha0_ext S2 ADE = -0.0085 [-0.0229, +0.0065], not separated. Its gross loss +0.0264 is 72 % in slow / slowing windows
(S2C +0.0189): refcv8 ends ~+0.25 m too FAR at 2 s where the human stays slow or keeps braking; its lon head argmax is
ACCELERATE in 72 % of slow_both windows.

Windows: S2A = dv2 in {slow_both, stopping} (434), S2B = dv2 decelerating (424), both on the compliant S2 set, seed 0.
Subsample to <= 220 per set, stratified by episode (seeded), and record n. Arms, same per-window sampler seed across arms:
 * N native; F1 = force the LON6 class mapped from GT `lon_v7` (the mapping force_lon_reroll_r8.py already uses; unlabelled
   windows excluded and counted); F_slow = force HOLD (LON6 index 4) on S2A and BRAKE_TO (index 2) on S2B -- VERIFY both
   indices from source and quote file:line; on a 100-window seeded subset per set also Fown (own argmax class) and N2 (second
   sampler draw = the inference-noise floor).
Metric per window: d = ADE_arm(S2 0-2 s) - ADE_echo(S2), echo = the panel's `ha0_ext` for that window (as e1_decomp_r8 reads it).
PRIMARY statistic, pre-registered in CODE (a `decide()` pinned by a test with literal expectations and a red mutation, written
BEFORE any real number exists): on S2A u S2B, closure_X = (mean d_N - mean d_X) / mean d_N with a paired episode-cluster
bootstrap CI (B 2000), for X in {F1, F_slow}. Outcomes:
 * TIMING-CONTROLLABLE: closure_Fslow >= 0.50 with CI lower bound > 0, AND closure_F1 < 0.50 -- the near-term plan obeys the
   class; the defect is the lon class's TIMING (the label covers NOW+2..8 s).
 * IDENTITY: closure_F1 >= 0.50 with CI lower bound > 0 -- the head picks the wrong class.
 * DECODER: both closures' CI upper bounds < 0.25 -- forcing the class does not move the near-term plan.
 * INCONCLUSIVE otherwise (report n_needed).
Report beside it: per-set closures, KA (CPU N vs banked GPU `os` mean ADE difference against the banked inference floor, as
force_lon does), KB (Fown changes less than F_slow), and what E1 would read if S2A u S2B took the F_slow plans (arithmetic,
labelled as such, not a result).

## Source quotes for the two forced classes (verified 2026-10-18 in the launch tree `C:/lgs/Lfb47a58376/stack`)

* `tanitad/refs/refcv8_conditioning.py:63` -- `LON6: tuple[str, ...] = ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")` -> **BRAKE_TO = index 2, HOLD = index 4**; `:64` `LON6_V7_IDS = (0, 1, 3, 4, 5, 7)`.
* `tanitad/refs/refc_v3.py:2081` -- `RefCV3Model.set_r8_force(self, lat3=None, lon6=None, cons=None, min_alloc=None)`: *"The forced posterior reaches EVERY planner feed of the tactical layer (tac8 prior, phi, allocation, the factorised score). Consumed by the next forward."* A forced class is a +-20-logit one-hot (`_r8_apply_force`, `big = 20.0`).
* In code: `SLOW_CLASS6 = {"S2A": 4, "S2B": 2}`, pinned literally in `test_slow_classes_are_hold_on_S2A_and_brake_to_on_S2B`, and re-asserted against `FL.LON6` at run time (`assert_fl_internals`).

## Implementation readings (they add no threshold and move none; a reader who disagrees can recompute from the per-window table)

1. **Populations differ between the two closures, by the brief's own wording.** `closure_F1` is computed on the windows that HAVE an F1 arm (labelled, LON6-mapped; the rest are excluded and counted), with N paired on the same windows; `closure_F_slow` is computed on every window that has the arm. In the banked slow_both windows only 161 of 376 carry a GT `lon_v7` (VERDICT_ANALYST section 2), so F1 may rest on well under the full draw. The **like-for-like F_slow closure on the F1 windows is reported beside the primary** (`closure_Fslow_like_for_like_on_F1_windows`); it does not enter `decide()`.
2. An **undefined closure** (non-positive denominator, or no data) makes the outcome INCONCLUSIVE, never a pass. Bounds are compared exactly as written: point >= 0.50, lower > 0, upper < 0.25, F1 point < 0.50 (for TIMING).
3. **Mutually exclusive by construction:** TIMING needs F1 point < 0.50, IDENTITY needs F1 point >= 0.50, DECODER needs both uppers < 0.25; at most one fires whatever the evaluation order.
4. The sets are regenerated by `e1_decomp_r8.py`'s own loader and `build_partitions` (imported); the plan **refuses** (writes nothing) unless they count exactly 434 / 424.
5. `d_N` is the **CPU** native arm on the same windows (so N and X differ only by the force). A sensitivity with the banked GPU `os` as base is reported and labelled device-mixing.
6. A **partial roll** is analysed and labelled `PARTIAL` with per-set coverage; a verdict from a partial roll is the committed table applied to what exists, not a completed pre-registered read.
7. The analysis **refuses a verdict** (`REFUSED`, with the table's reading kept beside it) unless every recorded `forced6` equals what the plan says was forced -- the F_slow mechanism rebinds force_lon's module-global `CRUISE6` per single-set chunk, and this content check is what proves the rebinding took effect.

## Frozen artifacts

The code hashes are recorded in `HANDOFF.md` ("Frozen hashes") at the moment the launcher was armed (2026-10-18 21:31 +0200); `decide()` has not been edited since the tests that pin it were written.
