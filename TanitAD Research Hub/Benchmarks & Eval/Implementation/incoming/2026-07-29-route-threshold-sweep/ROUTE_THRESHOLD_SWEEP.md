# Task #43 — the route collapse is a CALIBRATION artifact. One constant buys +0.1251 balanced accuracy.

**PI go given 2026-07-29 ("43 go on the route head"). MEASURED same day**, v4 from-scratch 30k arm
(step 29,999), **727 judgeable windows of 881 / 40 episodes**, pod2. **No training. No GPU.**

## 0. Result

| | thr | **balanced acc** | plain acc | recall L / S / R | prec L |
|---|---|---|---|---|---|
| **current** | 0.7616 = tanh(1.0) | **0.4242** | 0.6162 | 0.231 / 1.000 / **0.041** | 0.907 |
| my shrinkage estimate | 0.5080 | 0.4976 | 0.6616 | 0.316 / 0.995 / 0.182 | 0.859 |
| **best** | **0.3500** | **0.5493** | **0.6864** | 0.392 / 0.967 / **0.289** | 0.830 |

⭐ **+0.1251 balanced accuracy (+29 % relative) from changing ONE HARD-CODED CONSTANT.**
**Right-turn recall rises 0.041 → 0.289 — a 7× increase — while straight recall barely moves
(1.000 → 0.967).** Baselines: always-straight majority 0.5420, 3-class chance 0.3333.

⭐ **It is not even a trade-off: plain accuracy ALSO improves (0.6162 → 0.6864).** The current
setting is **dominated on both metrics** — the collapse was pure waste, not a conservative choice.

## 1. Pre-registration, and which outcome fired

Both outcomes were committed in the script's docstring **before** any swept number was seen:

- **OUTCOME A** — balanced accuracy rises materially as `thr` falls, driven by turn RECALL at a cost
  in turn PRECISION ⇒ calibration artifact, one-line fix real.
- **OUTCOME B** — flat or falling at every `thr` ⇒ `curv_5s`'s ordering carries too little signal for
  any threshold to recover turns; the shrinkage story is wrong and the fix must be the regressor.

⇒ **OUTCOME A fired**, with one qualification: precision cost was **smaller than predicted**
(left 0.907 → 0.830) and plain accuracy *rose* rather than fell.

⚠️ **My shrinkage arithmetic was directionally right and quantitatively under-corrected.** From
R² 0.3142 I predicted ρ ≈ 0.56 ⇒ `thr ≈ tanh(0.56) ≈ 0.508`. That does help (+0.0734) but the
empirical optimum is **0.350**, well below it. The ρ = √R² shrinkage model is a useful pointer, not
a calibration method — treat it as having found the *direction*, not the value.

## 2. Why this works (the mechanism, restated against the numbers)

`goal_modes.scalars_to_goal` derives the route by hard threshold on ONE regressed scalar:
`graded = tanh(curv_5s / CURV_TURN_PER_M)`, `CURV_TURN_PER_M = 1/60 m`, fire at `|graded| ≥ tanh(1)`.
`curv_5s` is the **worst-fit** of the four scalars (**R² 0.3142, RMSE 0.0123 /m**), so its
predictions are shrunk toward zero and rarely clear a bar set at ~1.36× their own RMSE. Everything
gentler than a ~34 m-radius turn was being rounded to STRAIGHT.
**There is no route classifier and no route cross-entropy in this path at all** — which is why the
earlier "class-balanced/focal loss" proposal was aimed at a component that does not exist.

## 3. ⛔ What this does NOT establish

1. ⚠️ **The threshold was selected ON the evaluation split. 0.5493 is an UPPER BOUND on what
   recalibration buys, not a validated operating point.** It must be confirmed on data not used to
   pick it before anything is deployed.
   ⭐ *Mitigating, and worth weighing:* the optimum is a **broad plateau** — every `thr` in
   **0.25–0.40** yields balanced accuracy ≥ 0.535 — not a spike. A plateau that wide is much harder
   to explain as split-specific noise than a single lucky point would be.
2. **It does not improve the goal head.** `curv_5s` is still R² 0.3142; this only stops a bad
   decision rule from discarding the signal that is already there. Step 2 (improve the regressor)
   remains open and is now *separable* from the threshold question.
3. **It has not been shown to move ADE.** Better route agreement is necessary, not sufficient — the
   produced-goal arm's +0.0186 [−0.1711, +0.1940] vs constant velocity must be **re-measured** with
   the new threshold. That is the next run and it is cheap (one MODE-B eval).
4. **Right-turn precision (0.648 at the optimum) rests on few predictions** and inherits the small-n
   caveat from the parent analysis. Left is the well-estimated arm.

## 4. Next, in order

1. **Re-run MODE B produced with `thr = 0.35`** and paired-bootstrap `ade_0_2s` against the current
   arm on the same windows. **Pre-register:** the route fix is only worth deploying if it moves the
   closed-loop/ADE picture, not merely the route metric.
2. **Hold out a split** to confirm 0.35 (or pick it on train-side windows) before it becomes default.
3. **Make the threshold a config field, not a literal.** It is currently `tanh(1.0)` inline at
   `goal_modes.py:148`; a swept value cannot be adopted safely while it is a magic number.

## 5. Provenance

`code/route_threshold_sweep.py` (pre-registration in its docstring) · `raw_route_threshold_sweep.json`
· `raw_goalagree_v4fs-30k-produced.pt` (the per-window dump that made this free).
Enabled by a 2-file patch: `GoalAgreement.dump()` + `--agree_dump` wiring, which persists tensors the
class already held and previously discarded. `stack`: 55 passed / 1 skipped on the touched suites.
