# SPEC — refav1: does `loncomb3` beat doing nothing on the FULL eval grid? (pre-registered)

**Written 2026-09-27 ~01:20 Berlin by the TrainingFlyWheel, BEFORE any full-grid `loncomb3` data exists.**
Staged, not committed (agents never commit); the Master Mind lands it. The run may not start until this file is
staged and its blob verified.

## 1. Why this run, and why now
The only refav1 checkpoint (`refav1-b1-v72-ep3-speed`, step 21,109) **does not beat doing nothing** on the full
141-episode grid with the shipped cost: `cl − ha0` ADE **+0.0158 [+0.0007, +0.0315]**, separated, worse
(`taniteval/results/RESULT-refav1-21109-openloop.md:45,398`). On the 40-window / 8-episode p4 panel the planner
configuration `loncomb3` is the best of 30 arms and beats the strongest floor `ha0_ext` on LON along-track,
heading and yaw-rate (separated), but its ADE edge (−0.1033 [−0.2676, +0.0851]) is **not separated on 8
clusters** (`../2026-09-05-refav1-close-the-gaps/raw/refav1_margin_banked_2026-09-27/RESULT_HARVEST.md` §3).
The episode axis is the binding one; this run gives it 141 clusters. **Inference only — no training, same weights.**

## 2. Hypothesis
> **`H-REFAV1-LONCOMB3-FULL`** — with the step-21,109 checkpoint unchanged, the planner configuration `loncomb3`
> beats the do-nothing floor `ha0_ext` on ADE at T1 over the full 141-episode v7.2 EVAL grid, at two inference
> seeds, by more than twice the inference-seed replicate difference.

## 3. Arms (identical checkpoint, identical windows; only `--plan-seed` differs between them)
| arm | flags on top of the full-grid invocation (`taniteval/tools/REFAV1_ARM.md:196-207`) |
|---|---|
| **A1 `loncomb3_s0`** | `--cost-metric ccos --cost-weights <W_JERK 0.02, W_KAPPA 15.11245, W_VEND 64.2971504241507> --a-sustain-mode a0_shift --jerk-seam a0 --plan-seed 0` (= `2026-09-05-refav1-longitudinal/raw/queueLON3.sh:77`) |
| **A2 `loncomb3_s1`** | identical, `--plan-seed 1` — the **mandatory inference-seed replicate** (iCEM samples) |

Common: `--window-stride 40 --episodes-n 0 --no-navshuf`, the 141-clip eval split (`refav1-fp8-eval`), v7.2 EVAL
labels + nav (md5 `aa12c948…`), `--no-lead-block` (as every p4 arm; distance-keeping is not a bar here).
Floors `ha`, `ha0`, `ha0_ext` (T1) and `ol` (T0) are produced inside each arm.
⚠️ `W_VEND` is **inert** (the arm never passes `target_speed`); it is carried only so the triple equals `loncomb3`'s.

## 4. Bars — committed now
**Estimator:** paired episode-cluster bootstrap over the 141 episodes, n_boot 2,000
(`taniteval/tools/refav1_paired_delta.py`), all four families. **Tier:** T1 = self-action open loop
(⛔ NOT closed loop, per the open-vs-closed-loop ruling).

- **BAR-L1 (primary, gating):** `A1 − ha0_ext` AND `A2 − ha0_ext` ADE < 0 with each CI excluding 0,
  **AND** |mean lever| > 2 × |`A2 − A1`| ADE (the replicate floor).
- **Secondary, reported per family, never pooled:** FDE vs `ha0_ext`; LONGITUDINAL (speed MAE, along-track, accel);
  LATERAL (cross, heading, yaw-rate, masked); TACTICAL (lateral + longitudinal decision recall incl. `turn_left`);
  STRATEGIC **UNAVAILABLE** (no route label exists on this eval surface; n = 0, reason stated).
- **Committed predictions (from p4):** ADE ≈ −0.10 vs `ha0_ext`; **LON speed still WORSE than `ha0_ext`,
  separated** (p4: +0.1608); heading and yaw-rate better; `turn_left` recall ≈ 0 (every `W_KAPPA = 15.11` arm
  loses left turns). The two predicted losses are stated here so they cannot be dropped later.

## 5. Outcomes, all committed
| outcome | reading | consequence |
|---|---|---|
| **PASS** | BAR-L1 met | the first refav1 configuration shown to beat the strongest do-nothing floor on ADE at T1 on the full grid; LON speed and left turns are the named remaining losses |
| **FAIL-TIE** | either CI contains 0, or the lever ≤ 2× the replicate floor | reported as **FAILED**; the p4 edge did not survive 141 clusters. Next lever: the terminal-speed term (PI) |
| **FAIL-WORSE** | `A − ha0_ext` ADE > 0, separated | the p4 panel's turn-dense 8 episodes did not represent the grid (244/282 windows are LANE_KEEP) |
| **VOID** | floors not bit-identical between A1 and A2, or the known-value control fails | no reading |

## 6. Controls and provenance
- `ha`/`ha0`/`ha0_ext`/`ol` must be **bit-identical** between A1 and A2 (same windows, same floors).
- Known-value control `X − X = 0` inside the paired tool.
- **Deliberate-regression reference:** the shipped cost on this exact grid is already MEASURED worse than `ha0`
  (+0.0158 separated). ⚠️ It was run 2026-09-04/05 on Thor's older stack, so it is a reference, not a same-version
  arm; re-running it costs one more arm and is optional.
- Code: tip `b3f7ea6f` `stack/` + `taniteval/`; checkpoint md5 `1189bc020018c2c67ce03d566c390285`; md5s of the
  tool and every input recorded in the run log before the first window.
- Compute: the 141-episode inputs exist on Thor (~40 s/window ⇒ ~3.1 h/arm) or on the dev box after a LAN copy
  (~21 s/window on the 4060 ⇒ ~1.6 h/arm). Two arms. Not a training run, so the 2026-09-26 launch gate does not
  apply; it would for any retrain.

## 7. Out of scope
Arming `W_VEND` (PI-reserved), any turn-vocabulary change, any retrain.

## 8. Amendment A1 (2026-09-27 01:32 Berlin, staged-blob timestamp) — a zero-training blend with the kinematic hold
**Registered BEFORE any full-grid data AND before any refav1 blend number has been computed on any panel.**

**Hypothesis `H-REFAV1-BLEND-1`:** the position blend **b50 = 0.5 · `loncomb3` + 0.5 · `ha0_ext`** beats `ha0_ext`
on ADE at T1 over the full grid.
- **Why:** every REF-C arm measured last night showed the same complementarity — the zero-training blend
  `w·os + (1−w)·ha` beat the echo for refcv6 at 5k and 30k (separated at both inference seeds), refcv4b and
  refcv5-v2, and the fitted `w` rose to ≈ 0.5 (`Project Steering/SPEC_REFCV7.md` §1 NEW-1). refav1's `loncomb3`
  shows the same error shape on p4 against `ha0_ext` (better heading and yaw-rate, worse speed).
- **`w` is FIXED at 0.5** — the value the REF-C fits converged to — **not fitted on any refav1 data**. No other
  `w` is scored as primary.
- **Construction:** the per-step positions (`cl`, `[n, 10, 2]`) are blended; every family is recomputed from the
  blended positions by the unchanged path (`refav1_arm._components` inside `refav1_paired_delta.py`), via a
  synthetic dump whose `cl` is the blend and whose floors are untouched (the tool's bit-identical floor check
  must pass). **Zero GPU** — it reuses A1/A2's dumps.
- **Deployability:** `ha0_ext` is causal by construction — `a0` is the BACKWARD difference of measured speed and
  `κ0` the recorded curvature at t0; "every index is <= i0" (`taniteval/tools/refav1_arm.py:414-445`) — the same
  measured-t0-state class as refcv7's NEW-1 prior. The PI's 2026-09-02 ruling admits measured v0 at t0; that it
  covers `a0` and `κ0` at t0 is my reading, flagged for the PI rather than assumed.

**BAR-B1:** `b50(A1) − ha0_ext` AND `b50(A2) − ha0_ext` ADE < 0 with each CI excluding 0, AND the mean beyond
2 × |`b50(A2) − b50(A1)`|. Four families reported per family; `b50 − loncomb3` reported.
**Known-value controls (the blend's regression arms):** at `w = 0` the blend IS `ha0_ext` and must read exactly
**0.0000** against it on every metric; at `w = 1` it IS the arm and must reproduce the arm's own row exactly.
**Sensitivity (reported, never gating):** `w` fitted leave-one-episode-out on the full grid (fit on 140 episodes,
score the held-out one) — the deployable form of the fit.
**Outcomes:** PASS = BAR-B1 met (a zero-training refav1 planner output that beats the strongest do-nothing floor on
ADE); FAIL = otherwise, reported as FAILED. ⚠️ Any p4 blend number computed after this amendment is EXPLORATORY —
p4 is the panel on which `loncomb3` was selected as best, so it cannot confirm anything.

## 9. Amendment A2 (2026-09-27 ~01:45 Berlin) — the planner-free DAMPED HOLD is a stronger floor; the blend must beat it
**Added AFTER the exploratory p4 blend numbers (disclosed here), and still BEFORE any full-grid data. It only adds
a control and makes the test STRICTER; no bar in §4 or §8 is loosened.**

**Why.** Exploratory p4 (`raw/p4_exploratory/`, 40 windows / 8 clusters, T1): `b50(loncomb3) − ha0_ext` ADE
**−0.2723 [−0.4235, −0.1475]** — but the SAME blend applied to the planner-free plan `ha0` (straight, constant
speed), **`damp50 = 0.5·ha0 + 0.5·ha0_ext`**, already reads **−0.2152 [−0.3634, −0.0839]** against `ha0_ext`. About
80 % of the blend's ADE gain is DAMPING the kinematic extrapolation, available with no model at all. The planner's
own contribution over `damp50` was LONGITUDINAL (speed −0.0918 [−0.1635, −0.0280], along −0.0978
[−0.1507, −0.0458], separated) and NOT separated on ADE (−0.0571 [−0.1377, +0.0397]). ⇒ a floor must be given the
same affordance as the arm: the blend's floor is the blended do-nothing plan.

**BAR-B2 (the attribution bar, gating for any claim that the PLANNER helps):** `b50(A) − damp50` < 0 with CI
excluding 0, at BOTH seeds, on **ADE** (primary) — and reported on LON speed / LON along (where p4 predicts the
planner's contribution lives).
**Reported beside every blend row:** `damp50 − ha0_ext` (the stronger trivial floor) — if it is separated on the full
grid, `ha0_ext` stops being the programme's strongest do-nothing floor for refav1, and every earlier "vs `ha0_ext`"
reading must be restated against it.
**Committed prediction (from p4):** BAR-B1 likely PASSES (driven by damping); BAR-B2 on ADE is uncertain; the
planner's separated contribution, if any, is longitudinal.

## 10. Amendment A3 (2026-09-27 ~02:00 Berlin) — the damped hold-action, measured on the FULL grid; BAR-B2 must hold against it too
**Added after an EXPLORATORY zero-GPU read of the EXISTING shipped-cost full-grid dump (Thor, 2026-09-04, 141 episodes /
282 windows; `raw/full141_floor_exploratory/`), still BEFORE any `loncomb3` full-grid data; it only adds a floor.**
That dump predates `ha0_ext` (Rung A1), so the full-grid damped floor available in it is **`dampha = 0.5·ha0 + 0.5·ha`**.
MEASURED (141 clusters, T1, known-value control w=0 reads exactly 0.0000): `dampha − ha` ADE **−0.1559 [−0.2249, −0.0948]**,
`dampha − ha0` ADE **−0.1484 [−0.1846, −0.1160]**, both separated; `dampha − ha0` also better on every LON metric.
⇒ On this grid a model-free damped hold beats both undamped floors by ~0.15 m ADE.
**Change:** the new A1/A2 dumps carry `ha`, `ha0` and `ha0_ext`, so BOTH damped floors are reported (`damp50` from
`ha0_ext`, `dampha` from `ha`), and **BAR-B2 must hold against the stronger of the two** (the one with the lower ADE),
at both seeds. The stronger floor is also reported against the arm alone (`loncomb3 − dampha`).
