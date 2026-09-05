# The plan-level A/B: the cheap fix does NOT make refav1 drive — it makes it worse

**Row:** `D-REFAV1-DRIVE-AB` · **Class:** MEASURED, T1 · **Date:** 2026-09-05
**Panel:** 14 episodes / 28 windows, stride 40, `ccos` + Stage-B weights, step 21,109
**Instruments:** `tools/run_ab_targeted.sh`, `tools/analyze_targeted.py`
**Raw:** `raw/abt_arms.json`, records at `C:/Users/Admin/refav1_drive/abt/` (dev box)

---

## The result

| metric (T1) | `argmax` (control) | `prior τ=0.5` (lever) | delta |
|---|---|---|---|
| **ADE** | **1.6098** [1.074, 2.184] | **2.5025** [1.735, 3.468] | **+0.8927** |
| FDE | 3.3396 [2.341, 4.301] | 5.2108 [3.547, 7.036] | **+1.8712** |
| LON speed MAE | 0.5361 [0.416, 0.659] | 0.5311 [0.413, 0.656] | −0.0050 |
| LON along MAE | 0.7171 [0.516, 0.924] | 1.0943 [0.683, 1.635] | +0.3772 |
| LAT cross MAE | 1.3358 [0.807, 1.907] | 2.1590 [1.484, 2.948] | +0.8232 |
| LAT heading MAE | 12.07° [8.98, 15.05] | 21.13° [17.21, 24.95] | **+9.07°** |

**Every family degrades except longitudinal speed** — which is unchanged (−0.005), exactly
as expected from a *lateral-only* bias. That the one channel the lever does not touch is the
one channel that does not move is itself a control: the effect is where the intervention is.

**The lever did precisely what it was designed to do.** It drove the planner's
constant-velocity share from **0.214 → 0.000** and its straight-plan share from
**0.250 → 0.000**. It turned more. **Turning more made it worse.**

## ⛔ And both arms are far below the trivial floors

| floor (non-planner, identical across arms — verified) | ADE |
|---|---|
| `ha` (hold action) | **0.4417** [0.334, 0.565] |
| `ha0_ext` (the canonical integrator) | **0.4263** [0.338, 0.530] |
| `ha0` | 0.4360 [0.342, 0.539] |
| `ol` ⛔ **T0**, a WM diagnostic — never driving | 0.3142 [0.218, 0.430] |

refav1's planner reads **1.61** against a **0.43** floor — **3.6× worse**, and the lever
takes it to **5.9× worse**. The brief's success condition (*beat `ha` and `ha0_ext` at T1*)
is not approached from either side.

⚠️ **SCOPE, and it matters: this panel is TURN-HEAVY BY CONSTRUCTION.** These 14 episodes
were selected because the lever changes the decode there. On the full grid refav1's ADE is
≈0.55 against floors ≈0.52 — close. **So the 3.6× gap is not a corpus-wide number; it is
what refav1's deficit looks like *concentrated on the turning windows*.** That is the
informative reading: refav1's failure is not diffuse, it lives in the turns.

## What this settles

`MUST_WE_RETRAIN.md` predicted from the ROC that Step 1 *"helps but is UNLIKELY TO BE
SUFFICIENT"*, because recall is bought at ≈0.49 false turns per unit gained. **The
measurement is stronger than the prediction: at this setting Step 1 is not insufficient, it
is actively harmful.** It composes exactly with `D-REFAV1-CCOS-ARMS` (executing more turns
degraded every family) — now with the decision rule, rather than the cost metric, as the
thing that caused the extra turns. Two independent levers, same direction, same reason:
**the turns refav1 adds are, on balance, wrong turns.**

⇒ **Step 1 is CLOSED as a route to driving.** The lever remains a correct and useful
instrument — it is how Step 2's success will be measured — but no setting of it should be
shipped on this checkpoint.

⇒ **Step 2 is now the live rung, and its objective is confirmed to be the right one:** not
more turns, but a **wider margin gap** (today 0.297 logits). A goal head that fires more
often at today's precision makes driving worse; only a head that fires more *accurately*
can help.

## ⚠️ Status of this claim

**PENDING the replicate arm** (`argmax@seed1`, running at the time of writing). `icem_plan`
is stochastic, so `H-ESTIM-SEED-1` applies: +0.8927 is only a lever effect if it clearly
exceeds `|argmax@seed1 − argmax@seed0|` on the same windows. Until that reads, this is a
**necessary-not-sufficient** observation. ⚠️ Also **n = 28 windows / 14 clusters** — small,
and the per-arm intervals overlap ([1.07, 2.18] vs [1.74, 3.47]); the paired delta is the
decision-grade form and is computed by `tools/analyze_targeted.py`.

**Control still to be read first:** the 10 windows whose decode does NOT change must be
bit-identical between arms. If they are not, the targeted design's factorisation is void.
