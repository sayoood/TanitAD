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

## ✅ THE REPLICATE LANDED — the claim is now COMPLETE, and every control passes

| | `argmax@s0` | `prior050` (lever) | `argmax@s1` (**replicate**) | floor `ha0_ext` |
|---|---|---|---|---|
| **ALL 28 windows** | 1.6098 | 2.5025 | 1.7952 | 0.4263 |
| **18 CHANGED windows** | 1.6087 | **2.9972** | 1.7328 | 0.4809 |
| **10 UNCHANGED windows** | 1.6119 | **1.6119** | 1.9075 | 0.3280 |

**1. The design control PASSES.** On the 10 windows whose decode does not change, the two
arms are **BIT-IDENTICAL** — `max_abs_diff = 0.0`, lever delta exactly **0.0**. The targeted
design's premise (`marginal = (18/282) × conditional`) is therefore **verified, not
assumed**, and the aggregate is recoverable from this panel.

**2. `H-ESTIM-SEED-1` IS SATISFIED.** On the 18 windows where the lever acts:

| | delta vs `argmax@s0` |
|---|---|
| **LEVER** (`prior050`, same seed) | **+1.3886** |
| **REPLICATE** (`argmax`, seed 1, *zero levers moved*) | **+0.1242** |
| **ratio** | **11.2×** |

⇒ the lever's effect is **11.2× the rig's own run-to-run noise floor**. A separated CI alone
would have been necessary-not-sufficient; against the replicate it is now **sufficient**.
**The lever moved the metric, and it moved it the wrong way.**

⚠️ **And the noise floor is not negligible** — `argmax@s1 − argmax@s0` is **+0.1242** on
changed windows and **+0.2956** on unchanged ones, from changing *only* the iCEM seed. ⇒ on
this panel **any refav1 claim smaller than ≈0.30 m is indistinguishable from seed noise**.
That number should travel with every future arm-vs-arm comparison on this rig.

**3. On the windows where the lever fires, refav1 is 3.4× → 6.2× below the floor**
(1.609 → 2.997 against `ha0_ext` 0.481).

## Status of this claim — COMPLETE

⚠️ **Scope:** n = 28 windows / 14 clusters, one checkpoint, and a deliberately turn-heavy
panel. The verdict "Step 1 is closed" is supported on *these* windows against *this* noise
floor; it is not a corpus-wide ADE claim, and the package never makes one.
