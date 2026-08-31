# PRE-REGISTRATION — MM-E19: is action-deafness caused by the HORIZON?

**Written 2026-08-31, BEFORE the arm runs** · Master Mind · **Tier** T0 primary,
T1 escalation pre-committed · **Depends on** MM-E11 (O1-INERT), MM-E17 (attenuation
located), MM-E18 (the gain converged).

```yaml
hypothesis: MM-E19
question: MM-E18 showed the FiLM gain CONVERGED — the model could weight actions
          more and chose not to. If that is because over ONE TICK (0.1 s) the next
          latent is almost fully determined by the current one, then extending the
          rollout to 6.0 s should make actions matter TO THE LOSS, and the optimiser
          should allocate them more gain on its own.
one_variable: --o5-k  8 -> 60      (0.8 s -> 6.0 s, the §4b binding horizon)
matched_incumbent: postrain30k     (already measured under the same instrument)
```

## 1. Why this arm, and why it is not just "the horizon fix"

Three levers have been eliminated, each by reading weights rather than training:

| lever | verdict | cost |
|---|---|---|
| "the FiLM never left zero-init" | ⛔ REFUTED — it trained (‖W‖ 2.97/3.06/5.44) | minutes |
| "O1 was simply off" | ⛔ REFUTED — MM-E11, ratio FELL 0.40× | one 8.6 h arm |
| "the conditioning gain is starved" | ⛔ REFUTED — MM-E18, it CONVERGED | minutes |

⭐ **What survives is that the model is behaving CORRECTLY.** Over 0.1 s the scene
predicts itself, so the action's marginal contribution to O5 is tiny and a tiny gain is
the right allocation. **The action-deafness and the horizon are the same problem**, and
`--o5-k 60` is the one intervention MM-E11's objective-family exhaustion does not rule
out — at 6.0 s the ego has moved 60–170 m and the action becomes the dominant
determinant of where the scene goes.

⚠️ **This was predicted a week ago without a mechanism** — `PLAN_TO_THE_GOAL_2026-08-24`:
*"The 0.6 s horizon may be too short for actions to matter at all. If the ego's command
genuinely cannot move the scene in 6 ticks, no objective will make it."* MM-E11 is that
prediction confirmed: no objective made it.

## 2. Arm

`k60p30k` = `postrain30k`'s recorded launch line **verbatim** + `--o5-k 60`.

⛔ **`--horizons 1 2 4` is KEPT, deliberately, even though the trainer now refuses it for
new designs.** This is a *matched control against a banked incumbent*, and strict
comparability outweighs config hygiene: heads 2/4 are **proven to take exactly zero
gradient** (bit-identical across 10 snapshots), so they cannot influence the result,
while changing them would introduce a second config difference and a different
`state_dict`. The guard exists to stop new designs declaring dead horizons; it is not a
reason to break a control.

**Cost, MEASURED not extrapolated:** 2.9019 s/step ⇒ **24.2 h**, peak CUDA 6.27 GB.

## 3. Reads and outcomes, COMMITTED IN ADVANCE

**Primary:** the MM-E10/MM-E17 action-divergence probe — same corpus, same n, same
instrument. **Incumbent `postrain30k` h1 ratio = 0.00595.**

| outcome | criterion | consequence |
|---|---|---|
| **HORIZON-WORKS** | h1 ratio rises **≥10×** (to ≥0.06) | action-deafness was HORIZON-CAUSED; 6 s enters the recipe as a requirement, not a preference |
| **HORIZON-PARTIAL** | rises, but <10× | the horizon is *a* cause, not *the* cause; report the number, do not adopt alone |
| **HORIZON-INERT** | unchanged within noise | ⛔ action-deafness is NOT horizon-driven. Remaining candidates: the action REPRESENTATION itself, or the data. MM-E12 says actions are not redundant given scene, so information exists that nothing is using — that becomes the question |
| 🔶 VOID | C0 ≠ 0 or any scene_spread ≈ 0 | instrument fault, no verdict (C160) |

⭐ **SECONDARY, AND IT IS MECHANISTICALLY IMPLIED — so it is a real prediction, not a
second look at the same number:** if the horizon is what makes actions matter, the
optimiser should **allocate more gain on its own**. ⇒ `‖to_scale_shift‖` at 30k should
**exceed** the incumbent's converged **2.97 / 3.06 / 5.44**, and `‖act_emb.2‖` should
**stop declining** (the incumbent fell monotonically 9.5694 → 9.4015). ⚠️ If the ratio
rises but the gains do NOT, the ratio gain came from somewhere else and the mechanism
story is wrong — report both.

## 3b. ⭐ A THIRD READ, ADDED AT STEP ~600 — BEFORE ANY NUMBER EXISTS

**DRIFT, and it costs nothing extra.** MM-E18's mechanism implies it: drift is the
fraction of `Δz` predictable from `z_t` alone, and we measured it **SELF-DOMINATED**
(MM-E6 — the scene subspace carries ~20× less drift than a random subspace of equal
rank) and **trivially reducible** (MM-E4 — the shuffle control fired, so it is a
symptom). ⭐ If the predictor is scene-dominated *because* one tick barely moves the
scene, then **drift is the same artifact seen from another angle**: over 0.1 s, "predict
`z_{t+1}` from `z_t`" is nearly the identity, so a self-referential solution is close to
optimal. Over **60 ticks** the identity is useless and the rollout must actually
transport the scene.

| outcome | consequence |
|---|---|
| **drift FALLS materially at k=60** | ⭐ drift, action-deafness and the horizon are **ONE defect**, and the 6 s requirement addresses all three — the strongest possible result for the recipe |
| **drift unchanged** | ⛔ drift is INDEPENDENT of the horizon and needs its own lever; the pre-committed candidate remains the frozen/EMA teacher target |
| drift RISES | report it; a harder task may legitimately raise it, and per the anti-gate below that is not automatically a regression |

### ⛔ AMENDED AT STEP ~2,800 — THE DRIFT READ HAS LOW POWER, AND I AM SAYING SO BEFORE THE NUMBER

I wrote §3b without consulting `PROVEN_TRAINING_SETUP.md`. It records **the cleanest
separation in the campaign — eight arms, no overlap:**

| initialisation | drift fraction |
|---|---|
| **distilled** (3 arms) | 0.1753 / 0.1952 / 0.3650 |
| **scratch** (5 arms) | **0.6138 – 0.6416** — a **4 % band across unrelated recipes** |

> *"Five arms with different objectives, weights and schedules land within 4 % of each
> other. That is what a floor looks like. Objective design was not the variable;
> initialisation was."* — and **nine objective terms** (O1, O2, O3, O7–O11, PSG) failed
> to move it.

⛔ **`k60p30k` is a SCRATCH arm.** The prior therefore predicts drift ≈ **0.61–0.64**
almost regardless of what `o5_k` does. ⇒ **the outcomes in the table below are NOT
symmetric in what they license:**

* **"drift unchanged" is the EXPECTED result and is NEARLY UNINFORMATIVE.** ⛔ It must
  **not** be read as *"the horizon does not affect drift"* — it is what every scratch
  arm does, and reading the prior back as a finding is the error this amendment exists
  to prevent.
* **"drift falls materially" is a HIGH-INFORMATION surprise** — it would mean `o5_k`
  broke a floor that five unrelated recipes could not. That asymmetry is what makes the
  read worth taking at all.

⚠️ **The one reason not to discard the read outright:** `o5_k` is not another loss
weight. It changes the **task** — a 60-step rollout versus an 8-step one — so it is not
strictly a member of the "nine objective terms" class that failed. The read is
**weak-but-not-void**, and that is exactly how it must be reported.

⭐ **This weakens my own §3b prediction, before any number exists, on evidence I had not
consulted when I made it.** A prior that would have explained the result afterwards is
worth nothing; stated beforehand it changes what the result can mean.

### ⭐⭐ A SHARPER READ EXISTS, AND IT ALREADY HAS A BANKED CONTROL — added at step ~3,400

Looking for the drift instrument turned up a better quantity than drift level.
`latentmotion.py` (E-DEC-59, RFF+ridge onto the top-8 PCA directions) reports, for the
scratch arm `rdw8p30k` at **k=4** on a HELD-OUT split (80 clips / 7,680 rows):

| column | r | t |
|---|---|---|
| `z_t` (DRIFT / positive control) | **0.6718** | 134.84 |
| **`ego_state [w, a, v]`** | **0.0073** | 2.78 |
| `z_t + ego_state` | 0.6712 | 135.21 |
| constant (control) | **0.0000** | 0.00 |
| **ego marginal over drift** | **−0.0006** | **−0.48** |

⭐⭐ **THE EGO STATE — WHICH CONTAINS THE ACTION — ADDS NOTHING TO PREDICTING Δz BEYOND
`z_t` ALONE.** Marginal −0.0006 at t −0.48: not merely small, *not distinguishable from
zero*. And the constant control reads exactly 0.0000, so the panel is admissible.

⇒ ⭐⭐⭐ **THIS IS MM-E18 CONFIRMED FROM THE DATA SIDE, AND IT IS STRONGER THAN MY
MODEL-SIDE ARGUMENT.** I argued the optimiser *chose* a small action gain because the
action barely helps. This shows the action **genuinely does not predict the latent
transition at 0.4 s** — measured on held-out data, with no model's choices involved.
**The model is not ignoring a useful signal; it is correctly declining to use a useless
one.** Action-deafness is a *correct response to the horizon*, not a defect.

⛔ **⇒ THE PRIMARY DRIFT-SIDE READ FOR MM-E19 IS THE EGO MARGINAL, NOT THE DRIFT LEVEL.**
Drift level is floor-bound and low-power (previous amendment). The **ego marginal over
drift** is exactly the quantity the horizon hypothesis predicts should move, and it has
a banked reference and a working known-value control.

| outcome at k=60 | consequence |
|---|---|
| **ego marginal becomes POSITIVE and material** (t ≫ 2) | ⭐⭐ the horizon hypothesis is CONFIRMED on the data side: actions predict Δz once the horizon is long enough. The 6 s requirement becomes load-bearing for the whole recipe |
| **marginal stays ≈ 0** | ⛔ actions do not predict the latent transition at ANY horizon we can reach ⇒ the problem is the ACTION REPRESENTATION or the latent, and no horizon or objective fixes it |

⚠️ **Two scope limits, stated now.** (1) The banked reference is `rdw8p30k`, a *sibling*
scratch arm, not `postrain30k` — so the incumbent must be re-measured with this
instrument at k=4 **and** k=60 for a clean paired comparison; the banked row is a
reference, not the control. (2) `latentmotion.py` is run at **k=4**; extending it to
k=60 changes the horizon of the *probe* as well as of the arm, and both arms must be
read at both k values or the comparison confounds probe-horizon with arm-horizon.

⚠️ **Committed at step ~600 (and extended at ~3,400), with no drift number in hand for
this arm** — the same discipline as MM-E11's pre-read amendment. ⚠️ And the incumbent's drift must be read with
the **same instrument on the same corpus**, or this is not a comparison. ⛔ This is an
*additional read of an arm already running for another reason* — it must never be
described as "the drift experiment", because no drift-specific variable was manipulated.

## 4. ⛔ The anti-gates, committed before any number exists

* **O5 LOSS WILL BE HIGHER, AND THAT IS NOT A REGRESSION.** A 60-step rollout is a
  strictly harder task than an 8-step one. ⛔ `o5_loss` **must not be compared across
  different `o5_k`** — it is a different quantity, not a worse score. The timing probe
  already showed `o5_stepK` ≈ 0.10–0.21 at k=60 against step-1 error ≈ 0.05.
* **FEWER WINDOWS.** k=60 yields **319,002** windows; the §4b addendum prices the 6 s
  horizon at ~43 % fewer per episode. Any comparison to a k=8 arm must state that the
  two arms saw different sample counts — the parity CORPUS is identical, the window
  count is not.
* Per the MM-E11 precedent: a T0 prediction regression **with** an action-ratio gain is
  the intended trade and escalates to T1. A T0 regression with **no** gain is a dead arm.

## 5. What this cannot show

Nothing about driving (T0). Whether reach converts to skill needs T1 with the
hold-action control. ⚠️ Single seed: magnitudes below the (unmeasured at 30k) seed band
are **unresolved, not null**. The ≥10× criterion is set far outside any plausible seed
band precisely because the incumbent sits at 0.00595 and MM-E11's own arm moved 0.40×.
