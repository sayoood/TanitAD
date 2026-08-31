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
