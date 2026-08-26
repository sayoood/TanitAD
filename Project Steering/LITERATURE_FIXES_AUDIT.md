# The literature fixes: what I found, what I did with them, and what the corrected diagnosis actually calls for

**Written** 2026-08-26 · **Author** Master Mind · **Trigger** the PI: *"I thought
also, you found fixes from the literature"* — a fair challenge, and the audit below
is less flattering than the answer I would have given from memory.

---

## 1. ⛔ The audit

| paper | banked | did we act on it? |
|---|---|---|
| **ActSWM** (arXiv 2607.26712) — names our failure **"context collapse"**; fix = hinge on rollout separation **+ a frozen, randomly-initialised action readout** | ✅ | ⛔ **NEVER IMPLEMENTED.** Planned as O12, demoted, superseded by my own O13. |
| **ACT-Bench** (arXiv 2412.05337) — external action-controllability yardstick | ✅ (note: *"queued"*) | ⛔ **NEVER ADOPTED.** I wrote that we should adopt it and did not. |
| **Causal Confusion in IL** (arXiv 1905.11979) | ✅ | ✅ **Used** — it is the theoretical frame for E-DEC-48b. |
| **V-JEPA 2 / 2-AC** (arXiv 2506.09985) | ✅ | ⛔ not used |
| **Genie** (2402.15391) · **LAPO** (2312.10812) · **VPT-IDM** (2206.11795) | ⛔ **were not banked at all** — banked 2026-08-26 | — |

⛔⛔ **THE ORDERING ERROR, AND IT IS THE EXPENSIVE ONE.** ActSWM independently
reproduced our exact failure — *"a jointly trained readout increases separation at
the cost of prediction quality"*, fidelity 0.972 → 0.698, which is O11's signature
to the shape — **and published a method.** I planned it as O12, then **demoted it on
an argument** (*"it aims at the scene, where E-DEC-48b measured no action
information"*) **and built a bespoke term, O13, instead. O13 failed at +192.4 %.**

⇒ **The published method that named our failure was never run. The bespoke one was.**
That is the wrong order, and the operating standard's *"settle conflicts with
experiments, not deference"* does not license it — running the cheap published
baseline first **is** the experiment.

⚠️ **E-DEC-57 voids the reasoning I demoted O12 with — but does not rescue O12.**
A frozen readout mining a channel that is a units-conversion of ego motion still has
nothing to mine. ⇒ **The demotion was right by accident and wrong by argument**,
which is worth more to record than a demotion that was simply wrong.

---

## 2. ⭐ The corrected diagnosis points at a DIFFERENT literature

**E-DEC-57:** our `action` is `atan(L·κ)` from the corpus's raw `curvature` column,
and the closed form `v·tan(steer)/L` reproduces the measured yaw-rate at
**r = 0.9988**. ⇒ **We never had a command channel — we had the ego's realised
motion in other units.**

That changes which shelf the fix is on:

| the problem I *thought* we had | the problem we *actually* have |
|---|---|
| "the predictor ignores the action" | **"there is no genuine action to ignore"** |
| → ActSWM's frozen readout | → ⭐ **latent-action / inverse-dynamics models** |

**The matching family, now banked:**

- ⭐⭐ **VPT** (2206.11795) — train an **inverse dynamics model** on a *small*
  labelled set, then pseudo-label action-free video at scale. **The canonical answer
  to "we have observations but no commands."**
- ⭐⭐ **LAPO** (2312.10812, *Learning to Act without Actions*) — recover **latent
  actions from observation pairs alone**, then ground them to true actions with a
  small labelled set. **The grounding step is exactly what comma2k19's CAN data
  could supply.**
- ⭐ **Genie** (2402.15391) — a latent action model (VQ over observation pairs) plus
  dynamics conditioned on it, learned entirely from unlabelled video.
- **V-JEPA 2-AC** (2506.09985) — a compact action-conditioned predictor on a frozen
  video encoder with *limited* action data; the closest published recipe to our
  architecture.

⚠️ **All four assume the action is a genuine control input that is merely
UNLABELLED. Our case is worse: the label exists and is a KINEMATIC RESTATEMENT.**
That is not fatal — it makes our situation the *action-free* one, which is precisely
what these methods address — but it must be stated, not glossed.

---

## 3. ⇒ The cheapest discriminating experiment, before any GPU

**Question:** is a genuine command channel separable from the realised motion at
all, in any corpus we hold?

**Test (~1 h, CPU):** on **comma2k19**, compute `r(steering_wheel_angle, v·κ)` on the
same held-out protocol.

| outcome | what it means |
|---|---|
| **r ≈ 0.99** (as PhysicalAI) | **No observational corpus can settle this.** Action-conditioning needs **interventional** data, and the E-DEC-52 conclusion hardens into a provisioning decision. |
| **r clearly < 0.99** | ⭐ **We have a real action channel.** Then VPT/LAPO's recipe applies: learn an IDM on comma2k19's CAN, pseudo-label PhysicalAI, and **the ten failed objectives deserve exactly one honest rerun** — the first on a channel with causal content. |

⛔ **Run this before ActSWM's O12, before any weight sweep, and before a simulator.**
It is the one measurement that tells us which literature applies.

⚠️ **And adopt ACT-Bench regardless.** Every number in this campaign is our own
instrument; after seven retractions in one night, an external yardstick is not a
nicety. It is banked and has been "queued" for days.
