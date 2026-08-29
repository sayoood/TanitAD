# RESULT — P-RC21 RE-RUN (AMENDMENT 2): EXIT C. The lever is the FAN, not the reward.

**Date:** 2026-08-29 · **Owner:** TanitAD_TrainingFlyWheel · **Tier: T0 training-side**
**Pre-registration:** `PREREG_P_RC21.md` **AMENDMENT 2**, landed before any
proximity arm existed (third consecutive prereg ahead of its numbers).
**Evidence class:** MEASURED (ours) · ⚠️ NON-PARITY corpus · paired
episode-cluster bootstrap, 4000 reps, 15 clusters, eval-mode deterministic readout.

---

## 0. Verdict: EXIT C — and it terminates the reward-composition line

> *"C — neither: total ΔR1 ≈ 0 at every w. The barrier neutralised the easy gain
> without supplying a usable one; the objective is now flat inside the trust
> region ⇒ STOP the reward-composition line. The next lever is the FAN (its
> diversity/support), not the scoring of it."*

⛔ **Nothing improved.** In both anchored arms every component either declined or
stayed flat, and **`proximity` — the term this amendment added — did not move at
all** (weighted Δ **−0.0004** at w=1, **+0.0002** at w=10).

## 1. Paired deltas

| arm | ΔR1 | ΔR2 collision (pp) | ΔR3 sel-ADE (m) |
|---|---|---|---|
| **T1** w=1 +prox | **−0.0901** [−0.1018, −0.0792] **SEP** | +0.013 [−0.215, +0.273] | **+0.0947** [+0.037, +0.143] **SEP** |
| **T2** w=10 +prox | **−0.0191** [−0.0384, −0.0009] **SEP** | −0.013 [−0.137, +0.130] | **+0.0433** [+0.015, +0.073] **SEP** |
| T-reg (hackable) | −0.2712 [−0.314, −0.229] **SEP** | −0.241 [−1.58, +0.81] | +0.4279 [+0.281, +0.577] **SEP** |

✅ **T-reg degrades on both axes with separation ⇒ the readout is valid** and the
nulls below are measurements, not blindness.

## 2. The decomposition — the primary diagnostic

Weighted ΔR1 contribution, with the previous (no-proximity) arms as context:

| arm | collision | comfort | feasibility | headway | progress | **proximity** | SUM |
|---|---|---|---|---|---|---|---|
| ctx w=0 (no anchor, prev) | −0.0073 | +0.0115 | **+0.1101** | −0.0023 | +0.0151 | — | **+0.1272** |
| prev w=1 no-prox | −0.0017 | −0.0277 | −0.0355 | −0.0003 | −0.0007 | — | −0.0660 |
| prev w=10 no-prox | +0.0005 | −0.0220 | −0.0170 | −0.0003 | +0.0007 | — | −0.0380 |
| **T1 w=1 +prox** | −0.0001 | −0.0225 | −0.0672 | −0.0004 | +0.0001 | **−0.0004** | −0.0901 |
| **T2 w=10 +prox** | +0.0001 | −0.0154 | −0.0034 | −0.0001 | −0.0004 | **+0.0002** | −0.0191 |

⚠️ **`proximity` had to be measured SEPARATELY** — see §4. `R1` is scored with the
DEFAULT spec for cross-arm comparability, so it excludes the new term by
construction, and AMENDMENT 2's exit B (*"proximity contributes a non-trivial
share of ΔR1"*) was **unmeasurable as written**. Closed post-hoc from the saved
`ckpt_after.pt`.

## 3. ⭐ THE FINDING: the reachable directions do not contain "avoid obstacles"

> ⚠️ **§3 IS REFINED BY ADDENDUM 2 (below), MEASURED after this was written.**
> The claim *"the policy cannot reach it"* holds **under the trust region**
> (0.338 m = 9 % of the gap) but is **REFUTED unanchored** (2.169 m = 59 %). The
> finding is a **tension between reach and drift**, not a wall. Read §3 with the
> addendum.

Put the whole campaign's arms side by side and one fact organises all of them:

* **Without the anchor (w=0)** the policy moves a lot — but the *only* direction
  it finds is **`feasibility` +0.1101** (87 % of the gain), bought with **+44 %
  ADE**. "Tame the fan."
* **With the anchor (w≥1)** that direction is closed, and the policy finds
  **nothing else**: every component flat or negative, in both the no-proximity
  and proximity arms.
* **Adding a continuous, well-shaped, demonstrably live scene-grounded term
  changed nothing.** `proximity` applies on **72.2 %** of real windows with full
  spread across the fan (A0-style coverage check, independent of training), has
  real headroom (cold-start mean **−0.2502** on a [−1, 0] range) — and the
  trained policy moved it by **0.0004**.

⇒ **The problem is not that the reward cannot rank safety. It is that the POLICY
CANNOT REACH IT.** The decoder emits small offsets from a fixed anchor
vocabulary; under a trust region those offsets can barely move, and even
unconstrained they move toward blandness rather than toward repositioning
candidates out of near-miss bands. **Ranking is useless if the action space
cannot act on the ranking.**

⚠️ **Three explanations ruled out by the campaign's own arms**, which is what
makes this a conclusion rather than a guess:
1. *Not the estimator* — at w=0 the policy moved decisively (TRAIN-C3's fix is
   working; M-B9 stays deferred on this evidence).
2. *Not the TRAINING step budget* — the same 2,000 optimiser steps produced a
   large move at w=0. ⚠️ **This is a different quantity from the DENOISE step
   budget**, which Addendum 2 shows *does* bind: reach scales 1.68 → 3.19 m over
   steps 0 → 8. Training steps are ruled out; inference steps are not.
3. *Not an inert reward term* — proximity's coverage was measured independently
   at 72.2 % with full spread **before** these numbers existed.

## 4. ⚠️ A pre-registration defect I have to own

AMENDMENT 2's **exit B was not measurable with the readout as built.** It asked
whether *"proximity contributes a non-trivial share of ΔR1"*, but `R1` is
computed with the DEFAULT reward spec precisely so arms stay comparable — so the
new term is excluded from it by design. I wrote an exit condition against a
quantity my own instrument does not produce.

It was recoverable only because **`ckpt_after.pt` exists**, and that exists only
because TRAIN-C5 forced checkpoint saving three hours earlier. Without it this
readout would have been unresolvable without re-training.

**The rule:** ⛔ **a pre-registered exit condition must name a quantity the
readout actually emits.** Check the exit against the instrument's output schema
when the amendment is written, not when it is read.

## 5. What is established

✅ The **anchor** is correct and effective (previous result; unchanged here).
✅ The **readout** is sound — deterministic, paired, with a regression control
that separates on both axes.
✅ The **barrier** is well-shaped and live on real data — it simply is not
reachable by this policy.
⛔ **The DDv2 mechanism does not transfer to REF-C v2.1 under this action space.**
Three configurations of the reward, four anchor strengths, and no configuration
improved fan safety.

⛔ **NOT established:** anything about refcv3 (different architecture), anything
about driving (T0 only), and anything about DDv2's method *on its own stack* —
they reshape a fan we hold nearly fixed.

## 6. Next lever — the FAN, per the committed exit

⛔ **Do NOT iterate the reward further.** The committed consequence is that the
lever is the fan's diversity and support. Concretely, in priority order:

| # | item |
|---|---|
| 1 | **Measure the reachable set.** How far can the decoder's offsets actually move a candidate at `decoder_steps=2`? If the answer is ≪ the near-miss band width (~metres), no reward can fix this and the finding is structural. This is cheap, 0 training, and it is the first thing to do. |
| 2 | **Anchor vocabulary as the lever.** DDv2 re-clusters anchors; ours are FPS over a synthetic unicycle pool. A vocabulary that covers evasive manoeuvres would give the policy somewhere safe to move *to*. |
| 3 | **Only then** revisit RL — with a fan the policy can actually reshape. |
| 4 | M-B9 stays deferred; §3.1 rules the estimator out on this evidence. |

---

## ⭐ ADDENDUM 2 — THE REACHABLE SET, MEASURED. §3's claim is REFINED, and partly REFUTED.

`EXIT C item 1, run 2026-08-29. Reported against the gap it must close, and at
several denoise budgets (both Master Mind design notes).`

### The three quantities

| quantity | value | vs the gap |
|---|---|---|
| **GAP** — extra clearance a near-miss candidate needs to reach `d_safe`=5 m (3,696 candidates) | **median 3.669 m** (mean 3.398, p25 2.437) | — |
| **REACH-CEILING** — fan displacement from the raw anchor vocabulary @ steps=2 | **2.084 m** (2 s endpoint **3.076 m**) | 57 % |
| **REACH-TRAINED** @ w=1 — how far 2,000 RL steps actually moved the fan | **0.338 m** | **9.2 %** |
| REACH-TRAINED @ w=10 | 0.163 m | 4.4 % |
| **REACH-TRAINED @ w=0 (unanchored)** | **2.169 m** | **59 %** |

### ⛔ §3 SAID "THE POLICY CANNOT REACH IT". THAT IS TRUE ONLY UNDER THE ANCHOR.

Unanchored, the policy moved the fan **2.169 m** — essentially the whole ceiling,
and **59 % of the median gap**. It is not immobile; it is *mobile in the wrong
direction*. Under the trust region it moves **0.338 m**, 9 % of the gap, which is
genuinely too little to clear a near miss.

⇒ **The real finding is a TENSION, not a wall:** the anchor strength that stops
the drift (w ≥ 1) also cuts reach to under a tenth of what avoidance requires.
Every arm in this campaign chose between *"moves, but toward blandness"* and
*"barely moves at all"*. That is a **design axis with an identified trade-off**,
not a structural dead end — a materially more hopeful and more actionable
statement than the one §3 published, and the measurement is what separates them.

### ⭐ AND REACH SCALES WITH THE DENOISE BUDGET — so "dead" was budget-specific

| denoise steps | mean displacement | at the 2 s endpoint |
|---|---|---|
| 0 (classifier only) | 1.681 m | 1.793 m |
| **2 (deployed)** | 2.084 m | 3.076 m |
| 4 | 2.439 m | 4.140 m |
| **8** | **3.187 m** | **6.077 m** |

At the 2 s endpoint — where avoidance actually happens — **steps=8 reaches
6.08 m, which EXCEEDS the median gap of 3.67 m**, while the deployed steps=2
reaches 3.08 m and falls just short of it.

⇒ Per design note (b): *"structurally dead"* becomes **"dead at this inference
budget"**, which is different and fixable. ⚠️ With a real cost: DDv2 runs 2 steps
deliberately (45 FPS end-to-end), so more steps buys reach with latency, and that
is a P6/TanitDeploy trade rather than a free win.

### ⚠️ A measurement error I made and caught

The first run of this probe measured `out["offset"]` and reported reach
**identical at steps 0/2/4/8**. That is impossible for a quantity the diffusion
loop refines — and it was the tell. `out["offset"]` is the **classifier-pass**
offset (`refc.py:1364-1365`); the loop accumulates into `x`, returned as
`anchor_traj` (`:1394-1399`, `:1515`). I had measured a quantity that *cannot*
depend on `steps`, and its flatness across four budgets is what exposed it.

⭐ **Design note (b) is what caught it.** Had I measured only the deployed
steps=2, the wrong number would have looked entirely plausible and would have
supported a "structurally dead" conclusion that the corrected data refutes. A
sweep over a parameter you expect to matter is a control, not just extra coverage.

### Revised next lever

1. ⭐ **The trade-off is now the object of study, not the fan alone.** The
   question is whether ANY (`w_anchor`, denoise-steps) pair gives reach ≳ the gap
   *and* holds drift. The campaign never tested reach and anchor together.
2. **Anchor vocabulary still matters** — 57 % ceiling at the deployed budget means
   even a perfect policy starts short — but it is no longer the only lever.
3. ⚠️ Any step-count increase is a **latency** decision (P6), not ours alone.

---

## ⛔ ADDENDUM 3 — THE EXTRA REACH IS NOT USABLE. Addendum 2's escape hatch is CLOSED.

`Zero training. Cold model, same 120 windows, same DEFAULT spec, eval-mode
deterministic readout, paired episode-cluster bootstrap 4000 reps.`

Addendum 2 found reach scaling with the denoise budget and offered *"structurally
dead"* → *"dead at this inference budget"*. ⭐ **That hypothesis had to be tested
before it could be used as a lever, because a decoder can move further by moving
WORSE.** It does.

| `decoder_steps` | R1 reward | R2 fan collision | **R3 sel-ADE** | ΔR3 vs deployed (paired) |
|---|---|---|---|---|
| **2 (deployed, = the model's configured `diffusion_steps`)** | **+0.8918** | 10.35 % | **0.654 m** | — |
| 4 | +0.8361 | 10.43 % | 0.789 m | +0.135 [−0.008, +0.270] not sep |
| **8** | +0.8365 | 10.91 % | **1.529 m** | **+0.875 [+0.647, +1.096] SEPARATED** |
| **16** | +0.8206 | **12.13 %** | **3.319 m** | **+2.664 [+2.321, +3.025] SEPARATED** |

⛔ **Every axis moves the wrong way together.** ADE **+134 %** at steps=8 and
**+407 %** at steps=16; collision rate **rises** (+0.56 pp, +1.78 pp); composed
reward **falls**. The fan is not reaching further toward anything useful — **it is
diverging.**

**Mechanism, from source not speculation:** `refc.py:309` sets
`diffusion_steps: int = 2`, and every instantiated config (`:661`, `:682`, `:908`)
passes 2. Two steps is the **trained operating point**, not merely a latency
choice. Iterating the offset head beyond what it was trained to compose walks the
trajectory off-distribution — the extra displacement Addendum 2 measured is
exactly that walk.

⇒ ⛔ **Do not raise `decoder_steps` as a reach lever, and do not re-run any RL arm
at a larger budget.** Addendum 2's *"dead at this inference budget"* reading is
**withdrawn on measurement**: at every budget where the fan is still usable, reach
is ≈2.1 m against a 3.67 m gap.

⚠️ **What this does NOT withdraw:** Addendum 2's other half stands — unanchored,
the policy moves **2.169 m (59 % of the gap)**, so §3's *"cannot reach it"* remains
refuted *as a statement about the policy's mobility*. The tension is real and
unchanged: **reach and drift are the same degree of freedom**, and no setting of
the denoise budget separates them.

⭐ **The value of this measurement is that it cost nothing and closed a branch.**
A plausible, well-evidenced lever appeared at 17:00 and was dead by 17:20, for
zero GPU-hours of training — because the reach hypothesis was written down as
something falsifiable rather than adopted as a plan. *(And it is the honest
sequel to TRAIN-C7: the sweep that caught the wrong tensor also produced a
hypothesis, and a hypothesis is not a finding until its consequence is measured.)*

### The lever that survives

**Anchor vocabulary** (item 2, unchanged). Both addenda point at it: reach at the
usable budget is 2.08 m of the 3.67 m gap (57 %), the policy's own movement is
capped there, and the only remaining way to give it somewhere safe to move *to* is
to change **where the anchors are** — DDv2 re-clusters them from data; ours are FPS
over a synthetic unicycle pool.
