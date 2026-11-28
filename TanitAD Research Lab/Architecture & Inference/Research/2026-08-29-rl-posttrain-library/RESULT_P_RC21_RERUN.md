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

---

## ⭐⭐ ADDENDUM 4 — THE SAFETY TARGET IS MIS-CALIBRATED. Not the policy, not the vocabulary.

`Zero training. Two probes: anchor-vocabulary coverage (n=32 near-miss windows of
90) and target calibration on real agent TRACKS (n=71 windows with a followable
track). Evidence class: MEASURED (ours).`

### The number that reorganises the campaign

**On 45.1 % of all windows carrying a visible lead track, the HUMAN DRIVER'S OWN
FUTURE is inside `d_safe = 5.0 m`.**

| clearance threshold | human clears it (obstacles STATIC, as scored) | human clears it (real TRACKS) |
|---|---|---|
| 2 m | 77.5 % | 84.5 % |
| 3 m | 71.8 % | 76.1 % |
| **5 m (our `d_safe`)** | **54.9 %** | **56.3 %** |

⇒ The proximity barrier fires **against the demonstration distribution on nearly
half of all windows**. It is not a safety threshold in this corpus; it is a
threshold that marks ordinary competent driving as unsafe.

### And the vocabulary is NOT the constraint — the opposite of what I expected

On near-miss windows, coverage of a candidate clearing `d_safe` (N=128 anchors,
pool n=90, S=4, d=8):

| vocabulary | clears `d_safe` | best clearance |
|---|---|---|
| **CURRENT (the model's own bank)** | **87.5 %** | **8.586 m** |
| CLUSTERED (k-means on real futures) | 68.8 % | 6.482 m |
| RANDOM (same pool, no clustering) | **68.8 %** | **6.482 m** |

⛔ **Item 2 is dead, and its own floor control killed it.** CLUSTERED and RANDOM
agree **to three decimals on both columns** — re-clustering buys *nothing* over
random draws from the same pool, exactly the outcome the Master Mind's control was
built to detect. And both are **worse than the bank we already have**, because the
synthetic unicycle pool contains extreme manoeuvres real futures do not.

⇒ **The existing vocabulary already contains "solutions" on 87.5 % of near-miss
windows. The policy declines to take them — correctly, because they are evasive
manoeuvres the human never performs.** The trust region pulls toward the
demonstration distribution; the proximity barrier pushes away from it; and since
the barrier's threshold is set where the human lives, the two are in **direct
opposition by construction**. That is the tension Addendum 2 named, now with a cause.

### ⚠️ TWO ERRORS OF MINE INSIDE THIS ADDENDUM, BOTH CAUGHT BEFORE PUBLICATION

**(1) My oracle control was CIRCULAR, and its headline number is WITHDRAWN.**
I selected near-miss windows by the condition *"the human's clearance < `d_safe`"*
and then reported *"the human clears `d_safe` in 0.0 % of them"*. That 0.0 % is
**definitional, not measured** — a control cannot be evaluated on a subset defined
by the control's own outcome. ⛔ Do not quote it. The non-circular version is the
calibration table above, computed over **all** windows with a followable track,
and it carries the finding on its own. *(The coverage comparison is unaffected:
the vocabularies were never used to select the windows.)*

**(2) My proposed mechanism was WRONG, and the measurement refutes it.**
`rewards.py:240-241` states obstacles are "a static snapshot", so I predicted the
violations were an artifact of scoring the ego's 2 s **future** against agents
frozen at **t0** — car-following would then look like closing to zero every time.
I built the moving-obstacle version from the join's `track_id` field, propagating
each track to its own future position and mapping it back into the t0 ego frame.
**The gap is 54.9 % → 56.3 %: +1.4 pp.** The snapshot artifact is real and
negligible; it explains **none** of the effect. The threshold is simply too large.

⭐ I was one write-up away from publishing a mechanism that is not the mechanism.
It cost one probe to find out, and the probe existed only because the field needed
to test it was already in the data.

### What to do instead — calibrate the barrier to the demonstrations

Set `d_safe` at a **percentile of the human's own clearance distribution** rather
than at a round number. From the tracks-based column: p5 **0.787 m**, p25
**3.245 m**, p50 **5.916 m**. A barrier near **2 m** fires on **15.5 %** of
windows — rare, and rare in the places where clearance is genuinely unusual —
instead of on 45 %. ⚠️ This is a **specification** change, so it needs its own
pre-registration with both outcomes committed; it is not a knob to turn inside an
existing arm.

⛔ **What is NOT claimed:** that a recalibrated barrier will make RL work here.
Every prior null was measured against the mis-calibrated target, so they are
uninformative about the recalibrated one — but "uninformative" is not "promising".
The campaign's exit stands until a pre-registered arm says otherwise.

---

## ✅ ADDENDUM 5 — THE MISCALIBRATION IS CONFINED TO ONE CONSTANT. Our eval instruments are FINE.

`Zero training. n=120 windows (43 with an in-lane lead). Every constant tested
against the population meeting ITS OWN applicability gate — never a subset
selected by the outcome (TRAIN-C8). Evidence class: MEASURED (ours).`

**The question** (Master Mind extension): Addendum 4 found a reward constant that
flags competent human driving at a high rate. Is that a *shape of error* our EVAL
instruments share? If so, every "our arm violates X on Y % of windows" number we
have published inherits it.

**The answer is no.** Nine of ten constants are well-calibrated, including every
one that was self-labelled PROPOSED.

| constant | value | n | **flags the human** | human p50 | source |
|---|---|---|---|---|---|
| **`proximity_safe_m`** | 5.0 | 43 | **⛔ 34.9 %** | 8.472 m | `rl/rewards.py:388` |
| `target_time_gap_s` (T*) | 2.0 | 34 | ⚠️ 11.8 % | 2.992 s | `rl/rewards.py:294` |
| `ttc_min_s` (hard VETO) | 1.5 | 34 | 5.9 % | 2.992 s | `rl/rewards.py:329` |
| `kappa_max_1pm` | 0.2 | 120 | 4.2 % | 0.022 | `rl/rewards.py:76` |
| `jerk_max_mps3` | 8.0 | 120 | 3.3 % | 1.771 | `rl/rewards.py:78` |
| `pdm_a_lon_max_mps2` | 3.0 | 120 | 3.3 % | 0.669 | `pseudosim.py:803` **PROPOSED** |
| `pdm_a_lat_max_mps2` | 3.0 | 120 | 1.7 % | 0.634 | `pseudosim.py:803` **PROPOSED** |
| `a_max_mps2` | 4.0 | 120 | 0.8 % | 0.669 | `rl/rewards.py:75` |
| `pdm_yaw_rate_max_radps` | 0.95 | 120 | 0.8 % | 0.113 | `pseudosim.py:804` **PROPOSED** |
| `lat_acc_max_mps2` | 4.0 | 120 | **0.0 %** | 0.634 | `rl/rewards.py:77` |

⭐ **The `PROPOSED` comfort constants are the headline non-finding.** They were the
prime suspects — round numbers, imported from nuPlan/NAVSIM, with the source
docstring conceding *"their exact constants are not quotable"*. Measured against
this corpus they flag the human at **0.8–3.3 %**. They are fine, and the worry that
motivated this sweep is **refuted for them.**

✅ **And it corroborates an instrument decision made for a different reason.**
`pseudosim` drops `comfort` from the composite via its discriminative-range gate
(`COMPONENT_WEIGHTS` carries 0.0). This sweep shows *why* that is correct: the
human sits at p95 **2.13 / 2.31 / 6.78** against limits of **3.0 / 3.0 / 8.0**, so
nothing in the neighbourhood of real driving approaches them and the term cannot
discriminate. **The range gate caught a non-binding term without knowing it was
non-binding** — the instrument behaved correctly on evidence it did not have.

### ⚠️ Two honesty notes on the numbers above

**(1) `proximity_safe_m` reads 34.9 % here and 45.1 % in Addendum 4. Both are
right; they describe DIFFERENT POPULATIONS.** Addendum 4 gated on the join's own
filter (any agent ahead within `LEAD_MAX_GAP_M`); this table adds the in-lane
lateral filter (`|cy| ≤ 2 m`), which is the population the *lead* metrics use. ⛔
Neither number supersedes the other and neither may be quoted without its gate.
The conclusion is unchanged at either value: a third to a half of competent human
driving is flagged.

**(2) `target_time_gap_s = 2.0` is not mis-calibrated in the same direction — it is
arguably too AGGRESSIVE.** The human's median time gap is **2.992 s**, well
*above* our target, so `T*` rewards following **closer** than humans actually
drive. At 11.8 % flagged it is not urgent, and it is a graded peak rather than a
barrier, so it shapes rather than vetoes. ⚠️ Logged as a work item, not a defect:
the fix would be to set `T*` from the demonstration median, and that is a
specification change requiring its own both-outcomes prereg.

### The scope this fixes

⛔ **Do NOT escalate an eval-doctrine item to the PI.** The sweep was built to find
one and found the opposite: the four-families instruments largely **report** raw
quantities (`lead_metrics.distance_keeping` emits headway/time-gap/TTC as values,
not verdicts), and the thresholds that do exist are sound. **The miscalibration is
one constant in one reward term, and `PREREG_D_SAFE_CAL.md` already covers it.**

⭐ **A sweep that refutes its own motivating hypothesis is worth as much as one
that confirms it** — this one bounded the blast radius of Addendum 4 to a single
line of code, and it did so before anyone re-derived a published eval number.

---

## ⭐ ADDENDUM 6 — R5 LANDS, AND ITS FIRST READING CLOSES THE ARGUMENT

`The field PREREG_D_SAFE_CAL is gated on now exists, with tests, verified on the
real model. Its cold-start value was not part of the prereg and is reported here
as an observation, not an exit.`

**R5 = `R5_gt_clearance_violation_frac`** — the fraction of scored windows whose
**SELECTED** candidate (the trajectory that would actually be driven) is inside
`proximity_safe_m`. Distinct from R2, which is a property of all 128 fan
candidates; the gap between them is where reward hacking hides.

| | value |
|---|---|
| **REF-C v2.1 cold start, selected path** | **34.2 %** (n=15 episodes with obstacles) |
| **HUMAN driver, same gate** (Addendum 5) | **34.9 %** |
| model fan collision rate (R2) | 10.35 % |

⭐ **The model's driven path violates `d_safe = 5.0 m` at the same rate as the
human's — 34.2 % vs 34.9 %.** An imitation-trained planner reproducing the
demonstration distribution's clearance behaviour to within a point is the policy
working *correctly*. ⇒ At this threshold there was never a defect for the barrier
to remove: it was asking a well-imitating policy to stop imitating.

⚠️ **This is an observation, not a pre-registered exit, and it does not
pre-decide `D-SAFE-CAL`.** It is one arm on one corpus with 15 clusters and no
paired interval; the prereg's outcomes 1 and 2 both remain live. It is recorded
because it was measured while wiring the field, and because a number this close
would look like hindsight if it appeared after the arms ran.

### Implementation notes

* ⭐ **`clearance()` extracted to `rewards.py` as the ONE definition of "how
  close".** `_proximity` now shapes it into a barrier and R5 thresholds it;
  previously each recomputed it, which is how two notions of "close" drift apart
  and a report compares a barrier's with a readout's.
* ⛔ **Absence returns `nan`, never 0 and never +inf.** A window with no obstacles
  is UNDEFINED for clearance, and the readout drops it rather than scoring it
  clean — otherwise a thin obstacle join dilutes the violation rate toward zero
  and reports safety. That is TRAIN-C2's family (a statistic pooled where the
  quantity is undefined) and the E-DETECT-1 all-zero floor's.
* 9 new tests (`stack/tests/test_rl_clearance.py`), RL suite **149 passed**.
* ⚠️ **One test found a real documentation gap while failing:** the obstacle shape
  contract is `[B, 1, K, 2]`, and `[B, K, 2]` broadcasts correctly at `B == 1`
  while dying at `B > 1` — so the wrong shape survives a single-window smoke test.
  Now spelled out in the docstring and pinned by a test that asserts the loud
  failure.
