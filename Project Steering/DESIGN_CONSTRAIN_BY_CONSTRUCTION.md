# DESIGN — Constrain the candidate set by CONSTRUCTION, do not penalise it after the fact

**Master Mind, 2026-09-05.** Status: **DESIGN + a cheap discriminating test.** ⛔ Nothing here is a
result; every number cited is MEASURED today and named with its source.

---

## 1. The pattern nobody named

Five defects were measured today, in three different arms, by four different streams. They were
reported as five problems. **They are one problem.**

| # | defect | MEASURED |
|---|---|---|
| 1 | refav1's **lateral vocabulary** cannot express a normal road curve | commands κ ∈ {0, 0.08} only; corpus median |κ| **0.00085** (R 1176 m); **38.7 %** of turns expressible |
| 2 | refav1's **longitudinal vocabulary** cannot express "keep accelerating" | **29 of 40** windows decode `ADAPT_SPEED_FOR_CURVE`, whose canonical control is **`a == 0`** |
| 3 | refcv3's **offset head** emits paths the vocabulary never contained | anchors drivable at **0.48 g**, decode emits **4.11 g** — **8.56×** |
| 4 | refcv3's **generator** emits colliding trajectories | **764 / 30,720 = 2.487 %** of candidates; worst window **104 of 128 = 81 %** |
| 5 | refav1's **shipped cost** has the do-nothing plan as its optimum | control tensor **exactly zero on 40/40**, reached **by search** |

⭐ **Every one of them is a defect in the CANDIDATE SET, present before any selection or any
learning happens.** Either the set cannot contain the right answer (1, 2), or it is full of answers
that must never be chosen (3, 4), or its scoring makes the empty answer best (5).

⇒ **The programme has been repairing SELECTION and LEARNING for a defect that lives in GENERATION.**

## 2. What the evidence says about which repair works

Today ran the comparison by accident, three ways, on the same fan:

| repair | mechanism | result |
|---|---|---|
| ⭐ **make bad UNREPRESENTABLE** (`feasible_decode`) | project onto the feasible set; a violating path cannot be expressed | **envelope 0.8879 → 0.0000**, **96.87 %** of the friction gap, T1 cost **+0.0012 m** |
| **penalise bad** (RL veto + reward) | down-weight violating candidates | **~2.7 %** of the gap, ADE **+0.0362 separated worse**, and it **partially cancels** with the gate |
| **filter bad** (top-2 kinematic gate) | re-rank what was already emitted | `sel_envelope` **−31 %**, ADE not separated — real, but bounded by what the generator produced |

⭐⭐ **And the penalty has a failure mode the projection does not: a penalty strong enough to prevent
bad behaviour also prevents GOOD behaviour.** MEASURED on the `W_KAPPA` ladder — plans identical to
`ha0` climb **3/40 → 11/40 → 10/40 → 17/40 → 40/40** as the curvature charge rises, and at the top
**both turn recalls hit 0.0**. A penalty cannot separate *"do not turn badly"* from *"do not turn"*.
A projection can: it deletes the infeasible turn and leaves the feasible one untouched.

⇒ **PRINCIPLE: make the bad UNREPRESENTABLE and the good REPRESENTABLE. A penalty does neither
cleanly — it trades them against each other.**

⚠️ **The honest counter-evidence, stated:** expanding representability alone did **not** pay either.
The L=3 vocabulary was approved on an oracle table and realised **2.3 %**, because *"turns goaled
correctly" is 0.2811 for every magnitude* — the head could not choose among the new tokens. ⇒ the
principle has two halves and **both are required**: a representable set the model can also *select
within*.

## 3. The concrete proposal: extend the projection from FRICTION to COLLISION

`feasible_decode` today projects onto the **friction-feasible** set — it inverts the scorer's own
finite-difference map, clamps to the box + Kamm disc, and re-integrates, so an over-friction path is
**unrepresentable**.

⇒ **Do the same for contact.** Project onto the **collision-free** set: a candidate that would
intersect an agent's swept volume is not down-weighted, not vetoed — **it cannot be emitted.**

**Why this is now possible and was not last week:** a collision constraint needs agent geometry,
where a friction constraint needs only the trajectory. **`obstacle.offline` reached the batch
today** (2,308 episodes / 12.1 M boxes, md5-verified), and refcv5's **agent-slot decoder** predicts
agents at inference — so the projection can run on **predicted** agents, not GT, which is what makes
it deployable rather than an oracle.

⚠️ **Three things that must be got right, from today's own failures:**
1. ⛔ **Swept segments, not waypoints.** The collision predicate tested sampled points until today;
   the undetected corridor was **9.381 m at p95** and the fix found **+37.8 %** more contacts. A
   projection built on the point predicate would leave the same hole.
2. ⛔ **Project in CONTROL space, re-integrate.** The friction projection works because it inverts
   the *exact* map the scorer uses; a projection in position space would produce paths the
   integrator cannot follow — the `off_reach` +0.2311 that `P2-C4` failed on is the warning.
3. ⚠️ **Predicted agents carry error.** The projection must take a *margin* that reflects detection
   uncertainty, and the margin is a parameter with a cost curve — not a constant to be guessed.

## 4. The cheapest discriminating test — 0 GPU, on banked data

⭐ **No training is needed to know whether this works.** The fan bank already holds every candidate
and its contact flag.

1. **Apply the collision projection offline** to the banked fan (`fan_bank_base_240w.npz`,
   240 × 128) using the val40 agent join.
2. **Committed outcomes, both written first:**
   * **SUPPORTED** — `fan_contact` **2.487 % → a structural 0.000 %**, and oracle-ADE at matched
     `fan_peak_g` moves by less than the **0.0163 m** replicate floor.
   * **REFUTED** — contact does not reach a structural zero (⇒ the constraint is not expressible in
     control space), **or** ADE degrades past the floor (⇒ it is a real trade and must be priced,
     not shipped).
3. ⛔ **Controls that must read known values:** the projection **disabled** must return the fan
   **bit-identically**; a window with **no agents** must be **untouched**; and the **19 windows that
   carry all 764 colliders** must be exactly the windows that change — *if a collider-free window
   moves, the projection is touching something other than collision.*

⇒ **If it passes, RL post-training for collision avoidance is unnecessary — the same way it turned
out to be unnecessary for friction.** ⚠️ **And if it fails, that is the strongest possible argument
FOR the RL arm**, because it would show the constraint cannot be imposed by construction and must be
learned. **Either outcome decides the RL question**, which is why this test comes first and costs
nothing.

## 5. What this predicts elsewhere — falsifiable, and cheap to check

* **refav1's longitudinal blocker (#2) should be fixed by REPRESENTABILITY, not by a penalty.** The
  lateral analogue's penalty (`W_KAPPA`) found an interior optimum and then drove the planner to
  silence; a longitudinal penalty should do the same. ⇒ the fix is a canonical control that is **not
  identically zero**, plus a head able to select it — **both halves**, per §2.
* **v7f inherits the principle, not the code.** Its decoder and vocabulary are its own, so
  `feasible_decode` transfers as a **design**, not as a module — and the check is whether v7f's fan
  has the same gap between what its vocabulary can drive and what its decode emits. ⛔ That number
  does not exist for v7f and should.

⚠️ **This document is a hypothesis with a test attached, not a finding.** It is written down before
the test so it can fail.

---

# ADDENDUM 1 (same day) — THE PRINCIPLE IS THREE-PART, NOT TWO

⛔ **The body above says *make the bad unrepresentable and the good representable*. That is
INCOMPLETE, and two arms measured after it was written show why.**

| part | mechanism | evidence |
|---|---|---|
| **1. make the bad UNREPRESENTABLE** | a constraint with units | Kamm cap `peak_g` max **3.262 → 0.707 = μ**, free on the longitudinal family, turn decisions **bit-identical**; the friction projection closed **96.87 %** of the gap at **1.2 mm** |
| **2. make the good REPRESENTABLE** | vocabulary / candidate set | ⛔ **ALONE IT DOES NOTHING.** L=3 realised **2.3 %** (no chooser). `l3ladder` added rungs at R 500–125 m and **not one window of 40 realised one** — its spine is bit-for-bit the baseline's |
| **3. ⭐ GIVE THE SEARCH A REASON TO PREFER IT** | cost / preference | `W_KAPPA` alone produced curvatures **0.0115–0.0530 — exactly the band the rungs occupy — with no ladder at all** |

⭐⭐ **The mechanism, confirmed from BOTH directions:** supply the candidates without the
preference and nothing is picked; supply the preference without the candidates and the search
**synthesises** them. ⇒ `D-REFAV1-DRIVE-GATE`'s gate is **not a wall around the reachable set — it
is an ABSENCE OF PREFERENCE.**

⭐ **Operational consequence:** part **3 can SUBSTITUTE for part 2** whenever the search can
synthesise what it needs. A wider vocabulary is only worth its cost when the search **cannot reach**
the band — a cheap test that had never been run. ⚠️ And a wider vocabulary is **not free**:
`l3ladder` cost `lane_keep` recall **0.7143 → 0.5714** and goal FDE **2.9639 → 3.3200**.

⛔ **And parts do not compose arbitrarily: a CONSTRAINT cannot supply a PREFERENCE.** A cap can
forbid a curvature but cannot make a rung attractive ⇒ *ladder + cap* is the **less** informative
pairing and *ladder + `W_KAPPA`* is the informative one.

⇒ **The prediction this makes for the open blocker:** refav1's longitudinal gap has
`ADAPT_SPEED_FOR_CURVE`'s canonical control at `a == 0`, so **part 2 is missing AND part 3 has
nothing to prefer.** ⛔ **Fixing either alone will null** — exactly as the lateral side demonstrated
twice today. Written before the longitudinal arm runs, so it can fail.
