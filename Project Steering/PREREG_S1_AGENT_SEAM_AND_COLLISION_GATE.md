# PRE-REGISTRATION — the agent seam and the collision gate (`H-SEL-GATE-1`)

**Written 2026-09-17, BEFORE any arm has been run.** Both outcomes committed below.
PI decision the same day on `PI_DECISION_QUEUE` **item 19**: *"for 3 chose defult"* —
option (a), **carry it as a named arm with its own pre-registration**. This is that file.

**Status: PRE-REGISTERED, NOT RUN.** Nothing here is a result.

---

## 1. The measured headroom this exists to reach

MEASURED at zero GPU on 493 held-out windows (item 19):

| | |
|---|---|
| windows where the **selected** plan collides (untouched `refcv5-v2`) | **28 / 493** |
| …of which the fan **still held a collision-free candidate** | **28 / 28 = 100 %** |
| mean share of the fan that was collision-free there | **55.3 %** |
| oracle-repair ceiling from fixing collision-selection **alone** | **+0.0485 `sel_pdms` [+0.0228, +0.0801] = 62.4 % of the entire oracle gap** |
| windows it covers | **25 / 493 = 5.1 %** |

⭐ Reproduces on **four checkpoints in three training states** (`base` 28/28,
`base_repeat` 28/28, `L1-NORL-s0` 25/25, `L1-RL-s0` 37/37) ⇒ a **`refcv5-v2` property**,
not an artifact of the RL experiment that surfaced it.

---

## 2. ⛔ The three things this pre-registration must not let anyone forget

1. ⛔ **The 62.4 % is a T0 ORACLE CEILING, not an available gain.** `sel_nc` is scored
   against the **recorded future**. At inference the vision-only rule forbids that, so a
   deployable gate needs a **PREDICTED** occupancy. **The deliverable is what the
   predicted gate buys; the oracle arm only says how much room there is.**
2. ⛔ **An oracle gap needs a RANDOM control.** *(Standing rule, cost a landed claim
   2026-09-17: best − actual is large for a GOOD selector too.)* Random, actual and
   oracle are quoted **on the same windows**, or none of them is quoted.
3. ⛔ **Two levers, not one.** The agent channel is a **representation** change; the
   collision gate is a **hard constraint at selection time**. Repairing both in one arm
   makes a PASS non-attributable — the `--v2` conflation precedent (ten levers on two
   axes) and the C6 confound.

---

## 3. Hypothesis

**`H-SEL-GATE-1`** — *A hard selection constraint — never select a colliding candidate
while a collision-free one is in the fan — driven by a **predicted** occupancy from the
refcv6 perception heads, recovers a measurable share of the collision-selection gap at
T1, without harming the other three metric families.*

**`H-SEL-AGENT-1`** — *Re-opening the structured agent channel (`--agents on`) improves
collision-relevant selection, independently of the gate.*

⚠️ `H-SEL-AGENT-1` is **an arm whose tiny-rig exclusion may or may not generalise**. The
2-seed tiny-rig result that gated `--agents off` was about an **auxiliary training task**;
this is about a **selection constraint**. The two do not meet, and neither overturns the
other.

---

## 4. Arms — ⛔ ONE VARIABLE EACH

| arm | the ONE change | what it tests |
|---|---|---|
| **`S1-BASE`** | none (refcv6 baseline, `--agents off`, no gate) | the base every delta is read against |
| **`S1-AGENTS`** | `--agents on` | `H-SEL-AGENT-1`: the representation lever alone |
| **`S1-GATE-PRED`** | collision gate on **predicted** occupancy (BEV map + 3-D box heads) | `H-SEL-GATE-1`: **THE DELIVERABLE** |
| **`S1-GATE-ORACLE`** | collision gate on the **recorded future** | ⛔ **T0 ONLY — the CEILING.** Never a capability claim, never quoted as driving performance |
| **`S1-RANDOM`** | selection replaced by a uniform draw from the fan | ⛔ **the no-information control.** Without it the oracle gap is unreadable |
| **`S1-GATE-CONST`** | the gate wired to a **constant** occupancy (everything free) | ⛔ **the deliberate-regression arm** — see §6 |
| **`S1-BOTH`** | `--agents on` **and** the predicted gate | only if a single-lever arm clears its bar |
| **`S1-REPL`** | `S1-BASE`'s flags, **different seed, zero levers moved** | ⛔ the run-to-run floor |

**Held constant:** corpus, split, steps, batch, trunk (`resnet34` for the matrix;
`resnet101` when a card allows), **416 × 1024**, anchors, `n_anchors` 117, the fan, the T1
harness, the held-out episodes. Every arm runs **two seeds**.

⛔ **Preflight refuses on an ARGV DIFF, not on intent.**

---

## 5. Committed criteria

### 5.1 PRIMARY — `H-SEL-GATE-1` (T1)

* **SUPPORTED** — `S1-GATE-PRED` reduces the **collided-selection rate** vs `S1-BASE`
  with a paired episode-cluster bootstrap CI excluding zero, **on both seeds**, by more
  than the `S1-REPL` floor — **and** `ade_m` and the other three families are not worse
  than `S1-BASE` by more than that floor.
* ⛔ **FAIL-HARM** — any family is worse than `S1-BASE` by more than the floor on either
  seed. A gate that fixes collisions by driving worse everywhere else has not helped.
* **REFUTED** — the collided-selection rate does not move beyond the floor.
* **INCONCLUSIVE** — `S1-REPL` shows the rig noisier than the effect.

### 5.2 The ceiling, reported but never claimed

`S1-GATE-ORACLE` is reported **beside** `S1-GATE-PRED` as *"what a perfect collision
checker would have bought on these windows"*, stamped **T0**, with `S1-RANDOM` and
`S1-BASE` on the **same windows**. The quotable number is
`(pred − base) / (oracle − base)` — **the share of the ceiling actually reached.**

### 5.3 `H-SEL-AGENT-1`

**SUPPORTED** only if `S1-AGENTS` improves the collision-relevant selection statistic
over `S1-BASE`, both seeds, beyond the floor. ⛔ It is **not** supported by an ADE move:
item 19's whole point is that this seam is pre-registered **against the collision
statistic, not against ADE**.

---

## 6. ⛔ The deliberate-regression arm

`S1-GATE-CONST` runs the identical gate with an occupancy that says **everything is
free**. The gate then reorders nothing and **must recover nothing**.

If `S1-GATE-CONST` recovers as much as `S1-GATE-PRED`, then the gain is coming from the
**re-ranking machinery**, not from the occupancy signal, and a PASS on `S1-GATE-PRED`
means nothing. *(A gate that does not FAIL the reintroduced defect cannot certify the
fix.)*

---

## 7. Controls that must read known values

| control | what it must read |
|---|---|
| **`S1-RANDOM`** | the no-information value of the selection statistic. ⛔ Quoted on the **same windows** as actual and oracle, or none is quoted |
| **`S1-GATE-CONST`** | **exactly zero** recovery |
| **`S1-REPL`** | the run-to-run floor; an effect smaller than it is not an effect |
| **the fan's own collision-free share** | ≈ **55.3 %** on the affected windows, as MEASURED — if a run reports a wildly different share, the fan or the collision checker changed and no arm is readable |

---

## 8. Dependencies and the gate on starting

* ⛔ **`S1-GATE-PRED` cannot start before a predicted occupancy exists.** That is what the
  BEV map head and the 3-D box head are for; both are now wired (R2/R3) and the chain
  trains end to end at 416 × 1024, but **no arm has yet shown the predicted occupancy is
  good enough to gate on**. The first work item is therefore a **read of the predicted
  occupancy's quality against the SAM3 map GT** — not a gate arm.
* `S1-GATE-ORACLE` and `S1-RANDOM` need **no** trained perception and can run first: they
  bound the problem and make everything after them readable.

---

## 9. Tier and reporting

**T1 primary.** `S1-GATE-ORACLE` is **T0 and stamped as such.** Every number carries its
evidence class, tier, `n` and estimator (`paired_episode_cluster_bootstrap`). All four
metric families — longitudinal, lateral, tactical, strategic — **in addition** to ADE,
per family, never pooled.

---

## 10. What would make me abandon this direction

Committed in advance: if `S1-GATE-ORACLE` reproduces the **+0.0485** ceiling but
`S1-GATE-PRED` recovers **less than 10 %** of it on both seeds, then the headroom is real
and the **predicted occupancy is not good enough to reach it**. That is reported as a
refutation of `H-SEL-GATE-1` *as configured*, and the next work is **perception quality**
— the occupancy head's own accuracy against SAM3 map GT — not a better gate.
