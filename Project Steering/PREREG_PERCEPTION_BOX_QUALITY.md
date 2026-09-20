# PRE-REGISTRATION — box-head localisation quality (`E-PERCEP-BOX-1`)

**Written 2026-09-20 by TanitAD_TrainingFlyWheel, at the Master Mind's request, BEFORE any arm is
run. Status: PRE-REGISTERED, NOT RUN. 0 GPU-hours. Nothing here is a result.**

This is the arm `PREREG_S1`'s own §10 demands. S1 named the next work as *perception quality*;
this file turns that into a **number, a set of one-variable arms, and an honest statement of which
of them this box can answer.** Every figure quoted below is MEASURED and carries its artifact:
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-s1-collision-gate/`
(`RESULT.md`, `raw/box_centre_curve.json`, `raw/box_axis_probe.json`, `raw/gt_extent_probe.json`,
`raw/verdict_A8.json`).

---

## 1. The target, taken from the measurement

The S1 gate is scored by matching predictions to agents at a **2 m centre distance**. A detector
whose centre error exceeds that threshold cannot gate, whatever its loss curve says.

| population | today (A8, 5,000 steps) | target |
|---|---|---|
| **near-forward** — `x ∈ [0, 60] m, |y| ≤ 16 m`, the population a collision gate ACTS on (n = 129 pairs) | `box3d_centre` **6.06 m** (\|dx\| 3.09, \|dy\| 2.97) | ⭐ **< 2 m** — the primary bar |
| **all matched pairs** — the 32-nearest, 360° set the loss trains on (n = 623) | **12.04 m** (\|dx\| 5.36, \|dy\| 6.68) | **< 2 m** — reported, secondary |
| train-side `box3d_centre`, last 500 steps | **10.32 m** = 5.16× the bar | — |

⛔ **Both populations are stated in every report, and the NEAR-FORWARD one is the primary**,
because it is the only one the gate consumes. Quoting the 360° number alone would understate the
head on the gate's own question by ~2×; quoting the near-forward number alone would overstate the
detector. The error is **isotropic** (x-share 0.445 overall, 0.51 near-forward), so no arm below
assumes a depth-specific defect.

**Metric, estimator, tier.** Primary: the held-out matched-pair centre error per population,
through the **training matcher** (`agent_slots.match_slots`, Hungarian over every valid target —
the 2 m greedy matcher pairs only lucky hits and flatters a weak head). Estimator: **episode-cluster
bootstrap** over windows, paired when two arms share windows; `n` windows and `n` episodes printed.
Tier: **T0** (labels only, no planner in the loop). Secondary, reported never substituted: BEV AP
at 2 m and the velocity MAE against its zero-velocity floor (today +2.01 m/s [+0.87, +3.03], a
**PASS** that must not regress).

---

## 2. The arms — ONE variable each

| arm | the ONE change | where it can run | why it is a candidate |
|---|---|---|---|
| **`P1-STEPS`** | 5,000 → **15,000** steps, A8's config and corpus unchanged | ⭐ **dev box**, ~33 h | the curve is still falling (15.11 → 10.42 m between the 1k–3k and 3k–5k blocks) |
| **`P2-CORPUS`** | 124 clips → the full parity train corpus, steps fixed at 5,000 | ⛔ **POD ONLY** | 62 halfA episodes is one corpus size; nothing here bounds the scale effect |
| **`P3-SUPERVISION`** | the target population: 32-nearest **360°** → **gate-relevant** (`x ∈ [0, 60], |y| ≤ 16`) | ⭐ **dev box**, ~11 h | **49 %** of GT is BEHIND the ego and only **22.3 %** lies in the decode's natural range; the head spends capacity where no gate reads it |
| **`P4-DECODE`** | the decode normalisation (`x_fwd_m` 60 / `y_half_m` 16) widened to the measured target spread | ⭐ **dev box**, ~11 h | targets span **±190 m** against a 60 × 32 m scale. ⚠️ The decode is a **SCALE, not a clamp**, so this arm tests **conditioning**, never representability |
| **`P5-TRUNK`** | `resnet34` → `resnet101` (⚠️ **INHERITED, not re-verified here:** `…/2026-09-18-resnet101-8gb-fit` reports it fits the 8 GiB card; this arm re-checks the fit before it trains) | ⭐ **dev box**, ~15–20 h | capacity, the axis A7 does not test (A7 is ImageNet-vs-random ON resnet34) |

⚠️ **Evidence class of the hour figures: ESTIMATED**, derived from A8's MEASURED 3.7 s/step and
~66 min per held-out read (its 5,000 steps took **11.15 h**, `summary.json` `wallclock_s` 40,111).
`P1`'s ~33 h assumes the eval cadence stays at every 1,000 steps — 15 reads is 16.5 h of it, so a
wider cadence is the cheapest lever on the COST, and it changes no criterion.

**Held constant across every arm:** the corpus and split (`eval124clean` halfA train / halfB
held-out), 416 × 1024, the window grid, `--agent-pad 32`, the join and its `with_rates` /
`with_track_ids` flags, the loss weights, and the eval windows (the trainer's own seed-12345
1,000, of which **736** are S1-eligible over 59 clusters). ⛔ **Preflight refuses on an ARGV DIFF,
not on intent.**

---

## 3. Committed criteria — both outcomes, per arm

* **`P1-STEPS`** — **SUPPORTED** if near-forward centre error falls **below 2 m** by 15,000 steps,
  with the paired CI against A8's 5,000-step read excluding zero and the gap exceeding the
  replicate floor (§3.1). **REFUTED** if it does not: then **steps at this corpus size are not the
  lever**, and `P2-CORPUS` becomes the only remaining scale explanation. ⚠️ Running to 15,000 is a
  **measurement**, not an extrapolation — the 2× bound forbids *predicting* past 10,000, not
  *measuring* there, and no step count is promised in advance (today's fits are R² **0.41** and
  **0.09**, so none is quotable).
* **`P3-SUPERVISION`** — **SUPPORTED** if the near-forward error improves beyond the floor **while
  the label-density change is reported** (targets/window drop from ~18.7 to ~4.6, MEASURED).
  **REFUTED** if it does not move, i.e. the 360° supervision was not costing the near field.
  ⛔ **FAIL-HARM** if the far/behind error explodes in a way that would break any future 360° use —
  reported, not hidden.
* **`P4-DECODE`** — **SUPPORTED** if the error improves beyond the floor with no other change.
  **REFUTED** otherwise, and then the normalisation was never the constraint (which the
  scale-not-clamp reading already makes the likelier outcome — stated in advance so a null is not
  a surprise).
* **`P5-TRUNK`** — **SUPPORTED** if `resnet101` improves the near-forward error beyond the floor at
  matched steps. **REFUTED** otherwise ⇒ capacity is not the constraint at this corpus size.
* **`P2-CORPUS`** — ⛔ **not decidable here.** Committed anyway so the pod request can carry it:
  **SUPPORTED** if the error at fixed 5,000 steps improves beyond the floor on the full corpus.

### 3.1 The floor, and the replicate that measures it

⛔ **`box3d_centre` has NO measured run-to-run floor today.** `H-ESTIM-SEED-1` binds: a separated
CI answers *"would another draw of EPISODES say this?"*, never *"would another training run?"*.
⇒ **`P0-REPLICATE` runs FIRST**: A8's exact flags, a different `--seed`, same steps. Its
arm-to-arm difference IS the floor, and **no arm above may be called SUPPORTED by a margin smaller
than it.** (The only free comparison we have is on a different metric — A8 vs A3 map IoU at 2k
reads 0.56906 vs 0.57607, a ~0.007 gap — which says nothing about this one.)

### 3.2 Controls that must read known values

| control | must read |
|---|---|
| label density | `n_matched == n_target` on **100 %** of rows (it is a min(targets, queries) identity with 32 ≤ 100; today 37.44 vs 37.44). Any arm changing the population re-reports it |
| untouched base | A8's config re-run reproduces its curve within the `P0-REPLICATE` floor |
| ⛔ deliberate regression for `P3` | supervision restricted to **far/behind only** must **NOT** improve the near-forward error. If it does, the arm is measuring something else and `P3` is void |
| velocity term | must keep beating the zero-velocity floor (today **+2.01 m/s** [+0.87, +3.03]); a regression is FAIL-HARM |
| the gate hand-back | when near-forward centre < 2 m, `PREREG_S1`'s `S1-GATE-PRED` becomes testable and is re-run **unchanged** — the S1 rig is already built, tested and mutation-proven |

---

## 4. ⛔ The honest frame — what this box can answer, and what it cannot

**The arithmetic.** A8 = 5,000 steps × batch 2 = **10,000 windows ≈ 0.95 of one halfA epoch**
(62 episodes, 10,600 held-out windows on the other half), at **~3.7 s/step + ~66 min per held-out
read**, i.e. **11.15 h** for its 5,000 steps. Today the head sits at **5.16×** the bar. The
prescribed matched-step ratio (**×0.6896 per 2,000 steps**, the fallback because no exponent is
quotable at R² 0.41 / 0.09) implies **~13,885 steps** to reach 2 m — **past the 2× extrapolation
bound of 10,000**, therefore inadmissible as a promise.

* ⭐ **The dev box CAN answer:** `P0-REPLICATE` (~11 h), `P3-SUPERVISION` (~11 h),
  `P4-DECODE` (~11 h), `P5-TRUNK` (~15–20 h) and `P1-STEPS` (~33 h at 15,000 steps). Together
  ≈ **3.2–3.6 days** of card time, serialised, one job at a time.
* ⛔ **The dev box CANNOT answer `P2-CORPUS`**, and no arithmetic here can stand in for it: 124
  clips is **one** corpus size, so a scale claim would have **no second point to rest on**. It
  needs the pod, and it is the only one of the five that does.
* ⛔ **Neither can this box promise that `P1` reaches the bar.** It can only measure whether it
  does. If `P1` and `P3`–`P5` all come back REFUTED, the honest programme sentence is *"at this
  corpus size the box head does not reach a 2 m centre error, and `S1-GATE-PRED` stays untestable
  until `P2` runs"* — which is exactly the case the pod request has to fund.

*(This paragraph is the one the pod request's §5 `D-S1-DEP-BOX` should carry, beside the existing
"cannot say how many steps" sentence.)*

---

## 5. What would make me abandon this direction

Committed in advance: if `P0`'s replicate floor turns out **larger than the gap between 6.06 m and
2 m**, then this rig cannot resolve the levers at all and the next work is the **rig**, not the
head. And if `P3` (supervision) alone closes most of the gap, then the defect was never detector
capacity but **what we asked it to detect** — and every future perception read must state its
target population before its number, exactly as `PREREG_S1` now does.

<!-- P3-CORRECTION-AND-AMENDMENT-2026-09-20 -->
## ⛔ CORRECTION + AMENDMENT to `P3-SUPERVISION`, 2026-09-20 — one number withdrawn, one alternative pre-registered

Both come from the TrainingFlyWheel's P3 pre-build, measured on the TRAIN cache before any arm ran.
⛔ The pre-registered P3 arm itself is **UNCHANGED**; this adds a correction, a named alternative
and a CONDITIONAL control. Nothing here was written after seeing an arm's result, because no arm
has been run.

### ⛔ 1. "~18.7 targets/window" is WITHDRAWN (line 75)

The line reads *"targets/window drop from ~18.7 to ~4.6, **MEASURED**"* and carries **no artifact
path**. **4.6 is right** (halfA gate-relevant **4.634**). **18.7 reproduces at NONE of the four
scopings actually measured**: raw 360° **31.13** · delivered-after-pad **18.95** · over all
windows **18.27** · halfB **19.79**. ⚠️ Its nearest neighbours are the *delivered-after-pad*
figures, which is the likely origin — and that is exactly the error, because **raw** and
**delivered after a pad-32 truncation** answer different questions. ⇒ **The true raw drop is
31.13 → 4.634, i.e. 6.7x, not ~4x.** Registered as `RETR-2026-09-20-P3-TARGET-DENSITY`; the
`MEASURED` stamp without a path is what let it through.

⭐ **The census that replaces it** (halfA, per labelled window, `raw/p3_census_halfA.json`):
360° **31.13** (median 20, p95 97, max 149) → gate-relevant **4.634** (median 3, max 34); the gate
population is **14.88 %** of all supervised boxes, i.e. **~85 % of what the head trains on lies
outside the population the collision gate reads** — P3's hypothesis, quantified for the first time.
The pad-32 truncation binds on **34.33 %** of labelled windows under 360°. The grid was **proven,
not assumed**: 3/3 literals match on halfA and 3/3 on halfB against config files written by
*different runs on different days*, so the agreement is evidence about the grid rather than about
one process's determinism, and a mismatch — or the absence of any literal — refuses the bank.

### ⭐ 2. AMENDMENT: a named alternative for a P3 regression, and a CONDITIONAL control arm

⛔ **The risk, MEASURED through the real `slot_set_loss` on 400 halfA windows** (not a
re-derivation), using per-window count distributions rather than means — a mean would hide the
zero-target windows that are the whole point (`raw/p3_presence_balance_halfA.json`):

| | 360° | gate-relevant |
|---|---|---|
| matched targets / window | 19.02 | **4.855** (3.92x fewer) |
| presence **positive** weight share | 0.7013 | **0.3379** |
| windows with **zero** matched targets | 2.75 % | **26.75 %** |
| `loss_presence` | 0.2061 | 0.1123 |

⇒ the no-object term goes from a **30 % minority** of the presence loss to a **two-to-one
majority**. `NO_OBJECT_W = 0.1` (the DETR `eos_coef` convention) is implicitly calibrated for
~19 targets per 100 queries; P3 moves the head to ~4.9 **and does not touch it**.

⛔ **Pre-registered as a NAMED ALTERNATIVE, read from the presence terms and never inferred
afterwards:** if `P3-SUPERVISION` regresses, *"the presence / no-object balance shifted"* is a live
explanation with nothing to do with P3's hypothesis. Same defect class as the `--v2` conflation —
ten levers on two axes, result non-attributable.

⭐ **The control, SOLVED rather than tuned.** From `share = pos / (pos + (Q - pos)*W)`, the weight
that holds the gate arm's positive share **equal** to the 360° arm's is **`NO_OBJECT_W =
0.02173`** against today's 0.1 — a 4.6x change, derived from the identity, not swept.

⛔ **MASTER MIND RULING — the control arm is CONDITIONAL, and this is committed in advance:**
it runs **only if P3 moves the near-forward error beyond `P0`'s replicate floor, in either
direction**. If P3 is flat there is nothing to attribute and the arm costs a card slot for nothing;
if P3 moves, the pair separates *"concentrating supervision on the gate population helped"* from
*"the presence term re-balanced"*, and without it the result is **not quotable as P3's effect**.
⛔ `NO_OBJECT_W` is **NOT** changed in the main P3 arm — only in the control.

### ✅ 3. What was checked and did NOT need changing

`P3` is **not** confounded by the query budget. The TrainingFlyWheel drafted *"P3 is confounded,
56.9 % of windows over-queried"* and **retracted it before it shipped**: `--agent-queries 16`
configures the **2-D agents head**, not the refcv6 box3d decoder, which is built with
`n_queries=100` and stamps `refcv6_perception.n_queries: 100` in a real run. Re-measured with the
correct count, `frac_over_queries` is **0.0000** in both populations on both halves. ⇒ no
disentangling arm is needed. Class: a true quantity quoted outside its scope — the scope being
**which head a flag configures**.

⚠️ And the pre-build's own premise was **REFUTED, measured**: the whole index path costs **2.5 s**
against an ~11 h arm (**0.006 %**), so there was never any indexing cost to lift off the card. The
value of that pass is the census above, not the saving that motivated it — reported as such rather
than quietly re-purposed.
