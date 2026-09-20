# PRE-REGISTRATION — what "stayed on the drivable surface" means (`H-DAC-DEF-1`)

⛔ **STATUS: DRAFT, NOT APPROVED, NOT APPLIED.** Written by the DataFlyWheel 2026-09-20 at the
Master Mind's request, for the Master Mind to review and the PI to rule on. **No number anywhere in
the programme changes because this file exists.** Nothing here has been run.

⛔ **This repairs an INSTRUMENT. It is not a goalpost move.** The rule it would replace was written
before anyone had measured what it does to the recorded human, and every number it produced stays
on record beside whatever replaces it (§7).

---

## 1. The defect, stated as an instrument fault

MEASURED, 736 held-out windows, reproduced 736/736 against the landed round-trip
(`…/Research/2026-09-20-dac-human-zero-anatomy/`, landed `bcc993b`):

| | |
|---|---|
| `dac = 0` on the **recorded human** | **328 / 736 = 44.6 %** of windows ⇒ the human's own PDMS is 0 there |
| zeroed **only** by cells whose mass is **road surface** (drivable + lane line + crosswalk + arrow/text) | **264 / 328 = 80.5 %** |
| zeroed by the ego's footprint at **tick 0** — where the recorded car is on the road by construction | **63 windows** |
| violating samples on an **explicitly off-road** cell (sidewalk / kerb / hatching) | 804 / 4,431 = 18.1 %, in **33** windows |
| …of those 804, within **one cell (0.5 m)** of the mapped drivable corridor | **775 (96.4 %)**; median distance **0.429 m**, max 1.047 m |
| the mapped corridor's width where they occur | median **3.00 m**, against an ego width of **2.297 m** |

⭐ **The fault is a read-out, not a map.** `dac_from_drivable` reads the **"drivable" channel alone**
out of the map's **nine**, so paint the car is meant to drive over — lane lines, crosswalks, arrows —
reads as *not drivable*. Under the footprint the map reads **0.93–0.95** drivable against the corpus
floor's 0.31–0.35 over the whole grid: the map is doing its job.

⚠️ **A second, independent fault: resolution.** The grid is 0.5 m. Almost every explicitly off-road
corner sits **within one cell** of the drivable corridor, in corridors barely wider than the car. A
rule with no tolerance is asking the map a question finer than its own cell.

⛔ **Why this blocks `H-DDV2RL-3`:** `PREREG_D9_REWARD_REPAIR` multiplies progress by this term with
the **human as the reference**. A reference zeroed on 44.6 % of windows cannot calibrate a repair.

---

## 2. Hypothesis

**`H-DAC-DEF-1`** — *There exists a drivable-compliance rule, fixed in advance, that agrees with a
human adjudicator about whether the recorded car stayed on the drivable surface, on a stratified
sample of held-out windows, well enough to serve as the reference term in a human-referenced
reward.*

⛔ The hypothesis is about **agreement with what we mean**, never about the rate. A rule chosen for
producing a comfortable number is the defect this file exists to remove.

---

## 3. Candidate rules — fixed here, before any adjudication

All operate on the same object: the ego footprint's 4 corners at 41 ticks = 164 samples per
trajectory, against the t0 SAM3 map. A sample is **admissible** only where the cell is inside the
grid and seen; an inadmissible sample carries no evidence.

| id | rule | what it would MISS — stated before it is measured |
|---|---|---|
| **V0** | the current rule: any corner, any tick, **drivable channel** < 0.5 | nothing. It is the strictest, and it is wrong in the other direction: it fails the human on paint, on unlabelled cells, and on its own footprint at t0 |
| **P1** | **road surface** = drivable + lane line + crosswalk + arrow/text < 0.5 | a car driving along a **crosswalk** or the wrong side of a **lane line** reads compliant. P1 measures *"on the paved road"*, not *"in the correct lane"*, and it must be said that way wherever it is quoted |
| **P2** | **explicit off-road only**: edge + hatched + sidewalk ≥ 0.5 | ⛔ it is **silent wherever the map has no class**. "Seen, no map class" is **17 % of all violating samples** — a car on an unlabelled surface reads **compliant**. P2 therefore needs §4's abstention clause to be honest |
| **P3** | any of the above **plus a one-cell tolerance**: a corner counts only if it is ≥ 1 cell (0.5 m) inside the offending region | a genuine kerb clip of less than 0.5 m reads compliant. ⚠️ Its rate is **not yet measured** and must be measured **in the pre-registered run**, not now |
| **S1…S3** | strictness knobs on any of the above: ≥ 2 corners at one tick · ≥ 2 consecutive ticks · threshold 0.25 | they change how much evidence is needed, never which channels are read. MEASURED: every strictness-only variant of V0 leaves the human-zero rate at **≥ 19.6 %** |

Known rates, already landed and quoted here so nobody re-derives them: V0 **44.57 %**, P1 **8.70 %**,
P2 **4.48 %**, P1+S1 5.43 %, P1+S2 6.93 %, P2+S2 3.26 %. ⛔ These are **not** the selection
criterion — §5 is.

### 4. The unlabelled surface — decided here, not later

Every candidate must state what it does when the cell is **seen but carries no class**. Three
options, and the choice is **pre-committed before adjudication**:

* **(a) compliant** — no evidence of a violation is not evidence of one. Matches the existing
  treatment of *unseen* cells and keeps the term an **upper bound on compliance**, which the
  current docstring already declares.
* **(b) violating** — what V0 does today, by accident rather than by decision.
* **(c) abstain and report** — the window is scored with the term **withheld**, and the share of
  withheld windows is published beside every DAC/EP/PDMS number.

⭐ **The draft proposes (a) for the term and (c) for the reporting**: compliant for the score,
with the unlabelled share always published, so a corpus whose map degrades cannot quietly improve
its own score. **The PI rules.**

> ### ✅ RULED BY THE PI, 2026-09-20: **(a) for the term, (c) for the reporting**
>
> Verbatim: *"a) for the term, (c) for the reporting"*. ⇒ a cell that is **seen but carries no
> class** is **COMPLIANT**, and the **withheld / unlabelled share is published beside every
> DAC / EP / PDMS number**. ⛔ §4 is CLOSED; no candidate may restate the question.
>
> **What it changes, measured before asking.** `corpus_publisher.reasons()` thresholds
> `ego_future_path_on_drivable["real"]`, which is `drivable_hits / ALL path cells` — so unlabelled
> cells sit in the **denominator but not the numerator** and drag the ratio under 0.9. Under (a)
> they move into the numerator. ⭐ For the **54** clips carrying the flag *"near path unlabelled
> (seen, no class), **no path cell on a non-drivable class**"*, nothing under the path is on a
> non-drivable class, so their compliant share is **1.0 by construction** — they pass the 0.9
> threshold **outright, not marginally**. Projected **78** at corpus completion (`s4_price.json`).
>
> ⛔ **NOT applied to the running corpus tonight, deliberately.** The rule is a numerator change
> and is re-derivable from the **banked `.npz`** via `path_classes()` — **no re-inference**. But
> changing it mid-run would split the corpus into two populations judged by two rules. ⇒ apply it
> as **ONE re-tiering pass after production completes (~2026-09-22)**, so every clip is judged by
> the same rule. Nothing on Thor is edited while the supervisor holds its lock.
>
> ⚠️ **Scope.** This does **not** recover the 22 `GIVEN_UP` clips (different reasons, not
> published) nor the 9 *"path untestable (parked / stopped ego)"* clips — that is the zero-motion
> family and a separate question. ⚠️ And it changes those clips' **tier**, not their existence:
> flagged clips are already published, in `semantic_maps/gt_flagged/`.


---

## 5. How the rule is chosen — human adjudication, not a rate

⛔ **The criterion is agreement with a human judgement of the SAME windows**, because the question
is what the term is supposed to mean.

1. **Sample, fixed before labelling:** 60 windows, drawn with a stated seed —
   **20** where V0 fires and P1 does not (the paint cases), **20** where P1 and P2 disagree,
   **10** where every candidate fires, **10** where none does (the control stratum).
2. **What the adjudicator sees:** the projected render — the path in the t0 camera plus the map's
   class panel (`media_projected/`, the format already landed, whose four projection controls pass).
   sha12 only.
3. **Labels:** `on the drivable surface` · `over the boundary` · `cannot tell`. "Cannot tell" is
   reported, never silently dropped, and never counted as agreement.
4. **The bar, committed now:** a candidate is **adoptable** if, over the adjudicated windows that
   are not "cannot tell", it agrees with the adjudicator on **≥ 95 %**, with **neither** error
   direction above **5 %** — a rule that is right on average by being wrong in both directions is
   not adopted.
5. **Ties:** if more than one candidate clears the bar, the **simplest** one wins — fewest channels
   read, fewest knobs — and the choice is recorded with that reason.

### Both outcomes, committed in advance

| outcome | what we do, decided now |
|---|---|
| **A candidate clears the bar** | it is adopted, applied to **every** arm (§7), and `H-DDV2RL-3`'s gate is satisfied on this axis |
| **No candidate clears it** | ⛔ the DAC term is declared **unfit as a reward multiplier with the human as reference**. `PREREG_D9_REWARD_REPAIR` must then either drop the multiplier, or replace the human reference with one that is not scored by this term. **This outcome is a legitimate result**, not a failure to be re-run until it goes away |
| **The adjudication itself is inconclusive** (> 25 % "cannot tell") | the map's 0.5 m grid is too coarse to answer the question at the boundary, and that is reported as the finding; a finer instrument (camera-level segmentation) becomes the open item |

---

## 6. Controls — each must read an EXACT value

Run before any adjudication; if any misses, the run stops and reports the miss.

| control | required reading |
|---|---|
| a trajectory entirely inside a hand-checked drivable region | **exactly 1.0** under **every** candidate |
| a trajectory entirely on cells the map calls sidewalk / kerb | **exactly 0** under V0, P1, P2 *(P3: 0 once past its one-cell tolerance — stated per candidate)* |
| a trajectory over cells that are entirely **unseen** | **exactly 1.0** under every candidate — no evidence is not a violation |
| a trajectory over cells that are **seen with no class** | the value §4 decided, and **the same** under every candidate |
| the recorded human's own footprint **at tick 0** | ⭐ **exactly 1.0** under the adopted rule, on **every** window — the car is on the road at the moment the window opens, by construction. Today 63 windows fail this |
| re-scoring an arm with the map **absent** | byte-identical to its banked number (the term defaults to 1 when no map exists) |

The first four are already implemented and pass in
`…/2026-09-20-dac-human-zero-anatomy/code/dac_anatomy.py::known_value_controls`.

---

## 7. Application, and what stays on record

⛔ **Whatever is adopted is applied to EVERY arm, including banked ones.** A metric change that
touches only new arms manufactures a difference between old and new that is the metric, not the
model.

1. Every arm carrying a DAC / EP / PDMS number is **re-scored** under the adopted rule.
2. **Both numbers are published side by side**, with the rule named: `dac@v0` and `dac@<adopted>`.
   ⛔ The old numbers are **not** deleted, corrected away, or quietly replaced.
3. `MODEL_REGISTRY.md` gains a rule column; a registry row without one is incomplete.
4. Any landed CLAIM whose direction changes goes to `RETRACTION_LOG.md` with its class.
5. The adoption date is recorded, and every number is stamped with the rule that produced it.

## 8. Scope — and what is deliberately NOT in it

* **In scope:** `dac_from_drivable`, and therefore **DAC, EP and PDMS** on this corpus.
* **Not affected:** **NC** — `no_at_fault_collision`, S1's **primary** statistic — does not use DAC.
  S1's collided-selection verdict stands whatever happens here.
* ⛔ **OUT OF SCOPE: the occupancy head's GT channel.** A1's per-half floor and A3/A8's IoU are
  internally consistent **on the `drivable` channel**, and they measure the *head*, not the
  *compliance rule*. Changing what the head is trained to predict is a **separate question with its
  own pre-registration**; nothing in this file licenses touching it.
* Out of scope: the SAM3 map's own class definitions, and any change to the corpus.

## 9. What this draft does NOT claim

It does not claim P1, P2 or P3 is right. It does not claim the human never leaves the drivable
surface: **33 windows** carry an explicitly off-road corner, and one sample sits **1.047 m** past
the mapped edge. It claims only that the current read-out cannot tell those apart from a car
driving over a crosswalk — and that the difference is what the term is for.

<!-- MM-REVIEW-2026-09-20 -->
## ⭐ MASTER MIND REVIEW, 2026-09-20 — APPROVED with three amendments, binding on this file

The draft is adopted as written except for the three points below. Its two best choices are kept
exactly: the criterion is **adjudication, not a rate** ("picking by rate would be choosing the
comfortable number"), and §7's **apply-to-every-arm with both numbers published** is what keeps
this a repair rather than a goalpost move.

### A. ⛔ WHO JUDGES — the draft says "human adjudicator" and we must not quietly become one

A model grading the instrument it will then be scored by is not a human judgement, and calling it
one would be the very substitution this programme keeps catching. The protocol, pre-committed:

1. **Stage 1 — MODEL adjudication, BLIND.** The Master Mind labels all 60 windows from a **blinded
   pack**: shuffled ids, no stratum in the filename, and **nothing on the render naming which
   candidate fires**. The labels are landed **before** the key exists in the repo, so the order is
   verifiable from the commit graph rather than promised.
2. **Stage 2 — HUMAN spot-check.** The PI labels a stratified **12** of the same 60 (≈ 5 minutes),
   without seeing stage 1.
3. **Reading it:** if the PI agrees with the model on **≥ 10/12**, stage 1 carries the adjudication
   and every number is stamped `adjudicator: model, human-verified 12/60`. If not, ⛔ **the PI's
   labels replace the model's on those windows, the bar is re-read on the human subset ALONE**, and
   the disagreement is reported with its pattern (which stratum, which direction).
4. ⛔ No number from this file may be published as "human adjudicated" unless a human labelled it.

### B. ⛔ THE TICK-0 CONTROL NEEDS AN ESCAPE CLAUSE, OR A MAP DEFECT REJECTS A GOOD RULE

§6 requires the human's own footprint at tick 0 to read **exactly 1.0 under the adopted rule on
every window**. That is the right control — the car is on the road when the window opens, by
construction — but today's horizon probe shows the first offending corner is **NEAR and EARLY**
(median 0.7 s / 7.8 m), so near-field map error is a live cause. A window whose map is
demonstrably wrong **at t0** would then reject every candidate, including a correct one.
⇒ **Pre-committed:** a tick-0 failure is first DIAGNOSED. If the ego's own t0 footprint sits on a
cell the map gives no road class at all, the window is **excluded from the control and COUNTED**,
and the count is published with the result. ⛔ **Cap: 5 % of windows.** Beyond that the finding is
not about the rule at all — it is *"the near-field map is unreliable"*, reported as such, and no
candidate is adopted on that evidence.

### C. The withheld share is identical across arms BY CONSTRUCTION — say where the caveat bites

§4(c) publishes the unlabelled/withheld share beside every number, which is right. Add the reason
it is not a comparability threat *within* a panel: the map is a property of the WINDOW, not of the
arm, so two arms scored on the same windows withhold exactly the same set. ⇒ the caveat applies
**across corpora, geometries and map versions**, where the withheld share genuinely differs, and
a DAC comparison across those is inadmissible unless both shares are quoted.

### Unchanged and endorsed
§4's proposal **(a) compliant + (c) always publish the share** stands for the Master Mind; the PI
may overrule it. §8's exclusion of the occupancy head's GT channel stands. §9's refusal to claim
any candidate is right stands — and so does its admission that **33 windows carry a genuinely
off-road corner and one sits 1.047 m past the mapped edge**.
