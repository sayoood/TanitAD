# Adjudicating a geometric labeller against a VLM chain-of-thought — and whether the lane-detector package closes the NUDGE gap

**Package** `Data Engineering/Research/2026-08-30-label-source-adjudication` · **Author**
Research Lab agent (daily run 004, 2026-08-30) · literature + source reading only, 0 GPU.
**Scope, binding:** this package does **NOT** measure or adjudicate the 41.7 % lateral
disagreement (`D-LAT-AGREE`). The PI assigned that to the DataFlyWheel as a blocking work
item and they are on it. Everything below is **background FOR that decision** plus the
separate `D-NUDGE-ABSORB` question. No adjudication is asserted here.

Seeds: `D-NUDGE-ABSORB` (lane changes absorbed into `NUDGE`; does the banked
`2026-08-29-lane-detector-deployment` package close the gap?) and, as background only,
the literature on heterogeneous-source label disagreement.

---

## Part A — does the lane-detector package close the lane-reference gap?

**A1 — NO. It is a deployment PLAN, not a deployed detector; zero transfer has been
measured.** [MEASURED, repo read] `…/2026-08-29-lane-detector-deployment/RESULT.md` selects
the instrument (CLRerNet DLA-34-EMA, 81.55 CULane F1, Apache-2.0), establishes the geometry,
and sizes the run. It contains **no detector run on our corpus**. Its own **F4** is the
binding caveat: transfer must be MEASURED, never assumed, with published precedent spanning
mild degradation to *"the F1-score drops to near-zero"* under semantic shift [lib
`2507.18653`]. ⇒ `D-NUDGE-ABSORB`'s mask reason — *"pending the lane-detector reference"* —
**still stands**, and the two tokens stay in `NOT_YET_EXTRACTABLE`.

**A2 — ⭐ BUT the package's geometry finding ALREADY COVERS the `D-NUDGE-ABSORB` scope, and
nobody has connected the two packages.** [DERIVED from two MEASURED inputs] The lane
package's **F3** gives cylindrical-vs-pinhole deviation on our rig (256×640, `f_ref`
305.577, az ±60°): **2.0 % at ±20°, 7.9 % at ±40°, 17.3 % at ±60°**, and calls
|offset| ≤ ~2 m ⇒ ~±25° ⇒ **≤3.1 %** the near-pinhole ego corridor. A lane change crosses
the boundary of the *current* lane, which sits at **half a lane width ≈ 1.6–1.85 m**
(lane 3.2–3.7 m, `D-NUDGE-ABSORB`). **That boundary is inside the ≤3.1 % corridor.**
⇒ The as-is CLRerNet deployment is **geometrically sufficient for exactly the
`D-NUDGE-ABSORB` adjudication**, with no cylindrical→pinhole rectify primitive needed
(the primitive is only required for adjacent-lane inventory at ±40–60°, F6/consequence 2).

**A3 — The 39 `LANE_CHANGE` records are NOT in the package's waiting-consumer list, and
should be added as a third set.** [MEASURED, repo read] The package names **68
`turn_suppression` + 41 CoT `CORRIDOR_OFFSET`** clips as waiting consumers. The
`D-NUDGE-ABSORB` population — `g_tac.goals.LANE_CHANGE_L` 23 + `_R` 16 = **39 records**, all
39 labelled `NUDGE_*` — appears nowhere in it. At the package's own F5 sizing (2 Hz
keyframes, the 68+41 sets ⇒ "minutes"), **39 more clips is minutes-scale**. This is a
scope addition, not new work.

**A4 — ⚠️ A REGISTER CAVEAT HAS GONE STALE: the auditability gap `D-NUDGE-ABSORB` flags was
FIXED TODAY.** [MEASURED, source read `stack/tanitad/data/ego_manoeuvre.py`] The register
row states *"`ego_manoeuvre.py:318` decides NUDGE on `abs(lat) >= NUDGE_LAT_M` and then
**discards `lat`** — no numeric lateral field survives anywhere in the 4,719-record schema"*.
That is **no longer true**. The current file carries, at `:316-319`, the comment
*"The lateral track is computed UNCONDITIONALLY (2026-08-30) so `lat_peak_m` is populated on
turns too… It used to live inside the `not is_turn` branch and be thrown away immediately
after the comparison"*, and `lat_peak_m: float` is a **`Manoeuvre` dataclass field**
(`:112`), emitted at `:379` (`lat_peak_m=round(_lat_peak, 3)`). ⇒ The cheap durable fix the
row proposes is **already implemented**; the row should be updated so the fix is not done
twice. ⛔ **What is NOT established: whether the 4,719-record RELEASE blob was re-emitted
with the field.** Persisting it in the extractor does not retroactively populate a blob
built before today. **That is the check to run — it decides whether stratification by
|lat_peak_m| is available now or needs a re-emit.**

**A5 — Two structural claims in the row re-verified from source; one needs a qualifier.**
[MEASURED, source read] ✅ `lateral_class ∈ {JUNCTION_TURN_L/R, ROAD_BEND_L/R, NUDGE_L/R,
STRAIGHT}` — **no `LANE_CHANGE` class exists** (`:98`, the dataclass docstring line).
✅ `NUDGE_LAT_M = 1.0` (`:82`) with **no upper bound**, so a 3.2–3.7 m lane change satisfies
it by construction. ⚠️ **Qualifier the row omits:** the live condition is
`abs(lat[j]) >= NUDGE_LAT_M and abs(peak) >= 5.0` (`:335`) — a **yaw gate of 5° is
conjoined**. A lane change carries a yaw excursion so it still qualifies, but the row's
bare *"`abs(lat) >= NUDGE_LAT_M`"* is an incomplete quotation of the predicate and would
mis-predict any low-yaw lateral translation.

---

## Part B — background for the DataFlyWheel's `D-LAT-AGREE` decision (no adjudication asserted)

**B1 — ⭐⭐ THE ADJUDICATION FRAMING MAY BE MALFORMED, BECAUSE THE TWO SOURCES ARE PROBABLY
NOT INDEPENDENT.** [PUBLISHED, banked `2511.00088`] Alpamayo-R1 states its auto-labeling
pipeline *"provides the model with both raw video and auxiliary signals, **including the ego
vehicle's trajectory, dynamic states, and meta actions**"*, and that the teacher is a
frontier VLM (GPT-5 class). If our `alpamayo.*` fields came from that pipeline, then Source B
**saw Source A's own input**. ⇒ Source B is not a second opinion; it is partly a re-emission
of the first. ⛔ **This invalidates every consensus method by its own stated assumption** —
Dawid-Skene (`DOI 10.2307/2346806`), data programming [`1605.07723`], and the spectral
meta-learner family [`1407.7644`] are all identified *by conditional independence given the
true label*. [`2601.22336`] gives finite-K counterexamples where ignoring such dependence
**reverses** the aggregated prediction while every marginal accuracy is matched.
⚠️ **UNVERIFIED and load-bearing: I did not confirm our corpus's `alpamayo` field came from
this pipeline.** It is a 0-GPU provenance check (was the trajectory in the prompt for our
clips?) and it should be run **before** any aggregation method is chosen. This is the same
defect family as our own flagship v1 route head scoring 1.0000 by echoing its nav input.

**B2 — At n = 2 sources the disagreement rate is not evidence about either source.**
[ESTIMATED — my derivation from the structure of `1303.3257`/`1407.7644`, not a quoted
theorem] Two sources yield **one** observed agreement rate against (at least) two unknown
accuracies plus the class prior: underdetermined. "A is good, B is bad" and "B is good, A is
bad" fit the 41.7 % symmetrically. Breaking the symmetry requires a **third conditionally
independent source** or a **small gold set** — the literature offers no third route.
[`2202.02016`] makes the same point formally for transition-matrix identifiability.

**B3 — ⭐ FOR THE LATERAL AXIS SPECIFICALLY, THE REGISTER HAS ALREADY RULED, AND THE
LITERATURE INDEPENDENTLY PREDICTS IT.** [MEASURED (ours) + PUBLISHED (banked)]
`D-DATA-ALPA-LAT` is already **MEASURED and binding**: Alpamayo lateral **31.2 % vs 23.9 %
shuffled (p = 0.335)**, lane 20.0 % vs 19.5 % (p = 0.706) — *"AT CHANCE on this corpus and
must not supervise a lateral label"*. The VLM literature predicts exactly this failure
**a priori**: [`2507.20174`, LRR-Bench] tests left/right discrimination directly and finds
*"humans achieve near-perfect performance on all tasks, whereas current VLMs attain
human-level performance only on the two simplest tasks"*, with the best models near zero on
several; [`2310.19785`] finds all 18 VLMs evaluated perform poorly on spatial relations
(BLIP 56 % vs humans 99 %). [`2512.14044`] attributes driving-VLM hallucination to
*"reliance on ungrounded, text-based Chain-of-Thought reasoning"* — the mechanism behind our
own `D-DATA-COT-HALLUC`. ⇒ **On the lateral axis there may be no adjudication to make:
one source is already measured at chance, and that is the field's expected result, not a
surprise.** This is offered as convergent background — the DataFlyWheel owns the ruling.

**B4 — ⛔ TWO POPULAR METHODS ARE FALSE-POSITIVE GENERATORS HERE. Do not run them.**
[PUBLISHED, banked] **cleanlab / confident learning** [`1911.00068`] explicitly *"builds on
the assumption of a class-conditional noise process"*; **co-teaching** [`1804.06872`] selects
**small-loss** samples. A deterministic geometric rule's errors are a function of *x* by
construction — that is *definitionally* instance-dependent noise — and under a systematic
threshold offset both methods will select whichever convention dominates and report it as
clean. [`2110.12088`, CIFAR-10N/100N] is the empirical warrant that real annotation noise is
instance-dependent, not class-dependent. Same shape as the ridge-probe λ-on-test trap: a
method that *cannot see* the defect returns a confident number.

**B5 — What the literature DOES support, in cost order.** [PUBLISHED, banked]
(i) **Don't adjudicate — model both.** [`2110.05719`] puts **one prediction head per
annotator over a shared trunk** and never aggregates, matching or beating aggregate-then-
train on 7 tasks; this **preserves attribution**, which picking a winner destroys
permanently. (ii) **Soft targets over the annotation distribution** [`2511.14117`]: matches
or exceeds hard-label accuracy on every dataset tested, −32 % KL to the annotator
distribution. (iii) **Treat straight/nudge as ORDINAL with an adaptive boundary**
[`2509.02351`, ORDAC] — its premise is *"ordinal image classification where class boundaries
are often ambiguous"*; corrects rather than discards; Adience @40 % noise MAE 0.86 → 0.62.
Our disagreement's stated shape (a threshold/granularity difference, not a sign error) **is
the literature's definition of the case where adjudication is the wrong operation.**
(iv) **Baseline first with noise-ignorant ERM on frozen features** [`2411.00079`] — SOTA on
real instance-dependent noise for the cost of one linear head. (v) If a weighting is wanted,
**per-region source accuracies** [`2203.13270`, Liger] can represent "the rule is right in
the large-offset region, the VLM in the near-zero region"; and [`1805.08877`, Adversarial
Label Learning] is the one aggregator that **requires no conditional-independence
assumption**, hence the only one B1 does not disqualify.

---

## What this changes for TanitAD (≤3)

1. **Add the 39 `LANE_CHANGE` records as a third waiting set to the lane-detector run, and
   run it** — geometry is already established as sufficient for this scope (A2), the marginal
   cost is minutes (A3), and it converts `D-NUDGE-ABSORB` from *awaiting an instrument* to
   *measured*. Gate on measured F1/agreement from the sets, never on CULane numbers (A1).
2. **Run the 0-GPU Alpamayo provenance check before the DataFlyWheel picks an aggregation
   method** (B1): was the ego trajectory in the VLM's prompt for our clips? If yes, every
   consensus method is off the table by its own assumptions, and the register's existing
   `D-DATA-ALPA-LAT` ruling (B3) is likely the whole answer for the lateral axis.
3. **Update `D-NUDGE-ABSORB`'s auditability caveat — its fix landed today** (A4) — and check
   whether the RELEASE blob was re-emitted with `lat_peak_m`, since the extractor change does
   not retroactively populate a blob built before it. Correct the row's NUDGE predicate to
   include the conjoined 5° yaw gate (A5).

## Named empty searches

- **A published measurement of VLM lateral-manoeuvre-label accuracy against geometric ground
  truth on driving clips**: NOT FOUND (two probes: driving-VLM reliability, and
  annotation-quality literature). LRR-Bench and OmniDrive-R1 measure adjacent quantities
  (benchmark left/right; QA accuracy). Our 4,719-record two-way table would be novel — **but
  only against gold, not against the other source** (B2).
- **Any method for exactly two label sources where one is a deterministic program and the
  other is a learned model conditioned on that program's input**: NOT FOUND. Nearest are
  `2203.13270` and `1805.08877`; everything else assumes ≥3 sources, independence, or both.
- ⚠️ **The Alpamayo "92 % QA" figure is not a reliability number for Source B** — it is
  LLM-evaluator vs human-evaluator agreement on a curated 2 K set [`2511.00088`]. It must not
  enter a registry row as label accuracy.

## Banked primaries

NEW this package: `2507.20174` · `2310.19785` · `2512.14044` · `1407.7644` (⚠️ banked with an
empty title string — PDF present, index row needs the title) · `2601.22336` · `1805.08877` ·
`2203.13270` · `1911.00068` · `1804.06872` · `2110.05719` · `2511.14117` · `2509.02351` ·
`2411.00079` · `2110.12088` · `1605.07723`. CITED-BY UPDATED: `2511.00088` (Alpamayo-R1,
already banked). **No arXiv ID exists** for Dawid & Skene 1979 (`DOI 10.2307/2346806`),
Uma et al. *Learning from Disagreement* (`DOI 10.1613/jair.1.12752`), or Díaz & Marathe 2019
(`DOI 10.1109/CVPR.2019.00487`) — cited as PUBLISHED-SECONDARY, inadmissible for the
registry until banked by DOI.

⚠️ **Read-depth disclosure:** `1605.07723`, `1407.7644`, `1805.08877` and `2203.13270` were
sourced at abstract level by the search pass; the PDFs are now banked but their *body*
claims (identifiability conditions, minimum source count) have **not** been read at source.
Do not quote a body claim from them without opening the banked PDF.
