# The lane-change label is not derivable from the ego track — proved, and replaced

**Author:** Architecture & Inference agent, 2026-08-16. **Extends** (does not replace)
`PI_REVIEW_FINDINGS.md` in this directory. **Primary sources:** `PI_VERDICTS_2026-08-16.json`
(human adjudication) · `labels/engine_a_aug120.jsonl` + `labels/engine_a_w120val.jsonl` (the
Engine A recompute, cross-checked 201/201 against the shipped geometry) · the bridged ego
`npz` for all 201 aug120 clips · `stack/scripts/s2_derive.py`, `ph0_v2.py`, `refb_labels.py`
read from source.

---

## ⛔ THE BINDING RULING (Sayed, PI, 2026-08-16)

> *"stop emitting lane_target/Prepare Lane change from geometric gate. Prepare lanbe change
> should onyl be to follow route or follow the main road (no route set), it must be derived
> from the context"*

Recorded verbatim, with its date and attribution, per the `V6F_PLANNER_DESIGN.md` convention.
Three binding parts, all now in force in `stack/scripts/s2_derive.py` §LC:

1. **The geometric gate emits NEITHER token.** Removal, not retuning — `LC_MIN_LAT_M` is not
   adjusted.
2. **`PREPARE_LANE_CHANGE` is admissible ONLY in service of route following:** (a) a route is
   set and the ego's current lane does not serve it, or (b) no route is set,
   `FOLLOW_MAIN_ROAD` applies, and the current lane does not continue as the main road (lane
   ends / forced merge / exit-only lane).
3. **It must be derived from CONTEXT**, not from observed lateral displacement.

⭐ **What the ruling changes philosophically.** Every other S2 label is derived from what the
ego **DID**. This one is now derived from what the route **DEMANDS**. The two can disagree —
a driver may fail to reposition in time, or may change lane for reasons outside the route
(overtaking, courtesy). ⇒ **The ego's actual lane change is CORROBORATION, never the source**,
and "observed but not required" is a real category that stays visible in the record
(`corroboration.observed_lane_change.is_label_source: false`).

⇒ **`LANE_TARGET` leaves `g_str` emission entirely.** Under (2) a lane change is a **MEANS**,
not a **GOAL**. ✅ **Verified against the vocabulary** (read-only, no edit): `v6.py:151`
`STRATEGIC_GOAL_TOKENS` is the goal channel, `:157` `STRATEGIC_ACTION_TOKENS` the action
channel, documented *"emitted by S, condition S's OWN predictor"*. The split supports the
reading; there is no contradiction to report.

⚠️ **THE TOKENS STAY IN THE VOCABULARY.** `v6.py:3281` sizes `GoalVocabulary` embedding tables
from these tuples, and the live v6F S-W run on Thor resumes strictly against them. A vocabulary
entry with zero training support is safe and reversible; a changed embedding shape is not.
Nothing in `stack/tanitad/models/v6.py` was touched, and `s2_derive.check_vocab_drift()` still
returns `"checked"`.

---

## §1 Priority 1 — a NEGATIVE RESULT, and it is why the redefinition was needed

The ruling de-prioritised the hunt for a discriminator. The work was already done, and it is
worth recording because **it independently establishes that the ego track cannot carry this
label** — the ruling is not merely a policy preference.

### 1.1 The reproduction is exact

The gate `_gated_lc_event()` fires on **19/201 aug120 clips**, and those 19 are **exactly** the
19 rows the PI reviewed in `S1_LANE_TARGET` (set equality, zero unreviewed). Recomputing
`latmaneuver_from_future` from the bridged ego `npz` reproduces every banked `lat_m` to
**max |Δ| = 0.005 m** (rounding). Adjudicated: **14 wrong / 4 correct** (the 14 includes
`4bea0a51`, whose `v` stayed `null` while its note says wrong — as `PI_REVIEW_FINDINGS.md`
counted it). **MEASURED**, `review/code/lc_feat.py` → `review/raw/lc_feat.json`.

### 1.2 ⛔ NOTHING SEPARATES — 46 features, three horizons, exact tests

**46 candidate features** were tested: net heading change over the event window, the
constant-curvature (arc) residual of the lateral offset, step-vs-ramp logistic/quadratic fits,
curvature bidirectionality (out-and-back), yaw detrending, lateral velocity/acceleration peaks,
arc-length normalisation — at **4 s, 8 s and 12 s** horizons.

Because n=4 CORRECT vs n=14 WRONG, the two-sided Mann-Whitney p is computed **exactly by
enumerating all C(18,4) = 3,060 assignments**, plus an exact **max-statistic family-wise null**
over the same 3,060 (not the conservative Bonferroni approximation).

| feature | p_exact | p_FWER | Cliff's δ | median CORRECT | median WRONG | separation |
|---|---|---|---|---|---|---|
| `yaw_lin_r2` (4 s) | 0.0248 | **0.292** | −0.75 | 0.813 | 0.9805 | none |
| `yaw_detrend_rms` (4 s) | 0.0464 | 0.458 | +0.68 | 0.00963 | 0.00532 | none |
| `bidir` (4 s) | 0.0559 | 0.790 | +0.52 | 0.0095 | 0.000 | none |
| `abs_lat_resid_arc` (4 s) | 0.1268 | 0.789 | +0.54 | 1.48 m | 0.51 m | none |
| `abs_lat_f` (the gated quantity) | 0.7980 | 1.000 | −0.11 | 4.21 m | 4.07 m | none |
| *(41 others)* | ≥ 0.06 | ≥ 0.54 | — | — | — | none |

*(Medians are true medians — the mean of the two middle order statistics at even n. The first
version of `lc_stats.py` took the upper-middle element and reported `yaw_lin_r2` WRONG as 0.986
where it is 0.9805; corrected in the script and here. A median is a named estimator and must be
the one it is named after — the same discipline as the `overlapping_holdout_se` rule.)*

⛔ **NOT ONE FEATURE SEPARATES THE TWO GROUPS.** The minimum attainable two-sided p at this
n is **6.54 × 10⁻⁴**, reachable **only by perfect separation** — and **no feature separates
perfectly**. The best candidate, `yaw_lin_r2`, is directionally sensible (a real lane change
fits a constant-curvature arc *worse*, δ = −0.75, a large effect) but lands at **p_FWER = 0.29**
and its ranges overlap completely.

⚠️ **Power, stated honestly.** n=18 is small. This analysis can only detect *very large* effects.
What it **can** exclude is a clean discriminator — a feature that would actually fix the label
would have to separate near-perfectly, and that is exactly what is ruled out. A weak, partially
overlapping signal remains possible and is **NOT ESTABLISHED** here.
⛔ **No threshold was tuned.** The one threshold reported below uses `LANE_HALF_M = 1.75`, a
constant that already existed in `refb_labels.py`; at that value the arc-residual fires on
**2/4 CORRECT and 2/14 WRONG — precision 0.50**, which is not a fix.

### 1.3 ⭐ THE MECHANISM — and it needs no statistics at all

The two gates in the detector are **jointly satisfiable by pure road curvature**, and this is an
identity, not an inference.

`latmaneuver_from_future` (`refb_labels.py:1256`) admits a lane change when
`|net_yaw| ≤ LC_NET_YAW_MAX = 0.20 rad` over the 4 s horizon **and** `|lat| ≥ 1.75 m`;
`s2_derive._gated_lc_event` then requires `|lat_m| ≥ LC_MIN_LAT_M = 3.0 m`. For a
**constant-curvature arc** of arc length `L` with net heading change `Y`, the lateral offset in
the window-start frame is exactly `L(1 − cos Y)/Y`. At the yaw ceiling `Y = 0.20`:

> **lat_max(L) = 0.0997 · L**

so a **pure curve** reaches the 3.0 m gate at **L = 30.1 m**, i.e. at **v = 7.53 m/s = 27 km/h**
over the 4 s horizon. **Every one of the 19 gated clips is faster than that** (v_mean range
12.7 – 33.2 m/s, median 18.6).

⇒ **Above 27 km/h the yaw gate does not bound the lateral offset, and the detector cannot tell
a lane change from a road that bends.** No value of `LC_MIN_LAT_M` fixes this, because the
quantity being thresholded is not scale-free while the quantity being gated is.

**MEASURED per clip:** on **15 of the 19** gated clips the pure-arc prediction **alone** already
exceeds the 3.0 m gate, **with the same sign as the emitted event** — the entire "lane change"
is the road curving. On **13 of the 14** PI-WRONG clips a constant-curvature arc fits the heading
profile at **R² > 0.92** (median **0.9805**; the lone exception is `b9073ea5` at 0.195 — which is
also the clip with the largest arc residual, 2.23 m, i.e. the one WRONG clip that most resembles
a real lane change). Worked example, `51fd1c9f` (PI: wrong): 33.1 m/s,
L = 133 m, net_yaw = −0.121 rad ⇒ a circular arc of radius 1,100 m predicts **−8.03 m** of
lateral offset; the gate saw **−6.04 m** and called it a lane change.

⚠️ **This CORRECTS a detail of `PI_REVIEW_FINDINGS.md` §Defect 2.** That document's refuted
hypothesis blamed `engine_a_summary` building `pts` in a frame pinned to the **t0** heading.
Read from source, `latmaneuver_from_future` computes `lat_m` in the **window-start** frame
(`seg[0,2]`), not the t0 frame; only the direction-consistency filter uses the t0 frame. The
hypothesis was therefore mis-stated as well as refuted. The refutation of the **curvature**
prediction stands and is reconfirmed here: whole-clip `peak_kappa_per_m` does not separate
(δ = +0.04, p = 0.96), and neither does the within-window curvature (δ = −0.29, p = 0.44).

### 1.4 What this establishes

The distinction the PI is making is **relative to lane boundaries**, and the ego pose does not
contain them. Four metres of lateral movement is a lane change on a narrow road and ordinary
drift on a wide one; the ego track is identical. PhysicalAI-AV ships **no map, lane graph or
lane-boundary annotation** (dataset card, verbatim: *"we do not include open maps data"*;
`obstacle.offline`'s 10-class enum over 87,481 cuboids is all dynamic agents). ⇒ **The ruling's
redirection from "did the ego move sideways" to "does the route require a different lane" is
what dissolves the identifiability failure** — it replaces a question the ego track cannot
answer with one about the scene and the route.

---

## §2 Implementation — what changed, and the measured emission delta

All edits in `stack/scripts/s2_derive.py` (the **one home**, three consumers) and
`stack/scripts/ph1_fuse.py`.

| change | site |
|---|---|
| §LC block: the ruling verbatim, `LANE_CONTEXT_INPUTS`, `lane_change_requirement()` | `s2_derive.py` |
| `LANE_TARGET` branch **deleted** from `derive_g_str` | `s2_derive.py:~357` |
| `LANE_TARGET` removed from the ROUTE_TO **remap destination set** | `s2_derive.py` |
| VLM-primary fallback: `goal_kind=lane_target` now **abstains with a reason** | `s2_derive.py` |
| `PREPARE_LANE_CHANGE` branch now keyed on `lane_change_requirement()`, not `_gated_lc_event()` | `s2_derive.py:~483` |
| VLM-verb fallback also gated on the requirement | `s2_derive.py` |
| observation recorded as `corroboration.observed_lane_change` `{is_label_source: false}` on **both** blocks | `s2_derive.py` |
| `_gated_lc_event()` demoted in its docstring to a corroboration-only helper | `s2_derive.py:153` |
| `lane_context=None` wired through with the reason it is None | `ph1_fuse.py` |

`lane_change_requirement()` reads **the route and the lane context only**. It is behaviourally
unable to see `lane_change_events` — pinned by a test that fuzzes the observation across
`{lc_left, lc_right} × {−99, −4, 0, 4, 99} m` and asserts the requirement and the emitted token
never move.

### 2.1 The emission change — MEASURED on all 797 records

Re-derived with the new module over the full labelled set (201 aug120 + 600 w120val, with the
VLM symbols faithfully reconstructed from each record's own corroboration block —
`review/code/lc_emit.py`):

| token | banked v1 | new | delta |
|---|---|---|---|
| `LANE_TARGET` (g_str) | **80** (10.04 %) | **0** (0.00 %) | −80 |
| `PREPARE_LANE_CHANGE` (a_str) | **80** (10.04 %) | **0** (0.00 %) | −80 |
| `FOLLOW_MAIN_ROAD` | 395 | 474 | +79 |
| `HOLD_CORRIDOR` | 526 | 597 | +71 |
| `REDUCE_TO` | 85 | 94 | +9 |
| `NONE_ABSTAIN` | 13 | 14 | +1 |
| TURN_L/R · STOP_AT · PREPARE_STOP · RESUME_CRUISE · PREPARE_EXIT | — | — | **0 (unchanged)** |

**Exactly 80/797 records change (10.04 %) — precisely the affected set, and no other token
moves.** On aug120 specifically the gate's 19/201 = 9.5 % becomes **0/201**.

Where they went: 79 `LANE_TARGET → FOLLOW_MAIN_ROAD` (the clip's own **measured** `route=follow`
token, not a default-of-absence), 1 → `NONE_ABSTAIN` (a VLM `route_to` with no junction geometry
to remap to — the pre-existing gate, correctly). For `a_str`, 71 → `HOLD_CORRIDOR` and
**9 → `REDUCE_TO`**.

⭐ **Those 9 are a bonus defect fix.** The lane-change branch sat **above** `REDUCE_TO` in the
`elif` chain, so **9 real decelerations (net Δv ≤ −3.0 m/s) were being suppressed** by a
phantom lane change. They now emit.

⭐ **REGRESSION CHECK — the change is provably surgical.** Comparing v1 against v2 record by
record: **80 have a changed token; for the other 717 the `g_str` AND `a_str` blocks are
BYTE-IDENTICAL — 1434 of 1434 family-blocks**, apart from the two corroboration keys the ruling
adds (`lane_change_requirement`, `observed_lane_change`). Args, arg-masks, provenance, sources,
confidences and the existing corroboration all unchanged. Nothing outside the lane-change path
moved.

**End-to-end verification (MEASURED):** all **797/797** re-derived records **validate against the
authoritative schema** (`colab/s2_schema.py::validate`, 0 refused), and both drift checks —
`s2_derive.check_vocab_drift()` and `s2_schema.check_v6_drift()` — return `"checked"`, i.e. the
derivation's pins still equal the real `v6` module. ⚠️ `colab/s2_schema.py` needed **no change**
and was **not touched** (another agent owns `colab/`): the new `lane_context` parameter is
keyword-only with a `None` default, so its positional two-argument calls
(`s2_schema.py:190-191`) and the S2 v1 builder's (`s2_build_labels.py:226-227`) keep working
unchanged and simply get the honest default.

### 2.2 On `a_str` and abstention — a limitation, stated

`STRATEGIC_ACTION_TOKENS` has **no `NONE_ABSTAIN`**, and `s2_valid` in `train_v6_staged.py`
(`:1953`) is a **window-level** mask shared by `g_str` and `a_str`, so there is **no per-family
abstain channel** today. The 80 records therefore fall through to `HOLD_CORRIDOR`/`REDUCE_TO`
rather than abstaining.

I judge this **defensible, not a lie**: under the ruling a lane change not required by the route
is not a *strategic* action at all — repositioning to overtake is a **tactical** manoeuvre, and
`g_tac_lat` already carries `LANE_CHANGE_L/R` for it. `HOLD_CORRIDOR` is the honest strategic
action while the tactical layer does the work. ⚠️ **UNVERIFIED**: I have not measured how often
the 80 clips' true strategic action is something other than hold/reduce. **Escalation:** if the
PI wants true abstention here, it needs a per-family validity mask in the batch schema — a
trainer change I did **not** make, because the live v6F run is resuming.

---

## §3 Priority 2 — the context-based derivation, and what must be built

`lane_change_requirement()` is implemented and pinned. It needs four named inputs
(`LANE_CONTEXT_INPUTS`):

| input | meaning | available today? |
|---|---|---|
| `n_lanes_same_direction` | lanes on the ego's carriageway | ⚠️ VLM `lanes_visible` — **unreliable, see §5** |
| `ego_lane_idx` | ego's 0-based lane index from the right | ⚠️ VLM `lane_ego` — same instrument, same doubt |
| `route_lane_idx` | which lane serves the route / main road | ⛔ **DOES NOT EXIST** — needs a map/lane graph |
| `lane_continues` | does the ego lane continue (not exit-only)? | ⛔ **DOES NOT EXIST** |

⇒ **`required` is `None` (UNKNOWN) for every clip in the corpus today**, and `ph1_fuse.py`
passes `lane_context=None` **deliberately**, with the reason in a comment at the call site.

⛔ **I did NOT wire B1's `lanes_visible`/`lane_ego` in, on purpose.** Two of the four inputs are
missing outright, so the requirement could not be computed even with perfect lane counts; and
building on the VLM count would replace an *unidentifiable* label with a *confidently wrong*
one — the exact trade the brief forbids.

**What would have to be built**, cheapest first:

1. **`route_lane_idx` / `lane_continues` are the blocker, not the count.** PhysicalAI-AV has no
   map. **NuRec scenes ship `map.xodr`** (OpenDRIVE — banked finding: *"NuRec is open msgpack +
   gsplat works on Thor"*), which carries lane topology including exit-only and lane-end. That
   is the only in-programme source of a lane graph today, and it covers AlpaSim/NuRec scenes,
   **not** the PhysicalAI training corpus.
2. **A vision lane-boundary estimator** (lane-marking segmentation → lane index + count from the
   front-wide camera). This is the vision-only-at-inference path, and it also gives the
   corroboration signal §1.4 says the ego track cannot.
3. **Until either lands, `PREPARE_LANE_CHANGE` has no admissible source** and the honest
   emission is the route's own token. That is now the implemented behaviour.

---

## §4 Priority 3 — `"no agents"` cannot be produced by an absent measurement any more

**The defect** (`ph1_fuse.py:357`, before): `", ".join(census) or "no agents"` — an empty census
is falsy, so **absence of evidence rendered as evidence of absence**.

⭐ **MEASURED, and worse than the code alone suggests:** on the fused aug120 corpus
`"no agents"` appears on **119/201 records (59.2 %)**, and **115 of those 119 (96.6 %) already
carried `perception.absent` in the structured layer**. The record knew perception had not run,
and the prose contradicted it. That is what the PI read on `03ba450b`
(*"In the picture of frame 9 there is clear a car as incoming traffic"*).

⭐⭐ **AND ON THE PI'S OWN 19 REVIEWED CLIPS THE ATTRIBUTION IS 100 % CLEAN.** Of those 19:
**10 rendered `"no agents"`, and ALL 10 (10/10) had `perception.absent` set**; the other 9 had
≥1 real track. The PI explicitly wrote *"no agents wrong"* on **8** of them. ⇒ **Every single
`"no agents"` he saw was unavailability rendered as a confident negative — not one was a failed
detection, and not one was a true empty scene.** His complaint was exactly right, and it was a
reporting defect rather than a perception defect.

⚠️ **This REFINES `PI_REVIEW_FINDINGS.md`**, which states the census was empty *"for every clip
in the corpus"*. On aug120 that is **too strong**: 82/201 clips DO carry detections; 115 have
`perception.absent` (the known SAM3 gap) and 4 have zero tracks from a detector that did run.

**The fix** (`ph1_fuse.py`) makes the three states structurally distinct:

| state | when | prose |
|---|---|---|
| `unavailable` | `perception.absent` set, **or zero frames processed** | `agent census UNAVAILABLE (<reason>)` |
| `measured`, empty | detector ran on ≥1 frame, returned nothing | `0 agents detected` — a **count**, not a claim about the world |
| `measured`, non-empty | — | `2 car, 1 pedestrian` |

`census_state()` returns `{"state", "reason", "counts", "n_agents"}` and is **structurally
unable** to report a measured zero when perception did not run — the unavailable branches return
before the counts are consulted, and `n_agents` is `None` (never `0`) in that case. The
structured `perception.census` block is now in the fused record, so downstream reads a field
instead of parsing prose. The same rule is applied to the corroboration leg: `flagged_empty_urban`
is a **finding**, and an absent (or zero-frame) detector can no longer produce one.

Pinned by `test_absent_perception_never_renders_as_a_finding_about_the_world`,
`test_scenario_line_carries_the_census_state_not_a_bare_negative`,
`test_empty_census_cannot_produce_a_flagged_empty_verdict` — including an explicit assertion
that **no branch can emit the literal `"no agents"`**.

⚠️ **The fused records on disk are NOT re-fused** — this fixes the producer. Every `"no agents"`
in an existing S2 artifact is still the old fallback and must not be read as evidence.

---

## §5 Priority 4 — the lane COUNT: one complaint is a RENDERING defect, one is a real error

**Where it comes from:** the VLM's B1 pass, `ph0_v2.py:66` (`lanes_visible`, int 0–6) and
`lane_ego`, prompted at `ph0_v2.py:140`:

> **`lanes_visible` = lanes you can count on the ego's carriageway (0 if unclear).**

⭐ **That definition dissolves the PI's first complaint.** On `03ba450b` the PI wrote
*"the pipeline is saying there is one lane, better: there are two lanes, one ego lane and one for
oncoming traffic"*. The record says `lanes_visible=1, lane_ego=0, road_type=rural`. **By the
prompt's own definition the count is CORRECT** — a rural road with one lane each way has exactly
one lane on the ego's carriageway. The defect is the **rendering**: `scenario_line` printed
`"rural 1-lane"`, which reads as *a one-lane road*. **The pipeline and the PI were using two
different definitions of "lane count", and the prose hid which one was meant.**
⇒ Fixed: the phrase is now `rural 1-lane-ego-carriageway`, pinned by
`test_lane_count_phrase_names_its_scope`.

**The second complaint is not dissolved.** On `51fd1c9f` (`3 lane wrong`) the record says
`lanes_visible=3, lane_ego=1, highway, night`. The carriageway definition does not rescue this
one — the PI is counting the same carriageway. ⚠️ **UNVERIFIED as to the true count**: I cannot
adjudicate it without watching the frames, which is outside a no-GPU analysis pass. It is
plausibly a genuine VLM error on a night highway.

**Corpus-wide reliability (MEASURED, n=201, `review/code/lane_count.py`):**

| probe | result | reading |
|---|---|---|
| `lanes_visible` distribution | 1: 82 · 2: 92 · 3: 26 · 4: 1 | plausible shape |
| `lane_ego` distribution | 0: 114 · 1: 87 | plausible |
| `lane_ego ≥ lanes_visible` (impossible for a 0-based index) | **0/201** | internally self-consistent |
| `lanes_visible == 0` (the prompt's own *"0 if unclear"* escape) | **0/201** | ⛔ **never abstains** |
| B1 `conf` | **`"high"` on 201/201 = 100 %** | ⛔ **degenerate — a dead channel** |

⛔ **Two of these are decisive.** The VLM's self-reported confidence is **constant** across the
entire corpus, so it carries **zero information and must never gate anything**. And it **never
once** used its documented "unclear" escape. ⇒ **The lane count has no usable uncertainty
signal**: there is no way to tell a confident-and-right count from a confident-and-wrong one,
which is precisely the C77 family again — a channel that cannot report its own absence.

**Recommendation on emission:**

1. ✅ **Keep `lanes_visible`/`lane_ego` in the record** as VLM semantics with provenance. They
   are the only lane signal we have and are internally consistent.
2. ⛔ **Do NOT let either DECIDE a label** — not `PREPARE_LANE_CHANGE` (already enforced), and
   not any future lane-indexed arg — until it has a working abstention. n=2 adjudicated
   (one definitional, one plausibly wrong) is **far too small** to state an error rate, and I
   do not state one.
3. **The cheapest next step is an accuracy measurement, not a fix:** adjudicate `lanes_visible`
   on a stratified sample (~40 clips across urban/highway/rural × day/night) using the review
   sheet that already exists. Until then, "the lane count is unreliable" is supported by the
   **dead confidence channel and the never-used abstain**, not by a measured error rate.
4. **Fix the prompt's abstention behaviour** before trusting the field: `conf` at 100 % "high"
   and 0/201 "unclear" means B1 was never given a reason to hedge.

---

## §6 Test status — and ⛔ MY OWN RETRACTION about the baseline

### ⛔ RETRACTED: "there are 4 pre-existing failures in files I did not touch"

I measured the tree myself before my first edit and got **4 failed · 3570 passed**, and I wrote
that the suite was not green and that four failures pre-dated me in another agent's blast radius.
**That was WRONG, and the four failures were MINE — an artifact of MY SHELL, not of the tree.**

**MEASURED, the reconciliation:** all four fail in isolation too (so not a concurrent-edit
transient), but re-run with **`PYTHONUTF8=1` all four PASS**:

| invocation | result |
|---|---|
| `pytest -q <the 4 tests>` | **4 failed** — `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f` |
| `PYTHONUTF8=1 pytest -q <the same 4>` | **4 passed** |

The tests read UTF-8 fixtures; without `PYTHONUTF8=1` this Windows shell decodes them as cp1252
and they die on the first non-ASCII byte. **The failures describe my environment, not the code.**

⇒ **ROOT-CAUSE CLASS — and it is the `df`-on-a-pod family, with me as the one who fell in.**
A probe reported a property **of the measuring apparatus** and I read it as a property **of the
subject**. It is precisely the trap this project's CLAUDE.md opens with (`df` reporting the
cluster not the pod quota; `free`/`tegrastats` on Thor; `memory.usage_in_bytes` counting page
cache) — *"a probe that reports the wrong scope is worse than no probe, because it looks like an
answer."* I even wrote a confident narrative on top of it ("three incompatible baselines, the tree
moves under every measurement"), which **explained away the discrepancy instead of investigating
it.** The discrepancy was the signal.

⇒ **THE RULE I SHOULD HAVE FOLLOWED:** when my measurement disagrees with a banked artifact,
**suspect my apparatus before I conclude about the subject** — and the check is cheap
(re-run one failing test under a different environment). I had two independent signals that
should have triggered it: the failures sat in files with no plausible connection to each other,
and a `charmap` codec error is an *encoding* symptom, not a logic one.

✅ **What actually holds: the suite is GREEN**, matching the banked artifacts
(`…/2026-08-16-seam-instrument/raw/pytest_baseline_pre_edit.txt` and
`…/2026-08-16-sam3-dtype-fix/raw/pytest_full_suite.txt`, both **3574 passed, 0 failed**) and
matching the number in my original brief. **`pytest -q` passing is a valid gate, and I must meet
it** — which makes this correction load-bearing rather than cosmetic: under my wrong baseline I
would have shipped with "4 failures, not mine" as an accepted state.

⚠️ **Anyone running this suite on Windows must export `PYTHONUTF8=1`**, or it reports four
failures that do not exist.

### ⭐ The blast-radius question, answered by direct measurement

**`test_ph1_fuse.py` had ZERO failures at my pre-edit baseline** — the full `pytest -q` short
summary names every failure, and not one was in that file (this half of the baseline stands: the
encoding artifact above never touched `test_ph1_fuse.py`). ⇒ **`test_lane_change_gate_requires_
displacement_and_valid_follow` was PASSING**, so the geometric gate was faithfully pinned right
up to this ruling, and the 19/201 over-emission was **a correct implementation of a wrong
specification**, not a code defect. That distinction matters: it is why the fix is a redefinition
and not a bug fix.

That test broke when I removed the gate, exactly as predicted. It was **rewritten, not deleted** —
`test_geometric_gate_can_no_longer_produce_either_lane_change_token` now asserts the *inverse*
across a bracketing sweep (1.2 / 3.4 / 7.0 / 25.0 m, both directions), carries the PI's verbatim
ruling and the 14-of-18 adjudication in its docstring, and records *why* the gate went.

### My delta

| | count |
|---|---|
| `tests/test_ph1_fuse.py` before | 27 passed, 0 failed |
| `tests/test_ph1_fuse.py` after | **37 passed, 0 failed** |
| net new tests | **+10** (1 rewritten in place, 11 added) |

⚠️ **ONE REAL FAILURE WAS MINE, AND I FIXED IT.** A mid-work full run showed a 5th failure:
`test_v6_s2_loss.py::test_the_REAL_797_record_artifact_loads_with_the_published_census`. That one
was **not** an encoding artifact — my regenerated labels had landed in the globbed `labels/`
directory (§7.1). Relocated to `review/labels_v2/`; that test and `test_ph1_fuse.py` now pass
together (**75 passed**), and `labels/` is byte-for-byte back to its v1 contents.

⚠️ **The collected total drifts** (3574 → 3658 across the session) purely because agents are
ADDING tests. So the durable gate is not an absolute total but **"zero failures, and my change
adds N"**.

### The final measurement

**`cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q`, final tree state (390 s):**

> **1 failed · 3658 passed · 7 skipped · 2 xfailed**

The single failure is `test_v6_ladder_edges.py::test_after_init_from_exactly_the_intended_groups_
train[S-J]` — a file I never touched, exercising v6 stage-init groups. **Re-run in isolation it
PASSES, 4/4 parametrisations.** ⇒ a **concurrent-edit transient**: `stack/tanitad/models/v6.py` is
under another stream's active edit (the `AgentSlotDecoder` work) and was being written during my
390-second run. Not a defect, and not mine. *(This is the coordinator's own rule applied — re-run
a failure alone before reporting it — and it is the mirror image of C84: there I attributed MY
artifact to the tree, here I checked before attributing the tree's artifact to anyone.)*

The 3658 passing matches the banked `…/2026-08-16-seam-instrument/raw/pytest_after_final.txt`
exactly.

**Targeted verification of my own blast radius, all green:**

| scope | result |
|---|---|
| `test_ph1_fuse.py` | **37 passed** |
| `test_ph1_fuse.py` + `test_v6_s2_loss.py` + `test_v6_gstr_port.py` + `test_v6_factored_goal.py` | **180 passed** |

**The claim: my change adds 10 passing tests and introduces zero failures.**

**The 11 added tests, and the rule each pins:**

| test | pins |
|---|---|
| `test_geometric_gate_can_no_longer_produce_either_lane_change_token` | ruling §1 — removal, swept 1.2→25 m, both directions |
| `test_lane_target_is_never_emitted_by_any_path` | ruling §3 — geometry, VLM fallback and ROUTE_TO remap; **and asserts the token is still in `v6.STRATEGIC_GOAL_TOKENS`** |
| `test_observed_lane_change_is_recorded_as_corroboration_not_source` | ruling §4 — observation survives, `is_label_source: false`, `required` is `None` not `False` |
| `test_prepare_lane_change_needs_a_ROUTE_SERVING_requirement` | ruling §2 — emitted on a route requirement with **no** observed event; suppressed when the ego already serves the route **despite** an 8 m observed event |
| `test_lane_change_requirement_cannot_read_the_observation` | the structural half — fuzzes the observation over `{lc_left,lc_right} × {−99,−4,0,4,99} m`, token must not move |
| `test_lane_ends_requires_a_change_but_never_invents_a_direction` | args discipline on the lane-ends edge — direction slot UNSET, never a fabricated ±1 |
| `test_absent_perception_never_renders_as_a_finding_about_the_world` | C77 — three census states; **no branch may emit the literal `"no agents"`** |
| `test_scenario_line_carries_the_census_state_not_a_bare_negative` | the prose path |
| `test_empty_census_cannot_produce_a_flagged_empty_verdict` | the corroboration path |
| `test_lane_count_phrase_names_its_scope` | §5 — the carriageway scope is named, `0` is UNCLEAR not "0 lanes" |
| `test_fused_record_carries_the_machine_readable_census` | end-to-end through `main()`: `perception.census` reaches the RECORD in both states |
| `test_legacy_two_arg_scenario_line_still_reports_agents_it_has` | the legacy positional caller (`colab/s2_lab_lib.py:832`) must not be told "unavailable" with agents in frame |

---

## §7 ⛔ ESCALATIONS — decisions that need a human, not a README line

1. ⭐ **THE CORRECTED LABELS EXIST, LOAD, AND NOTHING CONSUMES THEM YET — ONE PATH CHANGE.**
   `review/labels_v2/` holds `s2_labels_aug120.jsonl` (201 records, 19 changed),
   `s2_labels_w120val.jsonl` (596, 61 changed) and a copy of `clip_index.json`.
   **VERIFIED THROUGH THE REAL PRODUCTION LOADER**, not just on disk —
   `s2_labels.load_s2_labels("review/labels_v2")` returns **797 records** with census:

   | | `LANE_TARGET` | `PREPARE_LANE_CHANGE` | `FOLLOW_MAIN_ROAD` | `HOLD_CORRIDOR` | `REDUCE_TO` |
   |---|---|---|---|---|---|
   | v1 (untouched) | **80** | **80** | 395 | 526 | 85 |
   | **v2** | **absent** | **absent** | 474 | 597 | 94 |

   v1 still loads to its published census exactly (395 / 80 / 13 / 59 / 137 / 113), which
   **proves it was not touched**. **The v1 files are deliberately UNTOUCHED** — they are the
   artifact the PI adjudicated and `PI_VERDICTS_2026-08-16.json` indexes them.
   ⇒ **`--s2-labels` must be REPOINTED at `review/labels_v2/`** before `w_s2_goal` is ever
   raised above 0. It is default-off today (`w_s2_goal: float = 0.0`), so nothing is currently
   training on the wrong labels — **that is the window in which to make the swap, and it closes
   the moment S2 supervision is switched on.**

   ⚠️ **Why its own directory and not a `_v2` suffix — MEASURED THE HARD WAY.**
   `load_s2_labels()` takes a **directory** and **globs `s2_labels_*.jsonl`**
   (`stack/scripts/s2_labels.py:268`). A `*_v2.jsonl` dropped into `labels/` was therefore read
   **alongside** v1, and the loader refused the whole set on a duplicate `clip_id` — turning
   `test_v6_s2_loss.py::test_the_REAL_797_record_artifact_loads_with_the_published_census` red.
   **The guard worked exactly as designed; my file placement was the defect.** Two lessons worth
   keeping: *a sibling file in a globbed directory is a modification of that directory*, and
   *I missed `s2_labels.py` because I searched for consumers of `s2_derive` and of
   `train_v6_staged`, never for a loader that reads the label FILES* — the "absence at one
   location is not absence" rule, applied to the wrong noun.
2. **The fused records on disk still carry the old prose.** §4 fixes the producer; a re-fuse is
   needed for `"no agents"` to disappear from existing artifacts — and that re-fuse should wait
   for the SAM3 backfill (`fa5c73b`) so it only happens once.
3. **`a_str` has no abstain channel** (§2.2). If the PI wants true abstention rather than
   `HOLD_CORRIDOR` fall-through, that needs a per-family validity mask in the batch schema — a
   trainer change I did **not** make while the v6F run is resuming.
4. **`HIERARCHY_VOCABULARY.md:67` now conflicts with the ruling.** It assigns `LANE_TARGET`'s
   evidence source as *"lateral displacement events (E4.1 LAT)"* — exactly the derivation just
   removed. Same line is mirrored in `Project Steering/Reports/2026-08-15-2200-campaign-science-
   addendum.md:54` and `S2_STRATEGIC_GAP.md:126`. **I did not edit them** (they are other
   streams' documents); they need the §LC pointer or they will re-seed the defect.
   ⚠️ Note `LANE_TARGET`'s own arg spec is `lane_offset_idx, deadline_m` — *a lane index*, which
   was never derivable from displacement either. The vocabulary always described a
   context-derived label; only the implementation was hindsight-derived.
5. **`S2_LOSS.md:148` quotes the old token census** (`FOLLOW 395 · … LANE_TARGET 80 · …`). Those
   80 are now 0. Not mine to edit; flagged.

---

## §8 Evidence classes

| claim | class | n |
|---|---|---|
| gate fires 19/201, and those are exactly the 19 reviewed rows | **MEASURED** (`lc_feat.py`) | 201 |
| recompute reproduces banked `lat_m` to 0.005 m | **MEASURED** | 19 |
| no feature separates WRONG/CORRECT at family-wise significance | **MEASURED** (exact enumeration, 3,060 assignments, 46 features × 3 horizons) | 18 |
| `lat_max(L) = 0.0997·L`; a pure arc clears the 3.0 m gate above 7.53 m/s | **MEASURED** (closed form + per-clip) | — |
| 15/19 gated clips fully explained by constant-curvature road following, same sign | **MEASURED** | 19 |
| `LANE_TARGET` 80→0, `PREPARE_LANE_CHANGE` 80→0, exactly 80/797 records change | **MEASURED** (`lc_emit.py`) | 797 |
| the other 717 records are BYTE-IDENTICAL in both families (1434/1434 blocks) | **MEASURED** | 797 |
| v2 loads through the production loader `s2_labels.load_s2_labels` | **MEASURED** | 797 |
| the 4 "pre-existing failures" I reported are a `PYTHONUTF8` artifact of my shell | ⛔ **RETRACTED → MEASURED** (C84) | 4 |
| 9 real `REDUCE_TO` labels were being suppressed by the lane-change branch | **MEASURED** | 797 |
| `"no agents"` on 119/201, of which 115 already carried `perception.absent` | **MEASURED** (`lane_count.py`) | 201 |
| on the PI's 19 reviewed clips: 10/10 `"no agents"` renderings had `perception.absent` | **MEASURED** | 19 |
| B1 `conf` = "high" on 201/201; `lanes_visible == 0` on 0/201 | **MEASURED** | 201 |
| `lanes_visible` is carriageway-scoped by definition | **MEASURED** (`ph0_v2.py:140`, read from source) | — |
| `51fd1c9f`'s true lane count | ⛔ **UNVERIFIED** — needs frame review | 1 |
| the lane count's corpus error RATE | ⛔ **NOT ESTABLISHED** — n=2 adjudicated | 2 |
| `HOLD_CORRIDOR` is the right fall-through for the 80 records | ⚠️ **HYPOTHESIS** (argued from the hierarchy split, not measured) | — |
| PhysicalAI-AV ships no map / lane graph | **PUBLISHED** (dataset card) + **MEASURED** (feature read-set) | — |

**Estimator note:** the adjudication is one binary verdict per clip, so the clip *is* the
cluster and the exact permutation test over C(18,4) is the correct instrument.
`taniteval/ci.py`'s paired episode-cluster bootstrap applies to per-window eval metrics and is
not applicable to this design; `overlapping_holdout_se` was not used anywhere.

---

## §9 Deliverables

All paths relative to the repo root; everything below is **in the repo and staged**, nothing
lives only in a scratchpad.

| artifact | path |
|---|---|
| the §LC ruling + `lane_change_requirement()` + emission removal | `stack/scripts/s2_derive.py` |
| three-state census, `lane_phrase`, wiring | `stack/scripts/ph1_fuse.py` |
| 11 new + 1 rewritten test (27 → 37 passing) | `stack/tests/test_ph1_fuse.py` |
| **this document** | `…/2026-08-16-s2-v1-labels/review/LANE_CHANGE_DEEP_REVIEW.md` |
| corrected labels + index (⚠️ **no consumer yet** — §7.1) | `…/review/labels_v2/{s2_labels_aug120,s2_labels_w120val}.jsonl` + `clip_index.json` |
| feature extraction over the ego npz | `…/review/code/lc_feat.py` → `raw/lc_feat.json` |
| exact permutation tests, 34 features (4 s) | `…/review/code/lc_stats.py` → `raw/lc_stats.json` |
| the arc-geometry mechanism + per-clip table | `…/review/code/lc_mech.py` |
| 3-horizon extension + corpus residual sweep | `…/review/code/lc_ext.py` → `raw/lc_ext.json` |
| lane-count + census audit over 201 fused records | `…/review/code/lane_count.py` → `raw/lane_count.json` |
| emission-delta measurement over 797 records | `…/review/code/lc_emit.py` → `raw/lc_emit.json` |
| the v2 label regeneration | `…/review/code/s2_relabel_v2.py` |

⚠️ **Inputs that live OUTSIDE the repo**: the bridged ego `npz` for the 201 aug120 clips and the
204 fused aug120 JSON records are read from the session scratchpad, having been pulled from
`Sayood/tanitad-ph0-aug120` by `code/s2_pull_ego.py`. They are dataset inputs, not deliverables,
and are re-pullable from HF; the derived JSON in `review/raw/` is banked so the analysis is
reproducible without them.

⛔ **Not touched, by instruction or by ownership:** `stack/tanitad/models/v6.py`, `colab/`,
`stack/scripts/ph0_sam3.py`, `Project Steering/MODEL_REGISTRY.md`,
`stack/tanitad/models/agent_slots.py`, `stack/taniteval/tools/seam_probe.py`, and the v1 label
files. ⚠️ Several of these DO show as modified in `git status` — that is **other agents' staged
work in a shared tree**, not mine.

✅ **VERIFIED for the one that matters:** the `v6.py` diff (another stream's `AgentSlotDecoder`
re-export) contains **no line touching `STRATEGIC_GOAL_TOKENS`, `STRATEGIC_ACTION_TOKENS`,
`GOAL_ARG_NAMES`, `LANE_TARGET` or `PREPARE_LANE_CHANGE`**, and the live shapes are unchanged at
**11 / 6 / 8**. ⇒ **The embedding tables are the same size they were, so the v6F S-W run's strict
resume is safe.** This was worth checking rather than asserting: "I did not edit it" and "it is
unedited" are different claims in a tree four agents are writing to.

⚠️ **REPRODUCING THE SUITE:** `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q` → **1 failed (a concurrent-edit transient, green in isolation) · 3658 passed · 7 skipped · 2 xfailed**.
**Both env vars are load-bearing** — without `PYTHONUTF8=1` four tests fail on cp1252 decoding of
UTF-8 fixtures (C84), and without `OMP_NUM_THREADS` torch spawns ~113 threads per process
(CLAUDE.md). Quote a suite result with the invocation that produced it, never the bare number.
