# PI review of the S2 v1 strategic labels — two independent defects, one already fixed at source

**Reviewer:** Sayed (PI), 2026-08-16. **Primary source:** `PI_VERDICTS_2026-08-16.json` (this
directory). **Reviewed section:** `S1_LANE_TARGET` only — 19 clips. Every other section
(`S2_TURN`, `S3_STOP_AT`, `S4_ROUTE_TO_DISPOSITION`, `S5_EXCLUDED_VAL`, `S6_FOLLOW_CONTROLS`)
was left `v=null`. ⛔ **`null` is UNREVIEWED, not agreement** — 25 of the 44 rows carry no
judgement and nothing here may be read as validating them.

## The tally

| verdict | n |
|---|---|
| `wrong` | 14 (13 explicit + `4bea0a51`, whose note says wrong while `v` stayed null) |
| `correct` | 4 |
| blank | 1 |

**≈ 78 % of the adjudicated LANE_TARGET labels are wrong.** The PI's summary: *"almost all
wrong … we need to deeply review the strategic actions of preparing lane change, in lot of
scenes no agents reported"*.

⚠️ This section is exactly the one the pre-review analysis had already flagged as carrying
**0/19 VLM corroboration**. The instrument that said *"these are unsupported"* was right; the
open decision was keep-vs-re-emit, and this review settles it.

---

## Defect 1 — "no agents" is not a perception judgement. It is an empty-string fallback.

`stack/scripts/ph1_fuse.py:357`:

```python
", ".join(f"{v} {k}" for k, v in sorted(census.items())) or "no agents",
```

When the detection census is empty the join yields `""`, which is falsy, so the `or` substitutes
the literal string **"no agents"**. The pipeline never concluded that no agents were present — it
had nothing to say, and the absence was rendered as a positive claim.

**Root cause: the SAM3 dtype bug** (`vitdet.py:71` → `perflib/fused.py` force-casts the ViT MLP to
bf16 while `fc2` stays fp32; the image path enters no autocast context). SAM3 raised on every
concept of every frame, so `census` was empty for **every clip in the corpus**. The PI's
observation on `03ba450b` — *"in the picture of frame 9 there is clear a car as incoming
traffic"* — is that failure surfacing in the labels.

**Status: FIXED AT SOURCE** (commit `fa5c73b`), and the corpus is re-detecting: 2,299 detections
where there were 0, zero errors, zero dead-control failures. ⚠️ **The fused labels have NOT been
re-fused yet** — they still carry `perception.absent`. Until the re-fuse lands, every "no agents"
in any S2 artifact is this fallback and must not be read as evidence.

⇒ **DURABLE FIX, independent of SAM3:** an absent measurement must never be rendered as a
confident negative. `"no agents"` and `"perception unavailable"` are different claims and the
schema must be able to say which. This is the C77 family — an empty result presented as a
finding — reappearing one layer up, in prose rather than in a count.

---

## Defect 2 — lane-change over-emission. **My proposed mechanism is REFUTED.**

One gate drives both tokens: `_gated_lc_event()` in `stack/scripts/s2_derive.py:143` feeds
`LANE_TARGET` (g_str, line 257) *and* `PREPARE_LANE_CHANGE` (a_str, line 370). A single false
positive therefore corrupts both the strategic goal and the strategic action.

**Base rate (MEASURED, `engine_a_aug120.jsonl`, n=201):** the gate fires on **19/201 = 9.5 %** of
clips. At the PI's ≈78 % error rate the true lane-change incidence is ≈2 %. The gate is not
mis-tuned at the margin; it is firing mostly on clips with no lane change at all.

### The hypothesis I tested, and why it failed

**HYPOTHESIS:** `ph0_pilot.engine_a_summary` builds `pts` in a frame pinned to the t0 heading for
the whole clip, so a vehicle merely following a bend accumulates lateral offset without bound and
the ≥3.0 m gate cannot separate "changed lane" from "road curved". Prediction: the WRONG clips
carry higher curvature than the CORRECT ones.

**MEASURED (`peak_kappa_per_m`, per clip):**

| group | κ median | κ mean | max\|lat_m\| median | n |
|---|---|---|---|---|
| WRONG | 0.0033 | 0.0038 | 4.45 m | 13 |
| CORRECT | 0.0043 | 0.0263 | 4.21 m | 4 |

**REFUTED.** The groups do not separate — the CORRECT median is *higher*, and the CORRECT range
(0.0015–0.0954) fully contains the WRONG range (0.0017–0.0061). Lateral displacement is likewise
indistinguishable (4.45 vs 4.21 m). Every clip in both groups has `route=follow`.
⚠️ n=4 in the CORRECT arm is too small for a powered test — but there is no effect visible to be
underpowered about, and the two distributions overlap completely.

*(A corpus-level comparison shows fired clips with LOWER curvature than non-fired ones, 0.0033 vs
0.0129 — but that comparison is CONFOUNDED and must not be quoted: the gate requires
`route=follow`, so the non-fired group contains every turning clip. Only the within-fired
WRONG-vs-CORRECT contrast above is admissible.)*

### What the refutation actually establishes

**The features the detector uses do not contain the distinction the PI is making.** Curvature,
windowed lateral displacement, route token and event count are statistically identical between
the labels he calls right and the ones he calls wrong. This is not a threshold that needs
retuning — it is a **measurement that does not carry the answer**.

The reason is structural: *a lane change is defined relative to lane boundaries*, and ego pose
does not contain them. Four metres of lateral movement is a lane change on a narrow two-lane
road and ordinary within-lane drift on a wide one; the ego track is identical in both cases.
**PhysicalAI-AV supplies no map, lane graph, or lane-boundary annotation** — the dataset card
states verbatim that open maps data is not included, and `obstacle.offline`'s 10-class enum over
87,481 cuboids is all dynamic agents. There is no lane reference anywhere in our inputs.

⇒ **`LANE_TARGET` / `PREPARE_LANE_CHANGE` are not derivable from the ego track**, and no tuning
of `LC_MIN_LAT_M` will make them so. The PI's third observation is the same gap from the other
side: *"the pipeline is saying there is one lane, better: there are two lanes"* and *"3 lane
wrong"* — the lane **count** is wrong too, and that is the VLM leg, not the geometry leg.

### ⛔ BINDING — THE GEOMETRIC GATE IS REMOVED; THE LABEL BECOMES REASON-BASED (Sayed, 2026-08-16)

**Sayed, verbatim:** *"stop emitting lane_target/Prepare Lane change from geometric gate. Prepare
lanbe change should onyl be to follow route or follow the main road (no route set), it must be
derived from the context"*

| | |
|---|---|
| ⛔ **Removed** | the geometric gate `_gated_lc_event()` as a source of `LANE_TARGET` (`s2_derive.py:257`) and `PREPARE_LANE_CHANGE` (`:370`). The decision is REMOVAL, not a retuned `LC_MIN_LAT_M`. |
| ✅ **Admissible** | `PREPARE_LANE_CHANGE` **only in service of route-following**: (a) a route is set and the ego's current lane does not serve it, or (b) no route is set, `FOLLOW_MAIN_ROAD` applies, and the current lane does not continue as the main road (lane ends, forced merge, exit-only). |
| ✅ **Source** | **CONTEXT** — the scene and the route. Never observed lateral displacement. |

⭐ **WHY THIS DISSOLVES THE IDENTIFIABILITY FAILURE.** The refuted derivation asked *"did the ego
move sideways?"* — a question the ego track can answer, but whose answer does not separate a lane
change from within-lane drift (that is exactly the refutation above). The ruling replaces it with
*"does the route require the ego to be in a different lane?"* — a question about context, which is
answerable, and which is the thing the strategic layer actually has to decide. The label stops
being an **observation** and becomes a **reason**.

⚠️ **This INVERTS the pipeline's derivation philosophy for this token, and that must be stated
wherever it is used.** Every other S2 label is derived from what the ego DID. This one is derived
from what the route DEMANDS. The two can disagree — a driver may fail to reposition in time, or
may change lane for reasons outside the route (overtaking, courtesy). ⇒ **The ego's observed lane
change becomes CORROBORATION, never the source**, recorded exactly as the VLM's judgement is.
"Observed but not required" is a real category and must not silently become a positive label.

⇒ **`LANE_TARGET` leaves `g_str` emission entirely.** Under this definition a lane change is a
MEANS, not a GOAL: the strategic goal is `FOLLOW_MAIN_ROAD` / `TURN_*` / `EXIT_*`, and
repositioning is the action serving it. `PREPARE_LANE_CHANGE` remains an `a_str` action under the
route-serving condition.

⚠️ **The honest failure mode to guard against:** the context this needs is lane count, ego lane
index, and which lane serves the route — and the PI's own review shows the VLM's lane count is
currently wrong (*"one lane"* where there are two; *"3 lane wrong"*). **Do not replace an
unidentifiable label with an unreliable one.** Where the context is not yet trustworthy the
correct emission is `NONE_ABSTAIN` plus a written specification of what must be built.

### Remaining recommendations (not yet ruled on)

1. **Rebuild the context signal from VISION, where lane boundaries actually are** — lane
   structure and lane-marking geometry, not ego displacement. This also aligns with the standing
   rule that inference is vision-only; here ego is not inadmissible, it is simply *insufficient*.
3. **Re-fuse after the SAM3 backfill lands** and re-derive, so Defect 1 stops contaminating every
   downstream judgement — including the human review, which had to look past "no agents" on
   every clip.
4. **Re-run the review on the other five sections**, which remain entirely unadjudicated. The
   ≈78 % error rate is established for LANE_TARGET *only*; generalising it to `TURN_*`,
   `STOP_AT` or `FOLLOW_MAIN_ROAD` would be the same over-reach this document is correcting.

## Evidence classes

| claim | class |
|---|---|
| verdict tally, PI notes | **MEASURED** (human adjudication, primary source in this directory) |
| `"no agents"` is an `or`-fallback on an empty census | **MEASURED** (`ph1_fuse.py:357`, read from source) |
| SAM3 returned zero detections corpus-wide pre-fix | **MEASURED** (census, 25/25 clips, and the fix's 0→2,299) |
| gate fires 19/201 | **MEASURED** (`engine_a_aug120.jsonl`, n=201) |
| curvature explains the WRONG/CORRECT split | ⛔ **REFUTED** (this document) |
| lane boundaries absent from PhysicalAI-AV | **PUBLISHED** (dataset card) + **MEASURED** (feature read-set) |
| true lane-change incidence ≈2 % | **ESTIMATED** (9.5 % base rate × 78 % error, n=18 adjudicated) |
