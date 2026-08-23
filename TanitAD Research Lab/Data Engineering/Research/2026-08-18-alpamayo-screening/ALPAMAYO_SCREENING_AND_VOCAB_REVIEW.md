# Alpamayo-2 output screening, and the tactical-vocabulary review

**Date:** 2026-08-18 · **Asked by the PI:** *screen what Alpamayo-2 is outputting and map it to our
vocabulary; do we need to add TURN_LEFT / TURN_RIGHT?*

**Evidence class:** the taxonomy below is `MEASURED` (ours) over **4,729 clips**, from
`…/2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy.json` (primary artifact `records.parquet`).
Two caveats travel with every number, from the artifact itself: **temperature 0.6, seed 42, ONE
draw per clip** — one draw is not the model's mode, and cross-draw stability is UNMEASURED; and the
clips are PhysicalAI-AV, which Alpamayo lists as **training data** — overlap UNRESOLVED. ⛔ **Alpamayo
is a TEACHER, not ground truth.**

---

## 1. WHAT ALPAMAYO-2 ACTUALLY EMITS — three axes, complete value sets

It is a **structured enum in text**: a small closed vocabulary emitted as free text on three
labelled lines. Not a typed field (no schema enforces it), not open text (the observed set is
small and closed). It must be **parsed**, and an unparsed row **counted, never coerced**.

### 1.1 LONGITUDINAL — 7 values, 0 unparsed — *magnitude-typed*

| value | n | % |
|---|---:|---:|
| Gentle Deceleration | 1,594 | 33.71 |
| Maintain Speed | 1,225 | 25.90 |
| Gentle Acceleration | 1,151 | 24.34 |
| Stop | 304 | 6.43 |
| Strong Deceleration | 267 | 5.65 |
| Strong Acceleration | 182 | 3.85 |
| Reverse | 6 | 0.13 |

### 1.2 LATERAL — 7 values, 304 unparsed — *steering-magnitude-typed*

| value | n | % |
|---|---:|---:|
| Go Straight | 2,504 | 52.95 |
| Steer Right | 1,020 | 21.57 |
| Steer Left | 649 | 13.72 |
| Sharp Steer Right | 135 | 2.85 |
| Sharp Steer Left | 115 | 2.43 |
| Reverse Right | 1 | 0.02 |
| Reverse Left | 1 | 0.02 |

### 1.3 LANE — 7 values, 304 unparsed — *lane-topological-typed*

| value | n | % |
|---|---:|---:|
| Lane Keep | 4,035 | 85.33 |
| **Turn Right** | **101** | **2.14** |
| **Turn Left** | **85** | **1.80** |
| Right Lane Change | 82 | 1.73 |
| Slightly Shift Left | 69 | 1.46 |
| Slightly Shift Right | 31 | 0.66 |
| Left Lane Change | 22 | 0.47 |

⭐ **The 304 unparsed on LATERAL and LANE are exactly the 304 `Stop` rows on LONGITUDINAL.** A
stopped vehicle emits ONE axis; the missing axes are **NOT-APPLICABLE, never a value to impute.**
That is a structural fact about the emitter, not a coverage defect — and it is the
`E_STOP_ROW_ONE_AXIS` stratum in the review sheet.

### 1.4 The number that justifies factoring, on its own

**1,921 of 4,729 clips (40.62 %)** declare a live action on **both** the longitudinal and lateral
axes simultaneously — *"which a single 5-way softmax over [lane_keep, turn_left, turn_right,
accelerate, brake_stop] cannot represent"*, verbatim from the artifact. 98 distinct joint cells are
observed. The four largest:

| longitudinal | lateral | lane | n |
|---|---|---|---:|
| Gentle Deceleration | Go Straight | Lane Keep | 891 |
| Maintain Speed | Go Straight | Lane Keep | 727 |
| Gentle Acceleration | Go Straight | Lane Keep | 534 |
| Gentle Deceleration | Steer Right | Lane Keep | 353 |

⇒ The retired 5-way head could not express **40.62 %** of what the teacher says. The factored v6
vocabulary is not a preference; it is the minimum type-correct form.

### 1.5 The reasoning field, since it decides the longitudinal mapping

`raw_json.cot` (Alpamayo's *"Chain-of-Causation"*): **100 % coverage**, 4,729/4,729 non-empty, but
only **1,103 distinct strings (23.32 %)** — median length 49 chars. `answer == cot` on **100 %** of
rows. Top string: *"Keep lane because the road ahead is clear."* (n=404). The siblings
`cot_auto_labeling` and `box` are **empty on every row**.

---

## 2. THE MAPPING TO OUR VOCABULARY — where it lands, and where it does not

v6's tactical action vocabulary (`stack/tanitad/models/v6.py`, the single source, imported by
REF-C v3 and REF-A v1):

* `TACTICAL_LAT_ACTIONS` = LANE_KEEP · LANE_CHANGE_L · LANE_CHANGE_R · ABORT_LC · NUDGE_L · NUDGE_R
* `TACTICAL_LON_ACTIONS` = FOLLOW · CRUISE · YIELD_MERGE · BRAKE_TO · CREEP · HOLD

| Alpamayo axis | our axis | status |
|---|---|---|
| LANE → `a_tac_lat` | lane-topological → lane-relative | **maps, except TURN_\*** (§3) |
| LONGITUDINAL → `a_tac_lon` | magnitude-typed → **reason-typed** | ⛔ **TYPE MISMATCH — does not map at all** |
| LATERAL (steering magnitude) | — | ⭐ **not mapped, and correctly so** (§2.2) |

### 2.1 The longitudinal axis does not map, and that is a type fact

`TACTICAL_LON_ACTIONS` is **reason-typed** (FOLLOW, YIELD_MERGE, BRAKE_TO…) while Alpamayo's axis is
**magnitude-typed** (Gentle Deceleration, Strong Acceleration…). *"Gentle Deceleration"* cannot say
whether the ego is FOLLOWing a lead or BRAKE_TO a stop line — **the reason decides, and the reason
lives in `cot`**. Every value therefore maps to `REASON_REQUIRED`: the leg casts **no** longitudinal
vote, and the axis value plus reason are recorded for a separate meta-action × reason label builder.

⚠️ The old substring path *did* cast a vote here, from a lottery over truncated `cot` text.
Removing it was the honest correction, not a regression.

### 2.2 ⭐ The steering-magnitude axis needs no token — it is already in the action space

Alpamayo's LATERAL axis (Go Straight / Steer / Sharp Steer, L/R) is a **magnitude**. Our action
space is `(a, κ)` at 10 Hz through a unicycle: **κ *is* steering magnitude, as a continuous
control**. Adding a discrete steering-magnitude token would re-encode, coarsely, a quantity the
planner already optimises continuously — the same principle by which jerk is a *parameterisation*
rather than a cost term. ⇒ **Recommend: no token. Use it as a validation signal** (does the planned
κ agree in sign and band with the teacher's declared steering?), which is free and is a real check
on the tactical decode.

---

## 3. ⭐ THE TURN QUESTION — the review, and a recommendation

**The gap, exactly:** 186 of 4,729 clips (**3.94 %**) carry LANE ∈ {Turn Left, Turn Right}, for
which `TACTICAL_LAT_ACTIONS` has **no member**. Today they are declared `NO_V6_TOKEN`: reported,
never coerced. They are not silently mislabelled — but they are **unusable**, and they are
junctions, i.e. the scenarios where tactical decisions matter most.

### 3.1 The argument against adding it — and it is a real argument

v6 already represents turns at the **strategic** level (route tokens). Putting TURN on the
**tactical lateral** axis mixes a **route-level, topological, 5–20 s** decision into an axis whose
other five members are **lane-relative, 2–6 s** actions. That is the same category error as the
retired 5-way head, committed on a different dimension — and this programme has paid for that once
already (D-TAC1: 9.68 % of longitudinal decisions destroyed by exactly such a conflation).

### 3.2 The argument for — and why I think it wins

1. **"Lane keep" is not true inside a junction.** A junction has no lanes to keep. The alternative
   to a TURN token is either a **false** LANE_KEEP or a permanent abstention on 3.94 % of the
   corpus. Neither is a representation.
2. **The strategic route token answers a different question.** Strategic says *which arm of the
   junction* the route takes; tactical says *what the vehicle is doing in the next 2–6 s*. During a
   turn these coincide in the word "left" and differ in referent, horizon, and update cadence.
   Coincidence of vocabulary is not identity of meaning.
3. **The type objection is answerable by declaration.** TURN_L/TURN_R are the only lateral tokens
   whose meaning is *conditioned on the strategic route being set*. That coupling can be **documented
   and tested** rather than hidden — which is strictly better than the current state, where the
   coupling exists in the world and is absent from the vocabulary.
4. ⭐ **The change is free TODAY and expensive on any later day.** Nothing has yet been trained on
   the factored vocabulary: REF-C v3 and REF-A v1 both read `N_LAT`/`N_LON` from `v6.py` and neither
   has a checkpoint. Widening 6 → 8 costs one edit and two head shapes now; after the first factored
   checkpoint it costs a retrain of every arm that carries the head.

### 3.3 The honest counterweight: 186 examples is thin

At 6 → 8 classes, TURN_L would hold **85** training examples and TURN_R **101**, before any split.
On a 40-episode val set the expected count is ~2 per class — **not enough to score**. ⇒ If adopted,
the tokens must be **admitted as representable but NOT as scoreable**: they may be emitted and
supervised, and any per-class metric on them must be refused for under-power rather than reported.
That is the same discipline `cost_fidelity` applies to n < 200.

### 3.4 RECOMMENDATION — adopt, with three conditions

> **ADD** `TURN_L`, `TURN_R` to `TACTICAL_LAT_ACTIONS` (6 → 8), **conditional on:**
> 1. a docstring stating that these two tokens are junction-traversal and are **route-conditioned**,
>    unlike the other six;
> 2. a test pinning that no per-class metric is reported for them below n = 200 (representable, not
>    scoreable);
> 3. the change landing **before** any factored checkpoint exists — i.e. now, or not until the next
>    planned retrain of all three lines.

⛔ **NOT APPLIED.** The vocabulary is binding across v6, REF-C v3 and REF-A v1; widening it is the
PI's call, not an agent's. The one-line diff and the two head shapes are ready.

### 3.5 What is NOT recommended

* **Do not** add a steering-magnitude token (§2.2) — κ already carries it, continuously.
* **Do not** add `Reverse` (6 clips, 0.13 %) or `Reverse Left/Right` (1 clip each) — below any
  usable count, and reverse is out of the operational domain the arms are evaluated in.
* **Do not** map the longitudinal axis by magnitude (§2.1) — it is a type error, and the previous
  substring path that did so was removed on purpose.

---

## 4. WHAT THIS SCREENING LEAVES OPEN

1. **One draw per clip at temperature 0.6.** Cross-draw stability is UNMEASURED. Before any of these
   proportions is quoted as a property of Alpamayo rather than of one sample, re-draw a subset with
   a different seed and report the agreement. Cheap, and it bounds every number above.
2. **Training-data overlap is UNRESOLVED.** PhysicalAI-AV is listed as Alpamayo training data; these
   labels may be partly memorised. This makes the teacher usable for *vocabulary coverage* (what can
   it express) and questionable for *accuracy* claims.
3. **The 2026-08-16 visual review is unadjudicated** — 0/42 clips judged. It is the instrument that
   would tell us whether the mapped labels are *right*, as opposed to *representable*. §3's
   recommendation rests on representability only.
4. **The `cot` field is 100 % covered but only 23 % distinct** — reason-typed longitudinal labels
   built from it will inherit that concentration, and the top string alone covers 404 clips.
