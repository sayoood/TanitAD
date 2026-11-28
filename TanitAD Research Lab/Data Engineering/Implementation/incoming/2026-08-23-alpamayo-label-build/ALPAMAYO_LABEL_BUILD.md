# The Alpamayo augmentation now yields a real tactical + strategic label set — 4,723 clips, zero GPU

**Owner:** tactical/strategic label-augmentation stream · 2026-08-23 · branch
`claude/exciting-kepler-4d4449`

**PI correction that produced this package (2026-08-23):** *"these are not the
tactical or strategic labels. And we said at first step try to leverage maximally
the alpamayo augmented data."* Both correct. The prior package measured corridor
geometry (metres of curve-fit residual), which is **not a token**; and the
Alpamayo mapping table existed in code but **had never been run over the
4,729-clip distribution**. This package does the actual first step.

---

## 0. TL;DR

| field | filled | tokens emitted |
|---|---|---|
| **`a_tac_lat`** | **4 723 / 4 723 = 100.00 %** | `LANE_KEEP` 85.35 % · `NOT_APPLICABLE` 6.42 % · `TURN_R` 2.14 % · `TURN_L` 1.80 % · `LANE_CHANGE_R` 1.74 % · `NUDGE_L` 1.44 % · `NUDGE_R` 0.66 % · `LANE_CHANGE_L` 0.47 % |
| **`g_str`** | **4 723 / 4 723 = 100.00 %** | `FOLLOW_MAIN_ROAD` 83.82 % · `STRAIGHT_THROUGH` 7.77 % · `STOP_AT` 4.47 % · `TURN_RIGHT` 2.14 % · `TURN_LEFT` 1.80 % |
| **`a_tac_lon`** | **1 648 / 4 723 = 34.89 %** | `FOLLOW` 19.29 % · `CRUISE` 7.83 % · `YIELD_MERGE` 3.94 % · `BRAKE_TO` 3.83 % |

⭐ **VOCABULARY COMPLETENESS: 0 unmappable phrases.** Every phrase on every axis of
the 4,729-clip distribution reaches a v6.1 token. This is the coverage measurement
`HIERARCHY_VOCABULARY.md` §4 asks for, and it **passes**.

**Cost: zero GPU, zero VLM, ~2 s of CPU.** The augmentation was already on disk.

---

## 1. Why this had not been run

`tac_str_labels.compose` has an Alpamayo-only path — `lon_from_alpamayo` (via the
`cot` field) and `strategic_from_alpamayo` (via `lateral`/`lane`). But its **only
caller** is `vlm_tac_compose.py`, which *requires* `--raw` (VLM generations) and
`--payload`. So the free, already-banked Alpamayo tier sat **gated behind a VLM leg
costing 25.6 T4-days** that is only partially run and has **0/42 human-reviewed
labels**.

`stack/scripts/alpamayo_label_build.py` removes that gate: **Alpamayo first, VLM as
enrichment later.**

---

## 2. Controls — all passing

| control | result |
|---|---|
| `control_leak` | **PASS** — 6 val40-overlapping clips excluded from the committed `alpamayo_val40_exclusions.json`, **0 surviving** in the output |
| `control_axis_totals` | **PASS** — every per-axis count sums to `n = 4 723` |
| `control_vocab_pin` | **PASS** — `[]` unknown tokens; every emitted token is a live member of the v6.1 tuples |
| `control_constant_only` | **PASS** — majority shares 0.8535 / 0.6511 / 0.8382, **no field degenerate** |

Verified by **content, not by file existence** (the C77 lesson): 4 723 rows read
back, values non-empty, and **every abstention carries a reason** — after the §3
fix, with zero exceptions.

---

## 3. The silent-abstention defect — FOUND, VISUALLY VALIDATED, FIXED

**Found.** `compose` returned `g_str = None` — the FIELD, not a `LabelField` — on
its fall-through, **bypassing** `LabelField.__post_init__`'s guard whose own text
reads *"an absent value MUST carry a reason"*. **1 944 of 4 723 clips (41.16 %)**
had `g_str` absent with an empty reason — the largest `g_str` category, explaining
nothing. 85 % of the gap was `lane == "Lane Keep"` with `lateral` a STEER.

`ROOT-CAUSE CLASS`: **a validity guard that lives on the object, bypassed by the
path that never constructs the object.**

### 3.1 VISUAL VALIDATION — and it REFUTED my proposed fix

I proposed mapping the whole Steer population to `FOLLOW_MAIN_ROAD`. The PI asked
for visual validation first. **10 clips rendered** from the parity corpus
(`frames/`, 3 frames each: t=0, band open, band end) with ego geometry
(`raw/steer_geom.json`). The population is **MIXED**:

| clip | measured | what the frames show | my blanket mapping |
|---|---|---|---|
| `24ae03a3` | -7.7 deg over 104.7 m @ 15.3 m/s | rural road, gentle curves — genuine route-following | correct |
| `092ad43d` | **-64.6 deg over 13.9 m @ 2.0 m/s** | **a JUNCTION — give-way triangle in frame 2, ego emerges onto a different road in frame 3** | **WRONG** |
| `14bc3af3` | **+79.0 deg over 44.7 m @ 5.1 m/s** | night rural road, sharp bend, curve-warning sign | turn-magnitude |

⛔ **`092ad43d` is visibly a junction turn and Alpamayo labelled it
`lane = "Lane Keep"`.** A goal-level turn/no-turn split would have been wrong on
Alpamayo's own field. **My assumption was refuted by looking.**

At scale (n = 201 Alpamayo-in-parity clips with poses): the Steer population's
`|dyaw|` is p50 **10.6 deg**, p90 **72.1 deg**, and **34 % exceed 20 deg** — while
Alpamayo's OWN declared turns exceed 30 deg only **20 %** of the time (n = 10). The
`lane` field is not a reliable turn detector in our window.
**CONTROL PASSES**: declared turns p50 12.2 deg vs `Lane Keep` p50 2.9 deg —
**4.3x separation**, so `|dyaw|` does carry turn signal. n = 10 declared turns is
far too small to set a threshold from, and none is set here.

### 3.2 The PI's design resolves it, and that is WHY it is right

**PI 2026-08-23:** *"no abstension"* and *"the strategic goals should follow the
goal follow route and the related actions turn left and turn right in x m."*

* **GOAL stays route-following** — `FOLLOW_MAIN_ROAD` for the whole gap
  population. It is already THE declared default when no navigation route is set
  (PI 2026-08-11).
* **The turn becomes an ACTION** — `TURN_LEFT`/`TURN_RIGHT(within_m)` in `a_str`,
  derived from **geometry**, not from Alpamayo's `lane`.

That is exactly what makes the `092ad43d` mislabelling **harmless**: the goal is
correct either way, and the manoeuvre is carried by a channel that does not
consult the unreliable field. A goal-level split would have propagated the error.

### 3.3 Fixed

`strategic_from_alpamayo` now covers any `lateral` under `lane keep`, gives the
lane-change path a route-following goal, and its fall-through emits
`FOLLOW_MAIN_ROAD` **with a reason** instead of a hole.

⇒ **`g_str`: 58.84 % -> 100.00 %, silent absences 1 944 -> 0.** Not degenerate
(majority share 0.8382).

⛔ **AND A GUARD THE FIX ALMOST BROKE.** Adding the default made
`strategic_from_alpamayo` return a goal on the lane-change path, which won the
earlier branch and rendered the `lane_target_is_admissible` **contradiction
refusal unreachable** — caught immediately by
`test_declared_lane_change_refuses_a_contradicting_lane_target`. *"No abstention"*
removes abstentions that are **GAPS**, never those that are **CONTRADICTIONS
between legs**; `LABEL_PIPELINE_CONFIRMATION.md` §2 calls that case *"the design
WORKING"*. The refusal is now an **OVERRIDE**, not a branch, so no future default
can defeat it by ordering. 181 tests green across the three consumer suites.

### 3.4 ⛔ STILL OWED — the `a_str` turn action

`TURN_LEFT`/`TURN_RIGHT(within_m)` is **not yet implemented**, for two reasons the
PI should see:

1. **`STRATEGIC_ACTION_TOKENS` has no turn token** — it is
   `(PREPARE_LANE_CHANGE, HOLD_CORRIDOR, REDUCE_TO, PREPARE_EXIT, PREPARE_STOP,
   RESUME_CRUISE)`. Adding turns **sizes an embedding table** and the live v6F run
   resumes tensor-strict, so it must be an **APPENDED v6.2 tuple behind a version
   switch defaulting to v6.1** — the `TACTICAL_LAT_ACTIONS_V61` discipline
   verbatim. That is a vocabulary change, hence escalated not made.
2. ⚠️ **`within_m` needs POSES, and only 201 of 4 723 labelled clips (4.3 %) are
   in the parity corpus.** The GOAL needs no poses and reaches 100 %; the ACTION
   is capped at 4.3 % until the remaining Alpamayo clips are ingested. **This is
   the binding constraint on the turn action and it is not a code problem.**

## 4. ⚠️ Limits — what this does NOT license

* **No correctness claim.** This measures **coverage and provenance**, not accuracy.
  **No human has reviewed a single one of these 4 723 labels**; the 42-clip visual
  review still stands at **0/42**. Coverage is not correctness.
* **`a_tac_lon` at 34.89 % is the Alpamayo-only floor**, not the ceiling. The 12-clip
  three-tier sample reached 83.3 % *with* the VLM leg. The VLM is enrichment on top
  of this, and its value is now quantifiable: **+48 pp on the longitudinal axis**.
* **The window-alignment defect is NOT addressed here.**
  `…/2026-08-19-alpamayo-screening/WINDOW_ALIGNMENT_DEFECT.md` measured that
  Alpamayo's *magnitude* describes a different interval than our 2–6 s window
  (sign agreement 2/12 on our window vs 7/12 on the first half) and that **4/12
  clips contradict the poses on every window tested**. These labels use Alpamayo's
  **class and reason**, not its magnitude, which is the axis that defect attacks —
  but the residual contradiction rate is a **known label-noise floor** and is
  carried, not closed.
* **`g_tac` (tactical GOALS) is still 0 %** — untouched by this package. These are
  tactical *actions* (`a_tac_lat`/`a_tac_lon`) and strategic *goals* (`g_str`).

---

## 5. ⛔ ESCALATION — one decision, one confirmation

1. **DONE — `Steer L/R` -> `FOLLOW_MAIN_ROAD`** per the PI directive; `g_str` is
   now 100 % filled with 0 silent absences (§3.3).
2. **Approve the v6.2 `a_str` tuple** (append `TURN_LEFT`, `TURN_RIGHT`) so the
   turn action can be built — §3.4. Note its 4.3 % pose-coverage cap.
3. **Ingest more Alpamayo clips into the parity corpus?** 4 522 of 4 723 labelled
   clips have no poses, so no geometric argument (`within_m`, turn timing) can be
   derived for them. This is the ceiling on every argument-bearing token.
4. **Confirm before I wire `g_tac_geom` into `vlm_tac_compose`** — unchanged from
   the previous package; it alters the label-stream schema.

---

## 6. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| the label builder (Alpamayo tier, 4 controls, unmappable log) | `repo:stack/scripts/alpamayo_label_build.py` | no — staged |
| ⭐ **the label set — 4 723 clips** | `repo:…/2026-08-23-alpamayo-label-build/raw/alpamayo_labels_v61.jsonl` | no — staged |
| the census (coverage, legs, flags, controls, absence characterisation) | `repo:…/raw/alpamayo_label_census_v61.json` | no — staged |
| **visual validation — 10 rendered clips** (3 frames each) | `repo:…/frames/*.jpg` | no — staged |
| ego geometry for those 10 (dyaw, arc, lateral end) | `repo:…/raw/steer_geom.json` | no — staged |
| turn-vs-curve separation over 201 parity clips + control | `repo:…/raw/turn_split.json` | no — staged |
| the two probe scripts | `repo:…/code/extract_frames.py`, `…/code/turn_split.py` (also `thor:~/gtac/`) | no — staged |
| the `strategic_from_alpamayo` fix + contradiction override | `repo:stack/tanitad/lake/tac_str_labels.py` | no — staged |
| this document | `repo:…/2026-08-23-alpamayo-label-build/ALPAMAYO_LABEL_BUILD.md` | no — staged |

Inputs consumed (pre-existing, unmodified): `…/2026-08-16-tactical-labels/raw/
a1_alpamayo_taxonomy_per_clip.jsonl` (4 729 rows) and
`…/2026-08-18-alpamayo-screening/alpamayo_val40_exclusions.json`.

**Nothing lives on only one disk. Staged, never committed, never pushed.**
