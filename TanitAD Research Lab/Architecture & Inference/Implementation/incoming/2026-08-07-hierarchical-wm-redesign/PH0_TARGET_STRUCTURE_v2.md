# PH0 target structure v2 — a schema a 9B can actually satisfy

**PI instruction (2026-08-12):** *"Review your prompt and provide the target structure for
the models, it must be solvable."*

**Status:** DESIGN. Supersedes the `ph0-v0.1` single-shot schema for engine B.
**Decided stack (PI, 2026-08-12):** engine B = **Qwen3.5-9B**, engine C = **SAM3**,
engine A = **algorithmic integrated ego path + dynamics**, engine D = Alpamayo records.

---

## 1. The measurement that forces this

MEASURED on pod4, 8-clip mini pilot, arm `Qwen/Qwen3.5-9B`, after both infrastructure bugs
were fixed (decode threads pinned, correct vision auto-class):

| quantity | value |
|---|---|
| clips run to completion | 5 / 8 |
| clips **schema-valid with a passing action** | ⛔ **1 / 8** |
| hard failures, all identical | 3 / 8 — `no parseable JSON object in model output` (after retry) |
| wall clock | 82–185 s/clip · VRAM peak 18.0 GB |

The model is not the bottleneck: it loads, it watches 40 frames, it reasons for ~2.5 minutes,
and it emits text. **The bottleneck is what we ask it to emit.**

## 2. Why `ph0-v0.1` is not solvable in one shot

`PROMPT_PASS1` requests **ONE strict JSON object** carrying:

| burden | detail |
|---|---|
| top-level sections | **5** (`scenario`, `domain`, `signs`, `strategic.goal`, `strategic.actions`) |
| nesting depth | **4** (`strategic.actions[].constraints.within_m`) |
| unbounded arrays | **2** (`scenario.agents[]`, `signs[]`, plus `strategic.actions[]`) |
| distinct field types | ~30 |
| closed vocabularies to respect simultaneously | 5 (11 goal kinds · 6 verbs · 4 constraint slots · 5 sign kinds · 3 sources) |
| per-item pixel regression | `bbox:[x0,y0,x1,y1]` + `frame_idx` on **every** sign and **every** agent |
| free-form generation | `text_ocr` verbatim, `behaviour`, `position_rel`, `reason` |
| continuous values | `confidence` floats in [0,1] × every goal, action and domain |
| prose rules to hold in working memory | **6 binding rules**, incl. a conditional (`route_to` only with `evidence.sign_idx`) |
| output budget | `max_new_tokens = 2048` — long enough that a truncation lands mid-object |

Every one of those is individually reasonable. Together, in one autoregressive pass with no
grammar constraint, the probability of emitting a *fully* conforming object is the product of
~30 independent chances to slip — and one slip anywhere invalidates the whole record. That is
exactly the observed 1/8.

**⛔ The deeper design error, and it is ours, not the model's:** the schema asks the VLM for
three different *kinds* of thing at once, only one of which it is good at.

| the ask | who should actually answer it |
|---|---|
| `constraints.within_m` · `by_time_s` · `at_arc_m` · `hold_for_s` · `args.v_target_ms` | ⛔ **Engine A.** These are METRIC quantities. Engine A integrates the ego path and computes them exactly; a VLM estimating metres from video is *hallucinating by construction*. |
| `bbox` · `frame_idx` · agent extents | ⛔ **Engine C (SAM3).** Pixel-accurate localisation is a segmentation job. VLM bbox regression is its weakest modality. |
| `illumination` · `weather` · `road_type` · `domain` · sign `kind`/`state` · **`text_ocr`** · goal `kind` · action `verb` | ✅ **Engine B.** Symbol choice and text reading — what a VLM is uniquely for. |

⇒ **The organising principle for v2: the VLM chooses SYMBOLS, the algorithm supplies NUMBERS,
SAM3 supplies PIXELS.** v0.1 asked the VLM for all three and then measured it on the union.

## 3. The target structure — 4 small calls, each independently checkable

Each call is a **flat object**, **bounded arity**, **enums over free text**, and a **small
token budget** so truncation cannot chop a record. Every call is separately valid or invalid,
so one bad call costs one field, not the whole clip.

### B1 — SCENE (1 call/clip · ~60 tokens · no arrays, no nesting)
```json
{"illumination":"day|dusk|night|dark",
 "weather":"clear|rain|snow|fog|unclear",
 "road_type":"highway|urban|rural|junction|unclear",
 "domain":"highway|urban|roundabout|intersection|rural|unclear",
 "lanes_visible":0,
 "lane_ego":0,
 "conf":"low|med|high"}
```
7 scalars, all closed. `lanes_visible`/`lane_ego` are small ints 0–6. **`conf` is a 3-level
enum, not a float** — a float is a free generation site with no verification value; three
levels carry the same decision content and cannot be malformed.

### B2 — SIGN READING (1 call/clip · ≤6 items · ~150 tokens · NO bboxes)
```json
{"n_signs":0,
 "signs":[{"kind":"light|speed|nav|stop|yield|other",
           "state":"red|amber|green|none",
           "text":"",
           "applies_to_ego":true}]}
```
`n_signs` is emitted **first** so the array length is committed before the items — this alone
removes the commonest truncation failure. `text` is verbatim OCR or `""` (the one free-form
field we genuinely need and cannot get elsewhere). **No `bbox`, no `frame_idx`** — grounding
moves to B3/C.

### B3 — SIGN GROUNDING (1 call **per sign** · ~30 tokens)
```json
{"sign_idx":0,"frame_idx":0,"visible":true,"bbox":[0,0,0,0]}
```
One sign, one frame, one box. This is also the **hand-off point to SAM3**: B3 gives a coarse
box, SAM3 refines it to a mask, and disagreement between them is a *measurable* grounding
signal instead of an unverifiable claim. If SAM3 is available, B3 may be skipped entirely and
SAM3 prompted with the sign's `kind`+`text` directly.

### B4 — TACTICAL/STRATEGIC SYMBOLS (1 call/clip · ~120 tokens · **no metrics**)
```json
{"goal_kind":"keep_corridor|lane_target|exit_left|exit_right|turn_left|turn_right|
              straight_through|route_to|stop_at|follow_main_road|none_abstain",
 "goal_evidence_sign":null,
 "actions":[{"verb":"prepare_lane_change|hold_corridor|reduce_to|prepare_exit|
                     prepare_stop|resume_cruise",
             "direction":"left|right|none"}],
 "conf":"low|med|high"}
```
⭐ **Every metric slot is gone.** No `within_m`, `by_time_s`, `at_arc_m`, `hold_for_s`,
`v_target_ms`, no `source`, no per-action `confidence`, no `reason`. The VLM picks the verb;
**fusion attaches Engine A's measured numbers to it.** `source` is then *derived* — `signage`
if `goal_evidence_sign` is non-null, `path` if only Engine A supports it, `vlm-fused` if both.

### A → the numbers (no VLM involved)
Engine A already integrates the ego path; it supplies, per chosen verb:
`v_target_ms` (from the speed profile), `within_m` / `at_arc_m` (arc length to the manoeuvre),
`by_time_s` / `hold_for_s` (from the time base), plus the dynamics the PI named — speed,
acceleration, yaw-rate and curvature along the integrated path. These are **measurements**,
so they need no verification pass.

## 4. What makes it solvable, stated as mechanisms

1. **Grammar-constrained decoding is mandatory, not an optimisation.** With a JSON-schema FSM
   over the token stream, `no parseable JSON object` becomes *impossible by construction* —
   it eliminates the entire 3/8 hard-failure class rather than retrying it. Every field above
   is a closed enum, a bounded int, a bool, or one short string, so the grammar is small.
2. **Bounded arity.** `n_signs` before `signs[]`; `actions[]` capped at 3. No unbounded array
   can run past the token budget.
3. **Small budgets per call** (≤256 tokens) — truncation cannot land mid-object.
4. **Enums beat floats.** Confidence as `low|med|high` removes ~6 free generation sites/clip.
5. **One job per call.** A failure is attributable and costs one field. Under v0.1 a single
   malformed bbox destroyed the scene classification, the OCR and the strategic goal with it.
6. **Nothing is asked twice.** Anything Engine A measures or SAM3 localises is never requested
   from the VLM, so there is no disagreement to adjudicate and no invitation to hallucinate.

## 5. Pre-registered gate for v2 (both outcomes bound in advance)

Same 8 clips, same arm, same prompt hash discipline. Re-measure:

| quantity | v0.1 MEASURED | v2 CONFIRM threshold |
|---|---|---|
| clips with **all four** calls valid | 1 / 8 | **≥ 6 / 8** |
| hard `no parseable JSON` failures | 3 / 8 | **0 / 8** (grammar makes it structural) |
| B1 scene valid | — | ≥ 7 / 8 |
| B4 goal ∈ vocabulary | — | ≥ 7 / 8 |

⛔ **If v2 still lands below 6/8 with grammar-constrained decoding, the failure is NOT format
and the next lever is the arm, not the schema** — that is the pre-registered REFUTE branch,
and it is the only condition under which a larger model becomes the answer.

⚠️ Scope stamp: n = 8 clips is a smoke, not a measurement. The 50-clip PH1 run is what
produces quotable rates; these thresholds exist to decide whether PH1 is worth launching.

## 6. Open, and not guessed

- **SAM3's real API is not yet introspected.** Installed on pod4 with `--no-deps` (torch
  verified intact at 2.8.0+cu128); dependency closure walked to `pycocotools`. The wiring of
  B3 → SAM3 above is the *intended* seam, not a verified one. `sam2` was never installed, so
  engine C has been silently skipping — moving to SAM3 loses nothing.
- **`alpamayo_rows = 0` on every clip** — engine D contributed nothing to the measured run.
  Either the mini clips are absent from `records.parquet` or the join key is wrong.
  Undiagnosed; flagged rather than assumed.
- Which constrained-decoding library the pod supports is unverified (`outlines` /
  `xgrammar` / transformers-native). To be probed before implementation, not assumed.
