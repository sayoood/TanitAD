# SAM3 extraction v2 — contours, oriented extents, a scene channel, and the 115-clip re-run

**Package owner:** arch-inf agent, 2026-08-16 · branch `agent/arch-inf-20260803`
**PI instruction (verbatim):** *"rerun the whole 115 sam3 clips and before you do this use the
opportunity to optimize the extraction, include the contour segmentation of the agents in addition
to the bounding boxes, include additional classes like guardrails, road markings, road, ego lane,
road curbs…"*

---

## 0. TL;DR

| | v1 (as banked) | v2 |
|---|---|---|
| per-detection geometry | box + RLE mask | box + RLE mask + **polygon contour** + **oriented extent** |
| vocabulary | 7 agent concepts | 7 agent + **4 scene concepts**, in a **separate channel** |
| ego lane | — | **DERIVED** from markings + curbs, never prompted |
| detection floor | **0.5** (vendor default, nobody chose it) | **0.25** (PI decision) |
| wall-clock | 23.03 s/clip | **29.82 s/clip — 1.295×** |
| bytes/clip | 49 257 B | **120 005 B compact — 2.44×** (317 109 B at `indent=1` — 6.44×) |
| peak GPU | 4.231 GB | **4.241 GB** — +10 MB |
| record address | `sam3_backfill/` | `sam3_backfill_v2/` — the floors are never mixed |

⛔ **THE BIGGEST RESULT IS NOT ON THAT TABLE, AND IT IS A RETRACTION.** Building the contour
exposed that **`rle_rows` in the entire v1 corpus is FLATTENED and cannot redraw its own mask** —
2 496 detections, every one of them. `Sam3Processor` returns masks shaped **[N, 1, H, W]**, `_rows_rle`
iterated that as if it were [N, H, W], and every banked run came out `[0, flat_start, flat_end)`.
It passed every check anyone ran. §5, and `RETRACTION_LOG.md` C85.

⭐ **AND THE CLASS THE PI ASKED FOR THAT IS NOT PROMPTABLE IS THE INTERESTING ONE.** `ego lane`
returns **0** detections; `my lane` returns **6.0 per frame** with a median mask of 4 098 px² — a
large, plausible, unfalsifiable region, six of them, for a thing there is exactly one of. §2.

---

## 1. What changed in the engine

All of it is in `stack/scripts/ph0_sam3.py`, pinned by 25 new tests in
`stack/tests/test_ph0_sam3.py`. Suite: **3710 passed / 0 failed / 7 skipped / 2 xfailed**
(`cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q`; baseline 3689 + 21 net new at the time of
that run, 25 after the RLE fix).

### 1.1 Contours — and why they are traced on the pixel-CORNER lattice

`contour_of_mask(mask, tol_px=1.0, max_pts=48)` returns

```
contour_xy       FLAT [x0,y0,x1,y1,…] integer lattice points
contour_tol_px   the tolerance ACTUALLY used (not the one requested — see the cap)
contour_area_px  the polygon's own area
contour_n_loops  how many closed loops the mask had (>1 = fragmented or holed)
```

The boundary is traced on the **pixel-corner lattice** (crack following), not through boundary-pixel
centres. That is not a refinement, it is the difference between a usable number and a biased one:

> a polygon through pixel CENTRES under-counts area by roughly half the perimeter. A 3×3 blob has
> 9 mask pixels and a centre-polygon area of **4** — −55 %. The MEASURED median `car` box on this
> corpus is **188 px²** and `traffic light` **34 px²**
> (`…/2026-08-16-sam3-concept-reliability/` §2), so that bias would be the dominant term in every
> oriented extent derived from it, and it would look like a snug fit rather than a defect.

On the corner lattice the enclosed area equals the pixel count **exactly**, so the only area error is
the one the RDP tolerance buys — which is the number the brief asks for:

**MEASURED, 1 143 contours over the 5 pilot clips (`raw/p2_pilot_size.json`), tol = 1.0 px:**

| quantity | value |
|---|---|
| median signed area error | **−1.5 % to −2.9 %** per clip (−0.0149, −0.0219, −0.0292, 0.0000, −0.0196) |
| p90 signed area error | +2.9 % to +6.7 % |
| max abs error | 0.13 – 0.66 |
| contours with >1 loop | **118 / 1 143 = 10.3 %** |

⚠️ **The error is SIGNED and both signs mean different things.** RDP can only *lose* detail, so it
biases small and negative — that is the median. The positive tail is a different mechanism:
`contour_of_mask` keeps the **largest outer loop only**, so a detection with a hole comes out
*larger* than its mask, and a fragmented detection loses its other pieces. The large max values are
that, not simplification. ⇒ **The contour is a lossy summary and the RLE stays the primitive.** Both
are banked; `contour_area_px` vs `mask_area_px` audits every detection without a re-run.

The point cap is real and self-reporting: when 48 points is not enough, the tolerance doubles until
it is, and the record carries the tolerance that was actually used, not the one requested.

### 1.2 The oriented extent — the reason a contour was worth adding

`oriented_extent(contour_xy)` → `obb_cxcylwa = [cx, cy, long, short, angle_deg]`, the minimum-area
rectangle over the contour's convex hull.

⭐ **This is the agent-slot decoder's target tuple `(cx, cy, yaw, l, w)` in image space, and it is
exactly what a box cannot express.** `box_xyxy` has no angle — a car at 30° and the same car at 0°
share a box. MEASURED on a synthetic diagonal bar the extent recovers 45° with long/short > 4 while
the axis-aligned box of the identical mask is square (`test_oriented_extent_recovers_an_angle_a_box_cannot_express`).

⚠️ **IT IS AN IMAGE-SPACE ORIENTATION, NOT A YAW.** Turning it into a vehicle heading needs the
ground-plane homography and nothing here does that. `angle_deg` folds into [0, 180) because a
180° flip is the same extent — pinned by test, because a consumer reading it as a heading would
silently get a quantity defined only up to that flip.

### 1.3 The scene channel — structurally separate, on purpose

```python
AGENT_CONCEPTS = ["car","truck","bus","pedestrian","cyclist","traffic light","traffic sign"]
SCENE_CONCEPTS = ["lane marking", "road curb", "guardrail", "road marking"]
LIVENESS_CONCEPTS = ["road", "sky"]            # unchanged, still the C77 control
```

Scene detections land in `frames[*].scene` / `per_scene_hits` / `n_scene_det_total`. They do **not**
enter `frames[*].det`, `per_concept_hits` or `n_det_total`.

⛔ **Because those three are a CONTRACT, not a container.** `ph1_fuse.py` builds per-object tracks
out of `frames[*].det` and filters hazards on `t["concept"] in ("car","truck","bus","pedestrian",…)`;
`content_census` and three reports sum `per_concept_hits`. Pouring lane markings in would create
dozens of meaningless one-frame "tracks" per clip **and silently move numbers those documents
quote**, with nothing to reveal it. Pinned by
`test_scene_concepts_are_a_separate_channel_from_the_agent_contract` and
`test_a_v1_consumer_reads_a_v2_record_unchanged`.

**One v1 key does change meaning, deliberately:** `n_err_total` / `err_kinds` now span BOTH channels.
An error census that silently omitted a channel would be C77's own defect rebuilt. `n_err_agent` /
`n_err_scene` recover the v1 quantity.

**Stuff is not things, and the record says so.** `CONCEPT_KIND` marks every concept
`thing` / `stuff_instanced` / `stuff_extended` / `stuff_region`, because
`per_scene_hits["lane marking"] = 141` is **not** "141 lane markings" — it is *141 painted segments
separately grounded*. SAM3 returns a dashed line as one detection per dash and an extended structure
chopped by occlusion. A count of stuff is not an object count, and per-instance detections are
meaningful for `road marking` (an arrow, a stop line) and close to meaningless for `road`.

### 1.4 `road`, the control, and the one thing I did NOT do

The PI named `road` as a class. **It is not in `SCENE_CONCEPTS`**, and this is the one place I did
not take the instruction literally:

`road` is half of the **C77 liveness positive control**. Adding it to the measured vocabulary would
make *"the engine is live"* and *"the scene channel produced something"* the **same event** — a
reader could certify the scene channel with a number the scene channel itself produced. Spelling the
class `road surface` to dodge the string collision would be a fig leaf: MEASURED, `road` and
`road surface` return **2.6 vs 2.7 det/frame at hit rate 1.0/1.0 and median mask 6 923 vs 6 731 px²**
— the same object under a synonym, whose failures would correlate with the control's.

⇒ **The resolution, which costs nothing:** `liveness_probe` now banks its own detections with full
geometry (`keep_det=True`), so **road is extracted as a box + mask + contour**, in the control's own
block, outside every measured total. **The price, stated plainly: on the ONE control frame per clip,
not on all six run frames.** Per-frame road segmentation is available for the asking but requires
re-basing the control onto a different concept pair first — and `road` (83/83 clips) and `sky`
(81/83) are the only two concepts MEASURED reliable enough to be that control. **That is a PI
decision, escalated, not taken here.**

### 1.5 The detection floor is now verified, not assumed

`Sam3Processor(model, confidence_threshold=…)` was read from vendor source and had **never been
executed** — the brief flagged it UNVERIFIED. **MEASURED: the ctor kwarg exists and takes**
(`confidence_threshold_set_via: "ctor kwarg"`, every pilot and production record). `build_processor`
now reads the value back off the object and **refuses to detect** if it did not take, because a
corpus built at an unknown floor is unattributable *and the floor is invisible in the payload* —
it shows up only as rows that are not there.

### 1.6 Every consumer, named — touched, changed-by-default, or left alone

| consumer | status | detail |
|---|---|---|
| `stack/scripts/ph0_sam3.py` | **rewritten** | the engine |
| `stack/tests/test_ph0_sam3.py` | **+25 tests** | contours, extents, scene channel, ego lane, the `(1,H,W)` fix, the threshold verification |
| `colab/s2_lab_lib.py` | **extended** | `BACKFILL_V2_PREFIX`; `sam3_leg(scene_concepts=…, contours=…, meta=…)` — **default is v1 shape**, v2 is asked for; `content_census(require_schema=, require_conf=)`; `bank_json(indent=)` |
| `colab/RUNNER.md` | **updated** | v2 supersedes the notebook for the SAM3 leg; the measured v2 budget row |
| `stack/scripts/ph1_fuse.py` | **untouched, VERIFIED** | `build_tracks` on a real v2 record: **141 tracks, 0 scene concepts leaked** (`raw/v2_integration_check.json`) |
| `stack/scripts/ph0_rich_overlay.py` | **untouched — but see C85** | needs no code change; its output was wrong because the *producer* was. It does not draw contours or the scene channel — a v2 renderer lives in this package instead of editing a file I do not own |
| ⚠️ `stack/scripts/aug120_pipeline.py` | **untouched, BEHAVIOUR CHANGES** | it shells out to `ph0_sam3.py` **without `--concepts`/`--scene-concepts`**, so it now inherits the v2 defaults and will emit schema-v2 records on the next batch — including the 4 472-clip build. Its `SAM3_CENSUS` gate reads `n_det_total` / `n_err_total`, both still present, so it does not break. **`--scene-concepts ''` restores the v1 shape exactly.** Named here because a default change is still a change |
| `content_census` / `done_set` | **extended, back-compatible** | v1 records still census correctly; `require_*` only bind when passed |

---

## 2. ⭐ WHICH OF THE PI'S CLASSES ARE PROMPTABLE — MEASURED, not assumed

**Protocol** (`code/p1_prompt_probe.py`, `raw/p1_prompt_probe.json`): 21 candidate prompts × 2 frames
× 5 gap clips = **210 scorings, n = 10 frames**, one encode per frame, `confidence_threshold=0.25`,
frames 179×448, 87.5 s total. Synonyms shipped together because SAM3's text side is CLIP and there is
no way to reason out which surface form it has seen.

| PI class | prompt | det/frame | frames hit | score med | mask med px² | verdict |
|---|---|---|---|---|---|---|
| road markings | **`lane marking`** | 5.2 | 0.5 | 0.416 | 28 | ✅ **adopted** |
| | **`road marking`** | 10.8 | **0.7** | 0.396 | 49 | ✅ **adopted** |
| | `lane line` | 5.1 | 0.5 | 0.337 | 41 | redundant |
| | `white lane line` | 8.5 | 0.5 | 0.424 | 41 | redundant |
| | `painted road marking` | 2.2 | 0.2 | 0.591 | 22 | too rare |
| road curbs | **`road curb`** | 5.9 | **0.8** | 0.317 | 468 | ✅ **adopted** |
| | `curb` | 5.2 | 0.8 | 0.351 | 372 | equivalent |
| | `kerb` | **0.0** | 0.0 | — | — | ⛔ **does not ground** |
| | `sidewalk` | 5.7 | 0.7 | 0.351 | 536 (max 14 390) | different object |
| guardrails | **`guardrail`** | 0.7 | 0.4 | 0.328 | 958 | ✅ **adopted, thin** |
| | `guard rail` | **0.0** | 0.0 | — | — | ⛔ **does not ground** |
| | `crash barrier` | **0.0** | 0.0 | — | — | ⛔ **does not ground** |
| | `metal barrier` | 0.4 | 0.2 | 0.640 | 820 | weaker |
| road | `road` | 2.6 | **1.0** | 0.515 | 6 923 | control, §1.4 |
| | `road surface` | 2.7 | 1.0 | 0.497 | 6 731 | same object |
| | `drivable area` | **0.0** | 0.0 | — | — | ⛔ **does not ground** |
| ego lane | `ego lane` | **0.0** | 0.0 | — | — | ⛔ **not promptable** |
| | `the lane the car is driving in` | **0.0** | 0.0 | — | — | ⛔ **not promptable** |
| | `current lane` | **0.0** | 0.0 | — | — | ⛔ **not promptable** |
| | ⚠️ `my lane` | **6.0** | 0.9 | 0.336 | **4 098** (max 15 270) | ⛔ **rejected — see below** |

⛔ **THE `my lane` RESULT IS THE DANGEROUS OUTCOME, NOT THE GOOD ONE.** Three of the four ego-lane
phrasings return nothing, which is honest. The fourth returns **six detections per frame, each a
~4 000 px² region, on 9 of 10 frames** — for a thing there is **exactly one of**. A count of six is
prima-facie evidence that the model is grounding *"a lane-ish region"*, not *the ego's lane*, and
nothing downstream could falsify it: it looks exactly like a correct answer. Shipping it would have
been a derived quantity smuggled in as a perception.

⇒ **Four of the five requested classes are directly promptable** (`road marking`, `lane marking`,
`road curb`, `guardrail`, plus `road` which is the control). **`ego lane` is DERIVED** —
`ph0_sam3.derive_ego_lane`, §3. **Six of the twenty-one candidate prompts return zero and cannot be
guessed from the word** (`kerb`, `guard rail`, `crash barrier`, `drivable area`, and three of four
ego-lane phrasings) — which is the reason this probe existed instead of a vocabulary chosen at a
desk.

⚠️ **Scores sit low and that is the 0.25 floor earning its keep.** Median score for the adopted scene
prompts is **0.317–0.416**. At the old vendor default of **0.5 the entire scene channel would have
been discarded at inference** and un-recoverable without re-detecting every clip.

---

## 3. Ego lane — derived, and what it unblocks

`derive_ego_lane(scene_dets, frame_wh)` → `ego_lane.frames[<frame_idx>]`, class
**`DERIVED-ESTIMATED`**, `derived_from: ["lane marking", "road curb"]`.

Method: keep boundary detections whose **footpoint** (centre of the mask's lowest row — *not* the box
centre, which for a long diagonal marking floats metres from the paint) lies in the bottom 25 % of
the frame; cluster them in x, because SAM3 returns a dashed line as one detection per dash and three
dashes of one line are **one boundary**; the ego sits at the image centre column;
`lane_idx_est = (boundaries left of ego) − 1`, `lane_width_px = right.x − left.x`.

⭐ **WHAT IT SUPPLIES — and ⛔ A CORRECTION TO MY OWN BRIEF, because the obvious summary is wrong.**

The PI ruled that `PREPARE_LANE_CHANGE` must be derived from CONTEXT
(`…/2026-08-16-s2-v1-labels/review/PI_REVIEW_FINDINGS.md` §BINDING). The task brief that reached me
described the blockers as *"`route_lane_idx` (which lane the ego is in) and `lane_continues`"*. **The
primary source says otherwise**, and I only caught it by going to it —
`…/review/LANE_CHANGE_DEEP_REVIEW.md` §3 lists **four** inputs with individually different status:

| `s2_derive.LANE_CONTEXT_INPUTS` | meaning (verbatim) | status before this work | after |
|---|---|---|---|
| `n_lanes_same_direction` | lanes on the ego's carriageway | ⚠️ VLM `lanes_visible` — unreliable | ✅ **vision-derived** `n_lanes_est` (upper bound) |
| `ego_lane_idx` | *"ego's 0-based lane index from the right"* | ⚠️ VLM `lane_ego` — same instrument | ✅ **vision-derived** `lane_idx_est` |
| `route_lane_idx` | *"which lane serves the route / main road"* | ⛔ DOES NOT EXIST — needs a map/lane graph | ⛔ **still does not exist** |
| `lane_continues` | does the ego lane continue (not exit-only)? | ⛔ DOES NOT EXIST | ⛔ **still does not exist** |

⇒ ⛔ **`PREPARE_LANE_CHANGE` IS NOT UNBLOCKED BY THIS WORK.** `route_lane_idx` is *not* "which lane
the ego is in" — it is which lane serves the **route**, and together with `lane_continues` it needs
lane **topology**, which no camera frame contains. What this delivers is the review's own **option
2** — *"a vision lane-boundary estimator (lane-marking segmentation → lane index + count from the
front-wide camera) … the vision-only-at-inference path"* — which replaces the two ⚠️ VLM-sourced
inputs with vision-only ones and satisfies the vision-only-at-inference rule. Two of four remain
missing, and the honest emission stays the route's own token.

⚠️ **`lane_idx_est` counts 0-based FROM THE RIGHT**, because that is `ego_lane_idx`'s own definition.
My first implementation counted from the left; inventing a second convention for the same quantity
is how two correct numbers become one wrong one. `ego_lane.index_convention`, `ego_lane.supplies` and
`ego_lane.does_not_supply` are written into every record so the limit travels with the data, and the
correction is pinned by `test_ego_lane_says_which_of_the_four_lane_inputs_it_does_NOT_supply`.

⚠️ **`n_lanes_est` is an UPPER BOUND, not `n_lanes_same_direction`** — on an undivided road the
boundaries of the oncoming carriageway are in the count too.

⚠️ **AND IT IS THIN. MEASURED: the ego lane is bounded on BOTH sides on 9 of 29 pilot frames (31 %).**
The rest return `lane_idx_est: null` with an explicit `reason` — *"ego lane not bounded on both
sides (left 1, right 0)"* or *"no boundary detection in the near field"* — rather than a number that
looks like a measurement. That is the honest state of it: `lane marking` fires on ~50 % of frames,
and both sides at once less often still. **It is an input to a label, never a label**, and four ways
it is wrong (camera-centreline assumption — MEASURED false across PhysicalAI's two rigs; counts from
the leftmost VISIBLE boundary; curb-plus-line double counting; single-frame, unsmoothed) are written
into the function's own docstring rather than into a footnote.

---

## 4. ⛔ SIZE IT BEFORE YOU RUN IT — three arms, one session, five clips

`code/p2_pilot_size.py`, `raw/p2_pilot_size.json`. The v2 run changes **two** things at once — the
schema and the detection floor — so a single before/after would report their sum as the schema's
cost. That is the `--v2` conflation defect. Three arms, same five clips, same decoded bytes, one
kernel, threshold moved by assignment (a second `build_sam3_image_model` would add 3.58 GB and
pollute the process-global peak counter):

| arm | schema | conf | wall s/clip | det | bytes/clip `indent=1` | bytes/clip compact | peak GPU |
|---|---|---|---|---|---|---|---|
| **A** | v1 | 0.50 | **23.03** | 174 | **49 257** | 15 734 | 4.231 GB |
| **B** | v1 | 0.25 | 23.97 | **458** | 97 615 | 31 638 | 4.235 GB |
| **C** | v2 | 0.25 | **29.82** | 458 + **685 scene** | **317 109** | **120 005** | 4.241 GB |

| growth | value | reading |
|---|---|---|
| wall B/A | **1.041×** | the floor is nearly free — 2.63× the detections for 4 % of the time |
| wall C/B | **1.244×** | 4 more concepts + 1 143 contours |
| **wall C/A** | **1.295×** | **⇒ 115 clips ≈ 57.2 GPU-min** |
| bytes B/A | 1.982× | 2.63× the detections |
| bytes C/B | 3.249× | scene channel + contours |
| **bytes C/A** | **6.438×** | ⛔ **the brief's ~3× tripwire FIRES** |
| **bytes C/A, compact JSON** | **2.436×** | the lossless fix, below |
| peak GPU | **+10 MB** | not a constraint |

⛔ **THE TRIPWIRE FIRED AND HERE IS THE ANSWER — no detections are trimmed.** MEASURED: `bank_json`
wrote `indent=1`, and with `indent=1` every element of every nested list gets its own line. A v2
record is *mostly* nested lists (RLE runs, contour points), so **64 % of the file is whitespace**.
Compact separators give **the identical information at 120 005 B/clip = 2.44×** the v1 records as
banked — under the tripwire. Absolute corpus size: **115 × 120 KB ≈ 13.8 MB**.

⇒ **The right lever was the encoding, not the data.** Trimming — a `scene_min_score`, a top-K per
concept — was implemented as an available option and **deliberately not used**: the whole argument
for lowering the floor to 0.25 is that discarding at write time is irreversible while filtering at
read time is free. Spending that argument on 100 KB would have been incoherent. `bank_json` gained
an `indent` parameter (default unchanged at 1); the v2 run passes `indent=None`.

---

## 5. ⛔ RETRACTION — v1's `rle_rows` is FLATTENED and cannot redraw its own mask

**Found by building the contour. `crack_loops` asserts `ndim == 2` and returned nothing; the strict
reader found the permissive one's bug.**

MEASURED on a live T4: `Sam3Processor.set_text_prompt` returns `masks` of shape **`[N, 1, H, W]`**.
Every consumer indexed `masks[i]` and treated the result as 2-D. Exactly one of them was wrong in a
way that mattered:

* `mask_area_px` — `.sum()` is shape-agnostic ⇒ **correct** in v1;
* `_rows_rle` — `enumerate()` over a `(1, H, W)` array yields **one** item whose `row` is the whole
  `(H, W)` plane, so `np.flatnonzero` returns **flattened** indices.

MEASURED, clip `0089a096`, a `car` at box `[54.4, 82.1, 66.0, 94.1]` on a **448**-wide frame:

```
banked v1:  [[0, 36794, 36800], [0, 37240, 37250], [0, 37688, 37698], … ]   <- 12 runs, all row 0
correct  :  [[82, 58, 64],      [83, 56, 66],      [84, 56, 66],      … ]
```

**Why nothing caught it:** the run lengths still sum to `mask_area_px` (120), the JSON is
well-formed, the field is present, the count is right. It is C77's shape again — *a well-formed
artifact whose content is wrong* — and C18's — *a check scoped to the container*.

**Blast radius:**

| consumer | effect |
|---|---|
| `ph0_rich_overlay.draw_masks` / `ph0_sam3.draw_masks` | ⭐ **MEASURED, not inferred** (`raw/c85_overlay_proof.json`): banked v1 record `0089a096`, 60 detections, **962 runs — all on row 0, max column 63 539, only 2 of 962 inside a 448-wide frame**. Rendering them through the real `draw_camera` with and without `rle_rows` differs by **0 pixels**. ⇒ **the banked `*_rich.mp4` overlays show boxes and NO mask fill.** |
| all 2 496 v1 detections | `rle_rows` unusable **as documented** |
| `mask_area_px`, `box_xyxy`, `score`, `per_concept_hits`, `n_det_total`, the liveness control | **unaffected** — every number in `SAM3_CONCEPT_RELIABILITY.md` and `SAM3_DTYPE_FIX.md` stands |

⚠️ **THE DATA IS RECOVERABLE — the schema was the lie, not the pixels.** A v1 run decodes as
`row = start // W`, `col = start % W` (a run never straddles a row boundary, because a mask row's
run ends at the row's end). **No re-detection is required to rescue v1's masks.**

**Fix:** `ph0_sam3.as_2d_mask()` squeezes leading singleton axes and **raises** on a shape it cannot
interpret; `_rows_rle` and `read_outputs` go through it; every detection now also banks `mask_hw`,
which makes the encoding self-checking (36 794 is a legal column index in *some* frame size — it is
not one in a 448-wide frame). Pinned by
`test_the_real_mask_shape_is_N_1_H_W_and_the_rle_must_survive_it` and
`test_a_mask_that_is_not_two_dimensional_raises_instead_of_guessing`.

**Root-cause class:** *a serialiser that accepted a shape it never asserted, next to a reducer that
was shape-agnostic and therefore agreed with it.* The generalisation: **when two derivations of the
same object disagree in strictness, believe the strict one** — and prefer a function that raises to
one that copes.

---

## 6. The 115-clip re-run

⛔ **ALL 115, NOT THE 32 RESIDUAL.** 83 were banked at `confidence_threshold=0.5`; mixing two floors
in one corpus makes every downstream number unattributable, and **nothing in the data reveals it** —
a floor is visible only as rows that are not there.

⛔ **AND IT BANKS TO ITS OWN PREFIX**, `sam3_backfill_v2/`. Overwriting v1 in place would make the
corpus mixed for the whole length of the run, and free-Colab reclaimed the T4 three times during the
last pass, so "the whole length of the run" is not a hypothetical window. v1 stays intact and
quotable — it is the primary source of the concept-reliability study.

**Completion criterion, by CONTENT (C77), and stricter than v1's:**

```
liveness control present  AND  zero error entries
AND schema_version >= 2   AND  engine.confidence_threshold == 0.25
```

The last two are the new half: a v1 record is present, non-empty, error-free **and live** while still
being the wrong record. `engine.confidence_threshold` is stamped into every record by `sam3_leg`
precisely so this is checkable.

### 6.1 Result

<!-- CENSUS -->

---

## 7. Escalations — these are decisions, not notes

1. ⛔ **`RETRACTION_LOG.md` C85 (§5) needs to reach the overlay owner.** `ph0_rich_overlay.py` is not
   mine and needs **no code change** — the bug was in the producer — but **every overlay video
   rendered from a v1 record has no mask fill**, and anyone comparing a v1 overlay against a v2 one
   will read that as a regression in the opposite direction.
2. ⛔ **Per-frame `road` segmentation requires re-basing the liveness control** (§1.4). PI decision.
3. ⚠️ **`ph1_fuse` does not read the scene channel at all.** v2 records fuse exactly as v1 records do
   — which is the additive-schema promise working — but `route_lane_idx` / `lane_continues` will not
   appear in a fused record until the fuser is taught to read `ego_lane`. **Owner: the fusion
   package.** Naming it here rather than in a README, because an integration request that lived in a
   README went unread for 10 days.
4. ⚠️ **The v1 `sam3_backfill/` corpus still holds 32 C77 records.** v2 supersedes it wholesale; the
   v1 prefix should be marked superseded rather than repaired.

---

## 8. Reproduce

```
# dev box, one session, in order
python code/ship.py tanitad-sam3v2                     # closure + token + kernel bring-up
colab exec -s tanitad-sam3v2 -f code/p1_prompt_probe.py --timeout 2300
colab exec -s tanitad-sam3v2 -f code/p2_pilot_size.py   --timeout 2900
colab exec -s tanitad-sam3v2 -f code/p3_run115_v2.py    --timeout 9000
python code/hf_v2_census.py --out raw/v2_census.json    # INDEPENDENT far-side check
```

⚠️ **`PYTHONUTF8=1` is required for every `colab` invocation** — colab-cli 0.6.0 opens the script
with the locale codec (cp1252 here) and any file carrying a ⛔/⚠️ dies with
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f` before a line reaches the VM. MEASURED.

⚠️ **A re-ship is a NO-OP without a module purge.** The kernel persists across `colab exec`, so
extracting a newer file over an already-imported module changes nothing: `import ph0_sam3` returns
the cached one and the verify-gate passes on the OLD code. `bootstrap_v2.py` purges `sys.modules`
before re-importing, and its verify-gate then greps the specific fixes (`SCHEMA_VERSION >= 2`, the
scene/control disjointness, the corner-lattice area invariant, the `(1, H, W)` RLE fix) out of the
loaded module. This is the pod-checkout-drift trap with a Colab accent.
