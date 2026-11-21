# Label extraction rebuilt on the FULL 26.2 h Alpamayo corpus — v7 frozen

**Date** 2026-08-24 (overnight) · **Owner** Data FlyWheel (P2) · **Status** shipped, staged
**Report** `report/report.html` — 12 random scenes with frames, labels and my verdict
**Artifact** https://claude.ai/code/artifact/fba22185-2983-4f7b-b661-0a248cb1349a

---

## Headline (all MEASURED)

| | |
|---|---|
| clips labelled | **4,719 / 4,719**, 0 refused = **26.2 h** |
| full-corpus runtime | **30 s** (150 clips/s) |
| goal/action contradictions | **21.7 % → 0.21 %** (both directions checked) |
| Alpamayo lateral agreement | **63.9 %** vs 41.6 % chance |
| Alpamayo longitudinal agreement | **63.5 %** on a calibrated scale |
| CoT tokens grounded by a box | 12.8 % — a ceiling set by the source, see below |

The PI's `/goal` — *"prepare the pipeline to run tomorrow early in the whole 26 h
corpus"* — is met and exercised: the run is not merely prepared, it has been
executed end-to-end on the full corpus.

## The corpus was never the constraint — the pull was

Egomotion covered 968 of 4,729 clips. The chunk zips are ~38 MB each and hold
~200 clips, of which the Alpamayo selection needs ~3.4 — so whole-chunk
downloading moved 53 GB to extract 4 GB, MEASURED at 2 clips/min = **~26 hours**,
past the deadline.

HF serves these files with `Accept-Ranges`, so `code/pull_egomotion_range.py`
range-reads each remote zip's central directory and fetches ONLY the members we
need. **4,800/4,800 clips in 20 minutes, 1.58 GB, 0 errors** — 97x less traffic.
The same trick pulls camera video for a review sample
(`code/pull_camera.py`): 2.05 GB chunks, ~10 MB per clip.

## What the augmentation actually contains (RETRACTION_LOG C142)

I had stated that `meta_action` was "not reachable locally" and the Alpamayo data
"limited". Both wrong, and the PI was right to push back. The dataset is
`Sayood/tanitad-alpamayo2-augmentation` — **23,644 rows, 4,729 clips, 26.3 h,
FIVE tasks**, of which the pipeline used parts of one:

| task | coverage | value | verdict |
|---|---|---|---|
| `meta_action` | 100 % | 3 axes + CoT | **best lateral corroborator**, 69.9 % vs 41.6 % chance |
| `auto_labeling` | 100 % | typed, time-stamped motion segments (97.1 % parse) + forcing object | **timing and vocabulary**, not axis agreement |
| `grounding_via_vqa` | 93.3 % | 2D boxes | accurate — verified pixel-level; **confirms only, never refutes** |
| `vqa` | 99.9 % | 21 categories | ⛔ SAMPLED — ~0.3 % coverage per question, spot-checks only |
| `trajectory` | 100 % | Alpamayo's own path + ADE/FDE | unused so far |

⭐ **The longitudinal axis is calibrated and magnitude-bearing**, which nothing in
the programme had noticed: Strong Decel **−5.88** · Gentle Decel −1.73 · Maintain
+0.03 · Gentle Accel +1.93 · Strong Accel **+6.00** m/s, **Cohen's d = 1.59**.
⚠️ `Stop` is a STATE (v0 = 1.11 m/s, 66.7 % already below 1 m/s), not a
deceleration — mis-mapping those 48 clips is most of why a naive agreement read
30.7 %.

⚠️ **The anchors differ: Alpamayo `t0` = 5.1 s, ours = 8.0 s.** Every
cross-source number published before today compared moments 2.9 s apart.
Scoring each source on its own window: **54.9 % → 62.1 %**, no data changed.

## Seven defects, all found by looking at frames or by a control

| defect | before | after |
|---|---|---|
| `"_L" in "FOLLOW_LANE"` — every straight clip scored a LEFT turn (C143) | 24.9 % | **63.9 %** |
| a −94° turn belonged to NEITHER layer (started <6 s, ran to 14.8 s) | dropped | both layers |
| EVADE with no lateral motion — 74 % of emissions | 932 | **238** |
| goal and action used two different turn detectors | 219 | **0** |
| stop-then-launch labelled `STOP_POINT` + `ACCELERATE` | 101 | 6 |
| ⭐ **negated terms read as present** — the 408x inversion (C145) | 533 | **18** |
| a **stopped bus** matched no obstacle class, so the clip got `FOLLOW_LANE` alone | missed | EVADE |

## Three of tonight's errors were mine, inside the fixes

1. **A `contradicted` grounding state** — *"the clip has boxes, none of kind X,
   so X is absent"*. Invalid: there is exactly **ONE** grounding question per
   clip, and **3,246 clips with boxes were never asked about pedestrians**.
   Grounding can CONFIRM, never REFUTE (C144). I had reasoned this correctly for
   `vqa` in the same file and failed to carry it one field over.
2. **Boxes read as pixels** when they are 0–1000 normalised — they FIT a
   1920×1080 frame, so nothing errored, and a `Pedestrian` landed on a blank
   wall. Rescaled, it lands exactly on a real pedestrian (C144).
3. **A one-sided coherence check.** It tested "TURN goal but no turn action" and
   never the converse, so a fix I made broke **70 clips** the other way and the
   check certified them. Both directions are tested now.

⭐ **And the data corrected my own eyes TWICE, both on night scenes.** I read
`683d37fb` as an empty road and called its EVADE a false positive — the box shows
a pedestrian walking in the headlights. I read `472944a4`'s growing light as an
oncoming headlight and reported a missing `REACT_ON_ONCOMING` — the box shows a car
ahead with RED TAIL LIGHTS, and the source says "merging vehicle from the left", so
`YIELD + MERGE + GAP_TARGET` was right all along. **A negative from a 400 px
thumbnail is not a negative**, and at night I cannot separate headlights from
tail-lights at that scale.

## Vocabulary FROZEN (PI instruction)

*"we will stabilize now the vocabulary and keep it constant"*. `test_vocab_v7_frozen.py`
pins the exact tuples AND their order — they are tensor dimensions, so appending
is a migration and inserting is a corruption. It caught a real bug immediately:
`REDUCE_TO_FOLLOW_ROUTE` was removed from the vocabulary on PI instruction while
the emitter still emitted it. A runtime `vocab_v7.assert_frozen()` now guards
tokens assembled by concatenation, which static analysis cannot see.

**Definitions now live in `vocab_v7.py` beside the tuples**, with a test asserting
every frozen token has one and no definition outlives its token — the PI lost them
between two documents, and a markdown file is what allowed that.

Sizes: strategic goals **8** · strategic actions **7** (REDUCE_TO removed) ·
tactical goals **22** · lat actions **8** · lon actions **8** · nav **3**.

## ⭐ The biggest defect was found by a CONTROL, not a test (C145)

Nothing was failing and the tokens looked plausible one clip at a time. Splitting
the corpus by Alpamayo's OWN negative — 853 clips stating nothing is critical —
and requiring the token rates to run in a known direction:

| token | "nothing critical" | names a component | ratio |
|---|---|---|---|
| `TRAFFIC_LIGHT_REACT` | **60.4 %** | 0.1 % | **408x** ⛔ |
| `EVADE_IN_CORRIDOR` | 7.0 % | 2.0 % | 3.6x ⛔ |
| `GAP_TARGET` | 0.7 % | 16.1 % | 0.04x ✓ |
| `YIELD` | 3.6 % | 17.0 % | 0.21x ✓ |

Cause: *"…with **no** lead vehicle pedestrians cyclists **traffic lights** or
obstacles…"* — six negated terms under one `no`. **515 of 533 emissions were this
artefact.** After `cot_negation.py`: 533 -> 18, ratio 408.3 -> 1.19, and the
COLOURED light tokens barely moved (RED −5, GREEN −4) — noise removed, signal kept.

⚠️ **I had probed negation earlier the same night, measured "3.5 % — small", and
moved on.** That probe searched `no` within 40 characters of a term and could not
express a six-term enumeration. **An instrument that cannot detect X in the form X
takes will report X is rare.**

## The proposed "top fix" was refuted by checking its scope

I planned to use the explicit `none` to suppress contradicting CoT tokens. All 853
are worded *"the first 2 seconds"* — from Alpamayo's 5.1 s anchor that is
**−2.9 to −0.9 s relative to ours**, which does not overlap our window. It cannot
suppress anything. It remains valuable as a CONTROL, which is how it earned its keep.

## PI review 2026-08-27 — three more defects, from one screenshot (C146)

| what the PI flagged | what was wrong | before | after |
|---|---|---|---|
| strategic extracted at 6 s | bands were **0-6 s / 6+**; they are `OPERATIVE (0,2) · TACTICAL (2,6) · STRATEGIC (8,30)`. Symptom: `TURN_RIGHT_FOLLOW_ROUTE by_time_s: 6.0` | **435** | **0** |
| tactical must be 2-6 s | 0-2 s belongs to the OPERATIVE layer; and this exposes a REAL GAP — **nothing owns 6-8 s**, so those manoeuvres are now reported as `unassigned` (54 clips, 1.1 %) rather than absorbed | 0-6 s | 2-6 s |
| long curves read as turns | radius came from **peak instantaneous curvature** (40 m where the arc says **144 m**), and there was NO speed condition | 987 | **615** |

**The turn gate, calibrated against an independent reference** (Alpamayo's own
`motion_analysis` labels — 349 turns vs 1,468 keep-lane, text-derived, owing
nothing to our geometry):

    |dyaw| >= 15 deg AND R_arc <= 140 m AND v_min <= 8.0 m/s
    -> precision 70.6 %, recall 61.3 %, F1 0.656   (vs 54.1 % on dyaw alone)

⚠️ **The PI's mechanism was right, the signal was not.** He predicted turns show
strong deceleration. In the 2-6 s band a turning ego is **ACCELERATING** (median
**+2.2 m/s**) — it is already exiting — so `dv` separates nothing. Turn clips are
already slow **4 s before the anchor** (5.4 vs 14.0 m/s); the braking happens
outside every window we observe. ⇒ the usable signature is **absolute speed**,
and it removes **70 of 159 false positives (44 %)**.

⚠️ **I re-committed the two-detector defect while fixing it:** gating the GOAL and
not the ACTION left **329 clips (7.0 %)** with `lat=TURN` and no turn goal.
Coherence 0.40 % -> 7.16 % -> **0.21 %**. Pinned by `test_bands_and_turns.py`,
which checks the contract in BOTH directions.

## PI review 2026-08-27 (2) — one MERGE question, three defects, then a fourth (C147)

The PI asked why `59b57590` emitted `MERGE` when its CoT never mentions one. It
doesn't. The only `merg` in the whole text is **"potential door-opening/merge
hazards"** — a hypothetical risk CLASS in a compound noun. ⚠️ Negation scoping
(C145) cannot catch this: "potential" is IRREALIS, not negation.

**But the question found bigger things beside it:**

| defect | scale | after |
|---|---|---|
| `MERGE` from a hazard noun phrase | 3 of 129 (2.3 %) | 129 -> **82** |
| ⭐ **`LANE_CHANGE_L/R` had NO extractor at all** | **174 clips** state a lane change; **0** tokens | **39 emitted** |
| `components_analysis` is a numbered list; only entry 1 was read | 197 clips carry 2-6 entries | 3,107 -> **3,733 entries** |
| ⭐⭐ **13 of 52 frozen tokens (25 %) were never emitted on the corpus** | strategic layer: 3/8 goals, 2/7 actions reachable | **43/52**, strategic 4/8 and 5/7 |

**Why only 39 of 176 lane changes survive — and why that is right.** Measuring
cross-track residual AFTER removing the road's own fitted arc, against a matched
control: +0.07 m (2-6 s), +0.41 m (Alpamayo's window), **+1.02 m** (-3..+8 s).
Real and growing, but the median claimed lane change **never displaces a full
lane width (~3.5 m)** in anything we observe. Loosening the gate would
manufacture labels. ⚠️ The FIRST version of that measurement used deviation from
the initial heading and put the CONTROL at 7.95 m — any curve produces that.
**A displacement measure without the road's arc removed measures the road.**

⭐ **The durable fix: `vocab_v7.NOT_YET_EXTRACTABLE` + `test_vocab_reachability`.**
`LANE_CHANGE_L` was frozen, defined in the matrix, covered by the freeze test —
and could never be produced. **Freezing a token is not the same as being able to
emit one.** Every frozen token must now either have an extraction path or be
listed with a REASON; 9 remain, each naming what it would take.

## Still wrong (ranked)

1. **Traffic lights under-read on rich scenes.** `d452ea24` shows red lights at
   −4 s that no token captures — its CoT only discusses the bus. A source-coverage
   limit, not an extraction bug: the honest ceiling of a CoT-driven approach.
2. **19 clips (0.40 %)** carry a goal/action contradiction — 10 `lat=TURN` with no
   turn goal, 9 `STOP_POINT` with a non-braking action.
3. **Grounding reaches ~13 % of CoT tokens**, capped by the one sampled question
   per clip. Raising it needs the grounding task re-run with more questions.
4. **Alpamayo's `trajectory` task is banked but unused.**

## ⚠️ NOT MINE — a bit-identity break in another stream's in-flight work

`stack/scripts/train_v6_staged.py` carries an **uncommitted** LIT-3 change
(`o1_stopgrad_factual`) whose comment claims *"DEFAULT False => incumbent loss
bit-identical"*. **That claim is false.** MEASURED: 5 guard tests fail with it;
stashing ONLY that file gives **136 passed, 0 failed**. I restored their file
byte-identically and did not touch it. Escalated as a task, not left in a doc.

## Deliverable manifest

| artifact | where |
|---|---|
| `alpamayo_records.py` · `alpamayo_fusion.py` · `alpamayo_structured.py` · `cot_negation.py` | repo `stack/tanitad/data/` (NEW) |
| `egomotion_source.py` (flat store) · `vocab_v7.py` (freeze + definitions) | repo `stack/tanitad/` (MODIFIED) |
| `s2_geom_emit_v7.py` (fusion, band split) · `s2_run_corpus.py` (NEW) | repo `stack/scripts/` |
| `test_vocab_v7_frozen.py` · `test_alpamayo_fusion_sides.py` · `test_cot_negation.py` · **`test_bands_and_turns.py`** · **`test_cot_lane_change.py`** | repo `stack/tests/` (NEW) |
| full corpus labels, 4,719 clips | `raw/s2_labels_v7.jsonl.gz` (1.29 MB) + `C:/Users/Admin/tanitad-wt/_s2build/v7_ship/` |
| pull tooling | `code/pull_egomotion_range.py`, `code/pull_camera.py` |
| review report | `report/report.html` + the artifact URL above |
| egomotion store, 4,800 clips | `C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/` (dev box) |
| Alpamayo records | `C:/Users/Admin/tanitad-data/alpamayo/records.parquet` (dev box) |
| retractions | `Project Steering/RETRACTION_LOG.md` C142-C147 |
| register | `Project Steering/GOALS_AND_CLAIMS.md` D-DATA-ALPA-FULL |

**Tests:** 151 green on the affected surface; full suite 4,669 passed with the 5
LIT-3 failures above, which are not on this surface.
