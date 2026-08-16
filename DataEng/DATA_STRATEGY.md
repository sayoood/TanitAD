# TanitAD Data Strategy (v3.0, 2026-08-17 — supersedes v2.0 of 2026-08-15)

> **Role of this file.** `Project Steering/HANDOVER_TO_LOCAL_2026-08-15.md` §1.A cites it as *"the
> strategy"*, so it is a **cited owner** and may not go stale. It is deliberately an **INDEX with
> the decisions on it**: numbers live in the owner documents named below and in
> `Project Steering/MODEL_REGISTRY.md`, never here. Every number it *does* carry is stamped with
> its **evidence class, its n, and its CORPUS** — because three of the live disagreements in this
> programme are corpus-scope errors, not measurement errors (§4.3).
>
> **What changed in v3.0.** v2.0 was written on 2026-08-15 and was overtaken within 48 hours by
> four things that change the *strategy*, not merely the numbers: the perception leg went from
> zero-yield to working, the detection floor became a **declared decision**, the biggest data job
> was **reclassified from labelling to extraction** by a PI correction, and the multi-leg
> corroboration architecture was **measured and found not to do what it was designed to do** (§5.3).
>
> ⚠️ **Searched before rewriting** (absence at one location is not absence): no newer owner
> document exists. `…/Data Engineering/TANITDATASET_V1_STRATEGY.md` is still the **plan of
> 2026-07-25**, predating A2 and PH0 execution; `DATA_LAKE_ARCHITECTURE.md`, `OWN_DATASET_PLAN.md`
> and `Research/DATASET_LANDSCAPE.md` are narrower and older. ⇒ this file remains the owner.

## 0. The one thing to read first — unchanged

**We are data-limited by roughly two orders of magnitude, and compute-limited by none.**
(`…/2026-08-07-hierarchical-wm-redesign/V6_DATA_REQUIREMENT.md`, MEASURED from the running job.)

| quantity | value |
|---|---|
| training corpus | **2,400 episodes ≈ 13.3 h** |
| model | **336.5 M** params |
| task-matched reference **Orbis 2** (hierarchical driving WM, 1,067 M, fine-tuned on **our** corpus) | **5,890 h**, ratio **1 : 443** |
| **hours per M-param, Orbis 2 vs v6** | **5.52 vs 0.040** ⇒ **139× under, size-normalised** |

⚠️ **Evidence stamp, still unresolved:** the Orbis-2 rows are **PUBLISHED-via-snippet**
(`ORBIS2_ANALYSIS.md`, *"one confidence step below PUBLISHED-exact"*) — arXiv was egress-blocked.
**Confirm against the PDF before any of them decides a GPU-day.** Open since 2026-08-15 with **no
named owner** (§11).

⇒ **Only two levers can close a 443× gap:** **P0** a larger corpus under a **new, declared** parity
key (*never a silent widening*; eval set unchanged) or **P1** a frozen pretrained encoder that
imports someone else's hours. Everything else — train longer, saliency sampling, near-duplicate
pruning, **and every layer in §3–§5** — is worth **single-digit factors**. §7 makes this precise;
it is the sentence most often misread in this document.

## 1. Corpus roles (who provides what)

| Corpus | Role | Actions | License position |
|---|---|---|---|
| **PhysicalAI-AV (PRIMARY RICH, D-012)** | urban diversity backbone; 1,727 h, 25 countries, 2,500+ cities | egomotion (poses → yaw-rate/accel) | **use now, resolve later**; every experiment tagged `data:physicalai` |
| comma2k19 (BOOTSTRAP + PUBLIC ANCHOR) | real-CAN action grounding; **all public open-loop numbers** until licenses resolve | real CAN (steer, speed) | MIT — clean |
| NVIDIA synthetic (SIM-DATA ARM, D-014) | pre-rendered long-tail: emergency, lane change, weather | scenario egomotion | Cosmos-Drive-Dreams CC-BY-4.0 |
| **AlpaSim / NuRec** | the closed-loop arm that actually works — runs bare on an A40, **not** docker compose; `volume.nurec` is gzip+msgpack, `map.xodr` answers part of the strategic-map gap | sim-exact | per NuRec scene terms |
| nuScenes-mini | D8 OOD probes only (never trained on) | ego pose | research |
| Own GoPro / OpenDV / YouTube | Phase 1 H7 pseudo-labeling scale-up | via IDM (H7) | own / public |
| ~~CARLA on RunPod~~ · ~~MetaDrive~~ | superseded by AlpaSim / retired per D-014 | — | — |

⛔ **Two corpus facts that are settled and must stop being re-asked** (five independent probes):
PhysicalAI-AV ships **no map, lane graph, junction annotation, roundabout label, traffic-light
feature or route/goal signal** — the card says verbatim *"we do not include open maps data"* — and
`egomotion` carries **no lat/lon/GNSS**, so **OSM map-matching on our traces is impossible**. The
strategic-brain topology must come from AlpaSim, an external corpus, or an auto-labeller.

⚠️ **The rig caveat stands:** PhysicalAI front-wide has **two camera rigs** (cy ≈ 543 rig A,
cy ≈ 755 rig B). Crop around the **per-clip** principal point; a geometric-centre crop is ~215 px
wrong for rig B.

### 1.1 ⭐ NEW — the train-corpus obstacle join (2026-08-17)

**MEASURED (ours)**, artifact `…/Data Engineering/Implementation/incoming/2026-08-17-train-obstacle-join/raw/train2400_agents.meta.json`:

| | |
|---|---|
| corpus | the **parity train corpus** `physicalai-train-e438721ae894`; **2,400 episodes requested** |
| joined | **2,308 episodes · 433,040 frames · 12,122,129 agent boxes** · `visible_frac` 0.4106 |
| the 92 not joined | fully accounted **by name**: `no_obstacle` 79 · `registration_failed` 10 · `bad_clip` 3 (2,308 + 92 = 2,400 exactly) |
| verification | md5 `24cbdca8…`, plus a **read-back through the real consumer** (`train_p8_occupancy.JoinFileReader` → 433,040 records / 2,308 clips / occlusion flags present) — not a file-size check |

This is what finally makes `obstacle.offline` a **train-side** signal rather than a pod-side
curiosity, and it is the sixth PhysicalAI feature the programme reads.
⛔ **Two gaps, both work items, neither a number problem.** (a) The HF location
(`Sayood/tanitad-ph0-aug120` → `joins/`) is asserted in **exactly one prose line** and appears in
**zero code paths** — evidence class **INHERITED-from-prose**, and the round-trip md5 check is
*narrated, not banked*. (b) There is **no `MODEL_REGISTRY.md` row** for the join. Until both land,
cite the meta sidecar, not the HF path.
⚠️ Carry forward: the F-18 slot probe already returned **NEGATIVE (D1) at step 9000** on a 61-clip
non-parity subset. This join does not change that result.

## 2. ⭐ The strategic frame that changed — where the bound actually sits

v2.0 treated augmentation as one programme with one bottleneck. It is **three layers with three
different economics**, and as of 2026-08-17 **none of them is bounded by labelling any more**:

| layer | status | what bounds it now | class of cost |
|---|---|---|---|
| **1 — Alpamayo-2 Super** (§3) | ⭐ **labels COMPLETE for all 4,729 clips** | **our own 120° video extraction** | **CPU / IO, schedulable** |
| **2 — PH0 perception** (§4) | ⭐ **repaired**; 201/201 aug120 coverage | concept reliability + a **declared floor** | GPU, small |
| **3 — PH1 fusion + label legs** (§5) | ⛔ **architecturally defective** | **leg independence**, not leg count | **code fixes, ~0 GPU** |

⇒ **The strategy this implies.** Stop spending decisions on *"can we get more labels?"* — we have
them. Spend them on (a) **scheduling extraction**, (b) **fixing three cheap code defects that
currently corrupt the labels we already own** (§5.3, §5.4), and (c) the **only two levers that
touch the 443× deficit** (§7). Layers 1–3 are label-side; per `V6_DATA_REQUIREMENT.md` §3.4 they
help S-T/S-S and are **no help at all for S-W**.

## 3. LAYER 1 — Alpamayo-2 Super: the labels are done

**Owner of the numbers: `Project Steering/MODEL_REGISTRY.md` §11.1.**

**MEASURED** from `records.parquet` (sha256 `ecae276d…`, verified against the HF far side):
**23,644 rows / 4,729 unique clips / 5 tasks / 12 columns**, `seed` 42, `model_id
nvidia/Alpamayo2-Super` on 100 % of rows, **zero errors**, **78.4 wall-hours** at **59.6 s/clip** —
**~3.4× cheaper** than the DESIGN doc's ~200 s/clip ESTIMATE. Published as
`Sayood/tanitad-alpamayo2-augmentation`; **no raw sensor data** (rows join by `clip_id` + `t0_us`).
Delivery **4,729 / 4,800 selected = 98.52 %**.

⛔ **One quantisation arm only** — `NF4-backbone-4bit-UNVALIDATED` on **100 %** of rows. There is
no bf16 arm, therefore **no quantized-vs-full comparison inside the dataset**.
⚠️ **The dataset card understates the completeness hole by 356×** (claims 4,800 / 23,999 and *"one
task row missing"*; measured, **356 of 24,000 rows missing**). The four stratified road classes
(urban 1,884 · intersection_rich 1,241 · highway 384 · unstructured 83) are **100 % complete** —
every dropout falls in the unlabelled remainder. Accounting in registry §11.1.

### 3.1 ⭐ THE CORRECTION THAT MOVES THE STRATEGY — extraction, not labelling

**PI correction, mid-task, 2026-08-16** (`…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md:26`):
> *"Alpamayo's labels are complete for all 4,729… The bound is on OUR video, not the labels."*

⇒ **MEASURED**: the labels need nothing regenerated — full camera rig, 4,729 clips, 0 unparsable
rows. The missing artifact is **our** `<clip_id>.v2ep.pt`, 256×640 cylindrical
`camera_front_wide_120fov`. *"The path to scale is extracting more video — schedulable CPU compute
— not re-labelling."*

⚠️ **This reclassifies the biggest data job (§6) from a decision into a schedule.** v2.0 framed
the 4,472 as a gap in the augmentation; it is a gap in our own cache.

⛔ **Two n's that must never be merged** (banked as `_two_n_rule` in
`raw/a3_three_leg_agreement.json`): the **label-side n is 4,729** (complete); the
**agreement-side n is 201** (bounded by our w120 cache). Any sentence mixing them is wrong.

⛔ **Three things NOT to do**, verbatim from `ALPAMAYO2_SUPER_ANALYSIS.md`: do not reposition
TanitAD as *"beating Alpamayo"* (34 B vs 0.3 B, 6 cameras vs 1, 115,000 h vs ~13 h); do not
fine-tune Alpamayo as our deliverable; do not quote its numbers as a target on our corpus **until
the contamination question is settled**.

## 4. LAYER 2 — PH0 perception: repaired, and the floor is now a decision

### 4.1 The zero-yield defect and its fix

⛔ **SAM3 was returning ZERO detections corpus-wide.** **MEASURED**, corpus = the **115** aug120
clips with no SAM3 record: `n_det_total = 0` over the **101** clips capturable before the repair
overwrote records in place, plus an independent **25-clip seed-0 sample, 25/25 zero**
(`…/2026-08-16-sam3-dtype-fix/raw/census_before_101clips.json`; C77 in `RETRACTION_LOG.md`).
⚠️ *"Corpus-wide zero" rests on 101 + a 25-clip sample, not a full-115 pre-census — the run
overwrites in place, so a complete before-census is unrecoverable.*

**Root cause (MEASURED, from source):** `sam3/model/vitdet.py:71` routes its MLP through
`perflib/fused.py::addmm_act`, which force-casts to **bf16**, while `:74` is a plain **fp32**
`nn.Linear` ⇒ `mat1 and mat2 must have the same dtype`. Reachable **only** from `Sam3Processor` —
the image path we use; every *video* entry point hides it under a process-wide bf16 autocast.
**Fix:** `stack/scripts/ph0_sam3.py:242 install_dtype_agreement()`, invoked at `:366` *before any
forward*.

**After (current, v2 pass):** **115 / 115** clips, **9,505 agent** + **23,116 scene** detections at
floor **0.25**, `PASS: true` (`…/2026-08-16-sam3-extraction-v2/raw/v2_census.json`).

⛔ **NUMBER TRAP — `2,496` is two different quantities and they must never collide.** It is the v1
pass's **total detections over 83 clips at floor 0.5**, *and* the v2 pass's **`traffic sign`
per-concept count over 115 clips at floor 0.25**. Always state prefix + floor + n.
⚠️ The v1 pass is superseded, not merged: all **115** were re-run into a **separate prefix**
(`sam3_backfill_v2/`) because *"mixing two floors in one corpus makes every downstream number
unattributable."*
⚠️ **C85:** the entire v1 `rle_rows` corpus (all 2,496 detections) is **flattened and cannot redraw
its own mask**. ⚠️ Two entries in `RETRACTION_LOG.md` carry the id **C85** — disambiguate if citing.

### 4.2 ⭐ The detection floor is a DECISION, not a default

`CONF_THRESHOLD_DEFAULT = 0.25` (`stack/scripts/ph0_sam3.py:423`), replacing the vendor default
0.5. Recorded in source at `:417`:
> *"0.25 is a DECISION (PI, 2026-08-16), not a default anyone inherited — which is what 0.5 was."*

**Why it matters strategically, and why it is asymmetric:** the floor is **destructive at write
time** (`:402`). Detections below it are never banked, so **lowering the floor later costs a full
re-detect (~26 GPU-h)** while raising it is free filtering. **MEASURED** justification: the minimum
banked score across all 2,496 v1 detections was **exactly 0.5000** (n = 2,496, corpus = the 83
repaired clips) — i.e. the old floor was silently defining the corpus. Enforced at read time:
`p3_run_manifest.json` requires `engine.confidence_threshold == 0.25`; `v2_census.json` reports
`conf.wrong 0`.

⛔ **The decision is NOT recorded in `Project Steering/`.** Probed `MODEL_REGISTRY.md`,
`DECISIONS.md`, `RETRACTION_LOG.md`, `BACKLOG.md` — **no row**. It exists only as a source comment
and a commit body. **A corpus-defining, destructive-at-write-time choice with no steering record is
exactly how a threshold becomes folklore.** Work item, §11.

### 4.3 ⛔ STANDING RULE — the sign numbers are on three different populations

Three measurements are live, they do **not** measure the same thing, and the reconciliation is
**explicitly PARTIAL**. Never quote one without its corpus:

| number | what it measures | **corpus** | n |
|---|---|---|---|
| **2 / 23 ≈ 9 %** | VLM-box ↔ SAM3-box **location** agreement on the same frame | **PH0 pilot frames** | 49 frames → 23 both-fire |
| **~⅔ "no sign at all"** | SAM3 sign-box **content**, visual adjudication | **`w120val`** (600 clips, 4,048 sign detections) | 31 crops |
| **precision 0.880** [0.795, 0.958] | SAM3 sign-box **content**, labelled + bootstrap | **`aug120`** (83 records, 538 sign detections) | 64 / 33 clips |

⚠️ **The last two disagree and it is UNRESOLVED.** Both candidate mechanisms (max-area selection,
tight crop) were **tested and REFUTED**. *"What remains uncontrolled is the corpus, so the honest
statement is 'not on aug120', not 'G1 was wrong'."* (`NEXT_4472_BUILD_INPUTS.md:63-69`, which
carries the corpus-tagged table — reuse it verbatim rather than re-deriving.)
⇒ **B3 (sign-box grounding) stays DEMOTED to diagnostic-only**, and **PH1 drops signage-text fields
by default**. Open work item, 0 GPU, ~2 h: run the aug120 adjudication on the **w120val** sign leg
before any val-side sign label is trusted.

## 5. LAYER 3 — PH1 fusion and the label legs

### 5.1 Coverage, restated (v2.0's table was overtaken)

| | aug120 | w120val (baseline) |
|---|---|---|
| clips fused | **201 / 201**, zero failures | 600 / 600 |
| with the Alpamayo layer | **201 (100 %)** | 56 (9.3 %) |
| **SAM3 records available** | ⭐ **201 / 201** *(was 86 = 42.8 % in v2.0)* | 596 (99.3 %) |

⛔ **Never pool these columns** — disjoint populations, and a 3-voter and a 2-voter tactical
majority are different instruments.
⚠️ **v2.0's `corroborations 88 / conflicts 10` on aug120 are now stale in BOTH directions** and are
withdrawn from this file. They were computed with SAM3 absent on 115 clips (two of six checks could
not fire), and **the corpus has not been re-fused since the repair**. Do not quote them; re-fuse
first.
⚠️ `alpamayo_rows = 0` on every PH0 **pilot** clip remains **undiagnosed** (`PH0_COVERAGE_AUDIT.md`
§5); the fuser later joins correctly at 201/201.

### 5.2 ⭐ Retirement discipline — a signal that cannot be computed emits `not_computable`

Two fabricated signals were **deleted rather than tuned** in 2026-08-16, and both are now
mechanically prevented from returning. This is the policy, not two incidents:

**(a) `goal_evidence: grounded` is RETIRED.** `ph1_fuse.py:114` `GOAL_EVIDENCE_RETIRED =
("grounded", "provisional")`; the verdict is now `not_computable` (`:574`) and the surviving fact
is the honest one — **`sign_like_object_present`** (`:573`, true iff `n_sign_tracks > 0`).
**MEASURED, n = 201, corpus aug120:** `grounded` **15/201 (7.46 %) → 0/201**, and
`sign_like_object_present` fires on **exactly the same 15 clips**. ⛔ **AST-pinned** —
`stack/tests/test_ph1_fuse.py` asserts the tokens are gone *and cannot be re-emitted from source*.
Three converging causes: the sign TEXT gate closed at **0/31** (`G1_RESULT.md`); §4.3's reliability
finding; and **0 of the 4** sampled `route_to` verdicts even cited a navigation sign.

**(b) The geometric lane-change gate is REMOVED — PI ruling, 2026-08-16**, verbatim in
`…/2026-08-16-s2-v1-labels/review/LANE_CHANGE_DEEP_REVIEW.md:12-16`: *"stop emitting
lane_target/Prepare Lane change from geometric gate…"*.
**MEASURED, n = 797** (corpus: **201 aug120 + 596 w120val**), artifact `review/raw/lc_emit.json`:
`LANE_TARGET` **80 (10.04 %) → 0**, `PREPARE_LANE_CHANGE` **80 → 0**.
**All 80 carry `engine_a.route.token == "follow"` with `token_valid: true` — 80/80**, i.e. the route
engine had already said they were route-following.
⚠️ **Two precision points that a summary loses.** The emitted `g_str` splits **79 →
`FOLLOW_MAIN_ROAD` + 1 → `NONE_ABSTAIN`**, not 80 → one token. And *"false positives"* is a PI
adjudication on a **19-clip subsample (14 wrong / 4 correct)**, **not an 80/80 adjudication** — the
4 correct ones are **re-homed as tactical** `a_tac: LANE_CHANGE_L/R`, not discarded.
⭐ **Bonus, MEASURED:** the gate sat above `REDUCE_TO` in the elif chain and was **suppressing 9
real decelerations** (`REDUCE_TO` 85 → 94). A false-positive gate was also causing false negatives
on a different axis.
⇒ **Engine A is now the primary `g_str`/`a_str` labeller; the VLM is demoted to corroboration.**

### 5.3 ⛔ THE ARCHITECTURAL FINDING — three legs, one trustworthy, and the vote is not a vote

**Owner: `…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` + `raw/a3_three_leg_agreement.json`.**
**MEASURED, corpus aug120.** This is the most consequential data finding since v2.0.

**Leg presence (n = 4,729):** `has_alpamayo` **4,729** · `has_vlm` **201** · `has_engine_a` **201** ·
`all_three` **201** · `alpamayo_only` **4,528**.

**(a) The VLM's longitudinal channel is a CONSTANT, and it is a code defect.** `vlm_lon3 =
{decelerate: 162, None: 39}` — **n = 162**, one value on every clip where it speaks. κ against both
other legs is **exactly 0.0000** — *"not 'no skill', but a constant."* Cause, from source: `reduce_to`
(the VLM's only deceleration verb) maps to **nothing** on either axis (**49/270 = 18.1 %** silently
dropped), and `hold_corridor` — a **lateral** verb — matches the `LON_RULES` substring `"hold"` and
becomes `HOLD` (**159/270 = 58.9 %**). ⇒ **A code fix, not a re-labelling job.**

**(b) The VLM and ego legs are NOT independent — they share their input.** `_ego_prompt_mode ==
'past'` on **201/201**; the VLM's prompt block carries `motion` and `turning`, and the ph1 ego voter
reads **exactly those two fields** (`ph1_fuse.py:318-325`). The κ signature is the fingerprint:

| pair | κ (LAT) |
|---|---|
| **VLM ↔ ego** | **0.7608** |
| Alpamayo ↔ ego | 0.2089 |
| Alpamayo ↔ VLM | 0.1717 |

⇒ ⛔ **A 2-of-3 majority satisfied by {ego, VLM} is ONE SOURCE COUNTED TWICE.** The corroboration
count is therefore **not evidence of agreement between independent observers**, and "corroborations"
must not be reported as such until the legs are separated.

**(c) Alpamayo is trustworthy only *relatively*, and it states its own bound.** Reason-token
self-consistency **78.06 %** ⇒ **≈22 % expected label error**. And **correctness against a human is
⛔ UNMEASURED — no human has reviewed one label.**

**Strategic consequence.** The multi-leg design was meant to buy independence. Measured, it buys
one usable leg with a 22 % error bar. ⇒ **Do not scale the fusion to 4,472 clips until (a) and (b)
are fixed** — scaling a vote that double-counts one voter multiplies the defect by 22×.
⚠️ **There is currently no tactical loss term in `V6LossWeights` at all**, so nothing downstream
consumes these labels yet. That is the window in which to fix them.

### 5.4 ⛔ Live provenance defect — the banked corpus lies about its own inputs

Every fused aug120 record stamps `_provenance.vlm = "vision"` while its own v2 source records
`_ego_prompt_mode = 'past'`. **MEASURED 201/201 contradicted.** `ph1_fuse.py:556-561` is **correct
at HEAD** — **the banked corpus predates the fix and has not been re-fused.** ⇒ Any consumer
reading provenance from the banked records is reading a false claim of vision-only derivation,
which is directly adjacent to the binding vision-only rule. **Re-fuse is a work item with no owner
and no date (§11).**
⚠️ Related, unclosed: the 115 fused records still carry `perception.absent =
AUG120_SAM3_STAGE_GAP`, whose owner is named as *"the aug120-fusion package"* — **a document, not a
person**.

## 6. The biggest remaining data job — the 4,472-clip w120 build

**Owner: `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md`.**

Of the **4,729** augmented clips, **257** have w120 caches and the **201 runnable were processed and
pushed** ⇒ **4,472 clips have no w120 cache.**

⭐ **Reclassified by §3.1: this is an EXTRACTION job on schedulable CPU, not a labelling job.** The
source stores the 120° camera as chunked zips + per-feature parquet chunks, so the path is
chunk-index → `v2_compressed.py build --only-clips` (scoped in the stop runbook, **not started**).

⛔ **Three costs exist and they are three different jobs. Name which one you mean.**

| job | cost | class |
|---|---|---|
| **w120 extraction** of the 4,472 | **≈6.8 h at 8 shards** (24.1 h single-shard), **≈179 GB** | **ESTIMATED**, from a measured 19.4 s/clip |
| a **second A2 pass** over the 4,472 | **~78 GPU-h** at the realised 59.6 s/clip | MEASURED cost basis |
| a **full SAM3 re-detect** at a lower floor | **~26 GPU-h** | ESTIMATED (`ph0_sam3.py:411-412`) |

⛔ **PARITY RULING, required in writing before the first batch.** The **257 existing** clips are
canonical parity members — rebuilding them is a **re-cache**, not a re-selection. The **4,472 are a
genuinely new selection** and are admissible **only as a separate, declared labelled corpus**, never
as a silent widening of `physicalai-train-e438721ae894`.

⛔ **And a consumer decision that is still open:** the reliability threshold is *"a consumer
decision, not a corpus property"* — presence-flag at 0.5 vs per-detection supervision at ≥ 0.70,
which retains only **274 / 538 = 50.9 %**. **Decide in writing before the build** (§11).

## 7. ⛔ What the augmentation does and does NOT buy — do not misprice this

The augmentation set is **4,729 clips ≈ 26 h**, which *"roughly doubles the corpus"* against 13.3 h
in training. Against Orbis-2's 5,890 h that moves the size-normalised deficit from **139× → ~70×** —
**still two orders short.**

⇒ **Stated plainly: PH0/PH1/A2 augmentation is NOT a fix for the S-W data deficit.** Per
`V6_DATA_REQUIREMENT.md` §3.4, label-side curation *"applies to our S-T/S-S stages (goal supervision
on a frozen trunk), where a few thousand well-chosen labelled clips genuinely may suffice — which is
a real and encouraging result for the PH0 pipeline, and **no help at all for S-W**."*

**The augmentation stream and the corpus-size stream answer DIFFERENT questions; conflating them
misprices both.** The S-W levers remain **P0** (new-parity-key enlargement) and **P1** (frozen
pretrained encoder) — and per `ORBIS2_ANALYSIS.md` §5.1, **P1 is still "the best-supported unrun
experiment in the programme"**, and still unrun as of 2026-08-17.
⚠️ The cost we own on P1: *"DINOv2B is 3-channel, narrow-FOV, non-driving-pretrained. Ours is
9-channel wide-FOV cylindrical. The adapter that bridges that is real work and is precisely what
the arm must measure."*
⚠️ Do not read LIMO/s1 as *"we can learn driving physics from 13 hours"* — same S-T/S-S-only caveat.

## 8. Training composition

⚠️ **v1.0's mix (~60 % PhysicalAI / ~25 % comma2k19 / ~15 % MetaDrive) never shipped and is
withdrawn** — MetaDrive is retired (D-014) and every trained arm to date is **PhysicalAI-AV only**,
on the canonical parity corpus `physicalai-train-e438721ae894` (2,376 episodes, skip-hash
`f09e44db`), or its declared w120 sibling. **Parity is sacred: anything that re-selects episodes
breaks cross-arm comparability and must be refused.** Corpus enlargement (P0) is therefore a **new,
declared parity key**, never a silent widening, with the evaluation set unchanged.

- Validation: held-out episodes per corpus, real-only. Public numbers **comma2k19-only** until §9
  resolves.
- **Semantic-coverage audit (standing metric):** per corpus and per mix, report the scenario-tag
  distribution. The A2 road-class labels (`aug_road_class.json`) are **ego-derived** — admissible as
  **labels** under the binding rule, ⛔ **never as an inference-time input**.
- ⚠️ Same test applies to every label in §5: **ask whether an input at inference contains something
  the label was derived from.** The `_ego_prompt_mode = 'past'` finding (§5.3b, §5.4) is exactly
  this failure caught on the label-production side.

## 9. License management — the firewall

1. **Tagging (now):** every experiment record lists its corpora (`data:` tags); LEADERBOARD rows
   carry the tag so exposure is auditable in one grep.
2. **Firewall (now):** public claims / demos / publications use **comma2k19 + own data** only.
   ⚠️ v1.0 said *"comma2k19 + MetaDrive (+ own data)"*; MetaDrive is retired, so the firewall is
   **narrower** than it reads there.
   ⛔ **The A2 augmentation set is INSIDE the firewall, not outside it.** It is licensed
   `nvidia-physicalai-derivative` and inherits PhysicalAI-AV's position — publishing it on HF does
   **not** make it publicly claimable.
3. ⛔ **UNRESOLVED CONFLICT, and it is recorded rather than papered over.**
   `TANITDATASET_V1_STRATEGY.md` classes PhysicalAI/Alpamayo as *"commercial-OK for internal AV dev
   but no-derivatives → firewalled, recipe-only"*; `ALPAMAYO2_SUPER_ANALYSIS.md` cites an
   **OpenMDW-1.1 derivative permission**. **These are not obviously the same reading, and the
   disagreement is recorded here rather than resolved by whichever document was read last.**
   ⇒ **PI decision** (§11).
4. **Resolution paths (decide at Phase-0 exit):** (a) seek NVIDIA permission/partnership;
   (b) replicate headline results on open corpora (OpenDV/BDD100K via H7 pseudo-labels + own data);
   (c) commission own urban collection (GoPro rig, Phase 1/2) — see `OWN_DATASET_PLAN.md`.

## 10. Flywheel outlook (Phase 1+)

comma2k19-trained inverse dynamics (H7) pseudo-labels OpenDV/YouTube/GoPro → data-efficiency slope
experiment (C2 headline) → continual-learning loop (H10) writes surprise episodes back into
training. **Unchanged and still unstarted.**

## 11. ⛔ Open decisions — and who owns them

*(The brief for v3.0 required this section. Every row below was found with **no named owner**; the
"owner" column states who it must be, not who it currently is. A maintenance contract with no owner
is why v2.0 went stale in 48 hours.)*

| # | decision | owner | blocks |
|---|---|---|---|
| 1 | **License conflict** (§9.3) — no-derivatives vs OpenMDW-1.1 | **PI** (role named in-doc; no person, no date, no trigger) | every public claim on A2 |
| 2 | **Parity ruling for the 4,472** — separate declared corpus, not a widening (§6) | **PI** | the whole w120 build |
| 3 | **Reliability threshold for the build** — presence-flag @0.5 vs supervision @≥0.70 (retains 50.9 %) (§6) | **PI / the consuming stream** | the whole w120 build |
| 4 | **Record `CONF_THRESHOLD_DEFAULT = 0.25` in `Project Steering/`** (§4.2) — currently source-comment only | **PI** (the decision is his; the *recording* is an agent task) | audit trail of a corpus-defining, destructive choice |
| 5 | **Re-fuse the aug120 corpus** — false `_provenance.vlm = "vision"` on 201/201, fix is at HEAD (§5.4) | ⛔ **unassigned** | anything reading provenance |
| 6 | **Fix the VLM `lon3` mapping + separate the ego leg** (§5.3a/b) | ⛔ **unassigned** — code fixes, ~0 GPU | scaling fusion to 4,472 |
| 7 | **Confirm Orbis-2 rows against the PDF** (§0) — open since 2026-08-15 | ⛔ **unassigned** | any GPU-day decided on the 443× |
| 8 | **`MODEL_REGISTRY.md` row + banked HF verification for the obstacle join** (§1.1) | ⛔ **unassigned** | citing the join by location |
| 9 | **Adjudicate the w120val sign leg** (§4.3) — 0 GPU, ~2 h | ⛔ **unassigned** | any val-side sign label |
| 10 | **Re-run the 4 pod-produced SAM3 records with zero detections and no liveness control** | ⛔ **unassigned** | *"none of their zeros is quotable until they are re-run"* |
| 11 | **Per-family validity mask** if the PI wants true abstention on the ex-lane-change 80 (§5.2b) | **PI** | the 80 currently fall through to `HOLD_CORRIDOR`/`REDUCE_TO` |
| 12 | **Own the maintenance contract of this file** | ⛔ **unassigned** | this file's credibility — v2.0 went stale in 48 h |

---

## Document map — where each data fact actually lives

| topic | owner |
|---|---|
| A2 augmentation set: counts, cost, holes, license | **`Project Steering/MODEL_REGISTRY.md` §11.1** |
| its design + PI brief | `…/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md` |
| Alpamayo-vs-us comparison, leverage ranking | `…/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md` |
| data requirement, lever ranking, Orbis-2 gap | `…/2026-08-07-hierarchical-wm-redesign/V6_DATA_REQUIREMENT.md`, `ORBIS2_ANALYSIS.md` |
| PH0 pre-registration, validation, coverage | `…/2026-08-07-hierarchical-wm-redesign/PREREG_PH0_VLM.md`, `PH0_PIPELINE_VALIDATION.md`, `PH0_COVERAGE_AUDIT.md` |
| PH1 fusion counts + the SAM3 gap (⚠️ pre-repair) | `…/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md` |
| the 4,472 build inputs, corpus-tagged sign table, parity ruling | `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` |
| **SAM3 dtype defect + repair** | `…/Data Engineering/…/2026-08-16-sam3-dtype-fix/SAM3_DTYPE_FIX.md` |
| **SAM3 v2 extraction, the 0.25 floor, scene concepts** | `…/Data Engineering/…/2026-08-16-sam3-extraction-v2/SAM3_EXTRACTION_V2.md` |
| **SAM3 concept reliability (precision, aug120)** | `…/Data Engineering/…/2026-08-16-sam3-concept-reliability/` |
| **`goal_evidence` retirement + AST pin** | `…/Architecture & Inference/…/2026-08-16-evidence-and-flake/EVIDENCE_AND_FLAKE.md` |
| **S2 strategic labels + the lane-change ruling** | `…/Data Engineering/…/2026-08-16-s2-v1-labels/review/LANE_CHANGE_DEEP_REVIEW.md` |
| **the 80 re-homed as tactical** | `…/Architecture & Inference/…/2026-08-16-integration-close/INTEGRATION_CLOSE.md` |
| **three-leg agreement / tactical label validation** | `…/Data Engineering/…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` |
| **train obstacle join** | `…/Data Engineering/…/2026-08-17-train-obstacle-join/TRAIN_OBSTACLE_JOIN.md` |
| sign TEXT gate (0/31) | `Project Steering/G1_RESULT.md` |
| VLM augmentation plan (2026-07-25, the *intent*) | `…/Data Engineering/TANITDATASET_V1_STRATEGY.md` |
| corpus parity key, caches, episode contract | `MODEL_REGISTRY.md` §0.1 |
| own-data collection + licensing verdict | `…/Data Engineering/OWN_DATASET_PLAN.md` |
| corpus landscape (one row per corpus) | `…/Data Engineering/Research/DATASET_LANDSCAPE.md` |

**Maintenance contract.** Refresh at every augmentation-phase boundary and whenever a corpus,
parity key, license position or **label-leg architecture** changes. ⛔ **Numbers are quoted from the
registry or the raw artifact, never from this file** — this file points, the owners measure.
⚠️ **v2.0 went stale in 48 hours because this contract has no owner** (§11 row 12). Until it has a
name, treat any figure here older than the newest `…/incoming/` package as suspect.
