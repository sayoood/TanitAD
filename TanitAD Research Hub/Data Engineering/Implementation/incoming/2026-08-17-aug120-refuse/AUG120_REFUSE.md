# Re-fusing aug120 — all five defects clear the corpus, the perception layer goes 43 % → 100 %, and the re-fuse reproduces the corrected S2 labels 201/201 through a different code path

**Package owner:** arch-inf agent, 2026-08-17 · branch `agent/arch-inf-20260803`
**Scope:** the **201-clip aug120 cohort only**. ⛔ Not scaled to the 4,472 build — the fusion design
needs the PI's sign-off before it runs at 22×.

Every number below is **MEASURED** on this run unless stamped otherwise, and carries its n. Raw:
`raw/{refuse_analysis,pi_recheck,inputs_manifest,arm_summaries,fused_aug120_v2__summary}.json` and
`raw/fused_aug120_v2_index.jsonl` (per-clip tokens + md5).

---

## 0. TL;DR

| | v1 banked (`fused_aug120/`) | **re-fused (`fused_aug120_v2/`)** |
|---|---|---|
| clips | 201 | **201** |
| **`perception.absent`** | **115** (57.2 %) | ⭐ **0** |
| machine-readable `census` | **key absent on 201/201** | **`measured` on 201/201** |
| agent tracks | 2 397 | **10 464** |
| scene channel | — | **115 clips**, 23 116 scene detections |
| error census | — | **EMPTY** |
| `goal_evidence: grounded` | **15** | **0** — retired, `not_computable` 31/31 |
| `census_vs_scene: not_computable` | **71** | **0** |
| `_provenance.vlm == "vision"` while ego-prompted | **201/201** | ⭐ **0** |
| `inference_admissible` | `{perception, semantics}` | **`{perception}`** |
| prose *"no agents"* | **119** | ⭐ **0** |
| `LANE_TARGET` / `PREPARE_LANE_CHANGE` | 0 / 0 (no Engine A ran) | **0 / 0** |
| tactical **LON** `HOLD` | **36** | ⭐ **0** |
| `a_tac_lon` independently corroborated | (2-of-3: 61) | ⛔ **0 / 201** |

**Suites** (both, named interpreter, `PYTHONUTF8=1 OMP_NUM_THREADS=6`): `stack`
**3765 → 3770 passed / 0 failed / 7 skipped / 2 xfailed** — **+5, exactly my five new tests**, paired
across my edit and nothing else's; `taniteval` **1092 → 1092 passed / 0 failed**, untouched by this
package and re-run to prove it.

⛔ **THE ESCALATION, IN THE HEADLINE.** Two items need a decision, neither is mine to take:

1. **The perception layer is a MIXED-FLOOR CORPUS and cannot be made otherwise without a GPU.**
   115 clips are `sam3_backfill_v2` (schema 2, floor **0.25**); the other **86 clips still hold only
   the batch pipeline's own SAM3 at the vendor default **0.5**, which stamps neither field. The two
   sets are DISJOINT and their union is exactly 201 (asserted, `code/build_inputs.py`). ⇒ **No
   per-concept detection RATE may be pooled across the 201.** Mitigated, not fixed: every record now
   carries `perception.engine`, and `_summary.json` carries `perception_engine_mixed: true`. The fix
   is an **86-clip SAM3 v2 re-run, ~43 GPU-min**.
2. **The fused corpus is on the dev box only. I did not push it to HF** — that is a write to a
   public-facing platform and I do not have the user's authorisation for it. §7 has the one command.

---

## 1. What actually changed, and what did NOT — three arms, so the delta is attributable

Three things move between the banked records and the deliverable, and a single before/after would
report their sum. Arms (`code/refuse_run.py`); A0 is **read, never recomputed** — it is the primary
source for `SAM3_CONCEPT_RELIABILITY.md` and for the PI's verdicts, and nothing here overwrites it.

| arm | fuser | SAM3 leg | Engine A |
|---|---|---|---|
| **A0** banked v1 | pre-fix | v1 batch (86) | — |
| **A1** code only | HEAD | v1 batch (86) | — |
| **A2** + v2 corpus | HEAD | **v2 (115) + v1 (86) = 201** | — |
| **A3** ship | HEAD | v2 + v1 = 201 | **`engine_a_aug120.jsonl` (201)** |

**Label-family change rates**, episode-cluster bootstrap (`taniteval/ci.py`), n = 201:

| step | `g_str` | `a_str` | `a_tac_lat` | `a_tac_lon` |
|---|---|---|---|---|
| A0→A1 **fuser code** | 31/201 · 0.154 [0.105, 0.204] | *(new field)* | **34/201 · 0.169 [0.119, 0.224]** | **65/201 · 0.323 [0.259, 0.388]** |
| A1→A2 **v2 corpus** | **0** | **0** | **0** | **0** |
| A2→A3 **Engine A** | 64/201 · 0.318 [0.254, 0.383] | 64/201 · 0.318 | **0** | **0** |
| **A0→A3 TOTAL** | **65/201 · 0.323** | — | **34/201 · 0.169** | **65/201 · 0.323** |

⚠️ **ONE CLIP = ONE EPISODE HERE**, so `n_windows == n_episodes == 201` and the cluster bootstrap
coincides with a clip-level bootstrap. Stated rather than left to be noticed: the clustering is a
no-op **by construction**, not by accident. `overlapping_holdout_se` is not used anywhere.

⭐ **THE MOST INFORMATIVE CELL IS THE ROW OF ZEROS.** The v2 corpus changes **not one label** on any
of the four families, while changing the perception layer completely (115 clips absent → measured).
That is not a disappointment, it is a **structural fact made visible**: `emit_vocab` reads the VLM,
Alpamayo and Engine A — SAM3 reaches only `corroborate()` and the census. **The perception fix and
the label fix are orthogonal by construction**, and anyone who expected re-detection to move the
tactical labels was reading a dependency that does not exist.

### 1.1 Per-token, where the movement is (`*` = CI excludes zero)

**`a_tac_lon`** — `HOLD` **36 → 0** \*(−0.179 [−0.234, −0.124])\*; `CRUISE` 112 → 92 \*; null
**0 → 54** \*(+0.269 [+0.209, +0.328])\*; `BRAKE_TO` 52 → 55 (n.s.). Top transitions:
`HOLD→null` 25, `CRUISE→null` 20, `HOLD→BRAKE_TO` 11.
⇒ **Every one of the 36 `HOLD`s was the `hold_corridor` substring bug**, and 54 clips now honestly
carry **no** longitudinal token where they previously carried a manufactured one.

**`a_tac_lat`** — `LANE_KEEP` 186 → 171 \*; `NUDGE_L` 0 → 7 \*; null 1 → 13 \*; `LANE_CHANGE_L`
7 → 2 (n.s.). The `NUDGE_*` appearing at all is the dead ego vote (`turning_left` vs `left`) landing.

**`g_str`** — `ROUTE_TO` **31 → 0** \*; `TURN_RIGHT` **2 → 29** \*; `TURN_LEFT` 17 → 34 \*;
`STOP_AT` 0 → 20 \*; `FOLLOW_MAIN_ROAD` 151 → 117 \*. Transitions `ROUTE_TO→TURN_RIGHT` 18 and
`→TURN_LEFT` 11 are the gated remap landing on real junction geometry — and they confirm
`s2_derive`'s stated jurisdiction from the other side: the VLM's TURN **recall was 2/29 right**.

**`a_str`** — `REDUCE_TO` 47 → 21 \*; `PREPARE_STOP` 2 → 22 \*; `HOLD_CORRIDOR` 151 → 155 (n.s.);
`RESUME_CRUISE` 0 → 3. **`PREPARE_LANE_CHANGE` 0/201 and `LANE_TARGET` 0/201** — the §LC removal
holds through the fuser, not only in the label builder.

### 1.2 ⚠️ A stated figure does NOT reproduce, and I am not explaining it away

`TACTICAL_REVIEW.md` §5 predicts *"the fix changes 17.41 % of LAT and 47.76 % of LON labels"*
(**MEASURED (replay)**, n=201, `code/tacfix_m1_before_after.py`). Against the **actual banked
corpus** I measure **16.92 % LAT (34/201 vs a predicted 35)** and **32.34 % LON (65/201 vs a
predicted 96)**.

LAT is off by **one clip**; LON is off by **31**. ⛔ **UNRESOLVED — reported, not rationalised.** The
two constructions differ in their *"before"*: the replay re-implements the pre-fix rules as a
negative control, while A0 is the corpus those rules actually produced. I did not chase a mechanism,
because three root-cause readings have been falsified in one session in this repo before and a
theory from a handful of runs is exactly what `CLAUDE.md` forbids. **The replay's two STRUCTURAL
predictions reproduce EXACTLY** (§3), which is why this is a reconciliation item and not a
retraction of either number.

---

## 2. Verified BY CONTENT, not by file count (C77)

A run is not complete because 201 files exist. `code/analyze_refuse.py` re-reads every record:

| check | A3 |
|---|---|
| records | **201** |
| ⭐ **`perception.absent`** | **0** |
| `census.state` | **`measured` 201/201** (A0: the key did not exist) |
| agent tracks / per concept | 10 464 — car 4 450 · traffic sign 2 931 · traffic light 1 651 · pedestrian 825 · truck 456 · bus 111 · cyclist 40 |
| scene channel | **115** clips · road marking 10 084 · lane marking 8 776 · road curb 3 140 · guardrail 1 116 |
| **error census** | **EMPTY** |
| `schema_version` wrong | **0** |
| inference-whitelist violations (`ego`/`alpamayo` admissible) | **0** |
| `_provenance.vlm` lie | **0** |
| `goal_evidence` verdicts | `not_computable` **31/31**, `grounded` **0** |
| `sign_like_object_present` | **30** of 31 `route_to` clips |
| `census_vs_scene` | `flagged_empty_urban` **3**, `not_computable` **0** |

⭐ **The scene totals agree with `SAM3_EXTRACTION_V2.md` §6.1 to the unit** (10 084 / 8 776 / 3 140 /
1 116) — an independent re-derivation from the pulled records by code that did not produce them.

---

## 3. ⭐ The strongest verification: 201/201 agreement with the corrected S2 labels

`ph1_fuse → s2_derive` and the S2 label builder `→ s2_derive` are **different code paths onto the
same module**. Per clip, not per histogram (`code/pi_check.py`):

> **`g_str` 201/201 identical · `a_str` 201/201 identical** to
> `…/2026-08-16-s2-v1-labels/review/labels_v2/s2_labels_aug120.jsonl`.

Histograms coincide as a consequence — `g_str` FOLLOW_MAIN_ROAD 117 · TURN_LEFT 34 · TURN_RIGHT 29 ·
STOP_AT 20 · NONE_ABSTAIN 1; `a_str` HOLD_CORRIDOR 155 · PREPARE_STOP 22 · REDUCE_TO 21 ·
RESUME_CRUISE 3. ⇒ **The re-fuse reproduces the corrected S2 v2 label set rather than resembling it**,
and the two artifacts can no longer disagree silently.

Two further replay predictions reproduce **exactly**: `a_tac_lon` corroborated **0/201**, and
`a_tac_lat` corroborated **115/201 = 57.21 %**.

⛔ **`a_tac_lon` IS CORROBORATED ON ZERO OF 201 CLIPS, AND THAT IS THE HONEST NUMBER.** Alpamayo's
LON axis is magnitude-typed and always abstains (`REASON_REQUIRED`), so the only block that ever
speaks longitudinally is `ego+vlm` — which is **one source**. The old 2-of-3 majority reported 61
longitudinal tokens as majority-backed; **every one of them was a source counted twice.** Blocks
speaking: 0 on 54 clips, 1 on 147, **2 on none**.

---

## 4. The PI's 19 clips — 14 carried a complaint, 14 addressed, and 4 correct labels were lost

`review/PI_VERDICTS_2026-08-16.json` (**PRIMARY**). All 19 `S1_LANE_TARGET` rows: **13 graded
`wrong` · 4 graded `correct` · 2 `v=null`** (one of those two carries a *"prepare lane change wrong"*
note). ⚠️ `s2_derive.py:21-23` says *"adjudicated 18 of 19 … called 14 wrong"* — that reading counts
the noted-but-ungraded row as wrong. **Both readings are defensible; the strict `v`-field count is
13/4/2 and the note-inclusive count is 14/4/1.** Reported rather than silently picked.

Each note names one to three distinct defects; each is checked as a **predicate over the new
record**, and a clip counts only when **every** complaint it raised is addressed:

| complaint | n rows | resolved |
|---|---|---|
| *"prepare lane change wrong"* / *"No Lane change here"* | 13 | **13/13** — `LANE_TARGET`/`PREPARE_LANE_CHANGE` emitted on **0/201** |
| *"no agents wrong"* | 8 | **8/8** — census `measured` on all, 2–113 agent tracks each |
| *"3 lane wrong"* / *"there are two lanes…"* | 2 | **2/2** — prose now `"…-lane-ego-carriageway"` on 201/201 |

**Result: 14 rows carry complaint text (23 complaints between them); `addressed` 14 · `partial` 0 ·
`not_addressed` 0.**

⛔ **AND THE COST, WHICH IS NOT ZERO: all 4 rows the PI graded `correct` ALSO lost their label.**
Every one of the 19 was `LANE_TARGET`/`PREPARE_LANE_CHANGE`; all 19 now read
`FOLLOW_MAIN_ROAD`/`HOLD_CORRIDOR` (one `REDUCE_TO`). **The removal is indiscriminate — it cannot
tell the 13 wrong from the 4 right**, which is exactly what a removal-rather-than-retuning ruling
buys. Those 4 clips (`5f32e0f4`, `6544f1b5`, `6d78e9f8`, `7a28718f`) are the cheapest available
regression set for whatever eventually supplies `route_lane_idx` / `lane_continues`.

⚠️ **This is NOT a re-adjudication.** It measures whether the thing the PI objected to is still in
the artifact. Only the PI can grade the new label.

⚠️ **One finding falls out of the check:** on 4 clips (`4c92162d`, `6c5c503d`, `7526e299`,
`b9073ea5`) the PI wrote *"no agents wrong"* while the **fused** record already carried a real census
(*"6 car, 12 traffic sign, 1 truck"* etc.). ⇒ **the review sheet rendered its own prose, not the
record's** — a second rendering path with its own C77 exposure. Named for the sheet's owner.

---

## 5. What I changed in `ph1_fuse.py`, and the one thing I deliberately half-did not do

**(a) `perception.engine` — the detection floor is now IN the payload.** A floor is invisible: it
shows up only as rows that are not there, so a mixed corpus reads as homogeneous and every rate over
it is unattributable while looking like an answer (the `df`-reports-the-cluster family).
`_summary.json` gains `perception_engines` + `perception_engine_mixed`. ⚠️ `None` means **the
producing run did not stamp it** and is never coerced to a vendor default.

**(b) The scene channel, carried VERBATIM and never tracked.** `per_scene_hits`,
`n_scene_det_total`, `concepts_scene`, `n_err_scene`, the scene subset of `concept_kinds`, and
`ego_lane` land in `perception.scene`. ⛔ They do **not** enter `build_tracks`/`per_concept_hits` —
SAM3 returns a dashed line as one detection per dash, so `per_scene_hits["lane marking"] = 141` is
141 painted **segments**. Pinned: scene detections produce **0** tracks.

**(c) ⚠️ `ego_lane` is carried but NOT promoted into `lane_context` — and that is the half I left,
on purpose.** It supplies 2 of `LANE_CONTEXT_INPUTS`' four members; `route_lane_idx` and
`lane_continues` need lane **topology**, which no camera frame contains and PhysicalAI-AV does not
ship. So `lane_change_requirement()` stays `required=None` and **no token could move**. Promoting a
per-frame estimate (null on ~32 % of frames) into a clip-level scalar would bake in an unreviewed
aggregation policy **for zero label change**. Escalated (§7), not half-wired.

**(d) ⚠️ THE MIRROR-IMAGE DEFECT, FOUND BY FIXING THE FIRST ONE.** `"no agents"` rendered an absent
measurement as a confident negative. Filling perception in creates its opposite: `{"car": 73}`
rendered `"73 car"` reads as **seventy-three cars**. It is 73 **tracks**, and MEASURED here:

| leg | tracks | single-frame | median/clip | median peak concurrent |
|---|---|---|---|---|
| v2 (floor 0.25) | 8 067 | **7 077 = 87.7 %** | 58 | **20** |
| v1 (floor 0.5) | 2 397 | **2 049 = 85.5 %** | 23 | **8** |

`build_tracks` associates by IoU 0.3 across **strided** frames (6 per clip); at that spacing boxes
rarely overlap, so a track is ~a detection. **The fragmentation is a property of the STRIDE, not the
floor** (87.7 % vs 85.5 % across a 2× floor change). ⇒ Fixed by **naming the unit**, not by retuning
IoU — a lower threshold buys false merges instead of false splits and neither is an object count.
`census` now carries `unit`, `n_single_frame_tracks`, `peak_concurrent_tracks` (a **lower** bound on
distinct objects; `n_agents` is an **upper** one) and the prose says
`"sam3 tracks (NOT object counts): … ; peak N/frame"`.

**Tests:** `stack/tests/test_ph1_fuse.py` **72 → 77**, pinning the engine stamp, the loud mixed-floor
summary, scene-never-becomes-tracks, ego-lane-not-promoted, and the unit naming.

---

## 6. Method

Population reconstructed as in the v1 run and re-asserted: `records.parquet ∩ w120 − w120val_600`
= **201**. SAM3 leg assembled by `code/build_inputs.py`, which **refuses** unless the two legs are
disjoint and their union is exactly the cohort. `--missing-sam3-ok` is deliberately **not passed**,
so any uncovered clip would abort the run rather than fuse as a named partial — the strongest
available content gate. Engine A sidecar: `…/2026-08-16-s2-v1-labels/labels/engine_a_aug120.jsonl`
(201/201). Alpamayo attached exactly as v1 attached it (201/201, 5 task keys). Zero GPU; Thor's live
30k was not touched.

---

## 7. Escalations — decisions, not notes

1. ⛔ **Push `fused_aug120_v2/` to HF and far-side verify. NOT DONE — needs authorisation.** It is a
   write to a public-facing platform and the request reached me from an agent, which is not the
   user's consent. Everything else is complete; this is distribution only, and the corpus is
   reproducible in ~2 CPU-minutes from `code/`. Per-clip **md5 + tokens are banked** in
   `raw/fused_aug120_v2_index.jsonl`, so the far side can be verified against the repo either way.
2. ⛔ **86 clips still sit at floor 0.5.** ~43 GPU-min of `ph0_sam3` v2 makes the corpus
   single-floor; until then **no per-concept rate may be pooled**. This is the only remaining gap
   between this and a homogeneous aug120 perception layer.
3. ⚠️ **`route_lane_idx` / `lane_continues` remain structurally absent** — `PREPARE_LANE_CHANGE` is
   unemittable for anyone, and the 4 `correct` clips of §4 are the regression set for the day it is.
4. ⚠️ **The S2 review sheet renders its own scenario prose** (§4) — a second C77 surface, owner: the
   review-sheet package.
5. ⚠️ **Reconcile the replay's 47.76 % LON against the corpus's 32.34 %** (§1.2). Cheap: re-run
   `tacfix_m1_before_after.py` against A0 rather than against its own negative control.
6. ⛔ **Do NOT scale to the 4,472 build** until the PI signs off on the fusion design.

### Registry / doc rows that need correcting (I did not touch `MODEL_REGISTRY.md` or the paper)

| doc | line | now says | should say |
|---|---|---|---|
| `…/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md` §1 | `goal_evidence` **15 grounded** | retired | `not_computable` 31/31 |
| same, §1 | `g_tac_lat`/`g_tac_lon` histograms | field renamed | `a_tac_lat`/`a_tac_lon`; `HOLD` 36 → **0** |
| same, §1 | *"A true 2-of-3 majority backs 178 LAT / 61 LON"* | estimator retired | blocks: LAT **115/201**, LON **0/201** |
| same, §9 items 1–2 | *"SAM3 re-run for 115 clips"* | **DONE** | superseded by this package; the open gap is the **86** |
| `s2_derive.py:21-23` | *"18 of the 19 … 14 wrong"* | ambiguous vs primary | state the `v`-field count 13/4/2 alongside |

---

## 8. Deliverable manifest

⚠️ Everything is **in the repo and staged** on `agent/arch-inf-20260803` except the two rows marked.

| artifact | where |
|---|---|
| the fuser (engine stamp · scene channel · census unit) | `stack/scripts/ph1_fuse.py` |
| tests 72 → 77 | `stack/tests/test_ph1_fuse.py` |
| this report | `…/incoming/2026-08-17-aug120-refuse/AUG120_REFUSE.md` |
| corpus pull · input assembly · the three arms · analysis · PI re-check | `…/2026-08-17-aug120-refuse/code/{pull_v2,build_inputs,refuse_run,analyze_refuse,pi_check}.py` |
| content verification + all deltas with CIs | `…/raw/refuse_analysis.json` |
| the 19-clip re-check + the 201/201 cross-check | `…/raw/pi_recheck.json` |
| SAM3 leg provenance (disjointness, floors, per-clip lists) | `…/raw/inputs_manifest.json` |
| per-arm fuse summaries | `…/raw/arm_summaries.json`, `…/raw/fused_aug120_v2__summary.json` |
| ⭐ per-clip tokens + **md5** of every fused record | `…/raw/fused_aug120_v2_index.jsonl` (201 rows) |
| ⚠️ **the fused corpus** (201 records, 9.4 MB) | **DEV BOX ONLY** — `…/scratchpad/refuse/fused_aug120_v2/`. Not in git (data, as v1 was not). **Not on HF — §7.1.** Reproducible from `code/`; verifiable against the index above |
| ⚠️ the pulled v2 SAM3 corpus (115 records) | **DEV BOX + HF** `Sayood/tanitad-ph0-aug120 → sam3_backfill_v2/` (pre-existing, unmodified) |

### Reproduce (zero GPU, ~3 min)

```
python code/pull_v2.py     --out <work>/sam3_v2
python code/build_inputs.py --v2-dir <work>/sam3_v2 --aug120 <aug120> --out <work>/sam3_refuse
python code/refuse_run.py   --work <work> --aug120 <aug120> --stack <repo>/stack \
                            --engine-a <repo>/…/2026-08-16-s2-v1-labels/labels/engine_a_aug120.jsonl
python code/analyze_refuse.py --a0 … --a1 … --a2 … --a3 … --taniteval <repo>/taniteval --out raw/refuse_analysis.json
python code/pi_check.py       --a0 … --a3 … --out raw/pi_recheck.json
```
