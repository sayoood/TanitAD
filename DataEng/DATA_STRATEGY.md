# TanitAD Data Strategy (v4.0, 2026-08-18 — supersedes v3.0 of 2026-08-17)

> **Role of this file.** `Project Steering/HANDOVER_TO_LOCAL_2026-08-15.md` §1.A cites it as *"the
> strategy"*, so it is a **cited owner** and may not go stale. It is deliberately an **INDEX with
> the decisions on it**: numbers live in the owner documents named below and in
> `Project Steering/MODEL_REGISTRY.md`, never here. Every number it *does* carry is stamped with
> its **evidence class, its n, and its CORPUS** — because three of the live disagreements in this
> programme were corpus-scope errors, not measurement errors, and **one of them has since been
> settled and was neither** (§4.3).
>
> ⚠️ **The handover cites this file as "v2.0". Cite it by PATH, not by version** — it has been
> refreshed twice since that sentence was written, and the version in the citation will keep
> rotting. Do not read the handover's "v2.0" as evidence about what this file currently says.
>
> ⚠️ **Section renumbering in v4.0:** v3.0's **§11 (open decisions) is now §12.** §11 is a new
> section (standing data-engineering hygiene). **Cite this file by section HEADING, not by number** —
> and that rule now applies to every document this file points at, several of which were rewritten
> in place this week, voiding every line citation into them.

---

> ## What changed in v4.0 — and why it is a strategy change, not a number refresh
>
> v3.0 was written on 2026-08-17 and was overtaken inside 24 hours by **five** things:
>
> 1. ⛔ **The augmentation corpus is not disjoint from the training corpus, and never was** — the
>    aug120 perception cohort **IS** the parity-train intersection, exactly. The rate that matters
>    is **78.21 %**, and the *worse* direction is that the scheduled 4,472-clip build would pull
>    **15 % of the deployed val set into training** (§1.2). This changes the **admissibility** of
>    the biggest planned data job, not its cost.
> 2. ⭐ **The membership question is now ANSWERABLE** on any host, with no pod, via a committed
>    per-clip digest oracle. The old non-overlap assumption was **unanswerable, not lazy** — and it
>    is now a one-line call (§1.2b).
> 3. ⭐ **The perception layer is UNIFIED** — 201/201 at one floor, one schema, residual zero. The
>    mixed-floor hazard that made every pooled per-concept rate inadmissible is **closed** (§4.1b).
> 4. ⭐ **The sign disagreement is RESOLVED, and it was an INSTRUMENT, not a corpus** (C87). v3.0's
>    §4.3 "UNRESOLVED" is **withdrawn**; the SAM3 sign channel is **released as a presence flag**
>    with stated limits (§4.3).
> 5. ⭐ **The extraction economics were 4.7× pessimistic** and concurrency is now measured, not
>    guessed: the extraction may run **alongside live training** at a real but ~6×-under-threshold
>    cost (§6.2).
>
> Two of v3.0's open decisions **closed by execution** (re-fuse; VLM `lon3` mapping) and one
> **closed by measurement** (the sign adjudication). Four new ones opened. §12 is the ledger.

## 0. The one thing to read first — UNCHANGED from v3.0

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
named owner** (§12).

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

⚠️ **The rig caveat stands, and it has now bitten a launch path.** PhysicalAI front-wide has **two
camera rigs** (cy ≈ 543 rig A, cy ≈ 755 rig B). Crop around the **per-clip** principal point; a
geometric-centre crop is ~215 px wrong for rig B. See §6.3 — the w120 pilot's first launch attempt
would have silently produced a mis-cropped corpus, and the crash was the *good* outcome.

⚠️ **Say which read-set you mean — never the bare phrase "our ingest".** That phrase is what let the
feature count rot four times. **The counts are PINNED to source** by
`stack/tests/test_physicalai_feature_readset.py` (**MEASURED, 9 tests, all passing 2026-08-18**);
do not hand-edit them, and note the test's failure message names the documents to fix.

| layer | reads | of 36 |
|---|---|---|
| `physicalai_r0.py` (r0 clip selection) | `egomotion`, `camera_front_wide_120fov` | **2** |
| `physicalai.py` (the **episode build**) | + `camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions` | **5** |
| **program-wide** (incl. the pod-side side-car join) | + `obstacle.offline` | **6** |

### 1.1 The train-corpus obstacle join (2026-08-17) — and one carried caveat now WITHDRAWN

**MEASURED (ours)**, artifact `…/incoming/2026-08-17-train-obstacle-join/raw/train2400_agents.meta.json`,
owning document `…/2026-08-17-train-obstacle-join/TRAIN_OBSTACLE_JOIN.md` §*The result (MEASURED)*:

| | |
|---|---|
| corpus | the **parity train corpus** `physicalai-train-e438721ae894`; **2,400 episodes requested** |
| joined | **2,308 episodes · 433,040 frames · 12,122,129 agent boxes** · `visible_frac` 0.4106 |
| the 92 not joined | fully accounted **by name**: `no_obstacle` 79 · `registration_failed` 10 · `bad_clip` 3 (2,308 + 92 = 2,400 exactly) |
| verification | md5 `24cbdca8…`, plus a **read-back through the real consumer** (`train_p8_occupancy.JoinFileReader` → 433,040 records / 2,308 clips / occlusion flags present) — not a file-size check |

This is what makes `obstacle.offline` a **train-side** signal rather than a pod-side curiosity, and
it is the sixth PhysicalAI feature the programme reads (§1's table).

⚠️ **Reusable method, worth knowing before anyone re-runs it:** the join needs **poses, not video**.
`corpus_first_clips()` takes a manifest fast-path — a **13 MB** `_v2manifest.pt` against an **80 GB**
corpus — so the whole thing ran on the dev box while Thor was mid-30k, off a metadata-only mirror.
`PYTHONUTF8=1` is required or the script dies printing its own warning glyph. ⚠️ **Its background log
was 0 bytes for the entire 3-hour run while the job was healthy** — for this job the artifact is the
progress indicator, not the log.

⛔ **Two gaps, both work items, neither a number problem.** (a) The HF location
(`Sayood/tanitad-ph0-aug120` → `joins/`) is asserted in **exactly one prose line** and appears in
**zero code paths** — evidence class **INHERITED-from-prose**. (b) There is **no `MODEL_REGISTRY.md`
row** for the join. Until both land, cite the meta sidecar, not the HF path.

⛔ **WITHDRAWN from v3.0 — the carried F-18 caveat.** v3.0 closed this subsection with *"the F-18
slot probe already returned NEGATIVE (D1) at step 9000… This join does not change that result."*
**D1 is WITHDRAWN** (commit `1ebd261`): the slot probe **failed its own positive control** — handed
a tensor encoding the frame's own ground-truth boxes it scored **6.319 m and lost to a constant**,
while a ridge on that identical tensor recovered the lead gap at **1.016 m, r = +0.979**. An
instrument that cannot read the answer off a tensor containing it was never measuring the world
model. Root-cause class: **an instrument validated only by NEGATIVE controls** — five negative
controls were structurally blind to it, because the defect *only bites a working arm*.
⇒ **Do not quote D1 in either direction.** The repaired probe (`--n-queries 16`) gives the first K1
PASS in F-18, but on one seed against a measured 3.096 m seed spread. **Path to a valid D1:
{oracle, latent, null} × 16 queries × ≥3 seeds, re-read at 30 k — ~9 fits, no trunk compute.** The
join's real contribution is that this re-read can now run on the **full parity train corpus**
instead of a 61-clip non-parity subset.

### 1.2 ⛔⛔ NEW AND MOST CONSEQUENTIAL — the augmentation corpus is INSIDE the training corpus

**Owner: `…/incoming/2026-08-18-alpamayo-parity-exclusion/ALPAMAYO_PARITY_EXCLUSION.md`; registry
row `MODEL_REGISTRY.md` §12.4; classes **C112** (finding) and **C113** (correction) in
`RETRACTION_LOG.md`.**

**RETRACTED:** the standing claim that the Alpamayo augmentation set does **not** overlap the parity
corpus. It was assumed from provenance — *"different source ⇒ disjoint"* — and never computed.

⭐ **MEASURED BY ME 2026-08-18, re-derived in this repo through the committed oracle with no pod and
no GPU** (`…/incoming/2026-08-18-data-strategy-refresh/code/verify_strategy_claims.py` →
`raw/strategy_claims_verified.json`, `ALL_CLAIMS_REPRODUCED: true`). These are not inherited from
`contamination.json`; they are recomputed from the primary id lists and the digest sets.

| direction | measurement | n / denominator | what it means |
|---|---|---|---|
| **catalogue rate** *(do not quote this one)* | **201 / 4,729 = 4.2504 %** | the whole A2 record set | flattering — it counts clips that **do not exist as video** |
| ⛔ **buildable rate TODAY — QUOTE THIS** | **201 / 257 = 78.2101 %** | only 257 of the 4,729 have w120 video built | an Alpamayo eval split buildable today is **REF-A-I-JEPA scale (~80 %) train-contaminated** |
| ⛔⛔ **the worse direction** | **6 / 40 = 15.0 %** of the canonical val episodes are inside the A2 record set | 40 deployed val episodes; 40/40 `clip_sha8` agreements against a second source | not *"an eval split contains train clips"* but **"a train corpus is about to swallow the deployed val"** — the set behind **every published open-loop number** (881 stride-8 windows) |

⇒ **ROOT-CAUSE CLASS (C113): a contamination rate quoted over the CATALOGUE rather than over the
BUILDABLE SET. The denominator that flatters is the one that is easy to count** — the same family
as `df` reporting the cluster instead of the pod quota.

⭐ **AND THE 201 DO NOT *COINCIDE WITH* THE aug120 PERCEPTION COHORT — THEY **ARE** IT.**
**MEASURED BY ME:** `fused_aug120_v2_index.jsonl`, `fused_aug120_v3_index.jsonl` and the banked
exclusion list `alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt` all carry **201** ids whose
sorted-id sha256 is **`80632f17292eb5fc484956338a0aad7b40f91ff66502faa11a5cad49f9a0439e`** — one
digest, three files, **set equality True**. Through the oracle the cohort is **201/201 inside parity
train** and **0/201 inside the deployed val**.
**The mechanism is one expression** in `stack/scripts/aug120_pipeline.py` —
`todo = (records ∩ w120_corpus) − done`, where the w120 corpus **is** the parity geometry sibling.
⇒ **The cohort was SELECTED FROM the train corpus.** *A matching count between two sets is a prompt
to test set EQUALITY, not a coincidence to note.*

⇒ **WHAT THIS DOES AND DOES NOT COST US TODAY.**
- ✅ **Blast radius on published numbers is ZERO** — all 73 `taniteval/results/*.json` were opened;
  `registry.py`'s three eval corpora contain no Alpamayo corpus; the aug120 numbers that exist are
  **label-quality only**. The ADE numbers near the word "Alpamayo" are the *Alpamayo-2-Super model*
  on the 290-clip OOD-val corpus, a different corpus.
- ⛔ **The aug120 label set is a TRAIN leg, not a held-out one.** Anything that reads it as an eval
  split is reading train. The genuine held-out leg is **`w120val`: 600 clips, 0 in parity train**.
- ⛔ **The trigger for the val-swallowing direction is already scheduled** — the 4,472-clip build
  (§6). **No existing guard fires**: `parity.py` §9 checks a cache against *its own* corpus digest,
  and an augmentation corpus is a different corpus by construction.

### 1.2b ⭐ The membership oracle — how to actually use it

**Why it did not exist before, which matters more than the fix.** The parity manifest carries only
`clip_id_sha256_sorted` — a digest of the **whole sorted list**. That is a set **identity**; it
**cannot test one element**. The clip ids are gated-confidential and live only on pods. So *"is clip
X in the parity train split?"* **had no answer on any other host** — and an unanswerable question
gets answered by provenance. ⇒ **The assumption was unanswerable, not lazy, and the fix is the
missing ORACLE, not a reminder.**

**Committed, in-repo, no pod access, re-walked on every test run:**

| file | n | what it holds |
|---|---|---|
| `stack/tanitad/data/parity_train_clip_digests.json` | **2,400** | `sha256(clip_id)` for every parity **TRAIN** clip |
| `stack/tanitad/data/deployed_val40_clip_digests.json` | **40** | `sha256(clip_id)` for the canonical **val40** deployment |

**The two calls, and which one you need:**

```python
from tanitad.data import parity

# CONSUMING an eval split — "does it contain TRAIN clips?"  (parity.py §10)
parity.assert_eval_clips_disjoint_from_parity_train(clip_ids, label="my-eval")   # refuses
kept, dropped, rec = parity.filter_eval_clips(clip_ids, label="my-eval")         # constructs
parity.assert_v2_eval_cache(cache_dirs, label="my-eval")                         # same, for a cache dir

# BUILDING a train / augmentation corpus — "does it swallow the deployed val?"  (parity.py §10b)
parity.assert_train_clips_disjoint_from_deployed_val(clip_ids, label="alpamayo-4472")
kept, dropped, rec = parity.filter_train_clips(clip_ids, label="alpamayo-4472")
```

- **`filter_*` where a split is CONSTRUCTED** (you own membership); **`assert_*` where a split is
  CONSUMED**. `filter_*` returns `n_in`/`n_kept`/`n_dropped` so a report **can never quote the
  pre-filter n**.
- The assert is **PASS/FAIL per split, never a percentage threshold** — stated reason: a threshold
  would have waved the 4.3 % case through.
- The only escape hatch is `sanctioned_audit="<why>"`. It takes the **reason, not a boolean**,
  prints the disclosure, and stamps `decision_grade: False` on the returned record.
- 🔒 **Membership exact, enumeration impossible.** Every printed or raised message is **counts
  only**; ids come back only from `clips_in_parity_train` / `clips_in_deployed_val`, in-process,
  and those disclose nothing the caller did not already supply. The digest set is a one-way image
  and cannot be run backwards into ids.
- ⭐ **The mint refuses to write a wrong file.** `stack/scripts/make_parity_clip_digests.py` will not
  emit a digest set unless its source reproduces the committed corpus digest (the
  `register_v2_geometry_sibling` contract) — *"a digest set minted from anything but the registered
  clip split would silently authorise the wrong exclusions."* The val40 subset cannot reproduce a
  corpus digest by construction, so it is proved instead by **40/40 sha8 agreements with an
  independently banked second source**, and refused without them. `load_clip_digests` self-checks
  count, duplicates and a `digest_of_digests` on every read.
- **Pinned by `stack/tests/test_eval_contamination.py` — MEASURED, 17 tests, all passing
  2026-08-18.** Each guard was neutered in turn and required to go red (4, 8 and 1 failures).
- ⚠️ **Derived, never hand-listed.** The question asked is *"is this clip in the parity train
  split?"*, so **the next 4,472 clips need no list update.**

⛔ **A live contradiction, flagged not fixed — PI decision (§12 row 13).** `parity.py` §9 states
*"the repo carries only the digests"*. **MEASURED: it is false.** `…/2026-08-17-thor-concurrency-pilot/`
commits **4,729 + 201 raw PhysicalAI-AV clip ids in plaintext** (`alpamayo_clip_ids.txt`,
`alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt`), both **tracked** — verified by
`git ls-files`, at the **package root, not under `raw/`** (my first probe used the `raw/` path and
returned empty; *absence at one location is not absence*). Deleting them breaks the pin; keeping
them contradicts the stated gating rule. **Not an agent's call.**

⚠️ **A gap the oracle does NOT cover:** there is **no parity-VAL 600-clip oracle**. §10b covers the
**deployed 40**, which is what every published statistic is quoted over. Overlap with the other 560
is not a leak but is a comparability hazard; minting it needs the 600 ids, which live only on a pod.

## 2. The strategic frame — where the bound actually sits

v2.0 treated augmentation as one programme with one bottleneck. It is **three layers with three
different economics**, and as of 2026-08-18 **none is bounded by labelling** — but one is now bounded
by something new:

| layer | status | what bounds it now | class of cost |
|---|---|---|---|
| **1 — Alpamayo-2 Super** (§3) | **labels COMPLETE for all 4,729 clips** | ⛔ **a PARITY RULING (§1.2), then our own 120° video extraction** | **decision, then CPU / IO — schedulable, and now proven concurrent-safe** |
| **2 — PH0 perception** (§4) | ⭐ **UNIFIED — 201/201, one floor, one schema** | ⛔ **durability: it lives on ONE DISK** | ~0 GPU; a push, not a run |
| **3 — PH1 fusion + label legs** (§5) | ⭐ **two of three defects FIXED and re-fused**; one remains | **leg independence**, not leg count | **code fixes, ~0 GPU** |

⇒ **The strategy this implies.** Stop spending decisions on *"can we get more labels?"* — we have
them. Spend them on (a) **the parity ruling that now gates the extraction**, (b) **banking the
unified perception corpus off one disk**, (c) **separating the ego and VLM legs**, and (d) the
**only two levers that touch the 443× deficit** (§7). Layers 1–3 are label-side; per
`V6_DATA_REQUIREMENT.md` §3.4 they help S-T/S-S and are **no help at all for S-W**.

## 3. LAYER 1 — Alpamayo-2 Super: the labels are done

**Owner of the numbers: `Project Steering/MODEL_REGISTRY.md` §11.1.** ✅ **This row is current.**

**MEASURED** from `records.parquet` (sha256 `ecae276d…`, verified against the HF far side):
**23,644 rows / 4,729 unique clips / 5 tasks / 12 columns**, `seed` 42, `model_id
nvidia/Alpamayo2-Super` on 100 % of rows, **zero errors**, **78.4 wall-hours** at **59.6 s/clip** —
**~3.4× cheaper** than the DESIGN doc's ~200 s/clip ESTIMATE. Published as
`Sayood/tanitad-alpamayo2-augmentation`; **no raw sensor data** (rows join by `clip_id` + `t0_us`).
Delivery **4,729 / 4,800 selected = 98.52 %**. Independently re-verified 2026-08-16 by a second,
separately written probe — all headline counts reproduced exactly.

⛔ **One quantisation arm only** — `NF4-backbone-4bit-UNVALIDATED` on **100 %** of rows. There is
no bf16 arm, therefore **no quantized-vs-full comparison inside the dataset**.
⚠️ **The dataset card understates the completeness hole by 356×** (claims 4,800 / 23,999 and *"one
task row missing"*; measured, **356 of 24,000 rows missing**). The four stratified road classes
(urban 1,884 · intersection_rich 1,241 · highway 384 · unstructured 83) are **100 % complete** —
every dropout falls in the unlabelled remainder. Accounting in registry §11.1.

### 3.1 The correction that moved the strategy — extraction, not labelling

**PI correction, mid-task, 2026-08-16** (`…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md`,
§*the bound*):
> *"Alpamayo's labels are complete for all 4,729… The bound is on OUR video, not the labels."*

⇒ **MEASURED**: the labels need nothing regenerated — full camera rig, 4,729 clips, 0 unparsable
rows. The missing artifact is **our** `<clip_id>.v2ep.pt`, 256×640 cylindrical
`camera_front_wide_120fov`. *"The path to scale is extracting more video — schedulable CPU compute —
not re-labelling."*

⚠️ **This reclassified the biggest data job from a decision into a schedule. §1.2 has now
re-classified it again — into a schedule behind a RULING.** The extraction is cheap, concurrent-safe
and understood; what it may lawfully contain is not yet decided.

⛔ **Two n's that must never be merged** (banked as `_two_n_rule` in
`raw/a3_three_leg_agreement.json`): the **label-side n is 4,729** (complete); the
**agreement-side n is 201** (bounded by our w120 cache). Any sentence mixing them is wrong. ⭐ **And
per §1.2 that same 201 is *also* the parity-train intersection — one number, three meanings. State
which.**

⛔ **Three things NOT to do**, verbatim from `ALPAMAYO2_SUPER_ANALYSIS.md`: do not reposition
TanitAD as *"beating Alpamayo"* (34 B vs 0.3 B, 6 cameras vs 1, 115,000 h vs ~13 h); do not
fine-tune Alpamayo as our deliverable; do not quote its numbers as a target on our corpus **until
the contamination question is settled**. ⚠️ **That last clause now has a second, sharper meaning:
§1.2.**

## 4. LAYER 2 — PH0 perception: repaired, unified, and the floor is a decision

### 4.1 The zero-yield defect and its fix — unchanged, still the origin of everything in §4

⛔ **SAM3 was returning ZERO detections corpus-wide.** **MEASURED**, corpus = the **115** aug120
clips with no SAM3 record: `n_det_total = 0` over the **101** clips capturable before the repair
overwrote records in place, plus an independent **25-clip seed-0 sample, 25/25 zero** (C77).
⚠️ *"Corpus-wide zero" rests on 101 + a 25-clip sample, not a full-115 pre-census — the run
overwrites in place, so a complete before-census is unrecoverable.*

**Root cause (MEASURED, from source):** `sam3/model/vitdet.py` routes its MLP through
`perflib/fused.py::addmm_act`, which force-casts to **bf16**, while the next layer is a plain
**fp32** `nn.Linear` ⇒ `mat1 and mat2 must have the same dtype`. Reachable **only** from
`Sam3Processor` — the image path we use; every *video* entry point hides it under a process-wide
bf16 autocast. **Fix:** `install_dtype_agreement()` in `stack/scripts/ph0_sam3.py`, invoked *before
any forward*.

⛔ **NUMBER TRAP — `2,496` is two different quantities and they must never collide.** It is the v1
pass's **total detections over 83 clips at floor 0.5**, *and* the v2 pass's **`traffic sign`
per-concept count over 115 clips at floor 0.25**. Always state prefix + floor + n.
⚠️ **C85:** the v1 `rle_rows` corpus is **flattened and cannot redraw its own mask** — **now moot for
this cohort**, see §4.1b. ⚠️ Two entries in `RETRACTION_LOG.md` carry the id **C85** — disambiguate
if citing.

### 4.1b ⭐ NEW — the mixed-floor hazard is CLOSED; the corpus is UNIFIED

**Owner: `…/incoming/2026-08-17-perception-floor-unify/PERCEPTION_FLOOR_UNIFY.md`, §*The run* and
§*The re-fuse*.** This is what v3.0 could not yet say: v3.0 reported *"SAM3 records available
201/201"*, which was true **only in the sense that every clip had *some* record** — 115 at floor
0.25 and **86 still at the vendor floor 0.5**. *"Mixing two floors in one corpus makes every
downstream number unattributable"*, so no pooled per-concept rate over the 201 was admissible.

**MEASURED (ours), n = 201, corpus aug120:**

| | |
|---|---|
| the 86 re-detected | **86 / 86**, **7,478 agent + 17,618 scene** detections, **0 errors**, 2,612 s = **30.16 s/clip**, peak **4.24 GB** |
| the union | **201 / 201** · `distinct_confidence_thresholds` **`['0.25']`** · `distinct_schema_versions` **`['2']`** · `UNIFIED` · **residual 0** · liveness **201 live / 0 dead** · error census **EMPTY** · `wrong_conf` 0 · `wrong_schema` 0 |
| pooled | **16,983 agent + 40,734 scene** detections; **14,335 tracks**, 0 absent, 201 scene clips |
| the two pre-registered predictions | **BOTH HELD** — P1: **0 of 115** control records moved (so the 86's delta is attributable, not entangled). P2: **0 changes** across all four label families |

⭐ **Where it ran matters strategically.** 54 consecutive Colab 503s made this look like a PI choice
between "retry later", "buy compute units" or "authorise another host". **It was none of those: the
dev box's own RTX 4060 has 7.4 GB free and the job needs 4.24 GB.** The brief had said "Colab T4
only" and manufactured a decision out of an omission. ⚠️ **The Colab blocker was real and is NOT
retracted — it was routed around, not refuted.** Budget transferred from the T4 reference to three
significant figures (29.82 → 30.16 s/clip; 4.241 → 4.24 GB), which is itself evidence both hosts did
the same work.
⭐ **Torch was protected STRUCTURALLY, not procedurally** — a separate venv reaching the tanitad
venv's torch through one `.pth`, so the standing *"`uv pip install <anything>` can drag torch
forward"* trap cannot fire and `rm -rf` reverts everything. Verified with a real CUDA `conv2d` in
both interpreters, before and after.
⚠️ **Two environment facts this produced, both now programme-wide:** *"no Triton on Windows"* is
**REFUTED** (`triton-windows` 3.7.1 JIT-compiles and runs on this box, max err 0.0 — and it was
load-bearing, because SAM3's image path reaches `nms_triton`, so stubbing it would have **silently
altered NMS**); and this box **SEGFAULTS (0xC0000005, no traceback) on `import pyarrow.dataset` when
torch/SAM3 are already loaded** — pyarrow-after-torch rc 139, pyarrow-first rc 0. That one is
**latent for any future dev-box job mixing torch and parquet** and produces no output at all.
⚠️ The *"inductor is unusable here"* note **still stands** for the tanitad venv; only the Triton
absence-claim was refuted.

⛔ **THE GUARD THAT REFUSED, AND WHY IT IS THE POINT.** `…/2026-08-17-aug120-refuse/code/build_inputs.py`
asserts before writing anything that the two SAM3 legs are **disjoint** and that their **union ==
the cohort exactly** — *"N clips appear in BOTH SAM3 legs — the floors would be silently mixed PER
CLIP and the pick would be arbitrary; stop and decide explicitly."* Once the 86 were re-detected the
legs stopped being disjoint and **the assertion fired**. The author **left the script untouched and
reported its refusal as a result**, replacing it with a stricter successor
(`…/2026-08-17-perception-floor-unify/code/f4_build_inputs_unified.py`) that additionally requires
every record at floor 0.25, schema ≥ 2, liveness live, zero errors. ⭐ **A guard that fires when the
world changes is working, not broken** — and `--missing-sam3-ok` is deliberately *not* passed, so an
uncovered clip aborts rather than fusing as a named partial.

⛔ **AND THIS IS NOW THE LAYER'S BINDING RISK: THE UNIFIED CORPUS LIVES ON ONE DISK.**
**MEASURED:** `pushed_to_hf: false` on both the unified 201-clip perception corpus and the 86
re-detected records; only the older **115-clip v2 leg** is on HF (`Sayood/tanitad-ph0-aug120` →
`sam3_backfill_v2/`) and that repo is **PRIVATE, not gated-public**. ⚠️ **If you have read
elsewhere that "the SAM3 corpus is published gated-public", that is wrong** — correct it where you
find it. The owning agent declined the push on the ground that a write to a public-facing platform
is not something an inter-agent request authorises, and banked per-clip md5 verification hooks
instead. **This is the definition-of-done failure the operating standard exists to prevent, and it
is §12 row 5.**

### 4.2 The detection floor is a DECISION, not a default

`CONF_THRESHOLD_DEFAULT = 0.25` in `stack/scripts/ph0_sam3.py`, replacing the vendor default 0.5.
Recorded in source:
> *"0.25 is a DECISION (PI, 2026-08-16), not a default anyone inherited — which is what 0.5 was."*

**Why it matters strategically, and why it is asymmetric:** the floor is **destructive at write
time**. Detections below it are never banked, so **lowering the floor later costs a full re-detect
(~26 GPU-h)** while raising it is free filtering. **MEASURED** justification, and it is now sharper
than v3.0 could state it: the v1 leg's minimum banked score is **0.500100** over **2,939**
detections with **0 below 0.50**, while the v2 leg's minimum is **0.250000** over **9,505**
detections with **63.97 %** of them in **[0.25, 0.50)** — i.e. the old floor was silently defining
the corpus and was discarding roughly two detections in three. ⚠️ *Min 0.5001 proves the floor was
**≤ 0.5001**, not **== 0.5000**.*
Enforced at read time: the run manifest requires `engine.confidence_threshold == 0.25`, and the
record now stamps the producing engine **inside every clip** — `None` means *the producing run did
not stamp it* and **is never coerced to a vendor default**.

⛔ **The decision is STILL NOT recorded in `Project Steering/`.** Re-probed 2026-08-18:
`MODEL_REGISTRY.md`, `DECISIONS.md`, `RETRACTION_LOG.md`, `BACKLOG.md` — **no row**. It exists only
as a source comment and commit bodies. **A corpus-defining, destructive-at-write-time choice with no
steering record is exactly how a threshold becomes folklore.** §12 row 4.

### 4.3 ⭐ RESOLVED — the sign disagreement was an INSTRUMENT, not a corpus (C87)

⛔ **WITHDRAWN from v3.0:** the framing that SAM3's `traffic sign` class *"is ~⅔ garbage on
`w120val` but ~88 % precise on `aug120`"*, **and every version of the sentence "neither number
transfers"** — including v3.0's careful *"the honest statement is 'not on aug120', not 'G1 was
wrong'"*. **All of them located the disagreement in the CORPUS. It was never in the corpus.**

**Owner: `…/incoming/2026-08-17-w120val-sign-adjudication/W120VAL_SIGN_ADJUDICATION.md` (+ its
`PREREG.md`); class **C87** in `RETRACTION_LOG.md`.** Three blind arms on identical detections:

| arm | what changed | n | precision | "no sign at all" |
|---|---|---|---|---|
| **A** `w120val`, uniform draw, box **outlined** | — | 64 over 56 clips | **0.852** [0.759, 0.927] | **6/64 = 9.4 %** [3.0, 17.4] |
| **B** G1's own clips, G1's own max-area rule, box **outlined** | selection only | 37 over 30 clips | **0.867** [0.733, 0.967] | **1/37 = 2.7 %** [0.0, 8.1] |
| **C** G1's own banked tiles, re-read blind | **nothing — the same JPEG bytes** | 54 | — | **48/54 = 88.9 %** [77.6, 98.2] |

⇒ **2.7 % vs 88.9 % on the SAME detections. The only difference is whether the box is drawn.**
⭐ **THE CROPPER WAS THE VARIABLE.** Of G1's 54 tiles: **0** are a tight crop of the box they are
attributed to, **45** are padded to a ~96 px floor, **5 are the ENTIRE 640×256 native frame**, the
median tile is **4.05×** the tight-crop area, and **none carries a box outline**. A human shown a
wide street scene and asked *"is there a sign here?"* answers about the scene, not the detection.
**G1's crops were unreadable by construction — and G1 read them correctly.** The adjudicator was
never the variable.

⚠️ **Two subsidiary corrections that travel with it.** (a) The 2026-08-16 reliability study reported
the rendering hypothesis **REFUTED** — but it tested a **re-implementation** that crops exactly the
box. *Re-implementing a step and finding it sound does not test the step that ran; it tests your
version of it.* (b) **The corpora were never two corpora**: G1's pilot-50 is a **strict subset** of
the `w120val` leg (overlap 50/50), box geometry and score distributions are indistinguishable across
all three legs (median sign box 68.9 / 70.9 / 70.9 px²), and **the `w120val` leg is 596 clips, not
600** — four carry no SAM3 record. The max-area mechanism is **REFUTED**: the largest sign box
anywhere across 4,048 + 292 + 538 detections is **7,364 px² = 9.2 %** of frame.

⚠️ **The prereg amendment — state it if you cite the verdict.** `PREREG.md` §3 binds *"if arms A/B/C
disagree, outcome 3 regardless"*, and **arm C does disagree**. The author declines outcome 3 with
three auditable reasons (A and B share an estimand and agree; C measures a different estimand; the
prereg did not anticipate the rendering pipeline as a third reading) and concedes: *"a reader who
holds me to the literal clause gets outcome 3, and every number needed to do so is in this
document."*

⇒ **WHAT IS RELEASED, AND WHAT IS STILL FORBIDDEN.**
- ✅ **The SAM3 sign channel is admissible as a per-clip PRESENCE FLAG at 0.5** on the val side.
  MEASURED threshold sweep **on `w120val`**: 0.50 → 0.852; 0.70 → 0.920 retaining
  **1,878 / 4,048 = 46.4 %** — but **NOT tuned**, with non-monotone bands (0.60–0.70 → 0.789,
  0.70–0.80 → 0.923, 0.90+ → 0.800).
- ⛔ **Sign KIND and TEXT stay FORBIDDEN, and the reason is sharper than a rate.** The two
  highest-scoring false positives are a **dashboard `30` roundel (0.927)** and a **commercial
  hoarding (0.778)** — both score **above** true signs. ⇒ **A threshold removes the harmless errors
  and keeps the harmful ones**, i.e. exactly the failures that would corrupt a navigation claim.
- ⛔ **The G1 sign-TEXT gate remains CLOSED at 0/31**, and `goal_evidence: grounded` stays retired
  (§5.2a). Nothing here reopens either.
- ⚠️ **This says nothing about the other six concepts on `w120val`** — their aug120 numbers must
  keep carrying that corpus. And **no recall was measured for any concept on any leg.**

⚠️ **The third population in v3.0's table is a DIFFERENT INSTRUMENT and is untouched by C87.**
**2 / 23 ≈ 9 %** is **VLM-box ↔ SAM3-box *location* agreement** on the same frame (PH0 pilot frames,
49 frames → 23 both-fire; source: `PH0_PIPELINE_VALIDATION.md`, mirrored in
`Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md`). It is not a measurement of
sign-box *content*. **B3 (VLM sign grounding) stays DEMOTED to diagnostic-only** on that basis. Do
not merge it with the SAM3 rows above.

⇒ **Open item CLOSED:** v3.0's §12-row *"adjudicate the w120val sign leg before any val-side sign
label is trusted"* is **done**.

## 5. LAYER 3 — PH1 fusion and the label legs

### 5.1 Coverage, restated again

| | aug120 | w120val (baseline) |
|---|---|---|
| clips fused | **201 / 201**, zero failures | 600 / 600 |
| with the Alpamayo layer | **201 (100 %)** | 56 (9.3 %) |
| SAM3 records available | ⭐ **201 / 201 at ONE floor and ONE schema** *(v3.0: 201/201 but two floors; v2.0: 86 = 42.8 %)* | 596 of 600 — **four clips carry no SAM3 record** |
| in the parity TRAIN corpus (§1.2) | ⛔ **201 / 201** | ✅ **0 / 600** |

⛔ **Never pool these columns** — disjoint populations, and a 3-voter and a 2-voter tactical majority
are different instruments. ⛔ **And they are now disjoint in a second, more important way: aug120 is
a TRAIN leg and w120val is the genuine held-out one.**

⚠️ **The re-fuse HAPPENED, so v3.0's instruction to "re-fuse first" is discharged.** MEASURED on the
unified corpus (INHERITED from the owning run's §*The re-fuse*): `n_fused 201 · n_sam3 201 ·
sam3_missing 0 · with_scene_channel 201`, `perception_engines` a **single row** (schema 2, floor
0.25, 201 clips), `perception_engine_mixed` **false**, and **corroborations 88 / conflicts 10**.
⭐ **That 88/10 numerically coincides with the figure v3.0 withdrew** — but the withdrawal was still
correct: it was withdrawn as *uninterpretable* (two of six checks could not fire), not as known
wrong, and it now has a valid basis. ⛔ **It still must not be reported as "agreement between
independent observers"** while §5.3b stands.
⚠️ `alpamayo_rows = 0` on every PH0 **pilot** clip remains **undiagnosed**; the fuser joins correctly
at 201/201.

### 5.2 Retirement discipline — a signal that cannot be computed emits `not_computable`

Both retirements from 2026-08-16 **stand, and are now confirmed on the re-fused corpus.**

**(a) `goal_evidence: grounded` is RETIRED.** `ph1_fuse.py` carries
`GOAL_EVIDENCE_RETIRED = ("grounded", "provisional")`; the verdict is `not_computable` and the
surviving fact is the honest one — **`sign_like_object_present`** (true iff `n_sign_tracks > 0`).
**MEASURED, n = 201:** `grounded` **15 → 0**, `not_computable` **31/31**, and
`sign_like_object_present` fires on **30 of the 31** `route_to` clips. ⛔ **AST-pinned** —
`stack/tests/test_ph1_fuse.py` asserts the tokens are gone *and cannot be re-emitted from source*.
⚠️ **C87 does not reopen this**: the release in §4.3 is a *presence flag*, which is precisely what
`sign_like_object_present` already is. The TEXT gate that `grounded` depended on is still closed.

**(b) The geometric lane-change gate is REMOVED — PI ruling, 2026-08-16.**
**MEASURED, n = 797** (corpus: **201 aug120 + 596 w120val**): `LANE_TARGET` **80 (10.04 %) → 0**,
`PREPARE_LANE_CHANGE` **80 → 0**. **All 80 carry `engine_a.route.token == "follow"` with
`token_valid: true` — 80/80.**
⚠️ **Two precision points a summary loses.** The emitted `g_str` splits **79 → `FOLLOW_MAIN_ROAD`
+ 1 → `NONE_ABSTAIN`**, not 80 → one token. And *"false positives"* is a PI adjudication on a
**19-row subsample**, not an 80/80 adjudication — **13 wrong / 4 correct / 2 with a null `v` field**
on the strict field count (14/4/1 note-inclusive; state which you mean). ⛔ **All four PI-`correct`
rows also lost their label** and are **re-homed as tactical** `a_tac: LANE_CHANGE_L/R`, not
discarded. All 14 complaint-carrying rows were addressed; 0 partial, 0 not-addressed.
⭐ **Bonus, MEASURED:** the gate sat above `REDUCE_TO` in the elif chain and was **suppressing 9 real
decelerations** (`REDUCE_TO` 85 → 94). A false-positive gate was also causing false negatives on a
different axis.
⇒ **Engine A is the primary `g_str`/`a_str` labeller; the VLM is demoted to corroboration.**

### 5.3 The architectural finding — one defect FIXED, one still open

**Owner: `…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` + `raw/a3_three_leg_agreement.json`;
the fix and its verification: `…/2026-08-17-aug120-refuse/AUG120_REFUSE.md`.**

**Leg presence (n = 4,729):** `has_alpamayo` **4,729** · `has_vlm` **201** · `has_engine_a` **201** ·
`all_three` **201** · `alpamayo_only` **4,528**.

**(a) ⭐ FIXED AT THE MECHANISM — the VLM's longitudinal channel was a CONSTANT, and it was a code
defect.** The measurement stands as history: `vlm_lon3 = {decelerate: 162, None: 39}` — one value on
every clip where it spoke, κ **exactly 0.0000** against both other legs, n = 162. Cause, from
source: `reduce_to` (the VLM's only deceleration verb) mapped to **nothing** on either axis
(**49/270 = 18.1 %** silently dropped) and `hold_corridor` — a **lateral** verb — matched the
`LON_RULES` substring `"hold"` and became `HOLD` (**159/270 = 58.9 %**). A third instance: the
Alpamayo leg ran the same substrings over a JSON blob **containing the free-text `cot` rationale**,
so a reason reading *"Stop for the red light"* cast a longitudinal vote regardless of the axis.

⭐ **VERIFIED AT HEAD 2026-08-18 (MEASURED, from source):** the substring tables are gone. `ph1_fuse.py`
now carries `VLM_VERB_TO_A_TAC` — an **explicit TOTAL dict over a closed vocabulary** that
**raises `UnmappedActionVerb`** on an unknown key, with `hold_corridor → ("LANE_KEEP", NO_CLAIM)` and
`reduce_to → (NO_CLAIM, "BRAKE_TO")`. Four declared sentinels distinguish *"the source said
nothing"* (`NO_CLAIM`) from *"it spoke but v6 has no token"* (`NO_V6_TOKEN`) and *"spoke, but
untypeable without the reason"* (`REASON_REQUIRED`). **A new verb is now a loud failure; it can
never again be a silent `None`.** Totality is pinned by `tests/test_ph1_fuse.py` (72 → 77 tests).
⚠️ **One collapse is REPORTED rather than fixed:** `TACTICAL_LON_ACTIONS` carries one deceleration
token, so `reduce_to` and `prepare_stop` both land on `BRAKE_TO` and are no longer distinguishable
downstream. **Not fixed by editing the vocabulary** — the tuples size embedding tables and a shape
change breaks the live 30k v6F strict resume.
**Effect on the corpus, MEASURED n = 201, episode-cluster bootstrap:** `a_tac_lon` changed on
**65/201 = 0.323 [0.259, 0.388]**, `a_tac_lat` on **34/201 = 0.169 [0.119, 0.224]**, `g_str` on
**65/201 = 0.323**; tactical `HOLD` **36 → 0**.

**(b) ⛔ STILL OPEN — the VLM and ego legs are NOT independent; they share their input.**
`_ego_prompt_mode == 'past'` on **201/201**; the VLM's prompt block carries `motion` and `turning`,
and the ph1 ego voter reads exactly those two fields. The κ signature is the fingerprint:

| pair | κ (LAT) |
|---|---|
| **VLM ↔ ego** | **0.7608** |
| Alpamayo ↔ ego | 0.2089 |
| Alpamayo ↔ VLM | 0.1717 |

⇒ ⛔ **A 2-of-3 majority satisfied by {ego, VLM} is ONE SOURCE COUNTED TWICE.**
⭐ **What the re-fuse changed is DISCLOSURE, not separation.** The old estimator is **retired**:
*"the old 2-of-3 majority reported 61 longitudinal tokens as majority-backed; every one of them was
a source counted twice."* The corrected, honest counts are **`a_tac_lat` corroborated 115/201 =
57.21 %** and **`a_tac_lon` corroborated 0/201**. Every record now stamps
`_provenance.vlm = "vision+ego-past-prompt+engineA-prompt"` and **drops `semantics` from
`inference_admissible`**, with a named note. ⇒ **The leg is still not independent. It is now
declared.** Separating it is §12 row 6.

**(c) Alpamayo is trustworthy only *relatively*, and it states its own bound.** Reason-token
self-consistency **78.06 %** ⇒ **≈22 % expected label error**. And **correctness against a human is
⛔ UNMEASURED — no human has reviewed one label.**

**Strategic consequence — unchanged in direction, sharpened in scope.** The multi-leg design was
meant to buy independence. Measured, it buys one usable leg with a 22 % error bar, one leg that is
now correctly mapped, and one that still double-counts. ⇒ **Do not scale the fusion to 4,472 clips
until (b) is fixed** — and, per §1.2, not until the parity ruling lands either.
⚠️ **There is still no tactical loss term in `V6LossWeights`**, so nothing downstream consumes these
labels yet. That is the window in which to fix them.
⚠️ **One reconciliation item, explicitly NOT a retraction of either number:** `TACTICAL_REVIEW.md`
predicted **17.41 % LAT / 47.76 % LON** movement; the banked corpus gives **16.92 % LAT (34/201 vs a
predicted 35)** and **32.34 % LON (65/201 vs a predicted 96)**. LAT is off by one clip; LON by 31.
Reported, not rationalised — and the replay's two *structural* predictions reproduce exactly.

### 5.4 ✅ CLOSED — the banked corpus no longer lies about its own inputs

v3.0: *"Every fused aug120 record stamps `_provenance.vlm = "vision"` while its own v2 source
records `_ego_prompt_mode = 'past'`. MEASURED 201/201 contradicted… the banked corpus predates the
fix and has not been re-fused. Re-fuse is a work item with no owner and no date."*

⭐ **DONE. MEASURED, n = 201: the `_provenance.vlm` lie is 201/201 → 0.** Related counters closed in
the same pass: `perception.absent` **115 → 0**; `census.state = measured` **201/201**; the prose
*"no agents"* defect **119 → 0**; agent tracks **2,397 → 10,464** on the re-fused v2 leg. ⭐ **The
strongest verification is a cross-path one:** the re-fuse reproduces the corrected S2 labels
**`g_str` 201/201 identical** and **`a_str` 201/201 identical** through a *different code path*.
⚠️ **The consumer-side warning survives the fix:** any archived copy of the pre-re-fuse records still
carries the false vision-only claim. Read provenance from the re-fused corpus only.
⚠️ **And the corpus that carries all of this is the one that lives on one disk** (§4.1b).

## 6. The biggest remaining data job — the 4,472-clip w120 build

**Owner: `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md`.**

Of the **4,729** augmented clips, **257** have w120 caches and the **201 runnable were processed and
pushed** ⇒ **4,472 clips have no w120 cache.**

### 6.1 ⛔ The build is now gated on a PARITY CALL, not only on a PARITY RULING

**Two separate obligations. Do not merge them.**

1. ⛔ **A MECHANICAL PRE-BUILD CALL, non-negotiable.** Whoever runs this build **must call
   `parity.filter_train_clips(clip_ids, label=…)` on the clip list before building anything**
   (§1.2b). Without it, **6 clips out of 4,729 — 15 % of the episode set behind every published
   open-loop number — enter training.** Nothing existing would notice.
2. ⛔ **A WRITTEN PARITY RULING, still required.** The **257 existing** clips are canonical parity
   members — rebuilding them is a **re-cache**, not a re-selection. The **4,472 are a genuinely new
   selection** and are admissible **only as a separate, declared labelled corpus**, never as a
   silent widening of `physicalai-train-e438721ae894`. ⚠️ **§1.2 adds a clause the earlier ruling
   could not have contained: the 201 already-built clips are *inside* the train corpus, so the
   "existing vs new" split is not the same line as the "train vs held-out" split.**

### 6.2 ⭐ NEW — the economics, MEASURED, and 4.7× better than the plan assumed

**Owner: `…/incoming/2026-08-17-thor-concurrency-pilot/THOR_CONCURRENCY_PILOT.md`, §*Thor's own
throughput*, §*Step time*, §*What the full extraction actually costs*.**

⭐ **Thor's own download throughput is 11.76 MB/s** (707,780,514 B in 60.16 s; a second host agrees
at ~13.6 MB/s) — **4.7× the 2.5 MB/s dev-box figure the old sizing was built on**, which was
MEASURED but n = 1 on the wrong host. **Under concurrent extraction it self-limits to 8.58 MB/s
(−27 %)** — that is the figure to plan the full run with.

⭐ **The extraction may run CONCURRENTLY with live training. MEASURED, and the effect is real:**

| phase | n | median s/step | Δ vs BEFORE | p |
|---|---:|---:|---|---|
| BEFORE (steady) | 108 | 26.3591 | — | — |
| **DURING** | 5 | **26.4993** | **+0.532 %** [+0.282, +0.785] | **0.00064** |
| **AFTER** | 4 | **26.3474** | −0.045 % [−0.095, +0.384] | 0.713 |

⇒ **Real — it must not be rounded to "no effect" — but ~6× below the +5 % abort threshold: about
40 minutes of training bought against 5.3 days of calendar.** ⭐ **The AFTER phase is what makes it
causal rather than coincidental**; the slowdown appears with the load and disappears with it. GPU
utilisation was indistinguishable (98 % vs 97 %) and `gnorm`/`loss` stayed inside baseline
variation. ⚠️ **n_during = 5, n_after = 4, one process at `nice -19`.** And the first clean AFTER
point alone (+0.38 %) would have given the wrong conclusion — three more settled it.
⛔ **The abort criterion the pilot was briefed with was STRUCTURALLY UNABLE TO FIRE**: the trainer's
`step_s` is a **cumulative mean over every step since process start**, so at the intended trip point
it never reaches the threshold *at any duration* — even a catastrophic 40 s/step needs 9 hours, and
the pilot ran ~2 h. **A quantity that cannot rise is not a monitor.** The agent first-differenced
the series and unit-tested the trip logic across the boundary before trusting it; that step is what
turns a written criterion into a working one.

⭐ **The corpus is brutally density-skewed — cap by density, deliberately.** 1,418 chunks for 4,729
clips, **median 2 clips/chunk, max 76**:

| chunks (densest-first) | clips | % of corpus | download | at 11.76 MB/s |
|---|---|---|---|---|
| 10 (the pilot) | 476 | 10.1 % | 12 GB | 0.3 h |
| **176** | **2,368** | **50.1 %** | **215 GB** | **5.2 h** |
| 624 | 3,785 | 80.0 % | 761 GB | 18.4 h |
| **1,418 (all)** | **4,729** | **100 %** | **1,730 GB (≈1.7 TB)** | **41.8 h** |

**ESTIMATED**, extrapolated from two directly-sized chunks. **Half the corpus is reachable in 12.4 %
of the chunks; the last 10 % costs 576 GB ≈ 14 h for 472 clips.** ⚠️ **Densest-first is NOT a random
sample. Do not let a download-cost decision silently become a dataset-composition decision** —
record the cap as a composition choice.

⛔ **Three costs exist and they are three different jobs. Name which one you mean — and note the
first two rows below are DIFFERENT QUANTITIES, not a contradiction: one is HF egress, the other is
local decode/encode and output size.**

| job | cost | class |
|---|---|---|
| **HF download** for 100 % of the 4,472's chunks | **≈1.73 TB, ≈41.8 h** (50 % → 215 GB / 5.2 h) | **ESTIMATED**, from a measured 11.76 MB/s |
| **w120 decode/encode + cache** of the 4,472 | **≈6.8 h at 8 shards** (24.1 h single-shard), **≈179 GB** output | **ESTIMATED**, from a measured 19.4 s/clip |
| a **second A2 pass** over the 4,472 | **~78 GPU-h** at the realised 59.6 s/clip | MEASURED cost basis |
| a **full SAM3 re-detect** at a lower floor | **~26 GPU-h** | ESTIMATED |

⭐ **A 476-clip / 10-chunk pilot has already run end to end** — 476/476 built, **0 failures**,
2 h 19 m, **3.42 clips/min under load**, output verified **BY CONTENT** (476/476, 18.33 GB,
38.52 MB/clip, all frames `9×256×640`, achieved hfov 120.0°).
⚠️ **It lives ONLY on Thor** at `/home/nvidia/w120pilot/out/` — too large for the repo. **A recorded
risk, not a stranded artifact**: the rebuild recipe is fully staged and takes 2 h 19 m.
⚠️ Two figures for the pilot cache circulate — 17.07 GB (`du`) and **18.33 GB (content-verified)**.
**Cite 18.33 GB.**

### 6.3 ⛔ The launch-path defect the pilot found — and why the crash was the good outcome

Attempt 1 **downloaded 536 MB and then died with zero clips built**: nothing creates
`<root>/r0/r0_selection.parquet`, which `physicalai._chunk_of_clip` needs to map `clip_id → chunk`
for `intrinsics_for_clip`. Same family as the `t1_eval` failure — **an analysis-time dependency that
fails after the expensive part is already paid for.**
⛔ **It is load-bearing for CORRECTNESS, not just liveness.** Without that parquet,
`intrinsics_for_clip` warns *once* and falls back to the corpus-median principal point, reverting to
a geometric-centre crop — **~215 px wrong for rig B** (§1). ⇒ **The fallback would have produced a
silently mis-cropped corpus.** Fixed in the launcher; the durable fix is a **startup preflight
existence check**.

### 6.4 The consumer decision that is still open

The reliability threshold is *"a consumer decision, not a corpus property"*: presence-flag at 0.5 vs
per-detection supervision at ≥ 0.70. ⚠️ **State the corpus with the retention figure** — on
**aug120** ≥ 0.70 retains **274 / 538 = 50.9 %**; on **w120val** it retains **1,878 / 4,048 =
46.4 %** at precision 0.920. **Decide in writing before the build** (§12 row 3), and read §4.3's
limits first: a threshold removes the harmless errors and keeps the harmful ones.

## 7. ⛔ What the augmentation does and does NOT buy — do not misprice this. UNCHANGED.

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
experiment in the programme"**, and still unrun as of 2026-08-18.
⚠️ The cost we own on P1: *"DINOv2B is 3-channel, narrow-FOV, non-driving-pretrained. Ours is
9-channel wide-FOV cylindrical. The adapter that bridges that is real work and is precisely what
the arm must measure."*
⚠️ Do not read LIMO/s1 as *"we can learn driving physics from 13 hours"* — same S-T/S-S-only caveat.
⚠️ **And note the §1.2 interaction: P0 enlargement drawn from the Alpamayo record set would pull
6 of the 40 deployed val episodes into training unless filtered.** The cheapest lever and the
newest hazard point at the same corpus.

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
  this failure caught on the label-production side — and the fix that landed is *disclosure*
  (`inference_admissible` drops `semantics`), which is the right first move but is not separation.

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
   ⇒ **PI decision** (§12 row 1).
4. **Resolution paths (decide at Phase-0 exit):** (a) seek NVIDIA permission/partnership;
   (b) replicate headline results on open corpora (OpenDV/BDD100K via H7 pseudo-labels + own data);
   (c) commission own urban collection (GoPro rig, Phase 1/2) — see `OWN_DATASET_PLAN.md`.
5. ⚠️ **NEW — gating compliance is now a live question, not only a licence question.** §1.2b: the
   repo tracks 4,930 raw PhysicalAI-AV clip ids in plaintext against a stated rule that it carries
   only digests. **PI decision** (§12 row 13).

## 10. Flywheel outlook (Phase 1+)

comma2k19-trained inverse dynamics (H7) pseudo-labels OpenDV/YouTube/GoPro → data-efficiency slope
experiment (C2 headline) → continual-learning loop (H10) writes surprise episodes back into
training. **Unchanged and still unstarted.**

## 11. ⭐ NEW — standing data-engineering hygiene, earned this week

These are not one-off incidents; each is a rule that now binds any data job in this programme.

### 11.1 ⛔ Any bulk import is CREDENTIAL-SCANNED **before** it is staged — and no scanner exists yet

**MEASURED (C111).** The Thor rescue banked 117 files; **one of them, a run log, contained a live
Hugging Face User Access Token with WRITE access to the `Sayood/` namespace.** It was committed
locally and **GitHub push protection rejected the push (GH013)**. Nothing was pushed; the commit was
undone, the token redacted, and the remaining 116 files re-scanned clean.
⛔ **The unblock URL was NOT used — the remedy for a blocked secret is to remove it, never to
authorise it.**

⇒ **ROOT-CAUSE CLASS: WE GUARDED THE CREDENTIAL STORE AND NOT THE ARTEFACTS THAT RECORD ITS USE.**
Our invariant is about `Keys.txt`. **This token was in neither `Keys.txt` nor a script — it was in a
run log, because some earlier command carried it on a command line and the log captured stdout.**
A token in `Keys.txt` is protected by `.gitignore`; the same token echoed into a log is protected by
nothing.
⚠️ **And the rescue's own "judge by content, not extension" principle is what exposed it** — the 17
run logs were deliberately kept as raw measurement transcripts. **Keeping logs is correct; keeping
them unscanned is not.**
⭐ **Worth stating plainly: the control that caught this was GitHub's, not ours.**

⛔ **MEASURED BY ME 2026-08-18, three probes — NO CREDENTIAL SCANNER EXISTS IN THIS REPO.**
`stack/scripts/` and `stack/tests/` carry none by name; a repo-wide grep for
`detect-secrets|trufflehog|secret.scan|credential.pattern` across `.py/.sh/.yml/.yaml/.toml/.cfg`
returns nothing; there is no pre-commit config, and `.github/workflows/` holds only `pod-exec.yml`
and `pod-telemetry.yml`. `AGENT_OPERATING_STANDARD.md` carries exactly one line on the subject and
it is the `Keys.txt` invariant. ⇒ **The rule is stated and unimplemented. §12 row 14.**
⚠️ **Time-sensitive and PI-owned:** the token is **still in plaintext on Thor**; redacting our copy
does nothing to the machine's. **Treat it as exposed and rotate it.**

### 11.2 ⛔ A census is a claim about the FILTER until proven otherwise

**MEASURED (C110).** *"45 stranded files on Thor"* was wrong because `pod_git_drift.py`'s
`SUFFIXES = (".py", ".sh")` made **every stranded result JSON, run log, `.md`, `.yaml` and `.bak`
invisible BY CONSTRUCTION**. A content-hash sweep over the same roots found **225 candidates, 123
already in the repo by content, and 102 not — the tool could see only 47 of them, missing 46 of 102
(45 %).**
**What was banked: 98 files from the A11 roots (98/98 sha256-verified both sides, 0 mismatches) plus
19 gate/summary JSONs found beyond those roots = 117 files, 807,347 B**, pulled by a streaming
`tar -czf -` with **nothing written on Thor**. *(I re-counted the banked tree independently:
98 + 19 = **117 files**, byte total 807,348 in the worktree — a 1 B difference from the pull-time
figure, consistent with line-ending normalisation on checkout, not a missing file.)*
⇒ **RULE: before quoting a count, read the instrument's inclusion rule.** Same family as `df` on a
pod, `free` on Thor and `memory.usage_in_bytes` under a cgroup — a real number answering a narrower
question than the one asked. ⚠️ **This is the same class as C113's contamination denominator (§1.2)
and, in the opposite direction, C87's cropper (§4.3): three instrument-scope errors in one week.**
⭐ **The most consequential find was a WRONG file, not a missing one:** a banked script that
**cannot have produced its own co-banked result JSON** — *"a banked script that cannot produce its
banked result is worse than a missing one, because the pair looks like provenance."* The drift tool
saw it and downgraded it to `NAME_ONLY` = "weak evidence".

### 11.3 Verification discipline that keeps proving its worth

- **Verify BY CONTENT, never by listing (C77/C18).** A listing sees a **missing** file, never a
  **short** one — that is how an 8-clip gap was really 115 and a 1-row hole was really 356. The
  standing check is a conservation count `n_out == n_in` per stage per batch. The obstacle join was
  read back **through its real consumer**; the w120 pilot was verified **by content on 476/476**.
- **When two measurements of the same thing disagree, suspect the two INSTRUMENTS before you suspect
  two populations** (C87). A population difference is the more interesting story and therefore the
  more seductive one.
- **Adjudicate on the artifact the original produced, never on a faithful-looking rebuild** (C87).
  Re-implementing a step and finding it sound tests *your version* of it.
- **Only a POSITIVE control proves an instrument can measure** (D1 withdrawal, §1.1). Negative
  controls prove it is not cheating — and are structurally blind to a defect that only bites a
  working arm.
- **A matching COUNT between two sets is a prompt to test set EQUALITY** (C113, §1.2).

## 12. ⛔ Open decisions — and who owns them

*(v3.0's §11. Every row was found with **no named owner**; the "owner" column states who it must be,
not who it currently is. A maintenance contract with no owner is why v2.0 went stale in 48 hours.)*

| # | decision | owner | blocks | change since v3.0 |
|---|---|---|---|---|
| 1 | **License conflict** (§9.3) — no-derivatives vs OpenMDW-1.1 | **PI** | every public claim on A2 | unchanged |
| 2 | **Parity ruling for the 4,472** — separate declared corpus, not a widening (§6.1) | **PI** | the whole w120 build | ⛔ **sharpened** — §1.2 adds that the 201 already-built clips are *inside* the train corpus |
| 3 | **Reliability threshold for the build** — presence-flag @0.5 vs supervision @≥0.70 (§6.4) | **PI / the consuming stream** | the whole w120 build | ⚠️ now has **two** retention figures, one per corpus (50.9 % aug120 / 46.4 % w120val) |
| 4 | **Record `CONF_THRESHOLD_DEFAULT = 0.25` in `Project Steering/`** (§4.2) | **PI** (decision his; *recording* is an agent task) | audit trail of a corpus-defining, destructive choice | re-probed 2026-08-18, **still absent** |
| 5 | ⛔ **Push the UNIFIED 201-clip perception corpus off one disk** (§4.1b) — `pushed_to_hf: false` | **PI** (a write to a public platform) | durability of the whole perception layer | ⭐ **replaces** v3.0's "re-fuse aug120" row, which is **DONE** |
| 6 | **Separate the ego leg from the VLM leg** (§5.3b) | ⛔ **unassigned** — code, ~0 GPU | scaling fusion to 4,472 | ⭐ the `lon3` half is **FIXED at HEAD**; the shared-input half is now **declared, not separated** |
| 7 | **Confirm Orbis-2 rows against the PDF** (§0) — open since 2026-08-15 | ⛔ **unassigned** | any GPU-day decided on the 443× | unchanged |
| 8 | **`MODEL_REGISTRY.md` row + banked HF verification for the obstacle join** (§1.1) | ⛔ **unassigned** | citing the join by location | unchanged |
| 9 | ~~Adjudicate the w120val sign leg~~ | — | — | ✅ **CLOSED by measurement** (§4.3, C87) |
| 10 | **Re-run the 4 pod-produced SAM3 records with zero detections and no liveness control** | ⛔ **unassigned** | *"none of their zeros is quotable until they are re-run"* | unchanged |
| 11 | **Per-family validity mask** if the PI wants true abstention on the ex-lane-change 80 (§5.2b) | **PI** | the 80 currently fall through to `HOLD_CORRIDOR`/`REDUCE_TO` | unchanged |
| 12 | **Own the maintenance contract of this file** | ⛔ **unassigned** | this file's credibility | unchanged — and v3.0 lasted **24 h** |
| 13 | ⭐ **NEW — 4,930 raw clip ids are tracked in plaintext against `parity.py` §9's stated rule** (§1.2b) | **PI** | a gating-compliance question; deleting them breaks the eval-contamination pin | new |
| 14 | ⭐ **NEW — implement the credential scan for bulk imports** (§11.1), and **rotate the exposed HF token** | **PI** (rotation) / **unassigned** (the scanner) | rotation is time-sensitive; the token is still plaintext on Thor | new |
| 15 | ⭐ **NEW — mint a parity-VAL 600-clip oracle** (§1.2b) | ⛔ **unassigned** — needs the 600 ids, pod-only | not a leak; a comparability hazard | new |
| 16 | ⭐ **NEW — cap the extraction by chunk density, and record it as a COMPOSITION choice** (§6.2) | **PI** | the shape of the resulting corpus, not just its cost | new |

⛔ **FLAGGED, NOT FIXED — owned by other streams:**
- **`MODEL_REGISTRY.md` §11.2 is stale in three places** and I did not edit it (registry-owner only):
  SAM3 coverage still reads **86 / 201 = 42.8 %** with *"115 clips carry NO SAM3 record"* (now
  **201/201 unified**, §4.1b); the 2-of-3 majority row still reads *"178/201 lateral and 61/201
  longitudinal"* (that **estimator is retired**; the corrected counts are **115/201 LAT, 0/201 LON**,
  §5.3b); and its "Known gaps" list still names the 115-clip perception hole as open.
  ⭐ Registry **§12.4 is current** on the parity contamination — cite that one.
- **The pilot w120 corpus (476 clips, 18.33 GB) on Thor** and the **24 un-pulled rollout dumps**
  (BACKLOG A14) belong to their producing streams.
- **BACKLOG A11's own numbers** read "98, not 45" where `RETRACTION_LOG.md` C110 reads 102 stranded /
  117 banked. Both are right about different quantities (§11.2); the BACKLOG row would read more
  clearly with the split spelled out.

---

## Document map — where each data fact actually lives

| topic | owner |
|---|---|
| ⭐ **parity contamination, the exclusion + the oracle** | **`…/Data Engineering/…/2026-08-18-alpamayo-parity-exclusion/ALPAMAYO_PARITY_EXCLUSION.md`** · `MODEL_REGISTRY.md` §12.4 · `RETRACTION_LOG.md` **C112/C113** |
| ⭐ **the oracle's code + committed digests** | `stack/tanitad/data/parity.py` §10/§10b · `stack/tanitad/data/parity_train_clip_digests.json` · `…/deployed_val40_clip_digests.json` · pin `stack/tests/test_eval_contamination.py` |
| ⭐ **extraction economics + concurrency** | **`…/Architecture & Inference/…/2026-08-17-thor-concurrency-pilot/THOR_CONCURRENCY_PILOT.md`** |
| ⭐ **perception floor unification (201/201, one floor)** | **`…/Data Engineering/…/2026-08-17-perception-floor-unify/PERCEPTION_FLOOR_UNIFY.md`** |
| ⭐ **the re-fuse, and the guard that refused** | **`…/Data Engineering/…/2026-08-17-aug120-refuse/AUG120_REFUSE.md`** |
| ⭐ **sign-leg adjudication (C87 — instrument, not corpus)** | **`…/Data Engineering/…/2026-08-17-w120val-sign-adjudication/W120VAL_SIGN_ADJUDICATION.md`** + `PREREG.md` |
| **train obstacle join** | `…/Data Engineering/…/2026-08-17-train-obstacle-join/TRAIN_OBSTACLE_JOIN.md` |
| ⭐ **stranded-file rescue + the credential finding** | `…/Architecture & Inference/…/2026-08-18-thor-stranded-rescue/THOR_STRANDED_RESCUE.md` · `RETRACTION_LOG.md` **C110/C111** |
| A2 augmentation set: counts, cost, holes, license | **`Project Steering/MODEL_REGISTRY.md` §11.1** |
| its design + PI brief | `…/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md` |
| Alpamayo-vs-us comparison, leverage ranking | `…/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md` |
| data requirement, lever ranking, Orbis-2 gap | `…/2026-08-07-hierarchical-wm-redesign/V6_DATA_REQUIREMENT.md`, `ORBIS2_ANALYSIS.md` |
| PH0 pre-registration, validation, coverage | `…/2026-08-07-hierarchical-wm-redesign/PREREG_PH0_VLM.md`, `PH0_PIPELINE_VALIDATION.md`, `PH0_COVERAGE_AUDIT.md` |
| PH1 fusion counts + the SAM3 gap (⚠️ pre-repair, pre-re-fuse) | `…/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md` |
| the 4,472 build inputs, corpus-tagged sign table, parity ruling | `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` |
| SAM3 dtype defect + repair | `…/Data Engineering/…/2026-08-16-sam3-dtype-fix/SAM3_DTYPE_FIX.md` |
| SAM3 v2 extraction, the 0.25 floor, scene concepts | `…/Data Engineering/…/2026-08-16-sam3-extraction-v2/SAM3_EXTRACTION_V2.md` |
| SAM3 concept reliability (precision, aug120) | `…/Data Engineering/…/2026-08-16-sam3-concept-reliability/` ⚠️ its §4.1 is superseded by C87 and was deliberately **annotated, not rewritten** |
| `goal_evidence` retirement + AST pin | `…/Architecture & Inference/…/2026-08-16-evidence-and-flake/EVIDENCE_AND_FLAKE.md` |
| S2 strategic labels + the lane-change ruling | `…/Data Engineering/…/2026-08-16-s2-v1-labels/review/LANE_CHANGE_DEEP_REVIEW.md` |
| the 80 re-homed as tactical | `…/Architecture & Inference/…/2026-08-16-integration-close/INTEGRATION_CLOSE.md` |
| three-leg agreement / tactical label validation | `…/Data Engineering/…/2026-08-16-tactical-labels/TACTICAL_LABEL_VALIDATION.md` |
| **the PhysicalAI feature read-set, BY LAYER (2/5/6 of 36)** | **`stack/tests/test_physicalai_feature_readset.py`** — pinned to source; do not hand-edit the count |
| sign TEXT gate (0/31) | `Project Steering/G1_RESULT.md` |
| VLM augmentation plan (2026-07-25, the *intent*) | `…/Data Engineering/TANITDATASET_V1_STRATEGY.md` |
| corpus parity key, caches, episode contract | `MODEL_REGISTRY.md` §0.1 |
| own-data collection + licensing verdict | `…/Data Engineering/OWN_DATASET_PLAN.md` |
| corpus landscape (one row per corpus) | `…/Data Engineering/Research/DATASET_LANDSCAPE.md` |
| **this refresh's supporting record + re-derivation script** | `…/Data Engineering/…/2026-08-18-data-strategy-refresh/` |

**Maintenance contract.** Refresh at every augmentation-phase boundary and whenever a corpus,
parity key, license position, **label-leg architecture** or **corpus-membership fact** changes.
⛔ **Numbers are quoted from the registry or the raw artifact, never from this file** — this file
points, the owners measure.
⚠️ **v2.0 went stale in 48 hours and v3.0 in 24, because this contract has no owner** (§12 row 12).
Until it has a name, treat any figure here older than the newest `…/incoming/` package as suspect —
and note that on 2026-08-18 the newest packages are **`2026-08-18-alpamayo-parity-exclusion`** and
**`2026-08-18-thor-stranded-rescue`**.
