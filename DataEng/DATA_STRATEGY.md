# TanitAD Data Strategy (v2.0, 2026-08-15 — supersedes v1.0 of 2026-07-06)

> **What changed and why this rewrite happened.** v1.0 (2026-07-06, implementing D-009/D-010/D-012)
> was correct for Phase 0 and then went **a month and a half stale**: it predates Alpamayo-2 Super's
> release (2026-08-04), the entire augmentation campaign (Aug 6–15), the PH0 VLM+SAM3 line, PH1
> fusion, and the Orbis-2 data-requirement framing — yet
> `Project Steering/HANDOVER_TO_LOCAL_2026-08-15.md` §1.A still cites it as *"strategy in
> `DataEng/DATA_STRATEGY.md`"*. **A stale pointer in a handover is how the next session loses a
> stream.** v2.0 is deliberately an **INDEX with the decisions on it**: the numbers live in the
> owner documents named below and in `Project Steering/MODEL_REGISTRY.md`, never here.
>
> ⚠️ **Searched before rewriting** (rule: absence at one location is not absence): no newer owner
> document exists. The nearest candidate,
> `TanitAD Research Hub/Data Engineering/TANITDATASET_V1_STRATEGY.md`, is the **VLM-augmentation
> PLAN of 2026-07-25** — it predates the A2 run and PH0 execution and describes what we intended,
> not what was built. `DATA_LAKE_ARCHITECTURE.md` (07-13), `OWN_DATASET_PLAN.md` (07-13/17) and
> `Research/DATASET_LANDSCAPE.md` (07-18) are all narrower and older still. ⇒ this file remains
> the owner, brought current.

## 0. The one thing to read first

**We are data-limited by roughly two orders of magnitude, and compute-limited by none.**
(`…/2026-08-07-hierarchical-wm-redesign/V6_DATA_REQUIREMENT.md`, MEASURED from the running job.)

| quantity | value |
|---|---|
| training corpus | **2,400 episodes ≈ 13.3 h** |
| model | **336.5 M** params |
| ⭐ task-matched reference: **Orbis 2** (hierarchical driving WM, 1,067 M, fine-tuned on **our** corpus) | **5,890 h**, ratio **1 : 443** |
| ⭐ **hours per M-param, Orbis 2 vs v6** | **5.52 vs 0.040** ⇒ **139× under, size-normalised** |

⚠️ **Evidence stamp:** the Orbis-2 rows are **PUBLISHED-via-snippet** (`ORBIS2_ANALYSIS.md`,
*"one confidence step below PUBLISHED-exact"*) — arXiv was egress-blocked. **Confirm against the
PDF before any of them decides a GPU-day.** ⚠️ The Chinchilla row in the same doc reads itself
down: *"the constant is not transferable. What IS transferable is the order of magnitude."*

⇒ **Only two levers can close a 443× gap:** **P0** a larger corpus under a **new, declared**
parity key (*"never a silent widening"*; the eval set unchanged) or **P1** a frozen pretrained
encoder that imports someone else's hours. Everything else — train longer, saliency sampling,
near-duplicate pruning — is worth **single-digit factors**.

## 1. Corpus roles (who provides what) — carried forward from v1.0, corrected

| Corpus | Role | Actions | License position |
|---|---|---|---|
| **PhysicalAI-AV (PRIMARY RICH, D-012)** | urban diversity backbone; 1,727 h, 25 countries, 2,500+ cities | egomotion (poses → yaw-rate/accel) | **use now, resolve later**; every experiment tagged `data:physicalai` |
| comma2k19 (BOOTSTRAP + PUBLIC ANCHOR) | real-CAN action grounding; **all public open-loop numbers** until licenses resolve | real CAN (steer, speed) | MIT — clean |
| NVIDIA synthetic (SIM-DATA ARM, D-014) | pre-rendered long-tail: emergency, lane change, weather | scenario egomotion | Cosmos-Drive-Dreams CC-BY-4.0 |
| **AlpaSim / NuRec** *(new since v1.0)* | the closed-loop arm that actually works — runs bare on an A40, **not** docker compose; `volume.nurec` is gzip+msgpack, `map.xodr` answers part of the strategic-map gap | sim-exact | per NuRec scene terms |
| ~~CARLA on RunPod~~ | superseded in practice by AlpaSim for closed-loop | — | — |
| ~~MetaDrive~~ | retired per D-014 | — | — |
| nuScenes-mini | D8 OOD probes only (never trained on) | ego pose | research |
| Own GoPro / OpenDV / YouTube | Phase 1 H7 pseudo-labeling scale-up | via IDM (H7) | own / public |

⛔ **Two corpus facts that are now settled and must stop being re-asked** (five independent
probes): PhysicalAI-AV ships **no map, lane graph, junction annotation, roundabout label,
traffic-light feature or route/goal signal** — the card says verbatim *"we do not include open
maps data"* — and `egomotion` carries **no lat/lon/GNSS**, so **OSM map-matching on our traces is
impossible**. The strategic-brain topology must come from AlpaSim, an external corpus, or an
auto-labeller (§2).

⚠️ **The rig caveat stands:** PhysicalAI front-wide contains **two camera rigs** (cy ≈ 543 rig A,
cy ≈ 755 rig B). Crop around the **per-clip** principal point; a geometric-centre crop is ~215 px
wrong for rig B.

## 2. AUGMENTATION LAYER 1 — Alpamayo-2 Super (DELIVERED)

**Owner of the numbers: `Project Steering/MODEL_REGISTRY.md` §11.1.** Design:
`…/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md`. Comparison analysis:
`…/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md`.

**Delivered, MEASURED from `records.parquet` (sha256-verified against the HF far side):**
**23,644 rows / 4,729 clips / 5 tasks / 12 columns**, `seed` 42 and
`model_id nvidia/Alpamayo2-Super` on 100 % of rows, **zero errors**, **78.4 wall-hours** at
**59.6 s/clip** for the full battery — **~3.4× cheaper than the DESIGN doc's ~200 s/clip
ESTIMATE**. Published as `Sayood/tanitad-alpamayo2-augmentation`, **no raw sensor data** (rows
join back by `clip_id` + `t0_us`).

⛔ **One quantisation arm only** — `NF4-backbone-4bit-UNVALIDATED` on 100 % of rows. **There is no
bf16 arm and therefore no quantized-vs-full comparison inside the dataset.**
⚠️ **The dataset card understates the completeness hole by 356×**: it claims 4,800 clips / 23,999
rows and *"one task row missing"*; measured, **356 of 24,000 rows are missing** (81 selected clips
produced nothing, 10 delivered clips were never in the manifest, 1 clip lacks its
`grounding_via_vqa` row). The four stratified road classes (urban 1,884 · intersection_rich 1,241 ·
highway 384 · unstructured 83) are nonetheless **delivered 100 % complete** — every dropout falls in
the unlabelled remainder. Full accounting in registry §11.1.

**Why this layer exists — the strategic-supervision unlock.** Our route head degenerated to a
constant predictor (`{left 0, straight 1737, right 0}`) precisely because PhysicalAI ships no map.
**Alpamayo can manufacture the strategic supervision PhysicalAI-AV does not ship** — ranked
Tier-1 #2 in `ALPAMAYO2_SUPER_ANALYSIS.md` §8, *"the single biggest unlock available to us"*. Its
meta-actions (3-axis, with severity) and Chain-of-Causation traces are tactical-level supervision:
a second, independent teacher for stage 0, and **source #1 of the hierarchy vocabulary**
(`HIERARCHY_VOCABULARY.md` §1).

⛔ **Three things NOT to do, verbatim from the analysis:** do not reposition TanitAD as *"beating
Alpamayo"* (34 B vs 0.3 B, 6 cameras vs 1, 115,000 h vs ~13 h); do not fine-tune Alpamayo as our
deliverable; do not quote its numbers as a target on our corpus **until the contamination question
is settled**.

## 3. AUGMENTATION LAYER 2 — PH0 (VLM + SAM3) and PH1 fusion

**Owners:** `…/2026-08-07-hierarchical-wm-redesign/{PREREG_PH0_VLM.md, PH0_PIPELINE_VALIDATION.md,
PH0_COVERAGE_AUDIT.md}` · `…/Data Engineering/Implementation/incoming/2026-08-15-aug120-fusion/`.

**PH0** runs a VLM over keyframes for tactical/strategic semantics, with **SAM3** supplying
mask-level grounding — *"the closest admissible thing to the missing map"* (lane/road-surface
geometry PhysicalAI's labels never had), plus a cross-check on `obstacle.offline`. **Usage stays
labeling-side only.**

⛔ **B3 (sign-box grounding) is DEMOTED to diagnostic-only.** Measured agreement with SAM3 on the
*exact same frame* is **2 / 23 ≈ 9 %**. *(The earlier "0/8" was a confound — frames up to ~3.5 s
apart were compared.)* ⚠️ **What this does NOT show:** that the VLM's sign *classification* is
wrong — B2 and B3 are different claims. B3 is off the decision path; B1/B2/B4 continue.
⇒ **PH1 drops signage-text fields entirely** by default in the fusion gate (a G1 FAIL on all three
arms is the reason).

**PH1 fusion, MEASURED 2026-08-15 (`AUG120_FUSION_RESULT.md`), stated BESIDE, never pooled —
these are disjoint populations:**

| | aug120 (this run) | val-600 (baseline) |
|---|---|---|
| clips fused | **201 / 201**, zero failures | 600 / 600 |
| corroborations | 88 | 175 |
| conflicts | 10 | 41 |
| with the Alpamayo layer | **201 (100 %)** | 56 (**9.3 %**) |
| SAM3 records available | **86 (42.8 %)** | 596 (99.3 %) |

⚠️ **The aug120 corroboration count is a FLOOR, not a rate:** with SAM3 absent on 115 clips, two of
six corroboration checks cannot fire. ⛔ **Never pool these two columns** — a 3-voter and a 2-voter
tactical majority are different instruments.
⚠️ **The SAM3 gap was understated 14× in the stop runbook** ("batch_00184, 8 clips" → truly
**115 of 201, in every batch**); root cause: `aug120_pipeline.py` passed `--n` to the bridge and
the VLM and **omitted it for SAM3**, whose default is 4. ⇒ **A listing probe sees a MISSING file
but never a SHORT one. Count records against expected.**
⚠️ `alpamayo_rows = 0` on every PH0 pilot clip — engine D contributed nothing at pilot time,
**undiagnosed** (`PH0_COVERAGE_AUDIT.md` §5); the fuser later joins it correctly at 201/201.

## 4. THE BIGGEST REMAINING DATA JOB — the 4,472-clip w120 build

**Owner: `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md`** (five things that must be fixed
or decided **before** the first batch).

Of the **4,729** augmented clips, **257 have w120 caches**, 56 were already done, and the **201
runnable were all processed and pushed** ⇒ **4,472 clips have no w120 cache.**

⛔ **The blocker is a BUILD, not a model.** The source stores the 120° camera as chunked zips +
per-feature parquet chunks, so the path is chunk-index → `v2_compressed.py build --only-clips`
(scoped in the stop runbook, **not started**). Ranked **#4 in the open-work order** and *"the
single biggest data job; also unlocks the G1 native-res re-run."*
**Cost basis (MEASURED, not estimated):** a second A2 pass over the 4,472 is **~78 GPU-hours** at
the realised 59.6 s/clip — not the design doc's ~200 s/clip.

## 5. ⛔ What the augmentation does and does NOT buy — do not misprice this

The augmentation set is **4,729 clips ≈ 26 h**, which *"roughly doubles the corpus"* against the
13.3 h in training. Against Orbis-2's 5,890 h that moves the size-normalised deficit from
**139× → ~70×** — **still two orders short.**

⇒ **Stated plainly: PH0/PH1 augmentation is NOT a fix for the S-W data deficit.** Per
`V6_DATA_REQUIREMENT.md` §3.4, label-side curation *"applies to our S-T/S-S stages (goal
supervision on a frozen trunk), where a few thousand well-chosen labelled clips genuinely may
suffice — which is a real and encouraging result for the PH0 pipeline, and **no help at all for
S-W**."*

**The augmentation stream and the corpus-size stream answer DIFFERENT questions. Conflating them
misprices both.** The S-W levers are **P0** (new-parity-key corpus enlargement) and **P1** (frozen
pretrained encoder) — and per `ORBIS2_ANALYSIS.md` §5.1, **P1 is now "the best-supported unrun
experiment in the programme."** ⚠️ The cost we still own on P1: *"DINOv2B is 3-channel,
narrow-FOV, non-driving-pretrained. Ours is 9-channel wide-FOV cylindrical. The adapter that
bridges that is real work and is precisely what the arm must measure."*
⚠️ And do not read LIMO/s1 as *"we can learn driving physics from 13 hours"* — same S-T/S-S-only
caveat.

## 6. Training composition

⚠️ **v1.0's mix (~60 % PhysicalAI / ~25 % comma2k19 / ~15 % MetaDrive) never shipped and is
withdrawn** — MetaDrive is retired (D-014) and every trained arm to date is **PhysicalAI-AV only**,
on the canonical parity corpus `physicalai-train-e438721ae894` (2,376 episodes, skip-hash
`f09e44db`), or its declared w120 sibling. **Parity is sacred: anything that re-selects episodes
breaks cross-arm comparability and must be refused.** Corpus enlargement (P0) is therefore a
**new, declared parity key**, never a silent widening, with the evaluation set unchanged.

- Validation: held-out episodes per corpus, real-only. Public numbers **comma2k19-only** until the
  license question resolves (§7).
- **Semantic-coverage audit (standing metric):** per corpus and per mix, report the scenario-tag
  distribution. The A2 road-class labels (`aug_road_class.json`) are **ego-derived** — admissible
  as labels under the binding rule, ⛔ **never as an inference-time input.**

## 7. License management — the firewall, updated

1. **Tagging (now):** every experiment record lists its corpora (`data:` tags); LEADERBOARD rows
   carry the tag so exposure is auditable in one grep.
2. **Firewall (now):** public claims / demos / publications use **comma2k19 + own data** only.
   ⚠️ *v1.0 said "comma2k19 + MetaDrive (+ own data)"; MetaDrive is retired, so the firewall is
   narrower than it reads in v1.0.*
   ⛔ **The A2 augmentation set is INSIDE the firewall, not outside it.** It is licensed
   `nvidia-physicalai-derivative` and inherits PhysicalAI-AV's position — publishing it on HF does
   **not** make it publicly claimable. (`TANITDATASET_V1_STRATEGY.md` classes PhysicalAI/Alpamayo
   as *"commercial-OK for internal AV dev but no-derivatives → firewalled, `recipe-only`"* — that
   classification and the OpenMDW-1.1 derivative permission cited in
   `ALPAMAYO2_SUPER_ANALYSIS.md` **are not obviously the same reading, and the disagreement is
   recorded here rather than resolved by whichever document was read last.** ⇒ **PI decision.**)
3. **Resolution paths (decide at Phase-0 exit):** (a) seek NVIDIA permission/partnership;
   (b) replicate headline results on open corpora (OpenDV/BDD100K via H7 pseudo-labels + own data);
   (c) commission own urban collection (GoPro rig, Phase 1/2) — see `OWN_DATASET_PLAN.md`.

## 8. Flywheel outlook (Phase 1+)

comma2k19-trained inverse dynamics (H7) pseudo-labels OpenDV/YouTube/GoPro → data-efficiency slope
experiment (C2 headline) → continual-learning loop (H10) writes surprise episodes back into
training. **Unchanged from v1.0 and still unstarted.**

---

## Document map — where each data fact actually lives

| topic | owner |
|---|---|
| Alpamayo-2 augmentation set: counts, cost, holes, license | **`Project Steering/MODEL_REGISTRY.md` §11.1** |
| its design + PI brief | `…/Benchmarks & Eval/Research/2026-08-06-alpamayo-augmentation/DESIGN.md` |
| Alpamayo-vs-us comparison, leverage ranking | `…/Research/2026-08-05-alpamayo2-super/ALPAMAYO2_SUPER_ANALYSIS.md` |
| PH0 VLM/SAM3 pre-registration, validation, coverage | `…/2026-08-07-hierarchical-wm-redesign/PREREG_PH0_VLM.md`, `PH0_PIPELINE_VALIDATION.md`, `PH0_COVERAGE_AUDIT.md` |
| PH1 fusion counts + the SAM3 gap | `…/Data Engineering/Implementation/incoming/2026-08-15-aug120-fusion/AUG120_FUSION_RESULT.md` |
| the 4,472-clip build | `…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` |
| data requirement, lever ranking, Orbis-2 gap | `…/2026-08-07-hierarchical-wm-redesign/V6_DATA_REQUIREMENT.md`, `ORBIS2_ANALYSIS.md` |
| VLM augmentation plan (2026-07-25, the *intent*) | `…/Data Engineering/TANITDATASET_V1_STRATEGY.md` |
| corpus parity key, caches, episode contract | `MODEL_REGISTRY.md` §0.1 |
| own-data collection + licensing verdict | `…/Data Engineering/OWN_DATASET_PLAN.md` |
| corpus landscape (one row per corpus) | `…/Data Engineering/Research/DATASET_LANDSCAPE.md` |

**Maintenance contract:** refresh at every augmentation-phase boundary and whenever a corpus,
parity key or license position changes. ⛔ **Numbers are quoted from the registry or the raw
artifact, never from this file** — this file points, the owners measure.
