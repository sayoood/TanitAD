<title>Posted speed-limit supplier — map.xodr does not cover the corpus</title>

# Posted speed limit: `map.xodr` is not the supplier — it covers 16 of 4,572 train clips, and its numbers are mislabelled

**2026-09-13 · Research Lab (LAB-RUN-012) · Data Engineering · serves the Master Mind's STANDING RESEARCH QUESTION (injected 2026-09-10)**
⛔ **Tier: none** — a supplier census and literature survey; no model runs. ⛔ **Nothing here derives a ceiling from ego dynamics** (the refuted path, R² 0.9702).

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⛔⛔ **The premise of the standing question is wrong for TRAINING.** `map.xodr` ships with NuRec scenes, and NuRec 26.04 is **1,607 PhysicalAI clips** — **0.52 %** of the 306,152-clip corpus. **Intersected with our clip sets: 16 / 4,572 v7.2-train (0.35 %), 1 / 147 v7.2-eval, 17 / 4,729 B1 Alpamayo ids.** ⇒ as a training supplier it is **sparser than the CoT sign readings it was meant to replace (31 / 4,572)**. "Full coverage" is true of *AlpaSim scenes*, not of *our corpus*. | MEASURED `raw/nurec_corpus_overlap.json` (same-breath controls: 1,607/1,607, 4,572/4,572, 147/147 ids UUID-shaped; non-zero overlap proves one namespace) |
| **F2** | ⛔ **And the xodr number cannot be read as written.** Our own 2026-08-02 join (138 roads, 0 exceptions) showed the exporter writes **km/h labelled `unit="mph"`**, and on the scene we hold **293 of 299 ego samples snap to 40 km/h roads while the ego drives 60–73 km/h**. | INHERITED `…/Architecture & Inference/Research/2026-08-02-nurec-xodr-map/XODR_MAP.md` §6 (MEASURED there) |
| **F3** | ⭐ **New structural fact: the limit is a per-ROAD attribute, not a sign-anchored one.** Every one of 219 roads carries its speed on `<type>` (town 40/50, motorway 70), and **219/219** lane-level `<speed>` records merely copy it (0 disagreements). Vendor `DeepMap, Inc.`, `autolabels:v0`. A per-road autolabel explains F2's value error without needing a sign. | MEASURED `raw/xodr_speed_census.json` (one scene, n = 1 — scope stated) |
| **F4** | ⭐ **Where the xodr IS the right supplier: closed-loop evaluation.** For the ≤ 1,607 AlpaSim scenes it gives a zero-ego-provenance limit at full coverage — once the unit is corrected and the value is sanity-checked per scene. ⚠️ **And the 16 train-overlap clips are now a closed-loop LEAK LIST** — any AlpaSim number on those scenes is on training clips. | MEASURED (F1) + HYPOTHESIS (per-scene validity) |
| **F5** | ⛔ **Correction to our own record.** `CLOSED_LOOP_FEASIBILITY_2026-09-03.md` §9.2 says the NuRec and Alpamayo *"id namespaces are simply different"*. **They are not** — 17 ids intersect exactly. The check it called uninformative is informative. | MEASURED |
| **F6** | **External map-sourced limits exist, in three corpora, none of them ours:** **Waymo Open Motion** `LaneCenter.speed_limit_mph` (per lane); **nuPlan** lane polylines with a speed-limit attribute (8 cameras on the sensor subset; four cities); **NuRec** xodr (F1–F3). ⛔ **Argoverse 2 has NO speed field** — its `LaneSegment` lists 12 attributes, none a limit. | PUBLISHED-CODE (WOMD `map.proto`); PUBLISHED lib `2403.04133` abstract-only (nuPlan); PUBLISHED-CODE (AV2 user guide) |
| **F7** | **OSM `maxspeed` is not a route for PhysicalAI** (no GNSS — standing, unaffected) and weak even where matchable: **~12 % of OSM roads carry maxspeed globally**, 84 % in Sweden. | RELAYED (OSM wiki / NLnet project page, not re-verified) |
| **F8** | ⚠️ **The vision-derivable route has instruments, but the best published fine-grained sign number is weak.** TS-1M: **>1 M images, 454 classes** (abstract-only). MVV full text: on Mapillary **signs** (task T3) **DINOv2 0.484 vs InternVL-3-9B 0.467** accuracy — *"slightly higher"*, **no interval, no seeds** — while the abstract says DINOv2 *"consistently outperforms all VLM baselines"*. ⛔ **And MVV's fine categories are sign TYPES ("Speed Sign"), not speed VALUES** — nobody in this pass measured *reading the number*. ⇒ the concession is in the body, not the abstract: **fine-grained sign recognition from driving crops is ~50 % accurate for the best model reported.** | PUBLISHED lib `2508.02047` **FULL TEXT READ** (v1 2025-08-04, Table 3) · lib `2603.23034` abstract-only |

## 1 · The supplier table, updated (extends the 2026-09-10 census rows 5–6)

| supplier | layer | provenance at inference | train coverage | eval coverage | verdict |
|---|---|---|---|---|---|
| VLM sign readings in CoT (09-10 row 5) | augmented v7.2 | vision (camera sign) | **31 / 4,572** | **0 / 147** | ✅ admissible, too sparse |
| ⭐ **`map.xodr` (NuRec)** | external, per-clip | map autolabel, zero ego | **16 / 4,572** | **1 / 147** | ⚠️ admissible **for AlpaSim eval only**; unit bug + value error |
| dedicated sign detector over our frames | new model | vision | **UNMEASURED** — lower bound 69 CoT *mentions* | UNMEASURED | ⭐ cheapest open lever (§3) |
| road-context → limit regressor trained on nuPlan/WOMD labels | new model, external labels | vision | 4,572 / 4,572 by construction | 147 / 147 | ⛔ **PI question** (§2 risk b) |
| OSM `maxspeed` | external | map | 0 (no GNSS) | 0 | ⛔ no supplier |

## 2 · Five-dimension analysis (per item)

**(i) NuRec `map.xodr` — PUBLISHED-RELEASE-NOTE (HF card 26.04) + MEASURED census**

| dim | |
|---|---|
| RELEVANCE ⭐⭐⭐ | The Master Mind's standing question; the LONGITUDINAL family (88.7 % of the oracle gap). |
| CONSEQUENCE | Retires "xodr gives full coverage" as a *training* plan. Re-scopes it to AlpaSim evaluation. Adds a leak list. |
| COMBINATION | Pairs with 09-10's finding that the admissible CoT supplier is 147× too sparse: **both admissible suppliers are sub-1 % on train.** Any train-time max-speed input needs a *model* supplier, not a *label* supplier. |
| CHANCES / RISKS | Upside: a clean eval-time input for closed loop. Risks: autolabel values wrong (F2), unit mislabel (F2), 12-month licence expiry (card), per-road not per-sign (F3). |
| EXPERIMENT | **`E-DE-XODR-1` (0 GPU, pod-free):** for every locally held NuRec scene, snap ego to lanes and compare posted limit (km/h-corrected) with ego speed percentiles. **Committed:** limit < ego p50 on > 20 % of drive-time ⇒ xodr values are not trustworthy as a limit input even for eval; ≤ 5 % ⇒ admissible for AlpaSim with the unit fix. Blocked: only **2** renderable scenes are held locally. |

**(ii) Sign-recognition benchmarks — `2508.02047` MVV (full text), `2603.23034` TS-1M (abstract-only)**

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | The only route that is vision-only, admissible, and runs over *our* frames. |
| CONSEQUENCE | Moves the question from "which corpus has a limit" to "how often is a sign visible in a 20 s PhysicalAI clip" — a measurable rate we do not have. |
| COMBINATION | MVV's T1–T4 show a training-free DINOv2 exemplar feature-matcher at or above 1–14 B VLMs, and **3–4 days vs hours** to label 2,000 crops — the same *representation-over-size* pattern as today's Deployment and A&I findings. A frozen DINO-class reader is compatible with the vision-only rule. ⚠️ But at **0.484** on sign types, a reader would need its own validation before it supplies anything. |
| CHANCES / RISKS | Upside: coverage could be ≫ 31 because the CoT only mentions signs the VLM judged salient. Risks: **a sign is visible for seconds; the limit binds for kilometres** — a 20 s clip may simply not contain one; persistence across clips is impossible (no drive id — row 39 (e)); **value reading (50 vs 60) is an unmeasured task** — MVV stops at the type. |
| EXPERIMENT | **`E-DE-SIGN-1` (4060, after the live MM arm exits):** run an off-the-shelf speed-sign detector+classifier over the **147 eval + a 500-clip train sample** at 2 Hz. **Committed:** clip-level sign-present rate ≥ 20 % ⇒ build the reader as the supplier; 2–20 % ⇒ supplier for a *sparse, persistent* input with an explicit "unknown" token; < 2 % ⇒ **no in-corpus vision supplier exists** and the input channel is eval-only (AlpaSim). Controls: a no-sign crop set (must read ≈ 0 %) and the 31 CoT-read clips (must read ≥ 80 %). |

**(iii) External map-labelled corpora — nuPlan (`2403.04133`), WOMD (`map.proto`), AV2 (user guide)**

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | The route to a *dense* model supplier: learn "posted limit from scene appearance" where labels exist, apply to our frames. |
| CONSEQUENCE | Would give 100 % coverage by construction — and would change parity only if it re-selected episodes (it does not; it adds a column). |
| COMBINATION | ⛔ Collides with the **PI's 2026-08-03 goal-input ruling**: a road-context→limit model reads the same scene cues as the situation classifier. It is *not* that classifier's output, so it is not excluded by the letter of the rule — but the attribution argument behind the rule applies. |
| CHANCES / RISKS | Upside: dense, vision-only, zero ego. Risks: (a) domain shift — nuPlan is four cities, PhysicalAI 25 countries; (b) the PI admissibility question above; (c) an implicit limit is a *prior*, not a posted fact. |
| EXPERIMENT | Deferred behind `E-DE-SIGN-1` and the PI decision — see PI-queue item. |

## 3 · What this changes (≤3)

1. ⛔⭐⭐ **Re-scope the standing question:** `map.xodr` = **AlpaSim eval-time supplier**, not a training supplier. The training supplier question is now **`E-DE-SIGN-1`** (sign-present rate over our frames).
2. ⛔ **Adopt the 16 train-overlap NuRec clip ids as a closed-loop exclusion list** (`raw/nurec_corpus_overlap.json` → `overlap_ids`) and correct `CLOSED_LOOP_FEASIBILITY_2026-09-03.md` §9.2.
3. **Every xodr speed read goes through a unit correction (×1.609344 reverse) and a per-scene ego-percentile sanity check** before it enters any input or metric.

## 4 · Stopping condition (Rule Zero)

Done-condition is (3b) — **the remaining levers are blocked, and named:** `E-DE-SIGN-1` needs the 4060 (held by the live MM BEV-head arm) and a sign-detector weight download; `E-DE-XODR-1` needs more NuRec scenes than the 2 held locally (a pull from the gated HF repo — PI/MM decision on disk); the dense model-supplier route needs a **PI admissibility ruling**.

## 5 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Data Engineering/Research/2026-09-13-posted-speed-limit-supplier/RESULT.md` |
| Census code | `repo:…/code/xodr_speed_census.py`, `repo:…/code/nurec_corpus_overlap.py` |
| Raw | `repo:…/raw/xodr_speed_census.json`, `repo:…/raw/nurec_corpus_overlap.json`, `repo:…/raw/search_log.md` |
| NuRec `clip_ratings_26.04.csv` (88,411 B, sha256 `8075c222…`) | ⚠️ **session scratchpad only** — licence-gated NVIDIA file, deliberately **not** copied into the repo; the derived counts and overlap ids are |
