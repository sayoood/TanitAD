<title>Speed-sign reader selection — our splits are 94 % Vienna-convention, a third at night</title>

# Posted-limit supplier, part 2: the reader must be EUROPEAN and NIGHT-capable, and the pre-download choice is now narrow

**2026-09-15 · Research Lab (LAB-RUN-013) · Data Engineering · serves the Master Mind's STANDING RESEARCH QUESTION (max-speed supplier) + prepares `DE13-1` / `E-DE-SIGN-1`**
⛔ **Tier: none.** A corpus census plus a literature selection; no model runs, no weights downloaded. ⛔ **Nothing here derives a limit from ego dynamics** (the refuted path).

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **Our training split is NOT the corpus's geography.** The corpus is **50.7 % United States** (155,360 / 306,152), but **v7.2-train is 93.7 % Vienna-convention Europe (4,283 / 4,572) and 6.3 % US (289)**. **v7.2-eval is 91.2 % Europe (134 / 147) and 8.8 % US (13).** The split is near-uniform over 25 countries (US 289, DE 287, FR 281, DK 281, IT 278, SE 278 …). ⇒ **A speed-sign reader for our clips is a red-circle km/h reader first.** A US-trained (LISA / MUTCD "SPEED LIMIT NN" mph) reader would miss > 90 % of our frames by construction. | MEASURED `raw/sign_system_census.json` (4,572/4,572 and 147/147 ids matched to `data_collection.parquet`) |
| **F2** | ⭐⭐ **About a third of our clips are at night: 34.4 % of train and 34.7 % of eval** (hour < 6 or ≥ 20, local collection hour). | MEASURED, same file. ⚠️ `hour_of_day` is the dataset's own field; "night" by clock hour ignores season and latitude, so this is a proxy for darkness |
| **F3** | ⭐⭐ **Night breaks day-trained sign readers, and the published fix is data, not model size.** INTSD (full text): on an 8-class shared taxonomy with **size-matched** training sets, a classifier trained on daytime data reads **71.63 ± 0.77 %** on night crops; day + night reads **92.50 ± 1.30 %**. A YOLO-TS detector trained on day data reads **mAP@50 5.31 ± 1.02** at night. *"not recoverable from daytime imagery alone."* ⚠️ India, 41 classes, and **"speed" is ONE class with no value**. | PUBLISHED lib `2511.17183` v3 Table 4 (⚠️ column alignment taken from pypdf text extraction, not a rendered table) |
| **F4** | ⚠️ **The one global detection set that carries both sign systems is research-only, and Europe is a sampling target, not a majority.** MTSD: **52,453 fully annotated images, 257,543 boxes, 400 classes**, sampled to *"20 % North America, 20 % Europe, 20 % Asia …"*. Its derived data are *"available for academic research and approved applications"*. | PUBLISHED lib `1909.04422` (full text) |
| **F5** | ⭐ **A European, 3D-mapped sign ground truth exists for validation.** KITScenes Multimodal: Lanelet2 HD maps with **220 German StVO sign classes (120 observed)** mapped in 3D. ⚠️ **Whether speed VALUES are encoded is UNVERIFIED** (not in the summary read; StVO 274 carries the value as a sub-code). It covers 5.7 h in Karlsruhe, Frankfurt and Sindelfingen only. | PUBLISHED lib `2606.02956` (abstract + HF paper page; full text banked, not read) |
| **F6** | ⚠️ **ETSD merges six Vienna-convention countries (BE, HR, FR, DE, NL, SE): > 80,000 crops, 164 classes.** It covers 6 of our 24 European countries. Classification only (crops, no detection). | RELAYED (IEEE abstract via search, not banked) |

## 1 · The reader decision table (feeds the download request `E-DE-SIGN-1` needs)

| candidate training source | sign system | value classes | night | licence (as found) | covers our split? | verdict |
|---|---|---|---|---|---|---|
| **GTSRB / GTSDB** | DE (Vienna) | 8 speed values (20–120) — ⚠️ from general knowledge, not re-read today | ✗ daytime (INTSD Table 1 lists GTSDB as non-night) | UNVERIFIED | ⚠️ one country; value set fits most of EU | ⭐ cheapest value reader for a **pilot** |
| **ETSD** | 6 EU countries | inside 164 classes | ✗ | UNVERIFIED | ⚠️ 6 / 24 EU | candidate; licence unread |
| **MTSD** | global, both | *maximum-speed-limit* sub-classes (count UNVERIFIED here) | partial | academic / approved use | ✅ both systems | ⭐ best **coverage**; research-only |
| **LISA** | US MUTCD | speedLimit classes | ✗ | academic | ⛔ 6.3 % of train | ⛔ wrong system for us |
| **INTSD** | India | "speed" (no value) | ✅ | public | ⛔ | ⭐ only as **night augmentation** evidence |
| **KITScenes maps** | DE (3D, StVO) | UNVERIFIED | seasons | UNVERIFIED | validation only | ⭐ **validation GT**, not training |

## 2 · Five-dimension analysis

**(i) The census (F1, F2) — MEASURED**

| dim | |
|---|---|
| RELEVANCE ⭐⭐⭐ | The standing MM question. It decides which reader `E-DE-SIGN-1` should download. |
| CONSEQUENCE | Rewrites `E-DE-SIGN-1`'s design before it spends anything: the detector must be **European-first**, and its sign-present rate must be reported **separately for day and night**, or a night miss reads as "no sign exists". |
| COMBINATION | The 09-13 package's 31 CoT sign readings were never stratified by country. If they skew US, the "≥ 80 % on CoT-read clips" control is testing the wrong system. Pairs with row 39(e): no drive id, so a sign cannot persist across clips; a night miss is therefore unrecoverable too. |
| CHANCES / RISKS | Upside: a European reader has one value vocabulary (km/h round numbers) and a regular shape (red circle). Risk: 24 countries with variant fonts and supplementary plates (zones, time-limited, wet-road); US 6 % would stay unread. |
| EXPERIMENT | Folded into the revised `E-DE-SIGN-1` in §3. |

**(ii) Night sign recognition — INTSD `2511.17183` (full text)**

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | 34 % of our clips (F2). |
| CONSEQUENCE | A day-trained reader's night sign-present rate is **not evidence of absence**. A 5.31 mAP@50 day-trained detector would report "no signs" for most of a third of our data. |
| COMBINATION | Same asymmetry as our own **"absence needs two probes"** rule, now applied to a model: a reader that cannot see at night is one probe, and the absence it reports is a probe artefact. |
| CHANCES / RISKS | Chance: INTSD shows the gap is closable with size-matched night data. Risk: India, no speed values; transfer to European retroreflective signs under headlights is unmeasured. |
| EXPERIMENT | **Day/night stratified control inside `E-DE-SIGN-1`** (below). |

**(iii) Training and validation sources — MTSD `1909.04422`, KITScenes `2606.02956`, ETSD (RELAYED)**

| dim | |
|---|---|
| RELEVANCE ⭐⭐ | They decide the download the MM/PI must approve. |
| CONSEQUENCE | ⛔ **MTSD's "academic research and approved applications" terms are compatible with Lab research use and INCOMPATIBLE with anything that ships.** Same shape as the Alpamayo 1.5 licence split (register N-3). A reader trained on it may supply a *research* input only. |
| COMBINATION | KITScenes gives a 3D-mapped German sign GT. If its Lanelet2 regulatory elements carry values, it is the first **map-sourced European** check on a vision reader: exactly the independent reference the 09-13 package lacked (the xodr values were wrong on the one scene held). |
| CHANCES / RISKS | Risks: German-only validation; value field unverified; ETSD licence unread. |
| EXPERIMENT | **`E-DE-KITS-1` (0 GPU):** read KITScenes' map schema (full text + devkit) and count speed-limit regulatory elements carrying a numeric value. **Committed:** ≥ 1 valued element per km of the 162 km ⇒ adopt as reader validation GT; no value field ⇒ the line closes and we say so. |

## 3 · What this changes (≤3)

1. ⭐⭐⭐ **Revise `E-DE-SIGN-1` before approval:** the reader is **Vienna-convention first** (GTSRB-value pilot, MTSD for coverage); report the sign-present rate **stratified by country system × day/night**. Add a **night negative-control check**: if the night rate is < ¼ of the day rate on the same countries, the night half is VOID (a reader failure, not an absence). The committed 20 % / 2 % branches then apply to the **day** stratum only.
2. ⛔ **Licence gate on the reader:** anything trained on MTSD is research-input-only. Record it next to N-3 so a later shipping decision cannot inherit it silently.
3. ⭐ **Stratify the 31 CoT sign readings by country system** (0 GPU, a join with `data_collection.parquet`) before they serve as `E-DE-SIGN-1`'s positive control.

## 4 · Stopping condition (Rule Zero)

(3b) — the remaining levers are named and blocked on decisions: `E-DE-SIGN-1` needs a **weight/dataset download approval** (GTSRB-class pilot weights or MTSD; filenames and sizes to be stated in the request) and the 4060, which is free today; the dense model-supplier route still needs the PI ruling **DE13-4**. `E-DE-KITS-1` is 0 GPU and unblocked.

## 5 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Data Engineering/Research/2026-09-15-speed-sign-reader-selection/RESULT.md` |
| Census code | `repo:…/code/sign_system_census.py` |
| Raw | `repo:…/raw/sign_system_census.json`, `repo:…/raw/search_log.md` |
| Primaries banked today | lib `1909.04422` (MTSD), `2511.17183` (INTSD), `2606.02956` (KITScenes) |
| Input metadata (not copied: NVIDIA dataset file) | `devbox:C:/Users/Admin/tanitad-data/physicalai/metadata/data_collection.parquet` (306,152 rows) |
