<title>KITScenes as speed-limit validation GT — the line CLOSES, and the licence closes it twice</title>

# `E-DE-KITS-1`: no speed-limit VALUE field is documented at three independent probes, and CC BY-NC 4.0 closes the line independently of that question

**2026-09-17 · Research Lab (LAB-RUN-014) · Data Engineering · serves the Master Mind's STANDING RESEARCH QUESTION (max-speed supplier), part 3; executes the 09-15 package's pre-registered `E-DE-KITS-1`**
⛔ **Tier: none.** A licence-and-schema read; nothing downloaded, no weights, no model run. ⛔ **Nothing here derives a limit from ego dynamics** (the refuted path).

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⛔⭐⭐ **The pre-committed branch fires: no speed-limit VALUE field is documented, at three independent probes.** The paper states traffic signs are *"classified based on 220 German road traffic code classes (with 120 observed)"* — **class labels, not values or sub-codes**. The devkit README and the HF card both describe the same taxonomy as *"120 traffic-sign classes (**GTSIGN-220** taxonomy)"* and neither documents a value, `maxspeed`, or a Lanelet2 `SpeedLimit` regulatory element. `E-DE-KITS-1` committed: *"no value field ⇒ the line closes and we say so."* **We say so.** | PUBLISHED lib `2606.02956` + PUBLISHED-REPO (devkit README) + PUBLISHED-CARD (HF), all retrieved 2026-09-17 |
| **F2** | ⛔⭐⭐⭐ **And the licence closes it a second time, independently of F1: KITScenes-Multimodal is CC BY-NC 4.0** with additional dataset terms, **gated** on HuggingFace. The devkit code is Apache-2.0; **the data is non-commercial.** ⇒ even if the value field existed, it could supply a **research** validation reference only — and never anything that ships. | PUBLISHED-CARD + PUBLISHED-REPO |
| **F3** | ⚠️ **The cost was not priced in the 09-15 plan and it is decisive on its own: 4.58 TB.** For 5.7 h / 162 km / 1,007 scenarios across three German cities. As a *validation* GT for a speed-sign reader — the only role proposed for it — that is an unfavourable ratio, and it must be stated next to the adoption question rather than discovered after an approval. | PUBLISHED-CARD |
| **F4** | ⭐ **What KITScenes genuinely is, recorded so the refusal is not mistaken for a dismissal:** *"the most complete HD maps of any public autonomous driving dataset"*, Lanelet2 over **62 km²**, 29 road-feature classes, signs and lights **assigned to the lanes they govern via topological links**, annotated as reprojection-accurate 3D shapes with orientation, and **validated in closed-loop trials with the Autoware stack**. That is a real strength; it is simply not the thing our standing question needs. | PUBLISHED-CARD / lib `2606.02956` |
| **F5** | ⛔⭐⭐ **A re-find that saves a line from being re-proposed (V-1: a re-find is a free finding). The OSM route is CLOSED for our corpus, and it was already settled.** Today's scan surfaced `maxspeed`, `maxspeed:type`, `source:maxspeed` and `osm-legal-default-speeds` (a library inferring *default legal* limits from OSM tags) — a genuinely good supplier **for traces that carry coordinates**. ⛔ `CLAUDE.md` already records that PhysicalAI `egomotion` carries **no lat/lon/GNSS** — coordinates are clip-local metres — so **OSM map-matching on our traces is impossible**. ⇒ OSM is not a candidate and is not proposed. | INHERITED (`CLAUDE.md`, settled at 4 probes) + PUBLISHED-WIKI (today) |

| **F6** | ⭐⭐⭐ **MEASURED, and it is the decisive number of this package: the CoT sign-mention set is 69 clips on TRAIN and EXACTLY ZERO on EVAL — and the zero is REAL, not a broken read.** `cot_tokens.speed_limit` is true on **69 / 4,572** train records and **0 / 147** eval records. ⛔ A zero is a claim about the probe until a same-breath control reads non-zero: the **traffic-light** flag in the same field, on the same pass over the same file, reads **26 / 147 on eval**. ⇒ the tokeniser is reading eval records; there simply is no speed-limit mention in any of them. | MEASURED `raw/cot_sign_stratify.json` (record counts asserted: 4,572 ✅ / 147 ✅) |
| **F7** | ⭐⭐ **The train stratification fires `USABLE-CONTROL` on its pre-committed bar — but only for a split we cannot score on.** Of the 69: **66 Vienna-convention, 3 US** (bar: ≥ 20 Vienna ✅); **33 night / 36 day** (bar: ≥ 5 night ✅) — night is **47.8 %** of the sign mentions against 34.4 % of the corpus, so signs are *over*-represented at night, not under. Road class: urban 40, highway 25, intersection 4. Spread over 20 countries (Portugal 11, France 9, Italy 6, Denmark 6). ⭐ **Independent cross-check:** every one of the 69 countries agrees with `data_collection.parquet` — **69 agree / 0 disagree / 0 missing** — so the label's own `strata.country` is not self-reported drift. | MEASURED, same file |
| **F8** | ⚠️ **Scope correction to our own prior number, found by measuring it: 69 is the MENTION count, not the 31 "readings".** `cot_tokens.speed_limit` is a **boolean**; it carries no value. 09-13 recorded both figures — *"31 / 4,572"* as readings and *"lower bound 69 CoT mentions"* — and today's 69 reproduces the **mention** line exactly. ⇒ today's stratification describes the **69 mentions**; how many carry an extractable *number* is a different question this field cannot answer, and the 31 was not re-derived. | MEASURED + INHERITED (09-13) |

**Verdict: `E-DE-KITS-1` CLOSED on its pre-registered branch, for two independent reasons (F1 value field, F2 licence). `E-DE-SIGN-2` RAN and fires `USABLE-CONTROL` on train — and returns a MEASURED ZERO on eval (F6).**

---

## 1 · The supplier table, updated (supersedes nothing; adds two rows and closes one)

| candidate | posted-limit VALUE? | licence | covers our split? | verdict |
|---|---|---|---|---|
| **KITScenes maps** | ⛔ **undocumented at 3 probes** | **CC BY-NC 4.0**, gated | DE only, 162 km | ⛔ **CLOSED** (F1 + F2) |
| **OSM `maxspeed`** | ✅ yes, with `maxspeed:type` provenance | ODbL | ⛔ **unusable — our traces have no coordinates** | ⛔ **CLOSED** (F5, pre-settled) |
| **AlpaSim / NuRec `map.xodr`** | ⚠️ carries a limit field; **values were WRONG on the one scene held** (09-13) | NVIDIA terms | scenes only, not the train corpus | ⚠️ open, unresolved |
| **GTSRB-class value reader** | ✅ 8 speed values | UNVERIFIED | ⭐ Vienna-convention, fits 93.7 % of train | ⭐ pilot candidate (09-15) |
| **MTSD** | ⚠️ max-speed sub-classes, count unverified | research / approved use only | ✅ both systems | ⭐ coverage candidate, **research-input only** |
| **VLM CoT sign readings** | ✅ real posted signs | ours | ⛔ **31 / 4,572 train, 0 / 147 eval** | positive control only |

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐ | The MM's standing question (the max-speed input channel is blocked on a **supplier**, not on code). It decides whether `E-DE-SIGN-1` has an independent validation reference, which is what the 09-13 package lacked when the xodr values proved wrong. |
| **CONSEQUENCE** | ⛔ **`E-DE-SIGN-1` has no map-sourced European validation GT.** Its reader must therefore be validated against the only in-corpus reference we hold — the **31 CoT sign readings** — which 09-15 already flagged as needing country stratification, and which covers **0 of 147 eval clips**. ⇒ a reader could be built and could not be *validated on the eval split at all*. That is a harder blocker than the one the 09-15 plan recorded, and it should be stated in the download request rather than discovered after approval. |
| **COMBINATION** | ⭐ **F2 is the third instance of one pattern in this thread: every candidate supplier is licence-restricted in a way that splits research use from shipping** — MTSD *"academic research and approved applications"* (09-15), Alpamayo 1.5's split (register N-3), and now KITScenes CC BY-NC. ⭐ **And F2 is the exact mirror of today's frontier finding that NVIDIA moved the whole Alpamayo line to OpenMDW-1.1 (permissive, commercial)** — the *opponent's* supply got less restricted this month while our candidate suppliers stayed non-commercial. ⚠️ F1's failure shape is also familiar: a taxonomy named *"220 classes"* invites the reading *"220 sign types including each speed value"*, which is the `w120` identifier-token trap — **a count that encodes a class list, not a value space**. |
| **CHANCES / RISKS** | **Upside of closing:** two named routes are off the list for stated reasons, so neither returns as a re-search (row 39 discipline). **Risks / honest limits:** (a) ⚠️ **all three probes are DOCUMENTATION, not the map data** — a `SpeedLimit` regulatory element could exist and be undocumented, and Lanelet2 *natively supports one*; settling that needs a gated 4.58 TB download, which F2 makes not worth requesting; (b) the GTSIGN-220 class list itself was not obtained, so *"no sub-codes"* rests on three descriptions agreeing, not on the enumeration. |
| **EXPERIMENT** | **`E-DE-SIGN-2` — pre-committed above, then RUN in this same pass (0 GPU, seconds).** Committed bar: *"≥ 20 Vienna-convention AND ≥ 5 night ⇒ usable as `E-DE-SIGN-1`'s positive control; skewed US or day-only ⇒ the control tests the wrong sign system and `E-DE-SIGN-1` has no validation reference."* **Result: `USABLE-CONTROL` on train (66 Vienna, 33 night) — and F6's measured 0 / 147 on eval.** ⇒ the next experiment is now **`E-DE-SIGN-3` (0 GPU):** re-derive the **31 valued** readings from the CoT text (the boolean does not carry a value, F8) and stratify *those*. **Committed in advance:** if ≥ 20 of the valued readings are Vienna **and** the value parses to a legal km/h step, the positive control for `E-DE-SIGN-1` is fully specified; if the valued subset is < 10, the control is too thin to validate a reader and `E-DE-SIGN-1` must carry an explicit "no validation on eval, weak validation on train" clause into its approval request. |

## 3 · What this changes (≤3)

1. ⛔ **Close KITScenes as a speed-limit validation GT**, for the value-field reason **and** the CC BY-NC reason, and record both so the line is not re-opened by the first reason alone.
2. ⛔⭐⭐ **Add to `E-DE-SIGN-1`'s request the MEASURED fact that it has NO validation reference on the eval split at all** — **0 / 147**, with a same-breath traffic-light control reading **26 / 147**, so the zero is the corpus and not the probe (F6). A reader that cannot be validated where it will be scored is not yet a proposal. ⚠️ This also **weakens `E-DE-SIGN-1`'s own pre-committed control**, which reads *"the 31 CoT-read clips (must read ≥ 80 %)"* — those clips are **all in train**, so that control can never run on the eval half of the planned sweep.
3. ⭐ **Record OSM as CLOSED-BY-COORDINATES on the standing-empty register (row 39)**, with the reason, so the next pass does not re-discover `maxspeed`.

## 4 · Stopping condition (Rule Zero)

**(3a) for `E-DE-KITS-1`, and (3a) again for `E-DE-SIGN-2` — both ran and both fired a pre-registered
branch.** ⭐ The pass explicitly did **not** stop at the KITScenes closure: the next lever was named,
and then **executed in the same run** (F6–F8), which is what turned a literature refusal into the
programme's first measured statement about validation coverage on the eval split. The next lever
after that (`E-DE-SIGN-3`) is again 0 GPU and unblocked, and is pre-registered above.

**Remaining levers on this thread, and what blocks each:** **`E-DE-SIGN-1`** — a weight/dataset
**download approval** (PI/MM) and the 4060, which was **not free today** (four `python.exe` in
`nvidia-smi`'s compute-apps list, 100 % util); the **dense model-supplier** route — PI ruling
**DE13-4**; **`map.xodr` values** — more NuRec scenes than the 2 held locally.

## 5 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Data Engineering/Research/2026-09-17-kitscenes-map-speed-values/RESULT.md` |
| Code | `repo:…/code/cot_sign_stratify.py` |
| Raw | `repo:…/raw/cot_sign_stratify.json` (69 train clips with country/day-night/road-class + the parquet cross-check; eval empty with its control), `repo:…/raw/search_log.md` |
| Primary | lib `2606.02956` (KITScenes; banked 09-15, **cited-by updated today**) — plus the devkit README and HF card, recorded with retrieval date (web pages, not bankable as PDFs) |
| Inputs (not copied) | `repo:TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_{train,eval}.jsonl.gz` · `devbox:C:/Users/Admin/tanitad-data/physicalai/metadata/data_collection.parquet` |
