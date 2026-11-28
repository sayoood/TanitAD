<title>CoT speed-sign values — 43 valued readings, not 31, all on legal steps; the positive control is SPECIFIED</title>

# `E-DE-SIGN-3`: re-derived from the CoT text — **43 valued clips in train (42 Vienna-convention), 0 in eval, 100 % on legal km/h / mph steps ⇒ `SPECIFIED`**. The programme's quoted **"31 of 4,572" does not reproduce**

**2026-09-18 · Research Lab (LAB-RUN-015) · Data Engineering · serves LR14-10 and the MM standing question (max-speed INPUT supplier)**
⛔ **Pre-registered:** `SPEC.md` + `code/cot_sign_values.py` pinned in `raw/prereg_pin.json` before the run. 0 GPU; text only; ⛔ **nothing derived from ego**.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **43 of 4,572 train clips carry a VALUED speed-limit reading in the CoT; 42 are Vienna-convention (km/h), 1 is US (35 mph).** All 43 are on the independently authored legal-step set (**43/43 = 100 %**). Day 25 / night 18. Values: 30 ×13, 50 ×8, 60 ×6, 70 ×5, 40 ×3, 100 ×3, 20 ×2, 80, 120, 35 mph. **0 clips carry two different values.** | MEASURED `raw/cot_sign_values.json` |
| **F2** | ⭐⭐ **Verdict `SPECIFIED`** (≥ 20 Vienna valued **and** ≥ 95 % legal). ⇒ `E-DE-SIGN-1`'s positive control is **fully specified on train**: 42 clips with a sign *number* a reader must reproduce. | MEASURED against the SPEC §4 table |
| **F3** | ⭐⭐ **The boolean flag is a leaky index in BOTH directions.** 69 flags; **41** of them carry a value, **28 do not** (mentions such as *"4. Type: speed limit sign"* with no number), and **2 valued clips are NOT flagged.** ⇒ neither `speed_limit == True` nor the 69 is a count of readings, and any pipeline keyed on the flag misses 2/43 (4.7 %) of the valued supply. | MEASURED |
| **F4** | ⛔⭐ **"31 of 4,572" does not reproduce.** This parser, which passed its literal and mutation controls, finds **43**. The 31 has no artifact path in the 09-17 package, and 09-17's own JSON counts only *mentions* (69). ⇒ **the standing-question text quotes a number whose provenance I cannot locate.** Per `CLAUDE.md` ("a number carries its arm and its artifact path"), it should be replaced by **43 (valued) / 69 (mentions)**, with this path. ⚠️ Not a retraction of anyone's measurement: a different parser may have produced 31. **It is an unsourced figure, now superseded by a sourced one.** | MEASURED + provenance audit |
| **F5** | ⛔ **Eval is still 0/147**, flags and values alike, so the validation gap on eval (`E-DE-SIGN-2`) is unchanged. `SPECIFIED` is a **train-only** statement. | MEASURED |

## 1 · Controls

| control | bar | read | |
|---|---|---|---|
| record counts | 4,572 / 147 | **4,572 / 147** | ✅ |
| flag count = 09-17 | 69 | **69** | ✅ same corpus |
| parser literal tests (5) | all pass | **5/5** | ✅ |
| mutation (naive first-integer parser) | must fail ≥ 1 | fails **≥ 1** (e.g. reads `"1. Type: speed limit sign"` → 1) | ✅ the tests discriminate |
| ⭐ manual precision audit (added, not pre-registered; it can only lower the count) | — | **115 matches in 43 clips, 0 false positives on inspection** (`raw/match_audit.txt`) | ✅ |

⚠️ **The legal-step check is weak by construction.** 30/50/60/70 are legal *and* the most common numbers in any speed sentence, so 100 % legality alone could not catch a mis-parse. The **manual audit** is the check that actually bounds precision. Recall is **not** bounded: a phrasing none of the four patterns covers would be missed, so **43 is a lower bound**.

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | The MM's standing question (max-speed INPUT channel blocked on a supplier); `E-DE-SIGN-1`'s approval request; tactical-goal vocabulary (`D-TLIGHT-1` family: labels that exist but never reach training). |
| **CONSEQUENCE** | `E-DE-SIGN-1` can go to the PI with a **specified train-side control** (42 Vienna readings, 18 at night) **and** an explicit *"no validation reference on eval (0/147)"* clause. That is the re-write LAB-RUN-014 §5.6 asked for. |
| **COMBINATION** | These 43 are the **only admissible, ego-free, in-corpus posted-limit values** the programme holds (the ego-future supplier is the refuted nav-echo path, R² 0.9702). Paired with today's Band-C scan (OSM `maxspeed` is crowd-sourced and sparse; our corpus has **no GNSS**, so OSM map-matching stays impossible), the supplier options are unchanged: **a sign reader** (validated on these 42) or **`map.xodr` from AlpaSim/NuRec scenes**. |
| **CHANCES / RISKS** | **Upside:** a real oracle for a sign reader at zero cost. **Risks:** (a) n = 42 bounds any reader-accuracy CI to about ±15 pp at 95 %; (b) the CoT is itself a VLM reading (`provenance: vlm-cot`), so this validates *reader-vs-VLM agreement*, not reader-vs-truth; (c) 30 km/h is 30 % of the set, so the class balance is skewed. |
| **EXPERIMENT** | **`E-DE-SIGN-4` (proposed DE18-1, 0 GPU):** freeze these 43 clip-ids + values as `sign_value_oracle_v1.json`, with sha256, as the fixed validation set for `E-DE-SIGN-1`. **Committed:** a candidate reader is admissible only if exact-value agreement is **≥ 80 %** with a Wilson 95 % lower bound **≥ 65 %** on the 42 Vienna clips. Below that, the reader is refused before any corpus-wide run. |

## 3 · Stopping condition (Rule Zero)

**(3a).** The committed bar was cleared (`SPECIFIED`). The supplier itself remains **blocked on a PI decision** (`E-DE-SIGN-1` needs a weight/dataset download approval), and it now has the validation set it lacked.

`Deliverables: SPEC.md · RESULT.md · code/cot_sign_values.py · raw/{prereg_pin.json, cot_sign_values.json, cot_sign_values.log, match_audit.txt, search_log.md}`
