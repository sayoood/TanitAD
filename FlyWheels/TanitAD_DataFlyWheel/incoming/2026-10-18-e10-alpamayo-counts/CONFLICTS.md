# CONFLICTS — BACKLOG E10 (2026-10-18)

Scope: every place where a number the programme prints about the Alpamayo augmentation disagrees with the
bytes. Counts: `counts.json` (produced by `count_alpamayo.py`; ids below are its ids and the registry §11.3 ids).
Paper = `Paper/TANITAD_PAPER.md`, blob `c9c56706` at tip `d6a4ce9`. **The paper was NOT edited.**

**Headline: 25 of the 26 Alpamayo-augmentation figures the paper prints reproduce exactly. One conflicts (C-1).**
Two further items are outside the paper (C-2 `CLAUDE.md`, C-3 the BACKLOG row itself) and three are
notes that change how a number should be read (N-1..N-3).

---

## C-1  Paper vs registry vs bytes — "4.3 % contaminated" divides by the wrong set

| paper line | text (paper blob `c9c56706`) |
|---|---|
| L198–L199 | "**(5)** **201 of the 4,729 Alpamayo augmentation clips are already inside the parity training corpus**, making any evaluation split built on it **4.3 % contaminated** — the same class as the ~80 % REF-A I-JEPA leak, at smaller scale" |
| L3073 | "any evaluation split built on the Alpamayo augmentation is 4.3 % train-contaminated" (§7.19e) |
| L3924 | "4.3 % of that corpus is already inside the parity training selection (§7.19e)" (§10 Roadmap, item 11) |
| L6600 | "any eval split on it is **4.3 % contaminated**, the REF-A I-JEPA class" (changelog) |

**What the bytes say** (`count_alpamayo.py`, all MEASURED 2026-10-18):

| ledger id | quantity | value |
|---|---|---|
| `B03.overlap_listing` / `B03.overlap_oracle` | A2 clips inside the parity-train clip set (id listing / committed sha256 oracle) | **201 / 201**, equal to the banked 201-line exclusion list as a set |
| `B04.catalogue_pct` | 201 ÷ **4,729** (all A2 clips = the catalogue) | **4.2504 %** — the paper's "4.3 %" reproduces arithmetically |
| `B05.a2_with_w120` | A2 clips that have a w120 cache (A2 ∩ S2 clip index) | **257** (201 aug120 + 56 w120val) |
| `B06.buildable_pct` | 201 ÷ **257** (a split can only contain clips that exist) | **78.2101 %** |

The arithmetic is right; **the denominator is the catalogue, not the buildable set**, so the claim
"any evaluation split built on [the augmentation] is 4.3 % contaminated" is wrong by **18.4×**, and the
framing "at smaller scale" is inverted — the buildable split is at REF-A-I-JEPA scale (~80 %).
`MODEL_REGISTRY.md` §12.4 already says exactly this (C113, 2026-08-18: *"THE 4.3 % FIGURE IS WRONG IN THE
FLATTERING DIRECTION … QUOTE 78.21 %"*). The paper never received it: the string "78.21" occurs **0** times in
the paper blob (`number_locator["78.21"].paper_lines == []`; control: "4.3 %" returns L199, L3073, L3924, L6600
plus one unrelated hit at L5906, an nrmse).

**Proposed fix (NOT applied — brief forbids editing the paper):** at L199, L3073, L3924 and L6600 replace
"4.3 %" with "**78.21 %** (201 of the 257 Alpamayo clips that have a w120 cache; 4.3 % = 201/4,729 is the
catalogue rate and is not the rate any buildable split would see)", and delete "at smaller scale" at L199–L200
and the matching clause at L3074. Cite registry §12.4 and §11.3 ids `B04`/`B05`/`B06`.
What stays correct and need not change: the 201 itself (L198, L3068, L3894, L6599), the exclusion-list
constraint (L3894–L3896, L3922–L3925), and "disqualifying as an evaluation split".

**Root-cause class** (already named in the registry): a contamination rate quoted over the CATALOGUE rather
than over the BUILDABLE SET — the denominator that flatters is the one that is easy to count.

---

## C-2  `CLAUDE.md` rule 2 — "779 … WITH THE COLOUR ATTACHED" (not the paper)

`CLAUDE.md` says: *"779/4,572 records carry a GT traffic-light tactical goal WITH THE COLOUR ATTACHED —
RED 376, GREEN 363, YELLOW 22, colourless TRAFFIC_LIGHT_REACT 18"*.

Every component reproduces to the unit (JSON-parsed, and a second method — a raw-bytes `zcat | grep -c`
over the gz file — gives the same 376 / 363 / 22 / 18 / 779, with a nonexistent-key control reading 0):

| ledger id (`D03.train_TL`) | value |
|---|---|
| `TRAFFIC_LIGHT_REACT_RED` / `_GREEN` / `_YELLOW` / colourless `TRAFFIC_LIGHT_REACT` | **376 / 363 / 22 / 18** (sum 779) |
| records with any TL goal | **779** of 4,572 |
| records with a **coloured** TL goal | **761** |
| records with only the colourless goal (`state: "unknown"`) | **18** |
| provenance | `vlm-cot` on 779/779 |

⇒ the sentence is internally inconsistent: **779 is the any-TL-goal total, 761 is "with the colour attached"**,
and the 18 it lists in the same breath are the ones *without* it. Suggested wording: "779/4,572 records carry a
GT traffic-light tactical goal; 761 with the colour (RED 376, GREEN 363, YELLOW 22), 18 colourless". Layer:
**augmented label release** (v7.2) — unchanged from CLAUDE.md. v7.2 eval has 26 TL goals on 147 records
(RED 8, GREEN 15, YELLOW 3), all coloured.

---

## C-3  BACKLOG E10's premise is stale — the paper does not carry these counts INHERITED

`Project Steering/BACKLOG.md` row E10: *"give the Alpamayo augmentation counts their own registry row (the paper
carries them INHERITED)"*. At paper blob `c9c56706` §7.15 (L2521) already states the counts as
*"MEASURED and registry-anchored — `MODEL_REGISTRY.md` §11.1"*; the two surviving mentions of "INHERITED"
(L2523–L2524, L6489) are the **retraction of the old card-derived 4,800 / 23,999**, not a live INHERITED flag.
Registry §11.1/§11.2 existed from 2026-08-16. What was genuinely missing — the bytes (sha256), the runnable
command per count, and the v7.2 / parity layers — is what registry §11.3 now supplies. The row can close.

---

## Notes that change how a number should be read (not paper conflicts)

* **N-1 Brief path does not exist.** The brief names the v7.2 release at
  `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-09-04-v72-label-release/raw/…`. At `d6a4ce9` (and on
  `D:/Projects/TanitAD`) it is at
  `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/`
  (blobs `d643acd0…` train, `f587ec60…` eval; md5 `0ff90213…` / `aa12c948…` match its README and registry §4.5).
  Counted bytes: sha256 `1f557d1c…` (train), `78d908f0…` (eval).
* **N-2 "98.52 % of the manifest" (registry §11.1).** 4,729 / 4,800 includes the 10 delivered clips that were
  never in the manifest; the in-manifest delivered fraction is 4,719 / 4,800 = **98.31 %** (`A12.clip_pct` note).
  The row-level 23,644 / 24,000 = 98.52 % is unaffected. §11.1 was left untouched; the nuance is in §11.3.
* **N-3 Parity denominators: 2,400 vs 2,376.** The membership oracle and the cache listing both enumerate **2,400**
  clip ids; `CLAUDE.md` and `parity.py:80-81` call the corpus **2,376 episodes** after a 24-clip corrupt skip-set.
  The 201 is membership in the 2,400. Whether any of the 201 is among the 24 skipped is not answerable from these
  artifacts — it would move 201/257 by at most the number of skipped clips among them.

## Corroborations worth recording (agree with earlier work, independently re-derived)

* Registry §12.4 "6 of the 40 canonical val episodes (15.0 %) are inside the Alpamayo record set" → `B08.val40` =
  **[40, 6, "40/40", 0]**.
* 2026-09-18 `…/2026-09-18-refcv6-eval-parity-clean/raw/oracle_intersections.json`: `v72_eval_INTER_parity_train`
  = 11 → ledger `D02.parity_overlap_breakdown` = **[201, 190, 11, 0, 0]** (190 in v7.2 train, 11 in v7.2 eval).
* NEW (in neither the paper nor §11.1/§12.4): all 4,572 v7.2 train and 147 v7.2 eval clip ids are A2 clip ids;
  (train ∪ eval) equals A2 ∩ manifest (4,719) exactly; `cot_source.cot` is byte-equal to A2's `meta_action` cot on
  4,572/4,572 and 147/147, and `chain_of_causation` to A2's `auto_labeling` on 4,572/4,572 and 147/147.
