# E10 RESULT — Alpamayo-augmentation counts re-derived from the bytes (2026-10-18)

Brief: Master Mind 2026-10-18, BACKLOG row **E10**. CPU only, nothing committed or pushed, paper not edited.
Evidence class of everything below: **MEASURED, 2026-10-18** (`count_alpamayo.py`), except where marked.

## Headline — what needs a decision or a landing

1. **Land registry §11.3** (new subsection + one edited index line in §11). Base blob
   `git rev-parse d6a4ce9:"Project Steering/MODEL_REGISTRY.md"` = **`8a0ece41f9fbd75a579c4edbe980465a85740129`**;
   edited file = blob **`66e73dc0588c5b07ec46ffb8a9063ca4354533bf`** (618,056 B; base 582,189 B). Diff vs base:
   **+249 / −1 lines**, LF preserved (0 CR in base, 0 in edit), nothing else touched. ⚠️ The lander must confirm the
   registry blob on the tip it lands on is still `8a0ece41…` — other agents edit that file; if not, re-apply the
   inserted block (it sits between the closing `---` of §11.2 and `## 12.`).
2. **One paper conflict, decision needed (paper NOT edited): the "4.3 % contaminated" claim** at paper L199, L3073,
   L3924, L6600. `201 / 4,729 = 4.2504 %` is the CATALOGUE rate; a split can only contain clips that exist, and the
   rate over the buildable set is `201 / 257 = 78.2101 %` (**18.4×** larger). Registry §12.4 (C113) already says
   "quote 78.21 %"; the paper contains 0 occurrences of "78.21". Proposed edit text is in `CONFLICTS.md` C-1.
3. **`CLAUDE.md` rule 2 wording**: "779 … WITH THE COLOUR ATTACHED" — 779 is the any-TL-goal total; **761** have a
   colour, **18** are colourless. Every component number (376/363/22/18/779) reproduces. `CONFLICTS.md` C-2.
4. **BACKLOG E10's premise was stale** (paper had carried the counts as MEASURED + registry-anchored since
   2026-08-16); the row can close — status line at the end.

## The counts, each with its LAYER (full ledger: `counts.json`, 114 counts; registry §11.3)

Layers per CLAUDE.md rule 2: **PUBLISHED** = NVIDIA's corpus · **EPISODE** = our episode build · **AUGMENTED** = our
augmented label release.

| layer | count | value | ledger id |
|---|---|---|---|
| PUBLISHED | clips in PhysicalAI-AV (`data_collection.parquet` rows) | **306,152** | `PUB1` |
| PUBLISHED | A2 clips found in it / share | 4,729 of 4,729 / 1.5447 % | `PUB3`, `PUB4` |
| AUGMENTED (A2) | rows / distinct clips / columns | **23,644 / 4,729 / 12** | `A01`–`A03` |
| AUGMENTED (A2) | rows per task: trajectory, meta_action, auto_labeling, vqa / grounding_via_vqa | 4,729 each / **4,728** | `A04.*` |
| AUGMENTED (A2) | selection manifest → expected rows | 4,800 clips → 24,000 | `A08.*` |
| AUGMENTED (A2) | zero-row clips / off-manifest clips / short-clip deficit | **81 (−405) / 10 (+50) / 1 (−1)** | `A09.*` |
| AUGMENTED (A2) | rows missing vs 24,000 | **356 = 1.4833 %** | `A11.*` |
| AUGMENTED (A2) | delivered clips by road class: urban / intersection_rich / highway / unstructured / unlabelled | **1,884 / 1,241 / 384 / 83 / 1,137** (labelled 3,592, all 5-row complete) | `A14.*` |
| AUGMENTED (A2) | wall time / s per clip | 78.36 h / 59.65 s | `A15.*` |
| EPISODE | parity-train membership | 2,400 clip ids (see N-3 in CONFLICTS) | `B02.*` |
| EPISODE | A2 clips inside parity train | **201** (listing = sha256 oracle = banked exclusion list) | `B03.*` |
| EPISODE | …over all A2 clips (catalogue) / A2 clips with a w120 cache / …buildable rate | 4.2504 % / **257** (201 + 56) / **78.2101 %** | `B04`–`B06` |
| EPISODE | A2 clips with NO w120 cache | **4,472 (94.57 %)** | `B05.no_w120*` |
| EPISODE | val-40 episodes inside A2 | **6 of 40** (sha8 cross-check 40/40) | `B08.val40` |
| AUGMENTED (aug120) | fused clips / with Alpamayo layer | **201 / 201**; A2 join = 1,005 rows = 201 × 5 | `C04`, `C08` |
| AUGMENTED (aug120) | SAM3 covered / gap / ratio to the "8 clips" first recorded | **86 (42.79 %) / 115 (57.21 %) / 14.375×** | `C03`, `C06`, `C07` |
| AUGMENTED (aug120) | corroborations / conflicts (run accounting) | 88 / 10 | `C04` |
| AUGMENTED (v7.2) | records train / eval | **4,572 / 147** (all clip ids ∈ A2; train ∪ eval = A2 ∩ manifest = 4,719 exactly) | `D01`, `D02` |
| AUGMENTED (v7.2) | TL goals train: RED / GREEN / YELLOW / colourless | **376 / 363 / 22 / 18** | `D03.train_TL` |
| AUGMENTED (v7.2) | records with any TL goal / with a coloured one | **779 / 761** (eval: 26 / 26 of 147) | `D03.*` |
| AUGMENTED (v7.2) | `cot_source.cot` byte-equal to A2 `meta_action` cot | **4,572/4,572 and 147/147** | `D06.*` |
| EPISODE∩AUGMENTED | where the 201 sit: v7.2 train / v7.2 eval / off-manifest | **190 / 11 / 0** | `D02.parity_overlap_breakdown` |

**Paper cross-check:** 26 Alpamayo-augmentation figures located in the paper (every line machine-checked to hold the
quoted form); **25 MATCH, 1 CONFLICT** (PC25, the 4.3 % denominator). The paper's evidence class for these figures is
already MEASURED + registry-anchored (L2521), not INHERITED.

## How this was verified (and what the checks can and cannot show)

* `count_alpamayo.py` exits 0 only if its assertions hold: `alpamayo_clip_ids.txt` == parquet clip set; the parity
  listing reproduces the committed oracle digest; exclusion list == listing overlap == oracle overlap;
  aug120 `todo` == v2 union == S2 aug120 split == exclusion list; 257/201; batch accounting == summary;
  v7.2 `cot` byte-equal to A2.
* **Byte-deterministic:** two runs from different cwd, `cmp` identical (a first run was not — a set-ordered list of
  number forms under hash randomisation; fixed by a total sort).
* **Sensitivity control:** deleting ONE entry from a copy of the selection manifest moved 13 ledger values and turned
  five paper claims to CONFLICT and `D02.union_vs_a2_in_manifest` to NO — the counter reads what it is fed.
  ⚠️ The same control showed `A10` (the 24,000 − 405 + 50 − 1 closure) still closes: it is a bookkeeping identity,
  not an independent check, and the registry says so.
* **Second-method recounts** that share no code with the script: TL goals by raw `zcat | grep -c` over the gz bytes
  (376 / 363 / 22 / 18 / 779, nonexistent-key control = 0); A2 rows from the parquet footer (23,644) and
  `pyarrow.compute` unique/value_counts (4,729; 4,729 ×4 + 4,728); zero-row 81 / off-manifest 10 by set difference.
* **Runs on the `D:` checkout too, with a finding:** `--skip-paper-pins` run against `D:/Projects/TanitAD` gives 114 counts
  **byte-identical** to the tip run, but that checkout's paper blob is `b2d844da` (not `c9c56706`) and its registry blob is
  `f47911d5` (not `8a0ece41`) — **`D:` lags the mirror tip** — and 11 of the 17 counted inputs differ in sha256/size there
  because the text files are CRLF in the worktree. Hence the registry hashes the LF blob bytes and gives git blob ids
  (registry §11.3 "Limits"). Without the flag the script exits 1 on `D:` because the paper-line pins fail loudly, by design.
* Pins: `records.parquet` sha256 `ecae276d…` equals registry §11.1's pin; v7.2 md5s equal the release README and
  registry §4.5; the oracle digest `e61a0455…` reproduces from the listing.
* A UTF-8 trap bit once (`UnicodeEncodeError` on a `∩` print under cp1252) — the script and its runner use
  `PYTHONUTF8=1` and write ASCII-safe progress lines.

## Limits (also in registry §11.3 "Limits")

* "Has a w120 cache" (257) is read from the S2 clip index, not from the w120 video (Thor/pod-side).
* The HF dataset card (4,800 / 23,999 / "one" missing) is not banked; those values stay INHERITED. `A13` is
  MEASURED ÷ INHERITED and says so.
* `PUB5` / `A18.est_hours` assume 20 s per clip (ESTIMATED).
* §11.2's vocabulary tallies (`g_str` 151/31/17/2, voter majorities, 15 `grounded`) are NOT re-derived — the per-record
  fused files are HF-only; the aug120 numbers here come from the run's accounting JSON and label-source map.
* v7.2 lineage is verified field-for-field only for `cot_source.cot` and `chain_of_causation`.
* The brief's path `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-09-04-v72-label-release/…` does not exist at
  `d6a4ce9`; the release is under `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/`.

## Deliverable manifest (nothing is committed; everything below lives ONLY in `C:/Users/Admin/e10/` until landed)

| artifact | intended repo path | local source | exists in |
|---|---|---|---|
| counting script | `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-18-e10-alpamayo-counts/count_alpamayo.py` | `C:/Users/Admin/e10/count_alpamayo.py` | local only (sha256 `c5b023a4…`) |
| count ledger (114 counts, 26 paper claims, artifact sha256s) | `…/2026-10-18-e10-alpamayo-counts/counts.json` | `C:/Users/Admin/e10/counts.json` | local only |
| conflicts | `…/2026-10-18-e10-alpamayo-counts/CONFLICTS.md` | `C:/Users/Admin/e10/CONFLICTS.md` | local only |
| this report | `…/2026-10-18-e10-alpamayo-counts/RESULT.md` | `C:/Users/Admin/e10/RESULT.md` | local only |
| edited registry (§11.3 added) | `Project Steering/MODEL_REGISTRY.md` | `C:/Users/Admin/e10/MODEL_REGISTRY.md` | local only; base blob `8a0ece41…` |
| landing manifest | — | `C:/Users/Admin/e10/paths.json` | local only |

`counts.json` holds counts only — no clip id, no pod address, no token. The inputs are repo blobs at `d6a4ce9` (listed
with git blob ids in registry §11.3) plus three uncommitted files named by sha256: A2 `records.parquet`
(`ecae276d…`, 25,970,018 B), A2 `selection_manifest.json` (`7d9f4a79…`, 340,800 B), `data_collection.parquet`
(`9624750c…`, 11,339,400 B).

## BACKLOG E10 status line (draft — replaces row E10 in `Project Steering/BACKLOG.md`)

```
| ~~E10~~ | ~~**Registry hygiene**: give the Alpamayo augmentation counts their own registry row (the paper carries them INHERITED)~~ ✅ **DONE 2026-10-18, 0 GPU** — registry **§11.3**: 114 counts re-derived from the bytes by `count_alpamayo.py`, each with artifact sha256 + counting expression + LAYER (`…/incoming/2026-10-18-e10-alpamayo-counts/`). ⚠️ The premise was stale (paper MEASURED + registry-anchored since 2026-08-16); **25 of 26 paper figures reproduce**; the one conflict is the paper's "4.3 % contaminated" (L199/L3073/L3924/L6600) — bytes and §12.4 say **78.21 %** (201/257); paper NOT edited → PI/Master Mind call. Also: CLAUDE.md "779 … with the colour" = 761 coloured + 18 colourless. | 0-GPU | nothing |
```
