# Deliverable manifest — `comma-yaw-reissue` (2026-07-27)

**HEAD at start:** `37ccfea` · **branch:** `agent/benchmarks-eval-20260721`
**STAGED, NEVER PUSHED.** Nothing committed, nothing pushed, no branch switch.
**Nothing lives in only one place** — every artifact below is in the repo working tree and staged.
**No pod was touched** (pod1 training, pod2 small validation, pod3 sibling sync all untouched);
this pass ran entirely on the dev box and required no GPU.

---

## A. New artifacts (this agent produced them)

| # | artifact | where it lives | note |
|---|---|---|---|
| 1 | `COMMA_YAW_REISSUE.md` | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-comma-yaw-reissue/` | **the deliverable** — inventory, corrections, stale-pending list, escalations |
| 2 | `raw/comma_yaw_inventory.json` | same dir | machine-readable inventory: **75 locations**, each with file, line/JSON path, corpus scope, `heading_repair`, `v_min`, verdict, and what was done |
| 3 | `raw/comma_yaw_anchor.json` | same dir | the anchor measurement + the **honesty condition**, per corpus; also carries the speed-bin defect table and the repair audit, read verbatim from the v3 artifacts |
| 4 | `MANIFEST.md` | same dir | this file |

**Build provenance (scripts, scratchpad only — deliberately NOT staged):**
`…/scratchpad/cyr_build_anchor.py` and `…/scratchpad/cyr_extend_inventory.py` generate (2) and (3)
by *reading* existing result JSONs. They perform **no measurement** — the only arithmetic is means
over per-seed values that were already stored, and percentage deltas between two stored values, both
labelled in the output. They are one-shot generators over artifacts that are themselves staged, so
they add reproducibility surface without adding a maintained dependency. **If a reviewer wants them
in the repo, say so and they will be staged** — the JSONs they emit are the durable artifacts.

## B. Existing files CORRECTED or ANNOTATED (superseded values preserved in every case)

| # | file | change |
|---|---|---|
| 5 | 🔴 `repo:Project Steering/MODEL_REGISTRY.md` | **THE FLAGGED DIFF** — §8.1 #6 gains a `LABEL-PROTOCOL CORRECTION 2026-07-27 (C29)` block. **Additive only; no existing figure altered.** |
| 6 | `repo:Benchmarks & Eval/LEADERBOARD.md` | §7.3 comma `yaw R² 0.000` marked ⚠️STALE-PENDING + correction block; **states the FAIL verdict is unchanged** |
| 7 | `repo:Project Steering/LOOP_STATE.md` | pod3 IDM entry annotated; NO-GO left standing |
| 8 | `repo:…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json` | added `label_protocol_reissue_2026_07_27` + one `usage_caveat`. **No existing value altered** — both yaw fields re-read after the edit; JSON re-parsed clean |
| 9 | `repo:Project Steering/RETRACTION_LOG.md` | C5 row gains a forward pointer to C29 (deletion was the wrong fix); row not rewritten |
| 10 | `repo:Paper/TANITAD_PAPER.md` | §H7 and §7.9 `yaw ≈ 0` annotated |
| 11 | `repo:Project Steering/PROGRAM_OVERVIEW.md` | H7 `yaw ≈ 0` annotated inline |
| 12 | `repo:…/2026-07-26-idm-v2/IDM_V2_RESULTS.md` | §3.2 yaw table gains the three-protocol table; B0/B1/V3sB/V3wB marked stale-pending |
| 13 | `repo:…/2026-07-26-idm-v2/IDM_DIAGNOSIS.md` | comma yaw **ceiling 0.352** annotated — the ceiling is a property of the broken label; the inference built on it is retracted |
| 14 | `repo:…/2026-07-26-idm-youtube-db-retry/DB_RETRY.md` | comma `0.0719` + defect #2 annotated (defect now closed) |
| 15 | `repo:…/2026-07-24-idm-pipeline-derisk/RESULTS_idm_pipeline_derisk.md` | *"comma cannot test yaw pseudo-labels"* — conclusion overturned |
| 16 | `repo:…/2026-07-24-branchb-transfer-eval/v1-encoder-char/RESULTS_v1_encoder_char.md` | caveat **C6** — disqualification lifted |
| 17 | `repo:…/2026-07-24-branchb-transfer-eval/MANIFEST.md` | *"untestable on comma"* annotated |
| 18 | 🔴 `repo:stack/tanitad/data/comma2k19.py` | **factual error corrected** — docstring said the deployed head reads `0.83` on repaired labels; it reads **`0.8108`** (`0.83` is a *retrained* arm). Comment only |
| 19 | 🔴 `repo:stack/tests/test_comma2k19.py` | same error, second copy — corrected. Comment only |

**Nothing outside a comment or a document was modified. No executable path was touched.**
⛔ `Project Steering/Mission Plan.md` was **not opened**.

## C. Verification

| check | result |
|---|---|
| `stack/` `pytest -q` | **1534 passed, 12 skipped** — exact match to the brief's baseline, **zero new skips** |
| `taniteval/` `pytest -q` | **661 passed** — exact match |
| edited JSON re-parses | ✅ `idm_head_v1_card.json` loads; both yaw values byte-identical to before |
| staging verified with `git ls-files --cached` | see §D — **not** by exit code |

## D. Staging

All 19 paths above staged into the working tree with `git add`. Because `git add` can silently no-op
on a file in a **new** directory, presence was verified with **`git ls-files --cached`**, not exit
codes. See the report for the verified listing.

## E. 🔴 Escalations — integration needed, and NOT left as a note in a README

1. **The comma loader still defaults to `HEADING_MODE_LEGACY`** — every new comma build still gets
   broken yaw labels unless the caller opts in. Needs an owner to flip it and rebuild.
   *(Carried from IDM_V3 escalation #1; still open.)*
2. **Re-score `idm_head_v1` on its own 9,420-window val with the repair on** — the highest-value
   stale-pending item, **no GPU training required**, only a scoring pass with the persisted head.
3. **The YouTube-IDM rotation channel was gated on a conclusion C29 overturns** — *"comma
   disqualified for yaw"* is lifted. That line's owner should decide whether to re-open it.
4. **`IDM_V3.md` §0 says `0.83` where its own §4.2 says `0.8108`** — its author should reconcile.
5. **Two roundings of one comma yaw number are in circulation** (`0.000` vs `0.0005`).

## F. Single-copy risk

**None.** Every deliverable is in the repo working tree and staged. The only files that exist in one
place are the two scratchpad build scripts (§A), which are regenerable and whose outputs are staged.
