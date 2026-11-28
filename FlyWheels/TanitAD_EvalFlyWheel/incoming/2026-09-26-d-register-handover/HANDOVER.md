# D:-only steering lines → APPEND / REPLACE blocks for the tip (never whole files)

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-d-register-handover/` · **For** the Master Mind (single
committer) · **Date** 2026-09-26 · **Tip classified against** `da8400b7` · **Evidence** MEASURED unless stated.

## Why

The Master Mind found lines in D:'s `RETRACTION_LOG.md` and `GOALS_AND_CLAIMS.md` that are not on the tip, while D:'s
copies are otherwise ~3,700 lines behind it — so a whole-file land would revert the tip. It asked for **append blocks**.

⛔ **"Not on the tip" is not a sufficient test.** A line can be missing from the tip because it was WRITTEN on D:
(hand it over) or because the tip later CORRECTED it (re-adding it would resurrect the superseded text). And a line
written on D: can be an edit of an existing tip row rather than content. `code/d_only_register_lines.py` separates them:

| class | meaning | handed over? |
|---|---|---|
| `HISTORICAL` | in some branch version of the file (`HEAD..tip`, every version read) — the tip since changed it | ⛔ never |
| `SHA12_REDACTION` | an existing tip row with its clip handles rewritten to `sha12:` | ⛔ no — the PI ruled 2026-09-17 not to redact banked evidence (`tools/clipid_scan.py` docstring) |
| `OTHER_EDIT` | any other edit of an existing tip row | listed below for a decision; never auto-applied |
| `DESCENDANT` | the tip row survives in the D: row **character-complete**, and D: adds to it | ✅ **REPLACE** (cannot revert anything) |
| `NEW` | no tip line within similarity 0.5, **or** the nearest one had over half its tokens changed (a different item from the same template) | ✅ **APPEND**, verbatim, with an insertion anchor |

## The two registers (what you asked for)

| file | D:-only lines | NEW | DESCENDANT | SHA12_REDACTION | OTHER_EDIT | HISTORICAL |
|---|---:|---:|---:|---:|---:|---:|
| `RETRACTION_LOG.md` (23 branch versions read) | 17 | **17** | 0 | 0 | 0 | 0 |
| `GOALS_AND_CLAIMS.md` (88 branch versions read) | 26 | **3** | **1** | 21 | 1 | 0 |

* **`APPEND_RETRACTION_LOG.md`** — one region, D: lines 17029–17049: **R25** (2026-09-26), *"the planner's rule is
  PDM-shaped"* → it is NAVSIM v2-shaped while the harness scores v1. The id is free on the tip (its R-series ends at
  R24); the tip's newer entries use `RETR-2026-09-26-*`, so rename it at your discretion. Anchor: after tip line
  17670 — the tip has 118 newer lines after that point, so appending at the END is equally valid.
* **`APPEND_GOALS_AND_CLAIMS.md`** — `D-REFE-OPSWITCH-1`, `D-REFE-COMFORT-1`, `D-REFE-RULE-1` (none of the three ids
  is on the tip); they go directly after `D-REFE-SEL-1` in the same table.
* **`REPLACE_GOALS_AND_CLAIMS.md`** — `D-REFE-SEL-1`: the tip row's 1,960 characters are all kept, plus 1,496 added
  (the epoch-12 snapshot UPDATE and its artifact paths).
* `OTHER_EDIT`, optional: row `D-REFCV5-ORACLE-UNRUNNABLE`, character 797 — the tip has `⛛` (U+26DB) where D: has
  `⛔` (U+26D4), in *"`layer.agent_gate`. ⛛ **PARITY, BITWISE…"*. The tip's glyph looks like a slip; one character.
* The 21 `SHA12_REDACTION` lines are D: rewrites of existing rows (each clip handle on the tip replaced by its
  `sha12:` form, nothing else changed). They are recorded in `raw/d_only_register_lines.json` by line number, class and sha256 only — **no text**, so
  this package copies no clip handle into a new file.

## ⚠ Whose lines these are

**Not the EvalFlyWheel's.** This session never wrote either register this week (its only register write was on
2026-09-20, `RETR-2026-09-20-W5-SIBLING-FILE-LINE`, which is on the tip — the positive control for the method). Every
NEW / DESCENDANT line above cites files of **`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`**
(`eval/SPEC_NAVTEST.md`, `eval/rule_mismatch_diag.py`, `eval/RESULT_E6_sub200_ep012.md`, `refe/planner.py`) — the REFe
stream. Its `LANDING_READY.txt` lists **`Project Steering/GOALS_AND_CLAIMS.md` and `RETRACTION_LOG.md` as WHOLE FILES**
in its batches 1, 2, 9, 11 and 12. ⛔ Those entries must never be landed as files (the ~3,700-line revert); these
blocks are the safe equivalent. Worth a word with the REFe owner before landing its rows under its name.

## Same hazard, two more files (optional — not asked for; same tool, same checks)

REFe's batches also list `MODEL_REGISTRY.md` and `PI_DECISION_QUEUE.md` whole-file. Their D:-only lines:

| file | NEW | DESCENDANT | OTHER_EDIT | HISTORICAL |
|---|---:|---:|---:|---:|
| `MODEL_REGISTRY.md` | 4 | 1 | 1 | 2 |
| `PI_DECISION_QUEUE.md` | 10 | 0 | 0 | 0 |

* ⭐ **`PI_DECISION_QUEUE.md` — four PI decisions from today exist ONLY on D:** (*"I choose B and follow the paper
  version"* ~11:05; *"I accept the finish time extension"* ~13:35; *"no, do not stop the label pipeline at step 9,771"*
  ~18:05; *"do 2 and 3"* ~20:28). On the branch, the REFe item of ~10:40 (*"how should it be supervised?"*) still reads
  `🔴 NEW ITEM` with no decision. Block: `optional_other_steering/APPEND_PI_DECISION_QUEUE.md` (D: 1538–1556, anchored
  after tip line 1725).
* `MODEL_REGISTRY.md` — REFe's `**Status**` row extended (REPLACE, 909 characters kept + 603 added) and four new field
  rows (recipe change at step 3,708, finish vs budget, label_version 3, selection rule) anchored after it.
  `OTHER_EDIT` D: line 5258 (`**Declared departures from the paper**`) changes tokens inside an existing tip row —
  the REFe owner should reconcile it; it is not emitted.
* ⛔ **The 2 `HISTORICAL` lines are exactly why "not on the tip" is not enough:** D: lines 2782 and 2796 are the
  pre-correction yaw-rate readings that `RETR-2026-09-26-YAWMASK` corrected today. Re-appending them would undo that.

## Verification

* Every emitted region and replacement row equals D:'s worktree lines **verbatim** — 6/6, checked by an independent
  reader that re-derives each block from its marker's line numbers, not by the emitting code.
* `tools/clipid_scan.scan_text` on all 10 package files: **0 UUIDs** (positive control: a planted UUID reads 1).
  Called directly because the scanner's `DEFAULT_GLOBS` cover only `Project Steering/**` and `Research Lab/**` — a
  `--root` run on this package would have scanned zero files and read clean. Backticked 8-hex tokens in the REFe blocks
  are file hashes (three resolve as git blobs; `train.py`, `model.py`, `navsim_dac.py` and an md5 by context).
* `tools/secret_scan.py --tree`: 10 files, **0 blocking, 0 advisory**.
* `tests/test_d_only_register_lines.py` — **11 passed**, literal lines only (synthetic clip handles). Three source
  mutations in a scratch copy each turn exactly one test RED: dropping the same-row rule (a REFe heading was once read
  as an edit of a refcv7 heading from the same template), dropping the `SHA12_REDACTION` class, and letting a region
  drop a line that is also on the tip.

## Manifest

| artifact | where |
|---|---|
| `APPEND_RETRACTION_LOG.md`, `APPEND_GOALS_AND_CLAIMS.md`, `REPLACE_GOALS_AND_CLAIMS.md` | this package (staged) |
| `optional_other_steering/{APPEND_MODEL_REGISTRY.md, REPLACE_MODEL_REGISTRY.md, APPEND_PI_DECISION_QUEUE.md}` | this package (staged) |
| `code/d_only_register_lines.py`, `tests/test_d_only_register_lines.py` | this package (staged) |
| `raw/d_only_register_lines.json`, `raw/d_only_other_steering.json` (line numbers, classes, row ids, sha256 — no text) | this package (staged) |
| D:'s register copies themselves | ⛔ untouched. They are **staged whole-file** in D:'s shared index (`M `, by another stream) — a loaded gun for any pathspec-free commit; not mine to unstage. |
