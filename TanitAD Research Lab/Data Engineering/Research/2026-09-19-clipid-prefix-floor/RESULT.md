# Item 17 guard: the prefix floor is recorded — **530 prefixes in 53 files, against a NAMED list**

**Date** 2026-09-19 · **Owner** DataFlyWheel · **Asked by** Master Mind (the guard-owner task it offered) · **0 GPU**
**Closes** the gap flagged under item 17 in `PI_DECISION_QUEUE.md`: *"The prefix half has no stored floor."*
⛔ Nothing redacted, nothing rewritten. The PI's 2026-09-17 ruling stands: the banked record stays,
and the guard stops it growing.

## The result

`tools/clipid_baseline.json`, regenerated from tip **`e0c31f2`** (1,788 markdown files exported with
`git archive`, the same count `ls-tree` lists):

| | |
|---|---|
| UUID floor | **120 UUIDs in 27 files** — every per-file count **identical** to the ruling-day baseline; no new UUID-bearing file grandfathered |
| prefix floor | **530 prefixes in 53 files** (Project Steering 114 · Research Lab 416) — the ruling day's own 530 in 53 files, reproduced |
| the list it was counted against | `_clips_list`: **n 4,719**, **sha256 `a48251e89c7a8603…`** (= the v7 corpus id), named *"v7 corpus a48251e89c7a8603 (4,719 clip ids)"* |
| what it stores | per-file counts and the list's **digest** — no id and no prefix (checked against all 306,152 known ids and their prefixes: 0 in the file) |

MEASURED · `raw/verify_floor.json`. Re-writing the floor with the final tool gives a **byte-identical** file.

## Why the floor has to be NAMED

The same tree reads **530** prefixes against the 4,719-clip corpus and **870** against the
306,152-clip index. A floor that does not know its list turns *"which list?"* into false growth (an
index run against a corpus floor: 870 > 530) or hides real growth (a corpus run against an index
floor). So the baseline records the list's size and sha256 — the programme's existing corpus-id
convention — and **a run with any other list is refused, not compared** (`ClipListMismatch`). A
same-size list with **one** id swapped is refused too; a check on the count alone would pass it.

## What changed — `tools/clipid_scan.py`, `tools/tests/test_clipid_scan.py`

| change | why |
|---|---|
| `list_identity(ids)` → `{n, sha256_sorted}` over the sorted **set** | names and verifies the list without storing it |
| `--write-baseline --clips L` records `_clips_list`; `--clips-name` labels it | the floor carries its list |
| `--clips` with a different list ⇒ **`ClipListMismatch`** | see above |
| `--clips` against a baseline with no floor ⇒ **`ClipListMismatch`**, *"NO prefix floor"* | before: red on every legacy prefix |
| the default run compares the **UUID half only** | without a list the prefix half is not counted; comparing its 0 with the floor reports phantom "cleanups" (13 of them, measured — S9a below) |
| `--write-baseline` **refuses** if any file was unreadable | a floor written over unopened files under-counts, and the next real edit to one reads as a leak |
| the list file is read as **utf-8-sig** | a PowerShell-written list starts with a BOM, which would glue itself to the first id |

Unchanged: `UUID_RE` (imported by `a7_bank.py`), `scan_text`, `scan_tree` and the three refusals in
`compare_to_baseline`. Tests: **+8, now 18 — all pass** (`raw/pytest_clipid_scan.txt`); the 10
original tests are untouched.

## Green on legacy, red on growth — on the real tree

Each scenario runs the CLI as an operator would, on the tip export (S6–S8 on a copy of it).
MEASURED · `raw/verify_floor.json`, `raw/verify_floor.txt`.

| | run | result |
|---|---|---|
| S1 | default (UUID half) | **NO-GROWTH** · 120 UUIDs in 27 files · 0 shrank |
| S2 | `--clips` the v7 corpus list | **NO-GROWTH** · 530 prefixes · 67 files carry an identifier · 0 shrank |
| S3 | `--clips` the list **rebuilt** from the mirror's mp4 names, reversed, with BOM + CRLF | **NO-GROWTH** — a rebuilt list is the same list |
| S4 | `--clips` the 306,152-id index | ⛔ **refused**, `ClipListMismatch` |
| S5 | `--clips` 4,719 ids with **one swapped** for an index id | ⛔ **refused**, `ClipListMismatch` (same n, other digest) |
| S6 | **+1 prefix** in a file that carries 2 | ⛔ **refused**, `LeakGrew`: `…/2026-09-06-mm-decisions.md:prefixes 2 -> 3` |
| S7 | a **new file** carrying one prefix | ⛔ **refused**, `LeakGrew`: `new files: … (+0 uuid, +1 prefix)` |
| S8 | **−1 prefix** in that file (a cleanup) | **NO-GROWTH**, reported as `prefixes 2 -> 1` — a cleanup is never blocked |
| S8b | the copy restored | **NO-GROWTH**, 530 |

## Proof the tests can fail — mutation, not inspection

`code/mutation_audit.py` weakens one guard at a time in a copy of the tool and re-runs the suite.
Each anchor occurs exactly once in the source (asserted): a mutant applied to the wrong copy of a
line proves nothing. MEASURED · `raw/mutation_audit.json`.

| mutant | result |
|---|---|
| a · the digest check reduced to a count check | **1 red** — exactly `…DIFFERENT_list_of_the_SAME_size…` |
| b · a baseline with no named list is silently accepted | **1 red** — exactly `…WITHOUT_a_named_list…` |
| c · the prefix half compared although not counted | **1 red** — exactly `…does_not_report_the_prefix_floor_as_a_SHRINK` |
| d · a baseline written over unreadable files | **1 red** — exactly `…NOT_written_over_UNREADABLE_files` |
| e · the list identity depends on order and duplicates | **1 red** — exactly `…SAME_list_written_differently…` |
| f · the BOM not stripped | **1 red** — the same test |
| g · the `_`-key filter removed before comparing | **survives — an equivalent mutant.** The comparison walks the scan's files and never the baseline's keys, so `_clips_list` is never visited. Kept as defence; ⚠️ not load-bearing today |

## Landing: the three files go together — and either half alone is still safe

| if only this lands | default run | `--clips` corpus | `--clips` index |
|---|---|---|---|
| the new **baseline** with the old tool (S9) | NO-GROWTH, plus **13 phantom "shrank"** lines | NO-GROWTH | ⛔ **false red** (870 against a 530 floor) — the confusion the named list exists to stop |
| the new **tool** with the old baseline (S10) | NO-GROWTH | refused, *"NO prefix floor"* | refused |

Neither combination reads green on real growth.

## The list is never banked — how the next run rebuilds it

The 4,719 ids are clip ids, so they stay out of the repo. Two independent local sources reproduce the
digest, and their rebuilds are byte-identical (MEASURED · `raw/rebuild_list.txt`):

* the local mirror's file names, `C:\Users\Admin\tanitad-data\physicalai\camera\camera_front_wide_120fov\<clip id>.mp4`;
* the corpus's own `camera/camera_sha256.json` keys (private HF `Sayood/tanitad-v7-training-corpus`;
  a local copy sits under `D:\Projects\TanitAD-artifacts\_s2build-copy-20260919\`).

```
python code/rebuild_v7_list.py <temp>/v7_ids.txt --mirror C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov
python tools/clipid_scan.py --baseline tools/clipid_baseline.json --clips <temp>/v7_ids.txt
```

`code/rebuild_v7_list.py` writes the list **only** if it reproduces the recorded digest, and refuses a
path inside the repository. Both refusals were exercised: a source missing one clip (n 4,718) and an
output under `tools/` were refused, and nothing was written.

## Stale once this lands — flagged, not edited

* `Project Steering/PI_DECISION_QUEUE.md`, item 17: *"⚠️ The prefix half has no stored floor…"*. The
  Master Mind owns the queue.
* My own `2026-09-19-pi-queue-audit/RESULT.md` says the same. A dated note is **appended** there.

## Reproduce

```
python -m pytest tools/tests/test_clipid_scan.py                     # 18 pass
python code/mutation_audit.py <repo root> raw/mutation_audit.json     # a–f killed, g survives
python code/verify_floor.py --export <git archive of the tip's .md> … # arguments in its docstring
```

🔒 No clip id in any file here or in the three tool files: a scan over all 306,152 known ids, their
8-hex prefixes and their 12-hex tails finds **0**, against a control that reads the legacy prefixes
in `GOALS_AND_CLAIMS.md`. The guard itself, run over the tip with this package added, reads
**NO-GROWTH** on both halves.
