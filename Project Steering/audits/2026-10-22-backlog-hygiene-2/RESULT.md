# BACKLOG-HYGIENE-2 -- RESULT

**Package:** `C:/Users/Admin/par18/blhyg/` | **Base:** mirror tip `1664143ba9bfb9ad45aeb095a31a69dd6c5b7c30` (blob `Project Steering/BACKLOG.md` sha256 `064960afc7329c47...`, 175,213 B, LF) | 0 GPU | nothing committed, nothing pushed, no git working tree touched.

## HEADLINE -- needs the Master Mind to LAND (3 repo files + 1 script run)

1. Land `code/tools/backlog_drift.py`, `code/tools/tests/test_backlog_drift.py` and the **NEW** `code/tools/tests/fixtures/backlog_rows_2026-10-22.md` (new directory: use the lander, not a bare `git add` -- the "silent no-op in a newly created directory" trap applies).
2. Run `python code/apply_hygiene.py "Project Steering/BACKLOG.md" "Project Steering/BACKLOG.md"` on whatever the tip is at landing time. It is idempotent and refuses (exit 2, nothing written, reason named) if a target row is gone, duplicated, or has lost its closure evidence; `--allow-absent` skips a row a later sweep already pruned.
3. **One row still reads stale after the sweep, by design: `ZG-WIRE`** (DONE 2026-10-18 but its own text lists Open items (a)-(c)). The gate therefore still exits 1 until you decide it. Details under "Left for the MM".

## The brief's diagnosis was one layer too shallow -- MEASURED

The brief said the tool "reads only the leading verdict". It does not: `declares_done` already scans **every** bold span for a first-word closure verb. **The 12 listed rows were never parsed at all.** `TABLE_ROW` accepts only `[A-Z]\d+` ids (`A11`, `R4`); the ~80 rows added from 2026-10-17 carry ids like `EP-R1UB`, so the tool neither read them OPEN nor caught their closure -- it never saw them. The one guard against invisible rows (`UNPARSED`) covered bullets only. None of the 91 OPEN rows came from that family.

Attribution (`raw/attribution_parse_fix_only.json`, the new parser with CLOSED-LATER switched off): **the parse fix alone catches 10 of the 12 listed rows** (as DONE-BUT-OPEN). The new grammar is what catches `PRIVGEOM-DAC` (`UPDATE <date>: superseded`) and `L2-FRESH-STEPA` (`READ 1 DONE`), plus 4 more the list did not name (`PRIVGEOM-PROBE`, `NAVTEST-HEADBANK` CANCELLED, `D-SPACE`, `L2-STEPB-GATE`); and it supplies the label split ("the lead lies" vs "the row is honest, the id is not struck").

## Before / after (tool verdict counts, same file)

| | rows (distinct ids) | CLOSED | CLOSED-ELSEWHERE | **CLOSED-LATER** | **DONE-BUT-OPEN** | STALE-BLOCKER | OPEN | rc |
|---|---|---|---|---|---|---|---|---|
| OLD tool, tip bytes (`raw/before_old_tool.txt`) | 118 | 5 | 14 | - | 7 | 1 | **91** | 1 |
| NEW tool, tip bytes (`raw/before_new_tool.txt`) | **199** | 5 | 14 | **25** | **28** | 1 | 126 | 1 |
| NEW tool, swept bytes (`raw/after_new_tool.json`) | 199 | **59** | 14 | 0 | **1** (ZG-WIRE) | 0 | 125 | 1 |

Drift rows: 8 -> **54** (the other 46 were invisible) -> **1**. 59 closed = 5 + 52 struck + 2 struck by MM ruling. OPEN 126 -> 125 = +C2 (restated, no longer STALE-BLOCKER), -2 (SEARCH-P2-CONFIRM, LATENCY-BUDGET now struck).

Tests (`raw/tests_*.txt`): OLD tests on tip bytes **37 passed**; both test files on the swept bytes **98 passed** (`test_backlog_drift.py` 93 + `test_backlog_claim_check.py`); `test_apply_hygiene.py` **17 passed**. `backlog_claim_check.py` on the swept file: **0 new problems**, 70 -> 48 (it skips closed rows; `raw/claim_check_problems_{before,after}.txt`).

## Part 1 -- the tool (`code/tools/backlog_drift.py`, +188/-5; tests +318/-11)

* **Named-id family is parsed.** `NAMED_TABLE_ROW`, gated on an ITEM table (header first cell `#` or `id`): an indented timing table elsewhere has first cells `G-LIVE`, `**G-SUITE**` that are not backlog items.
* **UNPARSED now covers item-table rows** (it covered bullets only). An item-table row no id regex reads is printed, never swallowed.
* **`CLOSED-LATER`** (fails the gate, advice line printed). Fires when the row's item cell OPENS with an open-type bold lead (`NEXT, QUEUED, ACTIVE, OPEN, SCHEDULED, BUILDING, RUNNING, PROPOSED, REGISTERED, LAUNCH, DRAFT, CONDITIONAL, BLOCKED, BLOCKER, PLANNED, PENDING, FIX`) and a LATER bold span is one of:
  * `dated` -- first word DONE/CLOSED/SUPERSEDED/STRUCK/DISCHARGED/WITHDRAWN (+ the existing completed/cleared/retracted, + CANCELLED) **and a full date within 24 chars**; compounds (`DONE-marker`) refused outright, so the date is a second lock;
  * `update` -- `UPDATE <date>: <closure word>` (`UPDATE <date>: SEALED` does not fire);
  * `resolved` -- `RESOLVED <date>` **only under a BLOCKER/BLOCKED lead** (`D-SPACE`, `L2-STEPB-GATE`); anywhere else `resolved` is a sub-blocker's end or a decision id;
  * `qualified` -- see the judgement call.
  The lead list is a LABEL selector, not a firing condition (firing needs the later dated closure), so a wide lead set cannot raise a false alarm. A row that opens with the verdict, or with a struck/plain title, stays `DONE-BUT-OPEN` (`A9` pinned).
* JSON verdicts gain `closed_later_shape`; no consumer of the JSON exists in the repo (grep of the tip).

### The judgement call: `L2-FRESH-STEPA` ("READ 1 DONE ... O3", successor `UNSEEN-STRATUM-DATA` is its own row)
**Decided: strike it, and let the tool fire on it -- but only under a narrow rule.** A qualified DONE (`READ 1 DONE`, `STEP 1 DONE`, `S1 DONE`) is a PARTIAL-progress marker in every multi-step live row (`SEL-RANK`, `UNSEEN-STRATUM-DATA`, `P6-HN`, `R0E-PICK`, `PAPER-720`; all pinned as controls, 9 verbatim rows stay OPEN). It closes a row only as the row's **LAST bold span under an open-type lead**: rows are append-only, so a qualified DONE that nothing emphasised follows, under a lead still saying `NEXT`, is the row's final word. A later open marker keeps the row open (tested). For the row itself: its text says the brief's carve A is EMPTY so READ 1 (carve B) is the only read, all gates and the paired deltas are reported, and the O3 successor has its own row (which is itself CLOSED 2026-10-19). **MM: confirm or un-strike.** Un-striking one id cell reverts it; the tool will then flag it again.

### Additions beyond the brief (each one clause, removable, tested, mutation-killed)
`resolved` under a BLOCKER lead (2 live rows); `CANCELLED` as a closure word (`NAVTEST-HEADBANK`); the UNPARSED guard for item tables; the `FIX` lead.

### Tests and mutation proof
* Acceptance: every one of the 12 listed rows, **verbatim**, must fire as CLOSED-LATER, with the grammar branch pinned (`test_every_row_the_MM_LISTED...`).
* Controls (must not fire): a NEXT row mentioning "done" in prose; a bold `**DONE-marker discipline**` with and without a date; `**UPDATE <date>: SEALED**` plus its positive pair; `P7-WIRE-RESERVE` / `P7-WIRE-HEAD` verbatim; 9 verbatim multi-step rows; an undated closure; a qualified DONE with something after it; `resolved` under a non-blocker lead; `G-LIVE`/`G-SUITE` in a non-item table.
* The old live-file pin (`..._is_CURRENTLY_DRIFTED...`, rc==1 + ids A11..R4 on the live file) is **replaced, as its own docstring instructed on a sweep**: those seven rows + C2 are pinned on verbatim fixture rows; the live file is held to structure (gate runs, >= 30 named rows read, no UNPARSED). A live drift COUNT is deliberately not pinned (it would make `pytest` the backlog's gate).
* **Mutation: `raw/mutation_check.out` -- 12 of 12 killed, control green** (disable CLOSED-LATER: 27 RED; unparsed named family: 10 RED + 37 errors; no date requirement: 2; no compound lock: 1; qualified-not-last: 1; update branch: 2; resolved-any-lead: 1; item-table gate off: 2; UNPARSED swallowed: 2; classify ignores CLOSED-LATER: 15; lead gate removed: 6; CANCELLED dropped: 1). Harness `code/run_mutation_proof.py`; the tool's md5 verified unchanged.

## Part 2 -- the sweep (`code/apply_hygiene.py`, `out/BACKLOG.md`)

Convention: strike the id cell, keep every word (`BACKLOG.md:8`). **58 lines change, 0 deleted** (`raw/BACKLOG.diff`; the script verifies "insertions and `~~` only" per line before writing, and a test proves `apply()` enforces it). Tests: idempotent (byte-equal second pass), duplicates always refuse, evidence gone refuses, LF and CRLF preserved, 13 control rows byte-identical; mutation `raw/mutation_check_apply.out` **7 of 7 killed** (two survivors on the first run -- the no-deletion guard and EOL -- each earned a test).

**52 rows struck (evidence = the literal the script requires in the row at apply time):**

| id | tool verdict before | shape | tip line | closure evidence in the row's own text |
|---|---|---|---|---|
| A11 | DONE-BUT-OPEN | - | L26 | `DONE 2026-08-18` |
| A13 | DONE-BUT-OPEN | - | L32 | `DONE 2026-09-07` |
| B4 | DONE-BUT-OPEN | - | L45 | `DONE 2026-08-03` |
| C3 | DONE-BUT-OPEN | - | L57 | `CLOSED 2026-08-16` |
| R1 | DONE-BUT-OPEN | - | L124 | `DONE 2026-09-02 (D-REFAV1-LEAD-BLOCK)` |
| R2 | DONE-BUT-OPEN | - | L125 | `DONE 2026-09-02` |
| R4 | DONE-BUT-OPEN | - | L127 | `DONE 2026-09-02 19:26Z` |
| S4R-1 | DONE-BUT-OPEN | - | L530 | `DONE 2026-10-17` |
| P2F-1 | DONE-BUT-OPEN | - | L532 | `DONE 2026-10-18` |
| EP-1 | DONE-BUT-OPEN | - | L534 | `DONE 2026-10-18, 0 GPU` |
| SRTTC-DEV-APPLIER | DONE-BUT-OPEN | - | L538 | `DONE 2026-10-19` |
| P6-READ-PREP | DONE-BUT-OPEN | - | L547 | `DONE 2026-10-19` |
| LIT-EP-RL | DONE-BUT-OPEN | - | L565 | `DONE 2026-10-18, 0 GPU` |
| UNSEEN-STRATUM-DATA | DONE-BUT-OPEN | - | L569 | `CLOSED 2026-10-19 ~23:40: THE ROUTE DOES NOT EXIST` |
| SEARCH-STALE-T1 | DONE-BUT-OPEN | - | L579 | `CLOSED, NOT NEEDED` |
| P7-SPEC | DONE-BUT-OPEN | - | L583 | `DONE 2026-10-22` |
| SO-A16 | DONE-BUT-OPEN | - | L593 | `DONE 2026-10-17` |
| TEST-RED-1 | DONE-BUT-OPEN | - | L594 | `DONE 2026-10-18` |
| SO-CARVE | DONE-BUT-OPEN | - | L596 | `DONE 2026-10-18` |
| R0E-B2B5 | DONE-BUT-OPEN | - | L599 | `DONE 2026-10-19` |
| RL-1 | DONE-BUT-OPEN | - | L602 | `DONE 2026-10-19, 0 GPU` |
| RL-2 | DONE-BUT-OPEN | - | L603 | `DONE 2026-10-18, 0 GPU` |
| LOCK-STEAL-2 | DONE-BUT-OPEN | - | L604 | `DONE 2026-10-18, 0 GPU` |
| P6-RESUME-BANK | DONE-BUT-OPEN | - | L607 | `DONE 2026-10-20, for the NEXT ship` |
| HNSCREEN-CACHE-CONTENT-FP | DONE-BUT-OPEN | - | L609 | `DONE 2026-10-19` |
| HNSCREEN-FP-COST | DONE-BUT-OPEN | - | L610 | `DONE 2026-10-20` |
| HJ-UNITS | DONE-BUT-OPEN | - | L615 | `CLOSED 2026-10-18` |
| EP-R1UB | CLOSED-LATER | dated | L536 | `DONE 2026-10-19: outcome N, NONE-FEASIBLE` |
| NAVTEST-HEADBANK | CLOSED-LATER | dated | L537 | `CANCELLED 2026-10-19` |
| HEADROW-PROBE | CLOSED-LATER | dated | L539 | `DONE 2026-10-19: MIXED` |
| FASTNEG-CENSUS | CLOSED-LATER | dated | L540 | `DONE 2026-10-19: INCONCLUSIVE` |
| PICKRULE-AUDIT-NC | CLOSED-LATER | dated | L543 | `DONE 2026-10-19: ROW-MISS` |
| NC-FASTCAL | CLOSED-LATER | dated | L544 | `DONE 2026-10-19: NO-RECOVER` |
| EP-RELIAB-PICKS | CLOSED-LATER | dated | L545 | `DONE 2026-10-19: EP-OVERPREDICTED` |
| EP-ADMISSION-R1 | CLOSED-LATER | dated | L548 | `DONE 2026-10-19` |
| EP-ADMISSION-R2-CONFIRM | CLOSED-LATER | dated | L549 | `DONE 2026-10-19: SEAL AT 1.05` |
| PRIVGEOM-SAFE | CLOSED-LATER | dated | L550 | `DONE 2026-10-19: DAC INFO-UPSTREAM` |
| PRIVGEOM-CLEAN | CLOSED-LATER | dated | L553 | `DONE 2026-10-20 01:37: INCONCLUSIVE` |
| PRIVGEOM-DAC | CLOSED-LATER | update | L555 | `superseded by PRIVGEOM-SAFE` |
| PRIVGEOM-PROBE | CLOSED-LATER | update | L557 | `superseded by PRIVGEOM-DAC` |
| EPR1UB-GATE-CACHE | CLOSED-LATER | dated | L560 | `DONE 2026-10-19` |
| C2-GATE-SIBLING-LEAK | CLOSED-LATER | dated | L561 | `DONE 2026-10-19` |
| L1-STEPA-RUN | CLOSED-LATER | dated | L566 | `DONE 2026-10-19: O2 NOT ADMITTED` |
| L2-OFFPROP | CLOSED-LATER | dated | L567 | `DONE 2026-10-19: L2-ADMIT` |
| L2-FRESH-STEPA | CLOSED-LATER | qualified | L568 | `READ 1 DONE 2026-10-19 20:12: O3` |
| EPADM-SEARCHBANK | CLOSED-LATER | dated | L572 | `DONE 2026-10-19 ~23:55: ALL THREE DQ6 BARS MISSED` |
| EPADM-SEARCHBANK-S0 | CLOSED-LATER | dated | L573 | `CLOSED 2026-10-20, NOT BUILT` |
| SEARCH-DOSE | CLOSED-LATER | dated | L580 | `DONE 2026-10-20: O1_CONFIRMED` |
| R0E-B9B10 | CLOSED-LATER | dated | L600 | `DONE 2026-10-19` |
| HNSCREEN-TOKEN-NAMES | CLOSED-LATER | dated | L611 | `Row CLOSED.` |
| D-SPACE | CLOSED-LATER | resolved | L554 | `RESOLVED 2026-10-19 19:15` |
| L2-STEPB-GATE | CLOSED-LATER | resolved | L570 | `STEP B READ 2026-10-19` |

**5 MM rulings (2026-10-22), text INSERTED at the head of the item cell so the lead stops lying:**
* `SEARCH-P2-CONFIRM` -- id struck, `**CLOSED 2026-10-22: C2_FAIL_LOWER_CI**` (the C2 read, dPDMS -0.155 [-0.350, +0.037], is already in the row).
* `LATENCY-BUDGET` -- id struck, `**DECIDED 2026-10-20 (PI, D-LATENCY-BUDGET)**`. *Choice flagged: I struck it because a decided decision is not work; the MM said "relabel", so un-strike if you want it visible as a standing constraint.*
* `SEARCH-HOST-PORT`, `THOR-PICK-TICK`, `SEARCH-THOR-LATENCY` -- **not struck** (deferred is not closed), lead `**DEFERRED 2026-10-22 (MM ruling): to the optimisation phase**`; `SEARCH-THOR-LATENCY` also carries your sentence "The search is CLOSED on P2's scorer (C2, 2026-10-22); re-opens with a P6/P7 scorer."

**C2 (STALE-BLOCKER) -- restated, not struck.** The row's own text says its blocker is *UNVERIFIABLE, not cleared* while the markup struck it. I removed the two `~~` (words kept, tagged `[blocker NOT cleared ...]`) and appended a dated note: the one open item is a probe of whether `flagship-v2corpus-30k` finished; `MODEL_REGISTRY.md` section 1.7 on tip `1664143b` still reads RUNNING, ETA 2026-07-29T01:10Z, 85 days past, with the registry's own 2026-08-16 annotation that no completion row exists (`raw/registry_sec1_7_excerpt_tip1664143.txt`). I could not probe the pod, so "finished or dead" stays UNVERIFIED.

## Left for the MM (flagged, not changed)

| row | why not struck |
|---|---|
| **`ZG-WIRE`** (L612, still DONE-BUT-OPEN) | `DONE 2026-10-18` but the row lists **Open: (a)** trainer `summary.json {"done": true}` untouched on FAIL, **(b)** read the first real refcv7/8 ckpt by hand, **(c)** `ddv2_rl_refcv7.py` must gate `ckpt_latest.pt`. Striking would hide three live items. Either strike and split (a)-(c) into a new row, or restate. |
| `SEARCH-NAVTEST-PIPE` (not flagged) | last status "C2 RUNNING 2026-10-20"; the C2 read is recorded in `SEARCH-P2-CONFIRM` (2026-10-22), not in its own text. Evidence is cross-row, so I did not strike. |
| `EP-R1` (not flagged) | reads `BUILT + PRE-REGISTERED`; `EP-R1UB` ("EP-R1 closed at the fit; abandonment adopted") and `NAVTEST-HEADBANK` ("EP-R1 abandoned") say it is dead. Cross-row again. |
| `SH5-DAC-PROX` (not flagged) | BUILDING, with "UNSUPPORTED by PRIVGEOM-CLEAN", LOCAL/FULL DEPRIORITISED, grid 5 INSTRUMENT DEFECT, amend4 unsealed: partly closed, partly parked. |

**Pointers inside struck rows that have no row of their own** (all still readable in the struck row; promote if live): `RL-2` anchored RL arm BLOCKED on the A40, and its navtest look #2 which is conditional on RL-1 reading N-GAIN-LOST (RL-1 read N-BOTH, so the condition is not met); `RL-1` `H-REFE-RL1-NCRES` read needing a PREREG and the `run_v1.py:459` relpath fix; `SO-A16` bare-bash site `code/finalize_mini.py:156`; `PRIVGEOM-CLEAN` "PRECISION needs more logs (S0 build; PI decision)"; `HJ-UNITS` LEFT items (`p10_prior`/`v8f_mini`/`ra1_pooled` steer defaults, a pod-parity call); `HNSCREEN-FP-COST` pre-screen full-chain run on the run box; `HNSCREEN-CACHE-CONTENT-FP` S2 driver must call `finalize`; `R0E-B9B10` operator step `r0e_id_read.py`; `EPR1UB-GATE-CACHE` / `C2-GATE-SIBLING-LEAK` re-mint before any reuse.

## Observed, NOT changed (follow-ups)

* `declares_done` still reads a bold `**DONE-marker ...**` as a closure (first-word anchoring splits on `-`). Existing test `test_the_bullet_bold_PREFIX_is_restored...` pins that behaviour, so I did not touch it; the NEW branch refuses it.
* `--list-open` still lists `DEFERRED` / `PARKED` leads as OPEN (the three rows relabelled here, plus `FASTNEG-F25-RERUN`, `SH5-CREACH-GRID6`, `SEARCH-PESSIMISTIC`). A puller scanning that list still sees them; a DEFERRED/PARKED informational verdict is the natural next step.
* A row closed only by a **date-led** span (`**2026-10-22 07:53 Berlin, C2 READ ...: C2_FAIL_LOWER_CI**`, `SEARCH-P2-CONFIRM`) is still invisible to the tool; that is why it needed the ruling.
* `git archive` on this box exported the work-tree files with **CRLF** (autocrlf); every deliverable here is LF and blob-exact. Do not `cp` a CRLF checkout over the blob.

## Deliverable manifest

| artifact | where | note |
|---|---|---|
| `backlog_drift.py` (modified tool) | repo-candidate: `C:/Users/Admin/par18/blhyg/code/tools/backlog_drift.py` | **only copy**; diff `code/backlog_drift.tool.diff` |
| `test_backlog_drift.py` (93 tests) | `.../code/tools/tests/test_backlog_drift.py` | **only copy**; diff `code/test_backlog_drift.tests.diff` |
| verbatim-row fixture (NEW, 81 KB) | `.../code/tools/tests/fixtures/backlog_rows_2026-10-22.md` | **only copy**; new dir |
| `apply_hygiene.py` + `test_apply_hygiene.py` (17) | `.../code/` | **only copy**; one-shot, need not enter the repo |
| `run_mutation_proof.py`, `run_mutation_proof_apply.py` | `.../code/` | **only copy** |
| swept register | `.../out/BACKLOG.md` (sha256 `93d45d4a7a1d7513...`) | derivable: `apply_hygiene.py` on the tip blob |
| evidence | `.../raw/` (mutation outputs, before/after JSON+txt, test logs, BACKLOG.diff, tip blob, claim-check lists, registry 1.7 excerpt) | package only |
| test trees | `.../tree_before/`, `.../tree_after/` | scratch, reproducible |
