# AMEND1_CODING -- Amendment 1 (A1-2026-10-27) coded into the DAG-OACC reader

Spec: `E:/Projects/TanitAD-artifacts/dagland/dagoacc_amend1/AMEND1.md` (sha256 `44d7e879e5349e5791d1a6ad9ff18559cfd72ea63ce3ef634dcde16e9a33373e`). Pre-amendment files: `dagland/dagoacc_amend1/before/`. CPU only; no DAG-OACC value opened (the dry run hashes the pulls, as before); nothing staged or committed.

## What changed (all in `dagoacc_t1/`)

| file:line | change |
|---|---|
| `dagoacc_read.py:63-65` | `AMEND1_ID`, `AMEND1_SHA256`, `AMEND1_REPO_PATH` |
| `:108-130` | `OUTCOME_TABLE_PREREG` = the PREREG's six rows as printed (static-check target); `AMEND1_ROWS` rewrites exactly DOMINANT's condition and LONGITUDINAL FIXED's condition + committed next (item 3); `OUTCOME_TABLE` = in force. Order, other rows, NOT FIXED's HACC next unchanged |
| `:194-229` | `decide(controls_failed, sf, m1_worse)`: `dominant = pa and kp and d123 and crawl_pass`; `LONGITUDINAL FIXED = pa and kp and lon_all and not dominant` (the exclusion is in the row, not left to the order); `m1_worse` required for both seeds; `reading` removed. Adds `dominant_clauses_failed_per_seed` (D1/D2/D3/crawl) and `crawl_open`. UNPLACED kept as a defensive branch (`:132` text says so) |
| `:231-249` | `decide_unamended`: the old logic verbatim (default reading); `:268` `unamended_beside` -> `outcome_unamended` (descriptive, decides nothing); `ambiguity()` removed; `outcome_from` (`:826`) computes the decision first |
| `:251-261` | `outcome_block`: LF carries `reported_as` = "D1-D3 held, crawl open" or "not DOMINANT: seed s fails ..." |
| `:544-565`, `:608`, `:632-643` | `amend1_problems` (the coded rows must still be verbatim in AMEND1.md; untouched rows must equal the PREREG's); `--amend1` (default the E: copy) pinned by sha256 in `own_pin_problems`; provenance records it |
| `dagoacc_t1_chain.sh` | `AMEND1`/`AMEND1_SHA` (STOP if absent or different), `--amend1` in the static-check and reader calls, `amend1=` in PINS.txt, rc 3 text. `chain.diff` regenerated. LAUNCH command unchanged |
| `enumerate_cells.py:39-93` | `expected()` = amended table, `expected_unamended()` = old; also checks `decide_unamended`, and that no non-REFUSED cell matches two rows |
| `dagoacc_t1_selftest.py` | L1d-L1f (amendment text, pin, table in force), L8 rewritten (three gap reasons, crawl-open, clause lists, 36,864-combination sweep: never UNPLACED, exactly one row), L9, C6b/C11 (AMEND1 pin), H4b/H5b |

## Counts (524,288 cells)

| | before | after |
|---|---|---|
| UNPLACED | 255 | **0** |
| LONGITUDINAL FIXED | 3,837 | 4,095 |
| DOMINANT | 4 | 1 |
| NO EFFECT / COSTS POSITION / NOT FIXED / REFUSED | 196,608 / 49,152 / 12,288 / 262,144 | same |
| enumeration rc / disagreements | 3 / 0 | **0 / 0**, every row reached, non-REFUSED multi-matches 0 |
| words that change | | 258 = 255 UNPLACED->LF + 3 DOMINANT->LF (crawl open) |
| self-test | 244/244 | **287/287** |
| mutants | 105/105 (107; 2 unverified) | **127/127** (129; the same 2 real-run-only unverified, K09, K15); control green |

Mutants: R21, R22, R26, R27, R55 re-meant or replaced (the old R26 and R27 are equivalent or gone: the branch is unreachable, `DEFAULT_READING` is gone); R17-R20, R23, R24, R60, R62, R63 re-anchored (`decide_unamended` repeats the old lines); new R64-R81 and K17-K20 (crawl clause dropped, `not dominant` dropped, AMEND1 pin dropped, text check dropped, unamended feeding the decision, ...).

## Held fixed (proof)

The rehearsal on banked data (DAG-0 stands in; no DAG-OACC file): the hash of `{rows, t0_beside, controls_PASS}` is `95431ed6...` before and after; only the OUTCOME block moved (golden `9111eca9...` -> `0cef4bbd...`). Real-config dry run ends `ZZDAGOACCT1-DRY-OK` (22:41).

## Notes

* The amendment's crawl clause is coded as **M1** separated-worse; L2 is a LATERAL row inside D2 (so an L2 failure already breaks D2 and is named `D2`). The decision is identical to "L2 and M1".
* `decide()` now needs `m1_worse` for both seeds (ValueError otherwise): the crawl clause is never silently False.
* AMEND1.md is pinned in place (the dagland copy), not copied into `inputs/`; if it moves, pass `--amend1` / `DOT1_AMEND1`.
