# PLACEHOLDERS (report v39): every `{{LLR_...}}` the Master Mind fills after the LLR read

checks_v39.py must FAIL while any `{{` remains in the built report. Fill from the LLR `RESULTS.md` (same reader as CL / CLR; tier T1-kin, `reindexed` deciding), quoting numbers exactly and citing the repo path. No banned word and no T2 wording of the two-word forbidden kind (checks_v39.py scans for both).

| placeholder | file | what goes there |
|---|---|---|
| `{{LLR_RESULT_PENDING}}` | `sec_739_llr_result.md` (the whole file, one line) | the full subsection `**7.39.6 LLR: ... (`H-LEVELHEAD-2`, arm LLR).**` with Setup, Reading, the rules table, the 19-row tables (LLR − RIC0, attribution LLR − CLR on M1, composite LLRC − P10), families, floor, frame-freeze. Build the tables with `@@T <KEY> after=<substring>@@` directives once an `LLR` source key is added to `patch_v39.py` |
| `{{LLR_LABEL}}` | `sec_739_tail.md` (7.39.7 item 4), edits | the reader's outcome label, e.g. `P-FAILS-ATTR-NOT-SEPARATED-BETTER` |
| `{{LLR_ESTABLISH}}` | `sec_739_tail.md` | one sentence, MEASURED: M1 share closed per seed, guards, composite D1 / D2 / D3 |
| `{{LLR_LIMITS}}` | `sec_739_tail.md` | one sentence: floor reading per seed, single training run per seed |
| `{{LLR_NEXT}}` | `sec_739_tail.md` | the next arm or named blockers, per the A2 card's committed outcomes |
| `{{LLR_SENTENCE}}` | `edits_v39.md` (ST, abstract, 8.1, 8.3, 8.4 rows) | one results sentence reusing the numbers of `{{LLR_ESTABLISH}}` |

Counts in the part files: `sec_739_llr_result.md` 1 (`{{LLR_RESULT_PENDING}}`); `sec_739_tail.md` `{{LLR_LABEL}}`, `{{LLR_ESTABLISH}}`, `{{LLR_LIMITS}}`, `{{LLR_NEXT}}` once each; `edits_v39.md` `{{LLR_SENTENCE}}` in ST, ABS_LAST, NEXT_LAST, S81 item 29, R83 and ROW (1ay), `{{LLR_LABEL}}` in S81 item 29 and ROW (1ay).

LLR source key: `patch_v39.py` registers `RES_PATHS["LLR"]` automatically when `raw/2026-11-05-levelhead-llr-result/llr_read/RESULTS.md` exists at the tip (verify that path against the landed layout, `LLR_PATH` in the script). Once it does, `{{LAND_LLR}}` and `{{RES_LLR}}` fill-ins and `@@T LLR after=SUBSTR@@` table directives work in the section files; every table line must stay a line of that file.
