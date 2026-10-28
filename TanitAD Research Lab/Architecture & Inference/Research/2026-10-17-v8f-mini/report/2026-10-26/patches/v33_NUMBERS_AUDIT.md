# NUMBERS_AUDIT: report v33 (v32 + section 7.27 SPEEDSUB, 7.28 SPEEDSUB-T1, 7.29 LAUNCH-INTENT)

Audited deliverable (sha256): `V8F_SCIENTIFIC_REPORT.md` 8533bb8457038373fa1648bfd4fe48c1f4e6941c78bb18ed6567b99b637b26e0, built on source tip `8c1e9636`. The deliverable is the **final** version: section 7.29 carries the LAUNCH-INTENT read (HARMS) and no open slot (section I).

Method (mechanical, `audit_v33.py`): the v33 md was diffed against the v32 md on the tip (`8c1e9636388dbb105f6b1853bd95dfdccea9eb93`); every added or changed line was tokenised for numbers (sign kept, the Unicode minus read as `-`, thousands commas removed; a number is searched in its plain form and in its comma-formatted form at once, because the tokenizer drops the commas on both sides); each distinct signed token was looked up line by line in the source files below. `v33:N` are line numbers in `V8F_SCIENTIFIC_REPORT.md`. Sources are read from the blobs at the tip, except the one marked as outside the repo.

| source | role |
|---|---|
| `RESULT_V8F_SPEEDSUB.md` | the SPEEDSUB result, landed `af2dee4a` (primary source for 7.27; its tables are copied line for line) |
| `RESULT_V8F_SPEEDSUB_T1.md` | the SPEEDSUB-T1 result, landed `37d78ed1` (primary source for 7.28; its tables are copied line for line) |
| `PREREG_V8F_SPEEDSUB.md` | the SPEEDSUB design (wording and constants, not results) |
| `PREREG_V8F_SPEEDSUB_T1.md` | the SPEEDSUB-T1 design with Amendment 1 (wording and constants, not results) |
| `PREREG_V8F_LAUNCH_INTENT.md` | the LAUNCH-INTENT PREREG (landed `77234276`; primary source for 7.29: design, redesign disclosure, bars, arms, predicates, outcome table, NEXT-LAUNCH; its tables are copied line for line) |
| `PREREG_V8F_DAG_ORC.md` | the DAG-ORC design and Amendment 3 (the PI's routing); wording and constants |
| `RESULT_V8F_DAG_ORC.md` | the DAG-ORC result (cost lines, crawl rows) for the 7.26 update and the cross-references |
| `GOALS_AND_CLAIMS.md (register rows)` | only the register rows `H-V8F-SPEEDSUB-1`, `H-V8F-SPEEDSUB-T1-1`, `H-V8F-SPEEDSUB-T1-2`, `H-V8F-LAUNCH-INTENT-1`, `H-V8F-DAG-ORC-1`, `D-SPEEDBAND-EGO-ECHO-1` (the whole register is not a source) |
| `SUBREAD.json` | the SPEEDSUB read JSON (`raw/2026-10-28-speedsub-result/read/`) |
| `SPEEDSUB_T1_READ.json` | the SPEEDSUB-T1 read JSON (`raw/2026-10-28-speedsub-t1-read/`) |
| `SPEEDSUB_T1_BESIDE.json` | the logged-lane (bridge) beside JSON of SPEEDSUB-T1 |
| `REP_S0_SPEED_BIAS.md` | the Master Mind's diagnosis of REP s0 (sha256 `211601de…`) |
| `FINDING_1_ATTRIBUTION.md` | the attribution of the read-out's skill (sha256 `e89af4a0…`), banked at `842ec2ac` |
| `launch-intent-flag/RESULTS.md` | the lever's build results (tests, mutants, f) |
| `launch-intent-flag/MM_REDESIGN_2026-10-28.md` | the Master Mind's redesign note (bar and pairs), written before any LAUNCH-INTENT number |
| `launch-intent-flag/f_measure.json` | the measured GT-launch share f (49,618 of 306,566) |
| `launch-intent-v2/NOTES.md` | the reader and PREREG v2 notes (what the reader does not build) |
| `launch-intent-v2/enumerate_cells_li.out` | the enumeration of the 524,288 outcome cells |
| `launch-intent-chain/HANDOFF.md` | the box-2 chain handoff (self-test totals, pins) |
| `launch-intent-chain/NOTES.md` | the box-2 chain notes |
| `launch-intent-chain/PINS_FILLED.txt` | the pins of the chain (PREREG, reader, launch commit, trainer) |
| `launch-intent-chain/LAUNCH.txt` | the launch instructions of the chain |
| `DAGORC_READ.json` | the DAG-ORC read JSON (CTRL - P10 gaps, floors, REP - CTRL) |
| `RF_REP.json` | the replicate floors RF of DAG-ORC |
| `P0_RF0.json (rebase)` | the RF0 floors (DAG-0 - ORC-0), re-based file |
| `git log (cited commits)` | commit messages and dates of the commits named in the text |
| `RESULT_V8F_LAUNCH_INTENT.md` | the LAUNCH-INTENT result, landed `8c1e9636` (primary source for the read in 7.29; its tables are copied line for line) |
| `LAUNCH_INTENT_READ.json` | the read of record (reader output, sha256 `e451e1a6…`, banked at `d7dc9f11`): counts, ranges and shares are recomputed from it |
| `MLAUNCH.json` | the M-LAUNCH build (sha256 `f5ebb4b2…`): the four cells, the controls, the training-seed floor and the beside M2 block |
| `MLAUNCH NOTES.md` | the M-LAUNCH build notes (readings chosen, controls, design history, the post-hoc per-tick look) |
| `chain_v8f_launchintent.out` | the banked box-2 chain log (sha256 `ddae7b9e…`, CRLF read as LF): the launch times, the gates, the pulls, the DONE marker (MEASURED) |
| `R2C_NOTE.md` | the Master Mind's incident note on DAG-ORC r2 (repo path `raw/2026-10-28-dag-orc-result/notes/`, sha256 `df02ff2f…`): the REP provenance timing |
| `lit1_logs/chain.out` | the T1-kin chain log of the relaunch (dev box) |
| `lit1_logs/chain.out.prev_1028151251` | the T1-kin chain log of the stopped launch (RAM guard) |
| `lit1_logs/launch_intent_read_REFUSED.out` | the chain's own read with the launch-time reader (REFUSED on K-BUILD) |
| `read_A1/read_A1.out` | the console of the read of record |
| `check_result_filled.out` | the output of the RESULT's own checker (133 checks) |
| `REP launch-gate tokens (s0, s1)` | the launch-gate tokens of the two REP runs (they bind commit `8c24823f`) |
| `REP run records (s0, s1)` | the run records (`summary.json`) of the two REP runs (trainer seeds 10 and 11) |

**Totals.** 1162 added or changed lines (5 v32 lines replaced or removed); 1892 distinct signed numeric tokens (unsigned single digits 0-9 excluded as enumerations); 1880 found with the same sign in a source line (section A); 3 found only with a different or absent sign (section B); 5 found only in the v32 report (section C); 4 found nowhere (section D). B, C and D are each reviewed by hand in E.

⚠ **Limit of class A.** A token in A is a token that occurs somewhere in the source list, in a line the audit cannot tie to the claim; a coincidence cannot be told apart by this test. Sections G (own-section sources), F (claims resolved to lines) and H (recomputation from the banked JSON, assertion on mismatch) are the checks that tie decisive and derived numbers to their source.

## A. Tokens found with the same sign in a source (token -> v33 lines -> source:lines)

| token | v33 lines | source lines |
|---|---|---|
| `2026` | 3,1702,1702,1815,1970... | RESULT_V8F_SPEEDSUB.md:3,316,318,322...; RESULT_V8F_SPEEDSUB_T1.md:3,3,3,3...; PREREG_V8F_SPEEDSUB.md:3,3,9,12...; PREREG_V8F_SPEEDSUB_T1.md:3,5,133,194...; PREREG_V8F_LAUNCH_INTENT.md:3,3,3,3...; PREREG_V8F_DAG_ORC.md:3,3,37,194...; RESULT_V8F_DAG_ORC.md:3,3,3,3...; GOALS_AND_CLAIMS.md (register rows):1,2,2,3...; SPEEDSUB_T1_READ.json:7070,7070; REP_S0_SPEED_BIAS.md:3,46; FINDING_1_ATTRIBUTION.md:3,46; launch-intent-flag/RESULTS.md:3; launch-intent-flag/MM_REDESIGN_2026-10-28.md:1; launch-intent-v2/NOTES.md:3,63; launch-intent-chain/HANDOFF.md:1; launch-intent-chain/NOTES.md:1,28,47,54...; launch-intent-chain/PINS_FILLED.txt:1,4; launch-intent-chain/LAUNCH.txt:2,13,14,39...; DAGORC_READ.json:47,263,4072,4338...; git log (cited commits):1,1,1,4...; RESULT_V8F_LAUNCH_INTENT.md:3,3,3,161...; LAUNCH_INTENT_READ.json:5141,5511,5883,6253...; MLAUNCH.json:1651,1651,1669,1669; MLAUNCH NOTES.md:37; chain_v8f_launchintent.out:6,117; R2C_NOTE.md:2,4; lit1_logs/chain.out:14,14,16,16...; REP launch-gate tokens (s0, s1):5,27,28,51... |
| `10` | 3,3,1702,1702,1791... | RESULT_V8F_SPEEDSUB.md:3,46,274,316...; RESULT_V8F_SPEEDSUB_T1.md:3,3,3,3...; PREREG_V8F_SPEEDSUB.md:3,3,9,12...; PREREG_V8F_SPEEDSUB_T1.md:3,5,6,29...; PREREG_V8F_LAUNCH_INTENT.md:3,3,3,3...; PREREG_V8F_DAG_ORC.md:3,3,37,60...; RESULT_V8F_DAG_ORC.md:3,3,3,3...; GOALS_AND_CLAIMS.md (register rows):2,3,3,3...; SPEEDSUB_T1_READ.json:18,5174,5841,6507...; SPEEDSUB_T1_BESIDE.json:3; REP_S0_SPEED_BIAS.md:3,46; FINDING_1_ATTRIBUTION.md:3,46; launch-intent-flag/RESULTS.md:3; launch-intent-flag/MM_REDESIGN_2026-10-28.md:1,43; launch-intent-flag/f_measure.json:55,81,87; launch-intent-v2/NOTES.md:3,3,14,21...; launch-intent-v2/enumerate_cells_li.out:68; launch-intent-chain/HANDOFF.md:1; launch-intent-chain/NOTES.md:1,6,14,14...; launch-intent-chain/PINS_FILLED.txt:1,1,4; launch-intent-chain/LAUNCH.txt:2,13,14,39...; DAGORC_READ.json:47,263,3811,3901...; P0_RF0.json (rebase):12; git log (cited commits):1,1,1,4...; RESULT_V8F_LAUNCH_INTENT.md:3,3,3,6...; LAUNCH_INTENT_READ.json:4776,4903,5029,5076...; MLAUNCH.json:437,859,1581,1586...; MLAUNCH NOTES.md:33,33,35,37...; chain_v8f_launchintent.out:1,2,2,5...; R2C_NOTE.md:2,4; lit1_logs/chain.out:1,2,3,4...; lit1_logs/chain.out.prev_1028151251:1,2,3,4...; lit1_logs/launch_intent_read_REFUSED.out:12,13,23; check_result_filled.out:52,103,104; REP launch-gate tokens (s0, s1):5,27,28,40...; REP run records (s0, s1):5 |
| `28` | 3,1702,1702,1815,1970... | RESULT_V8F_SPEEDSUB.md:3,316,322,323...; RESULT_V8F_SPEEDSUB_T1.md:3,3,3,3...; PREREG_V8F_SPEEDSUB.md:3,3,3,3...; PREREG_V8F_SPEEDSUB_T1.md:3,5,194,196...; PREREG_V8F_LAUNCH_INTENT.md:3,3,3,3...; PREREG_V8F_DAG_ORC.md:432,434,473,497...; RESULT_V8F_DAG_ORC.md:3,3,3,3...; GOALS_AND_CLAIMS.md (register rows):3,3,3,3...; SPEEDSUB_T1_READ.json:7070,7070; REP_S0_SPEED_BIAS.md:3,46; FINDING_1_ATTRIBUTION.md:3; launch-intent-flag/RESULTS.md:3,34; launch-intent-flag/MM_REDESIGN_2026-10-28.md:1; launch-intent-v2/NOTES.md:3,3; launch-intent-chain/HANDOFF.md:1; launch-intent-chain/NOTES.md:1,6,28,47...; launch-intent-chain/PINS_FILLED.txt:1,4; launch-intent-chain/LAUNCH.txt:2,13,14,82...; DAGORC_READ.json:4696,4703,4726,4733...; git log (cited commits):1,1,1,4...; RESULT_V8F_LAUNCH_INTENT.md:3,3,3,378...; LAUNCH_INTENT_READ.json:6272; MLAUNCH.json:14; MLAUNCH NOTES.md:32,36; chain_v8f_launchintent.out:1,2,2,5...; R2C_NOTE.md:2,4; lit1_logs/chain.out:1,2,3,4...; lit1_logs/chain.out.prev_1028151251:1,2,3,4...; read_A1/read_A1.out:16; check_result_filled.out:136,136,136,136 |
| `20260803` | 3 | launch-intent-chain/LAUNCH.txt:18,43 |
| `8.3` | 3,2206,2541 | RESULT_V8F_SPEEDSUB_T1.md:326; RESULT_V8F_LAUNCH_INTENT.md:92 |
| `1.58926` | 3,46,2261,2263,2277... | PREREG_V8F_LAUNCH_INTENT.md:86,86,113,115...; GOALS_AND_CLAIMS.md (register rows):6; launch-intent-flag/RESULTS.md:43; launch-intent-flag/MM_REDESIGN_2026-10-28.md:42,43; launch-intent-flag/f_measure.json:31; launch-intent-v2/enumerate_cells_li.out:48; launch-intent-chain/NOTES.md:8,19; git log (cited commits):10; RESULT_V8F_LAUNCH_INTENT.md:1,8,326,327...; chain_v8f_launchintent.out:18,19,24,25...; lit1_logs/chain.out:49; check_result_filled.out:131 |
| `-0.1290` | 43,1766,1783,1791,1837... | RESULT_V8F_SPEEDSUB.md:15,34,37,46...; RESULT_V8F_SPEEDSUB_T1.md:355; PREREG_V8F_SPEEDSUB_T1.md:13; GOALS_AND_CLAIMS.md (register rows):4; REP_S0_SPEED_BIAS.md:26; git log (cited commits):31 |
| `-0.2107` | 43,1766,1770,1837,2877... | RESULT_V8F_SPEEDSUB.md:15,19,87,159; RESULT_V8F_SPEEDSUB_T1.md:355; PREREG_V8F_SPEEDSUB_T1.md:13; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:48; git log (cited commits):31 |
| `-0.0548` | 43,1766,1768,1783,1790... | RESULT_V8F_SPEEDSUB.md:15,17,34,44...; RESULT_V8F_SPEEDSUB_T1.md:355; PREREG_V8F_SPEEDSUB_T1.md:13; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:49; git log (cited commits):31 |
| `-0.0686` | 43,1766,1783,1789,1791... | RESULT_V8F_SPEEDSUB.md:15,34,37,44...; RESULT_V8F_SPEEDSUB_T1.md:355; PREREG_V8F_SPEEDSUB_T1.md:13; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:66; git log (cited commits):31 |
| `-0.1151` | 43,1766,1770,1837,2877... | RESULT_V8F_SPEEDSUB.md:15,19,87,160; RESULT_V8F_SPEEDSUB_T1.md:355; PREREG_V8F_SPEEDSUB_T1.md:13; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:67; git log (cited commits):31 |
| `-0.0232` | 43,1766,1768,1783,1790... | RESULT_V8F_SPEEDSUB.md:15,17,34,44...; RESULT_V8F_SPEEDSUB_T1.md:355; PREREG_V8F_SPEEDSUB_T1.md:13; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:68; git log (cited commits):31 |
| `-0.06035` | 43,1769,1783,1786,1789... | RESULT_V8F_SPEEDSUB.md:18,34,44,44; RESULT_V8F_SPEEDSUB_T1.md:7,21,27,54...; PREREG_V8F_SPEEDSUB.md:39,108,113; PREREG_V8F_SPEEDSUB_T1.md:13,111; GOALS_AND_CLAIMS.md (register rows):4,4,5; git log (cited commits):22,31 |
| `+6.7181` | 43,1771,2877,2958 | RESULT_V8F_SPEEDSUB.md:20,26; GOALS_AND_CLAIMS.md (register rows):4; git log (cited commits):31 |
| `+6.4923` | 43,1771,2877,2958 | RESULT_V8F_SPEEDSUB.md:20; GOALS_AND_CLAIMS.md (register rows):4; git log (cited commits):31 |
| `2000` | 43,44,47,51,1754... | RESULT_V8F_SPEEDSUB.md:6,37,79,206; RESULT_V8F_SPEEDSUB_T1.md:14,81; PREREG_V8F_SPEEDSUB.md:72,155; PREREG_V8F_SPEEDSUB_T1.md:80; PREREG_V8F_LAUNCH_INTENT.md:143,221; PREREG_V8F_DAG_ORC.md:22,100,290; RESULT_V8F_DAG_ORC.md:6,147; GOALS_AND_CLAIMS.md (register rows):3,3,4,4...; SUBREAD.json:87; SPEEDSUB_T1_READ.json:19,5175,5842,6508...; REP_S0_SPEED_BIAS.md:9; DAGORC_READ.json:3812,3902,3992,4026...; P0_RF0.json (rebase):13; RESULT_V8F_LAUNCH_INTENT.md:7,21,44,161...; LAUNCH_INTENT_READ.json:4777,4904,5030,5077...; MLAUNCH.json:7,1708; MLAUNCH NOTES.md:4 |
| `141` | 43,44,47,1750,1752... | RESULT_V8F_SPEEDSUB.md:6,43,43,68...; RESULT_V8F_SPEEDSUB_T1.md:14,14,72,77...; PREREG_V8F_SPEEDSUB.md:14,24,26,72...; PREREG_V8F_SPEEDSUB_T1.md:23,52,81,182; PREREG_V8F_LAUNCH_INTENT.md:128,139,141,143; PREREG_V8F_DAG_ORC.md:96,98,100; RESULT_V8F_DAG_ORC.md:5,6,43,44...; GOALS_AND_CLAIMS.md (register rows):3,3,4,4...; SPEEDSUB_T1_READ.json:19,209,246,284...; REP_S0_SPEED_BIAS.md:7,9; FINDING_1_ATTRIBUTION.md:29,29,33,34...; launch-intent-v2/NOTES.md:50,50; DAGORC_READ.json:585,599,613,627...; P0_RF0.json (rebase):13,26,33,49...; git log (cited commits):43; RESULT_V8F_LAUNCH_INTENT.md:6,7,21,44...; LAUNCH_INTENT_READ.json:1011,1025,1039,1053...; MLAUNCH NOTES.md:48; check_result_filled.out:41 |
| `0.28032` | 43,1822,1825,2877 | RESULT_V8F_SPEEDSUB.md:248,251; GOALS_AND_CLAIMS.md (register rows):4; FINDING_1_ATTRIBUTION.md:14; git log (cited commits):31 |
| `0.24751` | 43,1820,2877 | RESULT_V8F_SPEEDSUB.md:246; PREREG_V8F_SPEEDSUB.md:14; GOALS_AND_CLAIMS.md (register rows):4; FINDING_1_ATTRIBUTION.md:12; git log (cited commits):31 |
| `-0.0233` | 44,1894,2000,2059,2062... | RESULT_V8F_SPEEDSUB.md:144; RESULT_V8F_SPEEDSUB_T1.md:8,25,85,88...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:849,924,985,1590...; git log (cited commits):22,25 |
| `-0.0629` | 44,2000,2031,2059,2195... | RESULT_V8F_SPEEDSUB_T1.md:8,25,58,85...; RESULT_V8F_DAG_ORC.md:84,101,156,166; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:850,925; DAGORC_READ.json:1203,2136,2674,3892...; git log (cited commits):22,25 |
| `+0.0156` | 44,1821,2000,2059,2195... | RESULT_V8F_SPEEDSUB.md:247; RESULT_V8F_SPEEDSUB_T1.md:8,25,85,311...; GOALS_AND_CLAIMS.md (register rows):5; FINDING_1_ATTRIBUTION.md:22; git log (cited commits):22,25 |
| `-0.0076` | 44,2000,2059,2068,2196... | RESULT_V8F_SPEEDSUB_T1.md:8,25,85,94...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:855,1136,1226,6080; git log (cited commits):22,25 |
| `-0.0500` | 44,2000,2031,2059,2196... | RESULT_V8F_SPEEDSUB_T1.md:8,25,58,85...; GOALS_AND_CLAIMS.md (register rows):5; git log (cited commits):22,25 |
| `+0.0349` | 44,2000,2059,2196,2217 | RESULT_V8F_SPEEDSUB_T1.md:8,25,85,312...; GOALS_AND_CLAIMS.md (register rows):5; git log (cited commits):22,25 |
| `-0.0333` | 44,2000,2059,2197,2878... | RESULT_V8F_SPEEDSUB_T1.md:8,25,85,313...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:867,1560; git log (cited commits):22,25 |
| `-0.0832` | 44,2000,2031,2059,2197 | RESULT_V8F_SPEEDSUB_T1.md:8,25,58,85...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:868,1561; git log (cited commits):22,25 |
| `+0.0131` | 44,2000,2059,2197 | RESULT_V8F_SPEEDSUB_T1.md:8,25,85,313; GOALS_AND_CLAIMS.md (register rows):5; git log (cited commits):22,25 |
| `-0.1949` | 44,2000,2059,2149,2156... | RESULT_V8F_SPEEDSUB_T1.md:7,25,85,264...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:861,1348; git log (cited commits):22,25 |
| `-0.2619` | 44,2000,2059,2149,2156... | RESULT_V8F_SPEEDSUB_T1.md:7,25,85,264...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:862,1349; git log (cited commits):22,25 |
| `-0.1313` | 44,2000,2059,2149,2156... | RESULT_V8F_SPEEDSUB_T1.md:7,25,85,264...; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:863,1350; git log (cited commits):22,25 |
| `0.73` | 44 | RESULT_V8F_SPEEDSUB_T1.md:59,283; PREREG_V8F_SPEEDSUB_T1.md:15; PREREG_V8F_LAUNCH_INTENT.md:27; GOALS_AND_CLAIMS.md (register rows):5,5; REP_S0_SPEED_BIAS.md:39; launch-intent-flag/MM_REDESIGN_2026-10-28.md:8; git log (cited commits):34 |
| `1.17` | 44 | RESULT_V8F_SPEEDSUB_T1.md:59,283; PREREG_V8F_SPEEDSUB_T1.md:15; PREREG_V8F_LAUNCH_INTENT.md:27; GOALS_AND_CLAIMS.md (register rows):5,5; REP_S0_SPEED_BIAS.md:39; launch-intent-flag/MM_REDESIGN_2026-10-28.md:8; git log (cited commits):34 |
| `158` | 44,1994,2012,2012,2959 | RESULT_V8F_SPEEDSUB.md:224; RESULT_V8F_SPEEDSUB_T1.md:3,37,37; GOALS_AND_CLAIMS.md (register rows):5; git log (cited commits):22; MLAUNCH NOTES.md:37,38 |
| `+0.0008` | 44,1956,2211,2217,2541... | RESULT_V8F_SPEEDSUB.md:213; RESULT_V8F_SPEEDSUB_T1.md:10,338,351,371; GOALS_AND_CLAIMS.md (register rows):3,5; git log (cited commits):22; RESULT_V8F_LAUNCH_INTENT.md:92 |
| `-0.0185` | 44,2211,2217,2630,2878 | RESULT_V8F_SPEEDSUB_T1.md:10,338,351,371; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_BESIDE.json:181; git log (cited commits):22; RESULT_V8F_LAUNCH_INTENT.md:177; LAUNCH_INTENT_READ.json:3187,4801 |
| `12` | 45,1823,1993,2041,2041... | RESULT_V8F_SPEEDSUB.md:157,249,310,312; RESULT_V8F_SPEEDSUB_T1.md:3,15,68,68...; PREREG_V8F_SPEEDSUB.md:148; PREREG_V8F_LAUNCH_INTENT.md:7,47,49,49...; PREREG_V8F_DAG_ORC.md:300,317,317; RESULT_V8F_DAG_ORC.md:260,265,306; GOALS_AND_CLAIMS.md (register rows):6; FINDING_1_ATTRIBUTION.md:15,72; launch-intent-flag/RESULTS.md:30,49; launch-intent-flag/MM_REDESIGN_2026-10-28.md:1; launch-intent-v2/NOTES.md:10,12; launch-intent-chain/HANDOFF.md:4; launch-intent-chain/NOTES.md:37,54,55; git log (cited commits):16,19,25; RESULT_V8F_LAUNCH_INTENT.md:311,408,427; MLAUNCH NOTES.md:25,25,25; lit1_logs/chain.out:1,2,3,4...; lit1_logs/chain.out.prev_1028151251:12; lit1_logs/launch_intent_read_REFUSED.out:3,14,24,25; check_result_filled.out:85,94,136,137 |
| `174` | 46,2394,2481,2805,2819 | launch-intent-flag/f_measure.json:38; RESULT_V8F_LAUNCH_INTENT.md:363,443; chain_v8f_launchintent.out:18,19,24,25...; check_result_filled.out:67,100,118,120 |
| `206` | 47,2263,2412,2429,2960 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:3,25; check_result_filled.out:5,108 |
| `16` | 47,48,2393,2393,2394... | RESULT_V8F_SPEEDSUB_T1.md:432; PREREG_V8F_LAUNCH_INTENT.md:282; PREREG_V8F_DAG_ORC.md:252,438,488; RESULT_V8F_DAG_ORC.md:259,261; launch-intent-flag/RESULTS.md:31,32; launch-intent-v2/NOTES.md:10; launch-intent-v2/enumerate_cells_li.out:12,12,14,18...; git log (cited commits):22,22; RESULT_V8F_LAUNCH_INTENT.md:14,15,106,131...; MLAUNCH NOTES.md:34,34; chain_v8f_launchintent.out:78,79,80,81...; R2C_NOTE.md:7; lit1_logs/chain.out:15,16,30,31...; lit1_logs/launch_intent_read_REFUSED.out:2,3,4,5...; check_result_filled.out:8,100,110,111...; REP run records (s0, s1):31,91,184,244 |
| `-0.0215` | 48,2533,2544,2645,2660... | RESULT_V8F_DAG_ORC.md:70,151; GOALS_AND_CLAIMS.md (register rows):6; DAGORC_READ.json:1940,2478,4083; RESULT_V8F_LAUNCH_INTENT.md:14,95,194,210...; LAUNCH_INTENT_READ.json:3385,4403,5103,5378 |
| `-0.0423` | 48,2533,2544,2645,2781... | RESULT_V8F_LAUNCH_INTENT.md:14,95,194,348; LAUNCH_INTENT_READ.json:4404,5379 |
| `-0.0048` | 48,1848,2533,2544,2645... | RESULT_V8F_SPEEDSUB.md:98; P0_RF0.json (rebase):449; RESULT_V8F_LAUNCH_INTENT.md:14,95,194,348; LAUNCH_INTENT_READ.json:4405,5380 |
| `0.0136` | 48,2286,2310,2517,2533... | PREREG_V8F_LAUNCH_INTENT.md:169,194,346; GOALS_AND_CLAIMS.md (register rows):6,6; SPEEDSUB_T1_READ.json:2028,5579; SPEEDSUB_T1_BESIDE.json:172; launch-intent-flag/MM_REDESIGN_2026-10-28.md:50; launch-intent-v2/NOTES.md:16; launch-intent-v2/enumerate_cells_li.out:67; RESULT_V8F_LAUNCH_INTENT.md:14,64,91,95...; LAUNCH_INTENT_READ.json:765,840 |
| `+0.0431` | 49,2415,2420,2587,2608... | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:13,28,33,77...; check_result_filled.out:109 |
| `+0.0137` | 49,2079,2587,2772,2882 | RESULT_V8F_SPEEDSUB_T1.md:105,256; GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:13,77,339; check_result_filled.out:109 |
| `+0.0711` | 49,2587,2772,2882 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:13,77,339; check_result_filled.out:109 |
| `+0.0483` | 49,2415,2420,2588,2609... | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:13,28,33,78... |
| `+0.0105` | 49,2588,2773,2882 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:13,78,340 |
| `+0.0884` | 49,2588,2773,2882 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:13,78,340 |
| `-0.0294` | 50,2415,2420,2586,2607... | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:28,33,76,152...; LAUNCH_INTENT_READ.json:3159 |
| `-0.0522` | 50,1894,2586,2772,2882 | RESULT_V8F_SPEEDSUB.md:144; GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:76,339; LAUNCH_INTENT_READ.json:3160 |
| `-0.0077` | 50,2586,2772,2882 | PREREG_V8F_DAG_ORC.md:26; RESULT_V8F_DAG_ORC.md:101,156; GOALS_AND_CLAIMS.md (register rows):6; DAGORC_READ.json:1205,3894; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:76,339; LAUNCH_INTENT_READ.json:3161 |
| `-0.0341` | 50,2415,2420,2589,2610... | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:28,33,79,155...; LAUNCH_INTENT_READ.json:4503 |
| `-0.0655` | 50,2589,2773,2883 | RESULT_V8F_SPEEDSUB_T1.md:168; GOALS_AND_CLAIMS.md (register rows):6; SPEEDSUB_T1_READ.json:3887; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:79,340; LAUNCH_INTENT_READ.json:4504 |
| `-0.0027` | 50,2589,2773,2883 | RESULT_V8F_DAG_ORC.md:70,151; GOALS_AND_CLAIMS.md (register rows):6; DAGORC_READ.json:1672,2210,3817; git log (cited commits):4; RESULT_V8F_LAUNCH_INTENT.md:79,340; LAUNCH_INTENT_READ.json:4505 |
| `49` | 52,1921,1921,2738,2760... | RESULT_V8F_SPEEDSUB.md:157,159,160,167; RESULT_V8F_SPEEDSUB_T1.md:3,425,427,428...; PREREG_V8F_SPEEDSUB.md:155; PREREG_V8F_LAUNCH_INTENT.md:33,309; PREREG_V8F_DAG_ORC.md:270; GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):13,16; RESULT_V8F_LAUNCH_INTENT.md:3,378,430; MLAUNCH.json:7,16,20,29...; MLAUNCH NOTES.md:4,8,48,50...; check_result_filled.out:92 |
| `327` | 52,2738,2760,2797,2884 | PREREG_V8F_LAUNCH_INTENT.md:309; GOALS_AND_CLAIMS.md (register rows):6; RESULT_V8F_LAUNCH_INTENT.md:378; MLAUNCH.json:15,19,28,130...; MLAUNCH NOTES.md:8,43,49,50... |
| `+0.0348` | 52,2744,2752,2787 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:382; MLAUNCH NOTES.md:12,57 |
| `-0.0299` | 52,2744 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:382; MLAUNCH.json:50,1251; MLAUNCH NOTES.md:12 |
| `+0.0923` | 52,2577,2726,2744 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:129,286,382; MLAUNCH NOTES.md:12 |
| `+0.0282` | 52,2745 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:383; MLAUNCH NOTES.md:13 |
| `-0.0247` | 52,2219,2745 | RESULT_V8F_SPEEDSUB_T1.md:340; GOALS_AND_CLAIMS.md (register rows):6; SPEEDSUB_T1_BESIDE.json:199; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:383; MLAUNCH.json:152,1261; MLAUNCH NOTES.md:13 |
| `+0.0836` | 52,2745 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:383; MLAUNCH NOTES.md:13 |
| `+0.0243` | 52,2746 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:384; MLAUNCH NOTES.md:14 |
| `-0.0281` | 52,2061,2219,2746 | RESULT_V8F_SPEEDSUB_T1.md:87,340; GOALS_AND_CLAIMS.md (register rows):6; SPEEDSUB_T1_READ.json:944,5290; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:384; MLAUNCH.json:254,1271; MLAUNCH NOTES.md:14 |
| `+0.0846` | 52,2746 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:384; MLAUNCH NOTES.md:14 |
| `-0.0146` | 52,2747 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:385; MLAUNCH.json:355,1236,1279; MLAUNCH NOTES.md:15 |
| `-0.0607` | 53,2747 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:385; MLAUNCH.json:356,1281; MLAUNCH NOTES.md:15 |
| `+0.0286` | 53,2747 | GOALS_AND_CLAIMS.md (register rows):6; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:385; MLAUNCH NOTES.md:15 |
| `11` | 53,56,1921,1993,1995... | RESULT_V8F_SPEEDSUB.md:43,51,159,264...; RESULT_V8F_SPEEDSUB_T1.md:3,3,14,426...; PREREG_V8F_SPEEDSUB.md:178; PREREG_V8F_SPEEDSUB_T1.md:29; PREREG_V8F_LAUNCH_INTENT.md:24,24,24,28...; PREREG_V8F_DAG_ORC.md:60,207,252,361; RESULT_V8F_DAG_ORC.md:7,215,289,300; GOALS_AND_CLAIMS.md (register rows):3,3,3,6...; launch-intent-flag/RESULTS.md:3; launch-intent-flag/f_measure.json:62,82,88; launch-intent-v2/NOTES.md:21,23; launch-intent-v2/enumerate_cells_li.out:13,14,68; launch-intent-chain/HANDOFF.md:4; launch-intent-chain/NOTES.md:15,32,54,54; launch-intent-chain/LAUNCH.txt:61; DAGORC_READ.json:4701; git log (cited commits):1,1,4,10...; RESULT_V8F_LAUNCH_INTENT.md:3,8,17,316...; MLAUNCH.json:2,3,23; MLAUNCH NOTES.md:1,17,17,47...; chain_v8f_launchintent.out:18,19,30,31...; lit1_logs/chain.out:24,33,34; lit1_logs/launch_intent_read_REFUSED.out:32; check_result_filled.out:93,94,94,134; REP launch-gate tokens (s0, s1):86; REP run records (s0, s1):158 |
| `0.726175` | 56,1702,2277,2327,2450... | PREREG_V8F_LAUNCH_INTENT.md:22,87,101,117...; GOALS_AND_CLAIMS.md (register rows):6,6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:68; launch-intent-v2/enumerate_cells_li.out:48; launch-intent-chain/NOTES.md:50; RESULT_V8F_LAUNCH_INTENT.md:8,17,361,395 |
| `4.17852` | 56,2277,2375,2749,2804... | PREREG_V8F_LAUNCH_INTENT.md:24,88,101,295...; GOALS_AND_CLAIMS.md (register rows):6,6; launch-intent-flag/RESULTS.md:43; launch-intent-v2/enumerate_cells_li.out:48; launch-intent-chain/NOTES.md:50; RESULT_V8F_LAUNCH_INTENT.md:374,387,398; MLAUNCH.json:437,859,1586; MLAUNCH NOTES.md:17 |
| `0.1` | 1702,1702,2269,2323,2801... | RESULT_V8F_SPEEDSUB.md:371; PREREG_V8F_LAUNCH_INTENT.md:113,126; PREREG_V8F_DAG_ORC.md:49,51,52,61...; RESULT_V8F_DAG_ORC.md:7,27,251,295...; GOALS_AND_CLAIMS.md (register rows):3,3; DAGORC_READ.json:45,52,261; git log (cited commits):16,37; REP launch-gate tokens (s0, s1):33,79; REP run records (s0, s1):47,86,120,200... |
| `0.06035` | 1743 | PREREG_V8F_SPEEDSUB.md:31,37,38,38; PREREG_V8F_SPEEDSUB_T1.md:28; GOALS_AND_CLAIMS.md (register rows):4,5; SUBREAD.json:8; SPEEDSUB_T1_READ.json:873; check_result_filled.out:7 |
| `2399` | 1746,1746,1754,1807,1807... | RESULT_V8F_SPEEDSUB.md:6,7,7,65...; RESULT_V8F_SPEEDSUB_T1.md:14,14,15,15...; PREREG_V8F_SPEEDSUB.md:67,67,72,92...; PREREG_V8F_SPEEDSUB_T1.md:15,52,71,71...; PREREG_V8F_LAUNCH_INTENT.md:143; PREREG_V8F_DAG_ORC.md:100; RESULT_V8F_DAG_ORC.md:6,43,44,45...; GOALS_AND_CLAIMS.md (register rows):4; SPEEDSUB_T1_READ.json:19,206,207,209...; REP_S0_SPEED_BIAS.md:9; launch-intent-v2/NOTES.md:50; DAGORC_READ.json:585,599,613,627...; P0_RF0.json (rebase):13,26,33,49...; RESULT_V8F_LAUNCH_INTENT.md:7,76,77,78...; LAUNCH_INTENT_READ.json:1011,1025,1039,1053...; MLAUNCH.json:17; MLAUNCH NOTES.md:33,55; lit1_logs/launch_intent_read_REFUSED.out:2,5,8,11...; read_A1/read_A1.out:3,6,9,12...; check_result_filled.out:41 |
| `20261029` | 1746 | RESULT_V8F_SPEEDSUB.md:7; RESULT_V8F_SPEEDSUB_T1.md:15; PREREG_V8F_SPEEDSUB.md:67; PREREG_V8F_SPEEDSUB_T1.md:71 |
| `22` | 1748,1820,2442,2477,2477 | RESULT_V8F_SPEEDSUB.md:47,242,246,253...; RESULT_V8F_SPEEDSUB_T1.md:420; PREREG_V8F_SPEEDSUB.md:44,44,123,174...; PREREG_V8F_SPEEDSUB_T1.md:42,42; PREREG_V8F_DAG_ORC.md:193; GOALS_AND_CLAIMS.md (register rows):3; FINDING_1_ATTRIBUTION.md:12,12,46,65...; launch-intent-flag/RESULTS.md:35; git log (cited commits):13; RESULT_V8F_LAUNCH_INTENT.md:420,448,448; chain_v8f_launchintent.out:59,60,61,62; lit1_logs/chain.out:19,21,25; read_A1/read_A1.out:11,12 |
| `0.2475` | 1751,2233 | RESULT_V8F_SPEEDSUB.md:43; RESULT_V8F_SPEEDSUB_T1.md:359,371; PREREG_V8F_SPEEDSUB.md:27; PREREG_V8F_SPEEDSUB_T1.md:16 |
| `0.2724` | 1751,1826 | RESULT_V8F_SPEEDSUB.md:43,251; PREREG_V8F_SPEEDSUB.md:27 |
| `0.2807` | 1751 | RESULT_V8F_SPEEDSUB.md:43 |
| `0.2920` | 1751 | RESULT_V8F_SPEEDSUB.md:43 |
| `0.1207` | 1755,2298,2498,2505 | RESULT_V8F_SPEEDSUB.md:26; PREREG_V8F_SPEEDSUB.md:37,37; PREREG_V8F_LAUNCH_INTENT.md:151,182,344; PREREG_V8F_DAG_ORC.md:125; RESULT_V8F_DAG_ORC.md:39,43,234; SUBREAD.json:7; launch-intent-flag/MM_REDESIGN_2026-10-28.md:48; DAGORC_READ.json:530; P0_RF0.json (rebase):38; RESULT_V8F_LAUNCH_INTENT.md:48,52; LAUNCH_INTENT_READ.json:732; check_result_filled.out:123 |
| `0.9702` | 1758 | RESULT_V8F_SPEEDSUB.md:8; RESULT_V8F_SPEEDSUB_T1.md:357; GOALS_AND_CLAIMS.md (register rows):2 |
| `08` | 1761,1761,1970,1970 | RESULT_V8F_SPEEDSUB.md:3,3,3,3...; RESULT_V8F_SPEEDSUB_T1.md:3; PREREG_V8F_SPEEDSUB.md:54; PREREG_V8F_SPEEDSUB_T1.md:5; PREREG_V8F_LAUNCH_INTENT.md:55; git log (cited commits):34,37 |
| `05` | 1761,1761,2332 | RESULT_V8F_SPEEDSUB.md:3,3,310,311...; RESULT_V8F_SPEEDSUB_T1.md:422,434; PREREG_V8F_SPEEDSUB_T1.md:5; RESULT_V8F_DAG_ORC.md:261,261,263,273...; launch-intent-chain/HANDOFF.md:8; launch-intent-chain/NOTES.md:28,47; launch-intent-chain/LAUNCH.txt:14,82,88; git log (cited commits):31; R2C_NOTE.md:7,12,12,34...; lit1_logs/chain.out.prev_1028151251:18; read_A1/read_A1.out:32,33,34,35... |
| `36` | 1761,2444,2491 | RESULT_V8F_SPEEDSUB.md:3,310,311; RESULT_V8F_SPEEDSUB_T1.md:431,434; PREREG_V8F_LAUNCH_INTENT.md:33; PREREG_V8F_DAG_ORC.md:270,300,479; RESULT_V8F_DAG_ORC.md:234,261,349; launch-intent-flag/RESULTS.md:3,10,10,11; DAGORC_READ.json:4700,4700; git log (cited commits):19,19,19; RESULT_V8F_LAUNCH_INTENT.md:422,461; MLAUNCH NOTES.md:35; lit1_logs/chain.out:53; lit1_logs/launch_intent_read_REFUSED.out:4,5; check_result_filled.out:136 |
| `38` | 1761,1799 | RESULT_V8F_SPEEDSUB.md:3,57,311,312; RESULT_V8F_SPEEDSUB_T1.md:408,408,422,423...; RESULT_V8F_DAG_ORC.md:273; launch-intent-flag/RESULTS.md:37; git log (cited commits):28,28,40; lit1_logs/chain.out:36; lit1_logs/launch_intent_read_REFUSED.out:34 |
| `27` | 1761,1781,1793,2256,2478... | RESULT_V8F_SPEEDSUB.md:3,32,51,373; RESULT_V8F_SPEEDSUB_T1.md:365,416,424; PREREG_V8F_SPEEDSUB.md:3,131,148,168; PREREG_V8F_LAUNCH_INTENT.md:152; PREREG_V8F_DAG_ORC.md:3,3,224,338...; RESULT_V8F_DAG_ORC.md:3,260,271; GOALS_AND_CLAIMS.md (register rows):3,3,3,3...; launch-intent-v2/NOTES.md:63; DAGORC_READ.json:47,263,4701,4701...; git log (cited commits):19,28,31,43...; RESULT_V8F_LAUNCH_INTENT.md:449; lit1_logs/chain.out:22,30; read_A1/read_A1.out:34; REP launch-gate tokens (s0, s1):5,51 |
| `18` | 1762,1791,1970,2081,2085... | RESULT_V8F_SPEEDSUB.md:3,37,46; RESULT_V8F_SPEEDSUB_T1.md:135,139,142,264...; PREREG_V8F_SPEEDSUB_T1.md:42; RESULT_V8F_DAG_ORC.md:3,275,275; GOALS_AND_CLAIMS.md (register rows):5; FINDING_1_ATTRIBUTION.md:65; launch-intent-flag/RESULTS.md:10; git log (cited commits):19,28,28,37; chain_v8f_launchintent.out:6,117; lit1_logs/chain.out:37 |
| `2.1977` | 1767,1772,1773,1797,1797 | RESULT_V8F_SPEEDSUB.md:16,21,22,55... |
| `2.0687` | 1767 | RESULT_V8F_SPEEDSUB.md:16 |
| `2.1201` | 1767,1772,1773,1797,1797 | RESULT_V8F_SPEEDSUB.md:16,21,22,55... |
| `2.0515` | 1767 | RESULT_V8F_SPEEDSUB.md:16 |
| `0.0687` | 1769 | RESULT_V8F_SPEEDSUB.md:18 |
| `0.0083` | 1769,1789 | RESULT_V8F_SPEEDSUB.md:18,44,45; GOALS_AND_CLAIMS.md (register rows):4; DAGORC_READ.json:2108,2646,4263; git log (cited commits):31; LAUNCH_INTENT_READ.json:4489 |
| `+5.6709` | 1771,1781 | RESULT_V8F_SPEEDSUB.md:20,32; GOALS_AND_CLAIMS.md (register rows):4 |
| `+7.8141` | 1771 | RESULT_V8F_SPEEDSUB.md:20; GOALS_AND_CLAIMS.md (register rows):4 |
| `+5.3813` | 1771,1781 | RESULT_V8F_SPEEDSUB.md:20,32; GOALS_AND_CLAIMS.md (register rows):4 |
| `+7.6678` | 1771 | RESULT_V8F_SPEEDSUB.md:20; GOALS_AND_CLAIMS.md (register rows):4 |
| `8.9157` | 1772 | RESULT_V8F_SPEEDSUB.md:21 |
| `8.6124` | 1772 | RESULT_V8F_SPEEDSUB.md:21 |
| `-0.7164` | 1773,1803,1873 | RESULT_V8F_SPEEDSUB.md:22,61,123; PREREG_V8F_SPEEDSUB.md:9; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:57; git log (cited commits):31 |
| `-0.8182` | 1773,1803,1873 | RESULT_V8F_SPEEDSUB.md:22,61,123; PREREG_V8F_SPEEDSUB.md:9; SUBREAD.json:58 |
| `-0.6234` | 1773,1803,1873 | RESULT_V8F_SPEEDSUB.md:22,61,123; PREREG_V8F_SPEEDSUB.md:9; SUBREAD.json:59 |
| `1.4813` | 1773 | RESULT_V8F_SPEEDSUB.md:22 |
| `-0.6676` | 1773,1803,1873 | RESULT_V8F_SPEEDSUB.md:22,61,123; PREREG_V8F_SPEEDSUB.md:9; GOALS_AND_CLAIMS.md (register rows):4; SUBREAD.json:76; git log (cited commits):31 |
| `-0.7613` | 1773,1803,1873 | RESULT_V8F_SPEEDSUB.md:22,61,123; PREREG_V8F_SPEEDSUB.md:9; SUBREAD.json:77 |
| `-0.5753` | 1773,1803,1873 | RESULT_V8F_SPEEDSUB.md:22,61,123; PREREG_V8F_SPEEDSUB.md:9; SUBREAD.json:78 |
| `1.4526` | 1773 | RESULT_V8F_SPEEDSUB.md:22 |
| `0.1801` | 1774,1792 | RESULT_V8F_SPEEDSUB.md:23,46; GOALS_AND_CLAIMS.md (register rows):4; git log (cited commits):31 |
| `0.1028` | 1774,1792 | RESULT_V8F_SPEEDSUB.md:23,46; GOALS_AND_CLAIMS.md (register rows):4; git log (cited commits):31 |
| `8.4` | 1775 | RESULT_V8F_SPEEDSUB.md:24; PREREG_V8F_SPEEDSUB.md:38 |
| `9.0` | 1775 | RESULT_V8F_SPEEDSUB.md:24; PREREG_V8F_SPEEDSUB.md:38 |
| `0.0001` | 1787 | RESULT_V8F_SPEEDSUB.md:26; SPEEDSUB_T1_READ.json:297,415,533,651; P0_RF0.json (rebase):552,564; LAUNCH_INTENT_READ.json:4645,6082; REP run records (s0, s1):139,292 |
| `0.0604` | 1790,2287,2340,2349,2766 | RESULT_V8F_SPEEDSUB.md:46; PREREG_V8F_SPEEDSUB.md:37; PREREG_V8F_LAUNCH_INTENT.md:170,212,259; PREREG_V8F_DAG_ORC.md:160,162,391; SPEEDSUB_T1_READ.json:2338,3016,6703; launch-intent-flag/MM_REDESIGN_2026-10-28.md:24; launch-intent-v2/NOTES.md:12; DAGORC_READ.json:3253,3791; RESULT_V8F_LAUNCH_INTENT.md:355; LAUNCH_INTENT_READ.json:2065; check_result_filled.out:7,116 |
| `0.7` | 1792,2565 | RESULT_V8F_SPEEDSUB.md:46; PREREG_V8F_SPEEDSUB.md:22,129,129; PREREG_V8F_DAG_ORC.md:489; RESULT_V8F_DAG_ORC.md:265; GOALS_AND_CLAIMS.md (register rows):4; RESULT_V8F_LAUNCH_INTENT.md:117; R2C_NOTE.md:37 |
| `0.0` | 1797,1798,1798,1804,1804... | RESULT_V8F_SPEEDSUB.md:55,56,56,62...; RESULT_V8F_SPEEDSUB_T1.md:66,66,67,67...; PREREG_V8F_SPEEDSUB.md:70,82,83,86...; PREREG_V8F_SPEEDSUB_T1.md:89,90; PREREG_V8F_LAUNCH_INTENT.md:21,28,28,49...; PREREG_V8F_DAG_ORC.md:219,382; RESULT_V8F_DAG_ORC.md:296,297,300,301; SUBREAD.json:133,134,135,136...; SPEEDSUB_T1_READ.json:199,200,216,236...; SPEEDSUB_T1_BESIDE.json:44; FINDING_1_ATTRIBUTION.md:63,63; launch-intent-flag/RESULTS.md:14; launch-intent-v2/NOTES.md:14,14,33,33...; launch-intent-chain/NOTES.md:19,45,45; DAGORC_READ.json:531,532,533,534...; RF_REP.json:166; P0_RF0.json (rebase):61,84,107,123...; git log (cited commits):19,19; RESULT_V8F_LAUNCH_INTENT.md:387; LAUNCH_INTENT_READ.json:733,734,735,736...; MLAUNCH.json:927,928,929,934...; MLAUNCH NOTES.md:35,43,54,54...; REP launch-gate tokens (s0, s1):37,83; REP run records (s0, s1):52,59,205,212 |
| `1e-5` | 1801,1801,1804,1805,1981... | RESULT_V8F_SPEEDSUB.md:59,59,62,63; RESULT_V8F_SPEEDSUB_T1.md:69,77,404; PREREG_V8F_SPEEDSUB.md:53,86,89,90; PREREG_V8F_SPEEDSUB_T1.md:50,91,228; FINDING_1_ATTRIBUTION.md:59,65; git log (cited commits):28 |
| `6.63e-06` | 1801 | RESULT_V8F_SPEEDSUB.md:59 |
| `7.79e-06` | 1801 | RESULT_V8F_SPEEDSUB.md:59 |
| `1e-4` | 1802,1814,2039 | RESULT_V8F_SPEEDSUB.md:60,71,72,192; RESULT_V8F_SPEEDSUB_T1.md:66; PREREG_V8F_SPEEDSUB.md:87,97; PREREG_V8F_SPEEDSUB_T1.md:88; PREREG_V8F_DAG_ORC.md:66; MLAUNCH NOTES.md:34 |
| `4.00e-06` | 1802 | RESULT_V8F_SPEEDSUB.md:60 |
| `4.57e-06` | 1802 | RESULT_V8F_SPEEDSUB.md:60 |
| `5e-4` | 1803 | RESULT_V8F_SPEEDSUB.md:61; PREREG_V8F_SPEEDSUB.md:88 |
| `1e-6` | 1807 | RESULT_V8F_SPEEDSUB.md:65; PREREG_V8F_SPEEDSUB.md:92,170 |
| `99` | 1808,2043 | RESULT_V8F_SPEEDSUB.md:66; RESULT_V8F_SPEEDSUB_T1.md:70,415; PREREG_V8F_SPEEDSUB.md:93; PREREG_V8F_SPEEDSUB_T1.md:92,223 |
| `95` | 1808,1833,1869,1998,2032... | RESULT_V8F_SPEEDSUB.md:66,83,119; RESULT_V8F_SPEEDSUB_T1.md:23,58,70,79...; PREREG_V8F_SPEEDSUB.md:93; PREREG_V8F_SPEEDSUB_T1.md:92; PREREG_V8F_DAG_ORC.md:479; RESULT_V8F_DAG_ORC.md:33,263,265; GOALS_AND_CLAIMS.md (register rows):5; REP_S0_SPEED_BIAS.md:18; launch-intent-flag/RESULTS.md:29; RESULT_V8F_LAUNCH_INTENT.md:7,44,380; MLAUNCH NOTES.md:10,10; check_result_filled.out:42 |
| `1.0000` | 1808,1808,1808,1808 | RESULT_V8F_SPEEDSUB.md:66,66,66,66... |
| `4512` | 1809,2052 | RESULT_V8F_SPEEDSUB.md:67; RESULT_V8F_SPEEDSUB_T1.md:77,77; PREREG_V8F_SPEEDSUB.md:94; PREREG_V8F_SPEEDSUB_T1.md:56; PREREG_V8F_LAUNCH_INTENT.md:79,81; FINDING_1_ATTRIBUTION.md:63; launch-intent-flag/f_measure.json:4,15,16; chain_v8f_launchintent.out:18,19,24,25...; REP run records (s0, s1):9,162 |
| `306566` | 1809,2051,2051,2052,2276 | RESULT_V8F_SPEEDSUB.md:67,290; RESULT_V8F_SPEEDSUB_T1.md:77,77,77; PREREG_V8F_SPEEDSUB.md:94; PREREG_V8F_SPEEDSUB_T1.md:56; PREREG_V8F_LAUNCH_INTENT.md:79,80,80,83...; FINDING_1_ATTRIBUTION.md:48,63; launch-intent-flag/RESULTS.md:43; launch-intent-flag/f_measure.json:5,11,12,18...; git log (cited commits):28,28; chain_v8f_launchintent.out:18,19,24,25...; REP run records (s0, s1):10,163 |
| `4.99e-05` | 1814 | RESULT_V8F_SPEEDSUB.md:72 |
| `45` | 1814,2368,2479,2479,2760 | RESULT_V8F_SPEEDSUB.md:71; RESULT_V8F_SPEEDSUB_T1.md:425,426,434; PREREG_V8F_SPEEDSUB.md:148; PREREG_V8F_LAUNCH_INTENT.md:280; PREREG_V8F_DAG_ORC.md:248; launch-intent-v2/enumerate_cells_li.out:8; git log (cited commits):19; RESULT_V8F_LAUNCH_INTENT.md:450,450; MLAUNCH.json:1749,1794,1839,1884; MLAUNCH NOTES.md:68; chain_v8f_launchintent.out:97,98,99,100; lit1_logs/chain.out:45; lit1_logs/chain.out.prev_1028151251:18; lit1_logs/launch_intent_read_REFUSED.out:21,30 |
| `119` | 1814 | RESULT_V8F_SPEEDSUB.md:71 |
| `846` | 1816 | RESULT_V8F_SPEEDSUB.md:242; RESULT_V8F_SPEEDSUB_T1.md:359; FINDING_1_ATTRIBUTION.md:20,63,67 |
| `17` | 1821,1822,1970,1977,2278... | RESULT_V8F_SPEEDSUB.md:3,247,248,312...; RESULT_V8F_SPEEDSUB_T1.md:3,402,438; PREREG_V8F_SPEEDSUB.md:40; PREREG_V8F_SPEEDSUB_T1.md:42,42; PREREG_V8F_LAUNCH_INTENT.md:151,163,228; PREREG_V8F_DAG_ORC.md:125,194; RESULT_V8F_DAG_ORC.md:63,170,263,273...; GOALS_AND_CLAIMS.md (register rows):3; REP_S0_SPEED_BIAS.md:15; FINDING_1_ATTRIBUTION.md:13,14,65,65...; launch-intent-chain/HANDOFF.md:9; launch-intent-chain/LAUNCH.txt:39,101; DAGORC_READ.json:4072,4338,4696,4703...; git log (cited commits):1,4,28,40; RESULT_V8F_LAUNCH_INTENT.md:3,161,420,422...; LAUNCH_INTENT_READ.json:5141,5511,5883,6253...; MLAUNCH.json:1651,1669; chain_v8f_launchintent.out:6,117; lit1_logs/chain.out:14,14,16,16...; lit1_logs/launch_intent_read_REFUSED.out:37; read_A1/read_A1.out:3,4,5,6...; check_result_filled.out:86,93,93,137; REP launch-gate tokens (s0, s1):27,28,73,74 |
| `0.26313` | 1821 | RESULT_V8F_SPEEDSUB.md:247,251; GOALS_AND_CLAIMS.md (register rows):4; FINDING_1_ATTRIBUTION.md:13,22; git log (cited commits):31 |
| `+0.0328` | 1822 | RESULT_V8F_SPEEDSUB.md:248 |
| `0.30849` | 1823 | RESULT_V8F_SPEEDSUB.md:249; GOALS_AND_CLAIMS.md (register rows):4; FINDING_1_ATTRIBUTION.md:15; git log (cited commits):31 |
| `+0.0610` | 1823 | RESULT_V8F_SPEEDSUB.md:249 |
| `0.0156` | 1825 | RESULT_V8F_SPEEDSUB.md:251; SPEEDSUB_T1_READ.json:851,926 |
| `0.0328` | 1825 | RESULT_V8F_SPEEDSUB.md:251 |
| `+0.0080` | 1838,1933 | RESULT_V8F_SPEEDSUB.md:88,166 |
| `-0.0033` | 1838,1933,2072 | RESULT_V8F_SPEEDSUB.md:88,166; RESULT_V8F_SPEEDSUB_T1.md:98,152; SPEEDSUB_T1_READ.json:1691,3940,6553; P0_RF0.json (rebase):138 |
| `+0.0181` | 1838,1933,2712,2733 | RESULT_V8F_SPEEDSUB.md:88,166; RESULT_V8F_DAG_ORC.md:99; RESULT_V8F_LAUNCH_INTENT.md:272,300 |
| `+0.0039` | 1838,1933 | RESULT_V8F_SPEEDSUB.md:88,166; RESULT_V8F_DAG_ORC.md:73,154,242 |
| `-0.0013` | 1838,1848,1933 | RESULT_V8F_SPEEDSUB.md:88,98,166 |
| `+0.0092` | 1838,1851,1933,2072,2102... | RESULT_V8F_SPEEDSUB.md:88,101,166; RESULT_V8F_SPEEDSUB_T1.md:98,191,348; RESULT_V8F_DAG_ORC.md:107,123,162; RESULT_V8F_LAUNCH_INTENT.md:180 |
| `-0.0662` | 1839 | RESULT_V8F_SPEEDSUB.md:89,159 |
| `-0.1122` | 1839 | RESULT_V8F_SPEEDSUB.md:89,159 |
| `-0.0245` | 1839,2528,2634 | RESULT_V8F_SPEEDSUB.md:89,159; RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:4319,5251 |
| `-0.0379` | 1839 | RESULT_V8F_SPEEDSUB.md:89,160 |
| `-0.0638` | 1839 | RESULT_V8F_SPEEDSUB.md:89,160 |
| `-0.0130` | 1839 | RESULT_V8F_SPEEDSUB.md:89,160 |
| `-0.1403` | 1840,1892 | RESULT_V8F_SPEEDSUB.md:90,142,159 |
| `-0.2266` | 1840 | RESULT_V8F_SPEEDSUB.md:90,159 |
| `-0.0641` | 1840 | RESULT_V8F_SPEEDSUB.md:90,159 |
| `-0.0793` | 1840 | RESULT_V8F_SPEEDSUB.md:90,160 |
| `-0.1269` | 1840 | RESULT_V8F_SPEEDSUB.md:90,160; RESULT_V8F_DAG_ORC.md:84; DAGORC_READ.json:1060; RF_REP.json:233; LAUNCH_INTENT_READ.json:1486 |
| `-0.0337` | 1840,2068,2223,2629 | RESULT_V8F_SPEEDSUB.md:90,160; RESULT_V8F_SPEEDSUB_T1.md:94,344; SPEEDSUB_T1_READ.json:1227,6081; RESULT_V8F_LAUNCH_INTENT.md:176; LAUNCH_INTENT_READ.json:4519,5895 |
| `-0.0497` | 1841 | RESULT_V8F_SPEEDSUB.md:91,159 |
| `-0.0866` | 1841 | RESULT_V8F_SPEEDSUB.md:91,159 |
| `-0.0194` | 1841 | RESULT_V8F_SPEEDSUB.md:91,159 |
| `-0.0295` | 1841 | RESULT_V8F_SPEEDSUB.md:91,160 |
| `-0.0456` | 1841 | RESULT_V8F_SPEEDSUB.md:91,160 |
| `-0.0131` | 1841 | RESULT_V8F_SPEEDSUB.md:91,160 |
| `-0.1515` | 1844 | RESULT_V8F_SPEEDSUB.md:94,159 |
| `-0.2900` | 1844 | RESULT_V8F_SPEEDSUB.md:94,159 |
| `-0.0303` | 1844,1877 | RESULT_V8F_SPEEDSUB.md:94,127,159; MLAUNCH.json:390,478 |
| `-0.0429` | 1844 | RESULT_V8F_SPEEDSUB.md:94; RESULT_V8F_SPEEDSUB_T1.md:167; SPEEDSUB_T1_READ.json:3664; REP_S0_SPEED_BIAS.md:26 |
| `-0.1509` | 1844 | RESULT_V8F_SPEEDSUB.md:94 |
| `+0.0491` | 1844 | RESULT_V8F_SPEEDSUB.md:94 |
| `2350` | 1844 | RESULT_V8F_SPEEDSUB.md:94 |
| `2373` | 1844,1880,2640,2640,2640... | RESULT_V8F_SPEEDSUB.md:94,130; RESULT_V8F_SPEEDSUB_T1.md:118,118,118,118; RESULT_V8F_DAG_ORC.md:50; SPEEDSUB_T1_READ.json:998,1210,1422,1634...; DAGORC_READ.json:683,951,1221,1489...; P0_RF0.json (rebase):187,194; RESULT_V8F_LAUNCH_INTENT.md:189,189,189,189; LAUNCH_INTENT_READ.json:1109,1377,1647,1915... |
| `-0.2434` | 1845 | RESULT_V8F_SPEEDSUB.md:95 |
| `-0.6091` | 1845 | RESULT_V8F_SPEEDSUB.md:95 |
| `+0.0725` | 1845 | RESULT_V8F_SPEEDSUB.md:95 |
| `-0.2440` | 1845 | RESULT_V8F_SPEEDSUB.md:95,160 |
| `-0.5149` | 1845 | RESULT_V8F_SPEEDSUB.md:95,160 |
| `-0.0133` | 1845,1857,2065 | RESULT_V8F_SPEEDSUB.md:95,107,160; RESULT_V8F_SPEEDSUB_T1.md:91; SPEEDSUB_T1_READ.json:1408,5756 |
| `2328` | 1845,1846,1848 | RESULT_V8F_SPEEDSUB.md:95,96,98 |
| `2364` | 1845,1846,1848,1881,1882... | RESULT_V8F_SPEEDSUB.md:95,96,98,131...; SPEEDSUB_T1_READ.json:4194,4204,4224 |
| `-0.0041` | 1846,1874,2631 | RESULT_V8F_SPEEDSUB.md:96,124; P0_RF0.json (rebase):160; RESULT_V8F_LAUNCH_INTENT.md:178; LAUNCH_INTENT_READ.json:3203,4823 |
| `-0.0107` | 1846,1849 | RESULT_V8F_SPEEDSUB.md:96,99 |
| `+0.0018` | 1846 | RESULT_V8F_SPEEDSUB.md:96 |
| `-0.0047` | 1846,2064 | RESULT_V8F_SPEEDSUB.md:96; RESULT_V8F_SPEEDSUB_T1.md:90; SPEEDSUB_T1_READ.json:1610,5659 |
| `-0.0105` | 1846,2071,2163 | RESULT_V8F_SPEEDSUB.md:96; RESULT_V8F_SPEEDSUB_T1.md:97,275; SPEEDSUB_T1_READ.json:1470,6425 |
| `+0.0005` | 1846,1882 | RESULT_V8F_SPEEDSUB.md:96,132 |
| `+0.0020` | 1847,1848,1927 | RESULT_V8F_SPEEDSUB.md:97,98,163; FINDING_1_ATTRIBUTION.md:24 |
| `-0.0112` | 1847 | RESULT_V8F_SPEEDSUB.md:97; MLAUNCH.json:193 |
| `+0.0143` | 1847 | RESULT_V8F_SPEEDSUB.md:97 |
| `+0.0021` | 1847,1927,1955,2077,2226... | RESULT_V8F_SPEEDSUB.md:97,163,212; RESULT_V8F_SPEEDSUB_T1.md:103,317,347,372; RESULT_V8F_LAUNCH_INTENT.md:29,104,152,177 |
| `-0.0134` | 1847 | RESULT_V8F_SPEEDSUB.md:97; RESULT_V8F_DAG_ORC.md:99,154; DAGORC_READ.json:1443,4130; MLAUNCH.json:709 |
| `+0.0161` | 1847,2100 | RESULT_V8F_SPEEDSUB.md:97; RESULT_V8F_SPEEDSUB_T1.md:189 |
| `-0.0096` | 1848 | RESULT_V8F_SPEEDSUB.md:98,159 |
| `-0.0165` | 1848 | RESULT_V8F_SPEEDSUB.md:98,159 |
| `-0.0028` | 1848,2549 | RESULT_V8F_SPEEDSUB.md:98,159; RESULT_V8F_LAUNCH_INTENT.md:100; LAUNCH_INTENT_READ.json:3680 |
| `5.2` | 1849,1850,1885,1886,1926... | RESULT_V8F_SPEEDSUB.md:99,100,135,136...; RESULT_V8F_SPEEDSUB_T1.md:51,97,123,160...; PREREG_V8F_SPEEDSUB.md:161; PREREG_V8F_SPEEDSUB_T1.md:151; PREREG_V8F_LAUNCH_INTENT.md:169,194; PREREG_V8F_DAG_ORC.md:26; RESULT_V8F_DAG_ORC.md:55,81,107,123...; GOALS_AND_CLAIMS.md (register rows):6; SPEEDSUB_T1_READ.json:813,1043,1255,1467...; SPEEDSUB_T1_BESIDE.json:115,288; launch-intent-flag/MM_REDESIGN_2026-10-28.md:23; launch-intent-v2/enumerate_cells_li.out:67,67; DAGORC_READ.json:295,306,312,318...; RF_REP.json:15,97,213; P0_RF0.json (rebase):293,474; RESULT_V8F_LAUNCH_INTENT.md:64,91,95,99...; LAUNCH_INTENT_READ.json:340,346,352,426...; MLAUNCH NOTES.md:68; check_result_filled.out:107 |
| `-0.0319` | 1849 | RESULT_V8F_SPEEDSUB.md:99; MLAUNCH.json:1821; MLAUNCH NOTES.md:74 |
| `+0.0003` | 1849 | RESULT_V8F_SPEEDSUB.md:99; RESULT_V8F_DAG_ORC.md:79,160 |
| `-0.0065` | 1849 | RESULT_V8F_SPEEDSUB.md:99,160 |
| `-0.0136` | 1849 | RESULT_V8F_SPEEDSUB.md:99,160 |
| `-0.0006` | 1849 | RESULT_V8F_SPEEDSUB.md:99,160 |
| `2096` | 1849,1850,1917 | RESULT_V8F_SPEEDSUB.md:99,100,184 |
| `2102` | 1849,1850 | RESULT_V8F_SPEEDSUB.md:99,100; RESULT_V8F_SPEEDSUB_T1.md:123,129; SPEEDSUB_T1_READ.json:1048,1108,6401 |
| `-0.5435` | 1850,1926 | RESULT_V8F_SPEEDSUB.md:100,159,163 |
| `-1.4547` | 1850 | RESULT_V8F_SPEEDSUB.md:100,159 |
| `-0.0667` | 1850 | RESULT_V8F_SPEEDSUB.md:100,159 |
| `-0.3193` | 1850,1926 | RESULT_V8F_SPEEDSUB.md:100,160,163 |
| `-0.6480` | 1850 | RESULT_V8F_SPEEDSUB.md:100,160 |
| `-0.0600` | 1850 | RESULT_V8F_SPEEDSUB.md:100,160 |
| `+0.0013` | 1851,1955,2652 | RESULT_V8F_SPEEDSUB.md:101,212; RESULT_V8F_LAUNCH_INTENT.md:201 |
| `-0.0054` | 1851 | RESULT_V8F_SPEEDSUB.md:101 |
| `+0.0084` | 1851 | RESULT_V8F_SPEEDSUB.md:101 |
| `+0.0038` | 1851,1953,1959,2225 | RESULT_V8F_SPEEDSUB.md:101,210,216; RESULT_V8F_SPEEDSUB_T1.md:346 |
| `-0.0021` | 1851,2077,2226,2553,2652 | RESULT_V8F_SPEEDSUB.md:101; RESULT_V8F_SPEEDSUB_T1.md:103,317,347; SPEEDSUB_T1_READ.json:1317; RESULT_V8F_LAUNCH_INTENT.md:104,201; LAUNCH_INTENT_READ.json:4431,5424 |
| `-0.0042` | 1852,1885,1887,2227,2545... | RESULT_V8F_SPEEDSUB.md:102,135,137; RESULT_V8F_SPEEDSUB_T1.md:348; SPEEDSUB_T1_BESIDE.json:126; RESULT_V8F_LAUNCH_INTENT.md:96,200; LAUNCH_INTENT_READ.json:3342,5035 |
| `-0.0246` | 1852 | RESULT_V8F_SPEEDSUB.md:102; RESULT_V8F_DAG_ORC.md:85,167; DAGORC_READ.json:1882,2420,4045; MLAUNCH.json:221 |
| `+0.0163` | 1852,1957,1957,2072,2690 | RESULT_V8F_SPEEDSUB.md:102,214,214,216...; RESULT_V8F_SPEEDSUB_T1.md:98; RESULT_V8F_DAG_ORC.md:101; RESULT_V8F_LAUNCH_INTENT.md:250 |
| `+0.0104` | 1852,2071,2225 | RESULT_V8F_SPEEDSUB.md:102; RESULT_V8F_SPEEDSUB_T1.md:97,346 |
| `-0.0100` | 1852,1858,2120 | RESULT_V8F_SPEEDSUB.md:102,108; RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.0313` | 1852 | RESULT_V8F_SPEEDSUB.md:102 |
| `+0.0017` | 1853,2079,2100,2688,2689 | RESULT_V8F_SPEEDSUB.md:103; RESULT_V8F_SPEEDSUB_T1.md:105,152,189,256; RESULT_V8F_DAG_ORC.md:99; RESULT_V8F_LAUNCH_INTENT.md:248,249 |
| `-0.0183` | 1853,1883 | RESULT_V8F_SPEEDSUB.md:103,133; RESULT_V8F_SPEEDSUB_T1.md:168; SPEEDSUB_T1_READ.json:3888; P0_RF0.json (rebase):321 |
| `+0.0221` | 1853 | RESULT_V8F_SPEEDSUB.md:103 |
| `+0.0108` | 1853 | RESULT_V8F_SPEEDSUB.md:103 |
| `-0.0075` | 1853,2533,2540,2645,2651 | RESULT_V8F_SPEEDSUB.md:103; RESULT_V8F_DAG_ORC.md:82; DAGORC_READ.json:1033; RF_REP.json:222; RESULT_V8F_LAUNCH_INTENT.md:14,91,194,200...; LAUNCH_INTENT_READ.json:1459,3327,4417,5008...; check_result_filled.out:124 |
| `+0.0300` | 1853 | RESULT_V8F_SPEEDSUB.md:103 |
| `-0.3943` | 1854,1929 | RESULT_V8F_SPEEDSUB.md:104,159,164 |
| `-0.6422` | 1854 | RESULT_V8F_SPEEDSUB.md:104,159 |
| `-0.1651` | 1854 | RESULT_V8F_SPEEDSUB.md:104,159 |
| `-0.1950` | 1854,1929 | RESULT_V8F_SPEEDSUB.md:104,160,164 |
| `-0.3323` | 1854 | RESULT_V8F_SPEEDSUB.md:104,160 |
| `-0.0558` | 1854 | RESULT_V8F_SPEEDSUB.md:104,160 |
| `+0.6398` | 1855 | RESULT_V8F_SPEEDSUB.md:105,159,164 |
| `+0.3316` | 1855 | RESULT_V8F_SPEEDSUB.md:105,159 |
| `+0.9665` | 1855 | RESULT_V8F_SPEEDSUB.md:105,159 |
| `+0.3587` | 1855 | RESULT_V8F_SPEEDSUB.md:105,160,164 |
| `+0.1257` | 1855 | RESULT_V8F_SPEEDSUB.md:105,160 |
| `+0.6058` | 1855 | RESULT_V8F_SPEEDSUB.md:105,160 |
| `-0.0732` | 1856,2062,2159 | RESULT_V8F_SPEEDSUB.md:106,159; RESULT_V8F_SPEEDSUB_T1.md:88,271; SPEEDSUB_T1_READ.json:1378,5426 |
| `-0.1386` | 1856 | RESULT_V8F_SPEEDSUB.md:106,159 |
| `-0.0091` | 1856 | RESULT_V8F_SPEEDSUB.md:106,159; P0_RF0.json (rebase):450 |
| `+0.0109` | 1856,2069,2226,2735 | RESULT_V8F_SPEEDSUB.md:106; RESULT_V8F_SPEEDSUB_T1.md:95,347; RESULT_V8F_LAUNCH_INTENT.md:302 |
| `-0.0603` | 1856 | RESULT_V8F_SPEEDSUB.md:106; PREREG_V8F_SPEEDSUB.md:37 |
| `+0.0785` | 1856 | RESULT_V8F_SPEEDSUB.md:106; REP_S0_SPEED_BIAS.md:24 |
| `-0.2417` | 1857 | RESULT_V8F_SPEEDSUB.md:107 |
| `+0.2332` | 1857 | RESULT_V8F_SPEEDSUB.md:107 |
| `-0.0199` | 1857 | RESULT_V8F_SPEEDSUB.md:107 |
| `-0.1127` | 1857 | RESULT_V8F_SPEEDSUB.md:107 |
| `+0.0583` | 1857 | RESULT_V8F_SPEEDSUB.md:107 |
| `2367` | 1857,1857,1858,1858,1893... | RESULT_V8F_SPEEDSUB.md:107,107,108,108...; RESULT_V8F_SPEEDSUB_T1.md:119,119,120,120...; SPEEDSUB_T1_READ.json:1432,1442,1462,1644...; DAGORC_READ.json:1503,1517,1545; RESULT_V8F_LAUNCH_INTENT.md:190,190,191,191...; LAUNCH_INTENT_READ.json:4351,4365,4393,4619... |
| `+0.0153` | 1858 | RESULT_V8F_SPEEDSUB.md:108,159 |
| `+0.0050` | 1858,1953,1959,2072,2632 | RESULT_V8F_SPEEDSUB.md:108,159,210,216; RESULT_V8F_SPEEDSUB_T1.md:98; RESULT_V8F_LAUNCH_INTENT.md:179 |
| `+0.0255` | 1858 | RESULT_V8F_SPEEDSUB.md:108,159 |
| `+0.0052` | 1858 | RESULT_V8F_SPEEDSUB.md:108 |
| `+0.0208` | 1858,2079,2165 | RESULT_V8F_SPEEDSUB.md:108; RESULT_V8F_SPEEDSUB_T1.md:105,256,277 |
| `+0.0000` | 1859,1859,1859,1859,1859... | RESULT_V8F_SPEEDSUB.md:109,109,109,109...; RESULT_V8F_SPEEDSUB_T1.md:89,89,98,348; PREREG_V8F_SPEEDSUB.md:163,163,163; RESULT_V8F_DAG_ORC.md:73,73,154,242; RESULT_V8F_LAUNCH_INTENT.md:200,248,272 |
| `2107` | 1859,1859,1895,1895 | RESULT_V8F_SPEEDSUB.md:109,109,145,145 |
| `2240` | 1860,1860,1896,1896 | RESULT_V8F_SPEEDSUB.md:110,110,146,146; RESULT_V8F_DAG_ORC.md:265 |
| `2365` | 1861,1861,1897,1897 | RESULT_V8F_SPEEDSUB.md:111,111,147,147; SPEEDSUB_T1_READ.json:1858,1868,1888,2708...; DAGORC_READ.json:2849,2863,2891,3387...; P0_RF0.json (rebase):210,217,233,240...; LAUNCH_INTENT_READ.json:1661,1675,1703,2737... |
| `30` | 1862,1865,1898,1901,2074... | RESULT_V8F_SPEEDSUB.md:112,115,148,151...; RESULT_V8F_SPEEDSUB_T1.md:100,126,163,225...; PREREG_V8F_SPEEDSUB.md:148,160,163,163; PREREG_V8F_SPEEDSUB_T1.md:153; PREREG_V8F_LAUNCH_INTENT.md:21,21,35,43...; PREREG_V8F_DAG_ORC.md:115,194,277,434; RESULT_V8F_DAG_ORC.md:58,84,110,138...; SPEEDSUB_T1_READ.json:213,250,288,331...; SPEEDSUB_T1_BESIDE.json:142,315; launch-intent-flag/RESULTS.md:19,19,37,49; DAGORC_READ.json:298,334,351,368...; RF_REP.json:18,115,231; P0_RF0.json (rebase):362,477; git log (cited commits):19,19,19; RESULT_V8F_LAUNCH_INTENT.md:67,209,213,259...; LAUNCH_INTENT_READ.json:368,385,402,454...; MLAUNCH NOTES.md:39,68 |
| `2322` | 1862,1862,1865,1865,1898... | RESULT_V8F_SPEEDSUB.md:112,112,115,115... |
| `50` | 1863,1866,1899,1902,2075... | RESULT_V8F_SPEEDSUB.md:113,116,149,152...; RESULT_V8F_SPEEDSUB_T1.md:101,127,164,225...; PREREG_V8F_SPEEDSUB.md:163,163; PREREG_V8F_SPEEDSUB_T1.md:153; PREREG_V8F_LAUNCH_INTENT.md:198,228; PREREG_V8F_DAG_ORC.md:39,115,194; RESULT_V8F_DAG_ORC.md:59,85,111,167...; SPEEDSUB_T1_READ.json:1083,1295,1507,1719...; SPEEDSUB_T1_BESIDE.json:151,324; launch-intent-chain/HANDOFF.md:1; DAGORC_READ.json:299,335,352,369...; RF_REP.json:19,121,237; P0_RF0.json (rebase):387,478; RESULT_V8F_LAUNCH_INTENT.md:68,210,213,260...; LAUNCH_INTENT_READ.json:369,386,403,455...; lit1_logs/chain.out.prev_1028151251:1,2,3; check_result_filled.out:60,92,92 |
| `2319` | 1863,1863,1866,1866,1899... | RESULT_V8F_SPEEDSUB.md:113,113,116,116... |
| `80` | 1864,1867,1900,1903,2076... | RESULT_V8F_SPEEDSUB.md:114,117,150,153...; RESULT_V8F_SPEEDSUB_T1.md:102,128,165,225...; PREREG_V8F_SPEEDSUB.md:163,163; PREREG_V8F_SPEEDSUB_T1.md:153; PREREG_V8F_LAUNCH_INTENT.md:199,228; PREREG_V8F_DAG_ORC.md:115,194,479; RESULT_V8F_DAG_ORC.md:60,63,86,112...; SPEEDSUB_T1_READ.json:1093,1305,1517,1729...; SPEEDSUB_T1_BESIDE.json:160,333; launch-intent-flag/RESULTS.md:28,28; DAGORC_READ.json:300,336,353,370...; RF_REP.json:20,127,243; P0_RF0.json (rebase):412,479; RESULT_V8F_LAUNCH_INTENT.md:69,211,213,261...; LAUNCH_INTENT_READ.json:370,387,404,456... |
| `2294` | 1864,1864,1867,1867,1900... | RESULT_V8F_SPEEDSUB.md:114,114,117,117... |
| `-0.0109` | 1874,2539,2642 | RESULT_V8F_SPEEDSUB.md:124; RESULT_V8F_LAUNCH_INTENT.md:90,191; LAUNCH_INTENT_READ.json:3286,4949 |
| `-0.0235` | 1874 | RESULT_V8F_SPEEDSUB.md:124 |
| `+0.0011` | 1874 | RESULT_V8F_SPEEDSUB.md:124; PREREG_V8F_DAG_ORC.md:169 |
| `-0.0119` | 1874,2061,2219 | RESULT_V8F_SPEEDSUB.md:124; RESULT_V8F_SPEEDSUB_T1.md:87,340; SPEEDSUB_T1_READ.json:1156,5303 |
| `-0.0196` | 1874 | RESULT_V8F_SPEEDSUB.md:124; MLAUNCH.json:294 |
| `-0.3544` | 1875 | RESULT_V8F_SPEEDSUB.md:125; RESULT_V8F_SPEEDSUB_T1.md:257; SPEEDSUB_T1_READ.json:2602,7007 |
| `-0.4059` | 1875 | RESULT_V8F_SPEEDSUB.md:125 |
| `-0.3064` | 1875 | RESULT_V8F_SPEEDSUB.md:125 |
| `-0.3424` | 1875 | RESULT_V8F_SPEEDSUB.md:125 |
| `-0.3878` | 1875 | RESULT_V8F_SPEEDSUB.md:125 |
| `-0.2990` | 1875 | RESULT_V8F_SPEEDSUB.md:125 |
| `-0.7644` | 1876 | RESULT_V8F_SPEEDSUB.md:126 |
| `-0.8685` | 1876 | RESULT_V8F_SPEEDSUB.md:126 |
| `-0.6692` | 1876 | RESULT_V8F_SPEEDSUB.md:126 |
| `-0.7122` | 1876 | RESULT_V8F_SPEEDSUB.md:126 |
| `-0.8054` | 1876 | RESULT_V8F_SPEEDSUB.md:126 |
| `-0.6194` | 1876 | RESULT_V8F_SPEEDSUB.md:126 |
| `-0.0536` | 1877 | RESULT_V8F_SPEEDSUB.md:127 |
| `-0.0970` | 1877 | RESULT_V8F_SPEEDSUB.md:127 |
| `-0.0140` | 1877 | RESULT_V8F_SPEEDSUB.md:127 |
| `-0.0543` | 1877 | RESULT_V8F_SPEEDSUB.md:127 |
| `-0.0084` | 1877 | RESULT_V8F_SPEEDSUB.md:127; REP_S0_SPEED_BIAS.md:22 |
| `+0.0808` | 1880 | RESULT_V8F_SPEEDSUB.md:130 |
| `-0.1576` | 1880 | RESULT_V8F_SPEEDSUB.md:130 |
| `+0.3139` | 1880,2689 | RESULT_V8F_SPEEDSUB.md:130; RESULT_V8F_LAUNCH_INTENT.md:249 |
| `-0.0839` | 1880 | RESULT_V8F_SPEEDSUB.md:130 |
| `-0.3452` | 1880 | RESULT_V8F_SPEEDSUB.md:130 |
| `+0.0962` | 1880,2722 | RESULT_V8F_SPEEDSUB.md:130; RESULT_V8F_LAUNCH_INTENT.md:282 |
| `2349` | 1880 | RESULT_V8F_SPEEDSUB.md:130 |
| `+0.0570` | 1881 | RESULT_V8F_SPEEDSUB.md:131 |
| `-0.4042` | 1881 | RESULT_V8F_SPEEDSUB.md:131 |
| `+0.4881` | 1881 | RESULT_V8F_SPEEDSUB.md:131 |
| `-0.0760` | 1881 | RESULT_V8F_SPEEDSUB.md:131 |
| `-0.3837` | 1881 | RESULT_V8F_SPEEDSUB.md:131; RESULT_V8F_SPEEDSUB_T1.md:257; SPEEDSUB_T1_READ.json:1966,6977 |
| `+0.1851` | 1881 | RESULT_V8F_SPEEDSUB.md:131 |
| `2329` | 1881,1882,1884 | RESULT_V8F_SPEEDSUB.md:131,132,134 |
| `-0.0122` | 1882 | RESULT_V8F_SPEEDSUB.md:132 |
| `+0.0122` | 1882,2543,2642 | RESULT_V8F_SPEEDSUB.md:132; RESULT_V8F_LAUNCH_INTENT.md:94,191 |
| `-0.0032` | 1882,2553 | RESULT_V8F_SPEEDSUB.md:132; SPEEDSUB_T1_BESIDE.json:226; RESULT_V8F_LAUNCH_INTENT.md:104; LAUNCH_INTENT_READ.json:4756 |
| `-0.0099` | 1882,2099 | RESULT_V8F_SPEEDSUB.md:132; RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2230,5482; MLAUNCH.json:1789; MLAUNCH NOTES.md:73 |
| `+0.0027` | 1882,1884 | RESULT_V8F_SPEEDSUB.md:132,134 |
| `-0.0378` | 1883 | RESULT_V8F_SPEEDSUB.md:133 |
| `-0.0601` | 1883 | RESULT_V8F_SPEEDSUB.md:133 |
| `-0.0430` | 1883 | RESULT_V8F_SPEEDSUB.md:133 |
| `-0.0657` | 1883 | RESULT_V8F_SPEEDSUB.md:133 |
| `-0.0231` | 1883,2688 | RESULT_V8F_SPEEDSUB.md:133; RESULT_V8F_DAG_ORC.md:99; SPEEDSUB_T1_READ.json:2665; DAGORC_READ.json:2790,3328; RESULT_V8F_LAUNCH_INTENT.md:248; LAUNCH_INTENT_READ.json:1602,4854 |
| `-0.0044` | 1884 | RESULT_V8F_SPEEDSUB.md:134 |
| `-0.0120` | 1884 | RESULT_V8F_SPEEDSUB.md:134 |
| `+0.0022` | 1884,2071,2226,2541 | RESULT_V8F_SPEEDSUB.md:134; RESULT_V8F_SPEEDSUB_T1.md:97,347; PREREG_V8F_SPEEDSUB.md:10; RESULT_V8F_LAUNCH_INTENT.md:92 |
| `-0.0029` | 1884 | RESULT_V8F_SPEEDSUB.md:134 |
| `+0.0067` | 1884,2688 | RESULT_V8F_SPEEDSUB.md:134; RESULT_V8F_LAUNCH_INTENT.md:248 |
| `-0.0158` | 1885,2078,2652 | RESULT_V8F_SPEEDSUB.md:135; RESULT_V8F_SPEEDSUB_T1.md:104,255; RESULT_V8F_DAG_ORC.md:73; SPEEDSUB_T1_READ.json:1750,6963; DAGORC_READ.json:906; RF_REP.json:167; RESULT_V8F_LAUNCH_INTENT.md:201; LAUNCH_INTENT_READ.json:1332,4432,5425 |
| `-0.0381` | 1885 | RESULT_V8F_SPEEDSUB.md:135 |
| `-0.0108` | 1885 | RESULT_V8F_SPEEDSUB.md:135 |
| `-0.0220` | 1885,2068,2223 | RESULT_V8F_SPEEDSUB.md:135; RESULT_V8F_SPEEDSUB_T1.md:94,344; PREREG_V8F_DAG_ORC.md:24; RESULT_V8F_DAG_ORC.md:79 |
| `-0.0026` | 1885 | RESULT_V8F_SPEEDSUB.md:135 |
| `2097` | 1885,1886 | RESULT_V8F_SPEEDSUB.md:135,136 |
| `2104` | 1885,1886,2645 | RESULT_V8F_SPEEDSUB.md:135,136; SPEEDSUB_T1_READ.json:3598,3658,3810,3870; RESULT_V8F_LAUNCH_INTENT.md:194; LAUNCH_INTENT_READ.json:3869,3953,4407,4491 |
| `-0.7495` | 1886 | RESULT_V8F_SPEEDSUB.md:136 |
| `-1.6944` | 1886 | RESULT_V8F_SPEEDSUB.md:136 |
| `-0.2279` | 1886 | RESULT_V8F_SPEEDSUB.md:136 |
| `-0.5456` | 1886 | RESULT_V8F_SPEEDSUB.md:136 |
| `-1.0354` | 1886 | RESULT_V8F_SPEEDSUB.md:136 |
| `-0.1744` | 1886 | RESULT_V8F_SPEEDSUB.md:136 |
| `-0.0071` | 1887,2068,2072,2223 | RESULT_V8F_SPEEDSUB.md:137; RESULT_V8F_SPEEDSUB_T1.md:94,98,344; SPEEDSUB_T1_READ.json:1014,1479,6067,6540 |
| `-0.0192` | 1887 | RESULT_V8F_SPEEDSUB.md:137 |
| `+0.0042` | 1887,2077,2545 | RESULT_V8F_SPEEDSUB.md:137; RESULT_V8F_SPEEDSUB_T1.md:103,317; RESULT_V8F_LAUNCH_INTENT.md:96 |
| `-0.0142` | 1887,2651 | RESULT_V8F_SPEEDSUB.md:137; RESULT_V8F_DAG_ORC.md:83,164; DAGORC_READ.json:2122,2660,4277; P0_RF0.json (rebase):68; RESULT_V8F_LAUNCH_INTENT.md:200; LAUNCH_INTENT_READ.json:3610,5777 |
| `+0.0054` | 1887,1957,2072,2227 | RESULT_V8F_SPEEDSUB.md:137,214; RESULT_V8F_SPEEDSUB_T1.md:98,348 |
| `+0.1159` | 1888 | RESULT_V8F_SPEEDSUB.md:138 |
| `+0.0875` | 1888 | RESULT_V8F_SPEEDSUB.md:138 |
| `+0.1450` | 1888 | RESULT_V8F_SPEEDSUB.md:138 |
| `+0.1446` | 1888 | RESULT_V8F_SPEEDSUB.md:138 |
| `+0.1188` | 1888 | RESULT_V8F_SPEEDSUB.md:138; RESULT_V8F_DAG_ORC.md:71,152,241,244; GOALS_AND_CLAIMS.md (register rows):3 |
| `+0.1706` | 1888 | RESULT_V8F_SPEEDSUB.md:138 |
| `+0.0805` | 1889 | RESULT_V8F_SPEEDSUB.md:139 |
| `+0.0513` | 1889 | RESULT_V8F_SPEEDSUB.md:139; RESULT_V8F_SPEEDSUB_T1.md:166 |
| `+0.1108` | 1889 | RESULT_V8F_SPEEDSUB.md:139 |
| `+0.0988` | 1889 | RESULT_V8F_SPEEDSUB.md:139 |
| `+0.0738` | 1889 | RESULT_V8F_SPEEDSUB.md:139; RESULT_V8F_SPEEDSUB_T1.md:166 |
| `+0.1254` | 1889 | RESULT_V8F_SPEEDSUB.md:139 |
| `-2.2239` | 1890 | RESULT_V8F_SPEEDSUB.md:140 |
| `-2.5417` | 1890 | RESULT_V8F_SPEEDSUB.md:140 |
| `-1.9337` | 1890 | RESULT_V8F_SPEEDSUB.md:140 |
| `-2.0637` | 1890 | RESULT_V8F_SPEEDSUB.md:140 |
| `-2.3514` | 1890 | RESULT_V8F_SPEEDSUB.md:140 |
| `-1.7739` | 1890 | RESULT_V8F_SPEEDSUB.md:140 |
| `+0.6815` | 1891 | RESULT_V8F_SPEEDSUB.md:141 |
| `+0.1831` | 1891 | RESULT_V8F_SPEEDSUB.md:141 |
| `+1.2044` | 1891 | RESULT_V8F_SPEEDSUB.md:141 |
| `+0.2837` | 1891 | RESULT_V8F_SPEEDSUB.md:141 |
| `-0.1308` | 1891 | RESULT_V8F_SPEEDSUB.md:141 |
| `+0.7242` | 1891 | RESULT_V8F_SPEEDSUB.md:141 |
| `-0.1294` | 1892 | RESULT_V8F_SPEEDSUB.md:142 |
| `-0.2446` | 1892 | RESULT_V8F_SPEEDSUB.md:142 |
| `-0.0261` | 1892 | RESULT_V8F_SPEEDSUB.md:142 |
| `-0.0302` | 1892,2712 | RESULT_V8F_SPEEDSUB.md:142; RESULT_V8F_LAUNCH_INTENT.md:272; LAUNCH_INTENT_READ.json:2946,5590 |
| `+0.0734` | 1892 | RESULT_V8F_SPEEDSUB.md:142 |
| `-0.1541` | 1893 | RESULT_V8F_SPEEDSUB.md:143 |
| `-0.6931` | 1893 | RESULT_V8F_SPEEDSUB.md:143 |
| `+0.3614` | 1893 | RESULT_V8F_SPEEDSUB.md:143 |
| `-0.2168` | 1893 | RESULT_V8F_SPEEDSUB.md:143 |
| `-0.5473` | 1893 | RESULT_V8F_SPEEDSUB.md:143 |
| `+0.0024` | 1893 | RESULT_V8F_SPEEDSUB.md:143 |
| `-0.0080` | 1894 | RESULT_V8F_SPEEDSUB.md:144; RESULT_V8F_DAG_ORC.md:99,141,154 |
| `-0.0402` | 1894 | RESULT_V8F_SPEEDSUB.md:144 |
| `+0.0214` | 1894 | RESULT_V8F_SPEEDSUB.md:144; RESULT_V8F_DAG_ORC.md:73 |
| `+0.0034` | 1894,2069,2632 | RESULT_V8F_SPEEDSUB.md:144; RESULT_V8F_SPEEDSUB_T1.md:95; RESULT_V8F_LAUNCH_INTENT.md:179 |
| `60` | 1909 | RESULT_V8F_SPEEDSUB.md:162,176,296; PREREG_V8F_SPEEDSUB.md:160; PREREG_V8F_DAG_ORC.md:351,351 |
| `4.4` | 1911 | RESULT_V8F_SPEEDSUB.md:178,230; RESULT_V8F_LAUNCH_INTENT.md:320 |
| `695` | 1914,1914 | RESULT_V8F_SPEEDSUB.md:181,181; SPEEDSUB_T1_READ.json:3952 |
| `0.0000` | 1917,2055,2299,2300,2301... | RESULT_V8F_SPEEDSUB.md:184; RESULT_V8F_SPEEDSUB_T1.md:133; PREREG_V8F_LAUNCH_INTENT.md:183,184,185,186...; PREREG_V8F_DAG_ORC.md:476; RESULT_V8F_DAG_ORC.md:44,45,46,47...; GOALS_AND_CLAIMS.md (register rows):4; FINDING_1_ATTRIBUTION.md:18,65; RESULT_V8F_LAUNCH_INTENT.md:53,54,55,56... |
| `18.7` | 1923 | RESULT_V8F_SPEEDSUB.md:162 |
| `11.1` | 1923,1924 | RESULT_V8F_SPEEDSUB.md:162,162 |
| `18.4` | 1924,2568 | RESULT_V8F_SPEEDSUB.md:162; RESULT_V8F_LAUNCH_INTENT.md:120 |
| `92.7` | 1924 | RESULT_V8F_SPEEDSUB.md:162 |
| `97.4` | 1924 | RESULT_V8F_SPEEDSUB.md:162 |
| `0.05` | 1924 | RESULT_V8F_SPEEDSUB.md:162; DAGORC_READ.json:1941,2479,4084 |
| `0.03` | 1924 | RESULT_V8F_SPEEDSUB.md:162; MLAUNCH.json:480 |
| `40` | 1936 | RESULT_V8F_SPEEDSUB.md:202; RESULT_V8F_SPEEDSUB_T1.md:423,424; PREREG_V8F_SPEEDSUB.md:20,40,148; PREREG_V8F_LAUNCH_INTENT.md:119; RESULT_V8F_DAG_ORC.md:271,272,277; launch-intent-flag/RESULTS.md:36; launch-intent-flag/MM_REDESIGN_2026-10-28.md:46; lit1_logs/chain.out:39; lit1_logs/launch_intent_read_REFUSED.out:8,28,35 |
| `59` | 1936,1970 | RESULT_V8F_SPEEDSUB.md:3,202,307,308; RESULT_V8F_SPEEDSUB_T1.md:3,15,429,434...; PREREG_V8F_SPEEDSUB.md:20,40; PREREG_V8F_SPEEDSUB_T1.md:18,70; RESULT_V8F_DAG_ORC.md:261,349; git log (cited commits):16; lit1_logs/chain.out.prev_1028151251:10,11,12,13; lit1_logs/launch_intent_read_REFUSED.out:31,32,33,34... |
| `2334` | 1941 | RESULT_V8F_SPEEDSUB.md:196; SUBREAD.json:141,205 |
| `2.8940` | 1941 | RESULT_V8F_SPEEDSUB.md:196 |
| `2.7470` | 1941 | RESULT_V8F_SPEEDSUB.md:196 |
| `2.5404` | 1941 | RESULT_V8F_SPEEDSUB.md:196 |
| `12.2` | 1941 | RESULT_V8F_SPEEDSUB.md:196 |
| `7.5` | 1941 | RESULT_V8F_SPEEDSUB.md:196 |
| `2351` | 1942,1944,1945 | RESULT_V8F_SPEEDSUB.md:197,199,200; SUBREAD.json:142,144,145,206... |
| `1.9190` | 1942 | RESULT_V8F_SPEEDSUB.md:197 |
| `1.3020` | 1942 | RESULT_V8F_SPEEDSUB.md:197 |
| `0.7517` | 1942 | RESULT_V8F_SPEEDSUB.md:197 |
| `60.8` | 1942 | RESULT_V8F_SPEEDSUB.md:197 |
| `42.3` | 1942 | RESULT_V8F_SPEEDSUB.md:197 |
| `2317` | 1943 | RESULT_V8F_SPEEDSUB.md:198; SUBREAD.json:143,207 |
| `2.7953` | 1943 | RESULT_V8F_SPEEDSUB.md:198 |
| `2.6916` | 1943 | RESULT_V8F_SPEEDSUB.md:198 |
| `2.5570` | 1943 | RESULT_V8F_SPEEDSUB.md:198 |
| `8.5` | 1943 | RESULT_V8F_SPEEDSUB.md:198 |
| `5.0` | 1943 | RESULT_V8F_SPEEDSUB.md:198 |
| `2.2571` | 1944 | RESULT_V8F_SPEEDSUB.md:199 |
| `2.1425` | 1944 | RESULT_V8F_SPEEDSUB.md:199 |
| `1.8607` | 1944 | RESULT_V8F_SPEEDSUB.md:199 |
| `17.6` | 1944 | RESULT_V8F_SPEEDSUB.md:199 |
| `13.2` | 1944 | RESULT_V8F_SPEEDSUB.md:199 |
| `2.0927` | 1945 | RESULT_V8F_SPEEDSUB.md:200 |
| `1.8020` | 1945 | RESULT_V8F_SPEEDSUB.md:200 |
| `1.7143` | 1945 | RESULT_V8F_SPEEDSUB.md:200 |
| `18.1` | 1945 | RESULT_V8F_SPEEDSUB.md:200 |
| `4.9` | 1945 | RESULT_V8F_SPEEDSUB.md:200; git log (cited commits):37 |
| `0.1642` | 1953 | RESULT_V8F_SPEEDSUB.md:210 |
| `0.1926` | 1953 | RESULT_V8F_SPEEDSUB.md:210; LAUNCH_INTENT_READ.json:4147,6152 |
| `0.1859` | 1953 | RESULT_V8F_SPEEDSUB.md:210 |
| `0.2118` | 1953 | RESULT_V8F_SPEEDSUB.md:210; DAGORC_READ.json:1301,4002 |
| `0.0825` | 1953 | RESULT_V8F_SPEEDSUB.md:210 |
| `0.1238` | 1953 | RESULT_V8F_SPEEDSUB.md:210 |
| `+0.0217` | 1953,1959,2651 | RESULT_V8F_SPEEDSUB.md:210,216; GOALS_AND_CLAIMS.md (register rows):4; RESULT_V8F_LAUNCH_INTENT.md:200 |
| `+0.0404` | 1953,1959 | RESULT_V8F_SPEEDSUB.md:210,216 |
| `+0.0192` | 1953,1959 | RESULT_V8F_SPEEDSUB.md:210,216; GOALS_AND_CLAIMS.md (register rows):4 |
| `+0.0346` | 1953,1959,2630 | RESULT_V8F_SPEEDSUB.md:210,216; RESULT_V8F_LAUNCH_INTENT.md:177 |
| `0.0975` | 1954 | RESULT_V8F_SPEEDSUB.md:211 |
| `0.1113` | 1954 | RESULT_V8F_SPEEDSUB.md:211 |
| `0.1075` | 1954 | RESULT_V8F_SPEEDSUB.md:211 |
| `0.1280` | 1954 | RESULT_V8F_SPEEDSUB.md:211 |
| `0.0375` | 1954 | RESULT_V8F_SPEEDSUB.md:211; SPEEDSUB_T1_BESIDE.json:10; LAUNCH_INTENT_READ.json:2789,5014 |
| `0.0446` | 1954 | RESULT_V8F_SPEEDSUB.md:211; SPEEDSUB_T1_READ.json:3188; LAUNCH_INTENT_READ.json:2279,5436 |
| `+0.0100` | 1954 | RESULT_V8F_SPEEDSUB.md:211 |
| `-0.0067` | 1954,2100,2100,2712 | RESULT_V8F_SPEEDSUB.md:211; RESULT_V8F_SPEEDSUB_T1.md:167,189,189; SPEEDSUB_T1_READ.json:2026,2450,3666,5577...; RESULT_V8F_LAUNCH_INTENT.md:272; LAUNCH_INTENT_READ.json:2945,5589 |
| `+0.0275` | 1954 | RESULT_V8F_SPEEDSUB.md:211 |
| `+0.0167` | 1954 | RESULT_V8F_SPEEDSUB.md:211; RESULT_V8F_SPEEDSUB_T1.md:152; RESULT_V8F_DAG_ORC.md:73 |
| `+0.0025` | 1954,1956,1956,2533,2540... | RESULT_V8F_SPEEDSUB.md:211,213,213; RESULT_V8F_LAUNCH_INTENT.md:14,91,194,351; check_result_filled.out:124 |
| `+0.0296` | 1954,2113 | RESULT_V8F_SPEEDSUB.md:211; RESULT_V8F_SPEEDSUB_T1.md:208; RESULT_V8F_DAG_ORC.md:79,160 |
| `0.0196` | 1955 | RESULT_V8F_SPEEDSUB.md:212; DAGORC_READ.json:1045; RF_REP.json:226; LAUNCH_INTENT_READ.json:1471; MLAUNCH.json:289 |
| `0.0279` | 1955 | RESULT_V8F_SPEEDSUB.md:212; SPEEDSUB_T1_READ.json:3108; DAGORC_READ.json:892; RF_REP.json:161; LAUNCH_INTENT_READ.json:1318,2167,5263; MLAUNCH.json:325 |
| `0.0217` | 1955 | RESULT_V8F_SPEEDSUB.md:212; LAUNCH_INTENT_READ.json:4685,6146; MLAUNCH.json:711 |
| `0.0292` | 1955 | RESULT_V8F_SPEEDSUB.md:212; DAGORC_READ.json:977; RF_REP.json:198; LAUNCH_INTENT_READ.json:1403 |
| `0.0104` | 1955 | RESULT_V8F_SPEEDSUB.md:212; SPEEDSUB_T1_READ.json:1256,6410 |
| `0.0263` | 1955,2307,2514 | RESULT_V8F_SPEEDSUB.md:212; PREREG_V8F_LAUNCH_INTENT.md:191; RESULT_V8F_DAG_ORC.md:52,220; SPEEDSUB_T1_BESIDE.json:307; DAGORC_READ.json:497; P0_RF0.json (rebase):243,471; RESULT_V8F_LAUNCH_INTENT.md:61; LAUNCH_INTENT_READ.json:699,824 |
| `+0.0046` | 1955,2073 | RESULT_V8F_SPEEDSUB.md:212; RESULT_V8F_SPEEDSUB_T1.md:99; RESULT_V8F_DAG_ORC.md:83,164 |
| `+0.0029` | 1955 | RESULT_V8F_SPEEDSUB.md:212; RESULT_V8F_SPEEDSUB_T1.md:167 |
| `0.0067` | 1956,1956 | RESULT_V8F_SPEEDSUB.md:213,213; LAUNCH_INTENT_READ.json:2677,4847 |
| `0.0079` | 1956,1956 | RESULT_V8F_SPEEDSUB.md:213,213; SPEEDSUB_T1_READ.json:3070; LAUNCH_INTENT_READ.json:2113,4686,5185,6147 |
| `0.0075` | 1956,1956 | RESULT_V8F_SPEEDSUB.md:213,213; SPEEDSUB_T1_READ.json:1338,7031 |
| `-0.0008` | 1956,2073,2187,2228,2541 | RESULT_V8F_SPEEDSUB.md:213; RESULT_V8F_SPEEDSUB_T1.md:99,301,303,349; RESULT_V8F_DAG_ORC.md:78,220; SPEEDSUB_T1_READ.json:910,1276,6636; DAGORC_READ.json:975; RF_REP.json:196; P0_RF0.json (rebase):451; RESULT_V8F_LAUNCH_INTENT.md:92; LAUNCH_INTENT_READ.json:1401,3411 |
| `-0.0017` | 1956,2079,2227,2228 | RESULT_V8F_SPEEDSUB.md:213; RESULT_V8F_SPEEDSUB_T1.md:105,152,256,348...; RESULT_V8F_DAG_ORC.md:73,83,164; SPEEDSUB_T1_READ.json:1336,4152,7029; SPEEDSUB_T1_BESIDE.json:135,298; DAGORC_READ.json:637,1854,2392,4011; RF_REP.json:50; P0_RF0.json (rebase):114; LAUNCH_INTENT_READ.json:1063 |
| `0.0775` | 1957 | RESULT_V8F_SPEEDSUB.md:214 |
| `0.0959` | 1957 | RESULT_V8F_SPEEDSUB.md:214 |
| `0.0938` | 1957 | RESULT_V8F_SPEEDSUB.md:214 |
| `0.1121` | 1957 | RESULT_V8F_SPEEDSUB.md:214 |
| `0.0379` | 1957 | RESULT_V8F_SPEEDSUB.md:214; SPEEDSUB_T1_READ.json:1955 |
| `0.0654` | 1957 | RESULT_V8F_SPEEDSUB.md:214 |
| `+0.0063` | 1957,2072,2077,2226,2548... | RESULT_V8F_SPEEDSUB.md:214; RESULT_V8F_SPEEDSUB_T1.md:98,103,317,347; RESULT_V8F_LAUNCH_INTENT.md:99,194,200 |
| `+0.0263` | 1957,2228 | RESULT_V8F_SPEEDSUB.md:214; RESULT_V8F_SPEEDSUB_T1.md:349 |
| `+0.0283` | 1957 | RESULT_V8F_SPEEDSUB.md:214 |
| `81` | 1960 | RESULT_V8F_SPEEDSUB.md:220,225,228; PREREG_V8F_DAG_ORC.md:280; GOALS_AND_CLAIMS.md (register rows):4; git log (cited commits):31; MLAUNCH NOTES.md:35 |
| `+0.3374` | 1960 | RESULT_V8F_SPEEDSUB.md:225,228 |
| `+0.1762` | 1960 | RESULT_V8F_SPEEDSUB.md:225,228 |
| `-1` | 1970,2256 | RESULT_V8F_SPEEDSUB.md:3; RESULT_V8F_SPEEDSUB_T1.md:3,406; PREREG_V8F_SPEEDSUB.md:137,142; PREREG_V8F_SPEEDSUB_T1.md:164; launch-intent-v2/NOTES.md:4; MLAUNCH NOTES.md:3,26,34 |
| `07` | 1970,2492,2492 | RESULT_V8F_SPEEDSUB.md:3,307,308; RESULT_V8F_SPEEDSUB_T1.md:423; RESULT_V8F_DAG_ORC.md:3,3,3,263...; launch-intent-chain/LAUNCH.txt:13,88; RESULT_V8F_LAUNCH_INTENT.md:425,425; chain_v8f_launchintent.out:2; lit1_logs/chain.out:31,32; lit1_logs/chain.out.prev_1028151251:14,15,16,17...; lit1_logs/launch_intent_read_REFUSED.out:10,11; check_result_filled.out:94,94 |
| `43` | 1970,2476,2816 | RESULT_V8F_SPEEDSUB.md:3,312; RESULT_V8F_SPEEDSUB_T1.md:421; PREREG_V8F_DAG_ORC.md:318,318,318; GOALS_AND_CLAIMS.md (register rows):3; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:447; chain_v8f_launchintent.out:6,44,45,46...; lit1_logs/chain.out:44; lit1_logs/launch_intent_read_REFUSED.out:20 |
| `35` | 1970,2476 | RESULT_V8F_SPEEDSUB.md:3; RESULT_V8F_SPEEDSUB_T1.md:419,421,422; RESULT_V8F_DAG_ORC.md:263,272; launch-intent-flag/RESULTS.md:11,27; git log (cited commits):19; RESULT_V8F_LAUNCH_INTENT.md:447; lit1_logs/chain.out:17,30,35,49 |
| `23` | 1977 | RESULT_V8F_SPEEDSUB_T1.md:3,77,375,402...; PREREG_V8F_DAG_ORC.md:340; launch-intent-v2/NOTES.md:11,11; git log (cited commits):28; MLAUNCH NOTES.md:36; lit1_logs/chain.out:39,43; read_A1/read_A1.out:23,24; check_result_filled.out:59 |
| `21` | 1977,1993,1995,2441 | RESULT_V8F_SPEEDSUB.md:314; RESULT_V8F_SPEEDSUB_T1.md:3,3,3,77...; RESULT_V8F_DAG_ORC.md:263,277; git log (cited commits):1,25,34; RESULT_V8F_LAUNCH_INTENT.md:420; lit1_logs/chain.out:27,42; read_A1/read_A1.out:3,38; check_result_filled.out:93,136 |
| `44` | 1977,2393,2477,2816 | RESULT_V8F_SPEEDSUB_T1.md:3,77,402,415; PREREG_V8F_DAG_ORC.md:318; RESULT_V8F_DAG_ORC.md:263,282; launch-intent-flag/RESULTS.md:27; launch-intent-chain/LAUNCH.txt:13,88; git log (cited commits):4,22; RESULT_V8F_LAUNCH_INTENT.md:3,448; chain_v8f_launchintent.out:2,55,58; lit1_logs/chain.out:31,50; check_result_filled.out:93 |
| `20` | 1979,2729,2729,2822 | RESULT_V8F_SPEEDSUB.md:290; RESULT_V8F_SPEEDSUB_T1.md:16; PREREG_V8F_SPEEDSUB.md:131,148,172; PREREG_V8F_SPEEDSUB_T1.md:17,44; RESULT_V8F_DAG_ORC.md:274,276,276; REP_S0_SPEED_BIAS.md:3; FINDING_1_ATTRIBUTION.md:48; launch-intent-flag/RESULTS.md:28; launch-intent-v2/NOTES.md:14; RESULT_V8F_LAUNCH_INTENT.md:305,305; MLAUNCH NOTES.md:26; R2C_NOTE.md:12,12; lit1_logs/chain.out:17,18; check_result_filled.out:117,117,136,136 |
| `9.412e-06` | 1981,2050 | RESULT_V8F_SPEEDSUB_T1.md:77,404 |
| `77853e56` | 1982,2326 | RESULT_V8F_SPEEDSUB_T1.md:15; PREREG_V8F_SPEEDSUB_T1.md:30; PREREG_V8F_LAUNCH_INTENT.md:116; RESULT_V8F_DAG_ORC.md:7 |
| `41` | 1995,2397,2476,2476,2488 | RESULT_V8F_SPEEDSUB_T1.md:3,434; PREREG_V8F_LAUNCH_INTENT.md:319; PREREG_V8F_DAG_ORC.md:318,426; RESULT_V8F_DAG_ORC.md:274; launch-intent-flag/RESULTS.md:36; RESULT_V8F_LAUNCH_INTENT.md:447,447,458; chain_v8f_launchintent.out:40,41,42,43 |
| `09` | 1995 | RESULT_V8F_SPEEDSUB_T1.md:3,3,425,432...; PREREG_V8F_LAUNCH_INTENT.md:57; PREREG_V8F_DAG_ORC.md:380; RESULT_V8F_DAG_ORC.md:259,263; GOALS_AND_CLAIMS.md (register rows):1,2,2; REP_S0_SPEED_BIAS.md:3; git log (cited commits):31,34; R2C_NOTE.md:21,31,37; lit1_logs/launch_intent_read_REFUSED.out:22,31 |
| `29` | 1995 | RESULT_V8F_SPEEDSUB_T1.md:3,434,462,462; PREREG_V8F_LAUNCH_INTENT.md:52,53; git log (cited commits):19,28,28; MLAUNCH.json:14; MLAUNCH NOTES.md:36; lit1_logs/chain.out:23,24,25; read_A1/read_A1.out:28 |
| `47` | 1995,2479 | RESULT_V8F_SPEEDSUB_T1.md:3,424,426,427; launch-intent-flag/RESULTS.md:32; git log (cited commits):10; RESULT_V8F_LAUNCH_INTENT.md:450; chain_v8f_launchintent.out:101,102,103 |
| `48` | 1995,2393,2394,2394,2481... | RESULT_V8F_SPEEDSUB.md:314; RESULT_V8F_SPEEDSUB_T1.md:3; RESULT_V8F_DAG_ORC.md:261; launch-intent-flag/f_measure.json:39; RESULT_V8F_LAUNCH_INTENT.md:363,443; chain_v8f_launchintent.out:112,115,116,117; lit1_logs/chain.out:40,41,49; check_result_filled.out:100,120 |
| `454.4` | 1995 | RESULT_V8F_SPEEDSUB_T1.md:3,365,434; SPEEDSUB_T1_READ.json:7076 |
| `-0.39` | 2004,2195 | RESULT_V8F_SPEEDSUB_T1.md:29,311 |
| `-0.13` | 2004,2196 | RESULT_V8F_SPEEDSUB_T1.md:29,312 |
| `-3.23` | 2004 | RESULT_V8F_SPEEDSUB_T1.md:29 |
| `-0.55` | 2004,2197 | RESULT_V8F_SPEEDSUB_T1.md:29,313 |
| `2.2421` | 2005,2586 | RESULT_V8F_SPEEDSUB_T1.md:30; RESULT_V8F_DAG_ORC.md:240; RESULT_V8F_LAUNCH_INTENT.md:76 |
| `2.2189` | 2005 | RESULT_V8F_SPEEDSUB_T1.md:30 |
| `2.3201` | 2005,2588 | RESULT_V8F_SPEEDSUB_T1.md:30; RESULT_V8F_LAUNCH_INTENT.md:78 |
| `2.3125` | 2005 | RESULT_V8F_SPEEDSUB_T1.md:30 |
| `2.6446` | 2005,2587 | RESULT_V8F_SPEEDSUB_T1.md:30; RESULT_V8F_LAUNCH_INTENT.md:77 |
| `2.4497` | 2005 | RESULT_V8F_SPEEDSUB_T1.md:30 |
| `2.3317` | 2005,2589 | RESULT_V8F_SPEEDSUB_T1.md:30; RESULT_V8F_LAUNCH_INTENT.md:79 |
| `2.2984` | 2005 | RESULT_V8F_SPEEDSUB_T1.md:30 |
| `19` | 2023,2279,2294,2499,2500... | RESULT_V8F_SPEEDSUB_T1.md:50; PREREG_V8F_SPEEDSUB_T1.md:79; PREREG_V8F_LAUNCH_INTENT.md:163,178,346; PREREG_V8F_DAG_ORC.md:324; RESULT_V8F_DAG_ORC.md:63,63,89,89; GOALS_AND_CLAIMS.md (register rows):3,3; REP_S0_SPEED_BIAS.md:15; launch-intent-flag/RESULTS.md:3,34; launch-intent-v2/NOTES.md:14; P0_RF0.json (rebase):15,550,562; RESULT_V8F_LAUNCH_INTENT.md:292; lit1_logs/chain.out:26; check_result_filled.out:6 |
| `+0.1598` | 2023,2062,2189,2220 | RESULT_V8F_SPEEDSUB_T1.md:50,88,316,341 |
| `+0.0743` | 2023,2062,2190,2220 | RESULT_V8F_SPEEDSUB_T1.md:50,88,166,316... |
| `+0.2611` | 2023,2062,2190,2220 | RESULT_V8F_SPEEDSUB_T1.md:50,88,316,341 |
| `+1.3443` | 2023,2067,2190,2222 | RESULT_V8F_SPEEDSUB_T1.md:50,93,316,343 |
| `+0.2401` | 2023,2067,2190,2222 | RESULT_V8F_SPEEDSUB_T1.md:50,93,316,343 |
| `+2.5747` | 2023,2067,2190,2222 | RESULT_V8F_SPEEDSUB_T1.md:50,93,316,343 |
| `+0.3627` | 2023,2070,2190,2224 | RESULT_V8F_SPEEDSUB_T1.md:50,96,316,345 |
| `+0.1230` | 2023,2070,2190,2224 | RESULT_V8F_SPEEDSUB_T1.md:50,96,316,345 |
| `+0.6572` | 2023,2070,2190,2224 | RESULT_V8F_SPEEDSUB_T1.md:50,96,316,345 |
| `+0.0421` | 2025,2078,2150,2164,2733... | RESULT_V8F_SPEEDSUB_T1.md:9,52,104,255...; GOALS_AND_CLAIMS.md (register rows):5; git log (cited commits):22; RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0246` | 2025,2078,2151,2164,2688 | RESULT_V8F_SPEEDSUB_T1.md:52,104,255,276; GOALS_AND_CLAIMS.md (register rows):5; RESULT_V8F_LAUNCH_INTENT.md:248 |
| `+0.0612` | 2025,2078,2151,2164 | RESULT_V8F_SPEEDSUB_T1.md:52,104,255,276; PREREG_V8F_DAG_ORC.md:24; GOALS_AND_CLAIMS.md (register rows):5 |
| `+4.2928` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148; GOALS_AND_CLAIMS.md (register rows):5 |
| `+3.4386` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+5.1945` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+3.9857` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148; GOALS_AND_CLAIMS.md (register rows):5 |
| `+3.2389` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+4.7695` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+4.4851` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148; GOALS_AND_CLAIMS.md (register rows):5 |
| `+3.5747` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+5.4322` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+4.1886` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148; GOALS_AND_CLAIMS.md (register rows):5 |
| `+3.3996` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `+5.0549` | 2034,2048 | RESULT_V8F_SPEEDSUB_T1.md:75,148 |
| `55` | 2041,2391,2392,2491,2491 | RESULT_V8F_SPEEDSUB_T1.md:68,419,431,434; PREREG_V8F_SPEEDSUB_T1.md:90; RESULT_V8F_LAUNCH_INTENT.md:461,461; chain_v8f_launchintent.out:11,12,13,14; lit1_logs/chain.out:5,46,48,49...; read_A1/read_A1.out:19,20,29,30 |
| `1.9073e-06` | 2042,2042 | RESULT_V8F_SPEEDSUB_T1.md:69,69 |
| `7197` | 2042,2046 | RESULT_V8F_SPEEDSUB_T1.md:69,73; SPEEDSUB_T1_READ.json:183,184,185,220...; MLAUNCH NOTES.md:33; lit1_logs/launch_intent_read_REFUSED.out:2,5,8,11...; read_A1/read_A1.out:3,6,9,12... |
| `1.0` | 2043,2043,2043 | RESULT_V8F_SPEEDSUB.md:237,297; RESULT_V8F_SPEEDSUB_T1.md:70,70,70; PREREG_V8F_SPEEDSUB.md:131; PREREG_V8F_SPEEDSUB_T1.md:175; PREREG_V8F_LAUNCH_INTENT.md:52; PREREG_V8F_DAG_ORC.md:58,359; GOALS_AND_CLAIMS.md (register rows):3; SPEEDSUB_T1_READ.json:674,675,686,695...; MLAUNCH.json:14; MLAUNCH NOTES.md:36; REP run records (s0, s1):6,48,55,159... |
| `84` | 2047 | RESULT_V8F_SPEEDSUB_T1.md:74; PREREG_V8F_LAUNCH_INTENT.md:62 |
| `3.02e-05` | 2047 | RESULT_V8F_SPEEDSUB_T1.md:74 |
| `76.7` | 2051 | RESULT_V8F_SPEEDSUB_T1.md:77 |
| `-0.0072` | 2060,2068,2218 | RESULT_V8F_SPEEDSUB_T1.md:86,94,339; SPEEDSUB_T1_READ.json:934,1651,5180,6107 |
| `-0.0371` | 2060,2078,2218 | RESULT_V8F_SPEEDSUB_T1.md:86,104,255,339; SPEEDSUB_T1_READ.json:935,1751,5181,6964 |
| `+0.0209` | 2060,2218 | RESULT_V8F_SPEEDSUB_T1.md:86,339 |
| `+0.0179` | 2060,2218,2416,2608 | RESULT_V8F_SPEEDSUB_T1.md:86,339; RESULT_V8F_LAUNCH_INTENT.md:29,153 |
| `-0.0143` | 2060,2218 | RESULT_V8F_SPEEDSUB_T1.md:86,339; SPEEDSUB_T1_READ.json:1147,5194 |
| `+0.0516` | 2060,2218 | RESULT_V8F_SPEEDSUB_T1.md:86,339 |
| `-0.1061` | 2060,2157 | RESULT_V8F_SPEEDSUB_T1.md:86,269; SPEEDSUB_T1_READ.json:1358,5206 |
| `-0.1495` | 2060,2157 | RESULT_V8F_SPEEDSUB_T1.md:86,269; SPEEDSUB_T1_READ.json:1359,5207 |
| `-0.0664` | 2060,2157 | RESULT_V8F_SPEEDSUB_T1.md:86,269; SPEEDSUB_T1_READ.json:1360,5208 |
| `-0.0193` | 2060 | RESULT_V8F_SPEEDSUB_T1.md:86; SPEEDSUB_T1_READ.json:1570,5219 |
| `-0.0597` | 2060 | RESULT_V8F_SPEEDSUB_T1.md:86; SPEEDSUB_T1_READ.json:1571,5220 |
| `+0.0202` | 2060 | RESULT_V8F_SPEEDSUB_T1.md:86; RESULT_V8F_DAG_ORC.md:99,141,154 |
| `-0.0715` | 2061,2219 | RESULT_V8F_SPEEDSUB_T1.md:87,340; SPEEDSUB_T1_READ.json:945,5291 |
| `+0.0146` | 2061,2219,2225,2652 | RESULT_V8F_SPEEDSUB_T1.md:87,340,346; RESULT_V8F_LAUNCH_INTENT.md:201; MLAUNCH NOTES.md:15 |
| `-0.0616` | 2061,2219 | RESULT_V8F_SPEEDSUB_T1.md:87,340; SPEEDSUB_T1_READ.json:1157,5304; REP_S0_SPEED_BIAS.md:24 |
| `+0.0365` | 2061,2113,2219 | RESULT_V8F_SPEEDSUB_T1.md:87,208,340 |
| `-0.2064` | 2061,2158 | RESULT_V8F_SPEEDSUB_T1.md:87,270; SPEEDSUB_T1_READ.json:1368,5316 |
| `-0.2785` | 2061,2158 | RESULT_V8F_SPEEDSUB_T1.md:87,270; SPEEDSUB_T1_READ.json:1369,5317 |
| `-0.1393` | 2061,2158 | RESULT_V8F_SPEEDSUB_T1.md:87,270; SPEEDSUB_T1_READ.json:1370,5318 |
| `-0.0367` | 2061 | RESULT_V8F_SPEEDSUB_T1.md:87; SPEEDSUB_T1_READ.json:1580,5329 |
| `-0.0886` | 2061 | RESULT_V8F_SPEEDSUB_T1.md:87; SPEEDSUB_T1_READ.json:1581,5330 |
| `+0.0129` | 2061,2227 | RESULT_V8F_SPEEDSUB_T1.md:87,348 |
| `+0.0387` | 2062,2220,2547,2633 | RESULT_V8F_SPEEDSUB_T1.md:88,341; RESULT_V8F_LAUNCH_INTENT.md:98,180 |
| `-0.0176` | 2062,2220,2629 | RESULT_V8F_SPEEDSUB_T1.md:88,341; SPEEDSUB_T1_READ.json:955,5401; RESULT_V8F_LAUNCH_INTENT.md:176; LAUNCH_INTENT_READ.json:4250,5152 |
| `+0.0916` | 2062,2220 | RESULT_V8F_SPEEDSUB_T1.md:88,341 |
| `-0.1463` | 2062,2159 | RESULT_V8F_SPEEDSUB_T1.md:88,271; SPEEDSUB_T1_READ.json:1379,5427 |
| `-0.0057` | 2062,2159,2659 | RESULT_V8F_SPEEDSUB_T1.md:88,271; SPEEDSUB_T1_READ.json:1380,5428; RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:4715,6195 |
| `-0.1388` | 2062 | RESULT_V8F_SPEEDSUB_T1.md:88; SPEEDSUB_T1_READ.json:1591,5440 |
| `+0.0937` | 2062 | RESULT_V8F_SPEEDSUB_T1.md:88 |
| `-0.0081` | 2063 | RESULT_V8F_SPEEDSUB_T1.md:89; SPEEDSUB_T1_READ.json:965,5511; MLAUNCH.json:389 |
| `+0.0076` | 2063 | RESULT_V8F_SPEEDSUB_T1.md:89 |
| `-0.0034` | 2063,2063,2632,2712,2712 | RESULT_V8F_SPEEDSUB_T1.md:89,89,152; RESULT_V8F_DAG_ORC.md:99; SPEEDSUB_T1_READ.json:1176,1600,2876,3300...; DAGORC_READ.json:3057,3595; RESULT_V8F_LAUNCH_INTENT.md:179,272,272; LAUNCH_INTENT_READ.json:1869,2407,3483,5583... |
| `-0.0144` | 2063,2643 | RESULT_V8F_SPEEDSUB_T1.md:89; SPEEDSUB_T1_READ.json:1177,5524; RESULT_V8F_LAUNCH_INTENT.md:192; LAUNCH_INTENT_READ.json:4376,5339 |
| `+0.0065` | 2063,2226 | RESULT_V8F_SPEEDSUB_T1.md:89,347 |
| `+0.0101` | 2063,2100,2551,2642 | RESULT_V8F_SPEEDSUB_T1.md:89,189; RESULT_V8F_LAUNCH_INTENT.md:102,191 |
| `+0.0215` | 2063,2544 | RESULT_V8F_SPEEDSUB_T1.md:89; RESULT_V8F_LAUNCH_INTENT.md:95 |
| `-0.0124` | 2063 | RESULT_V8F_SPEEDSUB_T1.md:89; SPEEDSUB_T1_READ.json:1601,5550 |
| `+0.0043` | 2063 | RESULT_V8F_SPEEDSUB_T1.md:89 |
| `+0.0036` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `-0.1355` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90; SPEEDSUB_T1_READ.json:975,5621 |
| `+0.1564` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `+0.0091` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `-0.1350` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `+0.1376` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `-0.0621` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90; RESULT_V8F_DAG_ORC.md:82; SPEEDSUB_T1_READ.json:1398,5646; DAGORC_READ.json:764; RF_REP.json:105; LAUNCH_INTENT_READ.json:1190 |
| `-0.2061` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90; SPEEDSUB_T1_READ.json:1399,5647 |
| `+0.0685` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `-0.1682` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90; SPEEDSUB_T1_READ.json:1611,5660 |
| `+0.1280` | 2064 | RESULT_V8F_SPEEDSUB_T1.md:90 |
| `+0.0394` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91 |
| `+0.1431` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91 |
| `-0.0149` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91; SPEEDSUB_T1_READ.json:1196,5743 |
| `-0.0366` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91; SPEEDSUB_T1_READ.json:1197,5744 |
| `+0.0062` | 2065,2079,2165 | RESULT_V8F_SPEEDSUB_T1.md:91,105,256,277 |
| `-0.0391` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91; SPEEDSUB_T1_READ.json:1409,5757 |
| `+0.0102` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91 |
| `+0.0016` | 2065,2549,2634 | RESULT_V8F_SPEEDSUB_T1.md:91; RESULT_V8F_DAG_ORC.md:79,160; RESULT_V8F_LAUNCH_INTENT.md:100,181 |
| `-0.0342` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91; RESULT_V8F_DAG_ORC.md:72,153; SPEEDSUB_T1_READ.json:1621,5770; DAGORC_READ.json:1700,2238,3845 |
| `+0.0345` | 2065 | RESULT_V8F_SPEEDSUB_T1.md:91 |
| `+0.1714` | 2066,2221 | RESULT_V8F_SPEEDSUB_T1.md:92,342 |
| `-0.1698` | 2066,2221 | RESULT_V8F_SPEEDSUB_T1.md:92,342; SPEEDSUB_T1_READ.json:995,5848 |
| `+0.4806` | 2066,2221 | RESULT_V8F_SPEEDSUB_T1.md:92,342 |
| `+0.0260` | 2066,2217,2221,2572,2720 | RESULT_V8F_SPEEDSUB_T1.md:92,338,342,351; GOALS_AND_CLAIMS.md (register rows):5; RESULT_V8F_LAUNCH_INTENT.md:124,280 |
| `-0.2861` | 2066,2221 | RESULT_V8F_SPEEDSUB_T1.md:92,342; SPEEDSUB_T1_READ.json:1207,5861 |
| `+0.2939` | 2066,2221 | RESULT_V8F_SPEEDSUB_T1.md:92,342 |
| `-0.5464` | 2066,2160 | RESULT_V8F_SPEEDSUB_T1.md:92,272; SPEEDSUB_T1_READ.json:1418,5873 |
| `-1.0054` | 2066,2160 | RESULT_V8F_SPEEDSUB_T1.md:92,272; SPEEDSUB_T1_READ.json:1419,5874 |
| `-0.1855` | 2066,2160 | RESULT_V8F_SPEEDSUB_T1.md:92,272; SPEEDSUB_T1_READ.json:1420,5875 |
| `+0.0769` | 2066 | RESULT_V8F_SPEEDSUB_T1.md:92 |
| `-0.1957` | 2066,2761 | RESULT_V8F_SPEEDSUB_T1.md:92; SPEEDSUB_T1_READ.json:1631,5887; MLAUNCH.json:1719; MLAUNCH NOTES.md:72 |
| `+0.3433` | 2066 | RESULT_V8F_SPEEDSUB_T1.md:92 |
| `-0.4443` | 2067,2222 | RESULT_V8F_SPEEDSUB_T1.md:93,343; SPEEDSUB_T1_READ.json:1004,5957 |
| `-2.4850` | 2067,2222 | RESULT_V8F_SPEEDSUB_T1.md:93,343 |
| `+1.1922` | 2067,2222 | RESULT_V8F_SPEEDSUB_T1.md:93,343 |
| `-3.7761` | 2067,2161 | RESULT_V8F_SPEEDSUB_T1.md:93,273; SPEEDSUB_T1_READ.json:1428,5983 |
| `-5.9452` | 2067,2161 | RESULT_V8F_SPEEDSUB_T1.md:93,273; SPEEDSUB_T1_READ.json:1429,5984 |
| `-1.9696` | 2067,2161 | RESULT_V8F_SPEEDSUB_T1.md:93,273; SPEEDSUB_T1_READ.json:1430,5985 |
| `+0.0811` | 2067 | RESULT_V8F_SPEEDSUB_T1.md:93,166 |
| `-0.8026` | 2067 | RESULT_V8F_SPEEDSUB_T1.md:93; SPEEDSUB_T1_READ.json:1641,5997 |
| `+0.9589` | 2067 | RESULT_V8F_SPEEDSUB_T1.md:93 |
| `+0.0056` | 2068,2223 | RESULT_V8F_SPEEDSUB_T1.md:94,344 |
| `+0.0110` | 2068,2223 | RESULT_V8F_SPEEDSUB_T1.md:94,344 |
| `-0.0463` | 2068,2162 | RESULT_V8F_SPEEDSUB_T1.md:94,274; SPEEDSUB_T1_READ.json:1438,6093 |
| `-0.0747` | 2068,2162 | RESULT_V8F_SPEEDSUB_T1.md:94,274; SPEEDSUB_T1_READ.json:1439,6094 |
| `-0.0225` | 2068,2162 | RESULT_V8F_SPEEDSUB_T1.md:94,274; RESULT_V8F_DAG_ORC.md:82; SPEEDSUB_T1_READ.json:1440,6095; DAGORC_READ.json:765; RF_REP.json:106; P0_RF0.json (rebase):168; LAUNCH_INTENT_READ.json:1191 |
| `+0.0045` | 2068,2077,2226,2691 | RESULT_V8F_SPEEDSUB_T1.md:94,103,317,347; RESULT_V8F_LAUNCH_INTENT.md:251 |
| `+0.0173` | 2068 | RESULT_V8F_SPEEDSUB_T1.md:94 |
| `+0.0057` | 2069,2630 | RESULT_V8F_SPEEDSUB_T1.md:95; RESULT_V8F_LAUNCH_INTENT.md:177 |
| `-0.0111` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95; SPEEDSUB_T1_READ.json:1025,6178; P0_RF0.json (rebase):231 |
| `+0.0224` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95; RESULT_V8F_DAG_ORC.md:79 |
| `-0.0166` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95,372; RESULT_V8F_DAG_ORC.md:79,87,128,193...; GOALS_AND_CLAIMS.md (register rows):3; SPEEDSUB_T1_READ.json:1237,6191; DAGORC_READ.json:721,2177,2715; RF_REP.json:86; LAUNCH_INTENT_READ.json:1147 |
| `+0.0222` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95 |
| `-0.0056` | 2069,2659 | RESULT_V8F_SPEEDSUB_T1.md:95; SPEEDSUB_T1_READ.json:1448,6203; RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:3371,5083 |
| `-0.0229` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95; SPEEDSUB_T1_READ.json:1449,6204; MLAUNCH.json:911; MLAUNCH NOTES.md:21 |
| `+0.0124` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95 |
| `-0.0064` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95; SPEEDSUB_T1_READ.json:1660,6216; MLAUNCH.json:811 |
| `-0.0252` | 2069 | RESULT_V8F_SPEEDSUB_T1.md:95; PREREG_V8F_DAG_ORC.md:27; SPEEDSUB_T1_READ.json:1661,6217 |
| `+0.0585` | 2070,2224 | RESULT_V8F_SPEEDSUB_T1.md:96,345 |
| `-0.1062` | 2070,2224 | RESULT_V8F_SPEEDSUB_T1.md:96,345; SPEEDSUB_T1_READ.json:1035,6288 |
| `+0.2145` | 2070,2224 | RESULT_V8F_SPEEDSUB_T1.md:96,345 |
| `-0.1063` | 2070 | RESULT_V8F_SPEEDSUB_T1.md:96; SPEEDSUB_T1_READ.json:1458,6313 |
| `-0.2761` | 2070 | RESULT_V8F_SPEEDSUB_T1.md:96; SPEEDSUB_T1_READ.json:1459,6314 |
| `+0.0587` | 2070 | RESULT_V8F_SPEEDSUB_T1.md:96 |
| `-0.0644` | 2070 | RESULT_V8F_SPEEDSUB_T1.md:96; SPEEDSUB_T1_READ.json:1670,6326 |
| `-0.3605` | 2070 | RESULT_V8F_SPEEDSUB_T1.md:96; SPEEDSUB_T1_READ.json:1671,6327 |
| `+0.1731` | 2070 | RESULT_V8F_SPEEDSUB_T1.md:96 |
| `-0.0004` | 2071,2225,2652 | RESULT_V8F_SPEEDSUB_T1.md:97,346; GOALS_AND_CLAIMS.md (register rows):3,3,3; SPEEDSUB_T1_READ.json:1044,6397; RESULT_V8F_LAUNCH_INTENT.md:201; LAUNCH_INTENT_READ.json:3356,5055 |
| `-0.0175` | 2071,2225 | RESULT_V8F_SPEEDSUB_T1.md:97,346; RESULT_V8F_DAG_ORC.md:74,155; SPEEDSUB_T1_READ.json:1045,6398; DAGORC_READ.json:1995,2533,4138 |
| `+0.0145` | 2071,2225 | RESULT_V8F_SPEEDSUB_T1.md:97,346; RESULT_V8F_DAG_ORC.md:113,124 |
| `-0.0023` | 2071,2225 | RESULT_V8F_SPEEDSUB_T1.md:97,346; SPEEDSUB_T1_READ.json:1257,6411 |
| `+0.0240` | 2071,2225 | RESULT_V8F_SPEEDSUB_T1.md:97,346; RESULT_V8F_DAG_ORC.md:113,128 |
| `-0.0347` | 2071,2163 | RESULT_V8F_SPEEDSUB_T1.md:97,275; SPEEDSUB_T1_READ.json:1468,6423 |
| `-0.0674` | 2071,2163 | RESULT_V8F_SPEEDSUB_T1.md:97,275; SPEEDSUB_T1_READ.json:1469,6424 |
| `-0.0082` | 2071 | RESULT_V8F_SPEEDSUB_T1.md:97; SPEEDSUB_T1_READ.json:1680,6436 |
| `-0.0218` | 2071 | RESULT_V8F_SPEEDSUB_T1.md:97; SPEEDSUB_T1_READ.json:1681,6437 |
| `-0.0092` | 2072,2073,2227,2228,2632 | RESULT_V8F_SPEEDSUB_T1.md:98,99,348,349; RESULT_V8F_DAG_ORC.md:81; SPEEDSUB_T1_READ.json:1055,1065,6514,6624; DAGORC_READ.json:1018; RF_REP.json:215; RESULT_V8F_LAUNCH_INTENT.md:179; LAUNCH_INTENT_READ.json:1444,3216,4842 |
| `-0.0104` | 2072,2079,2227 | RESULT_V8F_SPEEDSUB_T1.md:98,105,256,348; SPEEDSUB_T1_READ.json:1267,1337,6527,7030 |
| `+0.0200` | 2072,2227 | RESULT_V8F_SPEEDSUB_T1.md:98,348; RESULT_V8F_DAG_ORC.md:74 |
| `+0.0187` | 2072 | RESULT_V8F_SPEEDSUB_T1.md:98 |
| `+0.0121` | 2073,2078,2228 | RESULT_V8F_SPEEDSUB_T1.md:99,104,255,349 |
| `+0.0359` | 2073,2228 | RESULT_V8F_SPEEDSUB_T1.md:99,349 |
| `-0.0325` | 2073,2228 | RESULT_V8F_SPEEDSUB_T1.md:99,349; SPEEDSUB_T1_READ.json:1277,6637 |
| `+0.0321` | 2073,2228 | RESULT_V8F_SPEEDSUB_T1.md:99,349 |
| `+0.0158` | 2073,2651 | RESULT_V8F_SPEEDSUB_T1.md:99; RESULT_V8F_DAG_ORC.md:83,164; RESULT_V8F_LAUNCH_INTENT.md:200 |
| `-0.0050` | 2073,2078,2079,2632,2688 | RESULT_V8F_SPEEDSUB_T1.md:99,104,105,255...; RESULT_V8F_LAUNCH_INTENT.md:179,248 |
| `+0.0367` | 2073 | RESULT_V8F_SPEEDSUB_T1.md:99 |
| `-0.0179` | 2073,2736 | RESULT_V8F_SPEEDSUB_T1.md:99; SPEEDSUB_T1_READ.json:1701,6663; RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0280` | 2073 | RESULT_V8F_SPEEDSUB_T1.md:99; PREREG_V8F_DAG_ORC.md:26 |
| `-0.0007` | 2077,2226,2236,2539,2642 | RESULT_V8F_SPEEDSUB_T1.md:103,317,347,372; SPEEDSUB_T1_READ.json:1104; RESULT_V8F_LAUNCH_INTENT.md:90,191; LAUNCH_INTENT_READ.json:3285,4948 |
| `-0.0059` | 2077,2226,2552,2645 | RESULT_V8F_SPEEDSUB_T1.md:103,317,347; SPEEDSUB_T1_READ.json:1105; RESULT_V8F_LAUNCH_INTENT.md:103,194; LAUNCH_INTENT_READ.json:4672,6121 |
| `-0.0052` | 2077,2151 | RESULT_V8F_SPEEDSUB_T1.md:103,279,317; PREREG_V8F_DAG_ORC.md:27; SPEEDSUB_T1_READ.json:1528 |
| `-0.0115` | 2077,2151 | RESULT_V8F_SPEEDSUB_T1.md:103,279,317; RESULT_V8F_DAG_ORC.md:17,69,240; GOALS_AND_CLAIMS.md (register rows):3; SPEEDSUB_T1_READ.json:1529; DAGORC_READ.json:1926,2464; P0_RF0.json (rebase):115 |
| `+0.0007` | 2077,2151,2539 | RESULT_V8F_SPEEDSUB_T1.md:103,279,317; GOALS_AND_CLAIMS.md (register rows):3; RESULT_V8F_LAUNCH_INTENT.md:90 |
| `-0.0003` | 2077 | RESULT_V8F_SPEEDSUB_T1.md:103,317; SPEEDSUB_T1_READ.json:1740; MLAUNCH.json:1808; MLAUNCH NOTES.md:74 |
| `-0.0049` | 2077 | RESULT_V8F_SPEEDSUB_T1.md:103,317; SPEEDSUB_T1_READ.json:1741; P0_RF0.json (rebase):306 |
| `+0.0308` | 2078 | RESULT_V8F_SPEEDSUB_T1.md:104,255 |
| `+0.0033` | 2078,2079,2690 | RESULT_V8F_SPEEDSUB_T1.md:104,105,255,256; RESULT_V8F_LAUNCH_INTENT.md:250 |
| `-0.0184` | 2078,2102,2661 | RESULT_V8F_SPEEDSUB_T1.md:104,191,255; SPEEDSUB_T1_READ.json:1327,1834,5784,6944; RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:3399,5123 |
| `+0.0267` | 2078 | RESULT_V8F_SPEEDSUB_T1.md:104,255 |
| `+0.0058` | 2078 | RESULT_V8F_SPEEDSUB_T1.md:104,255 |
| `+0.0075` | 2079,2540 | RESULT_V8F_SPEEDSUB_T1.md:105,256; RESULT_V8F_LAUNCH_INTENT.md:91 |
| `+0.0138` | 2079,2165 | RESULT_V8F_SPEEDSUB_T1.md:105,256,277 |
| `-0.0063` | 2079 | RESULT_V8F_SPEEDSUB_T1.md:105,256; SPEEDSUB_T1_READ.json:1760,7049 |
| `-0.0150` | 2079 | RESULT_V8F_SPEEDSUB_T1.md:105,256 |
| `15` | 2086,2367,2393,2417,2417... | RESULT_V8F_SPEEDSUB_T1.md:140,170,170; PREREG_V8F_SPEEDSUB_T1.md:167; PREREG_V8F_LAUNCH_INTENT.md:280,354; PREREG_V8F_DAG_ORC.md:246; RESULT_V8F_DAG_ORC.md:19,19,23,23...; GOALS_AND_CLAIMS.md (register rows):3; launch-intent-v2/enumerate_cells_li.out:6; P0_RF0.json (rebase):485; git log (cited commits):7,7,43; RESULT_V8F_LAUNCH_INTENT.md:3,30,30,30...; chain_v8f_launchintent.out:59,60,61,62...; R2C_NOTE.md:4; lit1_logs/chain.out:1,2,3,4...; lit1_logs/chain.out.prev_1028151251:1,2,3,4...; lit1_logs/launch_intent_read_REFUSED.out:15; check_result_filled.out:93,94,94,94... |
| `-0.7833` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:1784,5234 |
| `-0.9856` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:1785,5235 |
| `-0.6121` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:1786,5236 |
| `-0.6850` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186 |
| `-0.8833` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:1997,5248 |
| `-0.5135` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:1998,5249 |
| `-0.5449` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:2208,5260 |
| `-0.7564` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:2209,5261 |
| `-0.3517` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:2210,5262 |
| `-0.6785` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:2420,5273 |
| `-0.8810` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186 |
| `-0.5045` | 2097 | RESULT_V8F_SPEEDSUB_T1.md:186; SPEEDSUB_T1_READ.json:2422,5275 |
| `-0.7091` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:1794,5344 |
| `-0.8648` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:1795,5345 |
| `-0.5573` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:1796,5346 |
| `-0.5994` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2006,5357 |
| `-0.7792` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2007,5358 |
| `-0.4222` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2008,5359 |
| `-0.4277` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2218,5370 |
| `-0.6426` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2219,5371 |
| `-0.2186` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2220,5372 |
| `-0.6126` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2430,5383 |
| `-0.7887` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2431,5384 |
| `-0.4456` | 2098 | RESULT_V8F_SPEEDSUB_T1.md:187; SPEEDSUB_T1_READ.json:2432,5385 |
| `-0.7785` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:1804,5454 |
| `-1.1532` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:1805,5455 |
| `-0.4568` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:1806,5456 |
| `-0.4416` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2016,5467 |
| `-0.7129` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2017,5468 |
| `-0.1980` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188 |
| `-0.3301` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2228,5480 |
| `-0.6892` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2229,5481 |
| `-0.4242` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2440,5493 |
| `-0.7481` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2441,5494 |
| `-0.1227` | 2099 | RESULT_V8F_SPEEDSUB_T1.md:188; SPEEDSUB_T1_READ.json:2442,5495 |
| `-0.0228` | 2100 | RESULT_V8F_SPEEDSUB_T1.md:189; SPEEDSUB_T1_READ.json:1815,5565 |
| `+0.0257` | 2100 | RESULT_V8F_SPEEDSUB_T1.md:189 |
| `-0.0286` | 2100 | RESULT_V8F_SPEEDSUB_T1.md:189; RESULT_V8F_DAG_ORC.md:75,156; SPEEDSUB_T1_READ.json:2027,5578; DAGORC_READ.json:2009,2547,4152; MLAUNCH.json:363; MLAUNCH NOTES.md:15 |
| `+0.0136` | 2100,2226 | RESULT_V8F_SPEEDSUB_T1.md:189,347 |
| `-0.0174` | 2100 | RESULT_V8F_SPEEDSUB_T1.md:189; SPEEDSUB_T1_READ.json:2239,5591 |
| `+0.0373` | 2100 | RESULT_V8F_SPEEDSUB_T1.md:189 |
| `-0.0308` | 2100 | RESULT_V8F_SPEEDSUB_T1.md:189; SPEEDSUB_T1_READ.json:2451,5604 |
| `-0.2641` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:1824,5674 |
| `-0.6818` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:1825,5675 |
| `+0.1348` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190 |
| `-0.4115` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2036,5687 |
| `-0.7749` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2037,5688 |
| `-0.0871` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; PREREG_V8F_SPEEDSUB.md:10; SPEEDSUB_T1_READ.json:2038,5689 |
| `-0.1728` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2248,5700 |
| `-0.5489` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2249,5701 |
| `+0.1538` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190 |
| `-0.3812` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2460,5713 |
| `-0.7886` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2461,5714 |
| `-0.0297` | 2101 | RESULT_V8F_SPEEDSUB_T1.md:190; SPEEDSUB_T1_READ.json:2462,5715; P0_RF0.json (rebase):260 |
| `-0.1328` | 2102 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:1835,5785 |
| `+0.1534` | 2102,2119 | RESULT_V8F_SPEEDSUB_T1.md:191,220 |
| `-0.0874` | 2102 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:2046,5797 |
| `-0.1500` | 2102 | RESULT_V8F_SPEEDSUB_T1.md:191 |
| `-0.0307` | 2102 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:2048,5799 |
| `-0.0468` | 2102,2714 | RESULT_V8F_SPEEDSUB_T1.md:191; RESULT_V8F_DAG_ORC.md:81,123,162,192; SPEEDSUB_T1_READ.json:2258,3322,5810; DAGORC_READ.json:1826,2364,3977; RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:2437,6007 |
| `-0.1105` | 2102,2734 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:2259,5811; RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.0973` | 2102,2713 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:2470,5823; RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:2961,5611 |
| `-0.1557` | 2102 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:2471,5824 |
| `-0.0451` | 2102 | RESULT_V8F_SPEEDSUB_T1.md:191; SPEEDSUB_T1_READ.json:2472,5825 |
| `-2.4206` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:1844,5901 |
| `-3.1566` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:1845,5902 |
| `-1.7506` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:1846,5903 |
| `-2.0279` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:2056,5914 |
| `-2.7477` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:2057,5915 |
| `-1.3526` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:2058,5916 |
| `-0.6310` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203 |
| `-1.3850` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203 |
| `+0.1513` | 2108,2119 | RESULT_V8F_SPEEDSUB_T1.md:203,220 |
| `-1.7095` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:2480,5940 |
| `-2.4988` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:2481,5941 |
| `-0.9451` | 2108 | RESULT_V8F_SPEEDSUB_T1.md:203; SPEEDSUB_T1_READ.json:2482,5942 |
| `+3.0286` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `-0.2135` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204; SPEEDSUB_T1_READ.json:1855,6012 |
| `+6.2640` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+10.0287` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+6.4331` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+13.9961` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+15.5496` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+11.4219` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+20.1882` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+9.6555` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+6.0362` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+13.5666` | 2109 | RESULT_V8F_SPEEDSUB_T1.md:204 |
| `+0.0783` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.0549` | 2110,2563,2693 | RESULT_V8F_SPEEDSUB_T1.md:205; RESULT_V8F_LAUNCH_INTENT.md:115,253 |
| `+0.1043` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.1275` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.0871` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:157,205 |
| `+0.1752` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.1716` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.1294` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.2209` | 2110,2120 | RESULT_V8F_SPEEDSUB_T1.md:205,221 |
| `+0.1388` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `+0.0900` | 2110,2577,2726 | RESULT_V8F_SPEEDSUB_T1.md:205; RESULT_V8F_LAUNCH_INTENT.md:129,286 |
| `+0.1964` | 2110 | RESULT_V8F_SPEEDSUB_T1.md:205 |
| `-0.9563` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:1874,6231 |
| `-1.1041` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:1875,6232 |
| `-0.8301` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:1876,6233 |
| `-0.9708` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2086,6244 |
| `-1.1133` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2087,6245 |
| `-0.8394` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2088,6246 |
| `-0.9841` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2298,6257 |
| `-1.1328` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2299,6258 |
| `-0.8500` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206 |
| `-0.9582` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2510,6270 |
| `-1.1082` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206; SPEEDSUB_T1_READ.json:2511,6271 |
| `-0.8190` | 2111 | RESULT_V8F_SPEEDSUB_T1.md:206 |
| `-2.4032` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:1884,6341 |
| `-3.8112` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:1885,6342 |
| `-1.2015` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:1886,6343 |
| `-1.4846` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2096,6354 |
| `-2.4589` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2097,6355 |
| `-0.5725` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2098,6356 |
| `-1.3428` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2308,6367 |
| `-2.5587` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2309,6368 |
| `-0.2180` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207 |
| `-1.5896` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2520,6380 |
| `-2.7105` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2521,6381 |
| `-0.6001` | 2112 | RESULT_V8F_SPEEDSUB_T1.md:207; SPEEDSUB_T1_READ.json:2522,6382 |
| `+0.0425` | 2113,2631,2698 | RESULT_V8F_SPEEDSUB_T1.md:208; RESULT_V8F_DAG_ORC.md:82,163; RESULT_V8F_LAUNCH_INTENT.md:178,258 |
| `+0.0244` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208 |
| `+0.0643` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208 |
| `+0.0633` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208 |
| `+0.0971` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208; RESULT_V8F_DAG_ORC.md:109,164 |
| `+0.0967` | 2113,2722 | RESULT_V8F_SPEEDSUB_T1.md:208; RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.0614` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208; PREREG_V8F_DAG_ORC.md:27 |
| `+0.1422` | 2113,2735 | RESULT_V8F_SPEEDSUB_T1.md:208; RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0507` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208 |
| `+0.0755` | 2113 | RESULT_V8F_SPEEDSUB_T1.md:208 |
| `+0.1905` | 2119,2697 | RESULT_V8F_SPEEDSUB_T1.md:220; RESULT_V8F_DAG_ORC.md:108; RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.2318` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.2026` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220; RESULT_V8F_DAG_ORC.md:109,164; REP_S0_SPEED_BIAS.md:22 |
| `+0.1668` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.2416` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.1122` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.1971` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.1772` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.1380` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.2198` | 2119 | RESULT_V8F_SPEEDSUB_T1.md:220 |
| `+0.1517` | 2120,2721 | RESULT_V8F_SPEEDSUB_T1.md:221; RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.0898` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:160,221 |
| `+0.2127` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.1363` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.0713` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.1987` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.0604` | 2120,2340,2416,2420,2573... | RESULT_V8F_SPEEDSUB_T1.md:221; PREREG_V8F_LAUNCH_INTENT.md:214; PREREG_V8F_DAG_ORC.md:26,29; RESULT_V8F_DAG_ORC.md:18,22,113,209; GOALS_AND_CLAIMS.md (register rows):3,3,3,3; git log (cited commits):37; RESULT_V8F_LAUNCH_INTENT.md:29,33,81,125...; check_result_filled.out:122 |
| `+0.1250` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.1613` | 2120 | RESULT_V8F_SPEEDSUB_T1.md:221 |
| `+0.0980` | 2120,2220 | RESULT_V8F_SPEEDSUB_T1.md:221,341 |
| `-1.7428` | 2128,2699 | RESULT_V8F_SPEEDSUB_T1.md:229; RESULT_V8F_DAG_ORC.md:110; SPEEDSUB_T1_READ.json:2774; DAGORC_READ.json:2943,3481; RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:1755,5093 |
| `-2.2737` | 2128,2699 | RESULT_V8F_SPEEDSUB_T1.md:229; RESULT_V8F_DAG_ORC.md:110; SPEEDSUB_T1_READ.json:2775; DAGORC_READ.json:2944,3482; RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:1756,5094 |
| `-1.2861` | 2128,2699 | RESULT_V8F_SPEEDSUB_T1.md:229; RESULT_V8F_DAG_ORC.md:110; SPEEDSUB_T1_READ.json:2776; DAGORC_READ.json:2945,3483; RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:1757,5095 |
| `-1.8255` | 2128,2723 | RESULT_V8F_SPEEDSUB_T1.md:229; RESULT_V8F_DAG_ORC.md:110; SPEEDSUB_T1_READ.json:2986; DAGORC_READ.json:3211,3749; RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:2023,5835 |
| `-2.3433` | 2128,2723 | RESULT_V8F_SPEEDSUB_T1.md:229; RESULT_V8F_DAG_ORC.md:110; SPEEDSUB_T1_READ.json:2987; DAGORC_READ.json:3212,3750; RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:2024,5836 |
| `-1.3880` | 2128,2723 | RESULT_V8F_SPEEDSUB_T1.md:229; RESULT_V8F_DAG_ORC.md:110; RESULT_V8F_LAUNCH_INTENT.md:283 |
| `-1.4837` | 2128,2699 | RESULT_V8F_SPEEDSUB_T1.md:229; SPEEDSUB_T1_READ.json:3198; RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:2293,5463 |
| `-1.9587` | 2128,2699 | RESULT_V8F_SPEEDSUB_T1.md:229; SPEEDSUB_T1_READ.json:3199; RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:2294,5464 |
| `-1.0595` | 2128,2699 | RESULT_V8F_SPEEDSUB_T1.md:229; SPEEDSUB_T1_READ.json:3200; RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:2295,5465 |
| `-1.7135` | 2128,2723 | RESULT_V8F_SPEEDSUB_T1.md:229; SPEEDSUB_T1_READ.json:3410; RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:2561,6205 |
| `-2.1974` | 2128,2723 | RESULT_V8F_SPEEDSUB_T1.md:229; SPEEDSUB_T1_READ.json:3411; RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:2562,6206 |
| `-1.3049` | 2128,2723 | RESULT_V8F_SPEEDSUB_T1.md:229; SPEEDSUB_T1_READ.json:3412; RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:2563,6207 |
| `-1.6754` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:1924,6746 |
| `-2.1616` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:1925,6747 |
| `-1.2452` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:1926,6748 |
| `-1.8415` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2136,6759 |
| `-2.3319` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2137,6760 |
| `-1.4113` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2138,6761 |
| `-1.4009` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2348,6772 |
| `-1.8202` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2349,6773 |
| `-1.0285` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2350,6774 |
| `-1.6566` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2560,6785 |
| `-2.1108` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2561,6786 |
| `-1.2707` | 2129 | RESULT_V8F_SPEEDSUB_T1.md:230; SPEEDSUB_T1_READ.json:2562,6787 |
| `-4.5331` | 2131,2700 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_DAG_ORC.md:111; SPEEDSUB_T1_READ.json:2784; DAGORC_READ.json:2957,3495; RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:1769,5113 |
| `-5.6695` | 2131,2700 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_DAG_ORC.md:111; SPEEDSUB_T1_READ.json:2785; DAGORC_READ.json:2958,3496; RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:1770,5114 |
| `-3.5422` | 2131,2700 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_DAG_ORC.md:111; SPEEDSUB_T1_READ.json:2786; DAGORC_READ.json:2959,3497; RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:1771,5115 |
| `-4.6092` | 2131,2724 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_DAG_ORC.md:111; SPEEDSUB_T1_READ.json:2996; DAGORC_READ.json:3225,3763; RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:2037,5855 |
| `-5.6305` | 2131,2724 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_DAG_ORC.md:111; SPEEDSUB_T1_READ.json:2997; DAGORC_READ.json:3226,3764; RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:2038,5856 |
| `-3.6576` | 2131,2724 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_DAG_ORC.md:111; SPEEDSUB_T1_READ.json:2998; DAGORC_READ.json:3227,3765; RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:2039,5857 |
| `-4.1419` | 2131,2700 | RESULT_V8F_SPEEDSUB_T1.md:232; SPEEDSUB_T1_READ.json:3208; RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:2307,5483 |
| `-5.1720` | 2131,2700 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_LAUNCH_INTENT.md:260 |
| `-3.1830` | 2131,2700 | RESULT_V8F_SPEEDSUB_T1.md:232; RESULT_V8F_LAUNCH_INTENT.md:260 |
| `-4.8481` | 2131,2724 | RESULT_V8F_SPEEDSUB_T1.md:232; SPEEDSUB_T1_READ.json:3420; RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:2575,6225 |
| `-5.9214` | 2131,2724 | RESULT_V8F_SPEEDSUB_T1.md:232; SPEEDSUB_T1_READ.json:3421; RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:2576,6226 |
| `-3.8649` | 2131,2724 | RESULT_V8F_SPEEDSUB_T1.md:232; SPEEDSUB_T1_READ.json:3422; RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:2577,6227 |
| `-4.4812` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:1934,6808 |
| `-5.5861` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:1935,6809 |
| `-3.4515` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:1936,6810 |
| `-4.6270` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233 |
| `-5.6365` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2147,6822 |
| `-3.6953` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2148,6823 |
| `-4.1583` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2358,6834 |
| `-5.2174` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2359,6835 |
| `-3.2088` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2360,6836 |
| `-4.6535` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2570,6847 |
| `-5.7059` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2571,6848 |
| `-3.6993` | 2132 | RESULT_V8F_SPEEDSUB_T1.md:233; SPEEDSUB_T1_READ.json:2572,6849 |
| `-9.5624` | 2134,2701 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_DAG_ORC.md:112; SPEEDSUB_T1_READ.json:2794; DAGORC_READ.json:2971,3509; RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:1783,5133 |
| `-11.6843` | 2134,2701 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_DAG_ORC.md:112; SPEEDSUB_T1_READ.json:2795; DAGORC_READ.json:2972,3510; RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:1784,5134 |
| `-7.6124` | 2134,2701 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_DAG_ORC.md:112; SPEEDSUB_T1_READ.json:2796; DAGORC_READ.json:2973,3511; RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:1785,5135 |
| `-9.9249` | 2134,2725 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_DAG_ORC.md:112; SPEEDSUB_T1_READ.json:3006; DAGORC_READ.json:3239,3777; RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:2051,5875 |
| `-11.9750` | 2134,2725 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_DAG_ORC.md:112; RESULT_V8F_LAUNCH_INTENT.md:285 |
| `-7.9686` | 2134,2725 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_DAG_ORC.md:112; SPEEDSUB_T1_READ.json:3008; DAGORC_READ.json:3241,3779; RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:2053,5877 |
| `-9.3703` | 2134,2701 | RESULT_V8F_SPEEDSUB_T1.md:235; SPEEDSUB_T1_READ.json:3218; RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:2321,5503 |
| `-11.3441` | 2134,2701 | RESULT_V8F_SPEEDSUB_T1.md:235; SPEEDSUB_T1_READ.json:3219; RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:2322,5504 |
| `-7.5010` | 2134,2701 | RESULT_V8F_SPEEDSUB_T1.md:235; RESULT_V8F_LAUNCH_INTENT.md:261 |
| `-10.4419` | 2134,2725 | RESULT_V8F_SPEEDSUB_T1.md:235; SPEEDSUB_T1_READ.json:3430; RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:2589,6245 |
| `-12.5718` | 2134,2725 | RESULT_V8F_SPEEDSUB_T1.md:235; SPEEDSUB_T1_READ.json:3431; RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:2590,6246 |
| `-8.3877` | 2134,2725 | RESULT_V8F_SPEEDSUB_T1.md:235; SPEEDSUB_T1_READ.json:3432; RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:2591,6247 |
| `-9.6310` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236 |
| `-11.7915` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:1945,6871 |
| `-7.6545` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:1946,6872 |
| `-9.8671` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2156,6883 |
| `-11.8118` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2157,6884 |
| `-7.9544` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2158,6885 |
| `-9.5742` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2368,6896 |
| `-11.6457` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2369,6897 |
| `-7.6143` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2370,6898 |
| `-10.1261` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2580,6909 |
| `-12.2716` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2581,6910 |
| `-8.1095` | 2135 | RESULT_V8F_SPEEDSUB_T1.md:236; SPEEDSUB_T1_READ.json:2582,6911 |
| `3.23` | 2149 | RESULT_V8F_SPEEDSUB_T1.md:264 |
| `0.2885` | 2150 | RESULT_V8F_SPEEDSUB_T1.md:260,279 |
| `0.3306` | 2150 | RESULT_V8F_SPEEDSUB_T1.md:260,279 |
| `-1.1097` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:22 |
| `-1.2814` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:22 |
| `-0.9345` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:22 |
| `-0.7315` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:23 |
| `-0.8635` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:23 |
| `-0.5901` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:23 |
| `-1.0056` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:24 |
| `-1.1436` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:24 |
| `-0.8686` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:24 |
| `-1.1700` | 2168 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:25 |
| `-1.3210` | 2169 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:25 |
| `-1.0175` | 2169 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:25 |
| `-0.9913` | 2169 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:26 |
| `-1.1383` | 2169 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:26 |
| `-0.8487` | 2169 | RESULT_V8F_SPEEDSUB_T1.md:283; REP_S0_SPEED_BIAS.md:26 |
| `0.5` | 2171,2175,2208,2728,2751 | RESULT_V8F_SPEEDSUB.md:236; RESULT_V8F_SPEEDSUB_T1.md:287,289,328; PREREG_V8F_SPEEDSUB_T1.md:141; PREREG_V8F_LAUNCH_INTENT.md:28,221,221; PREREG_V8F_DAG_ORC.md:58,359,476,476; GOALS_AND_CLAIMS.md (register rows):3; launch-intent-flag/MM_REDESIGN_2026-10-28.md:67; launch-intent-v2/NOTES.md:12,32; RESULT_V8F_LAUNCH_INTENT.md:296,387; LAUNCH_INTENT_READ.json:876,907,938,969; MLAUNCH.json:1292,1296,1337,1378...; MLAUNCH NOTES.md:58; REP run records (s0, s1):45,49,54,198... |
| `0.4025` | 2172,2298,2505,2959 | RESULT_V8F_SPEEDSUB_T1.md:9,295,295; PREREG_V8F_LAUNCH_INTENT.md:182; RESULT_V8F_DAG_ORC.md:43,219,240,244; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:889; DAGORC_READ.json:509,581; RF_REP.json:3,26; git log (cited commits):22; RESULT_V8F_LAUNCH_INTENT.md:52; LAUNCH_INTENT_READ.json:711,778,1007 |
| `0.2309` | 2172,2959 | RESULT_V8F_SPEEDSUB_T1.md:9,295,295,295; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_READ.json:894; git log (cited commits):22 |
| `42.6` | 2172,2177 | RESULT_V8F_SPEEDSUB_T1.md:291,295 |
| `+0.4025` | 2177,2876,2955 | RESULT_V8F_SPEEDSUB_T1.md:291; PREREG_V8F_SPEEDSUB_T1.md:14; RESULT_V8F_DAG_ORC.md:69,219,234; GOALS_AND_CLAIMS.md (register rows):3; REP_S0_SPEED_BIAS.md:16; git log (cited commits):37 |
| `+0.2309` | 2177 | RESULT_V8F_SPEEDSUB_T1.md:291 |
| `0.574` | 2177 | RESULT_V8F_SPEEDSUB_T1.md:291 |
| `+0.2013` | 2177 | RESULT_V8F_SPEEDSUB_T1.md:291 |
| `+15.8553` | 2178 | RESULT_V8F_SPEEDSUB_T1.md:292; RESULT_V8F_DAG_ORC.md:77,221 |
| `+12.5214` | 2178 | RESULT_V8F_SPEEDSUB_T1.md:292 |
| `0.790` | 2178 | RESULT_V8F_SPEEDSUB_T1.md:292 |
| `21.0` | 2178,2562 | RESULT_V8F_SPEEDSUB_T1.md:292; RESULT_V8F_LAUNCH_INTENT.md:114 |
| `+7.9276` | 2178 | RESULT_V8F_SPEEDSUB_T1.md:292 |
| `+0.1324` | 2179 | RESULT_V8F_SPEEDSUB_T1.md:293; PREREG_V8F_SPEEDSUB_T1.md:14; RESULT_V8F_DAG_ORC.md:78,220; REP_S0_SPEED_BIAS.md:16 |
| `+0.0932` | 2179 | RESULT_V8F_SPEEDSUB_T1.md:293 |
| `0.704` | 2179 | RESULT_V8F_SPEEDSUB_T1.md:293 |
| `29.6` | 2179 | RESULT_V8F_SPEEDSUB_T1.md:293 |
| `+0.0662` | 2179 | RESULT_V8F_SPEEDSUB_T1.md:293 |
| `+0.0116` | 2185 | RESULT_V8F_SPEEDSUB_T1.md:299,303; RESULT_V8F_DAG_ORC.md:69,219,234; GOALS_AND_CLAIMS.md (register rows):3 |
| `-0.0141` | 2185 | RESULT_V8F_SPEEDSUB_T1.md:299; SPEEDSUB_T1_READ.json:913; MLAUNCH.json:1731; MLAUNCH NOTES.md:72 |
| `+0.8903` | 2186 | RESULT_V8F_SPEEDSUB_T1.md:300,303; RESULT_V8F_DAG_ORC.md:77,221 |
| `-0.3732` | 2186 | RESULT_V8F_SPEEDSUB_T1.md:300; SPEEDSUB_T1_READ.json:914 |
| `+0.0112` | 2187 | RESULT_V8F_SPEEDSUB_T1.md:301; PREREG_V8F_DAG_ORC.md:26; MLAUNCH NOTES.md:22 |
| `12.8` | 2200,2205 | RESULT_V8F_SPEEDSUB_T1.md:325,330 |
| `27.6` | 2200,2208,2876 | RESULT_V8F_SPEEDSUB_T1.md:321,328,330,372; RESULT_V8F_DAG_ORC.md:209,211 |
| `79.9` | 2205,2876 | RESULT_V8F_SPEEDSUB_T1.md:321,325,372; RESULT_V8F_DAG_ORC.md:202,202,211 |
| `-15.5` | 2205 | RESULT_V8F_SPEEDSUB_T1.md:325 |
| `66.9` | 2205 | RESULT_V8F_SPEEDSUB_T1.md:325; RESULT_V8F_DAG_ORC.md:206 |
| `19.5` | 2205 | RESULT_V8F_SPEEDSUB_T1.md:325,330 |
| `-0.8` | 2205 | RESULT_V8F_SPEEDSUB_T1.md:325 |
| `52.1` | 2206 | RESULT_V8F_SPEEDSUB_T1.md:326; RESULT_V8F_DAG_ORC.md:203 |
| `5.7` | 2206 | RESULT_V8F_SPEEDSUB_T1.md:326; PREREG_V8F_SPEEDSUB.md:148; PREREG_V8F_LAUNCH_INTENT.md:75; PREREG_V8F_DAG_ORC.md:37 |
| `52.0` | 2206 | RESULT_V8F_SPEEDSUB_T1.md:326; RESULT_V8F_DAG_ORC.md:207 |
| `21.3` | 2206 | RESULT_V8F_SPEEDSUB_T1.md:326 |
| `-3.4` | 2206,2569 | RESULT_V8F_SPEEDSUB_T1.md:326; RESULT_V8F_LAUNCH_INTENT.md:121 |
| `4.1` | 2207 | RESULT_V8F_SPEEDSUB.md:190; RESULT_V8F_SPEEDSUB_T1.md:327; RESULT_V8F_LAUNCH_INTENT.md:290 |
| `56.4` | 2207 | RESULT_V8F_SPEEDSUB_T1.md:327; RESULT_V8F_DAG_ORC.md:204 |
| `-19.7` | 2207 | RESULT_V8F_SPEEDSUB_T1.md:327 |
| `58.8` | 2207 | RESULT_V8F_SPEEDSUB_T1.md:327; RESULT_V8F_DAG_ORC.md:208,208 |
| `26.4` | 2207 | RESULT_V8F_SPEEDSUB_T1.md:327,330; RESULT_V8F_DAG_ORC.md:126,229 |
| `13.9` | 2207,2543 | RESULT_V8F_SPEEDSUB_T1.md:327; RESULT_V8F_LAUNCH_INTENT.md:94 |
| `1.9` | 2208 | RESULT_V8F_SPEEDSUB_T1.md:328; RESULT_V8F_DAG_ORC.md:265 |
| `46.0` | 2208 | RESULT_V8F_SPEEDSUB_T1.md:328; RESULT_V8F_DAG_ORC.md:205 |
| `-3.5` | 2208 | RESULT_V8F_SPEEDSUB_T1.md:328 |
| `5.4` | 2208 | RESULT_V8F_SPEEDSUB_T1.md:328,330 |
| `-0.0374` | 2217 | RESULT_V8F_SPEEDSUB_T1.md:338,351; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_BESIDE.json:9 |
| `+0.0375` | 2217,2564,2696 | RESULT_V8F_SPEEDSUB_T1.md:338,351; GOALS_AND_CLAIMS.md (register rows):5; RESULT_V8F_LAUNCH_INTENT.md:116,256 |
| `-0.0637` | 2217 | RESULT_V8F_SPEEDSUB_T1.md:338,351; GOALS_AND_CLAIMS.md (register rows):5; SPEEDSUB_T1_BESIDE.json:182 |
| `+0.0059` | 2218,2630 | RESULT_V8F_SPEEDSUB_T1.md:339; RESULT_V8F_LAUNCH_INTENT.md:177 |
| `-0.0226` | 2218 | RESULT_V8F_SPEEDSUB_T1.md:339; SPEEDSUB_T1_BESIDE.json:18; MLAUNCH.json:676 |
| `+0.0333` | 2218,2652 | RESULT_V8F_SPEEDSUB_T1.md:339; RESULT_V8F_LAUNCH_INTENT.md:201 |
| `-0.0010` | 2218 | RESULT_V8F_SPEEDSUB_T1.md:339 |
| `-0.0331` | 2218 | RESULT_V8F_SPEEDSUB_T1.md:339; SPEEDSUB_T1_BESIDE.json:191 |
| `+0.0305` | 2218 | RESULT_V8F_SPEEDSUB_T1.md:339 |
| `+0.0015` | 2219 | RESULT_V8F_SPEEDSUB_T1.md:340 |
| `-0.0377` | 2219 | RESULT_V8F_SPEEDSUB_T1.md:340; SPEEDSUB_T1_BESIDE.json:27 |
| `+0.0399` | 2219 | RESULT_V8F_SPEEDSUB_T1.md:340 |
| `-0.0753` | 2219 | RESULT_V8F_SPEEDSUB_T1.md:340; SPEEDSUB_T1_BESIDE.json:200 |
| `+0.0239` | 2219 | RESULT_V8F_SPEEDSUB_T1.md:340 |
| `+0.0312` | 2220,2735 | RESULT_V8F_SPEEDSUB_T1.md:341; RESULT_V8F_LAUNCH_INTENT.md:302 |
| `-0.0256` | 2220 | RESULT_V8F_SPEEDSUB_T1.md:341; SPEEDSUB_T1_BESIDE.json:36 |
| `+0.0881` | 2220 | RESULT_V8F_SPEEDSUB_T1.md:341 |
| `+0.0154` | 2220 | RESULT_V8F_SPEEDSUB_T1.md:341 |
| `+0.1916` | 2220 | RESULT_V8F_SPEEDSUB_T1.md:341 |
| `+0.2038` | 2221 | RESULT_V8F_SPEEDSUB_T1.md:342 |
| `-0.1045` | 2221 | RESULT_V8F_SPEEDSUB_T1.md:342; SPEEDSUB_T1_BESIDE.json:72 |
| `+0.4759` | 2221 | RESULT_V8F_SPEEDSUB_T1.md:342 |
| `+0.2147` | 2221 | RESULT_V8F_SPEEDSUB_T1.md:342 |
| `-0.0596` | 2221 | RESULT_V8F_SPEEDSUB_T1.md:342; SPEEDSUB_T1_BESIDE.json:245 |
| `+0.4457` | 2221 | RESULT_V8F_SPEEDSUB_T1.md:342 |
| `+0.6521` | 2222 | RESULT_V8F_SPEEDSUB_T1.md:343 |
| `-1.0941` | 2222 | RESULT_V8F_SPEEDSUB_T1.md:343; SPEEDSUB_T1_BESIDE.json:81 |
| `+2.2367` | 2222 | RESULT_V8F_SPEEDSUB_T1.md:343 |
| `+2.0000` | 2222 | RESULT_V8F_SPEEDSUB_T1.md:343 |
| `+1.0303` | 2222 | RESULT_V8F_SPEEDSUB_T1.md:343 |
| `+3.1066` | 2222 | RESULT_V8F_SPEEDSUB_T1.md:343 |
| `+0.0035` | 2223 | RESULT_V8F_SPEEDSUB_T1.md:344; REP_S0_SPEED_BIAS.md:24 |
| `-0.0097` | 2223 | RESULT_V8F_SPEEDSUB_T1.md:344; PREREG_V8F_SPEEDSUB.md:10; SPEEDSUB_T1_BESIDE.json:90 |
| `+0.0152` | 2223 | RESULT_V8F_SPEEDSUB_T1.md:344 |
| `+0.0019` | 2223 | RESULT_V8F_SPEEDSUB_T1.md:344 |
| `-0.0201` | 2223,2629 | RESULT_V8F_SPEEDSUB_T1.md:344; SPEEDSUB_T1_BESIDE.json:263; RESULT_V8F_LAUNCH_INTENT.md:176; LAUNCH_INTENT_READ.json:3173,4781 |
| `+0.0188` | 2223 | RESULT_V8F_SPEEDSUB_T1.md:344 |
| `+0.1149` | 2224 | RESULT_V8F_SPEEDSUB_T1.md:345 |
| `-0.0623` | 2224 | RESULT_V8F_SPEEDSUB_T1.md:345; SPEEDSUB_T1_BESIDE.json:108 |
| `+0.3086` | 2224 | RESULT_V8F_SPEEDSUB_T1.md:345 |
| `+0.3345` | 2224 | RESULT_V8F_SPEEDSUB_T1.md:345 |
| `+0.1144` | 2224 | RESULT_V8F_SPEEDSUB_T1.md:345 |
| `+0.5729` | 2224,2542,2641 | RESULT_V8F_SPEEDSUB_T1.md:345; RESULT_V8F_LAUNCH_INTENT.md:93,190 |
| `+0.0142` | 2225 | RESULT_V8F_SPEEDSUB_T1.md:346 |
| `-0.0045` | 2225 | RESULT_V8F_SPEEDSUB_T1.md:346; SPEEDSUB_T1_BESIDE.json:117 |
| `+0.0319` | 2225 | RESULT_V8F_SPEEDSUB_T1.md:346 |
| `+0.0277` | 2225 | RESULT_V8F_SPEEDSUB_T1.md:346 |
| `+0.0068` | 2226,2548 | RESULT_V8F_SPEEDSUB_T1.md:347; RESULT_V8F_LAUNCH_INTENT.md:99 |
| `+0.0004` | 2226 | RESULT_V8F_SPEEDSUB_T1.md:347; GOALS_AND_CLAIMS.md (register rows):3 |
| `+0.0071` | 2227,2651 | RESULT_V8F_SPEEDSUB_T1.md:348; RESULT_V8F_LAUNCH_INTENT.md:200 |
| `+0.0204` | 2227,2228,2634 | RESULT_V8F_SPEEDSUB_T1.md:348,349; RESULT_V8F_LAUNCH_INTENT.md:181 |
| `-0.0171` | 2227 | RESULT_V8F_SPEEDSUB_T1.md:348; SPEEDSUB_T1_BESIDE.json:299; P0_RF0.json (rebase):352; MLAUNCH.json:785 |
| `+0.0475` | 2228 | RESULT_V8F_SPEEDSUB_T1.md:349 |
| `-0.0079` | 2228 | RESULT_V8F_SPEEDSUB_T1.md:349; SPEEDSUB_T1_BESIDE.json:308 |
| `+0.0667` | 2228 | RESULT_V8F_SPEEDSUB_T1.md:349 |
| `1.6` | 2231 | RESULT_V8F_SPEEDSUB_T1.md:365 |
| `0.2793` | 2234 | RESULT_V8F_SPEEDSUB_T1.md:359,371; PREREG_V8F_SPEEDSUB_T1.md:197 |
| `6.1` | 2246 | RESULT_V8F_SPEEDSUB_T1.md:384; PREREG_V8F_SPEEDSUB_T1.md:34,117,124; RESULT_V8F_LAUNCH_INTENT.md:359 |
| `138` | 2253 | RESULT_V8F_SPEEDSUB_T1.md:365,406,417,427 |
| `305` | 2254 | RESULT_V8F_SPEEDSUB_T1.md:406; PREREG_V8F_SPEEDSUB_T1.md:175; PREREG_V8F_LAUNCH_INTENT.md:326; launch-intent-flag/MM_REDESIGN_2026-10-28.md:73 |
| `52` | 2256,2447,2477,2477,2848 | RESULT_V8F_SPEEDSUB_T1.md:365,428,434,434...; git log (cited commits):10,28; RESULT_V8F_LAUNCH_INTENT.md:8,430,448,448; lit1_logs/chain.out:2,3,40,42...; check_result_filled.out:92 |
| `77234276` | 2264,2848,2849 | launch-intent-chain/PINS_FILLED.txt:1,3,4; RESULT_V8F_LAUNCH_INTENT.md:3,430,432; check_result_filled.out:88 |
| `-1.78` | 2271 | PREREG_V8F_LAUNCH_INTENT.md:34; PREREG_V8F_DAG_ORC.md:300 |
| `-0.93` | 2271 | PREREG_V8F_LAUNCH_INTENT.md:34; PREREG_V8F_DAG_ORC.md:300 |
| `49618` | 2276 | PREREG_V8F_LAUNCH_INTENT.md:81,339; launch-intent-flag/RESULTS.md:43; launch-intent-flag/f_measure.json:21 |
| `0.161851` | 2276 | PREREG_V8F_LAUNCH_INTENT.md:81,339; launch-intent-flag/RESULTS.md:43; launch-intent-v2/enumerate_cells_li.out:48,48 |
| `1.3` | 2279,2287,2539 | PREREG_V8F_SPEEDSUB.md:22,148; PREREG_V8F_LAUNCH_INTENT.md:170,326; PREREG_V8F_DAG_ORC.md:327; GOALS_AND_CLAIMS.md (register rows):4,6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:24; RESULT_V8F_LAUNCH_INTENT.md:90 |
| `9.1` | 2279,2284 | PREREG_V8F_LAUNCH_INTENT.md:167; GOALS_AND_CLAIMS.md (register rows):6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:21 |
| `31.7106` | 2284,2306 | PREREG_V8F_LAUNCH_INTENT.md:167,190; RESULT_V8F_DAG_ORC.md:51,121,125,221; launch-intent-flag/MM_REDESIGN_2026-10-28.md:21; launch-intent-v2/enumerate_cells_li.out:67; DAGORC_READ.json:291,399,559 |
| `3.4726` | 2284 | PREREG_V8F_LAUNCH_INTENT.md:167; SPEEDSUB_T1_READ.json:2704; launch-intent-flag/MM_REDESIGN_2026-10-28.md:21; DAGORC_READ.json:2845,3383; LAUNCH_INTENT_READ.json:1657,4940 |
| `8.6844` | 2284 | PREREG_V8F_LAUNCH_INTENT.md:167; SPEEDSUB_T1_READ.json:2916; launch-intent-flag/MM_REDESIGN_2026-10-28.md:21; DAGORC_READ.json:3113,3651; LAUNCH_INTENT_READ.json:1925,5682 |
| `3.7` | 2284 | PREREG_V8F_LAUNCH_INTENT.md:167; PREREG_V8F_DAG_ORC.md:488; launch-intent-flag/MM_REDESIGN_2026-10-28.md:21; RESULT_V8F_LAUNCH_INTENT.md:9,183,226,412 |
| `1.1070` | 2284,2306,2513,2538,2542... | PREREG_V8F_LAUNCH_INTENT.md:167,190,346; GOALS_AND_CLAIMS.md (register rows):6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:50; launch-intent-v2/NOTES.md:16; RESULT_V8F_LAUNCH_INTENT.md:60,89,93,97... |
| `0.32` | 2284 | PREREG_V8F_LAUNCH_INTENT.md:167 |
| `0.13` | 2284 | RESULT_V8F_SPEEDSUB_T1.md:316; PREREG_V8F_LAUNCH_INTENT.md:167 |
| `0.2648` | 2285,2307 | PREREG_V8F_LAUNCH_INTENT.md:168,191; RESULT_V8F_DAG_ORC.md:52,122,126,220; launch-intent-flag/MM_REDESIGN_2026-10-28.md:22; launch-intent-v2/enumerate_cells_li.out:67; DAGORC_READ.json:292,400,560 |
| `0.0854` | 2285 | PREREG_V8F_LAUNCH_INTENT.md:168; SPEEDSUB_T1_READ.json:2714; launch-intent-flag/MM_REDESIGN_2026-10-28.md:22; DAGORC_READ.json:2859,3397; LAUNCH_INTENT_READ.json:1671,4960; MLAUNCH.json:1791 |
| `0.1352` | 2285 | PREREG_V8F_LAUNCH_INTENT.md:168; SPEEDSUB_T1_READ.json:2926; launch-intent-flag/MM_REDESIGN_2026-10-28.md:22; DAGORC_READ.json:3127,3665; LAUNCH_INTENT_READ.json:1939,5702 |
| `3.1` | 2285,2676 | PREREG_V8F_LAUNCH_INTENT.md:168; PREREG_V8F_DAG_ORC.md:173,184; RESULT_V8F_DAG_ORC.md:8,145,179; launch-intent-flag/MM_REDESIGN_2026-10-28.md:22; DAGORC_READ.json:4349; RESULT_V8F_LAUNCH_INTENT.md:163,231; LAUNCH_INTENT_READ.json:6261 |
| `2.0` | 2285 | RESULT_V8F_SPEEDSUB.md:238; PREREG_V8F_SPEEDSUB.md:37; PREREG_V8F_LAUNCH_INTENT.md:168; PREREG_V8F_DAG_ORC.md:58,359; GOALS_AND_CLAIMS.md (register rows):3; SPEEDSUB_T1_BESIDE.json:253; launch-intent-flag/MM_REDESIGN_2026-10-28.md:22 |
| `0.0526` | 2285,2307,2514,2539,2543... | PREREG_V8F_LAUNCH_INTENT.md:168,191,346; GOALS_AND_CLAIMS.md (register rows):6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:50; launch-intent-v2/NOTES.md:16; launch-intent-v2/enumerate_cells_li.out:67; RESULT_V8F_LAUNCH_INTENT.md:61,90,94,98...; LAUNCH_INTENT_READ.json:762,825 |
| `0.62` | 2285 | PREREG_V8F_LAUNCH_INTENT.md:168 |
| `0.39` | 2285 | PREREG_V8F_LAUNCH_INTENT.md:168 |
| `0.1740` | 2286,2310 | PREREG_V8F_LAUNCH_INTENT.md:169,194; RESULT_V8F_DAG_ORC.md:55,123,127; launch-intent-flag/MM_REDESIGN_2026-10-28.md:23 |
| `0.0443` | 2286 | PREREG_V8F_LAUNCH_INTENT.md:169; SPEEDSUB_T1_READ.json:2744; launch-intent-flag/MM_REDESIGN_2026-10-28.md:23; DAGORC_READ.json:2901,3439; LAUNCH_INTENT_READ.json:1713,5020 |
| `0.0529` | 2286 | PREREG_V8F_LAUNCH_INTENT.md:169; SPEEDSUB_T1_READ.json:2956; launch-intent-flag/MM_REDESIGN_2026-10-28.md:23; DAGORC_READ.json:3169,3707; LAUNCH_INTENT_READ.json:1981,5762 |
| `3.9` | 2286 | PREREG_V8F_LAUNCH_INTENT.md:169; RESULT_V8F_DAG_ORC.md:265; launch-intent-flag/MM_REDESIGN_2026-10-28.md:23 |
| `3.3` | 2286,2567 | PREREG_V8F_SPEEDSUB.md:148; PREREG_V8F_LAUNCH_INTENT.md:169; RESULT_V8F_DAG_ORC.md:8,176,331; launch-intent-flag/MM_REDESIGN_2026-10-28.md:23; RESULT_V8F_LAUNCH_INTENT.md:119,185 |
| `0.31` | 2286 | PREREG_V8F_LAUNCH_INTENT.md:169 |
| `0.26` | 2286,2522 | PREREG_V8F_LAUNCH_INTENT.md:169; RESULT_V8F_LAUNCH_INTENT.md:69 |
| `0.0776` | 2287,2316 | PREREG_V8F_LAUNCH_INTENT.md:170,200; RESULT_V8F_DAG_ORC.md:61,124,128,222; launch-intent-flag/MM_REDESIGN_2026-10-28.md:24; launch-intent-v2/enumerate_cells_li.out:67; DAGORC_READ.json:301,409,569; LAUNCH_INTENT_READ.json:4575,5975 |
| `0.0568` | 2287 | PREREG_V8F_LAUNCH_INTENT.md:170; SPEEDSUB_T1_READ.json:2804; launch-intent-flag/MM_REDESIGN_2026-10-28.md:24; DAGORC_READ.json:2985,3523; LAUNCH_INTENT_READ.json:1797 |
| `1.4` | 2287 | PREREG_V8F_LAUNCH_INTENT.md:23,125,170,320...; RESULT_V8F_DAG_ORC.md:311; GOALS_AND_CLAIMS.md (register rows):3; launch-intent-flag/RESULTS.md:53; launch-intent-flag/MM_REDESIGN_2026-10-28.md:24,73; launch-intent-chain/LAUNCH.txt:100 |
| `0.0096` | 2287,2316,2523,2541,2545... | PREREG_V8F_LAUNCH_INTENT.md:170,200,346; GOALS_AND_CLAIMS.md (register rows):6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:50; launch-intent-v2/NOTES.md:16; launch-intent-v2/enumerate_cells_li.out:67; DAGORC_READ.json:1840,2378,3997; RESULT_V8F_LAUNCH_INTENT.md:70,92,96,100...; LAUNCH_INTENT_READ.json:771,870 |
| `0.17` | 2287,2500,2509 | PREREG_V8F_LAUNCH_INTENT.md:170; RESULT_V8F_LAUNCH_INTENT.md:56 |
| `0.16` | 2287 | PREREG_V8F_LAUNCH_INTENT.md:170; SPEEDSUB_T1_READ.json:2967; DAGORC_READ.json:3184,3722; LAUNCH_INTENT_READ.json:1996,5789 |
| `0.0138` | 2298,2505 | PREREG_V8F_LAUNCH_INTENT.md:182; RESULT_V8F_DAG_ORC.md:43,219; SPEEDSUB_T1_READ.json:1548,7039; DAGORC_READ.json:488; P0_RF0.json (rebase):29,36,462; RESULT_V8F_LAUNCH_INTENT.md:52; LAUNCH_INTENT_READ.json:690,779; MLAUNCH.json:813 |
| `0.2414` | 2298,2505 | PREREG_V8F_LAUNCH_INTENT.md:182; RESULT_V8F_LAUNCH_INTENT.md:52; LAUNCH_INTENT_READ.json:753,780 |
| `0.8050` | 2298 | PREREG_V8F_LAUNCH_INTENT.md:182; RESULT_V8F_DAG_ORC.md:43,219,240,244 |
| `0.0796` | 2299,2506 | PREREG_V8F_LAUNCH_INTENT.md:183; RESULT_V8F_DAG_ORC.md:44; DAGORC_READ.json:489; P0_RF0.json (rebase):52,59,463; RESULT_V8F_LAUNCH_INTENT.md:53; LAUNCH_INTENT_READ.json:691,784 |
| `0.1592` | 2299,2506 | PREREG_V8F_LAUNCH_INTENT.md:183; RESULT_V8F_LAUNCH_INTENT.md:53; LAUNCH_INTENT_READ.json:754,785,4447,5453 |
| `0.3373` | 2299,2506 | PREREG_V8F_LAUNCH_INTENT.md:183; RESULT_V8F_DAG_ORC.md:44; DAGORC_READ.json:510,595; RF_REP.json:4,32; RESULT_V8F_LAUNCH_INTENT.md:53; LAUNCH_INTENT_READ.json:712,783,1021 |
| `0.6746` | 2299 | PREREG_V8F_LAUNCH_INTENT.md:183; RESULT_V8F_DAG_ORC.md:44; DAGORC_READ.json:284,392,552 |
| `0.0201` | 2300,2507 | PREREG_V8F_LAUNCH_INTENT.md:184; RESULT_V8F_DAG_ORC.md:45; DAGORC_READ.json:490; P0_RF0.json (rebase):75,82,464; RESULT_V8F_LAUNCH_INTENT.md:54; LAUNCH_INTENT_READ.json:692,789,2790,5015 |
| `0.0402` | 2300,2507 | PREREG_V8F_LAUNCH_INTENT.md:184; RESULT_V8F_LAUNCH_INTENT.md:54; LAUNCH_INTENT_READ.json:755,790 |
| `0.4596` | 2300,2507 | PREREG_V8F_LAUNCH_INTENT.md:184; RESULT_V8F_DAG_ORC.md:45,241,244; DAGORC_READ.json:511,609; RF_REP.json:5,38; RESULT_V8F_LAUNCH_INTENT.md:54; LAUNCH_INTENT_READ.json:713,788,1035 |
| `0.9192` | 2300 | PREREG_V8F_LAUNCH_INTENT.md:184; RESULT_V8F_DAG_ORC.md:45,241; DAGORC_READ.json:285,393,553 |
| `0.1323` | 2301,2508 | PREREG_V8F_LAUNCH_INTENT.md:185; RESULT_V8F_DAG_ORC.md:46; DAGORC_READ.json:491; P0_RF0.json (rebase):91,105,465; RESULT_V8F_LAUNCH_INTENT.md:55; LAUNCH_INTENT_READ.json:693,794 |
| `0.2646` | 2301,2508 | PREREG_V8F_LAUNCH_INTENT.md:185; RESULT_V8F_LAUNCH_INTENT.md:55; LAUNCH_INTENT_READ.json:756,795 |
| `0.5603` | 2301,2508 | PREREG_V8F_LAUNCH_INTENT.md:185; RESULT_V8F_DAG_ORC.md:46; DAGORC_READ.json:512,623; RF_REP.json:6,44; RESULT_V8F_LAUNCH_INTENT.md:55; LAUNCH_INTENT_READ.json:714,793,1049 |
| `1.1206` | 2301 | PREREG_V8F_LAUNCH_INTENT.md:185; RESULT_V8F_DAG_ORC.md:46; DAGORC_READ.json:286,394,554 |
| `0.0050` | 2302,2509 | PREREG_V8F_LAUNCH_INTENT.md:186; RESULT_V8F_DAG_ORC.md:47; RESULT_V8F_LAUNCH_INTENT.md:56 |
| `0.0100` | 2302,2302,2509 | PREREG_V8F_LAUNCH_INTENT.md:186,186; RESULT_V8F_DAG_ORC.md:47,141,242,244; RESULT_V8F_LAUNCH_INTENT.md:56 |
| `0.0017` | 2302,2509 | PREREG_V8F_LAUNCH_INTENT.md:186; RESULT_V8F_DAG_ORC.md:47,242,244; SPEEDSUB_T1_READ.json:1762,1814,2664,3728...; DAGORC_READ.json:513,2789,3327; RF_REP.json:7; RESULT_V8F_LAUNCH_INTENT.md:56; LAUNCH_INTENT_READ.json:715,798,1601,3767... |
| `0.1308` | 2303,2510 | PREREG_V8F_LAUNCH_INTENT.md:187; RESULT_V8F_DAG_ORC.md:48; DAGORC_READ.json:493; P0_RF0.json (rebase):137,151,467; RESULT_V8F_LAUNCH_INTENT.md:57; LAUNCH_INTENT_READ.json:695,804 |
| `0.2616` | 2303,2510 | PREREG_V8F_LAUNCH_INTENT.md:187; RESULT_V8F_LAUNCH_INTENT.md:57; LAUNCH_INTENT_READ.json:758,805 |
| `0.2540` | 2303,2510 | PREREG_V8F_LAUNCH_INTENT.md:187; RESULT_V8F_DAG_ORC.md:48; RESULT_V8F_LAUNCH_INTENT.md:57 |
| `0.5080` | 2303 | PREREG_V8F_LAUNCH_INTENT.md:187; RESULT_V8F_DAG_ORC.md:48 |
| `0.0041` | 2304,2511 | PREREG_V8F_LAUNCH_INTENT.md:188; RESULT_V8F_DAG_ORC.md:49; DAGORC_READ.json:494; P0_RF0.json (rebase):167,174,468; RESULT_V8F_LAUNCH_INTENT.md:58; LAUNCH_INTENT_READ.json:696,809 |
| `0.0082` | 2304,2511 | PREREG_V8F_LAUNCH_INTENT.md:188; RESULT_V8F_LAUNCH_INTENT.md:58; LAUNCH_INTENT_READ.json:759,810 |
| `0.1042` | 2304,2511 | PREREG_V8F_LAUNCH_INTENT.md:188; RESULT_V8F_DAG_ORC.md:49; DAGORC_READ.json:515,665; RF_REP.json:9,62; RESULT_V8F_LAUNCH_INTENT.md:58; LAUNCH_INTENT_READ.json:717,808,1091 |
| `0.2084` | 2304 | PREREG_V8F_LAUNCH_INTENT.md:188; RESULT_V8F_DAG_ORC.md:49; DAGORC_READ.json:289,397,557 |
| `0.3801` | 2305,2512 | PREREG_V8F_LAUNCH_INTENT.md:189; RESULT_V8F_DAG_ORC.md:50; DAGORC_READ.json:495; P0_RF0.json (rebase):197,469; RESULT_V8F_LAUNCH_INTENT.md:59; LAUNCH_INTENT_READ.json:697,814 |
| `0.7602` | 2305,2512 | PREREG_V8F_LAUNCH_INTENT.md:189; RESULT_V8F_LAUNCH_INTENT.md:59; LAUNCH_INTENT_READ.json:760,815 |
| `2.5074` | 2305,2512 | PREREG_V8F_LAUNCH_INTENT.md:189; RESULT_V8F_DAG_ORC.md:50; DAGORC_READ.json:516,679; RF_REP.json:10,68; RESULT_V8F_LAUNCH_INTENT.md:59; LAUNCH_INTENT_READ.json:718,813,1105 |
| `5.0148` | 2305 | PREREG_V8F_LAUNCH_INTENT.md:189; RESULT_V8F_DAG_ORC.md:50; DAGORC_READ.json:290,398,558 |
| `0.5535` | 2306,2513 | PREREG_V8F_LAUNCH_INTENT.md:190; RESULT_V8F_DAG_ORC.md:51,221; DAGORC_READ.json:496; P0_RF0.json (rebase):220,470; RESULT_V8F_LAUNCH_INTENT.md:60; LAUNCH_INTENT_READ.json:698,819 |
| `15.8553` | 2306,2513 | PREREG_V8F_LAUNCH_INTENT.md:190; RESULT_V8F_DAG_ORC.md:51,221; GOALS_AND_CLAIMS.md (register rows):3,5; SPEEDSUB_T1_READ.json:890; DAGORC_READ.json:517,693; RF_REP.json:11,74; RESULT_V8F_LAUNCH_INTENT.md:60; LAUNCH_INTENT_READ.json:719,818,1119 |
| `0.1324` | 2307,2514 | PREREG_V8F_LAUNCH_INTENT.md:191; RESULT_V8F_DAG_ORC.md:52,220; GOALS_AND_CLAIMS.md (register rows):3,5; SPEEDSUB_T1_READ.json:891; DAGORC_READ.json:518,707; RF_REP.json:12,80; RESULT_V8F_LAUNCH_INTENT.md:61; LAUNCH_INTENT_READ.json:720,823,1133 |
| `0.0320` | 2308,2515 | PREREG_V8F_LAUNCH_INTENT.md:192; RESULT_V8F_DAG_ORC.md:53; RESULT_V8F_LAUNCH_INTENT.md:62 |
| `0.0640` | 2308,2308,2515 | PREREG_V8F_LAUNCH_INTENT.md:192,192; RESULT_V8F_DAG_ORC.md:53; RESULT_V8F_LAUNCH_INTENT.md:62 |
| `0.0224` | 2308,2515 | PREREG_V8F_LAUNCH_INTENT.md:192; RESULT_V8F_DAG_ORC.md:53; SPEEDSUB_T1_READ.json:1026,6179; DAGORC_READ.json:519,989; RF_REP.json:13,202; RESULT_V8F_LAUNCH_INTENT.md:62; LAUNCH_INTENT_READ.json:721,828,1415 |
| `0.3985` | 2309,2516 | PREREG_V8F_LAUNCH_INTENT.md:193; RESULT_V8F_DAG_ORC.md:54; DAGORC_READ.json:499; P0_RF0.json (rebase):282,289,473; RESULT_V8F_LAUNCH_INTENT.md:63; LAUNCH_INTENT_READ.json:701,834 |
| `0.7970` | 2309,2516 | PREREG_V8F_LAUNCH_INTENT.md:193; RESULT_V8F_LAUNCH_INTENT.md:63 |
| `1.2242` | 2309,2516 | PREREG_V8F_LAUNCH_INTENT.md:193; RESULT_V8F_DAG_ORC.md:54; DAGORC_READ.json:520,735; RF_REP.json:14,92; RESULT_V8F_LAUNCH_INTENT.md:63; LAUNCH_INTENT_READ.json:722,833,1161 |
| `2.4484` | 2309 | PREREG_V8F_LAUNCH_INTENT.md:193; RESULT_V8F_DAG_ORC.md:54; DAGORC_READ.json:294,402,562 |
| `0.0068` | 2310,2517 | PREREG_V8F_LAUNCH_INTENT.md:194; RESULT_V8F_DAG_ORC.md:55; SPEEDSUB_T1_BESIDE.json:170; DAGORC_READ.json:500; P0_RF0.json (rebase):312,474; RESULT_V8F_LAUNCH_INTENT.md:64; LAUNCH_INTENT_READ.json:702,839 |
| `0.0870` | 2310,2517 | PREREG_V8F_LAUNCH_INTENT.md:194; RESULT_V8F_DAG_ORC.md:55; GOALS_AND_CLAIMS.md (register rows):3; RESULT_V8F_LAUNCH_INTENT.md:64 |
| `0.0183` | 2311,2518 | PREREG_V8F_LAUNCH_INTENT.md:195; RESULT_V8F_DAG_ORC.md:56; DAGORC_READ.json:501; P0_RF0.json (rebase):335,475; RESULT_V8F_LAUNCH_INTENT.md:65; LAUNCH_INTENT_READ.json:703,844 |
| `0.0366` | 2311,2518 | PREREG_V8F_LAUNCH_INTENT.md:195; RESULT_V8F_LAUNCH_INTENT.md:65; LAUNCH_INTENT_READ.json:766,845 |
| `0.0421` | 2311,2518 | PREREG_V8F_LAUNCH_INTENT.md:195; RESULT_V8F_DAG_ORC.md:56; SPEEDSUB_T1_READ.json:1538,6953; DAGORC_READ.json:522; RF_REP.json:16; RESULT_V8F_LAUNCH_INTENT.md:65; LAUNCH_INTENT_READ.json:724,843 |
| `0.0842` | 2311 | PREREG_V8F_LAUNCH_INTENT.md:195; RESULT_V8F_DAG_ORC.md:56; DAGORC_READ.json:296,404,564 |
| `0.0125` | 2312,2519 | PREREG_V8F_LAUNCH_INTENT.md:196; RESULT_V8F_DAG_ORC.md:57; DAGORC_READ.json:502; P0_RF0.json (rebase):358,476; RESULT_V8F_LAUNCH_INTENT.md:66; LAUNCH_INTENT_READ.json:704,849 |
| `0.0250` | 2312,2519 | PREREG_V8F_LAUNCH_INTENT.md:196; RESULT_V8F_LAUNCH_INTENT.md:66 |
| `0.0950` | 2312,2519 | PREREG_V8F_LAUNCH_INTENT.md:196; RESULT_V8F_DAG_ORC.md:57; RESULT_V8F_LAUNCH_INTENT.md:66 |
| `0.1900` | 2312 | PREREG_V8F_LAUNCH_INTENT.md:196; RESULT_V8F_DAG_ORC.md:57 |
| `0.1877` | 2313,2520 | PREREG_V8F_LAUNCH_INTENT.md:197; RESULT_V8F_DAG_ORC.md:58; DAGORC_READ.json:503; P0_RF0.json (rebase):367,383,477; RESULT_V8F_LAUNCH_INTENT.md:67; LAUNCH_INTENT_READ.json:705,854 |
| `0.3754` | 2313,2520 | PREREG_V8F_LAUNCH_INTENT.md:197; RESULT_V8F_LAUNCH_INTENT.md:67; LAUNCH_INTENT_READ.json:768,855 |
| `0.2592` | 2313,2520 | PREREG_V8F_LAUNCH_INTENT.md:197; RESULT_V8F_DAG_ORC.md:58; DAGORC_READ.json:524,791; RF_REP.json:18,116; RESULT_V8F_LAUNCH_INTENT.md:67; LAUNCH_INTENT_READ.json:726,853,1217 |
| `0.5184` | 2313 | PREREG_V8F_LAUNCH_INTENT.md:197; RESULT_V8F_DAG_ORC.md:58; DAGORC_READ.json:298,406,566 |
| `0.4449` | 2314,2521 | PREREG_V8F_LAUNCH_INTENT.md:198; RESULT_V8F_DAG_ORC.md:59; RESULT_V8F_LAUNCH_INTENT.md:68 |
| `0.8898` | 2314,2314,2521 | PREREG_V8F_LAUNCH_INTENT.md:198,198; RESULT_V8F_DAG_ORC.md:59; RESULT_V8F_LAUNCH_INTENT.md:68 |
| `0.3912` | 2314,2521 | PREREG_V8F_LAUNCH_INTENT.md:198; RESULT_V8F_DAG_ORC.md:59; DAGORC_READ.json:525,805; RF_REP.json:19,122; RESULT_V8F_LAUNCH_INTENT.md:68; LAUNCH_INTENT_READ.json:727,858,1231 |
| `1.0082` | 2315,2522 | PREREG_V8F_LAUNCH_INTENT.md:199; RESULT_V8F_DAG_ORC.md:60; DAGORC_READ.json:505; P0_RF0.json (rebase):417,433,479; RESULT_V8F_LAUNCH_INTENT.md:69; LAUNCH_INTENT_READ.json:707,864 |
| `2.0164` | 2315,2315,2522 | PREREG_V8F_LAUNCH_INTENT.md:199,199; RESULT_V8F_DAG_ORC.md:60; DAGORC_READ.json:300,408,568; RESULT_V8F_LAUNCH_INTENT.md:69; LAUNCH_INTENT_READ.json:770,865 |
| `0.5170` | 2315,2522 | PREREG_V8F_LAUNCH_INTENT.md:199; RESULT_V8F_DAG_ORC.md:60; RESULT_V8F_LAUNCH_INTENT.md:69 |
| `0.0048` | 2316,2523 | PREREG_V8F_LAUNCH_INTENT.md:200; RESULT_V8F_DAG_ORC.md:61,222; SPEEDSUB_T1_BESIDE.json:237; DAGORC_READ.json:506,1101; RF_REP.json:250; P0_RF0.json (rebase):456,480; RESULT_V8F_LAUNCH_INTENT.md:70; LAUNCH_INTENT_READ.json:708,869,1527 |
| `0.0388` | 2316,2523 | PREREG_V8F_LAUNCH_INTENT.md:200; RESULT_V8F_DAG_ORC.md:61,222; GOALS_AND_CLAIMS.md (register rows):3; DAGORC_READ.json:527,833; RF_REP.json:21,134; RESULT_V8F_LAUNCH_INTENT.md:70; LAUNCH_INTENT_READ.json:729,868,1259 |
| `3000` | 2323 | PREREG_V8F_LAUNCH_INTENT.md:7,90,113; PREREG_V8F_DAG_ORC.md:59,66,175,300...; RESULT_V8F_DAG_ORC.md:294,296,298,300; GOALS_AND_CLAIMS.md (register rows):3; launch-intent-chain/LAUNCH.txt:8,76; chain_v8f_launchintent.out:18,19,24,25...; R2C_NOTE.md:12,14; REP run records (s0, s1):3,141,156,294 |
| `49.6969` | 2323 | PREREG_V8F_LAUNCH_INTENT.md:113,127; PREREG_V8F_DAG_ORC.md:359; RESULT_V8F_DAG_ORC.md:294,296,298,300; launch-intent-flag/RESULTS.md:14; chain_v8f_launchintent.out:56,75,94,113; REP launch-gate tokens (s0, s1):34,80; REP run records (s0, s1):24,177 |
| `01` | 2330,2392,2392,2444,2491 | RESULT_V8F_SPEEDSUB.md:371; RESULT_V8F_SPEEDSUB_T1.md:420; PREREG_V8F_SPEEDSUB.md:150; PREREG_V8F_LAUNCH_INTENT.md:12,12; PREREG_V8F_DAG_ORC.md:61,70,438,488; RESULT_V8F_DAG_ORC.md:259,271,274; FINDING_1_ATTRIBUTION.md:65,65; launch-intent-flag/RESULTS.md:10,28,30; P0_RF0.json (rebase):6; RESULT_V8F_LAUNCH_INTENT.md:422,461; chain_v8f_launchintent.out:28,29,33; R2C_NOTE.md:12,16; lit1_logs/chain.out:10,53 |
| `13` | 2331,2391,2391,2392,2392... | RESULT_V8F_SPEEDSUB.md:308; RESULT_V8F_SPEEDSUB_T1.md:170,428,434,434; PREREG_V8F_SPEEDSUB_T1.md:42,42; PREREG_V8F_LAUNCH_INTENT.md:25,26,28,49...; PREREG_V8F_DAG_ORC.md:150,255,255,479; GOALS_AND_CLAIMS.md (register rows):3,3; FINDING_1_ATTRIBUTION.md:65,65; launch-intent-flag/RESULTS.md:33; launch-intent-v2/NOTES.md:40; launch-intent-chain/HANDOFF.md:1; git log (cited commits):10,13; RESULT_V8F_LAUNCH_INTENT.md:3,8,305,363...; chain_v8f_launchintent.out:1,2,5,6...; lit1_logs/chain.out:7,8,9,10...; check_result_filled.out:92,92,92,92... |
| `54` | 2331,2391,2391,2394,2481... | RESULT_V8F_SPEEDSUB_T1.md:417; PREREG_V8F_LAUNCH_INTENT.md:33; PREREG_V8F_DAG_ORC.md:271,321; RESULT_V8F_DAG_ORC.md:3,3; RESULT_V8F_LAUNCH_INTENT.md:363,431,443; chain_v8f_launchintent.out:1,2,5,6...; lit1_logs/chain.out:44,45; read_A1/read_A1.out:9; check_result_filled.out:100,120 |
| `56` | 2332 | RESULT_V8F_SPEEDSUB_T1.md:15; PREREG_V8F_SPEEDSUB_T1.md:18,70; PREREG_V8F_DAG_ORC.md:479; RESULT_V8F_DAG_ORC.md:261,272; GOALS_AND_CLAIMS.md (register rows):3,3; launch-intent-chain/HANDOFF.md:8; launch-intent-chain/NOTES.md:28,47; launch-intent-chain/LAUNCH.txt:14,82,88; R2C_NOTE.md:7,34,37; lit1_logs/chain.out:6; lit1_logs/launch_intent_read_REFUSED.out:2,3,4,5...; read_A1/read_A1.out:10,21,37 |
| `524288` | 2367,2960 | PREREG_V8F_LAUNCH_INTENT.md:280; PREREG_V8F_DAG_ORC.md:240,421; GOALS_AND_CLAIMS.md (register rows):3,6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:62; launch-intent-v2/NOTES.md:10; launch-intent-v2/enumerate_cells_li.out:1 |
| `262144` | 2367 | PREREG_V8F_LAUNCH_INTENT.md:280; PREREG_V8F_DAG_ORC.md:243; launch-intent-v2/enumerate_cells_li.out:3 |
| `261888` | 2367 | PREREG_V8F_LAUNCH_INTENT.md:280; PREREG_V8F_DAG_ORC.md:244; launch-intent-v2/enumerate_cells_li.out:4 |
| `128` | 2368 | PREREG_V8F_LAUNCH_INTENT.md:61,280,284; PREREG_V8F_DAG_ORC.md:249,254; launch-intent-v2/enumerate_cells_li.out:9,17; check_result_filled.out:136 |
| `64` | 2368,2466,2467,2468,2469 | PREREG_V8F_LAUNCH_INTENT.md:90,280; PREREG_V8F_DAG_ORC.md:250; launch-intent-v2/enumerate_cells_li.out:10; launch-intent-chain/LAUNCH.txt:49,49; RESULT_V8F_LAUNCH_INTENT.md:326,327,328,329; chain_v8f_launchintent.out:18,19,24,25... |
| `0.02` | 2380,2381,2451,2451,2453... | RESULT_V8F_SPEEDSUB.md:166; PREREG_V8F_LAUNCH_INTENT.md:99,101,126,239...; PREREG_V8F_DAG_ORC.md:132,210,210,384...; RESULT_V8F_DAG_ORC.md:259,295,295,297...; GOALS_AND_CLAIMS.md (register rows):3; SPEEDSUB_T1_READ.json:1268,6528; launch-intent-chain/NOTES.md:19,19,44; DAGORC_READ.json:652; RF_REP.json:57; git log (cited commits):43; RESULT_V8F_LAUNCH_INTENT.md:309,311,311,361; LAUNCH_INTENT_READ.json:1078; chain_v8f_launchintent.out:57,57,76,76... |
| `8168` | 2381 | PREREG_V8F_LAUNCH_INTENT.md:79,240 |
| `0.1619` | 2381,2451,2453 | PREREG_V8F_LAUNCH_INTENT.md:22,22,87,240; launch-intent-flag/RESULTS.md:50,50; RESULT_V8F_LAUNCH_INTENT.md:311; chain_v8f_launchintent.out:57,76,95,114 |
| `0.3315` | 2384,2460,2884 | PREREG_V8F_LAUNCH_INTENT.md:94,340; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:318; check_result_filled.out:128 |
| `0.3307` | 2384,2460 | PREREG_V8F_LAUNCH_INTENT.md:95,340; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:318; check_result_filled.out:128 |
| `0.3256` | 2384,2460,2883 | PREREG_V8F_LAUNCH_INTENT.md:96,340; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:318; check_result_filled.out:128 |
| `0.3281` | 2385,2460 | PREREG_V8F_LAUNCH_INTENT.md:97,340; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:318; check_result_filled.out:128 |
| `57` | 2392 | RESULT_V8F_SPEEDSUB_T1.md:15,432; PREREG_V8F_SPEEDSUB.md:148; PREREG_V8F_SPEEDSUB_T1.md:18,70; PREREG_V8F_DAG_ORC.md:479; launch-intent-flag/RESULTS.md:30; RESULT_V8F_LAUNCH_INTENT.md:443; chain_v8f_launchintent.out:15; lit1_logs/launch_intent_read_REFUSED.out:10,11,12,13... |
| `58` | 2392,2392 | RESULT_V8F_SPEEDSUB_T1.md:15; PREREG_V8F_SPEEDSUB_T1.md:18,70; PREREG_V8F_LAUNCH_INTENT.md:33,122,122; PREREG_V8F_DAG_ORC.md:272; RESULT_V8F_DAG_ORC.md:3,3; chain_v8f_launchintent.out:16,17,21; lit1_logs/chain.out:16; lit1_logs/chain.out.prev_1028151251:7,8,9; lit1_logs/launch_intent_read_REFUSED.out:22,23,24,25...; read_A1/read_A1.out:22,31 |
| `14` | 2392,2392,2392,2392,2392... | RESULT_V8F_SPEEDSUB_T1.md:170,418; PREREG_V8F_SPEEDSUB.md:160; PREREG_V8F_LAUNCH_INTENT.md:14,335; PREREG_V8F_DAG_ORC.md:150; launch-intent-flag/RESULTS.md:45,45; launch-intent-v2/NOTES.md:25; launch-intent-chain/PINS_FILLED.txt:1; git log (cited commits):28,28; RESULT_V8F_LAUNCH_INTENT.md:420,430,443,447...; chain_v8f_launchintent.out:22,23,27,28...; lit1_logs/chain.out:19,33,37; lit1_logs/launch_intent_read_REFUSED.out:26,33; check_result_filled.out:104,136 |
| `00` | 2392,2392,2848 | RESULT_V8F_SPEEDSUB_T1.md:3,77,434; PREREG_V8F_SPEEDSUB.md:148; RESULT_V8F_DAG_ORC.md:263,277; launch-intent-flag/MM_REDESIGN_2026-10-28.md:1; launch-intent-chain/NOTES.md:55; git log (cited commits):13,34; RESULT_V8F_LAUNCH_INTENT.md:3,430; chain_v8f_launchintent.out:22,23,27; lit1_logs/chain.out:7,8,9,30; lit1_logs/chain.out.prev_1028151251:14,15; lit1_logs/launch_intent_read_REFUSED.out:37; check_result_filled.out:92 |
| `34` | 2392 | RESULT_V8F_SPEEDSUB.md:307; launch-intent-flag/RESULTS.md:33; git log (cited commits):31; lit1_logs/chain.out:48 |
| `02` | 2392,2392,2393,2441,2476... | RESULT_V8F_SPEEDSUB.md:72,308,309,309...; RESULT_V8F_SPEEDSUB_T1.md:429,430; PREREG_V8F_LAUNCH_INTENT.md:318; PREREG_V8F_DAG_ORC.md:317,434,488; RESULT_V8F_DAG_ORC.md:259,271; launch-intent-flag/RESULTS.md:29; git log (cited commits):1; RESULT_V8F_LAUNCH_INTENT.md:420,443,447; chain_v8f_launchintent.out:34,35,39; lit1_logs/chain.out:11,12; read_A1/read_A1.out:3,4,5,6...; check_result_filled.out:93 |
| `33` | 2392 | RESULT_V8F_SPEEDSUB_T1.md:418,420,421,427...; PREREG_V8F_DAG_ORC.md:31; git log (cited commits):28; lit1_logs/chain.out:28 |
| `37` | 2392,2398,2398,2476,2479... | RESULT_V8F_SPEEDSUB.md:157; PREREG_V8F_LAUNCH_INTENT.md:326,326; PREREG_V8F_DAG_ORC.md:324,325; RESULT_V8F_DAG_ORC.md:306; launch-intent-flag/RESULTS.md:29; RESULT_V8F_LAUNCH_INTENT.md:447,450,450; MLAUNCH NOTES.md:39; lit1_logs/chain.out:39 |
| `25` | 2393,2478 | RESULT_V8F_SPEEDSUB_T1.md:462,462; PREREG_V8F_SPEEDSUB.md:135,148; PREREG_V8F_SPEEDSUB_T1.md:133; RESULT_V8F_DAG_ORC.md:306; git log (cited commits):28,28; RESULT_V8F_LAUNCH_INTENT.md:449; chain_v8f_launchintent.out:64,65,74,77; lit1_logs/chain.out:22,23; read_A1/read_A1.out:4,15 |
| `06` | 2393,2478,2479 | RESULT_V8F_SPEEDSUB.md:307,314,314; RESULT_V8F_SPEEDSUB_T1.md:431,432; PREREG_V8F_LAUNCH_INTENT.md:318; PREREG_V8F_DAG_ORC.md:317; RESULT_V8F_DAG_ORC.md:261,263,263,271...; GOALS_AND_CLAIMS.md (register rows):1,2; RESULT_V8F_LAUNCH_INTENT.md:449,450; chain_v8f_launchintent.out:82,83,84,93...; R2C_NOTE.md:2,4,12,16...; lit1_logs/chain.out.prev_1028151251:1,2,3,4...; lit1_logs/launch_intent_read_REFUSED.out:2,37; read_A1/read_A1.out:38 |
| `2.9` | 2394,2481,2805 | RESULT_V8F_LAUNCH_INTENT.md:363,443; check_result_filled.out:100,118 |
| `87` | 2396 | GOALS_AND_CLAIMS.md (register rows):6,6; launch-intent-v2/NOTES.md:12; git log (cited commits):7,13; check_result_filled.out:62 |
| `245` | 2396,2396 | GOALS_AND_CLAIMS.md (register rows):6,6; launch-intent-v2/NOTES.md:13,13,23; git log (cited commits):13,13 |
| `390` | 2396,2396 | launch-intent-chain/HANDOFF.md:4,4; launch-intent-chain/NOTES.md:54,54,54; git log (cited commits):10,10 |
| `185` | 2396,2396 | launch-intent-chain/HANDOFF.md:4,4; launch-intent-chain/NOTES.md:54,54,54; git log (cited commits):10,10 |
| `2.8` | 2397,2481,2806 | PREREG_V8F_LAUNCH_INTENT.md:119,320,322,323; PREREG_V8F_DAG_ORC.md:426,488; RESULT_V8F_DAG_ORC.md:265; GOALS_AND_CLAIMS.md (register rows):3,3,6; launch-intent-flag/MM_REDESIGN_2026-10-28.md:46; launch-intent-v2/NOTES.md:40; RESULT_V8F_LAUNCH_INTENT.md:363; check_result_filled.out:102 |
| `42` | 2397,2398 | RESULT_V8F_SPEEDSUB_T1.md:424,425,434,434; PREREG_V8F_LAUNCH_INTENT.md:319,319; PREREG_V8F_DAG_ORC.md:426; RESULT_V8F_DAG_ORC.md:263,278; lit1_logs/chain.out:15; lit1_logs/launch_intent_read_REFUSED.out:9,18,19,29...; check_result_filled.out:136,136 |
| `4.93` | 2398 | RESULT_V8F_DAG_ORC.md:265 |
| `-0.0214` | 2416,2609 | RESULT_V8F_LAUNCH_INTENT.md:29,154 |
| `-0.0015` | 2416,2610 | RESULT_V8F_LAUNCH_INTENT.md:29,155 |
| `341.6` | 2430,2818,2819 | RESULT_V8F_LAUNCH_INTENT.md:3; LAUNCH_INTENT_READ.json:6278; check_result_filled.out:108 |
| `24` | 2441,2477 | RESULT_V8F_SPEEDSUB.md:72,308,309,309...; RESULT_V8F_SPEEDSUB_T1.md:416,421; git log (cited commits):7; RESULT_V8F_LAUNCH_INTENT.md:3,420,448; chain_v8f_launchintent.out:63; lit1_logs/chain.out:21; read_A1/read_A1.out:13,14,32,33; check_result_filled.out:93,136 |
| `46` | 2441,2489 | RESULT_V8F_SPEEDSUB.md:251,251; RESULT_V8F_SPEEDSUB_T1.md:15,422; PREREG_V8F_SPEEDSUB_T1.md:18,70; FINDING_1_ATTRIBUTION.md:23; git log (cited commits):7,31; RESULT_V8F_LAUNCH_INTENT.md:3,420,459; lit1_logs/chain.out:18,32; check_result_filled.out:93 |
| `04` | 2442,2478,2478,2495 | RESULT_V8F_SPEEDSUB.md:307; RESULT_V8F_SPEEDSUB_T1.md:430,431; RESULT_V8F_DAG_ORC.md:261,263,272,273; launch-intent-flag/RESULTS.md:27; git log (cited commits):22,40; RESULT_V8F_LAUNCH_INTENT.md:420,427,449,449; chain_v8f_launchintent.out:78,79,80,81; R2C_NOTE.md:2,21; lit1_logs/chain.out:14; read_A1/read_A1.out:23,24,25,26... |
| `51` | 2447,2494,2848 | RESULT_V8F_SPEEDSUB_T1.md:416,427,429; PREREG_V8F_DAG_ORC.md:321; git log (cited commits):10; RESULT_V8F_LAUNCH_INTENT.md:8,427,430; lit1_logs/chain.out:1; lit1_logs/chain.out.prev_1028151251:4; read_A1/read_A1.out:5,6; check_result_filled.out:92,94 |
| `30727` | 2455 | PREREG_V8F_LAUNCH_INTENT.md:94; launch-intent-flag/f_measure.json:43; RESULT_V8F_LAUNCH_INTENT.md:313; chain_v8f_launchintent.out:57 |
| `191203` | 2455 | PREREG_V8F_LAUNCH_INTENT.md:94,99; launch-intent-flag/f_measure.json:42; RESULT_V8F_LAUNCH_INTENT.md:313; chain_v8f_launchintent.out:57,57 |
| `0.160704` | 2455 | RESULT_V8F_LAUNCH_INTENT.md:313; chain_v8f_launchintent.out:57 |
| `0.331451` | 2455 | RESULT_V8F_LAUNCH_INTENT.md:313; chain_v8f_launchintent.out:57; check_result_filled.out:55 |
| `30674` | 2456 | PREREG_V8F_LAUNCH_INTENT.md:95; launch-intent-flag/f_measure.json:50; RESULT_V8F_LAUNCH_INTENT.md:314; chain_v8f_launchintent.out:76 |
| `191396` | 2456 | PREREG_V8F_LAUNCH_INTENT.md:95; launch-intent-flag/f_measure.json:49; RESULT_V8F_LAUNCH_INTENT.md:314; chain_v8f_launchintent.out:76,76 |
| `0.160265` | 2456 | RESULT_V8F_LAUNCH_INTENT.md:314; chain_v8f_launchintent.out:76 |
| `0.330729` | 2456 | RESULT_V8F_LAUNCH_INTENT.md:314; chain_v8f_launchintent.out:76; check_result_filled.out:56 |
| `30068` | 2457 | PREREG_V8F_LAUNCH_INTENT.md:96; launch-intent-flag/f_measure.json:57; RESULT_V8F_LAUNCH_INTENT.md:315; chain_v8f_launchintent.out:95 |
| `191333` | 2457 | PREREG_V8F_LAUNCH_INTENT.md:96; launch-intent-flag/f_measure.json:56; RESULT_V8F_LAUNCH_INTENT.md:315; chain_v8f_launchintent.out:95,95; REP run records (s0, s1):147,151 |
| `0.15715` | 2457 | RESULT_V8F_LAUNCH_INTENT.md:315; chain_v8f_launchintent.out:95 |
| `0.325586` | 2457 | RESULT_V8F_LAUNCH_INTENT.md:315; chain_v8f_launchintent.out:95; check_result_filled.out:57 |
| `30367` | 2458 | PREREG_V8F_LAUNCH_INTENT.md:97; launch-intent-flag/f_measure.json:64; RESULT_V8F_LAUNCH_INTENT.md:316; chain_v8f_launchintent.out:114 |
| `191391` | 2458 | PREREG_V8F_LAUNCH_INTENT.md:97; launch-intent-flag/f_measure.json:63; RESULT_V8F_LAUNCH_INTENT.md:316; chain_v8f_launchintent.out:114,114; REP run records (s0, s1):300,304 |
| `0.158665` | 2458 | RESULT_V8F_LAUNCH_INTENT.md:316; chain_v8f_launchintent.out:114 |
| `0.328093` | 2458 | RESULT_V8F_LAUNCH_INTENT.md:316; chain_v8f_launchintent.out:114; check_result_filled.out:58 |
| `0.03420829772949219` | 2466 | RESULT_V8F_LAUNCH_INTENT.md:326; chain_v8f_launchintent.out:18,19 |
| `11.032110214233398` | 2466 | RESULT_V8F_LAUNCH_INTENT.md:326; chain_v8f_launchintent.out:18,19 |
| `14267.4716796875` | 2466 | RESULT_V8F_LAUNCH_INTENT.md:326; chain_v8f_launchintent.out:18,19 |
| `0.03153038024902344` | 2467 | RESULT_V8F_LAUNCH_INTENT.md:327; chain_v8f_launchintent.out:24,25 |
| `10.921006202697754` | 2467 | RESULT_V8F_LAUNCH_INTENT.md:327; chain_v8f_launchintent.out:24,25 |
| `10972.4638671875` | 2467 | RESULT_V8F_LAUNCH_INTENT.md:327; chain_v8f_launchintent.out:24,25 |
| `0.03741455078125` | 2468 | RESULT_V8F_LAUNCH_INTENT.md:328; chain_v8f_launchintent.out:30,31 |
| `10.58016586303711` | 2468 | RESULT_V8F_LAUNCH_INTENT.md:328; chain_v8f_launchintent.out:30,31 |
| `10940.1328125` | 2468 | RESULT_V8F_LAUNCH_INTENT.md:328; chain_v8f_launchintent.out:30,31 |
| `0.02295684814453125` | 2469 | RESULT_V8F_LAUNCH_INTENT.md:329; chain_v8f_launchintent.out:36,37 |
| `9.637072563171387` | 2469 | RESULT_V8F_LAUNCH_INTENT.md:329; chain_v8f_launchintent.out:36,37 |
| `46505.515625` | 2469 | RESULT_V8F_LAUNCH_INTENT.md:329; chain_v8f_launchintent.out:36,37 |
| `218` | 2486,2486 | RESULT_V8F_LAUNCH_INTENT.md:456,456 |
| `88` | 2486 | RESULT_V8F_LAUNCH_INTENT.md:456; MLAUNCH NOTES.md:35 |
| `222` | 2487 | launch-intent-flag/RESULTS.md:36; RESULT_V8F_LAUNCH_INTENT.md:457; MLAUNCH NOTES.md:32 |
| `178` | 2487 | RESULT_V8F_LAUNCH_INTENT.md:457 |
| `208` | 2488,2489 | RESULT_V8F_LAUNCH_INTENT.md:458,459 |
| `139` | 2488 | RESULT_V8F_SPEEDSUB_T1.md:418,421,422,423...; RESULT_V8F_LAUNCH_INTENT.md:458 |
| `140` | 2489 | RESULT_V8F_SPEEDSUB_T1.md:419,424,425,428; PREREG_V8F_LAUNCH_INTENT.md:326; launch-intent-flag/MM_REDESIGN_2026-10-28.md:73; launch-intent-chain/NOTES.md:54,54; RESULT_V8F_LAUNCH_INTENT.md:459 |
| `1756` | 2491,2819 | RESULT_V8F_LAUNCH_INTENT.md:461; check_result_filled.out:101,119 |
| `3040` | 2493 | RESULT_V8F_LAUNCH_INTENT.md:425; lit1_logs/chain.out.prev_1028151251:22 |
| `4096` | 2493 | RESULT_V8F_LAUNCH_INTENT.md:425; lit1_logs/chain.out.prev_1028151251:22 |
| `2048` | 2493,2494 | RESULT_V8F_LAUNCH_INTENT.md:425,426; lit1_logs/chain.out:2; lit1_logs/chain.out.prev_1028151251:2; check_result_filled.out:134 |
| `14.32` | 2500,2513 | RESULT_V8F_LAUNCH_INTENT.md:60 |
| `1.67` | 2505 | RESULT_V8F_LAUNCH_INTENT.md:52 |
| `2.12` | 2506,2508 | RESULT_V8F_LAUNCH_INTENT.md:53,55 |
| `11.43` | 2507 | RESULT_V8F_LAUNCH_INTENT.md:54 |
| `0.97` | 2510 | FINDING_1_ATTRIBUTION.md:46; RESULT_V8F_LAUNCH_INTENT.md:57 |
| `12.71` | 2511 | RESULT_V8F_LAUNCH_INTENT.md:58 |
| `3.30` | 2512 | RESULT_V8F_LAUNCH_INTENT.md:59 |
| `2.52` | 2514 | RESULT_V8F_DAG_ORC.md:265; RESULT_V8F_LAUNCH_INTENT.md:61 |
| `0.35` | 2515 | RESULT_V8F_LAUNCH_INTENT.md:62 |
| `1.54` | 2516 | RESULT_V8F_LAUNCH_INTENT.md:63 |
| `6.40` | 2517 | RESULT_V8F_LAUNCH_INTENT.md:64 |
| `1.15` | 2518 | RESULT_V8F_LAUNCH_INTENT.md:65 |
| `3.80` | 2519 | RESULT_V8F_LAUNCH_INTENT.md:66 |
| `0.69` | 2520 | RESULT_V8F_LAUNCH_INTENT.md:67 |
| `0.44` | 2521 | RESULT_V8F_LAUNCH_INTENT.md:68 |
| `4.04` | 2523 | RESULT_V8F_LAUNCH_INTENT.md:70 |
| `53` | 2526,2785 | RESULT_V8F_SPEEDSUB_T1.md:77,428,434; PREREG_V8F_SPEEDSUB.md:131,148; RESULT_V8F_DAG_ORC.md:259,265; lit1_logs/chain.out:4,41; lit1_logs/chain.out.prev_1028151251:5,6; read_A1/read_A1.out:7,8,17,18... |
| `-0.0094` | 2528,2634 | RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:3511,5623 |
| `158.1` | 2533,2544,2781 | RESULT_V8F_LAUNCH_INTENT.md:95,348 |
| `-0.0212` | 2533,2540,2645,2734 | RESULT_V8F_LAUNCH_INTENT.md:14,91,194,301...; LAUNCH_INTENT_READ.json:3328,5009; check_result_filled.out:124 |
| `55.1` | 2533,2540 | RESULT_V8F_LAUNCH_INTENT.md:91 |
| `-0.7305` | 2538,2641 | RESULT_V8F_LAUNCH_INTENT.md:89,190; LAUNCH_INTENT_READ.json:3271,4928 |
| `-1.7042` | 2538,2641 | RESULT_V8F_LAUNCH_INTENT.md:89,190; LAUNCH_INTENT_READ.json:3272,4929 |
| `+0.3980` | 2538,2641 | RESULT_V8F_LAUNCH_INTENT.md:89,190 |
| `+0.7305` | 2538 | RESULT_V8F_LAUNCH_INTENT.md:89 |
| `66.0` | 2538 | RESULT_V8F_LAUNCH_INTENT.md:89 |
| `+0.0113` | 2539,2642 | RESULT_V8F_LAUNCH_INTENT.md:90,191 |
| `-0.0037` | 2541 | P0_RF0.json (rebase):323; RESULT_V8F_LAUNCH_INTENT.md:92; LAUNCH_INTENT_READ.json:3412 |
| `-0.6672` | 2542,2641 | RESULT_V8F_LAUNCH_INTENT.md:93,190; LAUNCH_INTENT_READ.json:4348,5299 |
| `+1.7987` | 2542,2641 | RESULT_V8F_LAUNCH_INTENT.md:93,190 |
| `-0.5729` | 2542 | RESULT_V8F_LAUNCH_INTENT.md:93 |
| `-51.8` | 2542 | RESULT_V8F_LAUNCH_INTENT.md:93 |
| `-0.0073` | 2543,2642 | RESULT_V8F_LAUNCH_INTENT.md:94,191; LAUNCH_INTENT_READ.json:4361,5318 |
| `-0.0273` | 2543,2642 | RESULT_V8F_LAUNCH_INTENT.md:94,191; LAUNCH_INTENT_READ.json:4362,5319 |
| `+0.0073` | 2543 | RESULT_V8F_LAUNCH_INTENT.md:94 |
| `-0.0001` | 2545 | GOALS_AND_CLAIMS.md (register rows):3,3; P0_RF0.json (rebase):442; RESULT_V8F_LAUNCH_INTENT.md:96; LAUNCH_INTENT_READ.json:4488 |
| `+0.0083` | 2545 | RESULT_V8F_DAG_ORC.md:82,163; RESULT_V8F_LAUNCH_INTENT.md:96 |
| `-43.8` | 2545 | RESULT_V8F_LAUNCH_INTENT.md:96 |
| `+0.2594` | 2546,2641 | RESULT_V8F_LAUNCH_INTENT.md:97,190 |
| `-0.8091` | 2546,2641 | RESULT_V8F_LAUNCH_INTENT.md:97,190; LAUNCH_INTENT_READ.json:3540,5671 |
| `+1.3427` | 2546,2641 | RESULT_V8F_LAUNCH_INTENT.md:97,190 |
| `-0.2594` | 2546 | RESULT_V8F_LAUNCH_INTENT.md:97 |
| `-23.4` | 2546,2785 | RESULT_V8F_LAUNCH_INTENT.md:97 |
| `-0.0387` | 2547,2642,2782 | RESULT_V8F_LAUNCH_INTENT.md:98,191,349; LAUNCH_INTENT_READ.json:3553,5690 |
| `-0.0679` | 2547,2642,2782 | RESULT_V8F_LAUNCH_INTENT.md:98,191,349; LAUNCH_INTENT_READ.json:3554,5691 |
| `-0.0160` | 2547,2632,2642,2782 | RESULT_V8F_LAUNCH_INTENT.md:98,179,191,349 |
| `73.6` | 2547,2782 | RESULT_V8F_LAUNCH_INTENT.md:98,349 |
| `-0.0068` | 2548,2645 | P0_RF0.json (rebase):298; RESULT_V8F_LAUNCH_INTENT.md:99,194; LAUNCH_INTENT_READ.json:3595,5750 |
| `-0.0204` | 2548,2645,2652 | RESULT_V8F_LAUNCH_INTENT.md:99,194,201; LAUNCH_INTENT_READ.json:3596,4700,5751,6167 |
| `50.0` | 2548 | RESULT_V8F_LAUNCH_INTENT.md:99 |
| `+0.0060` | 2549 | RESULT_V8F_DAG_ORC.md:75,81; RESULT_V8F_LAUNCH_INTENT.md:100 |
| `-0.0016` | 2549 | RESULT_V8F_LAUNCH_INTENT.md:100 |
| `-16.7` | 2549 | RESULT_V8F_LAUNCH_INTENT.md:100 |
| `-0.2191` | 2550,2641 | RESULT_V8F_LAUNCH_INTENT.md:101,190; LAUNCH_INTENT_READ.json:4615,6040 |
| `-1.7774` | 2550,2641 | RESULT_V8F_LAUNCH_INTENT.md:101,190; LAUNCH_INTENT_READ.json:4616,6041 |
| `+1.2541` | 2550,2641 | RESULT_V8F_LAUNCH_INTENT.md:101,190 |
| `+0.2191` | 2550 | RESULT_V8F_LAUNCH_INTENT.md:101 |
| `19.8` | 2550 | RESULT_V8F_LAUNCH_INTENT.md:101 |
| `-0.0051` | 2551,2642 | RESULT_V8F_LAUNCH_INTENT.md:102,191; LAUNCH_INTENT_READ.json:4630,6061 |
| `+0.0291` | 2551,2642 | RESULT_V8F_LAUNCH_INTENT.md:102,191; MLAUNCH NOTES.md:73 |
| `-0.0101` | 2551 | RESULT_V8F_SPEEDSUB_T1.md:152; RESULT_V8F_DAG_ORC.md:73,87,128,154...; SPEEDSUB_T1_READ.json:3726; DAGORC_READ.json:1981,2179,2519,2717...; RESULT_V8F_LAUNCH_INTENT.md:102 |
| `-19.2` | 2551 | RESULT_V8F_LAUNCH_INTENT.md:102 |
| `+0.0178` | 2552,2645 | RESULT_V8F_LAUNCH_INTENT.md:103,194 |
| `+0.0474` | 2552,2645 | RESULT_V8F_LAUNCH_INTENT.md:103,194 |
| `-0.0178` | 2552 | RESULT_V8F_DAG_ORC.md:87,124,193; DAGORC_READ.json:1911,2449; RESULT_V8F_LAUNCH_INTENT.md:103 |
| `-130.9` | 2552,2785 | RESULT_V8F_LAUNCH_INTENT.md:103 |
| `+0.0070` | 2553,2634 | RESULT_V8F_DAG_ORC.md:99,154; RESULT_V8F_LAUNCH_INTENT.md:104,181 |
| `-21.9` | 2553 | RESULT_V8F_LAUNCH_INTENT.md:104 |
| `-30.2` | 2557,2576 | RESULT_V8F_LAUNCH_INTENT.md:128 |
| `-3.0` | 2558,2566,2570 | RESULT_V8F_LAUNCH_INTENT.md:118,122 |
| `+2.7420` | 2562,2692 | RESULT_V8F_LAUNCH_INTENT.md:114,252 |
| `+0.3388` | 2562,2692 | RESULT_V8F_LAUNCH_INTENT.md:114,252 |
| `+5.2598` | 2562,2692 | RESULT_V8F_LAUNCH_INTENT.md:114,252 |
| `+3.4726` | 2562,2692 | RESULT_V8F_DAG_ORC.md:103,202; RESULT_V8F_LAUNCH_INTENT.md:114,252 |
| `+0.6601` | 2562,2692 | RESULT_V8F_DAG_ORC.md:103; RESULT_V8F_LAUNCH_INTENT.md:114,252 |
| `+6.3308` | 2562,2692 | RESULT_V8F_DAG_ORC.md:103; RESULT_V8F_LAUNCH_INTENT.md:114,252 |
| `+0.0847` | 2563,2572,2693,2720 | RESULT_V8F_DAG_ORC.md:107; RESULT_V8F_LAUNCH_INTENT.md:115,124,253,280 |
| `+0.1200` | 2563,2693 | RESULT_V8F_LAUNCH_INTENT.md:115,253 |
| `+0.0854` | 2563,2693 | RESULT_V8F_DAG_ORC.md:104,203; RESULT_V8F_LAUNCH_INTENT.md:115,253; MLAUNCH NOTES.md:73 |
| `+0.0580` | 2563,2693 | RESULT_V8F_DAG_ORC.md:104; RESULT_V8F_LAUNCH_INTENT.md:115,253 |
| `+0.1175` | 2563,2693 | RESULT_V8F_DAG_ORC.md:104; RESULT_V8F_LAUNCH_INTENT.md:115,253 |
| `0.8` | 2563 | launch-intent-chain/LAUNCH.txt:15; RESULT_V8F_LAUNCH_INTENT.md:115 |
| `+0.0201` | 2564,2696 | RESULT_V8F_LAUNCH_INTENT.md:116,256 |
| `+0.0588` | 2564,2696 | RESULT_V8F_LAUNCH_INTENT.md:116,256 |
| `+0.0443` | 2564,2696 | RESULT_V8F_DAG_ORC.md:107,204; RESULT_V8F_LAUNCH_INTENT.md:116,256 |
| `+0.0232` | 2564,2696 | RESULT_V8F_DAG_ORC.md:107; RESULT_V8F_LAUNCH_INTENT.md:116,256 |
| `+0.0744` | 2564,2696 | RESULT_V8F_DAG_ORC.md:107; REP_S0_SPEED_BIAS.md:25; RESULT_V8F_LAUNCH_INTENT.md:116,256 |
| `15.3` | 2564 | RESULT_V8F_LAUNCH_INTENT.md:116 |
| `+0.0564` | 2565,2702 | PREREG_V8F_SPEEDSUB.md:10; RESULT_V8F_LAUNCH_INTENT.md:117,262 |
| `+0.0374` | 2565,2702 | RESULT_V8F_LAUNCH_INTENT.md:117,262 |
| `+0.0788` | 2565,2702 | RESULT_V8F_LAUNCH_INTENT.md:117,262 |
| `+0.0568` | 2565,2702 | RESULT_V8F_DAG_ORC.md:113,205; RESULT_V8F_LAUNCH_INTENT.md:117,262 |
| `+0.0378` | 2565,2702 | RESULT_V8F_DAG_ORC.md:113; RESULT_V8F_LAUNCH_INTENT.md:117,262 |
| `+0.0798` | 2565,2702 | RESULT_V8F_DAG_ORC.md:113; RESULT_V8F_LAUNCH_INTENT.md:117,262 |
| `+19.9019` | 2566,2692 | RESULT_V8F_LAUNCH_INTENT.md:118,252 |
| `+14.5378` | 2566,2692 | RESULT_V8F_LAUNCH_INTENT.md:118,252 |
| `+25.7101` | 2566,2692 | RESULT_V8F_LAUNCH_INTENT.md:118,252 |
| `+19.3282` | 2566,2692 | RESULT_V8F_LAUNCH_INTENT.md:118,252 |
| `+13.9956` | 2566,2692 | RESULT_V8F_LAUNCH_INTENT.md:118,252 |
| `+25.0836` | 2566,2692 | RESULT_V8F_LAUNCH_INTENT.md:118,252 |
| `+0.2107` | 2567,2693 | RESULT_V8F_LAUNCH_INTENT.md:119,253 |
| `+0.1535` | 2567,2693 | RESULT_V8F_LAUNCH_INTENT.md:119,253 |
| `+0.2708` | 2567,2693 | RESULT_V8F_LAUNCH_INTENT.md:119,253 |
| `+0.2180` | 2567,2693 | RESULT_V8F_LAUNCH_INTENT.md:119,253 |
| `+0.1586` | 2567,2693 | RESULT_V8F_LAUNCH_INTENT.md:119,253 |
| `+0.2848` | 2567,2693 | RESULT_V8F_LAUNCH_INTENT.md:119,253 |
| `+0.1071` | 2568,2696 | RESULT_V8F_DAG_ORC.md:71,152,241; RESULT_V8F_LAUNCH_INTENT.md:120,256 |
| `+0.0671` | 2568,2696 | RESULT_V8F_LAUNCH_INTENT.md:120,256 |
| `+0.1558` | 2568,2696 | RESULT_V8F_LAUNCH_INTENT.md:120,256 |
| `+0.1313` | 2568,2696 | RESULT_V8F_LAUNCH_INTENT.md:120,256 |
| `+0.0797` | 2568,2696 | RESULT_V8F_LAUNCH_INTENT.md:120,256 |
| `+0.1961` | 2568,2696 | RESULT_V8F_LAUNCH_INTENT.md:120,256 |
| `+0.0990` | 2569,2702 | RESULT_V8F_LAUNCH_INTENT.md:121,262 |
| `+0.0727` | 2569,2702 | RESULT_V8F_LAUNCH_INTENT.md:121,262 |
| `+0.1284` | 2569,2702 | RESULT_V8F_LAUNCH_INTENT.md:121,262 |
| `+0.0957` | 2569,2702 | RESULT_V8F_LAUNCH_INTENT.md:121,262 |
| `+0.0690` | 2569,2702 | RESULT_V8F_LAUNCH_INTENT.md:121,262 |
| `+0.1258` | 2569,2702 | RESULT_V8F_LAUNCH_INTENT.md:121,262 |
| `+8.9439` | 2570,2716 | RESULT_V8F_LAUNCH_INTENT.md:122,276 |
| `+5.3604` | 2570,2716 | RESULT_V8F_LAUNCH_INTENT.md:122,276 |
| `+12.9899` | 2570,2716 | RESULT_V8F_LAUNCH_INTENT.md:122,276 |
| `+8.6844` | 2570,2716 | RESULT_V8F_DAG_ORC.md:103,206; RESULT_V8F_LAUNCH_INTENT.md:122,276 |
| `+5.1690` | 2570,2716 | RESULT_V8F_DAG_ORC.md:103; RESULT_V8F_LAUNCH_INTENT.md:122,276 |
| `+12.5740` | 2570,2716 | RESULT_V8F_DAG_ORC.md:103; RESULT_V8F_LAUNCH_INTENT.md:122,276 |
| `+0.0964` | 2571,2717 | RESULT_V8F_LAUNCH_INTENT.md:123,277 |
| `+0.0650` | 2571,2717 | RESULT_V8F_LAUNCH_INTENT.md:123,277 |
| `+0.1321` | 2571,2717 | RESULT_V8F_LAUNCH_INTENT.md:123,277 |
| `+0.1352` | 2571,2717 | RESULT_V8F_DAG_ORC.md:104,207; RESULT_V8F_LAUNCH_INTENT.md:123,277 |
| `+0.0889` | 2571,2717 | RESULT_V8F_DAG_ORC.md:104; RESULT_V8F_LAUNCH_INTENT.md:123,277 |
| `+0.1915` | 2571,2717 | RESULT_V8F_DAG_ORC.md:104; RESULT_V8F_LAUNCH_INTENT.md:123,277 |
| `28.7` | 2571 | RESULT_V8F_LAUNCH_INTENT.md:123 |
| `+0.0462` | 2572,2720 | RESULT_V8F_LAUNCH_INTENT.md:124,280 |
| `+0.0716` | 2572,2720 | RESULT_V8F_LAUNCH_INTENT.md:124,280 |
| `+0.0529` | 2572,2720 | RESULT_V8F_DAG_ORC.md:107,208; RESULT_V8F_LAUNCH_INTENT.md:124,280 |
| `+0.0281` | 2572,2720 | RESULT_V8F_DAG_ORC.md:107; RESULT_V8F_LAUNCH_INTENT.md:124,280; MLAUNCH NOTES.md:14 |
| `12.7` | 2572 | RESULT_V8F_LAUNCH_INTENT.md:124 |
| `+0.0620` | 2573,2726 | RESULT_V8F_SPEEDSUB_T1.md:166; PREREG_V8F_SPEEDSUB.md:10; RESULT_V8F_LAUNCH_INTENT.md:125,286 |
| `+0.0415` | 2573,2726 | RESULT_V8F_LAUNCH_INTENT.md:125,286 |
| `+0.0870` | 2573,2726 | RESULT_V8F_DAG_ORC.md:81; RESULT_V8F_LAUNCH_INTENT.md:125,286 |
| `+0.0391` | 2573,2726 | RESULT_V8F_DAG_ORC.md:71,113,152,241; RESULT_V8F_LAUNCH_INTENT.md:125,286 |
| `+0.0864` | 2573,2726 | RESULT_V8F_SPEEDSUB_T1.md:154; RESULT_V8F_DAG_ORC.md:113; REP_S0_SPEED_BIAS.md:22; RESULT_V8F_LAUNCH_INTENT.md:125,286 |
| `-2.6` | 2573 | RESULT_V8F_LAUNCH_INTENT.md:125 |
| `+9.3556` | 2574,2716 | RESULT_V8F_LAUNCH_INTENT.md:126,276 |
| `+5.7951` | 2574,2716 | RESULT_V8F_LAUNCH_INTENT.md:126,276 |
| `+13.2237` | 2574,2716 | RESULT_V8F_LAUNCH_INTENT.md:126,276 |
| `+9.5748` | 2574,2716 | RESULT_V8F_LAUNCH_INTENT.md:126,276 |
| `+5.8282` | 2574,2716 | RESULT_V8F_LAUNCH_INTENT.md:126,276 |
| `+13.7093` | 2574,2716 | RESULT_V8F_LAUNCH_INTENT.md:126,276 |
| `2.3` | 2574 | RESULT_V8F_SPEEDSUB.md:371; RESULT_V8F_DAG_ORC.md:91; RESULT_V8F_LAUNCH_INTENT.md:14,83,126 |
| `+0.1445` | 2575,2717 | RESULT_V8F_LAUNCH_INTENT.md:127,277 |
| `+0.0951` | 2575,2717 | RESULT_V8F_LAUNCH_INTENT.md:127,277 |
| `+0.2028` | 2575,2717 | RESULT_V8F_LAUNCH_INTENT.md:127,277 |
| `+0.1343` | 2575,2717 | RESULT_V8F_LAUNCH_INTENT.md:127,277 |
| `+0.0858` | 2575,2717 | RESULT_V8F_LAUNCH_INTENT.md:127,277 |
| `+0.1912` | 2575,2717 | RESULT_V8F_LAUNCH_INTENT.md:127,277 |
| `-7.6` | 2575 | RESULT_V8F_LAUNCH_INTENT.md:127 |
| `+0.0767` | 2576,2720 | RESULT_V8F_LAUNCH_INTENT.md:128,280 |
| `+0.0413` | 2576,2720 | RESULT_V8F_LAUNCH_INTENT.md:128,280 |
| `+0.1232` | 2576,2720 | RESULT_V8F_LAUNCH_INTENT.md:128,280 |
| `+0.0589` | 2576,2720 | RESULT_V8F_LAUNCH_INTENT.md:128,280 |
| `+0.0334` | 2576,2720 | RESULT_V8F_LAUNCH_INTENT.md:128,280 |
| `+0.0896` | 2576,2720 | PREREG_V8F_DAG_ORC.md:169; RESULT_V8F_LAUNCH_INTENT.md:128,280 |
| `+0.0672` | 2577,2726 | RESULT_V8F_LAUNCH_INTENT.md:129,286 |
| `+0.0457` | 2577,2726 | RESULT_V8F_LAUNCH_INTENT.md:129,286 |
| `+0.0652` | 2577,2726 | RESULT_V8F_LAUNCH_INTENT.md:129,286 |
| `+0.0440` | 2577,2726 | RESULT_V8F_LAUNCH_INTENT.md:129,286 |
| `-3.1` | 2577 | RESULT_V8F_LAUNCH_INTENT.md:129 |
| `2.2128` | 2586 | SPEEDSUB_T1_READ.json:3798; RESULT_V8F_LAUNCH_INTENT.md:76 |
| `-1.2305` | 2586,2684 | RESULT_V8F_LAUNCH_INTENT.md:76,244; LAUNCH_INTENT_READ.json:2621 |
| `-1.4302` | 2586,2684 | RESULT_V8F_LAUNCH_INTENT.md:76,244; LAUNCH_INTENT_READ.json:2622 |
| `-1.0448` | 2586,2684 | RESULT_V8F_LAUNCH_INTENT.md:76,244; LAUNCH_INTENT_READ.json:2623 |
| `-1.2012` | 2586,2684 | RESULT_V8F_DAG_ORC.md:95; SPEEDSUB_T1_READ.json:2624; DAGORC_READ.json:2733,3271; RESULT_V8F_LAUNCH_INTENT.md:76,244; LAUNCH_INTENT_READ.json:1545 |
| `-1.3945` | 2586,2684 | RESULT_V8F_DAG_ORC.md:95; SPEEDSUB_T1_READ.json:2625; DAGORC_READ.json:2734,3272; RESULT_V8F_LAUNCH_INTENT.md:76,244; LAUNCH_INTENT_READ.json:1546 |
| `-1.0186` | 2586,2684 | RESULT_V8F_DAG_ORC.md:95; SPEEDSUB_T1_READ.json:2626; DAGORC_READ.json:2735,3273; RESULT_V8F_LAUNCH_INTENT.md:76,244; LAUNCH_INTENT_READ.json:1547 |
| `2.6877` | 2587 | RESULT_V8F_LAUNCH_INTENT.md:77 |
| `-0.7556` | 2587,2684 | RESULT_V8F_LAUNCH_INTENT.md:77,244; LAUNCH_INTENT_READ.json:3697 |
| `-1.0164` | 2587,2684 | RESULT_V8F_LAUNCH_INTENT.md:77,244; LAUNCH_INTENT_READ.json:3698 |
| `-0.4950` | 2587,2684 | RESULT_V8F_DAG_ORC.md:98,153; RESULT_V8F_LAUNCH_INTENT.md:77,244 |
| `-0.7987` | 2587,2684 | SPEEDSUB_T1_READ.json:3048; RESULT_V8F_LAUNCH_INTENT.md:77,244; LAUNCH_INTENT_READ.json:2083 |
| `-1.0586` | 2587,2684 | SPEEDSUB_T1_READ.json:3049; RESULT_V8F_LAUNCH_INTENT.md:77,244; LAUNCH_INTENT_READ.json:2084 |
| `-0.5428` | 2587,2684 | SPEEDSUB_T1_READ.json:3050; RESULT_V8F_LAUNCH_INTENT.md:77,244; LAUNCH_INTENT_READ.json:2085 |
| `2.3684` | 2588 | RESULT_V8F_LAUNCH_INTENT.md:78 |
| `-1.0749` | 2588,2708 | RESULT_V8F_LAUNCH_INTENT.md:78,268; LAUNCH_INTENT_READ.json:2889 |
| `-1.2743` | 2588,2708 | RESULT_V8F_LAUNCH_INTENT.md:78,268; LAUNCH_INTENT_READ.json:2890 |
| `-0.8735` | 2588,2708 | RESULT_V8F_LAUNCH_INTENT.md:78,268; LAUNCH_INTENT_READ.json:2891 |
| `-1.1231` | 2588,2708 | RESULT_V8F_DAG_ORC.md:95; SPEEDSUB_T1_READ.json:2836; DAGORC_READ.json:3001,3539; RESULT_V8F_LAUNCH_INTENT.md:78,268; LAUNCH_INTENT_READ.json:1813 |
| `-1.3301` | 2588,2708 | RESULT_V8F_DAG_ORC.md:95; SPEEDSUB_T1_READ.json:2837; DAGORC_READ.json:3002,3540; RESULT_V8F_LAUNCH_INTENT.md:78,268; LAUNCH_INTENT_READ.json:1814 |
| `-0.9159` | 2588,2708 | RESULT_V8F_DAG_ORC.md:95; SPEEDSUB_T1_READ.json:2838; DAGORC_READ.json:3003,3541; RESULT_V8F_LAUNCH_INTENT.md:78,268; LAUNCH_INTENT_READ.json:1815 |
| `2.2976` | 2589 | RESULT_V8F_LAUNCH_INTENT.md:79 |
| `-1.1457` | 2589,2708 | RESULT_V8F_LAUNCH_INTENT.md:79,268; LAUNCH_INTENT_READ.json:3965 |
| `-1.3570` | 2589,2708 | RESULT_V8F_LAUNCH_INTENT.md:79,268 |
| `-0.9417` | 2589,2708 | RESULT_V8F_LAUNCH_INTENT.md:79,268; LAUNCH_INTENT_READ.json:3967 |
| `-1.1116` | 2589,2708 | SPEEDSUB_T1_READ.json:3260; RESULT_V8F_LAUNCH_INTENT.md:79,268; LAUNCH_INTENT_READ.json:2351 |
| `-1.3180` | 2589,2708 | RESULT_V8F_LAUNCH_INTENT.md:79,268 |
| `-0.9119` | 2589,2708 | SPEEDSUB_T1_READ.json:3262; RESULT_V8F_LAUNCH_INTENT.md:79,268; LAUNCH_INTENT_READ.json:2353 |
| `+0.1797` | 2601,2629 | REP_S0_SPEED_BIAS.md:26; RESULT_V8F_LAUNCH_INTENT.md:144,176; check_result_filled.out:112 |
| `+0.0998` | 2601,2629 | RESULT_V8F_LAUNCH_INTENT.md:144,176; check_result_filled.out:112 |
| `+0.2726` | 2601,2629 | RESULT_V8F_LAUNCH_INTENT.md:144,176; check_result_filled.out:112 |
| `+0.7391` | 2601,2631,2776 | RESULT_V8F_LAUNCH_INTENT.md:144,178,342; check_result_filled.out:115 |
| `+0.4366` | 2601,2631,2776 | RESULT_V8F_LAUNCH_INTENT.md:144,178,342; check_result_filled.out:115 |
| `+1.0952` | 2601,2631,2776 | RESULT_V8F_LAUNCH_INTENT.md:144,178,342; check_result_filled.out:115 |
| `+0.0720` | 2601,2643 | RESULT_V8F_LAUNCH_INTENT.md:144,192 |
| `+0.0403` | 2601,2643 | RESULT_V8F_SPEEDSUB_T1.md:160; RESULT_V8F_LAUNCH_INTENT.md:144,192 |
| `+0.1109` | 2601,2643 | RESULT_V8F_LAUNCH_INTENT.md:144,192 |
| `+1.4444` | 2602,2644 | RESULT_V8F_LAUNCH_INTENT.md:144,193 |
| `+0.6003` | 2602,2644 | RESULT_V8F_LAUNCH_INTENT.md:144,193 |
| `+2.4872` | 2602,2644 | RESULT_V8F_LAUNCH_INTENT.md:144,193 |
| `+0.0493` | 2602,2630 | RESULT_V8F_LAUNCH_INTENT.md:144,177; check_result_filled.out:113 |
| `+0.0170` | 2602,2630 | RESULT_V8F_LAUNCH_INTENT.md:144,177; check_result_filled.out:113 |
| `+0.0801` | 2602,2630 | RESULT_V8F_LAUNCH_INTENT.md:144,177; check_result_filled.out:113 |
| `204` | 2615 | PREREG_V8F_LAUNCH_INTENT.md:243; launch-intent-v2/NOTES.md:14; RESULT_V8F_LAUNCH_INTENT.md:161; check_result_filled.out:4,121,137 |
| `-0.0393` | 2629 | RESULT_V8F_LAUNCH_INTENT.md:176; LAUNCH_INTENT_READ.json:3174,4782 |
| `-0.0031` | 2629 | RESULT_V8F_LAUNCH_INTENT.md:176; LAUNCH_INTENT_READ.json:3175,4783 |
| `+0.0074` | 2629 | RESULT_V8F_LAUNCH_INTENT.md:176 |
| `+0.0310` | 2629 | RESULT_V8F_LAUNCH_INTENT.md:176 |
| `-0.0744` | 2629 | RESULT_V8F_LAUNCH_INTENT.md:176; LAUNCH_INTENT_READ.json:4517,5893 |
| `-0.1220` | 2629 | RESULT_V8F_LAUNCH_INTENT.md:176 |
| `-0.0388` | 2630,2734 | RESULT_V8F_LAUNCH_INTENT.md:177,301; LAUNCH_INTENT_READ.json:3188,4802 |
| `-0.0249` | 2630,2712 | P0_RF0.json (rebase):144; RESULT_V8F_LAUNCH_INTENT.md:177,272; LAUNCH_INTENT_READ.json:3456,4022,5544,5960 |
| `-0.0250` | 2630 | RESULT_V8F_LAUNCH_INTENT.md:177 |
| `-0.0521` | 2630 | RESULT_V8F_LAUNCH_INTENT.md:177; LAUNCH_INTENT_READ.json:4532,5914 |
| `-0.0568` | 2631,2690 | RESULT_V8F_DAG_ORC.md:101; SPEEDSUB_T1_READ.json:2684; DAGORC_READ.json:2817,3355; RESULT_V8F_LAUNCH_INTENT.md:178,250; LAUNCH_INTENT_READ.json:1629,3201,4821,4893 |
| `-0.1137` | 2631 | RESULT_V8F_LAUNCH_INTENT.md:178; LAUNCH_INTENT_READ.json:3202,4822 |
| `-0.0186` | 2631 | RESULT_V8F_LAUNCH_INTENT.md:178; LAUNCH_INTENT_READ.json:4278,5192 |
| `+0.1079` | 2631 | RESULT_V8F_LAUNCH_INTENT.md:178 |
| `-0.2560` | 2631,2776 | RESULT_V8F_LAUNCH_INTENT.md:178,342; check_result_filled.out:115 |
| `-0.4105` | 2631,2776 | RESULT_V8F_LAUNCH_INTENT.md:178,342; LAUNCH_INTENT_READ.json:4546,5934; check_result_filled.out:115 |
| `-0.1222` | 2631,2776 | RESULT_V8F_LAUNCH_INTENT.md:178,342; LAUNCH_INTENT_READ.json:4547,5935; check_result_filled.out:115 |
| `+0.0212` | 2632 | RESULT_V8F_LAUNCH_INTENT.md:179 |
| `+0.0037` | 2632 | RESULT_V8F_DAG_ORC.md:75,156; RESULT_V8F_LAUNCH_INTENT.md:179 |
| `-0.0161` | 2632 | RESULT_V8F_LAUNCH_INTENT.md:179; LAUNCH_INTENT_READ.json:3484,5584 |
| `+0.0089` | 2632 | RESULT_V8F_LAUNCH_INTENT.md:179 |
| `-0.0039` | 2632 | RESULT_V8F_LAUNCH_INTENT.md:179; LAUNCH_INTENT_READ.json:4560,5954 |
| `+0.0117` | 2632,2652 | RESULT_V8F_DAG_ORC.md:71,87,222; RESULT_V8F_LAUNCH_INTENT.md:179,201 |
| `595` | 2632,2632,2632,2632 | RESULT_V8F_SPEEDSUB_T1.md:115,115,115,115; RESULT_V8F_DAG_ORC.md:47; SPEEDSUB_T1_READ.json:968,1180,1392,1604...; DAGORC_READ.json:641,909,1179,1447...; P0_RF0.json (rebase):118,125; RESULT_V8F_LAUNCH_INTENT.md:179,179,179,179; LAUNCH_INTENT_READ.json:1067,1335,1605,1873... |
| `70` | 2632,2632,2632,2632 | RESULT_V8F_SPEEDSUB_T1.md:115,115,115,115; RESULT_V8F_DAG_ORC.md:47; SPEEDSUB_T1_READ.json:968,1180,1392,1604...; DAGORC_READ.json:641,909,1179,1447...; P0_RF0.json (rebase):118,125; RESULT_V8F_LAUNCH_INTENT.md:179,179,179,179; LAUNCH_INTENT_READ.json:1067,1335,1605,1873...; check_result_filled.out:136 |
| `-0.0882` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180; LAUNCH_INTENT_READ.json:3230,4862 |
| `+0.1748` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180 |
| `-0.0740` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180 |
| `+0.0952` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180 |
| `+0.0590` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180 |
| `-0.0277` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180; LAUNCH_INTENT_READ.json:3498,5604 |
| `+0.1582` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180 |
| `-0.0382` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180; LAUNCH_INTENT_READ.json:4573,5973; MLAUNCH.json:1209 |
| `-0.1706` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180; LAUNCH_INTENT_READ.json:4574,5974 |
| `+0.0776` | 2633 | RESULT_V8F_LAUNCH_INTENT.md:180 |
| `712` | 2633,2633 | RESULT_V8F_SPEEDSUB_T1.md:116; SPEEDSUB_T1_READ.json:1614,4802,5014,5663; DAGORC_READ.json:1461; RESULT_V8F_LAUNCH_INTENT.md:180,180; LAUNCH_INTENT_READ.json:2695,3233,4577 |
| `72` | 2633,2633,2633,2634,2634... | RESULT_V8F_SPEEDSUB_T1.md:116,116,116,116...; PREREG_V8F_LAUNCH_INTENT.md:33; PREREG_V8F_DAG_ORC.md:272,300; RESULT_V8F_DAG_ORC.md:48,49,49; SPEEDSUB_T1_READ.json:978,988,1190,1200...; launch-intent-flag/RESULTS.md:30; DAGORC_READ.json:655,669,923,937...; P0_RF0.json (rebase):141,148,164,171; RESULT_V8F_LAUNCH_INTENT.md:180,180,180,181...; LAUNCH_INTENT_READ.json:1081,1095,1349,1363... |
| `710` | 2633,2633 | RESULT_V8F_DAG_ORC.md:48; SPEEDSUB_T1_READ.json:2040,2678,2890,5691; DAGORC_READ.json:923,1999,2537,2807...; RESULT_V8F_LAUNCH_INTENT.md:180,180; LAUNCH_INTENT_READ.json:1349,1619,1887,2963... |
| `73` | 2633 | RESULT_V8F_SPEEDSUB.md:251; RESULT_V8F_SPEEDSUB_T1.md:462; RESULT_V8F_DAG_ORC.md:48; SPEEDSUB_T1_READ.json:4590,4802; FINDING_1_ATTRIBUTION.md:23; DAGORC_READ.json:1999,2537; git log (cited commits):28; RESULT_V8F_LAUNCH_INTENT.md:180; LAUNCH_INTENT_READ.json:3501 |
| `-0.0359` | 2634 | RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:3244,4882 |
| `+0.0563` | 2634 | RESULT_V8F_LAUNCH_INTENT.md:181 |
| `-0.0605` | 2634 | RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:4320,5252 |
| `-0.0479` | 2634 | RESULT_V8F_SPEEDSUB_T1.md:162; SPEEDSUB_T1_READ.json:4038; REP_S0_SPEED_BIAS.md:23; RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:3512,5624 |
| `-0.0062` | 2634 | RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:4587,5993 |
| `-0.0363` | 2634 | RESULT_V8F_SPEEDSUB_T1.md:167; SPEEDSUB_T1_READ.json:4089; RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:4588,5994 |
| `+0.0189` | 2634 | RESULT_V8F_LAUNCH_INTENT.md:181 |
| `654` | 2634,2634 | RESULT_V8F_SPEEDSUB_T1.md:117,117; RESULT_V8F_DAG_ORC.md:49; SPEEDSUB_T1_READ.json:988,1624,4388,5024...; DAGORC_READ.json:1745,2283; RESULT_V8F_LAUNCH_INTENT.md:181,181; LAUNCH_INTENT_READ.json:3247,4591 |
| `652` | 2634 | RESULT_V8F_DAG_ORC.md:49; SPEEDSUB_T1_READ.json:1838,2050,2688,2900...; DAGORC_READ.json:2013,2551,2821,3089...; P0_RF0.json (rebase):164; RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:1633,1901,2439,4323 |
| `653` | 2634 | RESULT_V8F_SPEEDSUB_T1.md:117; SPEEDSUB_T1_READ.json:1200,2474,3538,4600...; DAGORC_READ.json:669,937,1207,1475; RESULT_V8F_LAUNCH_INTENT.md:181; LAUNCH_INTENT_READ.json:1095,1363,2709,2977... |
| `-0.4211` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:3257,4908 |
| `-0.6883` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:3258,4909 |
| `-0.1899` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:3259,4910 |
| `+0.0892` | 2640 | RESULT_V8F_SPEEDSUB_T1.md:157; RESULT_V8F_LAUNCH_INTENT.md:189 |
| `-0.2326` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:4334,5279 |
| `+0.3963` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189 |
| `+0.1038` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189 |
| `-0.2848` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:3526,5651 |
| `+0.5761` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189 |
| `-0.1749` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:4601,6020 |
| `-0.5225` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189; LAUNCH_INTENT_READ.json:4602,6021 |
| `+0.1333` | 2640 | RESULT_V8F_LAUNCH_INTENT.md:189 |
| `2366` | 2641,2641,2642,2642,2644... | RESULT_V8F_SPEEDSUB_T1.md:119,119,120,120...; RESULT_V8F_DAG_ORC.md:51,52,54; SPEEDSUB_T1_READ.json:1008,1018,1038,1220...; DAGORC_READ.json:697,711,739,965...; RESULT_V8F_LAUNCH_INTENT.md:190,190,191,191...; LAUNCH_INTENT_READ.json:1123,1137,1165,1391... |
| `-0.0191` | 2643 | RESULT_V8F_LAUNCH_INTENT.md:192; LAUNCH_INTENT_READ.json:3299,4968 |
| `-0.0365` | 2643 | RESULT_V8F_LAUNCH_INTENT.md:192; LAUNCH_INTENT_READ.json:3300,4969 |
| `-0.0030` | 2643 | RESULT_V8F_LAUNCH_INTENT.md:192 |
| `+0.0053` | 2643 | RESULT_V8F_LAUNCH_INTENT.md:192 |
| `+0.0234` | 2643 | RESULT_V8F_LAUNCH_INTENT.md:192 |
| `-0.0270` | 2643,2712 | RESULT_V8F_LAUNCH_INTENT.md:192,272 |
| `-0.0572` | 2643 | RESULT_V8F_LAUNCH_INTENT.md:192; LAUNCH_INTENT_READ.json:4644,6081 |
| `+0.0001` | 2643 | GOALS_AND_CLAIMS.md (register rows):3; REP_S0_SPEED_BIAS.md:25; RESULT_V8F_LAUNCH_INTENT.md:192 |
| `-0.0438` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193; LAUNCH_INTENT_READ.json:3313,4988; MLAUNCH.json:580 |
| `-0.1565` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193; LAUNCH_INTENT_READ.json:3314,4989 |
| `+0.0921` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193 |
| `+0.0906` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193 |
| `-0.0774` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193; LAUNCH_INTENT_READ.json:4390,5359 |
| `+0.2676` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193 |
| `-0.3908` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193; LAUNCH_INTENT_READ.json:4657,6100 |
| `-0.7329` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193; LAUNCH_INTENT_READ.json:4658,6101 |
| `-0.0748` | 2644 | RESULT_V8F_LAUNCH_INTENT.md:193; LAUNCH_INTENT_READ.json:4659,6102 |
| `2105` | 2645 | RESULT_V8F_SPEEDSUB_T1.md:123,123,129,129; RESULT_V8F_DAG_ORC.md:55,61; SPEEDSUB_T1_READ.json:1472,1532,1684,1744...; DAGORC_READ.json:753,837,1829,1913...; P0_RF0.json (rebase):302,309,446,453; RESULT_V8F_LAUNCH_INTENT.md:194; LAUNCH_INTENT_READ.json:1179,1263,1717,1801... |
| `135` | 2645,2645,2645,2645 | RESULT_V8F_SPEEDSUB_T1.md:123,123,123,123...; RESULT_V8F_DAG_ORC.md:55,55,61,61; SPEEDSUB_T1_READ.json:1048,1108,1260,1320...; DAGORC_READ.json:753,837,1021,1105...; P0_RF0.json (rebase):302,309,446,453; RESULT_V8F_LAUNCH_INTENT.md:194,194,194,194; LAUNCH_INTENT_READ.json:1179,1263,1447,1531... |
| `2106` | 2645,2645 | RESULT_V8F_SPEEDSUB_T1.md:123,129; RESULT_V8F_DAG_ORC.md:55,61; SPEEDSUB_T1_READ.json:1260,1320,2110,2170...; DAGORC_READ.json:1021,1105,1291,1375...; RESULT_V8F_LAUNCH_INTENT.md:194,194; LAUNCH_INTENT_READ.json:1447,1531,1985,2069... |
| `-0.0221` | 2651 | RESULT_V8F_SPEEDSUB_T1.md:162; SPEEDSUB_T1_READ.json:4040; RESULT_V8F_LAUNCH_INTENT.md:200; LAUNCH_INTENT_READ.json:4418,5405 |
| `+0.0150` | 2651,2652 | RESULT_V8F_LAUNCH_INTENT.md:200,201 |
| `+0.0079` | 2651,2686 | RESULT_V8F_LAUNCH_INTENT.md:200,246 |
| `+0.0371` | 2651 | RESULT_V8F_SPEEDSUB_T1.md:167; RESULT_V8F_DAG_ORC.md:70,79,151,160; RESULT_V8F_LAUNCH_INTENT.md:200 |
| `-0.0121` | 2652 | RESULT_V8F_DAG_ORC.md:83; DAGORC_READ.json:1046; RF_REP.json:227; RESULT_V8F_LAUNCH_INTENT.md:201; LAUNCH_INTENT_READ.json:1472,3624,5797 |
| `-0.0025` | 2652 | RESULT_V8F_LAUNCH_INTENT.md:201; LAUNCH_INTENT_READ.json:4699,6166 |
| `+0.0171` | 2652 | RESULT_V8F_LAUNCH_INTENT.md:201 |
| `-0.1265` | 2659 | RESULT_V8F_SPEEDSUB.md:226; RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:3369,5081 |
| `-0.2605` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:3370,5082 |
| `-0.0083` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:4445,5451 |
| `-0.1748` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:4446,5452 |
| `+0.1592` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209 |
| `+0.1814` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209 |
| `-0.1398` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:3638,5824 |
| `+0.5613` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209 |
| `-0.2141` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:4713,6193 |
| `-0.4437` | 2659 | RESULT_V8F_LAUNCH_INTENT.md:209; LAUNCH_INTENT_READ.json:4714,6194 |
| `2321` | 2659,2659,2659,2659 | RESULT_V8F_DAG_ORC.md:58; DAGORC_READ.json:795,1063,1871,2139...; RESULT_V8F_LAUNCH_INTENT.md:209,209,209,209; LAUNCH_INTENT_READ.json:1221,1489,3373,3641... |
| `-0.2295` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:3383,5101 |
| `-0.4581` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:3384,5102 |
| `-0.1178` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:4459,5471 |
| `-0.4129` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:4460,5472 |
| `+0.1810` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210 |
| `-0.1536` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:3651,5843 |
| `-0.5723` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:3652,5844 |
| `+0.2938` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210 |
| `-0.0448` | 2660 | SPEEDSUB_T1_BESIDE.json:236; RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:4727,6213 |
| `-0.3621` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210; LAUNCH_INTENT_READ.json:4728,6214 |
| `+0.2538` | 2660 | RESULT_V8F_LAUNCH_INTENT.md:210 |
| `2313` | 2660,2660,2660,2660 | RESULT_V8F_DAG_ORC.md:59; DAGORC_READ.json:809,1077,1885,2153...; RESULT_V8F_LAUNCH_INTENT.md:210,210,210,210; LAUNCH_INTENT_READ.json:1235,1503,3387,3655... |
| `-0.4006` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:3397,5121 |
| `-0.7914` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:3398,5122 |
| `-0.1193` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:4473,5491 |
| `-0.6477` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:4474,5492 |
| `+0.4084` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211 |
| `-0.3352` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:3665,5863 |
| `-1.1408` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:3666,5864 |
| `+0.5368` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211 |
| `-0.0523` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:4741,6233 |
| `-0.5763` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211; LAUNCH_INTENT_READ.json:4742,6234 |
| `+0.4724` | 2661 | RESULT_V8F_LAUNCH_INTENT.md:211 |
| `2290` | 2661,2661,2661,2661 | RESULT_V8F_DAG_ORC.md:60; DAGORC_READ.json:823,1091,1899,2167...; RESULT_V8F_LAUNCH_INTENT.md:211,211,211,211; LAUNCH_INTENT_READ.json:1249,1517,3401,3669... |
| `-0.7962` | 2685 | RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:2635,4787 |
| `-1.0078` | 2685 | RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:2636,4788 |
| `-0.6137` | 2685 | RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:2637,4789 |
| `-0.7761` | 2685 | RESULT_V8F_DAG_ORC.md:96; SPEEDSUB_T1_READ.json:2634; DAGORC_READ.json:2747,3285; RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:1559,4793 |
| `-0.9806` | 2685 | RESULT_V8F_DAG_ORC.md:96; SPEEDSUB_T1_READ.json:2635; DAGORC_READ.json:2748,3286; RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:1560,4794 |
| `-0.5996` | 2685 | RESULT_V8F_DAG_ORC.md:96; SPEEDSUB_T1_READ.json:2636; DAGORC_READ.json:2749,3287; RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:1561,4795 |
| `-0.4314` | 2685 | RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:3711,5157 |
| `-0.6628` | 2685 | RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:3712,5158 |
| `-0.2286` | 2685 | RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:3713,5159 |
| `-0.4388` | 2685 | SPEEDSUB_T1_READ.json:3058; RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:2097,5163 |
| `-0.6617` | 2685 | SPEEDSUB_T1_READ.json:3059; RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:2098,5164 |
| `-0.2409` | 2685 | SPEEDSUB_T1_READ.json:3060; RESULT_V8F_LAUNCH_INTENT.md:245; LAUNCH_INTENT_READ.json:2099,5165 |
| `-0.6994` | 2686,2751 | GOALS_AND_CLAIMS.md (register rows):6; RESULT_V8F_LAUNCH_INTENT.md:246,387; LAUNCH_INTENT_READ.json:2649,4807; MLAUNCH.json:873; MLAUNCH NOTES.md:21 |
| `-0.8611` | 2686 | RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:2650,4808 |
| `-0.5457` | 2686 | RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:2651,4809 |
| `-0.6809` | 2686 | RESULT_V8F_DAG_ORC.md:97; SPEEDSUB_T1_READ.json:2644; DAGORC_READ.json:2761,3299; RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:1573,4813 |
| `-0.8443` | 2686 | RESULT_V8F_DAG_ORC.md:97; SPEEDSUB_T1_READ.json:2645; DAGORC_READ.json:2762,3300; RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:1574,4814 |
| `-0.5214` | 2686 | RESULT_V8F_DAG_ORC.md:97; SPEEDSUB_T1_READ.json:2646; DAGORC_READ.json:2763,3301; RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:1575,4815 |
| `-0.1720` | 2686 | RESULT_V8F_LAUNCH_INTENT.md:246 |
| `-0.4096` | 2686 | RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:3726,5178 |
| `+0.0640` | 2686 | RESULT_V8F_LAUNCH_INTENT.md:246 |
| `-0.2213` | 2686 | SPEEDSUB_T1_READ.json:3068; RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:2111,5183 |
| `-0.4578` | 2686 | SPEEDSUB_T1_READ.json:3069; RESULT_V8F_LAUNCH_INTENT.md:246; LAUNCH_INTENT_READ.json:2112,5184 |
| `-0.8740` | 2687 | RESULT_V8F_LAUNCH_INTENT.md:247 |
| `-1.2470` | 2687 | RESULT_V8F_LAUNCH_INTENT.md:247 |
| `-0.5533` | 2687 | RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:2665,4829 |
| `-0.8172` | 2687 | RESULT_V8F_DAG_ORC.md:98; SPEEDSUB_T1_READ.json:2654; DAGORC_READ.json:2775,3313; RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:1587,4833 |
| `-1.1770` | 2687 | RESULT_V8F_DAG_ORC.md:98; RESULT_V8F_LAUNCH_INTENT.md:247 |
| `-0.4998` | 2687 | RESULT_V8F_DAG_ORC.md:98; SPEEDSUB_T1_READ.json:2656; DAGORC_READ.json:2777,3315; RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:1589,4835 |
| `-0.2144` | 2687 | RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:3739,5197 |
| `-0.5718` | 2687 | RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:3740,5198 |
| `+0.1072` | 2687,2697 | RESULT_V8F_LAUNCH_INTENT.md:247,257 |
| `-0.2569` | 2687 | SPEEDSUB_T1_READ.json:3078; RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:2125,5203 |
| `-0.5998` | 2687 | SPEEDSUB_T1_READ.json:3079; RESULT_V8F_LAUNCH_INTENT.md:247; LAUNCH_INTENT_READ.json:2126,5204 |
| `+0.0569` | 2687 | RESULT_V8F_LAUNCH_INTENT.md:247 |
| `-0.0164` | 2688 | RESULT_V8F_LAUNCH_INTENT.md:248; LAUNCH_INTENT_READ.json:2678,4848 |
| `+0.0314` | 2688 | RESULT_V8F_LAUNCH_INTENT.md:248 |
| `+0.0272` | 2688 | RESULT_V8F_DAG_ORC.md:99; RESULT_V8F_LAUNCH_INTENT.md:248 |
| `-0.0338` | 2688 | RESULT_V8F_LAUNCH_INTENT.md:248; LAUNCH_INTENT_READ.json:3754,5218 |
| `+0.0219` | 2688 | RESULT_V8F_DAG_ORC.md:75; RESULT_V8F_LAUNCH_INTENT.md:248 |
| `-0.0248` | 2688 | SPEEDSUB_T1_READ.json:3089; RESULT_V8F_LAUNCH_INTENT.md:248; LAUNCH_INTENT_READ.json:2140,5224 |
| `-0.2519` | 2689 | RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:2691,4867 |
| `-0.6506` | 2689 | RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:2692,4868 |
| `+0.0829` | 2689 | RESULT_V8F_LAUNCH_INTENT.md:249 |
| `-0.3318` | 2689 | RESULT_V8F_DAG_ORC.md:100; SPEEDSUB_T1_READ.json:2674; DAGORC_READ.json:2803,3341; RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:1615,4873 |
| `-0.6942` | 2689 | RESULT_V8F_DAG_ORC.md:100; SPEEDSUB_T1_READ.json:2675; DAGORC_READ.json:2804,3342; RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:1616,4874; MLAUNCH.json:102 |
| `-0.0242` | 2689 | RESULT_V8F_DAG_ORC.md:100; SPEEDSUB_T1_READ.json:2676; DAGORC_READ.json:2805,3343; RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:1617,4875 |
| `-0.4459` | 2689 | RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:3768,5238 |
| `+0.4482` | 2689 | RESULT_V8F_LAUNCH_INTENT.md:249 |
| `-0.0980` | 2689 | RESULT_V8F_LAUNCH_INTENT.md:249 |
| `-0.5251` | 2689 | SPEEDSUB_T1_READ.json:3099; RESULT_V8F_LAUNCH_INTENT.md:249; LAUNCH_INTENT_READ.json:2154,5244 |
| `-0.0537` | 2690 | PREREG_V8F_DAG_ORC.md:24; RESULT_V8F_LAUNCH_INTENT.md:250; LAUNCH_INTENT_READ.json:2705,4887 |
| `-0.1320` | 2690 | RESULT_V8F_LAUNCH_INTENT.md:250 |
| `+0.0207` | 2690,2712 | RESULT_V8F_LAUNCH_INTENT.md:250,272 |
| `-0.1232` | 2690 | RESULT_V8F_DAG_ORC.md:101; SPEEDSUB_T1_READ.json:2685; DAGORC_READ.json:2818,3356; RESULT_V8F_LAUNCH_INTENT.md:250; LAUNCH_INTENT_READ.json:1630,4894 |
| `-0.1074` | 2690 | RESULT_V8F_LAUNCH_INTENT.md:250; LAUNCH_INTENT_READ.json:3782,5258 |
| `+0.1580` | 2690,2697 | RESULT_V8F_LAUNCH_INTENT.md:250,257 |
| `+0.0279` | 2690 | RESULT_V8F_DAG_ORC.md:72; RESULT_V8F_LAUNCH_INTENT.md:250 |
| `-0.0978` | 2690 | SPEEDSUB_T1_READ.json:3109; RESULT_V8F_LAUNCH_INTENT.md:250; LAUNCH_INTENT_READ.json:2168,5264 |
| `+0.2093` | 2690 | RESULT_V8F_LAUNCH_INTENT.md:250 |
| `-3.0131` | 2691 | RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:2719,4914 |
| `-3.7375` | 2691 | RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:2720,4915 |
| `-2.3511` | 2691 | RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:2721,4916 |
| `-2.5920` | 2691 | RESULT_V8F_DAG_ORC.md:102; RESULT_V8F_LAUNCH_INTENT.md:251 |
| `-3.2391` | 2691 | RESULT_V8F_DAG_ORC.md:102; SPEEDSUB_T1_READ.json:2695; DAGORC_READ.json:2832,3370; RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:1644,4921 |
| `-1.9751` | 2691 | RESULT_V8F_DAG_ORC.md:102; SPEEDSUB_T1_READ.json:2696; DAGORC_READ.json:2833,3371; RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:1645,4922 |
| `-0.9179` | 2691 | RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:3796,5285 |
| `+0.9648` | 2691 | RESULT_V8F_LAUNCH_INTENT.md:251 |
| `-0.0846` | 2691 | RESULT_V8F_SPEEDSUB_T1.md:161; SPEEDSUB_T1_READ.json:3118,4242; RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:2181,5290; MLAUNCH.json:261; MLAUNCH NOTES.md:14 |
| `-0.9878` | 2691 | SPEEDSUB_T1_READ.json:3119; RESULT_V8F_LAUNCH_INTENT.md:251; LAUNCH_INTENT_READ.json:2182,5291 |
| `+0.8799` | 2691 | RESULT_V8F_LAUNCH_INTENT.md:251 |
| `-0.9811` | 2694 | RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:2761,4974 |
| `-1.1264` | 2694 | RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:2762,4975 |
| `-0.8486` | 2694 | RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:2763,4976 |
| `-0.9620` | 2694 | RESULT_V8F_DAG_ORC.md:105; RESULT_V8F_LAUNCH_INTENT.md:254 |
| `-1.1077` | 2694 | RESULT_V8F_DAG_ORC.md:105; SPEEDSUB_T1_READ.json:2725; DAGORC_READ.json:2874,3412; RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:1686,4981 |
| `-0.8319` | 2694 | RESULT_V8F_DAG_ORC.md:105; SPEEDSUB_T1_READ.json:2726; DAGORC_READ.json:2875,3413; RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:1687,4982 |
| `-0.9733` | 2694 | RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:3837,5344 |
| `-1.1219` | 2694 | RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:3838,5345 |
| `-0.8367` | 2694 | RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:3839,5346 |
| `-0.9786` | 2694 | SPEEDSUB_T1_READ.json:3148; RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:2223,5350 |
| `-1.1252` | 2694 | SPEEDSUB_T1_READ.json:3149; RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:2224,5351 |
| `-0.8423` | 2694 | SPEEDSUB_T1_READ.json:3150; RESULT_V8F_LAUNCH_INTENT.md:254; LAUNCH_INTENT_READ.json:2225,5352 |
| `-2.5056` | 2695 | RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:2775,4994 |
| `-3.8661` | 2695 | RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:2776,4995 |
| `-1.3292` | 2695 | RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:2777,4996 |
| `-2.4618` | 2695 | RESULT_V8F_DAG_ORC.md:106; SPEEDSUB_T1_READ.json:2734; DAGORC_READ.json:2887,3425; RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:1699,5000 |
| `-3.8097` | 2695 | RESULT_V8F_DAG_ORC.md:106; SPEEDSUB_T1_READ.json:2735; DAGORC_READ.json:2888,3426; RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:1700,5001 |
| `-1.2924` | 2695 | RESULT_V8F_DAG_ORC.md:106; SPEEDSUB_T1_READ.json:2736; DAGORC_READ.json:2889,3427; RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:1701,5002 |
| `-1.1458` | 2695 | RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:3851,5364 |
| `-2.4394` | 2695 | RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:3852,5365 |
| `-0.0147` | 2695 | RESULT_V8F_DAG_ORC.md:70; DAGORC_READ.json:864; RF_REP.json:149; RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:1290,3853,5366 |
| `-1.2365` | 2695 | SPEEDSUB_T1_READ.json:3158; RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:2237,5370 |
| `-2.4350` | 2695 | RESULT_V8F_LAUNCH_INTENT.md:255 |
| `-0.1556` | 2695 | SPEEDSUB_T1_READ.json:3160; RESULT_V8F_LAUNCH_INTENT.md:255; LAUNCH_INTENT_READ.json:2239,5372 |
| `+0.1967` | 2697 | RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.2378` | 2697 | RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.1508` | 2697 | RESULT_V8F_DAG_ORC.md:108; RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.2319` | 2697 | RESULT_V8F_DAG_ORC.md:108; RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.1409` | 2697 | RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.0985` | 2697 | RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.1861` | 2697 | RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.1484` | 2697 | RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.1921` | 2697 | RESULT_V8F_DAG_ORC.md:86; RESULT_V8F_LAUNCH_INTENT.md:257 |
| `+0.1546` | 2698 | RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.0912` | 2698 | RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.2194` | 2698 | RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.1396` | 2698 | RESULT_V8F_DAG_ORC.md:109; RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.0771` | 2698 | RESULT_V8F_DAG_ORC.md:109; RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.1997` | 2698 | RESULT_V8F_DAG_ORC.md:109; RESULT_V8F_LAUNCH_INTENT.md:258 |
| `-0.0292` | 2698,2698 | SPEEDSUB_T1_READ.json:3189; RESULT_V8F_LAUNCH_INTENT.md:258,258; LAUNCH_INTENT_READ.json:2280,3894,5431,5437 |
| `+0.1096` | 2698 | RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.0446` | 2698 | RESULT_V8F_LAUNCH_INTENT.md:258 |
| `+0.1130` | 2698 | RESULT_V8F_LAUNCH_INTENT.md:258 |
| `-1.8693` | 2699 | RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:2831,5087 |
| `-2.4537` | 2699 | RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:2832,5088 |
| `-1.3756` | 2699 | RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:2833,5089 |
| `-1.4919` | 2699 | RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:3907,5457 |
| `-1.9634` | 2699 | RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:3908,5458 |
| `-1.0845` | 2699 | RESULT_V8F_LAUNCH_INTENT.md:259; LAUNCH_INTENT_READ.json:3909,5459 |
| `-4.7626` | 2700 | RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:2845,5107 |
| `-5.9590` | 2700 | RESULT_V8F_LAUNCH_INTENT.md:260 |
| `-3.7194` | 2700 | RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:2847,5109 |
| `-4.2598` | 2700 | RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:3921,5477 |
| `-5.2985` | 2700 | RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:3922,5478 |
| `-3.3374` | 2700 | RESULT_V8F_LAUNCH_INTENT.md:260; LAUNCH_INTENT_READ.json:3923,5479 |
| `-9.9630` | 2701 | RESULT_V8F_LAUNCH_INTENT.md:261 |
| `-12.0882` | 2701 | RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:2860,5128 |
| `-7.9003` | 2701 | RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:2861,5129 |
| `-9.4896` | 2701 | RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:3935,5497 |
| `-11.4905` | 2701 | RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:3936,5498 |
| `-7.6284` | 2701 | RESULT_V8F_LAUNCH_INTENT.md:261; LAUNCH_INTENT_READ.json:3937,5499 |
| `-0.5233` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:2903,5529 |
| `-0.6966` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:2904,5530 |
| `-0.3707` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:2905,5531 |
| `-0.7030` | 2709 | RESULT_V8F_DAG_ORC.md:96; RESULT_V8F_LAUNCH_INTENT.md:269 |
| `-0.9087` | 2709 | RESULT_V8F_DAG_ORC.md:96; SPEEDSUB_T1_READ.json:2847; DAGORC_READ.json:3016,3554; RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:1828,5536 |
| `-0.5230` | 2709 | RESULT_V8F_DAG_ORC.md:96; RESULT_V8F_LAUNCH_INTENT.md:269 |
| `-0.7336` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:3979,5899 |
| `-0.9525` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:3980,5900 |
| `-0.5426` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:3981,5901 |
| `-0.6592` | 2709 | SPEEDSUB_T1_READ.json:3270; RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:2365,5905 |
| `-0.8610` | 2709 | RESULT_V8F_LAUNCH_INTENT.md:269 |
| `-0.4801` | 2709 | SPEEDSUB_T1_READ.json:3272; RESULT_V8F_LAUNCH_INTENT.md:269; LAUNCH_INTENT_READ.json:2367,5907 |
| `-0.5819` | 2710 | RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:2917,5549 |
| `-0.7575` | 2710 | RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:2918,5550 |
| `-0.4109` | 2710 | RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:2919,5551 |
| `-0.5876` | 2710 | RESULT_V8F_DAG_ORC.md:97; SPEEDSUB_T1_READ.json:2856; DAGORC_READ.json:3029,3567; RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:1841,5555 |
| `-0.7702` | 2710 | RESULT_V8F_DAG_ORC.md:97; SPEEDSUB_T1_READ.json:2857; DAGORC_READ.json:3030,3568; RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:1842,5556 |
| `-0.4119` | 2710 | RESULT_V8F_DAG_ORC.md:97; SPEEDSUB_T1_READ.json:2858; DAGORC_READ.json:3031,3569; RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:1843,5557 |
| `-0.6009` | 2710 | RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:3993,5919 |
| `-0.7823` | 2710 | RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:3994,5920 |
| `-0.4214` | 2710 | RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:3995,5921 |
| `-0.5758` | 2710 | SPEEDSUB_T1_READ.json:3280; RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:2379,5925 |
| `-0.7559` | 2710 | SPEEDSUB_T1_READ.json:3281; RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:2380,5926 |
| `-0.4022` | 2710 | SPEEDSUB_T1_READ.json:3282; RESULT_V8F_LAUNCH_INTENT.md:270; LAUNCH_INTENT_READ.json:2381,5927 |
| `+0.1377` | 2711 | RESULT_V8F_LAUNCH_INTENT.md:271 |
| `-0.2067` | 2711 | RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:2932,5570 |
| `+0.5134` | 2711 | RESULT_V8F_LAUNCH_INTENT.md:271 |
| `-0.6014` | 2711 | RESULT_V8F_DAG_ORC.md:98; SPEEDSUB_T1_READ.json:2866; DAGORC_READ.json:3043,3581; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:1855,5575 |
| `-0.9063` | 2711 | RESULT_V8F_DAG_ORC.md:98; SPEEDSUB_T1_READ.json:2867; DAGORC_READ.json:3044,3582; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:1856,5576 |
| `-0.3345` | 2711 | RESULT_V8F_DAG_ORC.md:98; SPEEDSUB_T1_READ.json:2868; DAGORC_READ.json:3045,3583; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:1857,5577 |
| `-0.6569` | 2711 | RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:4007,5939 |
| `-1.0141` | 2711 | RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:4008,5940 |
| `-0.3371` | 2711 | RESULT_V8F_SPEEDSUB_T1.md:258; SPEEDSUB_T1_READ.json:3452; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:4009,5941 |
| `-0.4009` | 2711 | SPEEDSUB_T1_READ.json:3290; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:2393,5945 |
| `-0.7081` | 2711 | SPEEDSUB_T1_READ.json:3291; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:2394,5946 |
| `-0.1184` | 2711 | SPEEDSUB_T1_READ.json:3292; RESULT_V8F_LAUNCH_INTENT.md:271; LAUNCH_INTENT_READ.json:2395,5947 |
| `+0.0162` | 2712 | RESULT_V8F_LAUNCH_INTENT.md:272 |
| `-0.0243` | 2712 | RESULT_V8F_SPEEDSUB_T1.md:152; RESULT_V8F_DAG_ORC.md:73,99; SPEEDSUB_T1_READ.json:2877,3515; DAGORC_READ.json:638,3058,3596; RF_REP.json:51; RESULT_V8F_LAUNCH_INTENT.md:272; LAUNCH_INTENT_READ.json:1064,1870,5596; MLAUNCH.json:260,1102,1103; MLAUNCH NOTES.md:14 |
| `+0.0254` | 2712 | RESULT_V8F_LAUNCH_INTENT.md:272 |
| `-0.4007` | 2713 | RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:2959,5609 |
| `-0.7827` | 2713 | RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:2960,5610 |
| `-0.4466` | 2713 | RESULT_V8F_DAG_ORC.md:100; SPEEDSUB_T1_READ.json:2886; DAGORC_READ.json:3071,3609; RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:1883,5615 |
| `-0.8200` | 2713 | RESULT_V8F_DAG_ORC.md:100; RESULT_V8F_LAUNCH_INTENT.md:273 |
| `-0.1686` | 2713 | RESULT_V8F_DAG_ORC.md:100; SPEEDSUB_T1_READ.json:2888; DAGORC_READ.json:3073,3611; RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:1885,5617 |
| `-0.5136` | 2713 | RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:4035,5979 |
| `-0.8890` | 2713 | RESULT_V8F_LAUNCH_INTENT.md:273 |
| `-0.1944` | 2713 | RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:4037,5981 |
| `-0.4713` | 2713 | SPEEDSUB_T1_READ.json:3310; RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:2421,5985; MLAUNCH.json:181 |
| `-0.8540` | 2713 | RESULT_V8F_LAUNCH_INTENT.md:273 |
| `-0.1546` | 2713 | SPEEDSUB_T1_READ.json:3312; RESULT_V8F_LAUNCH_INTENT.md:273; LAUNCH_INTENT_READ.json:2423,5987 |
| `-0.0817` | 2714 | RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:2973,5629 |
| `-0.1396` | 2714 | RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:2974,5630 |
| `-0.0318` | 2714 | RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:2975,5631; MLAUNCH.json:191 |
| `-0.0724` | 2714 | RESULT_V8F_DAG_ORC.md:101; SPEEDSUB_T1_READ.json:2896; DAGORC_READ.json:3085,3623; RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:1897,5635 |
| `-0.1384` | 2714 | RESULT_V8F_DAG_ORC.md:83,101; SPEEDSUB_T1_READ.json:2897; DAGORC_READ.json:778,3086,3624; RF_REP.json:111; RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:1204,1898,5636 |
| `-0.0085` | 2714 | RESULT_V8F_SPEEDSUB_T1.md:154; RESULT_V8F_DAG_ORC.md:101; SPEEDSUB_T1_READ.json:2898,4171; DAGORC_READ.json:3087,3625; RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:1899,5637 |
| `-0.1051` | 2714 | RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:4049,5999 |
| `-0.1677` | 2714 | RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:4050,6000 |
| `-0.0483` | 2714,2761 | RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:4051,6001; MLAUNCH.json:1720; MLAUNCH NOTES.md:72 |
| `-0.0951` | 2714 | SPEEDSUB_T1_READ.json:3320; RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:2435,6005 |
| `-0.1535` | 2714 | SPEEDSUB_T1_READ.json:3321; RESULT_V8F_LAUNCH_INTENT.md:274; LAUNCH_INTENT_READ.json:2436,6006 |
| `-1.9501` | 2715 | RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:2987,5656 |
| `-2.6594` | 2715 | RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:2988,5657 |
| `-1.2479` | 2715 | RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:2989,5658 |
| `-2.0539` | 2715 | RESULT_V8F_DAG_ORC.md:102; SPEEDSUB_T1_READ.json:2906; DAGORC_READ.json:3099,3637; RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:1911,5662 |
| `-2.7092` | 2715 | RESULT_V8F_DAG_ORC.md:102; SPEEDSUB_T1_READ.json:2907; DAGORC_READ.json:3100,3638; RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:1912,5663 |
| `-1.3705` | 2715 | RESULT_V8F_DAG_ORC.md:102; SPEEDSUB_T1_READ.json:2908; DAGORC_READ.json:3101,3639; RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:1913,5664 |
| `-1.9614` | 2715 | RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:4063,6026 |
| `-2.6722` | 2715 | RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:4064,6027 |
| `-1.2710` | 2715 | RESULT_V8F_LAUNCH_INTENT.md:275 |
| `-1.7865` | 2715 | SPEEDSUB_T1_READ.json:3330; RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:2449,6032 |
| `-2.5844` | 2715 | SPEEDSUB_T1_READ.json:3331; RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:2450,6033 |
| `-1.0282` | 2715 | SPEEDSUB_T1_READ.json:3332; RESULT_V8F_LAUNCH_INTENT.md:275; LAUNCH_INTENT_READ.json:2451,6034 |
| `-0.9021` | 2718 | RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:3029,5716 |
| `-1.0493` | 2718 | RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:3030,5717 |
| `-0.7638` | 2718 | RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:3031,5718 |
| `-0.9742` | 2718 | RESULT_V8F_DAG_ORC.md:105; SPEEDSUB_T1_READ.json:2936; DAGORC_READ.json:3141,3679; RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:1953,5722 |
| `-1.1152` | 2718 | RESULT_V8F_DAG_ORC.md:105; SPEEDSUB_T1_READ.json:2937; DAGORC_READ.json:3142,3680; RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:1954,5723 |
| `-0.8405` | 2718 | RESULT_V8F_DAG_ORC.md:105; SPEEDSUB_T1_READ.json:2938; DAGORC_READ.json:3143,3681; RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:1955,5724 |
| `-0.9788` | 2718 | RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:4105,6086 |
| `-1.1217` | 2718 | RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:4106,6087 |
| `-0.8478` | 2718 | RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:4107,6088 |
| `-0.9518` | 2718 | SPEEDSUB_T1_READ.json:3360; RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:2491,6092 |
| `-1.0961` | 2718 | SPEEDSUB_T1_READ.json:3361; RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:2492,6093 |
| `-0.8185` | 2718 | SPEEDSUB_T1_READ.json:3362; RESULT_V8F_LAUNCH_INTENT.md:278; LAUNCH_INTENT_READ.json:2493,6094 |
| `-0.4029` | 2719 | RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:3043,5736 |
| `-1.3012` | 2719 | RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:3044,5737 |
| `+0.4388` | 2719 | RESULT_V8F_LAUNCH_INTENT.md:279 |
| `-1.8473` | 2719 | RESULT_V8F_DAG_ORC.md:106; SPEEDSUB_T1_READ.json:2946; DAGORC_READ.json:3155,3693; RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:1967,5742 |
| `-2.9617` | 2719 | RESULT_V8F_DAG_ORC.md:106; SPEEDSUB_T1_READ.json:2947; DAGORC_READ.json:3156,3694; RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:1968,5743 |
| `-0.8501` | 2719 | RESULT_V8F_DAG_ORC.md:106; SPEEDSUB_T1_READ.json:2948; DAGORC_READ.json:3157,3695; RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:1969,5744 |
| `-1.9161` | 2719 | RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:4119,6106 |
| `-3.1968` | 2719 | RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:4120,6107 |
| `-0.8210` | 2719 | RESULT_V8F_LAUNCH_INTENT.md:279 |
| `-1.5251` | 2719 | SPEEDSUB_T1_READ.json:3370; RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:2505,6112 |
| `-2.5967` | 2719 | SPEEDSUB_T1_READ.json:3371; RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:2506,6113 |
| `-0.5934` | 2719 | SPEEDSUB_T1_READ.json:3372; RESULT_V8F_LAUNCH_INTENT.md:279; LAUNCH_INTENT_READ.json:2507,6114 |
| `+0.1972` | 2721,2721 | RESULT_V8F_DAG_ORC.md:108; RESULT_V8F_LAUNCH_INTENT.md:281,281 |
| `+0.1593` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.2363` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.1600` | 2721 | RESULT_V8F_DAG_ORC.md:108; RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.2364` | 2721 | RESULT_V8F_DAG_ORC.md:108; RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.1926` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.2331` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.1709` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.1297` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.2141` | 2721 | RESULT_V8F_LAUNCH_INTENT.md:281 |
| `+0.1384` | 2722 | RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.0746` | 2722 | RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.2006` | 2722 | RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.1371` | 2722 | RESULT_V8F_DAG_ORC.md:109; RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.0751` | 2722 | RESULT_V8F_DAG_ORC.md:109; RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.2001` | 2722 | RESULT_V8F_DAG_ORC.md:86,109,168; RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.1542` | 2722 | RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.2114` | 2722 | RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.1567` | 2722 | RESULT_V8F_LAUNCH_INTENT.md:282 |
| `+0.2155` | 2722,2736 | RESULT_V8F_LAUNCH_INTENT.md:282,303 |
| `-1.6440` | 2723 | RESULT_V8F_LAUNCH_INTENT.md:283 |
| `-2.0440` | 2723 | RESULT_V8F_LAUNCH_INTENT.md:283 |
| `-1.2864` | 2723 | RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:3101,5831 |
| `-1.9276` | 2723 | RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:4175,6199 |
| `-2.4624` | 2723 | RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:4176,6200 |
| `-1.4663` | 2723 | RESULT_V8F_LAUNCH_INTENT.md:283; LAUNCH_INTENT_READ.json:4177,6201 |
| `-4.7627` | 2724 | RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:3113,5849 |
| `-5.7624` | 2724 | RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:3114,5850 |
| `-3.8251` | 2724 | RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:3115,5851 |
| `-4.8929` | 2724 | RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:4189,6219 |
| `-5.9895` | 2724 | RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:4190,6220 |
| `-3.8792` | 2724 | RESULT_V8F_LAUNCH_INTENT.md:284; LAUNCH_INTENT_READ.json:4191,6221 |
| `-10.2601` | 2725 | RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:3127,5869 |
| `-12.4419` | 2725 | RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:3128,5870 |
| `-8.1915` | 2725 | RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:3129,5871 |
| `-10.4943` | 2725 | RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:4203,6239 |
| `-12.6644` | 2725 | RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:4204,6240 |
| `-8.4336` | 2725 | RESULT_V8F_LAUNCH_INTENT.md:285; LAUNCH_INTENT_READ.json:4205,6241 |
| `0.1627` | 2729 | RESULT_V8F_LAUNCH_INTENT.md:305; check_result_filled.out:117 |
| `+0.0718` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0195` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.1256` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0505` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0012` | 2733 | GOALS_AND_CLAIMS.md (register rows):3; RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0992` | 2733 | REP_S0_SPEED_BIAS.md:24; RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0682` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.1170` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0090` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `-0.0465` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0624` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `-0.0157` | 2733 | P0_RF0.json (rebase):299; RESULT_V8F_LAUNCH_INTENT.md:300 |
| `+0.0979` | 2733 | RESULT_V8F_LAUNCH_INTENT.md:300 |
| `-0.0792` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.1340` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.0274` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301; MLAUNCH.json:1853; MLAUNCH NOTES.md:75 |
| `-0.0873` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.1516` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `+0.0523` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `+0.0069` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `+0.0963` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.0517` | 2734 | RESULT_V8F_SPEEDSUB_T1.md:168; SPEEDSUB_T1_READ.json:4310; RESULT_V8F_LAUNCH_INTENT.md:301 |
| `+0.0066` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.0917` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `-0.1449` | 2734 | RESULT_V8F_LAUNCH_INTENT.md:301 |
| `+0.0901` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.1509` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0842` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0339` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.1378` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `-0.0450` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0670` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0731` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0157` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.1303` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0890` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.0341` | 2735 | RESULT_V8F_LAUNCH_INTENT.md:302 |
| `+0.1627` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.1074` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0344` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0867` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0487` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0009` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0970` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0295` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `-0.0280` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0882` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.0485` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `-0.0046` | 2736 | P0_RF0.json (rebase):351; RESULT_V8F_LAUNCH_INTENT.md:303 |
| `+0.1005` | 2736 | RESULT_V8F_LAUNCH_INTENT.md:303 |
| `-0.5560` | 2751,2787 | GOALS_AND_CLAIMS.md (register rows):6; RESULT_V8F_LAUNCH_INTENT.md:387; MLAUNCH NOTES.md:21 |
| `-0.3975` | 2751 | GOALS_AND_CLAIMS.md (register rows):6; RESULT_V8F_LAUNCH_INTENT.md:387; MLAUNCH.json:874; MLAUNCH NOTES.md:21 |
| `253` | 2760 | MLAUNCH.json:1838; MLAUNCH NOTES.md:68 |
| `255` | 2760 | MLAUNCH.json:1793,1883; MLAUNCH NOTES.md:68 |
| `-0.1200` | 2761 | MLAUNCH NOTES.md:72 |
| `+0.3824` | 2761 | MLAUNCH NOTES.md:73 |
| `+0.1367` | 2761 | MLAUNCH NOTES.md:73 |
| `+0.6489` | 2761 | MLAUNCH NOTES.md:73 |
| `-0.1880` | 2762 | MLAUNCH NOTES.md:74 |
| `-0.3169` | 2762 | MLAUNCH.json:1835; MLAUNCH NOTES.md:74 |
| `-0.0707` | 2762 | MLAUNCH.json:1836; MLAUNCH NOTES.md:74 |
| `-0.0975` | 2762 | MLAUNCH.json:1866; MLAUNCH NOTES.md:75 |
| `-0.2103` | 2762 | MLAUNCH.json:1867; MLAUNCH NOTES.md:75 |
| `-0.0040` | 2762 | MLAUNCH NOTES.md:75 |
| `0.0294` | 2765 | RESULT_V8F_LAUNCH_INTENT.md:355; check_result_filled.out:116 |
| `0.0483` | 2766 | RESULT_V8F_LAUNCH_INTENT.md:355; LAUNCH_INTENT_READ.json:3427; check_result_filled.out:116 |
| `+0.0199` | 2787 | MLAUNCH NOTES.md:21 |
| `0.2484` | 2802 | PREREG_V8F_LAUNCH_INTENT.md:101,348; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:361 |
| `0.2478` | 2802 | PREREG_V8F_LAUNCH_INTENT.md:101,348; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:361 |
| `0.2435` | 2802 | PREREG_V8F_LAUNCH_INTENT.md:101,348; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:361 |
| `0.2456` | 2802 | PREREG_V8F_LAUNCH_INTENT.md:101,348; launch-intent-v2/enumerate_cells_li.out:68; RESULT_V8F_LAUNCH_INTENT.md:361 |
| `133` | 2820 | git log (cited commits):1; check_result_filled.out:136,138 |
| `+0.1434` | 2876,2955 | RESULT_V8F_DAG_ORC.md:17,22,69,232...; GOALS_AND_CLAIMS.md (register rows):3,3,3; git log (cited commits):37 |
| `+0.0538` | 2876,2955 | RESULT_V8F_DAG_ORC.md:17,69,240; GOALS_AND_CLAIMS.md (register rows):3 |
| `+0.2342` | 2876,2955 | RESULT_V8F_DAG_ORC.md:17,69,240; GOALS_AND_CLAIMS.md (register rows):3 |
| `+0.1106` | 2876,2955 | RESULT_V8F_DAG_ORC.md:18,174; GOALS_AND_CLAIMS.md (register rows):3; git log (cited commits):37 |
| `+0.0722` | 2876,2955 | RESULT_V8F_DAG_ORC.md:18,174; GOALS_AND_CLAIMS.md (register rows):3; git log (cited commits):37 |
| `8.8` | 2876,2955 | RESULT_V8F_DAG_ORC.md:121,130,228; GOALS_AND_CLAIMS.md (register rows):3; git log (cited commits):37 |
| `33.8` | 2876,2955 | RESULT_V8F_DAG_ORC.md:124,130,229,231; GOALS_AND_CLAIMS.md (register rows):3; git log (cited commits):37 |
| `162` | 2955 | RESULT_V8F_DAG_ORC.md:3,14; GOALS_AND_CLAIMS.md (register rows):3; git log (cited commits):37 |

## B. Found only with a different sign or none (sign is a property of the sentence, reviewed by hand in E)

| token | v33 lines | unsigned form found at |
|---|---|---|
| `+28.7` | 2557,2558 | RESULT_V8F_LAUNCH_INTENT.md:123 |
| `+73.6` | 2785 | RESULT_V8F_LAUNCH_INTENT.md:98,349 |
| `+158.1` | 2785 | RESULT_V8F_LAUNCH_INTENT.md:95,348 |

## C. Found only in the v32 report (carried context, not a v33 source claim)

| token | v33 lines | v32 lines |
|---|---|---|
| `7.24` | 3,1741,1744 | 40,933,1739,1756,1802 |
| `7.25` | 3 | 41,1190,1293,1642,1740 |
| `7.26` | 3,2055,2173,2200,2236 | 42,1170,1170,1273,1276 |
| `8.1` | 2440,2442 | 1726 |
| `76` | 2526,2785 | 422 |

## D. Found nowhere (must be structural: section numbers, dates, identifiers, derived values)

| token | v33 lines |
|---|---|
| `7.27` | 3,43,1186,1738,1978 |
| `7.28` | 3,44,57,1186,1759 |
| `7.29` | 3,45,1702,1702,2237 |
| `29.3` | 2491 |

## E. Hand review of sections B, C and D

| token | review |
|---|---|
| `+28.7` | sign added in the prose range 'from -30.2 % to +28.7 %' (and 'between -3.0 % and +28.7 %'); the value 28.7 is the share closed in the s1 main curvature cell (RESULT table line 123, 28.7 %); the range and its ends are recomputed from the point deltas in section H |
| `+73.6` | sign added in the prose range 'from -23.4 % to +73.6 %' over the eight main-pair cells; 73.6 is the improvement / bar of the s1 main curvature cell (RESULT table lines 98 and 349, 73.6 %); the range is recomputed in section H |
| `+158.1` | sign added in the prose range 'from -130.9 % to +158.1 %' over all 16 cells; 158.1 is the improvement / bar of the s0 replicate L2 cell (RESULT table lines 95 and 348, 158.1 %); the range is recomputed in section H |
| `7.24` | section reference (the status line of v33 and the 7.24 / 7.26 update sentences say 'section 7.24' for the SPEEDREAD read); structural, a heading number of this report, also present in v32 |
| `7.25` | section reference (the status line names the DAGGER2 section); structural, a heading number of this report, also present in v32 |
| `7.26` | section reference (the DAG-ORC section; the status line, the update sentences and the 7.29 prose point to it); structural, a heading number of this report, also present in v32 |
| `8.1` | section reference (the established-claims list of the Discussion, named by a 7.29 sentence); structural, a heading number of this report, also present in v32 |
| `76` | derived count, 19 rows x 4 pair-seed cells (X - Y), named in the floors and the numbers prose; section H asserts 76 cells with 16 separated-better, 7 separated-worse and 53 not separated. The v32 occurrence is an unrelated 76 |
| `7.27` | section reference (heading number of the SPEEDSUB section, named in the status line, the abstract, the 7.28 and 7.29 prose and the 8.4 rows); structural, found in no source because it is this report's own numbering |
| `7.28` | section reference (heading number of the SPEEDSUB-T1 section, named in the status line, the abstract, the 7.29 prose and the 8.4 rows); structural, found in no source because it is this report's own numbering |
| `7.29` | section reference (heading number of the LAUNCH-INTENT section, named in the title line, the status line, the abstract, the 7.26 update sentence and the 8.x text); structural, found in no source because it is this report's own numbering |
| `29.3` | derived value, the twelve T1-kin step times 1756 s / 60 = 29.3 min; the sum and the quotient are recomputed in section H from the relaunch's chain log |

## F. Claims taken from a PREREG, a note or a log (wording and constants), resolved to source lines

| claim in the report | source:lines |
|---|---|
| 7.27: the question, the selection disclosure, F/2 = 0.06035, the arm `ego`, the shuffle control, tier T1-class open loop | `PREREG_V8F_SPEEDSUB.md`:37,20,3,3 |
| 7.27: the PASS, the 27 control flags, the reader line, the declared channel and its attribution, the rule-violation block, the follow-up | `RESULT_V8F_SPEEDSUB.md`:3,3,8,251,251,47,6,43,259,228 |
| 7.27: the attribution table (TRAIN fold 0: full 0.24751, no nav 0.26313, no limit 0.28032, neither 0.30849) | `FINDING_1_ATTRIBUTION.md`:12,13,14,15 |
| 7.28: design, the hit rule, the guards (G-WORSE, G-NEW, G-ADE, G-VIOL), the outcome order, the causal bucket, the prior | `PREREG_V8F_SPEEDSUB_T1.md`:4,194,210,17,121,38,119 |
| 7.28: the NULL, the 158 control flags, the reader line, the floor rule NOT COLLAPSE, the ranking of the next steps, the single-copy dumps | `RESULT_V8F_SPEEDSUB_T1.md`:3,3,295,295,279,371,372,317,225,243,245 |
| 7.28: the REP s0 diagnosis (all five speed cells slow against CTRL s0; REP s1 and A sit with CTRL; the Master Mind's, written before the read) | `REP_S0_SPEED_BIAS.md`:24,25,26 |
| 7.26 update: the PI's routing, B not trained, A stays HARMS, NEXT-CRAWL = LAUNCH-INTENT | `PREREG_V8F_DAG_ORC.md`:497,502,503,510 |
| 7.29: status, promotion, the claim row, the dose (f = 0.161851, W_A 1.58926, W_B 0.726175, W_HI 4.17852), the lever, the mask, the redesign disclosure | `PREREG_V8F_LAUNCH_INTENT.md`:3,3,81,86,87,5,35,49 |
| 7.29: K-MASK as registered (window-list sha256 equal to DAG-OACC's; launch_n / launch_eligible within 0.02 of f), the controls K-GATE, K-PAIR, K-INIT, K-SHARE, K-BUILD, K-P0, K-RC, K-ZERO, K-REP-LICENCE | `PREREG_V8F_LAUNCH_INTENT.md`:240,240,236,237,238,241,242,243,244,245,90 |
| 7.29: the chain's own K-MASK check also asserts launch_rows == launch_eligible (INHERITED from the chain notes) | `launch-intent-chain/NOTES.md`:19 |
| 7.29: the PREREG's reading choices confirmed at promotion: domain D of the replicate pair, M-LAUNCH moved only in both pairs at both seeds, no exempt rows | `PREREG_V8F_LAUNCH_INTENT.md`:28,217,176,204 |
| 7.29: the launch commit, the pins, the reader's not-built items (M-LAUNCH, anti-echo, M2, T0 families), the chain, the REP licence | `launch-intent-v2/NOTES.md`:30,38,12 |
| 7.29: the chain package (launch commit b09ac89f, trainer 0d708b3c, the self-test totals) and its pins | `launch-intent-chain/PINS_FILLED.txt`:2,2,7,7 |
| 7.29: the launch facts, MEASURED from the banked box-2 chain log: first stamped line 13:54, the predecessor's DONE marker, the refresh, the four gates (launched, ZZJOB-END, PASS), A_0 launched 14:02, the four pulls with run record PASS and K-SHARE PASS, the DONE marker at 16:48 and 'B is not queued' | `chain_v8f_launchintent.out`:1,2,10,12,16,22,28,34,39,55,74,93,112,117,117,5,5 |
| 7.29: the REP provenance: the Master Mind's incident note (repo path), REP s0 and the other pre-05:56 runs, the pinned relaunch at 8c24823f | `R2C_NOTE.md`:34,28,31 |
| 7.29: the REP launch-gate tokens bind commit 8c24823f at both seeds | `REP launch-gate tokens (s0, s1)`:6,52,8 |
| 7.29: the REP run records: trainer seeds 10 and 11 | `REP run records (s0, s1)`:5,158 |
| 7.29: Amendment 1 (appended at e28516e8, sha256 6ae79f2f): K-BUILD's data-module clause, pinned to DAG-ORC r2's build | `PREREG_V8F_LAUNCH_INTENT.md`:352,354,364,375,378 |
| 7.29: the RESULT: word, controls, tier, estimator, run record, disclosures (Amendment 1, RAM-guard stop, pin slip, reader label, single copies, coupling), next, what it cannot claim | `RESULT_V8F_LAUNCH_INTENT.md`:3,5,5,14,131,16,378,387,387,389,416,424,429,430,433,435,436,437,411 |
| 7.29: the RESULT's own checker: 133 checks, 0 failed | `check_result_filled.out`:138,139 |
| 7.29: M-LAUNCH build notes: the readings, the controls, the disclosed redesign of the controls, the post-hoc per-tick look, the beside M2 block | `MLAUNCH NOTES.md`:8,8,64,22,22,77 |
| 7.29: the T1-kin chain logs: the stopped launch (RAM guard), the relaunch, the REFUSED read of the chain | `lit1_logs/chain.out.prev_1028151251`:22 |
| 7.29: the PI's go as recorded in the DAG-ORC PREREG; the redesign note of the Master Mind (bar 2 x max(F, RF0), four same-seed pairs) | `launch-intent-flag/MM_REDESIGN_2026-10-28.md`:15,1 |
| Register rows: H-V8F-SPEEDSUB-1, H-V8F-SPEEDSUB-T1-1 and H-V8F-LAUNCH-INTENT-1 exist and carry the words the report uses | `GOALS_AND_CLAIMS.md (register rows)`:4,5,6,1 |
| Commits named in the text exist and carry the subject word (94a47f4b SPEEDSUB, 842ec2ac PREREG_V8F_SPEEDSUB_T1, af2dee4a RESULT_V8F_SPEEDSUB, 4cec09e7 Amendment 1, 37d78ed1 RESULT_V8F_SPEEDSUB_T1, 051259d4 Amendment 3, b09ac89f the flag, 77234276 the LAUNCH-INTENT PREREG, d8c9f7bf the queue stage) | `git log (cited commits)`:40,34,31,28,22,25,16,19,13,7 |

The launch facts of 7.29 are MEASURED from the banked box-2 chain log (`chain_v8f_launchintent.out`): the first stamped line 13:54, the refresh, the four gates (launched 13:57 / 13:58 / 14:00 / 14:01, ZZJOB-END 13:58:34 / 14:00:02 / 14:01:33 / 14:02:37, each PASS), A_0 launched 14:02, the pulls at 14:44 / 15:25 / 16:06 / 16:48 and the DONE marker at 16:48 (section H recomputes the 174 min). Nothing outside the repository is read by this audit. The REP provenance is `R2C_NOTE.md` (a Master Mind note in the repo, tagged INHERITED in the text) plus the two REP launch-gate tokens, which bind commit `8c24823f` (MEASURED).
⚠ **One source inconsistency, disclosed in the report and in `RESULT_V8F_LAUNCH_INTENT.md` section 8:** `PINS_FILLED.txt` in the chain package is stamped about 14:10, while the pins were filled about 13:50, before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00). The report quotes the stamp only with that disclosure; section J re-measures the commit times, the two pinned hashes and the log-derived times.

## G. Section-scoped check (a number in the prose of 7.27 / 7.28 / 7.29 must occur in that section's own sources)

Section 7.27 is checked against the SPEEDSUB result, PREREG, read JSON and the attribution file (and the T1 files it points forward to); 7.28 against the SPEEDSUB-T1 result, PREREG, read JSON, beside JSON, the REP s0 diagnosis and the register rows; 7.29 against the LAUNCH-INTENT PREREG, the DAG-ORC PREREG and result and read JSON, the LAUNCH-INTENT result, read JSON, M-LAUNCH JSON and notes, the box-2 chain log, the T1-kin chain logs, the REP tokens and run records, R2C_NOTE.md, the build, redesign, reader and chain notes and the commit messages. Table lines are verbatim copies and are not re-tested here (section I). Section numbers are structural, and a section-scoped match for them is not evidence. 16 tokens are not found in their own section's sources; each is reviewed:

| section:token | review |
|---|---|
| `7.27:7.27` | the section's own heading number; structural |
| `7.27:7.24` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.27:7.28` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.28:7.28` | the section's own heading number; structural |
| `7.28:7.27` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.28:7.26` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.28:7.29` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.29:7.29` | the section's own heading number; structural |
| `7.29:7.26` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.29:7.28` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.29:8.1` | section reference to another section of this report (a heading number), structural; the target section exists in this report (checked by checks_v33.py) |
| `7.29:29.3` | derived: the twelve T1-kin step times sum to 1756 s, 1756 / 60 = 29.3 min; recomputed in section H from the relaunch's chain log (the sources print the 12 step times, not the quotient) |
| `7.29:76` | derived: 19 rows x 4 pair-seed cells; section H asserts the 76 cells and their 16 / 7 / 53 split from the read JSON |
| `7.29:+28.7` | the sign of the upper end of the prose range of the share of the Y - P10 gap closed; the unsigned value 28.7 is in the RESULT's table; the range is recomputed in section H |
| `7.29:+73.6` | the sign of the upper end of the prose range of improvement / bar over the eight main-pair cells; the unsigned value is in the RESULT's table; recomputed in section H |
| `7.29:+158.1` | the sign of the upper end of the prose range of improvement / bar over all 16 cells; the unsigned value is in the RESULT's table; recomputed in section H |

## H. JSON cross-check of the decisive and the derived numbers

`SUBREAD.json`, `SPEEDSUB_T1_READ.json`, `SPEEDSUB_T1_BESIDE.json`, `DAGORC_READ.json`, `RF_REP.json`, `P0_RF0.json`, `f_measure.json`, `LAUNCH_INTENT_READ.json`, `MLAUNCH.json` and the banked box-2 chain log `chain_v8f_launchintent.out` (with the T1-kin chain log for the twelve steps) are read from the tip. Each row below is recomputed from the JSON or the log by `audit_v33.py` (an assertion fails the script on a mismatch) and the quoted snippet is found in the named part of v33. Derived values (shares, ratios, counts, sums, the dose, the box-2 span) are computed here, not copied. For 7.29 every table line that carries a number of the read (the predicate table, the 76 cells, the crawl, guard, R-NEW, R-T0 and family tables, the 19-row floors table, the slow-draw table, the M-LAUNCH cells, the run record, the gates, the box-2 and dev-box times) is rebuilt from the JSON or log and must equal the printed line; 160 such lines were rebuilt and matched (the rows marked as tables below summarise them).

| what | JSON key path | value in the JSON | snippet in v33 | found in |
|---|---|---|---|---|
| outcome PASS, F/2 = 0.06035, mlp_ego, readout sha a205609a, 27 control flags all true | `outcome; F_pass_m; selected_readout; readout_sha256; controls (counted)` | PASS; 0.06035; mlp_ego; a205609a…; 27 true | `all **27 control flags PASS**` | 7.27 |
| 27 control flags in the 8.4 row | `controls (counted)` | 27 | `all 27 control flags PASS` | row 1z |
| ADE6 ego - own, s0: delta and 95 % CI | `rows["0"]["ego"].{delta, lo, hi}` | −0.1290 [−0.2107, −0.0548] | `−0.1290 [−0.2107, −0.0548]` | 7.27, abstract, 8.1, 8.3, row 1z |
| ADE6 egoshuf - own (deliberate-regression arm), s0: separated-worse | `rows["0"]["egoshuf"].delta` | +6.7181 | `+6.7181` | 7.27, abstract, 8.1, row 1z |
| ADE6 gt - own (T0 ceiling), s0 | `rows["0"]["gt"].delta` | −0.7164 | `−0.7164` | 7.27 |
| CAPTURE = ego / gt, s0 (4 dp) | `rows["0"].capture; ego.delta / gt.delta` | 0.1801 | `0.1801` | 7.27 |
| ADE6 ego - own, s1: delta and 95 % CI | `rows["1"]["ego"].{delta, lo, hi}` | −0.0686 [−0.1151, −0.0232] | `−0.0686 [−0.1151, −0.0232]` | 7.27, abstract, 8.1, 8.3, row 1z |
| ADE6 egoshuf - own (deliberate-regression arm), s1: separated-worse | `rows["1"]["egoshuf"].delta` | +6.4923 | `+6.4923` | 7.27, abstract, 8.1, row 1z |
| ADE6 gt - own (T0 ceiling), s1 | `rows["1"]["gt"].delta` | −0.6676 | `−0.6676` | 7.27 |
| CAPTURE = ego / gt, s1 (4 dp) | `rows["1"].capture; ego.delta / gt.delta` | 0.1028 | `0.1028` | 7.27 |
| CAPTURE as a share of the GT cells' ceiling: 18 % and 10 % | `rows[<seed>].capture` | 18 % and 10 % | `18 % and 10 %` | 7.27 |
| s1 clears the bar by 0.0083 m (0.0686 - 0.06035 = 0.00825, half up) | `rows["1"]["ego"].delta; F_pass_m` | 0.0083 | `clears the bar by 0.0083 m only` | 7.27 |
| the two seeds' effects differ by 0.0604 m | `rows[<seed>]["ego"].delta` | 0.0604 | `differ by 0.0604 m` | 7.27 |
| the upper ends of the two intervals (-0.0548, -0.0232) lie above -0.06035 | `rows[<seed>]["ego"].hi; F_pass_m` | -0.0548, -0.0232 > -0.06035 | `(−0.0548, −0.0232) lies above −0.06035` | 7.27 |
| NULL, NULL-ONE, 1 hit of 4 (REP s0), 158 controls true, no guard fires, bar 0.06035 | `OUTCOME; label; flags.hits; controls (158); guards.*.fires; bar` | NULL; NULL-ONE; REP_s0 only; 158; none; 0.06035 | `**all 158 control flags PASS**` | 7.28 |
| 158 control flags in the 8.4 row | `controls (counted)` | 158 | `all 158 control flags PASS` | row 1aa |
| NULL-ONE in the abstract, 8.1, 8.3, the 8.4 row | `label` | NULL-ONE | `NULL-ONE` | abstract, 8.1, 8.3, row 1aa |
| ADE6 egoc - identity, CTRL s0: delta and 95 % CI | `d_ADE6_egoc_minus_identity.CTRL_s0.{delta, lo, hi}` | −0.0233 [−0.0629, +0.0156] | `−0.0233 [−0.0629, +0.0156]` | 7.28, abstract |
| ADE6 egoc - identity, CTRL s1: delta and 95 % CI | `d_ADE6_egoc_minus_identity.CTRL_s1.{delta, lo, hi}` | −0.0076 [−0.0500, +0.0349] | `−0.0076 [−0.0500, +0.0349]` | 7.28, abstract |
| ADE6 egoc - identity, REP s0: delta and 95 % CI | `d_ADE6_egoc_minus_identity.REP_s0.{delta, lo, hi}` | −0.1949 [−0.2619, −0.1313] | `−0.1949 [−0.2619, −0.1313]` | 7.28, abstract |
| ADE6 egoc - identity, REP s1: delta and 95 % CI | `d_ADE6_egoc_minus_identity.REP_s1.{delta, lo, hi}` | −0.0333 [−0.0832, +0.0131] | `−0.0333 [−0.0832, +0.0131]` | 7.28, abstract |
| ADE6 egoc - identity, CTRL_s0, as a point alone (8.1 item 16, 8.4 row 1aa) | `d_ADE6_egoc_minus_identity.CTRL_s0.delta` | −0.0233 | `−0.0233` | 8.1, row 1aa |
| ADE6 egoc - identity, CTRL_s1, as a point alone (8.1 item 16, 8.4 row 1aa) | `d_ADE6_egoc_minus_identity.CTRL_s1.delta` | −0.0076 | `−0.0076` | 8.1, row 1aa |
| ADE6 egoc - identity, REP_s1, as a point alone (8.1 item 16, 8.4 row 1aa) | `d_ADE6_egoc_minus_identity.REP_s1.delta` | −0.0333 | `−0.0333` | 8.1, row 1aa |
| the one hit as a multiple of the bar: 3.23 x | `d_ADE6_egoc_minus_identity.REP_s0.delta / bar` | 3.23 | `3.23 times the bar` | 7.28 |
| floor rule at s0: NOT COLLAPSE; ADE6 gap 0.4025 -> 0.2309 m; drop 42.6 % | `floor_rule_H_V8F_SPEEDSUB_T1_2[s0].{verdict, g_id, g_c}` | NOT COLLAPSE; 0.4025; 0.2309; 42.6 % | `falls from 0.4025 to 0.2309 m (a drop of 42.6 %)` | 7.28 |
| floor rule in the 8.4 row | `floor_rule_H_V8F_SPEEDSUB_T1_2` | 0.4025 -> 0.2309 | `ADE6 gap 0.4025 → 0.2309 m` | row 1aa |
| REP s0: 8 of 18 rows separated-better, 2 separated-worse; CTRL s0 and REP s1: none; CTRL s1: 3 separated-worse | `tables[egoc-identity].<run>.<row>.{delta, separated}` | {'CTRL_s0': (0, 0), 'CTRL_s1': (0, 3), 'REP_s0': (8, 2), 'REP_s1': (0, 0)} | `Of the 18 computed rows, 8 are separated-better and 2` | 7.28 |
| CTRL s0 and REP s1 move no row of 18; CTRL s1 moves the wrong way on three rows | `tables[egoc-identity].<run>.<row>.separated` | 0 / 0 / 3 worse | `CTRL s0 and REP s1 move no row of 18` | 7.28 |
| VIOL-ANY, REP_s0, egoc - identity | `tables[egoc-identity].REP_s0[VIOL-ANY].{delta, lo, hi}` | +0.0421 [+0.0246, +0.0612] | `+0.0421 [+0.0246, +0.0612]` | 7.28 |
| M1 crawl share, REP_s0, egoc - identity | `tables[egoc-identity].REP_s0[M1 crawl share].{delta, lo, hi}` | −0.0052 [−0.0115, +0.0007] | `−0.0052 [−0.0115, +0.0007]` | 7.28 |
| accel MAE, CTRL_s1, egoc - identity | `tables[egoc-identity].CTRL_s1[accel MAE].{delta, lo, hi}` | +0.1598 [+0.0743, +0.2611] | `+0.1598 [+0.0743, +0.2611]` | 7.28 |
| yaw-rate MAE, CTRL_s1, egoc - identity | `tables[egoc-identity].CTRL_s1[yaw-rate MAE].{delta, lo, hi}` | +1.3443 [+0.2401, +2.5747] | `+1.3443 [+0.2401, +2.5747]` | 7.28 |
| L1 lateral-accel MAE, CTRL_s1, egoc - identity | `tables[egoc-identity].CTRL_s1[L1 lateral-accel MAE].{delta, lo, hi}` | +0.3627 [+0.1230, +0.6572] | `+0.3627 [+0.1230, +0.6572]` | 7.28 |
| M1 crawl share is not separated at any of the four runs | `tables[egoc-identity].<run>[M1 crawl share].separated` | false x4 | `M1 crawl share moves at no run` | 7.28 |
| M1 at CTRL s0 and s1: -0.0007 and +0.0021 | `tables[egoc-identity].CTRL_s*[M1 crawl share].delta` | −0.0007 and +0.0021 | `−0.0007 and +0.0021` | 7.28 |
| ADE6 egoshufc - identity (deliberate-regression arm), CTRL s0, separated-worse | `tables[egoshufc-identity].CTRL_s0[ADE6].{delta, lo, hi}` | +4.2928 [+3.4386, +5.1945] | `+4.2928 [+3.4386, +5.1945]` | 7.28 |
| ADE6 egoshufc - identity (deliberate-regression arm), CTRL s1, separated-worse | `tables[egoshufc-identity].CTRL_s1[ADE6].{delta, lo, hi}` | +3.9857 [+3.2389, +4.7695] | `+3.9857 [+3.2389, +4.7695]` | 7.28 |
| ADE6 egoshufc - identity (deliberate-regression arm), REP s0, separated-worse | `tables[egoshufc-identity].REP_s0[ADE6].{delta, lo, hi}` | +4.4851 [+3.5747, +5.4322] | `+4.4851 [+3.5747, +5.4322]` | 7.28 |
| ADE6 egoshufc - identity (deliberate-regression arm), REP s1, separated-worse | `tables[egoshufc-identity].REP_s1[ADE6].{delta, lo, hi}` | +4.1886 [+3.3996, +5.0549] | `+4.1886 [+3.3996, +5.0549]` | 7.28 |
| the logged-lane bridge, ADE6 at CTRL s0 and s1: +0.0008 and -0.0185, not separated | `SPEEDSUB_T1_BESIDE.egol_minus_identity_l.CTRL_s*.ADE6.{delta, separated}` | +0.0008 and −0.0185 | `+0.0008 and −0.0185 m` | 7.28 |
| the bridge in the abstract and in 8.1 item 16 | `SPEEDSUB_T1_BESIDE.egol_minus_identity_l.CTRL_s*.ADE6.delta` | +0.0008 / -0.0185 | `+0.0008 / −0.0185 m` | abstract, 8.1 |
| share of the identity - P10 crawl gap that egoc closes at CTRL s0 / s1: at most 12.8 %; widens it on 3 of 8 cells | `tables[egoc-P10], tables[identity-P10] ([crawl rows], CTRL_s0 / s1).delta` | max 12.8 %; negative on 3 of 8 | `closes at most 12.8 % of the gap on any crawl row and widens it on 3 of 8 cells` | 7.28 |
| the gap table of 7.29: v1 bar, CTRL - P10 at s0 / s1, v1 bar / gap, v2 bar, v2 bar / gap, for the four crawl rows, recomputed cell by cell (an assertion fails the script on a mismatch) | `floors.F, P0_RF0.rows[r].RF0, RF_REP.RF[r], read_a.CTRL_P10.<seed>[r].delta` | min 1.3, max 9.1; every v2 bar below both gaps | `/ **1.1070** / 0.32 / 0.13 /` | 7.29 |
| v1 bar / gap ranges from 1.3 to 9.1 (min and max over the 8 crawl cells) | `(the same recomputation)` | 1.3 to 9.1 | `1.3 to 9.1 times` | 7.29 |
| the 19-row bar table of 7.29: F, RF0, v2 bar = 2 x max(F, RF0), RF, v1 bar = 2 x max(F, RF0, RF), recomputed row by row | `floors.F, P0_RF0.rows[r].RF0, RF_REP.RF[r]` | 19 rows equal | `/ ADE6 / 0.1207 / 0.0138 / **0.2414** / 0.4025 / 0.8050 /` | 7.29 |
| REP s0 sets RF (the larger /REP - CTRL/ of the two seeds) on 17 of 19 rows | `read_a.REP_CTRL.<seed>[<row>].delta` | 17 of 19 | `set RF on 17 of` | 7.29 |
| f = 49,618 / 306,566 = 0.161851 (the GT-launch share of the TRAIN windows) | `f_measure.json.{n_launch, n_windows_all, f}` | 49618 / 306566 = 0.161851 | `f = 49,618 / 306,566 = 0.161851` | 7.29 |
| W_A = (1 - f) / (2 f) - 1 = 1.58926 (launch samples hold 1/3 of the along-track weight) | `f_measure.json.f` | 1.58926 | `W_A = 1.58926` | 7.29 |
| W_B = (1 - f) / (3 f) - 1 = 0.726175 (1/4) | `f_measure.json.f` | 0.726175 | `W_B = 0.726175` | 7.29 |
| W_HI = (1 - f) / f - 1 = 4.17852 (1/2) | `f_measure.json.f` | 4.17852 | `W_HI = 4.17852` | 7.29 |
| W = 1.58926 in the abstract and in the 8.4 row 1ab | `f_measure.json.f` | 1.58926 | `1.58926` | abstract, row 1ab |
| the outcome table covers 524,288 = 2^19 cells: 262,144 + 261,888 + 1 + 15 + 3 + 45 + 128 + 64 | `(arithmetic on the PREREG's counts)` | 524288 | `524,288 cells` | 7.29 |
| HARMS, arm A not adopted, 206 controls all true | `OUTCOME.OUTCOME; OUTCOME.adopts; controls_PASS (counted)` | HARMS; False; 206 true | `all **206 reader controls PASS**` | 7.29 |
| the control groups and their counts (sum 206) | `controls_PASS keys, grouped by the text before ':'` | 35 groups, sum 206 | `STATIC 1, K-P0 1, K-BUILD 1, K-INIT 8, K-REGISTERED 8, K-PAIR 6, K-SHARE 8, K-MASK 4, K-REP-LICENCE 2, SUMMARY 8, K-GATE 8, K1 8, K2 8, K3 8, K5 8, K6 8, K7 8, GRID 8, T1-CONFIG 8, K2-logged 4, K3-logged 4, GRID-logged 4, T1-CONFIG-logged 4, K4 4, T0F-CKPT 4, T0-CKPT 8, DUMP-WINDOWS 8, GRID-COUNTS 8, R2-ADE6 8, C4-WINDOWS 6, C3-FLOOR 6, K-ZERO 10, ADE-LINK 4, T0-WINDOWS 4, K1-LI-ROWS 1` | 7.29 |
| the JSON condition verbatim | `OUTCOME.condition` | controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) ... | `controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed` | 7.29 |
| committed next verbatim | `OUTCOME.committed_next` | not adopted; read B under this table (train B first, with it... | `not adopted; read B under this table (train B first, with its own PI go).` | 7.29 |
| b_status verbatim | `cascade.b_status` | PENDING: train B (4 runs, share 1/4) through its gate, with ... | `PENDING: train B (4 runs, share 1/4) through its gate, with its own PI go, and read it under this same table` | 7.29 |
| reader fb3d2d7d, RF0 1608d70b, manifest 98d2cd3d, wall 341.6 s | `provenance.{reader_sha256, rf0_sha256, manifest_sha256, wall_s}` | fb3d2d7d; 1608d70b; 98d2cd3d; 341.6 | `Wall 341.6 s` | 7.29 |
| the four ADE6 points lie between 0.0294 and 0.0483 m | `read_a.{A_CTRL, Arep_REP}.<seed>[ADE6].delta` | [('s0 main', -0.0294), ('s0 rep', 0.0431), ('s1 main', 0.0483), ('s1 rep', -0.0341)] | `the four points lie between 0.0294 and 0.0483 m` | 7.29 |
| 2.7 the per-seed predicate table (9 lines: controls, R-TLAT, R-TLON, R-ADE, R-T0, R-NEW, R-MOVE, R-CRAWL, R-KEEP, R-NOBASE) rebuilt from rows.A.<seed>.pairs.<pair>.{guards, flags, domain, R-NEW_fires, R-NOBASE_violations, R-MOVE_rows, R-CRAWL_rows, T0_point, KEEP_point} | `rows.A` | 10 of 10 lines equal; R-ADE fails at s0 (replicate pair) and at s1 (main pair) | `/ R-ADE (T1-kin ADE6, X − Y, not separated-worse) / held: -0.0294 B / FAILS: +0.0431 W /` | 7.29 |
| 2.6 R-T0 and R-KEEP: the 4 points, all at most +0.0604 | `rows.A.<seed>.pairs.<pair>.{T0_point, KEEP_point}` | 4 of 4 lines equal | `/ s0 / main (A − CTRL) / +0.0021 / held / -0.0294 / true /` | 7.29 |
| 2.2 ADE6: the 4 lines (X − Y, means, X − P10, Y − P10) rebuilt from read_a.{A_CTRL, Arep_REP, A_P10, CTRL_P10, Arep_P10, REP_P10}.<seed>[ADE6] | `read_a.<pair>.<seed>.ADE6` | 4 of 4 lines equal | `/ s0 / main (A − CTRL) / ADE6 /` | 7.29 |
| 2.3 R-MOVE: the 16 crawl lines (X − Y, improvement = -delta, bar, improvement / bar, clears) rebuilt cell by cell | `read_a.<pair>.<seed>[crawl row]; floors.bar` | 16 of 16 lines equal | `/ s0 / replicate (A_rep − REP) / L2 curvature MAE (GT>=5.2) /` | 7.29 |
| 2.4 R-CRAWL: the 16 lines (X − P10, Y − P10, share of the gap closed from the point deltas) rebuilt cell by cell | `read_a.{A_P10, CTRL_P10, Arep_P10, REP_P10}.<seed>[crawl row]` | 16 of 16 lines equal | `/ s1 / main (A − CTRL) / curvature MAE /` | 7.29 |
| 2 of the 16 crawl cells separated-better, 1 clears its bar | `read_a.<pair>.<seed>[crawl rows]` | 2; 1 | `2 of the 16 cells are separated-better and 1 clears its bar` | 7.29 |
| share of the Y - P10 gap closed: 9 positive, 7 negative | `(Y - P10 - (X - P10)) / (Y - P10) from the point deltas` | 9 / 7 | `positive in 9 of the 16 cells and negative in 7` | 7.29 |
| share range over all 16 | `(the same)` | -30.2 to 28.7 | `from −30.2 % to +28.7 %` | 7.29 |
| main-pair share range | `(the same)` | -3.0 to 28.7 | `lies between −3.0 % and +28.7 %` | 7.29 |
| improvement / bar range over the 8 main-pair cells and over all 16 | `(2.3 recomputed)` | -23.4 to +73.6; -130.9 to +158.1 | `from −23.4 % to +73.6 % over the eight main-pair cells, from −130.9 % to +158.1 % over all 16` | 7.29 |
| the main-pair partner of the one clearing cell: L2 at s0 main, 55.1 % of the bar | `read_a.A_CTRL.s0[L2 curvature MAE (GT>=5.2)]` | −0.0075 [−0.0212, +0.0025]; 55.1 % | `reads −0.0075 [−0.0212, +0.0025], not separated, 55.1 %` | 7.29 |
| 2.5 R-NEW / R-NOBASE: domain D has 15 rows in all four pairs; the X − Y separated-worse rows, those worsening by at least the bar, and those whose X − P10 is separated-worse, rebuilt per pair | `rows.A.<seed>.pairs.<pair>.{domain, R-NEW_fires, R-NOBASE_violations}; read_a` | 4 of 4 lines equal; no row fires, no row violates | `/ s0 / main (A − CTRL) / CTRL − P10 not separated-worse on D / 15 /` | 7.29 |
| R-NEW condition (3) met, condition (1) fails: speed MAE, s1 main | `read_a.A_CTRL.s1[speed MAE]; read_a.A_P10.s1[speed MAE]` | +0.1797 [+0.0998, +0.2726] | `speed MAE +0.1797 [+0.0998, +0.2726]` | 7.29 |
| R-NEW condition (3) met, condition (1) fails: accel MAE, s1 main | `read_a.A_CTRL.s1[accel MAE]; read_a.A_P10.s1[accel MAE]` | +0.7391 [+0.4366, +1.0952] | `accel MAE +0.7391 [+0.4366, +1.0952]` | 7.29 |
| R-NEW condition (3) met, condition (1) fails: cross-track MAE, s1 main | `read_a.A_CTRL.s1[cross-track MAE]; read_a.A_P10.s1[cross-track MAE]` | +0.0720 [+0.0403, +0.1109] | `cross-track MAE +0.0720 [+0.0403, +0.1109]` | 7.29 |
| R-NEW condition (3) met, condition (1) fails: L1 lateral-accel MAE, s1 main | `read_a.A_CTRL.s1[L1 lateral-accel MAE]; read_a.A_P10.s1[L1 lateral-accel MAE]` | +1.4444 [+0.6003, +2.4872] | `L1 lateral-accel MAE +1.4444 [+0.6003, +2.4872]` | 7.29 |
| R-NEW condition (3) met, condition (1) fails: along-track MAE, s0 rep | `read_a.Arep_REP.s0[along-track MAE]; read_a.Arep_P10.s0[along-track MAE]` | +0.0493 [+0.0170, +0.0801] | `along-track MAE meets (3), +0.0493 [+0.0170, +0.0801]` | 7.29 |
| 3.1 census of separated cells per family (4 families x 4 pair-seed cells) recomputed from read_a | `read_a.{A_CTRL, Arep_REP}.<seed>[family rows].{separated_better, separated_worse, separated}` | 4 of 4 lines equal (16 census cells) | `/ LONGITUDINAL (6) / 2 / 0 / 4 /` | 7.29 |
| 3.2-3.5 the 17 family rows x 4 X − Y cells (68 cells) with their n, rebuilt | `read_a.{A_CTRL, Arep_REP}.<seed>[row]` | 17 of 17 lines equal | `/ speed MAE / lower /` | 7.29 |
| 3.8 X − P10 and Y − P10 for 19 rows x 2 seeds x 4 columns (152 cells), rebuilt | `read_a.{A_P10, CTRL_P10, Arep_P10, REP_P10}.<seed>[row]` | 38 of 38 lines equal | `/ ADE6 / lower / **-1.2305 [-1.4302, -1.0448] B** /` | 7.29 |
| not_built.M-LAUNCH verbatim | `not_built.M-LAUNCH` | X - Y of the mean (v_own - v_GT) at ticks 1-2 in t... | `X - Y of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps, in each pair: NOT BUILT (routes NEXT-LAUNCH by hand)` | 7.29 |
| not_built.anti_echo_block verbatim | `not_built.anti_echo_block` | longitudinal_claim_admissible, hold-v0, copy detec... | `longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT` | 7.29 |
| not_built.M2_and_L2_launch_subset verbatim | `not_built.M2_and_L2_launch_subset` | along-track MAE, speed MAE and L2 on the GT-launch... | `along-track MAE, speed MAE and L2 on the GT-launch subset; M2: NOT BUILT` | 7.29 |
| not_built.t0_four_families verbatim | `not_built.t0_four_families` | orc_families.py pointed at the arms and their cont... | `orc_families.py pointed at the arms and their controls (the T0 LONGITUDINAL / LATERAL / TACTICAL rows and distance keeping at T0): NOT RUN` | 7.29 |
| next_launch is null | `next_launch` | null | `The JSON field `next_launch` is null` | 7.29 |
| STRATEGIC not paused for v8F | `strategic_family` | NOT PAUSED for v8F (PI ruling 2026-10-17 | `STRATEGIC is not paused for v8F` | 7.29 |
| 76 X - Y cells: 16 separated-better, 7 separated-worse, 53 not separated | `read_a.{A_CTRL, Arep_REP}.<seed>[19 rows]` | 16 / 7 / 53 | `**16 are separated-better, 7 separated-worse and 53 not separated.**` | 7.29 |
| per pair and seed census of the 76 cells | `(the same)` | main s0 8 / 0 / 11, replicate s0 1 / 2 / 16, main s1 1 / 5 / 13, replicate s1 6 / 0 / 13 | `main s0 8 / 0 / 11, replicate s0 1 / 2 / 16, main s1 1 / 5 / 13, replicate s1 6 / 0 / 13` | 7.29 |
| 3 cells reach the bar improving, 6 reach it worsening, 5 of those separated-worse | `(impv >= bar over the 76 cells)` | 3; 6; 5 | `Three cells reach the bar in the improving direction` | 7.29 |
| DK-TG points | `read_a.A_CTRL.s1[DK-TG].delta; read_a.Arep_REP.s0[DK-TG].delta` | -0.0094; -0.0245 | `(−0.0094) and in the replicate pair at s0 (−0.0245)` | 7.29 |
| six cells reach the bar worsening, five separated-worse | `(impv >= bar over the 76 cells)` | 6; 5 | `Six cells reach it in the worsening direction and five of those are separated-worse` | 7.29 |
| RF / bar from 0.17 (DK-VIOL) to 14.32 (yaw-rate MAE), above 1 on 13 of 19 rows | `floors.RF / floors.bar` | 0.17; 14.32; 13 | `RF ÷ bar runs from 0.17 (DK-VIOL) to 14.32 (yaw-rate MAE) and exceeds 1 on 13 of the 19 rows` | 7.29 |
| 2.1 the 19-row floors table (F, RF0, bar = 2 x max(F, RF0), RF, RF / bar), rebuilt from floors.{F, RF0, bar, RF}; its F, RF0, bar and RF columns also equal the PREREG's committed 19-row table, cell by cell | `floors` | 19 of 19 lines equal | `/ ADE6 / lower / 0.1207 / 0.0138 / **0.2414** / 0.4025 / 1.67 /` | 7.29 |
| rows with separated, opposite-sign X - Y cells in the two pairs of a seed | `read_a.{A_CTRL, Arep_REP}.<seed>[row]` | [('s0', 'ADE6'), ('s1', 'ADE6'), ('s1', 'speed MAE'), ('s1', 'accel MAE'), ('s1', 'L1 lateral-accel MAE')] | `at s1, speed MAE, accel MAE and L1 lateral-accel MAE (at s0 only ADE6)` | 7.29 |
| accel MAE at s1: main and replicate | `read_a.{A_CTRL, Arep_REP}.s1[accel MAE]` | +0.7391 [+0.4366, +1.0952]; −0.2560 [−0.4105, −0.1222] | `accel MAE at s1 reads +0.7391 [+0.4366, +1.0952] in the main pair and −0.2560 [−0.4105, −0.1222] in the replicate pair` | 7.29 |
| slow-draw flag: no arm labelled; largest shift 0.1627 m/s; 13 of 20 cell intervals exclude 0 | `beside.A.slow_draw` | none; 0.1627; 13 | `the largest shift in magnitude over the 20 cells is 0.1627 m/s, and 13 of the 20 cell intervals exclude 0` | 7.29 |
| slow-draw table: 4 arm-pair lines (5 cells each, mean [lo, hi], label none) rebuilt from beside.A.slow_draw | `beside.A.slow_draw` | 4 of 4 lines equal | `/ s0 main (A − CTRL) / +0.0718 [+0.0195, +0.1256] /` | 7.29 |
| M-LAUNCH: the 4 cells (X − Y [95 % CI], not separated) rebuilt from MLAUNCH.json primary.cells; 327 windows / 49 episodes in every cell | `primary.cells.<cell>.x_minus_y; launch_subset` | 4 of 4 lines equal; M_LAUNCH = not moved | `/ main (A − DAG-OACC) / s0 / +0.0348 [−0.0299, +0.0923] / no /` | 7.29 |
| training draw alone, REP - DAG-OACC at s0, about 16 times the largest M-LAUNCH point (+0.0348) | `beside_training_seed_floor.REP_minus_CTRL_s0.x_minus_y` | −0.5560 [−0.6994, −0.3975]; ratio 15.98 | `is −0.5560 [−0.6994, −0.3975] m/s, about 16 times the lever's largest point (+0.0348)` | 7.29 |
| REP - DAG-OACC at s1 in the numbers item | `beside_training_seed_floor.REP_minus_CTRL_s1.x_minus_y.delta` | +0.0199 | `(+0.0199 at s1)` | 7.29 |
| M-LAUNCH build controls: 9 groups, all PASS | `controls.*.PASS` | 9 of 9 | `(all nine groups PASS)` | 7.29 |
| M2 beside, separated cell: along-track MAE main_s0 | `beside_M2.pairs.main_s0.rows.along_mae_m.x_minus_y` | −0.1200 [−0.1957, −0.0483] | `along-track MAE, main pair s0 −0.1200 [−0.1957, −0.0483] (better)` | 7.29 |
| M2 beside, separated cell: speed MAE main_s1 | `beside_M2.pairs.main_s1.rows.speed_mae_mps.x_minus_y` | +0.3824 [+0.1367, +0.6489] | `speed MAE, main pair s1 +0.3824 [+0.1367, +0.6489] (worse)` | 7.29 |
| M2 beside, separated cell: L2 curvature MAE replicate_s0 | `beside_M2.pairs.replicate_s0.rows.L2_curvature_mae_fast_1pm.x_minus_y` | −0.1880 [−0.3169, −0.0707] | `L2 curvature MAE, replicate pair s0 −0.1880 [−0.3169, −0.0707] (better)` | 7.29 |
| M2 beside, separated cell: speed MAE replicate_s1 | `beside_M2.pairs.replicate_s1.rows.speed_mae_mps.x_minus_y` | −0.0975 [−0.2103, −0.0040] | `speed MAE, replicate pair s1 −0.0975 [−0.2103, −0.0040] (better)` | 7.29 |
| M2 beside: 253 to 255 windows / 45 episodes for L2; 12 cells, 4 separated | `beside_M2.pairs.*.rows.L2_curvature_mae_fast_1pm.x_minus_y.{n_windows, n_episodes}` | 253..255 / 45; 12; 4 | `253 to 255 windows / 45 episodes for L2` | 7.29 |
| 4.3 K-SHARE / K-MASK / checkpoint prefix / argv sha16 for the 4 runs, rebuilt from the banked box-2 chain log (run record, K-SHARE and 'checkpoint sha256 (trained)' lines) | `chain_v8f_launchintent.out` | 4 of 4 lines equal | `/ A_0 (M2b_s0_launchintent) / 0 / 30727 / 191203 / 0.160704 / 0.331451 /` | 7.29 |
| 4.4 the four launch gates (W, trainer seed, probe batch, launch probe contribution, head_o grad diff, swap probe grad diff), rebuilt from the chain log's 'LAUNCH-INTENT gate: PASS' lines; head_o grad diff > 0 in all four | `chain_v8f_launchintent.out` | 4 of 4 lines equal | `/ A_0 / 1.58926 / 0 / 11 / 64 /` | 7.29 |
| the four weight shares at four decimals equal the PREREG's ESTIMATED shares (0.3315, 0.3307, 0.3256, 0.3281); all inside 1/3 +- 0.02 | `chain log K-SHARE lines` | 0.3256 .. 0.3315 | `(0.3315, 0.3307, 0.3256, 0.3281 at four decimals)` | 7.29 |
| weight share range in 8.1 | `chain log K-SHARE lines` | 0.3256 .. 0.3315 | `0.3256 to 0.3315 against 1/3` | 8.1 |
| box-2 time: first stamped line 13:54 to DONE marker 16:48 = 174 min = 2.9 h | `chain log: first and last 'MM-DD hh:mm' stamps` | 174 min; 2.9 h | `**174 min or 2.9 h**` | 7.29 |
| 174 min in the abstract | `(the same)` | 174 min | `174 min of box 2` | abstract |
| 9 box-2 run times: train launch minute, train ZZJOB-END, T0 eval launch minute, T0 eval ZZJOB-END for the 4 runs, rebuilt from the chain log | `chain_v8f_launchintent.out` | 4 of 4 lines equal | `/ A_0 (M2b_s0_launchintent) / 14:02 / 14:41:35 / 14:41 / 14:43:37 /` | 7.29 |
| the four gates: launched 13:57 / 13:58 / 14:00 / 14:01, ZZJOB-END 13:58:34 / 14:00:02 / 14:01:33 / 14:02:37 | `chain log: gate jobs` | [('13:57', '13:58:34'), ('13:58', '14:00:02'), ('14:00', '14:01:33'), ('14:01', '14:02:37')] | `(launched 13:57, 13:58, 14:00 and 14:01; ZZJOB-END 13:58:34, 14:00:02, 14:01:33 and 14:02:37)` | 7.29 |
| refresh ZZJOB-END 13:54:55; A_0 launched 14:02 | `chain log: refresh, t_M2b_s0_launchintent` | 13:54:55; 14:02 | `(ZZJOB-END 13:54:55, rc 0; box-2 code = tip `d8c9f7bf` at` | 7.29 |
| the pulls at 14:44 / 15:25 / 16:06 / 16:48 | `chain log: 'pulled to' lines` | [('M2b_s0_launchintent', '14:44'), ('M2b_s1_launchintent', '15:25'), ('M2b_s0_launchintentrep', '16:06'), ('M2b_s1_launchintentrep', '16:48')] | `14:44 (A_0), 15:25 (A_1), 16:06 (A_rep,0) and 16:48 (A_rep,1)` | 7.29 |
| the 12 T1-kin chain steps (causal, logged, fresh T0 per run) rebuilt from the relaunch's chain.out; they sum to 1756 s = 29.3 min | `lit1_logs/chain.out` | 4 of 4 lines equal; sum 1756 s = 29.3 min | `/ A_0 / 218 / 218 / 88 /` | 7.29 |
| the twelve steps sum to 1756 s, 29.3 min of dev-box CPU | `lit1_logs/chain.out: 'end ... rc=0 wall=<n>s'` | 1756 s; 29.3 min | `The twelve steps sum to 1756 s (**[M]**, arithmetic on the table), 29.3 min of dev-box CPU` | 7.29 |
| 12 steps in 1756 s in the Provenance bullet | `(the same)` | 1756 s | `12 steps in 1756 s` | 7.29 |

## I. Mechanical checks run

* Every table line inside 7.27 (134 lines) is a line of `RESULT_V8F_SPEEDSUB.md`, inside 7.28 (155 lines) of `RESULT_V8F_SPEEDSUB_T1.md`, and inside 7.29 (288 lines) of `PREREG_V8F_LAUNCH_INTENT.md` or `RESULT_V8F_LAUNCH_INTENT.md`, byte for byte (`patch_v33.py` extracts each table by an anchored heading and asserts membership in the source).
* Numbers of four or more integer digits (plain and comma forms are searched together): 58 in the added lines, each found in a source in the form(s) listed here: 2026 -> 2026; 20260803 -> 20260803; 2000 -> 2,000/2,000,/2000/2000,; 2399 -> 2,399/2,399,/2399/2399,; 20261029 -> 20261029; 4512 -> 4,512/4,512,/4512/4512,; 306566 -> 306,566/306,566,/306566/306566,; 2350 -> 2,350; 2373 -> 2,373/2373; 2328 -> 2,328; 2364 -> 2,364/2364; 2096 -> 2,096; 2102 -> 2,102/2102; 2367 -> 2,367/2367; 2107 -> 2,107; 2240 -> 2,240/2240; 2365 -> 2,365/2365; 2322 -> 2,322; 2319 -> 2,319; 2294 -> 2,294; 2349 -> 2,349; 2329 -> 2,329; 2097 -> 2,097; 2104 -> 2,104/2104; 2334 -> 2,334/2334,; 2351 -> 2,351/2351/2351,; 2317 -> 2,317/2317,; 77853e56 -> 77853e56; 7197 -> 7,197/7197/7197,; 77234276 -> 77234276; 49618 -> 49,618/49618,; 3000 -> 3,000/3000/3000,; 524288 -> 524,288; 262144 -> 262,144; 261888 -> 261,888; 8168 -> 8168; 30727 -> 30,727/30727/30727,; 191203 -> 191,203/191203/191203,; 30674 -> 30,674/30674/30674,; 191396 -> 191,396/191396/191396, ....
* Open outcome slots in this deliverable (a final version has none): 0: .
* Banned words, closed-loop wording, tier stamps, section order, superset and demotion checks, the offline render and the rebuild: see `v33_checks.txt`.

## J. Log, commit and hash cross-check of the LAUNCH-INTENT disclosures (re-measured from the repository)

Every row is read from a banked log, a commit, or a blob at the tip and the quoted snippet is found in the named part of v33 (an assertion fails the script on a mismatch). The times are the logs' own stamps and the commits' author dates; the Master Mind's brief is not a source here, which is why the 15:11 stop time and the 2,048 MB guard value of the T1-kin chain (not in any log) stay tagged INHERITED in the text.

| what | source | value in the source | snippet in v33 | found in |
|---|---|---|---|---|
| commit 77234276 (PREREG promoted) at 13:49:00 and commit d8c9f7bf (queue stage) at 13:51:52 bracket the pin fill | `git log --format=%ad` | 2026-10-28 13:49:00 ; 2026-10-28 13:51:52 | `before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00)` | 7.29 |
| commit e28516e8 (Amendment 1) at 15:24:46 | `git log --format=%ad` | 2026-10-28 15:24:46 | `It landed at `e28516e8` (15:24:46)` | 7.29 |
| RESULT landed 8c1e9636 at 17:43:02; the read banked at d7dc9f11 at 17:11:44 | `git log --format=%ad` | 2026-10-28 17:43:02 ; 2026-10-28 17:11:44 | `(landed `8c1e9636`, 17:43:02; the read banked at `d7dc9f11`, 17:11:44)` | 7.29 |
| the PREREG blob and the reader blob at 77234276 hash to 469bac1f and 5e950dcf (the pins the chain carries) | `git show 77234276:<path> / sha256` | 469bac1f ; 5e950dcf | `hash to the `469bac1f…` and `5e950dcf…` that the pins carry` | 7.29 |
| the amended PREREG has sha256 6ae79f2f at e28516e8, d7dc9f11 and the tip | `git show <rev>:Project Steering/PREREG_V8F_LAUNCH_INTENT.md / sha256` | 6ae79f2f | `gives the amended text sha256 `6ae79f2f…`` | 7.29 |
| PINS_FILLED.txt carries the 14:10 stamp | `launch-intent-chain/PINS_FILLED.txt` | PINS FILLED by the Master Mind 2026-10-28 ~14:10, after PREREG_V8F_LAUNCH_INTENT was promoted and la | `says the Master Mind filled the chain pins at ~14:10` | 7.29 |
| the provenance table of 7.29: 15 repo rows, each sha256 equal to the sha256 of the blob at d7dc9f11 (the PREREG row: the blob at e28516e8, d7dc9f11 and the tip); 4 checkpoint rows, each full sha256 begins with the 12-hex prefix of the chain log's 'checkpoint sha256 (trained)' line and occurs in the RESULT | `git show d7dc9f11:<path> / sha256; chain_v8f_launchintent.out; RESULT_V8F_LAUNCH_INTENT.md` | 19 of 19 equal | `/ read of record (reader output) / `read_A1/LAUNCH_INTENT_READ.json` / `e451e1a6ba44e1d3` | 7.29 |
| the stopped launch's last line is stamped 15:07:07 and reads 'LOW RAM available 3040 MB < 4096 MB'; its log holds no 'start' line for a rollout | `lit1_logs/chain.out.prev_1028151251` | 10-28 15:07:07 LI_s0_m10_causal_heads: waiting, a guard is LOW: LOW RAM available 3040 MB < 4096 MB / ram=3040MB disk_fr | `the stopped launch's last line, 15:07:07, reads` | 7.29 |
| the relaunch started at 15:12:51, its first rollout at 15:13:04, all eight rollouts show attempt 1 | `lit1_logs/chain.out` | 15:12:51; 15:13:04; attempts ['1', '1', '1', '1', '1', '1', '1', '1'] | `The relaunch started at 15:12:51 and its first rollout at 15:13:04, all eight rollouts in its log show attempt 1` | 7.29 |
| A_0's three T1-kin steps ran from 15:13:04 to 15:22:14 | `lit1_logs/chain.out` | 15:13:04 .. 15:22:14 | `A_0's three T1-kin steps ran from 15:13:04 to 15:22:14` | 7.29 |
| the chain's own read ran 16:55:55 to 17:01:36 and REFUSED on one failed control, K-BUILD | `lit1_logs/chain.out; lit1_logs/launch_intent_read_REFUSED.out` | 16:55:55 .. 17:01:36; 1 failed control: K-BUILD | `The chain's own read ran 16:55:55 to 17:01:36 and REFUSED` | 7.29 |
| the read of record: first console stamp 17:02:21; reader fb3d2d7d with PREREG 6ae79f2f on the console's first line; the console ends with ZZLAUNCH-INTENT-READ-DONE HARMS and ZZREAD-A1-RC 0 | `read_A1/read_A1.out` | 17:02:21; fb3d2d7d / 6ae79f2f; HARMS; rc 0 | `before the read of record started (first console stamp 17:02:21)` | 7.29 |
| the console's last lines | `read_A1/read_A1.out` | ZZLAUNCH-INTENT-READ-DONE HARMS; ZZREAD-A1-RC 0 | `*"ZZLAUNCH-INTENT-READ-DONE HARMS"* and *"ZZREAD-A1-RC 0"*` | 7.29 |
| the RESULT's own checker: 133 checks, 0 failed | `check_result_filled.out` | SUMMARY 133 checks passed, 0 failed | `(133 checks, 0 failed; **[I]**, its output in the repo)` | 7.29 |
| REP s0 trained and was evaluated before 05:56; REP s1 after the r2c relaunch on the branch pinned at 8c24823f | `R2C_NOTE.md` | Unaffected. A s0 and REP s0 trained and evaluated before 05:56. | `REP s0 trained and was evaluated before 05:56` | 7.29 |
| both REP launch-gate tokens bind commit 8c24823f; the run records carry trainer seeds 10 and 11 | `REP launch-gate tokens; REP run records` | 8c24823f x2; seeds 10, 11 | `bind commit `8c24823f` (**[M]**` | 7.29 |

