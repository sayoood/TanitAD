<title>Search log - I-1 quality half pre-screen (2026-09-09)</title>

# SEARCH LOG - I-1 QUALITY HALF PRE-SCREEN

`Retrieval date 2026-09-09, dev box, WebSearch / WebFetch. Every query recorded; every empty named.`
`The full 22-track scan for this day lives in Frontier Scan/Daily/2026-09-09/raw/search_log.md - this file records only the queries this package rests on.`

| # | query / route | hits | outcome |
|---|---|---|---|
| 1 | WebFetch `arxiv.org/html/2608.12939` | ok | **FULL TEXT.** The proof statement that divergence bounds "multi-step prediction error and planner cost"; **CEM selection regret 15.2 +/- 2.0 %**; prediction-error MAE 55.9 +/- 4.7 %; H=5 for the planning analysis. |
| 2 | `arxiv 2026 diffusion flow matching trajectory planning driving multimodal futures scorer` | 8 | FlowR2A `2606.24231` (reward-to-action distribution, unifying scoring-based dense supervision with anchor-based generation); WAM-Flow `2512.06112`; GuideFlow `2511.18729` (already read 2026-09-05). |
| 3 | in-programme re-read (no new retrieval) | - | I-1 cost half: 5.94x, 0.150 ms, breadth flat to N about 32 (`2026-09-01-terminal-value-vs-fan-scoring`). P-7: latency linear in K, 350-1,015 ms against a 100 ms budget. FS5-1: the ranking never sees the refined fan (201/201). |

## Named empties and gaps

| id | what | probes | verdict |
|---|---|---|---|
| **E-P1** | A published **matched-LATENCY** comparison of a learned terminal value against fan-scoring on a driving planner | 1 | **NOT FOUND at one probe.** One probe is not absence (CLAUDE.md rule 2), so this is recorded as "not found at the searched location" and **not** as "nobody has done this". A second probe with varied terms is proposed as a hygiene row. |
| **E-ICEM** | Any published treatment of **inference-seed variance** for a sampling planner in this comparison | 1 | **NOT FOUND at one probe.** Our own CLAUDE.md 0.30 m seed floor remains the only quantified statement we hold, and it is OURS (MEASURED), not published. |
| **E-LIB** | Library "already banked?" for `2608.12939` and `2606.24231` | 0 | **NOT RUN - INCONCLUSIVE.** Mount down (Errno 22). Banking owed (debt D-12). |
