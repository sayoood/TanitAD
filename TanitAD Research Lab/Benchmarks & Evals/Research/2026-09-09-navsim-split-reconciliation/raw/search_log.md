<title>Search log - NAVSIM split reconciliation (debt D-10) (2026-09-09)</title>

# SEARCH LOG - NAVSIM SPLIT RECONCILIATION (DEBT D-10)

`Retrieval date 2026-09-09, dev box, WebSearch / WebFetch. Every query recorded; every empty named.`
`The full 22-track scan for this day lives in Frontier Scan/Daily/2026-09-09/raw/search_log.md - this file records only the queries this package rests on.`

| # | query / route | hits | outcome |
|---|---|---|---|
| 1 | `arxiv 2026 NAVSIM v2 EPDMS navhard navtest leaderboard scoring` | 10 | The maintainers' statement discouraging **self-reported and unofficial "NAVSIM v2" splits**; navhard sized at **450 Stage-1 plus 5,462 Stage-2**; NAVSIM v2 Stage 2 confirmed as 3DGS counterfactual. |
| 2 | `arxiv 2026 world model latent action prediction driving September` | 9 | Surfaced Latent-WAM `2603.24581` (89.3 EPDMS, 104M). |
| 3 | WebFetch `arxiv.org/abs/2603.24581` | ok | Abstract, 104M, 89.3 EPDMS, 28.9 HD-Score. **Split NOT stated in the abstract.** |
| 4 | WebFetch `arxiv.org/html/2603.24581v1` | ok | **FULL TEXT results.** Table 1 rows (89.3 / 86.1 / 85.1 / 84.8), the phrase "12k evaluation scenarios", the 104M-vs-191M parameter split, the EMA frozen-SCWE note. |

## Named empties and gaps

| id | what | probes | verdict |
|---|---|---|---|
| **E-1** | **The split name for Latent-WAM's 89.3** | 2 (abstract, then full-text results section) | **NOT STATED.** The paper never writes navtest, navhard or navsafe. **The assignment of "12k evaluation scenarios" to a navtest-class split is OUR INFERENCE from the scenario count**, not a quotation, and is marked as such in the package. An appendix probe was not possible in this retrieval. |
| **E-2** | **Leaderboard status of the 89.3** | 1 | **No reference to the official leaderboard found.** One probe only, so this is a weak absence and is written as "reads as self-evaluated", never as "they did not submit". |
| **E-3** | Provenance of OUR 55.5 / 56.3 | 0 | **NOT PROBED - the G: mount was down all pass**, so `LEADERBOARD.md` and the register could not be re-read. **INCONCLUSIVE**, and the 0-GPU six-number extraction is proposed as a work item rather than guessed at. |
