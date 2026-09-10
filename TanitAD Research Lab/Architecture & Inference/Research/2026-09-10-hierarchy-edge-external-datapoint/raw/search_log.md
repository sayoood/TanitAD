# Search log — Architecture & Inference, 2026-09-10

`Shared retrieval with the day's frontier scan; full query list in ../../../../Frontier Scan/Daily/2026-09-10/raw/search_log.md`

| # | query / route | hits | outcome |
|---|---|---|---|
| A-1 | `world model autonomous driving latent September 2026 arxiv` | 8 | Drive-HWM `2609.03572` surfaced |
| A-2 | `arxiv 2609.03572 Drive-HWM hierarchical world model dynamic latent` | 7 | authorship + 2026-09-03 date confirmed; also `2604.03208` *Hierarchical Planning with Latent World Models* (**unread — proposed as a row-18 candidate**) |
| A-3 | `arxiv.org/abs/2609.03572` → `arxiv.org/html/2609.03572` | full text | Table III and Table IV extracted verbatim |
| A-4 | `arxiv.org/abs/2603.19312` → `arxiv.org/html/2603.19312v1` (LeWM) | full text | AdaLN conditioning; **no action-sensitivity control**; SIGReg; CEM 300×30, horizon 5 |
| A-5 | library check (`Library/library.json`) | — | ⚠️ **run AFTER the searches, violating V-1.** `2603.19312` was **already banked**; `2609.03572` was not |

## Empty / refused

| id | item | status |
|---|---|---|
| **A-E1** | parameter counts for Drive-HWM's "Fast only" / "Slow only" ablation arms | ⛔ **NOT STATED IN THE PAPER.** Probed the abstract, Table III, Table IV and the ablation text. Absence is the finding, not a retrieval failure |
| **A-E2** | seconds-per-timestep for Drive-HWM's `N = K = 8` | ⛔ **NOT STATED.** Their K=8 is therefore unit-incomparable with our K=8 (16 s) rung in backlog P-17 |
