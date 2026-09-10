# Search log — Deployment & Optimization, 2026-09-10

`Shared retrieval with the day's frontier scan; full query list in ../../../../Frontier Scan/Daily/2026-09-10/raw/search_log.md`

| # | query / route | hits | outcome |
|---|---|---|---|
| D-1 | `arxiv September 2026 JEPA joint embedding predictive architecture action conditioned` | 8 | `2601.00844` *Value-guided action planning with JEPA world models* surfaced |
| D-2 | `arxiv.org/abs/2601.00844` | metadata | LeCun et al., World Modeling Workshop 2026, submitted 2025-12-28 |
| D-3 | `arxiv.org/pdf/2601.00844` → local PyMuPDF extraction | **full text, 7 pp.** | Table 2 (all ten variants × three environments), the IQL loss, MPPI planner, 200/80 instance counts, both author-stated risks |
| D-4 | Drive-HWM `2609.03572` Table III | full text | `Ts` 25.6 / `Tf` 81.6 / `Tpeak` **107.2** / `Tavg` 84.8 — the P-7 amortisation datapoint **and its bound** |
| D-5 | `nvidia-smi --query-compute-apps` | — | GPU free of python compute (RTX 4060, 1,765 / 8,188 MiB, Windows shell processes only). **No GPU work was run; the proposed screen is 0-GPU** |
| D-6 | library check | — | ⚠️ **`2601.00844` was ALREADY BANKED.** V-1 violated: the library was checked after the web search, not before |

## Empty / not found

| id | item | status |
|---|---|---|
| **D-E1** | a direct comparison of a learned terminal-value head against fan/sampling breadth at equal rollout budget | ⛔ **NOT FOUND in `2601.00844`.** The paper compares *representation-shaping* variants under one fixed MPC planner; it never varies the planner's breadth. **I-1's question as literally posed is still unanswered in the literature** — what changed is that the paper relocates the lever to the latent |
| **D-E2** | whether `2601.00844`'s encoder is frozen in any variant | ⛔ **NOT STATED for a frozen case** — every listed variant trains the state encoder with some loss. The frozen-trunk blocker is our inference from the method's form, and is marked as such |
