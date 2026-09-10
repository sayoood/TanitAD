# Search log — Data Engineering, 2026-09-10

`Shared retrieval with the day's frontier scan; full query list in ../../../../Frontier Scan/Daily/2026-09-10/raw/search_log.md`

## The five routes to `2606.16533`

| route | method | result |
|---|---|---|
| 1 | fetch-tool on the PDF | ⛔ failed 2026-09-09 — PDF too large |
| 2 | landing page | ⛔ failed 2026-09-09 — metadata only |
| 3 | HTML | ⛔ failed 2026-09-09 — truncated mid-§4.2 |
| 4 | `arxiv.org/html/2606.16533v1` | ⛔ **failed today, identically** — truncated at the same sentence: *"This data pool n"* |
| **5** | **`urllib` byte download → local PyMuPDF extraction** | ✅ **SUCCEEDED.** v1: **42,560,348 B**, `%PDF-` header, valid `%%EOF`, 90 pp. v3: **43,862,955 B**, valid `%%EOF`, 119 pp. |

⭐ **The block was always SIZE, never access.** Three "unreachable" verdicts across two passes were one tool's limit on a fully public document.

## ⛔ The version trap

| version check | result |
|---|---|
| `arxiv.org/abs/2606.16533` version list | **v1, v2, v3** exist; current title *"Kairos: A **Regret-Aware** Native World-Action Model Stack for Physical AI"* |
| grep v1 (90 pp.) | `control-relevant` **×1** (unrelated VideoDiT ablation) · `two-level` **×3** (all *two-level batching*, an I/O technique) · `control-informative` **×0** · `regret` **×0** |
| grep v3 (119 pp.) | `control-relevant` **×42** · `regret` **×96** · claim present verbatim |

⛔ **Reading v1 alone would have produced a confident, false `REFUTED` on a true claim.** The version check was run only because `regret` returned **0** in a paper whose title contains *"Regret-Aware"* — the discrepancy that caught it.

## Queries

| # | query | hits | outcome |
|---|---|---|---|
| K-1 | `arxiv 2026 Kairos 2606.16533 data curation two-level video` | 8 | search summary asserted the two-level claim — ⚠️ **this is the RELAYED path that created debt D-8; it is a summary, not the paper** |
| K-2 | full-text grep of v3 for `Control-Relevant Event Filtering` | 1 section | the six event families, verbatim, **and the "does not yet compute CID directly" concession** |
| K-3 | full-text grep of both versions for curation numbers | — | level-1 figures only (PySceneDetect >95 % precision / 80 % recall; 5–40 s segments); ⛔ **no level-2 retention rate and no curation ablation exist in either version** |
| K-4 | library check | — | `2606.16533` **NOT banked** — one of only two genuinely new primaries today |

## Empty / not found

| id | item | status |
|---|---|---|
| **K-E1** | any CID (control information density) computation, threshold or score in Kairos | ⛔ **ABSENT BY THE PAPER'S OWN STATEMENT**: *"The current pipeline does not yet compute CID directly."* Absence confirmed by the source, not inferred from a failed search |
| **K-E2** | an ablation of curation against downstream model quality | ⛔ **NOT PRESENT in either version.** Two probes (v1 and v3 full-text). Level 1 is described but never measured |
| **K-E3** | thresholds for the aesthetic and motion filters | ⛔ **NOT STATED.** *"Samples falling below a predefined threshold"* — the threshold is never given |
