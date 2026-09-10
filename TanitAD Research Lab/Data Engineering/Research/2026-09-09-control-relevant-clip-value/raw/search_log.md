<title>Search log - control-relevant clip value (2026-09-09)</title>

# SEARCH LOG - CONTROL-RELEVANT CLIP VALUE

`Retrieval date 2026-09-09, dev box, WebSearch / WebFetch. Every query recorded; every empty named.`
`The full 22-track scan for this day lives in Frontier Scan/Daily/2026-09-09/raw/search_log.md - this file records only the queries this package rests on.`

| # | query / route | hits | outcome |
|---|---|---|---|
| 1 | `arxiv 2026 video dataset curation data selection world model training which clips matter` | 8 | Kairos `2606.16533`, Summer-22B `2603.00173`, VidaForge `2609.06652`, Cosmos curation `2511.00062`. |
| 2 | `arxiv 2026 retrieval augmented driving scenario embedding search memory bank nearest neighbor policy` | 10 | `2606.28383` zero-label JEPA complexity scorer; RealDrive `2505.24808`; Driving-RAG `2504.04419`. |
| 3 | WebFetch `arxiv.org/html/2606.28383v1` | ok | **FULL TEXT.** Score formula, four ablations with rho values, alpha = 0.996, AP 0.512 against 0.436 chance, 1,322 scenarios, 1,289,130 parameters. |
| 4 | WebFetch `arxiv.org/pdf/2606.16533` (Kairos) | **FAIL** | `maxContentLength size of 10485760 exceeded`. |
| 5 | WebFetch `arxiv.org/abs/2606.16533` | partial | Landing page only - abstract, no curation section. |
| 6 | WebFetch `arxiv.org/html/2606.16533v1` | partial | **Truncated mid-section 4.2 (Data Curation).** Collection strategy and shot-segmentation thresholds readable; the two-level curation mechanics are not. |

## Named empties and gaps

| id | what | probes | verdict |
|---|---|---|---|
| **E-K** | **Kairos's control-relevant event filtering mechanics** | **3 routes** (pdf too large, abs landing-only, html truncated) | **UNREAD.** The two-level-curation claim and the "a visually clean clip is not necessarily control-informative" line are **RELAYED from a search-engine summary and are barred from deciding anything in this package.** Carried as register debt D-8, which this pass makes worse rather than better. |
| **E-DV** | Whether the complexity score predicts **downstream training value** | 1 (full text) | **The paper concedes it does not test this**: it validates against scenario tags and anomaly-detection ranking only. This is the gap our own experiment is designed to fill, and it is why row 22's training arm is not skipped. |
| **E-LIB** | Library "already banked?" for `2606.28383` and `2505.24808` | 0 | **NOT RUN - INCONCLUSIVE.** Mount down all pass (Errno 22). Rule V-1 unsatisfied; banking owed (debt D-12). |
