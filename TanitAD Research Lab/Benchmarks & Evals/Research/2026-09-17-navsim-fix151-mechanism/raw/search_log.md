# search_log — E-BE-FIX151-1 (2026-09-17)

⭐ **V-1 applied: the Library was checked before the web.** `2601.05083` (DrivoR) and `2606.07170`
(TOAD) were already banked by the 09-15 pass, so no re-bank was needed; only the **issue primary**
and the **EPDMS composition** were new.

| # | query / URL | tool | hits | what it settled |
|---|---|---|---|---|
| Q1 | `github.com/autonomousvision/navsim/issues/151` | fetch | 1 | ⭐⭐ **F1** — title, mechanism, opened 2025-09-03, the `weighted_metrics_array` sentence, the 0.875 demonstration, and that the issue is closed with **no linked fix commit visible on the page** |
| Q2 | `NAVSIM EPDMS weighted_metrics_array multiplicative metrics ego progress drivable area human_penalty_filter fix` | search | 10 | ⭐⭐ **F2 / F3** — the partition: multiplicative **NC · DAC · DDC · TLC**; weighted **EP(5) · TTC(5) · LK(2) · HC(2) · EC(2)**. Cross-confirmed across two independent papers describing NAVSIM v2 (`2605.00066`, `2505.23129`), neither of them DrivoR — so the composition does not rest on the same source as the caption under test |
| Q3 | `arxiv/html/2608.20974v1` Table 1 caption | fetch | 1 | ⭐⭐ **F5** — *"Comparison with state-of-the-art methods on NAVSIM-v2 **navtest**"*; rows SparseDriveV2 90.1, Discrete-WAM 90.4, WA-JEPA 91.7; HUGSIM Tab. 2 HD-Score WA-JEPA 0.4462 vs DrivoR 0.3252, UniAD 0.3124, LTF 0.2310, VAD 0.1393 |

## Empty / partial searches (named, per charter §5.4)

| # | query | result | scope of the resulting statement |
|---|---|---|---|
| **E-BE1** | a **fix commit or PR** linked from issue #151 | ⛔ **EMPTY at one probe (the issue page itself)** | ⚠️ This supports only *"no fix is linked from the issue page"*. It does **NOT** support *"no fix commit exists"* — the repo history and release notes were **not** searched. `E-BE-FIX151-2` is registered to do exactly that, and until it runs the causal link from #151 to the substrate shift stays an inference |
| **E-BE2** | NAVSIM's **own source** for the EPDMS weights (5/5/2/2/2) | not probed this pass | The weights are PUBLISHED from papers *describing* NAVSIM, not read from its code. A version drift in the weights would not be caught by today's read. Named as a limitation in §3 rather than smoothed over |
| **E-BE3** | whether 09-13's **7 of 9 moved subscores** are multiplicative or weighted | not re-derived | F3's "unexplained component" is therefore conditional on an **INHERITED** identification. Folded into `E-BE-FIX151-2` as its second half |
