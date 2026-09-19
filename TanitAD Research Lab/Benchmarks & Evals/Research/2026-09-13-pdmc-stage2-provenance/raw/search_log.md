# search log — 2026-09-13 pdmc-stage2-provenance

| # | query / probe | route | hits | result |
|---|---|---|---|---|
| 1 | where did our 56.6 come from | `grep 56\.6` over `Frontier Scan/Daily/2026-09-02/RESULT.md`, `LEDGER_A5_benchmarks.md` | 3 | recorded 09-02 as "NAVSIM v2", later re-quoted as "navhard"; no arXiv id attached |
| 2 | `DrivoR arXiv NAVSIM v2 EPDMS 56.3 PDM-Closed` | WebSearch | 10 | **TOAD `2606.07170`** is the source of both 56.3 and 56.6 |
| 3 | `navhard two-stage EPDMS PDM-Closed 51.3 table NAVSIM v2 paper` | WebSearch | 10 | `2506.04218` Table 2 |
| 4 | TOAD Table 2 caption + split | WebFetch arxiv.org/html/2606.07170v1 | 1 | "NAVSIM-v2 navhard-two-stage … test set"; PDM-C cited to [8] Dauner CoRL'23 |
| 5 | TOAD Table 6 stage-wise | WebFetch same | 1 | stage-wise subscores extracted |
| 6 | Pseudo-Sim Table 2 stage-wise + aggregation | WebFetch arxiv.org/html/2506.04218v2 | 1 | v2 2025-08-27; `s1·s2`; subscores extracted |
| 7 | NAVSIM version history | WebFetch github.com/autonomousvision/navsim/releases | 1 | v2.0 (04 Mar), v2.1 / v2.1.1 / v2.1.2 (28 Apr); **no later release** |
| 8 | `navsim navhard_two_stage synthetic scenes update 2026 stage 2 regenerated EPDMS change GitHub issue` | WebSearch | 10 | **EMPTY for a documented Stage-2 change** (probe 1). ⚠️ A search snippet asserted *"244 Stage-1 / 4,164 Stage-2 scenarios"* for navhard — **UNATTRIBUTED** (GS-6): not found on `docs/splits.md` (probe 2, which lists no counts at all); barred |
| 9 | `docs/splits.md` | WebFetch | 1 | no scenario counts, no versioning — **EMPTY at a second probe** for a Stage-2 version statement |
| 10 | `docs/metrics.md` | WebFetch | 1 | Gaussian-kernel proximity weighting; multiplication of stages |
| 11 | CLOVER `2605.15120` v2 — a third PDM-C navhard row? | WebFetch html | 1 | **no PDM-Closed row** (EMPTY) — third-table fingerprint deferred to `E-BE-S1S2-1` |
| 12 | ⭐ V-1 AFTER the fact: is the banked `2506.04218` the same version as the HTML read? | PyMuPDF on `Library/papers/2506.04218_…pdf` | 1 | ⛔ **NO — banked = v3 (20 Mar 2026), HTML read = v2 (27 Aug 2025).** v3 Table 2 = 03/2026 leaderboard snapshot, PDM-C **56.6**; strings `51.3/88.1/83.1/25.4/73.7` absent (0), `56.6/90.5/86.6/29.7/74.2` present, control `94.4` ×3. **This probe reversed F4 of the first draft.** |
