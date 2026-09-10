# Search log — Benchmarks & Evals, 2026-09-10

`Shared retrieval with the day's frontier scan; full query list in ../../../../Frontier Scan/Daily/2026-09-10/raw/search_log.md`

| # | query / route | hits | outcome |
|---|---|---|---|
| B-1 | `NAVSIM v2 navhard EPDMS leaderboard scenario count split September 2026` | 9 | pointed to the maintainers' own Pseudo-Simulation paper |
| B-2 | `github.com/autonomousvision/navsim/blob/main/docs/splits.md` | 1 | ⛔ **PARTIAL — storage sizes only** (navtrain 14 GB/445 GB, navtest 983 MB/223 GB, navhard_two_stage 892 MB/31 GB, warmup_two_stage 27 MB/1.2 GB, private_test_hard_two_stage 14 MB/11 GB). **No scenario counts anywhere in the doc.** A named partial, not a failure |
| B-3 | `arxiv.org/html/2506.04218v2` (the NAVSIM v2 maintainers' paper) | full text | ⭐⭐⭐ **THE PRIMARY.** navhard *"450 Stage 1 and 5462 Stage 2 observations"*; PDM-Closed 51.3 / Latent TransFuser 23.1 / MLP 12.7 / ConstVel 10.9; EPDMS range [0, 1] |
| B-4 | `NAVSIM navtest 12000 scenarios count PDMS benchmark split definition` | 10 | navtest ≈ 12,000 samples (~136 test logs); full PDMS and EPDMS sub-metric formulas |
| B-5 | Drive-HWM `2609.03572` NAVSIM v2 row | full text | EPDMS **86.4** → stamped **navtest**; PDMS 93.8 → **NAVSIM v1**, a different metric |
| B-6 | library check | — | `2506.04218` **already banked** (V-1 again) |

## Empty / partial

| id | item | status |
|---|---|---|
| **B-E1** | a maintainers' primary stating navtest's scenario count | ⚠️ **NOT FOUND at two probes.** The ≈12,000 figure is `PUBLISHED-SECONDARY`, consistent across several 2026 method papers but not quoted from NAVSIM's own docs. **Marked as secondary in the ledger; a third probe should read the NAVSIM NeurIPS '24 paper directly** |
| **B-E2** | provenance of the **PDM-Closed = 56.6 on navhard** figure our own records carry | ⛔ **NOT RESOLVED THIS PASS.** It is a TanitAD-internal record (LAB-RUN-008), and reconciling it against the primary's 51.3 is the pre-registered experiment in RESULT.md, not something a web search settles |
| **B-E3** | split names attached to DriveFuture 55.5 and DrivoR 56.3 | ⛔ **STILL UNSTAMPED.** They match neither cluster's centre. Guideline T-4 continues to bar them |
