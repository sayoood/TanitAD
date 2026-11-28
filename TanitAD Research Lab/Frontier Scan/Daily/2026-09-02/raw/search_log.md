<title>Search log — frontier scan 2026-09-02</title>

# SEARCH LOG — 2026-09-02

`Every query, its outcome, and every EMPTY named (charter §5 rule 4 / §6 step 3).`
`⭐ V-1 applied FIRST: the Library was searched before the web. 6 of 8 primaries needed today were ALREADY BANKED.`

## Library-first probes (V-1)

| # | probe | result |
|---|---|---|
| L1 | `library.json` lookup for the four full-text debts (`2607.04732`, `2601.22156`, `2606.31672`, `2606.21775`) | ⭐ **ALL FOUR ON DISK** — 7.6 MB / 0.9 MB / 35.5 MB / 1.6 MB. Every one of the day's carried debts was readable at zero cost. |
| L2 | `library.json` lookup for `2606.31232` (Delta-JEPA) after the web surfaced it | ⛔ **ALREADY BANKED AND UNREAD.** `kb_add` returned *"already banked"*. |
| L3 | first path-existence probe for L1 | ⚠️ **FALSE NEGATIVE OF MY OWN MAKING** — I joined the stored repo-relative path onto the Library dir, doubling it, and read `exists=False` for all six. Re-probed correctly ⇒ all present. **Recorded because it is the exact "absence at one probe" failure the standard warns about, committed by the auditor.** |

## Band D — opponent doctrine (priority 1)

| # | query | outcome |
|---|---|---|
| D1 | `Wayve Waabi Mobileye technical blog 2026 lessons autonomous driving architecture doctrine` | ⚠️ **THIN** — trade-press roundups, no doctrine documents. Not empty, but nothing adjudicable. |
| D2 | `Mobileye Shashua 2026 physical AI keynote architecture thesis end-to-end vs compound AI` | ⭐⭐ **HIT** — surfaced *Compound AI: The framework powering scalable autonomy* (Mobileye blog) **and** the academic primary `1604.06915`. This became the day's Band-D item. |
| D3 | fetch `mobileye.com/blog/compound-ai-the-framework-powering-scalable-autonomy/` | HIT — full doctrine text, dated **2025-07-31**. |
| D4 | fetch `arxiv.org/abs/1604.06915` | HIT — abstract + framing. **Banked.** |
| D5 | ⭐ **COUNTER-SEARCH (protocol step 4, mandatory)**: `end-to-end learning outperforms modular pipeline autonomous driving evidence 2025 2026 scaling refutes decomposition` | ⭐⭐ **HIT, AND IT CONTRADICTS D2/D3/D4.** Surfaced `2603.16050` (*The Era of End-to-End Autonomy*) and `2412.09602` (*Hidden Biases of End-to-End Driving Datasets*). Directly opposed to Mobileye's doctrine ⇒ verdict is **CONTESTED**, not confirmed. |

## Band A

| # | query | outcome |
|---|---|---|
| A1 | `arxiv 2026 JEPA joint embedding predictive architecture representation collapse action conditioned latent world model new` | ⭐⭐⭐ **HIT** — Delta-JEPA `2606.31232`, LeWorldModel `2603.19312`, Sub-JEPA `2605.09241`, EB-JEPA `2602.03604`. A2 DEEP restored. |
| A2 | `arxiv 2026 vision-language-action model VLA driving action tokenization latency real-time` | HIT — `2512.16760` (survey), `2606.14010` (RT-VLA), `2502.02175` (VLA-Cache). A4 DEEP restored. |
| A3 | fetch `arxiv.org/pdf/2606.31232` via WebFetch | ⚠️ **FAILED** — returned raw PDF binary, unreadable by the fetch tool. **Recovered** by `kb_add` + local `pdftotext`. *(Method note: for arXiv PDFs, bank-then-extract beats WebFetch.)* |
| A4 | fetch `arxiv.org/html/2512.16760v1` | HIT — and it **REFUTED a number the search summary asserted** (see RESULT F6). |

## Band B

| # | query | outcome |
|---|---|---|
| B1 | local full-text grep of `2601.22156` for the conversion recipe | HIT — HALO pipeline, layer-selection criterion, 2.3B-token figure in context. |
| B2 | local full-text grep of `2606.21775` for the VLWM mechanism | ⭐⭐ HIT — the **action-as-token vs FiLM** argument, unreadable from the abstract. |
| B3 | local full-text grep of `2607.04732` for supervision requirements | ⭐ HIT — **debt D-5 answered decisively** (see RESULT F2). |
| B4 | `arxiv 2026 video dataset curation selection training data efficiency world model which clips matter pruning` | HIT (B9 SCAN) — MiniWorld `2608.01127`, Summer-22B `2603.00173`, WARP-RM `2606.28320`. ⚠️ The Open-Sora *70M→10M* figure is **RELAYED via a survey summary**, primary not read. |

## Band C

| # | query | outcome |
|---|---|---|
| C1 | `UNECE GRVA automated driving systems regulation 2026 ISMR DSSAD text adopted WP.29 document` | HIT — surfaced a **direct PDF path** to `ECE-TRANS-WP.29-GRVA-2026-02e.pdf` plus a legal analysis. |
| C2 | fetch `unece.org/sites/default/files/2026-01/ECE-TRANS-WP.29-GRVA-2026-02e.pdf` | ⛔ **HTTP 403 — THIRD FAILED ROUTE TO THE PRIMARY.** Debt **D-4 STANDS.** |
| C3 | fetch the Sidley legal analysis (2026-03-04) | HIT — ISMR/DSSAD/SMS quoted; **online learning and model-update re-approval NOT MENTIONED**. Third independent probe. |
| C4 | `NAVSIM navhard leaderboard 2026 results new state of the art EPDMS driving benchmark release August September` | HIT — DrivoR **56.3**, PDM-Closed **56.6**, CLOVER **48.3** navhard-two-stage, RAP-DINO **36.9**. ⚠️ **Different splits — must not be pooled (V-5).** First C4 numbers the programme holds. |

## ⛔ EMPTIES AND FAILURES NAMED

| id | what was sought | status |
|---|---|---|
| **E1** *(carried, now 3 probes)* | DINOv3-class encoders benchmarked on **BEV/driving** dense tasks | ⛔ **STILL EMPTY.** Today's A1/A2 sweeps surfaced no driving-BEV encoder benchmark. Third probe, three phrasings. **Do not re-run without a new trigger** — promote toward the standing empty-search register. |
| **E7** *(new)* | An **August/September 2026** NAVSIM or navhard leaderboard release note | ⛔ **EMPTY** — the search returned papers and a **March 2026** leaderboard snapshot only. The C4 numbers below are from paper claims, **not** from a leaderboard read. |
| **E8** *(new)* | UNECE GRVA **primary text** by any non-403 route | ⛔ **EMPTY AT THREE ROUTES** (direct PDF, prior pass's route, secondary-with-link). D-4 stands. |
| **E9** *(new)* | Any Wayve / Waabi / Zoox **doctrine document** dated 2026 | ⛔ **EMPTY at one probe (D1).** ⚠️ **One probe is not absence** — carried as a single-probe miss, explicitly NOT recorded as "no such document exists". Second probe owed next pass. |
| **F-A3** *(tool failure, not an empty)* | Delta-JEPA via WebFetch | Failed on PDF binary; recovered locally. Logged so the workaround is not re-derived. |

## Coverage this pass

**SCANNED:** D1 · A2 · A4 · A5 · B2 · B9 · B12 · B13 · C1 · C3 · C4 — **11 tracks.**
**DEEP:** D1 (Band D, 7-step) · A2 · A4 · A5 · B2 · B12 · B13 — **7 tracks.**
⛔ **NOT SCANNED: A1, A3, B1, B3, B4, B5, B6, B7, B8, B10, B11, C2 — 12 tracks.**
**Stated rather than smoothed over.** This pass was **depth-first by design** (amendment §8.1 priorities 1–2): four carried full-text debts were discharged from banked PDFs and Band D was adjudicated. It meets the Band-B minimum (3 deep-reads) and the Band-C sweep, and **fails the breadth mandate for the second consecutive day** — which is now a pattern, not a budget choice, and is escalated as such in the run summary.
