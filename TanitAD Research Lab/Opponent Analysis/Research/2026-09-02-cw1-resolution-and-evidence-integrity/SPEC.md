# E-OPP-CW1-1 — resolve correction-watch CW-1, and check the evidence base it rests on

`Research Lab · Opponent Analysis · daily pass 2026-09-02 · 0 GPU`

## Why this exists

`TanitAD Research Lab/Frontier Scan/LEDGER_A1_world_models.md:60-63`, verbatim:

> ⚠️ **CW-1 — OPEN CORRECTION-WATCH.** DriveFuture's headline navhard **55.5**
> cannot be reconciled with its own navhard ablation rows (**30.9–34.6**) from the
> text. **No DriveFuture navhard *level* may enter a comparability table until
> resolved.** The ablation *deltas* are usable (same table, same setting). Resolve
> by reading the leaderboard submission config.

CW-1 is the programme's **only** correction-watch, it fences a competitor level out
of every comparability table, and it re-appears as work in `LAB_BACKLOG.md:150`
(row P-5, 0 GPU) and `2026-08-31-LAB-RUN-004.md:104`.

## Hypotheses, both outcomes committed in advance

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-CW1-1** | The 55.5 and the 30.9–34.6 rows are the **same** configuration and genuinely irreconcilable. | The fence stands and should harden — the number is not trustworthy. | ⭐ They are **different settings**, the paper names the difference, and the fence should be narrowed to "admissible with its setting". |
| **H-CW1-2** | An independent primary reproduces DriveFuture's navhard level. | The level is externally corroborated. | It is not, and the fence stands regardless of H-CW1-1. |
| **H-OPP-INT** | The banked primaries are readable. | CW-1 is answerable from the bank, as designed. | ⛔ If a needed primary is unreadable, the finding is about the **evidence base**, and the question becomes how many others are. |

## Success criteria, committed in advance

1. ⛔ **PRIMARY SOURCES ONLY.** No aggregator, no blog, no summary. Every
   competitor number is read from a PDF we hold, and its library key is named.
2. A resolution must quote the **reconciling sentence verbatim** — "we think the
   settings differ" is not a resolution.
3. ⛔ **The Lab does not lift a fence.** A resolution is escalated as a
   recommendation with its evidence; un-fencing is a programme call.
4. If a competitor number is corrected, the **wrong** and **right** values are
   both stated with the primary's own words, so the correction is checkable.
5. Anything **not** verified is named as unverified rather than omitted.

## Method

* Read `2605.09701` (DriveFuture) and `2606.07170` (TOAD) from the banked Library.
* Cross-check every level against a **second** primary where one exists.
* Verify the fence's current observance by locating every repo site carrying a
  DriveFuture navhard number.
* ⛔ **Check the artifacts by CONTENT, not presence.** `os.path.exists` and a byte
  count are not evidence a paper is readable — that is the memmap-of-zeros class.
  `code/library_integrity.py` runs magic → `%%EOF` → parse over all 310 entries.

**Tier: N/A** — opponent-claim adjudication and artifact integrity, not a model
read. The four-metric-families rule binds *our* evals and does not bind a
competitor's published table; ⚠️ but note the programme's standing finding that
**EPDMS alone does not satisfy the four families** (D-EPDMS-FAM), which applies to
every number quoted here.

## ⛔ Scope

* No comparison between any competitor number and any TanitAD number is made or
  implied — we hold no navhard EPDMS row.
* The NAVSIM scoring-basis date fence (2025-04-28, 2025-09-29) is **unchecked**
  for these numbers; they are quoted as opponent-internal comparisons, which is
  the setting in which that is safest.
