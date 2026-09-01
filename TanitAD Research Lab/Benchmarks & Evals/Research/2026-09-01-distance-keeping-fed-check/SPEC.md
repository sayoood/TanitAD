<title>SPEC - is the distance-keeping family fed? (row 19, verify-first)</title>

# SPEC - E-LAB-BE-0901: is the LONGITUDINAL distance-keeping family FED on the current eval path?

**Trigger.** `LAB_BACKLOG` ranked row **19** (P1, **verify-first**): *"is the LONGITUDINAL
distance-keeping family FED on the current eval path (`win["lead"]` for the val episodes)? The
instrument (lead_metrics.py, D-LEAD-1-admitted) existed unfed as of 08-03."* Rationale on the
row: *"a binding family reported UNAVAILABLE while its instrument sits idle is a
criteria-completeness violation."*

**Why this row and not row 12 (D-EPDMS-FAM), which ranks higher.** Row 12's deliverable is the
decision-quality companion for **EPDMS rows**, and the EPDMS benchmark portfolio is row 3 —
`⛔PI`, still OPEN. Row 12's consumer therefore does not exist yet. Row 19 is fully unblocked and
changes what our **current** artifacts report, so it is the top *actionable* B&E row today.
⚠️ Stated explicitly because "top unblocked" was a judgement call, not a mechanical read.

**verify-first** means: the source predates 2026-08-22 and closure elsewhere is plausible.
Probe before working it — the answer may be "already closed", and that is a valid, valuable result.

## Hypotheses, both outcomes committed IN ADVANCE

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| H-DK-1 | The family is still UNFED on the eval path | the row stands; identify the exact gap and the cheapest wiring | **already closed** since 08-03 -> mark the row ✅ DONE and say what closed it |
| H-DK-2 | If unfed, the cause is a MISSING INSTRUMENT | build/port the instrument | the instrument exists and the gap is **supply** (`win["lead"]` never attached) -> a wiring task, not a science task |

**Pre-committed criterion.** H-DK-1 -> outcome B iff **any** banked artifact in
`taniteval/results/` carries a NUMERIC `distance_keeping.*` value. One is enough to close it.

## Method

1. **Source read** — does the instrument exist, is it admitted, and is a consumer wired to it?
   (`lead_source.py`, `lead_metrics.py`, `four_families.py`, `driving.py`.)
2. **Artifact census** — bucket every `taniteval/results/*.json` into
   NUMERIC / KEY-BUT-NON-NUMERIC / NO-KEY / **UNREADABLE**, and record the distinct
   unavailability reason strings.

CPU only. No GPU, no pod, no Thor.

## ⛔ The unreadable bucket is a first-class outcome, decided before the run

CLAUDE.md: the G: mount *"FORGES TEST FAILURES ... a G: failure count is admissible ONLY with
zero Errno 22 in the run"*. So:

- Counts are reported **over readable files**, with the readable denominator stated.
- Unreadable files are **NAMED and declared UNKNOWN** — never counted as "absent", and never
  silently dropped, because collapsing UNKNOWN into ABSENT lets a mount fault manufacture a
  cleaner result than the evidence supports.
- Any file that fails is retried, and a persistent failure is confirmed through a **second,
  independent API** before being called unreadable (absence found at one location is not absence).
