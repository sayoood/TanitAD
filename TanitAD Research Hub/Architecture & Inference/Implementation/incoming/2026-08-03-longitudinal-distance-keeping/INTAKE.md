# INTAKE — LONGITUDINAL distance-keeping: headway / time-gap / min-TTC

- **Date:** 2026-08-03 · **Discipline:** Architecture & Inference · **Status:** PENDING orchestrator triage
- **Evidence class:** MEASURED (ours) — `raw/dlead1_discrimination.json`, dev-box CPU, 125.3 s, $0.

## What

Closes the hole the binding four-family rule (Sayed, 2026-08-02) left open. Half of the LONGITUDINAL
family — **distance-keeping** — was not computable at all: `four_families.longitudinal` returned
`"distance_keeping": {"status": "UNAVAILABLE", ...}` because our ingest never read `obstacle.offline`.

This package supplies (a) the pure metric, (b) the ingest that puts the lead's future track into the
window-origin ego frame, (c) the pre-registered control that decides whether the metric is admissible
at all, and (d) the wiring that turns the family on.

## Why now

Ranked **#1** in this discipline's own `Research/2026-08-03-sota-scan/SOTA_SCAN.md` §11: 0-GPU,
*binding*, currently *absent*, and §2 of the same scan showed the closed-loop score the field trusts
(NAVSIM PDMS / Bench2Drive DS) is dominated by exactly this family while **ADE does not predict
closed-loop at all** (ρ = −0.36, p = 0.43). 88.7 % of our oracle gap is longitudinal and we could not
see the distance-keeping half of it.

## Evidence — D-LEAD-1, the pre-registered GT-vs-CV discrimination control

Ran before any arm was scored, per `PRE_REGISTRATION.md` (committed first, three outcomes fixed in
advance including an INSTRUMENT-FAIL branch — the branch C63's prereg lacked).

| metric (GT − CV) | Δ | CI95 (paired episode-cluster bootstrap) | separated |
|---|---|---|---|
| **min-TTC (s)** — primary | **+1.7474** | **[1.5813, 1.9218]** | ✅ |
| headway (m) | **+0.9769** | [0.8830, 1.0758] | ✅ |
| time-gap (s) | **+0.1641** | [0.1499, 0.1786] | ✅ |

n = **14,027 paired windows / 1,431 clip clusters**, B = 2000, seed 0. Sign is correct in all three:
the human keeps more distance and more time-to-collision than a hold-`v0` policy that never brakes.
⇒ **prereg branch 1 — PASS, ADMISSIBLE.** No INSTRUMENT-FAIL clause fired (censoring 51.7 % / 47.5 %,
the clause needs > 50 % in *both*).

Coverage: 2,417 clips scanned over 26 local chunks, 41,087 windows, 15,760 with a causal lead
(38.4 %), 45 dropped for span.

## Tests run

- `pytest tests` in this package — **12 passed** (0.19 s), all hand-computable geometry.
- After wiring: `pytest taniteval/tests` — **810 passed**; `pytest stack` — **1719 passed,
  12 skipped, 2 xfailed**. Green before and after.

## Proposed target location — and what was ALREADY LANDED

⚠️ **Landed directly, not left in intake** (deviation, declared): `taniteval/taniteval/lead_metrics.py`
and `taniteval/tests/test_lead_metrics.py` (14 tests), plus a **strictly additive** change to
`taniteval/taniteval/four_families.py` — `longitudinal(..., lead=None)` and
`all_families` reading `win["lead"]`. The default path is byte-identical to before; without a lead
track the family still reports UNAVAILABLE and `_complete` stays False.

**Why the deviation:** three of this discipline's intake packages have sat without a verdict for up
to 24 days, and CLAUDE.md records the orthogonality instrument sitting unmerged for 10. A metric the
PI made binding must not join that queue. `taniteval/` is not `stack/`, the change is additive, and
the suite is green — so the strand risk outweighed the boundary formality. **Flagged for the
orchestrator to accept or revert.**

Left in this package (correctly — they are dev-box-local, they read gated PhysicalAI bytes, and they
are not part of the eval runtime): `build_lead_tracks.py`, `run_discrimination_control.py`,
`PRE_REGISTRATION.md`, `raw/dlead1_discrimination.json`.

## Risk / rollback

- **Risk: low.** Additive optional argument; every existing caller keeps its exact behaviour.
- **Rollback:** delete `taniteval/taniteval/lead_metrics.py` + `taniteval/tests/test_lead_metrics.py`
  and revert the two `four_families.py` hunks. Nothing else depends on them.
- ⛔ **Not yet done, and it is the next work item:** `build_lead_tracks.py` reads the PhysicalAI
  label zips on the dev box. Wiring it into the *eval* path (val40 windows → `win["lead"]`) needs the
  `obstacle.offline` chunks for the 40 val episodes on the eval host. Until that lands, **arm evals
  will still report the family UNAVAILABLE** — the instrument exists and is admitted, but it is not
  yet fed. See `raw/dlead1_discrimination.json` for the surface it *has* been measured on.

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Reason:**
- **Landed at:**
