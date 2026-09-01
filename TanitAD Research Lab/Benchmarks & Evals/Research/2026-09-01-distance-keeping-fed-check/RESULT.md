<title>RESULT - the longitudinal distance-keeping family is wired but UNFED: 0 of 71 artifacts</title>

# RESULT — the distance-keeping family is admitted, wired, and fed on **zero** artifacts; all 28 that mention it carry a reason string the source itself calls stale

**Package** `E-LAB-BE-0901` · 2026-09-01 · Benchmarks & Evals · 0 GPU (CPU census)
**Serves** `LAB_BACKLOG` ranked row **19** (P1, verify-first)
**Class** `MEASURED` (ours) — census over `taniteval/results/`, `raw/dk_census.json`
**Tier stamp** n/a — an artifact census, not a model eval.

Row 19 asked: *"is the LONGITUDINAL distance-keeping family FED on the current eval path
(`win["lead"]` for the val episodes)?"* — because *"a binding family reported UNAVAILABLE while
its instrument sits idle is a criteria-completeness violation"*.

**Answer: NO, and the row's premise is confirmed — but the diagnosis needs one correction.**

---

## 1. Findings

### F1 — the instrument is present and admitted; the *refusal* was already retired `MEASURED (source)`

All three pieces exist in the main tree and are wired to each other:

| piece | path | state |
|---|---|---|
| lead registration | `taniteval/taniteval/lead_source.py` | present (`select_lead_causal`, `register_poses_to_time`, `LEAD_LAT_M 2.0`, `LEAD_MAX_GAP_M 80.0`) |
| the metrics | `taniteval/taniteval/lead_metrics.py` | present, **admitted by the pre-registered D-LEAD-1 control** |
| the consumer | `taniteval/taniteval/four_families.py` | computes `distance_keeping.mean_headway_min_m` / `mean_time_gap_min_s` / `mean_min_ttc_s` **whenever `win["lead"]` is supplied** |

`driving.py:65` records the refusal as **"REFUSAL RETIRED 2026-08-18"** and states the situation
precisely: *"These tier-0 dumps carry no `win['lead']` yet, so here it is an unwired WORK ITEM,
not an absence."*

⇒ **This is not a missing instrument and not a live refusal.** It is a supply gap between a
working producer and a working consumer.

### F2 — ⛔ the family is fed on **0 of 71** readable artifacts `MEASURED`

Census over every `*.json` in `taniteval/results/` (75 files):

| bucket | count |
|---|---|
| carries a **numeric** distance-keeping value | **0** |
| carries the key with a **non-numeric** (unavailable) value | **28** |
| carries no distance-keeping key at all | 43 |
| **could not be read** (see F4 — declared, not folded in) | **4** |

**Zero.** Not "few" — none. The family that CLAUDE.md's four-families rule calls binding, and
which owns **88.7 % of our oracle gap** (longitudinal), has never once been scored on a banked
artifact.

### F3 — ⭐ all 28 carry the **same stale reason string**, and the source already refutes it `MEASURED`

Every one of the 28 reads, identically:

> `"no lead-agent state exists (lead_state is a None stub)"`

But `driving.py:613-620` says of exactly this text:

> *"the old reason ('no lead-agent state exists') is a **STALE ABSENCE-CLAIM**: the state EXISTS
> (obstacle.offline join, lead_source registration, lead_metrics admitted by D-LEAD-1)"*

⇒ **The correction landed in the code's prose on 2026-08-18 and reached none of the artifacts.**
Anyone reading a banked result today is told the state does not exist, which the programme has
known to be false for two weeks. This is the **stale-absence-claim class** in its purest form —
the same family as the "2 of 36 features" rot that needed a pinning test, and as
`STALE_BLOCKER_SWEEP`. A prose fix cannot reach a JSON file.

⚠️ **One correction to row 19's framing.** The row says the instrument "existed unfed as of
08-03" and implies it merely needs feeding. Measured, there are **two** gaps, and only the first
is what the row describes:
1. **supply** — no dump carries `win["lead"]` (real, and the row is right about it);
2. **provenance** — every existing artifact asserts a reason the source calls stale. Re-running
   evals fixes (1); it does **not** retract the 28 artifacts already published with (2).

### F4 — ⚠️ 4 artifacts are UNREADABLE from the dev box, by two independent APIs `MEASURED`

`driving_flagship-30k.json`, `driving_flagship-nospeed.json`, `driving_flagship-speed.json`,
`driving_flagship-v16-ab-ft.json` (32–33 KB each) fail to open with `OSError [Errno 22]` via
Python, and with `MethodInvocationException` via .NET `File.ReadAllText`. They failed in **three
independent runs over ~25 minutes**, and two of them survived **8 spaced retries each**.

⛔ **They are declared UNREAD, not assumed empty.** Folding them into the "0 numeric" count would
be exactly the G:-mount trap CLAUDE.md warns about (*a G: failure count is admissible only with
zero Errno 22 in the run*). The headline is therefore **0 of 71**, with 4 outstanding.

⚠️ Note these four are the flagship tier-0 `driving.py` dumps — the ones whose own source says
they carry no `win["lead"]`. So the *expected* value is "unavailable", and the headline is very
unlikely to move. **But expectation is not measurement, and they stay declared.**

---

## 2. What this changes for TanitAD — 3 recommendations

1. ⭐ **Close the supply gap where the consumer already works.** `four_families` needs only
   `win["lead"]`. The wiring task is to have the eval path call `lead_source.select_lead_causal`
   over the val episodes' `obstacle.offline` join and attach the result per window. **No new
   instrument, no new science, no GPU** — this is the cheapest possible route to making a binding
   family stop reading UNAVAILABLE.
2. ⛔ **Do not re-publish the 28 artifacts' reason string.** When the family is fed, the stale
   text must be replaced, not merely superseded. **Pin it with a test**, the way the "2 of 36"
   count was pinned: assert that no artifact in `taniteval/results/` contains
   `"no lead-agent state exists"`. That is the only mechanism that has ever stopped this class —
   a prose correction demonstrably did not.
3. **Escalate the 4 unreadable artifacts as a data-integrity item.** Four banked eval results
   cannot be audited from the dev box. Whether that is Drive corruption or a local mount fault,
   **an unauditable artifact is not a quotable one**, and the registry may be citing them.

## 3. Deliverable manifest

| artifact | where | only place? |
|---|---|---|
| `SPEC.md` | `repo:` this package | staged |
| `code/dk_census.py` | `repo:` this package | staged |
| `raw/dk_census.json` | `repo:` this package | staged |
| `raw/search_log.md` | `repo:` this package | staged |
| `RESULT.md` (this file) | `repo:` this package | staged |

## 4. Evidence-class ledger

| claim | class | source |
|---|---|---|
| 75 files, 71 readable, 0 numeric, 28 unavailable, 43 no-key | MEASURED (ours) | `raw/dk_census.json` |
| the single reason string, 28× identical | MEASURED (ours) | same |
| 4 files unreadable via 2 independent APIs, 3 runs | MEASURED (ours) | same + `raw/search_log.md` |
| instrument present/admitted; refusal retired 2026-08-18 | MEASURED (source read) | `driving.py:65,613-620`, `four_families.py`, `lead_source.py`, `lead_metrics.py` |
| "88.7 % of our oracle gap is longitudinal" | INHERITED (CLAUDE.md four-families rule, **not re-verified here**) | `CLAUDE.md` |
| D-LEAD-1 admitted `lead_metrics` | INHERITED (source comment, not re-run) | `driving.py:65` |
