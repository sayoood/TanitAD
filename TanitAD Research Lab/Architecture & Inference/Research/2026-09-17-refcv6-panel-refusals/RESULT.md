# Refusals 2 and 9 — the REPORT-time half of §12, and why they are not launch checks

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: dev box, CPU**
**Closes:** the two ⚠️ rows in `PREREG_REFCV6_V2.ERRATUM-1.md` §E4 that needed building rather than
rewiring.

## The correction underneath this package

§12 introduced all ten refusals as *"the launch checker exits non-zero on any of these"*. Two of
them **cannot run at launch**: a control's known value and a tactical headline only exist once an arm
has produced a panel. Calling them launch checks is part of why they were never built — there was no
place to put them. They live in `tanitad/train/panel_refusals.py` and run against the emitted panel.

## Refusal 2 — a control must read its known value EXACTLY

⛔ `==`, and no tolerance, on purpose. The record carries the reason: an all-zero control read
**0.016340208** against a base rate of **0.016197554333** — **+0.881 %** — and the gap went unnoticed
because nothing had written down what the control was supposed to read.

Three ways to fail, and **the second is the one that matters**:

| | |
|---|---|
| no controls at all | a panel that cannot distinguish a working instrument from a broken one |
| **a control with no declared `expected`** | ⛔ **REFUSED.** An undeclared expectation turns a control into a *second measurement* — whatever it reads looks like the answer |
| `observed != expected` | exactly, no tolerance |

⭐ **Why exactness rather than a tolerance:** the moment a tolerance exists, the 0.881 % case has to
argue about its size. Exactness is the property that makes the check unarguable, and a test pins that
even a **1e-18** deviation is refused.

## Refusal 9 — no pooled tactical headline, no headline under the n = 200 floor

⛔ A `pool`-shaped key anywhere in the report is refused outright. Pooling across the 22 behaviours
turns a handful of well-supported classes into an average that reads as competence on all of them.

⭐ **The distinction the refusal preserves:** a token under the floor may be **REPORTED** — it must
not **LEAD**. **10 of the 22** behaviours sit under n = 200, so a headline is otherwise free to be
built almost entirely from the least-supported classes. An AP quoted without its `n` is also refused:
a number that cannot be placed against the floor gets quoted as if it had been.

## Both are mutation-proven

| module | guards removed | verdict |
|---|---|---|
| `prelaunch_v2.py` (refusals 3, 4) | 8 | **8 KILLED, 0 escaped**, baseline green |
| `panel_refusals.py` (refusals 2, 9) | 7 | **7 KILLED, 0 escaped**, baseline green |

**26 tests** across both. Raw: `raw/audit_prelaunch_v2.json`, `raw/audit_panel_refusals.json`.

⚠️ **The audit itself gained a guard in this pass.** It now selects its removal list **by module
stem** and refuses a module it has no list for. Without that, pointing it at `panel_refusals.py`
while it still held `prelaunch_v2.py`'s patterns would have reported `PATTERN-MISSED` for every
branch — and still printed a tidy summary. ⛔ Same class as the exit-code-2 defect the first audit
had: **a check that passes for the wrong reason is indistinguishable from a check that works.**

## §12 tally now

**8 ✅ · 1 ⚠️ · 1 ⛔**

| still open | why |
|---|---|
| ⚠️ **refusal 1** — one-variable, on parsed namespaces | `check_one_variable` takes `w_agent` / `w_tac_goal`, the 2026-09-10 lever set. The v2 variable is the **backbone**. ⛔ It cannot be rewired responsibly until the v2 launch commands exist — building a diff for commands nobody has written is speculation, not a guard. |
| ⛔ **refusal 10** — a capped `goal_pos_weight` quoted without its cap sentence | a rule about **prose**. Either it becomes a check on the panel's own fields (`capped: true` + `cap_note`), or it is **demoted from a refusal to a convention**. The PI's call. |

⛔ §11's OWED item stands in narrowed form: no GPU arm launches until refusal 1 is wired to the real
v2 launch commands and 10 is built or demoted.
