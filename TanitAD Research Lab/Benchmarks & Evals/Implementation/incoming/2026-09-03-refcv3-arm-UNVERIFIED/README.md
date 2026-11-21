# refcv3 T1 adapter — RESCUED, UNVERIFIED, NOT INTEGRATED

**Status: 🔶 QUEUED FOR TRIAGE. Do not import it, do not quote a number from it.**

## What this is

`refcv3_arm.py` (1,746 lines) is a partial HIER T1 adapter for refcv3, written on 2026-09-03
by a Benchmarks & Evals agent that was **killed mid-task by a model rate limit** before it
tested, documented or staged anything. The file existed **only** in the working mirror
`C:\Users\Admin\tanitad-wt\taniteval\tools\` — one disk, no backup — and was rescued into the
repo by the Master Mind at 09:20 Berlin under the anti-stranding rule (BACKLOG R14).

It is filed under `Implementation/incoming/` and **not** under `taniteval/tools/` on purpose:
placing it beside the working adapters would make an untested artifact look like a shipped
instrument, which is exactly how an inadmissible number reaches a report.

## What is missing (all of it declared, none of it done)

| the brief asked for | state |
|---|---|
| `taniteval/tools/refcv3_arm.py` | present, parses, **never executed** |
| `taniteval/tools/REFCV3_ARM.md` (the T1 contract it mirrors, §1–§3) | **absent** |
| `stack/tests/test_refcv3_arm.py` | **absent** — nothing pins any behaviour |
| the T1 definition for a supervised trajectory model, derived with file:line | **absent** — this was the deliverable the Master Mind had to review BEFORE any number could be quoted |
| a run against a real checkpoint | **never attempted** (refcv3 is still training) |

## ⛔ Known defect it carries

`recorded_controls` (≈ line 477) replays the recorded channel through the unicycle integrator
and calls it `kappa`, and its docstring asserts *"the MEASURED true-kappa channel"*. **That is
wrong.** `physicalai.py:620` writes `arctan(wheelbase · curvature)` — a road-wheel **steer
angle** — into `actions[:, 0]`. Integrating it as a curvature over-rotates by the wheelbase
factor; on refav1's eval slice the same defect cost 0.716 m of lateral error against a 0.053 m
floor. See register rows `C-STEER-CURVATURE-INTERFACE` and `D-STEER-INTERFACE-RESOLVED`, and
the bridge now available in `stack/tanitad/models/kinematic.py`
(`as_curvature` / `as_command`, `STEER_WHEELBASE_M`).

Whoever picks this up: the conversion is a **call-site** decision, because a planner candidate
(curvature) and a recorded action (steer) reach the same integrator in different units. Follow
`refav1_arm.py --action-units` rather than inventing a second convention.

## Why refcv3 still needs this

refcv3 has **no admissible T1 number and no instrument that can produce one** (register
`D-V7-READINESS-2026-09-02` §D). Its in-training eval is a T0 loss on 160 fixed windows and
must never be quoted as driving performance. The v7 dominance claim needs refcv3 read at T1
with the four families, the hold-action control, the constant-velocity control (`ha0`) and the
nav-shuffle control — what `refav1_arm.py` already does for refav1.

**Recommended next step:** do not resume this file blind. Re-derive the T1 definition first
(the killed agent never wrote it down), then decide whether to finish this draft or restart
from the current `refav1_arm.py`, which has since gained `ha0`, the trivial-profile instrument
and `--action-units` — none of which this draft knows about.
