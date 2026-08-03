# Research-agent cadence — DAILY (changed 2026-08-03)

**PI instruction, verbatim (2026-08-02):** *"screen our research agents for results and their queued
tasks, change their cronjob to let run them every day instead of one agent one day."*

Owner of this file: the orchestrator. Every agent prompt points here, so keep it accurate or the
agents read a lie.

---

## BEFORE — one agent per day (weekly rotation)

| agent | cron | fired |
|---|---|---|
| tools-devenv | `43 6 * * 1` | Mondays only |
| data-engineering | `43 6 * * 2` | Tuesdays only |
| architecture-inference | `43 6 * * 3` | Wednesdays only |
| benchmarks-eval | `43 6 * * 4` | Thursdays only |
| opponent-analyzer | `43 6 * * 5` | Fridays only |
| production-optimization | `43 6 * * 6` | Saturdays only |
| orchestrator | `23 14 * * 5` | Fridays only, after the others |

Each agent got **one shot per week**. The orchestrator — the only party permitted to write an INTAKE
verdict or merge an agent branch — got one shot per week too, and became the bottleneck.

## AFTER — every agent every day, staggered

| slot | agent | cron | local time |
|---|---|---|---|
| 1 | tools-devenv | `43 6 * * *` | 06:43 |
| 2 | data-engineering | `17 8 * * *` | 08:17 |
| 3 | architecture-inference | `51 9 * * *` | 09:51 |
| 4 | benchmarks-eval | `23 11 * * *` | 11:23 |
| 5 | opponent-analyzer | `49 12 * * *` | 12:49 |
| 6 | production-optimization | `29 14 * * *` | 14:29 |
| — | **orchestrator** (debt sweep) | `11 16 * * *` | 16:11 |

Unchanged, listed so the day reads end-to-end: pod-training-monitor `37 */6 * * *`, evening report
`53 17 * * *`, and the session-cron program reports at 07:57 / 12:57 / 17:57.

Times are **Europe/Berlin local** (the scheduler takes local cron, not UTC). The scheduler adds a few
minutes of deterministic jitter, so the wall-clock is a minute or two later than the table.

### Why staggered and not all at 06:43

Three reasons, each with a failure behind it:

1. **The git index is shared state.** CLAUDE.md records commit-swallowing twice in one session when
   several agents staged concurrently (`60265d3`, `3d41bd0`), and `git commit -- <pathspec>` segfaults
   on this repo, leaving a stale `.git/index.lock` that reads like contention but is debris. Six
   agents committing at the same minute makes that the normal case instead of the rare one.
2. **~1.5 h between slots** is longer than the tightened daily budget (75 min), so in the normal case
   only one agent is live at a time.
3. **Slot order preserves the old dependency chain.** The weekly prompts said "consume Monday's and
   Tuesday's outputs"; the same producers now run earlier the same morning, so tools-devenv →
   data-eng → arch-inf → bench-eval still holds, compressed from a week into a day.

---

## Rules that come with the daily cadence

1. **Scope one day, not one week.** One landed increment beats three half-finished ones. The bounded
   loop tightened from *3 iterations / 2 h / 25 searches* to **2 iterations / 75 min / 12 searches** —
   the agents now run 7× as often, so each run must be ~7× cheaper or the burn multiplies.
2. **First duty every run is your own debt**, before any new work: is your last branch merged into
   the tip? do your INTAKE packages have verdicts? Deliverables on a branch or a pod are NOT done.
3. **STATE.md LAST_RUN must advance to today.** A run that does not update STATE is invisible — on
   2026-08-03 Architecture & Inference's STATE read `2026-07-18` while the scheduler had recorded a
   run on 07-29, and Data Engineering's was 15 days stale.
4. **Two agents got a mode rule instead of a straight 7×**, because their weekly duty does not
   divide into a day:
   - **opponent-analyzer** — the outside world does not move in 24 h. Short delta-scan (one line if
     nothing changed — that is a valid result, not idling) + real backlog work; full sweep ≤ weekly.
   - **production-optimization** — its weekly duty was a compliance review **and** an optimization
     experiment. Alternate them day to day; state the mode at the top of the note.
5. **The orchestrator's daily job is the debt sweep, not a report.** ≥3 INTAKE adjudications and ≥2
   branch merges per day, oldest first. The weekly synthesis stays **Fridays only**.
6. **Gated ≠ idle.** If the top item is blocked, drop to the next unblocked one and execute it in the
   same run (CLAUDE.md, ⛔ NEVER IDLE).

---

## The debt this change is meant to burn down — measured 2026-08-03

Baseline so the trend is checkable. MEASURED, artifacts: `git branch --no-merged`, and
`Project Steering/AgentSchedule/intake_audit.py` over the 37 `INTAKE.md` files.

- **24 of 37 INTAKE packages un-adjudicated**, oldest **25 days**. 13 decided; 14 still carry the
  untouched template menu (`integrate / integrate-with-changes / defer / reject` with all four options
  listed is **not** a decision); 10 have no verdict content at all.
- **10 unmerged `agent/*` branches**, oldest 2026-07-10.
- **`main` is 128 commits behind** the working tip (`agent/benchmarks-eval-20260802`), 0 ahead.

Corroboration: the Data Engineering agent independently counted *"11 unmerged branches, 26
verdict-less INTAKEs, oldest 25 days"* on 2026-08-03 (commit `7a69d76`). Two independent counts, same
oldest-age; the 24-vs-26 spread is two packages whose verdict section holds a date but no decision.
That commit is itself on an unmerged branch — the debt list about stranded work was stranded.

**The queue grows faster than it is adjudicated — measured during this very session.** At 01:30 the
audit read **24 un-adjudicated of 37**; at 07:45 the same script read **25 of 38**. Production &
Optimization had filed `2026-08-03-thor-real-weights` in between — its second intake of the day. With
a weekly orchestrator, arrivals outrun verdicts, which is why the fix is a *daily sweep* and not
merely a bigger one.

**The orchestrator must republish these three numbers in PROJECT_STATE.md every day. They must fall.**
If after two weeks they have not, the daily cadence is not paying for itself and the change should be
reverted rather than defended.

---

## Rollback

The change is config-only and fully reversible; nothing was deleted.

1. Restore the seven prompts from `Project Steering/AgentSchedule/skills-backup-2026-08-03/`
   (verbatim copies of each `SKILL.md` as it stood before the change).
2. Restore the seven cron expressions from the BEFORE table above, via
   `mcp__scheduled-tasks__update_scheduled_task`.

To revert **one** agent to weekly (the likely outcome if a given agent produces churn), change only
that agent's cron back to its BEFORE value and restore its backed-up prompt. The slots are
independent.

## Known risk the PI should decide on

Scheduled-agent runs go from ~6/week to **49/week** — roughly **8× the API burn**. CLAUDE.md records
that uncontrolled fan-out *"exhausted the weekly API budget on 2026-07-21 and cost three agents'
work."* The tightened per-run budget (rule 1) is the mitigation, but it is a mitigation, not a proof.
If the budget binds, the cheapest correction is to keep the **orchestrator** daily — it is the one
clearing the backlog — and return the slower-moving research agents (opponent-analyzer,
production-optimization) to weekly.
