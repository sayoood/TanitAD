# TanitAD Research Hub — Operating Concept (v1.0, 2026-07-05)

**Mission (from the constitution).** Post-doc-quality weekly research + real MV implementation per
discipline, autonomous, incremental, and impossible to derail — because speed is decisive and the hub
must work correctly from week 1.

## 1. Structure

```
TanitAD Research Hub/
  HUB_CONCEPT.md                  this file
  INITIAL_RESEARCH_SYNTHESIS.md   kickoff research baseline (H0–H15)
  HYPOTHESIS_LEDGER.md            living status of all hypotheses
  agents/
    _common-protocol.md           shared rules incl. the loop concept — read FIRST by every agent
    tools-devenv-agent.md         Monday
    data-engineering-agent.md     Tuesday
    architecture-inference-agent.md  Wednesday
    benchmarks-eval-agent.md      Thursday
    opponent-analyzer-agent.md    Friday (morning)
    orchestrator-agent.md         Friday (afternoon) — health check + weekly synthesis
  <Discipline>/
    Research/                     KNOWLEDGE_BASE.md + dated research notes + STATE.md
    Implementation/               MV implementation increments (code/specs land in stack/ when runnable)
```

## 2. The weekly pipeline (sequential by design)

Mon Tools&DevEnv → Tue Data Engineering → Wed Architecture & Inference → Thu Benchmarks & Eval →
Fri Opponent Analyzer → Fri Orchestrator/Synthesizer.

Each agent reads the *current week's* outputs of the agents before it (Mon output feeds Tue, etc.).
The Orchestrator closes the week: agent health check, synthesis report, KPI/benchmark progress,
proposals for the MVP stream (which Sayed steers daily — agents propose, never override).

## 3. The loop concept (inside every agent run)

Every agent run is a bounded quality loop, not a single pass:

```
for iteration in 1..MAX (default 3):
    1. RECALL   read own STATE.md + KNOWLEDGE_BASE.md + this week's upstream outputs
    2. SEARCH   fresh sources (papers, releases, news, repos, videos) since last run
    3. ANALYZE  relevance + impact on our goals/hypotheses; discard noise aggressively
    4. PRODUCE  research note + knowledge-base delta + implementation increment
    5. CRITIQUE self-review against the agent's quality gates (in its prompt file)
    6. if all gates pass -> COMMIT (files + git) and stop; else loop with the critique as input
```

Hard bounds: MAX 3 iterations, wall-clock budget per run (default 2 h), web-search budget
(default ≤ 25 searches). An agent that cannot pass its gates within bounds commits what it has,
marks `QUALITY: partial` in its STATE.md, and the Orchestrator flags it Friday.

## 4. Knowledge bases (incremental, deduplicated)

`<Discipline>/Research/KNOWLEDGE_BASE.md` is the single accumulating asset per discipline:
curated entries `[date] [source] [1-3 line finding] [impact on H_x / phase] [link]`, newest first,
pruned monthly (stale/superseded entries moved to an ARCHIVE section). Research notes
(`YYYY-MM-DD-<slug>.md`) hold the full weekly analysis; the knowledge base holds only the distilled
deltas. Hypothesis-relevant findings ALSO update `HYPOTHESIS_LEDGER.md` (status + one-line evidence).

## 5. Interfaces to the rest of the project

- Agents never edit `Mission Plan.md`; proposals go to `Project Steering/Proposals/`.
- Implementation increments that are runnable code go into `stack/` via normal commits (tests must pass).
- Benchmarks & Eval agent owns `Benchmarks & Eval/LEADERBOARD.md` and `REGULATION_TRACE.md`.
- Every agent ends its run with the session-end ritual of `CONTINUATION_PROTOCOL.md` (state update,
  commit, push) — the hub is fully resumable at any point.

## 6. Scheduling

Agents are registered as scheduled jobs (weekly, per-day, off-peak minutes) on the dev machine via the
Claude scheduled-tasks system; each job's prompt is a one-liner pointing to the agent file, so the
agent definitions stay versioned in git and editable without touching the schedule. Registered
schedule: see `agents/_common-protocol.md` §Schedule.

## 7. Health & drift control (Orchestrator duties, Friday)

1. Verify each agent ran this week and passed gates (else: diagnose, re-run or flag).
2. Aggregate research + implementation deltas into `Project Steering/Progress Reports/YYYY-Www.md`.
3. Update benchmark/KPI progress table (from `Benchmarks & Eval/LEADERBOARD.md`).
4. Detect drift: plan vs reality, resource burn vs ledger, hypothesis statuses vs evidence.
5. Produce max 3 concrete, prioritized proposals for next week's MVP stream.
