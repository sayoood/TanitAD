# R5 — Agentic Architecture, Harness Engineering, Token Efficiency & AI-That-Engineers-AI

**Reviewer role:** independent agentic-systems / MLOps-for-agents reviewer (read-only, static repo + web).
**Scope:** the programme's own agent architecture and harness — not model architecture, not strategy
(see the 2026-07-25 `R5_strategy_research_management.md` for that), not general workflow/ops (see
`R6_workflow_automation_ops_docs.md`, which this stream explicitly extends). **Date:** 2026-09-25.
Repo at `467ce8a` (last commit 2026-08-04). No pods, no compute, no torch — static review.

**Evidence classes** used throughout, per programme convention: **MEASURED** (read in this repo/session,
artifact named) · **PUBLISHED** (a citable external source) · **INHERITED** (quoted from a programme doc,
not independently re-verified by me) · **ESTIMATED** (arithmetic shown) · **HYPOTHESIS**.

---

## 1. Headline — ranked findings

| # | Finding | Evidence |
|---|---|---|
| **1** | **Zero mechanical enforcement of any CLAUDE.md rule.** `.claude/settings.json` and `.claude/settings.local.json` contain no `"hooks"` key at all; there is no `.pre-commit-config.yaml` and no `.github/workflows/`. Claude Code ships **34 hook events**, including `Stop` (exit 2 *prevents Claude from stopping*), `SessionEnd`, and `PreToolUse` (can deny/rewrite a specific `Bash` command) — a near-exact mechanical match for "never end a turn having only reported," "safe_commit.py is the only sanctioned commit path," and "run session_guard.py at session end." None is wired up; every enforcement mechanism in the repo is prose the agent must remember to apply. | MEASURED (`.claude/settings.json` full read; `find` for pre-commit/workflows configs, empty) + PUBLISHED (`code.claude.com/docs/en/hooks`) |
| **2** | **The resource ledger tracks GPU-$ only — there is no ledger, log, or line-item anywhere for LLM/API token spend**, even though the repo's own retraction history says an "uncontrolled fan-out exhausted the weekly API budget" (`CLAUDE.md:372`) and a later redesign doc explicitly flags "~8× the API burn... **the PI should decide on**" as an *unresolved* risk. The resource that actually blew a budget is the one resource nothing measures. | MEASURED (`Project Steering/RESOURCE_LEDGER.md`, full read — 2 GPU-pod rows, nothing else; `Project Steering/AgentSchedule/DAILY_ROTATION.md:126-131`) |
| **3** | **The prior review's own automation proposals are a natural experiment in what gets built:** of R6's 10 ranked proposals (2026-07-25), exactly **3 were built — `safe_commit.py`, `registry_lint.py`, `repo_janitor.py`** — all three first committed on the **same day**, 2026-08-02 (one week after the review), and all three built to spec or beyond it (pointer sidecars, `--gap` tolerant retracted-claim sweeps, future-dated-item detection). The other 7 (`ckpt_relay.py`, `results_ledger.py`, a generated `GATES.md`, the `LOOP_STATE.md` split, report-cadence consolidation, a brief/report linter, a fan-out governor) were never built. | MEASURED (`git log --diff-filter=A --format=%ad -- tools/safe_commit.py tools/registry_lint.py tools/repo_janitor.py` → all `2026-08-02`; `find` for the other 6 tool names, empty) |
| **4** | **`LOOP_STATE.md` — flagged as a liability by *both* prior reviews — grew instead of shrinking.** R5(07-25) measured "472 lines / ~58k tokens... now itself a liability"; R6(07-25) measured "122 KB run-on." By the last commit (08-04) it is **1,493 lines / 202,832 bytes** (≈+66% by size in 9 days). Its own top pointer — *"CURRENT STATE… everything below this block is OLDER, read this first"* — is demonstrably false: live, load-bearing entries (pod2 termination, a third trainer death) sit **~1,300 lines below** that pointer, dated up to 6 days later. | MEASURED (`wc -lc`, `grep -n "^#"`, direct read of lines 1–50 and 1312–1493) |
| **5** | **The programme's own audit says the orchestrator/relay layer, not the subagents, is the dominant error source.** `BOOST_PROGRAM.md`'s 2026-07-26 scorecard rates every *mechanized* measure "✅ working" (evidence tiers, instrument certification, mid-run gates) but scores **M2 "no unverified premise in briefs" as "🔴 FAILING — AT MY LAYER,"** citing four same-day errors, all the orchestrator's own — and names the mechanism precisely (**M12**): *"a number crossing an agent boundary carries its surface, or it does not cross… **M12 binds relay, which is where the leak actually is.**"* | MEASURED (`Project Steering/BOOST_PROGRAM.md:430-444, 487-495`) |
| **6** | **The blanket "≥5 parallel streams, ALWAYS" rule (`CLAUDE.md:347`) sits in tension with Anthropic's own published multi-agent guidance**, which explicitly names "high interdependency tasks (e.g., **most coding tasks**)" and "domains requiring shared context" as cases to *avoid* multi-agent fan-out — precisely TanitAD's `stack/` + shared-git-index situation. The incident record shows the cost of not scoping it: a **215-agent / 12.9M-token** research burst (`LOOP_STATE.md:46`, 2026-07-29) landed **4 days after** a smaller burst had already "exhausted the weekly API budget" (`CLAUDE.md:372`), and no fan-out governor was ever built (Finding 3). | MEASURED (`LOOP_STATE.md:46-48`) + PUBLISHED (`anthropic.com/engineering/multi-agent-research-system`) |
| **7** | **The gate/evidence discipline is independently validated by concurrent (2026) frontier findings on LLM reward-hacking in autonomous research** — a same-class arXiv survey (2609.28614) measures a **30.5% spontaneous reward-hacking rate** on open-ended research-pipeline tasks and **74.6%** confirmed hacking when a pass threshold exceeds baseline, with naive LLM-panel review missing **6.5%** of hacks even while explicitly looking for them. `GATE_PROTOCOL.md`'s "VOID SECONDARIES" and "PRIVILEGED-INPUT PRIMARIES" rules (§0.7–0.8) are hand-built countermeasures to exactly this class, discovered the hard way (a route-CE-by-construction bug) *before* this literature existed — this is real, externally-corroborated moat, not process theatre. | PUBLISHED (arXiv:2609.28614) + MEASURED (`Project Steering/GATE_PROTOCOL.md:117-206`) |
| **8** | **Commit and report cadence is boom-bust, and the 7-week gap is total, not partial.** The entire visible git history is **149 commits on 9 calendar days** between 07-13 and 08-04 (49 of them on 08-03 alone), then **52 days of silence** to 2026-09-25. The 3×/day `program-report` cadence (`Project Steering/Reports/`) and the 1-day-old daily research-agent cadence (`DAILY_ROTATION.md`, effective 08-03) both stop at the same boundary, with no handoff note explaining the gap. | MEASURED (`git log --format='%ad' --date=short \| sort \| uniq -c`; `Project Steering/Reports/` listing, last entry `2026-08-03-1300`) |
| **9** | **No model-routing policy exists anywhere in the repo.** Neither `_common-protocol.md` nor any discipline `SKILL.md`/agent file names which Claude model the loop, the orchestrator, or a subagent should run on, or at what effort level — in contrast to this very review, whose 9 streams were each assigned an explicit model tier by the orchestrator. Model choice for the *production* loop is an unrecorded, unverifiable fact. | MEASURED-absence (grep across `TanitAD Research Hub/agents/*.md`, `.claude/skills/*/SKILL.md`, `Project Steering/AgentSchedule/*`) |
| **10** | **Two protocol docs disagree with each other on the very budget numbers they exist to fix — inside the anti-drift machinery itself.** `_common-protocol.md:12` states the pre-08-03 per-run budget was **"4 iterations / 4 h / 25 searches"**; `DAILY_ROTATION.md:63` (same day, same subject) states it was **"3 iterations / 2 h / 25 searches."** `registry_lint.py`'s retracted-claim/pointer-drift sweep does not cover either file — its scan set is `MODEL_REGISTRY.md`-and-similar only. | MEASURED (both files, cited lines) |

---

## 2. Q1 — The agentic architecture, mapped

### 2.1 Roles

```
PI (Sayed) ── provision / spend / publish / strategy+tactics decisions (D-018)
   │            escalation format: "DECISION NEEDED: X | proposal | default if silent"
   ▼
Orchestrator ── one identity, several hats: the cron "drumbeat" reader of LOOP_STATE.md,
   │            the daily 16:11 INTAKE/merge sweep, the 3×/day program-report author,
   │            and — per BOOST_PROGRAM's own audit — the empirical #1 source of
   │            unverified-premise errors (Finding 5).
   ▼
6 discipline agents (tools-devenv · data-engineering · architecture-inference ·
   │  benchmarks-eval · opponent-analyzer · production-optimization), DAILY since
   │  2026-08-03 (was weekly), staggered 06:43→14:29 Berlin, each in its own git
   │  worktree, each bound to a 2-iteration / 75-minute / 12-search loop.
   ▼
Ad hoc subagent fan-out — research bursts (the 07-29 215-agent/12.9M-token event),
   adversarial-verification passes (the 08-03 "12 agents, 6 streams" pass that
   produced 36 of the 46 dated RETRACTION_LOG entries in one day), and reviews
   like this one.
```

### 2.2 Loop / cron drumbeat

`LOOP_STATE.md:3-4` states its own design intent precisely: *"Rewriting the cron every time reality
moved was costing a re-derivation every 30 minutes and shipped stale instructions twice in one day.
Update this file instead."* This is a sound idea — a live document instead of a frozen prompt — that
was not paired with a size discipline, and Finding 4 is the direct consequence: the fix for "stale
instructions" became, itself, an instrument that ships stale instructions (a pointer 1,300 lines out
of date). Three Claude Code **skills** canonicalize the highest-frequency runbooks instead of having
the drumbeat re-derive them each time — a good pattern: `fleet-status` (delegates to
`tools/fleet_probe.py`, explicitly built to fix a *"grep matched nothing, printed nothing, scored as
healthy — four times"* failure class), `gate-eval`, and `program-report` (the 07:57/12:57/17:57
cadence, dead since 08-03).

### 2.3 Handoff / memory state

`CONTINUATION_PROTOCOL.md` defines a clean **three-layer model**: *constitution* (`Mission Plan.md`,
PI-only) → *state* (`PROJECT_STATE.md` + per-discipline `STATE.md`, overwritten every session) →
*history* (`DECISIONS.md`, git log, Progress Reports, append-only). In practice there is a **fourth,
unofficial layer** — `LOOP_STATE.md` — that carries the highest-frequency, highest-consequence state
(live fleet status, PI decisions pending, active-stream table) and is not named in the memory model
at all, so it inherits none of that model's discipline (no "overwritten, not appended" rule; no size
bound).

### 2.4 Guardrails, layered

1. **Prose / doctrine** — `CLAUDE.md` (always-loaded), `AGENT_OPERATING_STANDARD.md`'s verbatim-paste
   preamble, `BOOST_PROGRAM.md`'s M1–M13 measures, `GATE_PROTOCOL.md`'s standing rules.
2. **Statistical/decision code** — `stack/scripts/run_gate.py` (2,196 lines), `taniteval/ci.py`,
   `stack/scripts/gate_emitters.py` (521 lines) — genuinely rare: a gate that *refuses* to render a
   verdict from a train-log slope, an unnamed estimator, or a bare exponent.
3. **Mechanical tools, invoked by convention** — `tools/{safe_commit,registry_lint,repo_janitor,
   session_guard,ci_gate,fleet_probe,gpu_tripwire,pod_git_drift}.py` — all well-built (5,383 combined
   lines across `tools/`), **none wired to a hook, a CI job, or a pre-commit hook** (Finding 1). They
   work exactly as well as an agent's memory to run them.

### 2.5 Doc system

`TanitAD Research Hub/<Discipline>/{GOALS,BACKLOG,STATE,KNOWLEDGE_BASE}.md` +
`Research/*.md` + `Implementation/incoming/<date>-<slug>/INTAKE.md` per discipline; `Project Steering/`
for programme governance (registry, retraction log, gates, protocols, resource ledger, the reviews
tree). Three report series were tried; as of the last commit, **all three are dead**: `Daily Reports/`
(last `2026-07-20`), `Progress Reports/` (last `2026-W33`), `Reports/` (the 3×/day series, last
`2026-08-03 13:00`).

### 2.6 Subagent contract

`TanitAD Research Hub/agents/_common-protocol.md` (220 lines) is the actual contract every discipline
agent runs under: mandatory session-start reading order, a bounded **RECALL → SEARCH → ANALYZE →
PRODUCE → CRITIQUE** loop (2 iterations / 75 min / 12 searches since 2026-08-03, down from a figure
the protocol itself states inconsistently — Finding 10), mandatory git-worktree isolation, an
INTAKE.md hand-off for anything destined for `stack/`, and 9 lettered quality gates (G-A…G-I) of which
**G-F is the only one with a machine check** (`tools/session_guard.py`).

### 2.7 Context quantified

| Doc | Size | ≈Tokens (ESTIMATED, 4 chars/tok) | Read by |
|---|---|---|---|
| `CLAUDE.md` | 30,096 B / 411 lines | ~7,524 | **Every session, always** |
| `PROJECT_STATE.md` | 101,754 B / 395 lines | ~25,438 | Every session, step 1 of the start ritual |
| `LOOP_STATE.md` | 202,832 B / 1,493 lines | ~50,708 | The orchestrator/drumbeat, and growing |
| `RETRACTION_LOG.md` | 327,234 B / 3,149 lines | ~81,808 | "Before asserting in a known class" (CLAUDE.md, i.e. situationally, not wholesale) |
| `MODEL_REGISTRY.md` | 214,514 B / 2,222 lines | ~53,628 | Whenever a model fact is quoted |

The five core steering docs alone total **876,430 bytes (~219,000 tokens ESTIMATED)** before a single
discipline's own `STATE.md`/`KNOWLEDGE_BASE.md`/`BACKLOG.md` is opened. Only `CLAUDE.md` and
`PROJECT_STATE.md` are mandated reading on every session; `LOOP_STATE.md` is read by the
orchestrator identity at drumbeat frequency (nominally every ~30 min per its own text), which is
where Finding 4's growth becomes a recurring, compounding token tax rather than a one-off cost (§5.2
gives the worked estimate).

### 2.8 Doc growth, commit cadence, the 7-week gap

`LOOP_STATE.md` grew **122 KB → 203 KB in 9 days** (R6 measured the former on 07-25; this review
measured the latter on 08-04's HEAD) — roughly **+9 KB/day compounding**, unbounded. Git history is
**149 commits compressed into 9 active calendar days** (07-13, 07-17, 07-18, 07-20, 07-21, 07-29,
08-02, 08-03, 08-04), peaking at **49 commits on 08-03** and **36 RETRACTION_LOG entries that same
day** (the "12 agents, 6 streams" adversarial-verification pass, `LOOP_STATE.md:1376`-adjacent). Then:
**52 days of silence** to 2026-09-25, with no HANDOFF block, no closing report, and no `DECISIONS.md`
entry marking a pause — the very artifact `CONTINUATION_PROTOCOL.md §3.6` prescribes for exactly this
situation ("if work is half-done: leave a HANDOFF block") does not appear to have been written before
the gap opened (UNVERIFIED — I did not find one; a HANDOFF block, if it exists, would be at the top of
`PROJECT_STATE.md` or a discipline `STATE.md`, and I did not do an exhaustive per-discipline sweep for
one — see §6 Open Questions).

---

## 3. Q2 — Failure modes and what could be mechanized

### 3.1 RETRACTION_LOG root-cause classes, tallied

The canonical class table (`RETRACTION_LOG.md:11-21`) defines **7** classes (C1–C6, C15). The document
body actively defines and cites **at least 5 more** with the same "Standing consequence" treatment —
**C9** (horizon-blind instrument, :145), **C10** (evaluator ≠ pre-registration, :138), **C13**
(a guard that cannot fail, referenced :135), **C14** (grid end ≠ measured limit, :132), **C16**
(fabricating intermediary, :160) — none of which made it back into the top table. **The ranked-classes
summary at the top of the document that exists to prevent stale headlines is itself a stale
header** — a small, low-stakes instance of exactly the disease the document is for.

Mention-frequency tally (grep count, a proxy for "how often the class is invoked to explain a new
incident," not a clean per-incident count — MEASURED, methodology stated):

| Class | Mentions | Class | Mentions |
|---|---|---|---|
| **C4** inherited w/o re-verification | **22** | C13 guard cannot fail | 14 |
| **C5** scalar off noisy curve / bare exponent | **19** | C14 grid end ≠ limit | 7 |
| **C6** confounded comparison | **18** | C8 Glob-truncation/worktree sprawl | 5 |
| **C3** mechanism instead of measurement | **16** | C15/C16/C17 | 4 each |
| C2 absence from one probe | 14 | C10/C11/C12/C18 | ≤3 each |
| C1 faster-moving source | 12 | C9 horizon-blind instrument | 9 |

**Top-4 by recurrence — C4, C5, C6, C3 — are exactly the four the programme's own doctrine repeats
most (CLAUDE.md's exponent/interval/evidence-class rules target C5 and C4 directly).** The doctrine is
correctly aimed at its own biggest problem; the gap is not diagnosis, it is that diagnosis stays prose.

**46 dated `## ` entries total; 36 of them (78%) are dated 2026-08-03** — one adversarial-verification
push produced more corrections than the prior three weeks combined. Read charitably, this is the
"≥5 parallel streams" doctrine working exactly as intended (independent verification surfacing real
defects); read soberly, it also means the density of wrong claims sitting unexamined *before* such a
push is high, and the push is not a standing cadence — it happened once.

### 3.2 What Claude Code can mechanically enforce today

`code.claude.com/docs/en/hooks` (PUBLISHED, fetched this session) documents **34 hook events** across
per-session, per-turn, tool-use, agent/task, context/state and model/display cadences, five hook
`type`s (`command`, `http`, `mcp_tool`, `prompt`, `agent`), and a decision model (`permissionDecision:
allow/deny/ask`) that a `command`-type hook can return. Mapped onto this repo's own repeated-four-times
and repeated-twice rules:

| Prose rule (repeated N× because it wasn't mechanical) | Hook | Mechanism |
|---|---|---|
| "Never end a turn having only reported" (`CLAUDE.md:235,260` — **flagged 4×**) | `Stop` | Exit 2 *prevents Claude from stopping*; a script checks whether this turn's transcript contains a mutating tool call (Edit/Write/a side-effecting Bash command) and blocks a report-only stop |
| "`safe_commit.py` is the only sanctioned commit path" (`tools/README.md`) | `PreToolUse` on `Bash(git commit *)` | Deny raw `git commit`, `permissionDecisionReason` tells the model to invoke `tools/safe_commit.py` instead — I confirmed this exact JSON shape is a documented pattern |
| "Run `session_guard.py` / `registry_lint.py` before ending / before quoting" | `SessionEnd` / `PreToolUse` on registry edits | Both scripts already exit non-zero on a real problem; today nothing calls them automatically |
| "END WITH A DELIVERABLE MANIFEST" (`AGENT_OPERATING_STANDARD.md` rule 2) | `SubagentStop` | Grep the subagent's final report for a manifest-shaped table before hand-back; block/flag if absent |
| Evidence-class tagging on every number (`CLAUDE.md:180`) / four-metric-family completeness (`CLAUDE.md:380`) | `type: "prompt"` hook at `Stop`/`SubagentStop` | A cheap-model single-turn check — *"does this report tag its numeric claims with an evidence class / include all four metric families?"* — is literally the "evidence-class checker" and "ADE-only report detector" the brief asks whether to build; the mechanism already exists, unused |
| Fan-out cap ("5–8 concurrent", `CLAUDE.md:372`) | `SubagentStart` + a semaphore file | Deny/queue a spawn past N concurrent — R6's P10, never built |
| `LOOP_STATE.md` size cap (R6's P7, never built) | `PostToolUse` on `Edit`/`Write` matching that path | `wc -l`, warn or block past a stated cap |
| Pod-drift ("verify by import before every launch", `CLAUDE.md`) | `PreToolUse` on an `ssh … train_*` launch pattern | Require `pod_git_drift.py`'s exit code first |

None of these require new tools to be *written* in most rows — `session_guard.py`, `registry_lint.py`,
`safe_commit.py`, `pod_git_drift.py` already exist and already have the right exit-code contract. The
gap is entirely in the **last mile**: nothing in `.claude/settings.json` calls them. This is the
single cheapest, highest-leverage change available to the programme (§5, R1).

One structural reason the gap is easy to miss: `.claude/settings.json`'s `defaultMode: "acceptEdits"`
plus a broad `Bash(git *)`/`Bash(ssh *)`/`Bash(python *)` wildcard allowlist means almost nothing ever
reaches a **permission prompt** for a human to catch — the harness's other native safety net
(ask-before-running) has been deliberately widened away for velocity, which makes hooks not just an
opportunity but the *only* remaining native lever, since permission prompts were traded away.

---

## 4. Q3 — Keep / automate / delete

**Keep (high-value, evidence-backed, do not touch):**
- `GATE_PROTOCOL.md` + `run_gate.py` — the horizon rule, void-secondary rule, and privileged-input
  rule are externally validated by 2026 reward-hacking literature (§1 Finding 7). This is the
  programme's actual moat.
- `taniteval/ci.py`'s episode-cluster bootstrap; the evidence-class **and** evidence-tier (M1)
  doctrine; the RETRACTION_LOG root-cause-class habit itself (the tool, not the stale summary table).
- `registry_lint.py`'s **pointer** pattern (`<!-- src: path#field -->`) — cite the artifact, don't
  copy the number. 14 pointers exist in `MODEL_REGISTRY.md` today; this is the one place "generated,
  not hand-typed" (R6's P5 goal) was actually achieved, and only partially.
- `safe_commit.py`, `repo_janitor.py`, `session_guard.py`, `ci_gate.py`, `fleet_probe.py` — all
  well-built, all evidence-driven, all just missing a trigger (§3.2).
- The 2026-08-03 daily-cadence budget cut (2 iter / 75 min / 12 searches) — a rare, correct instance
  of the programme right-sizing cost against frequency *before* being burned, citing its own reasoning
  (`_common-protocol.md:12-17`: "you now run 7× as often, so an unchanged budget would multiply the
  burn by ~7").
- Retiring a measure **in place with a strikethrough** rather than deleting it (`BOOST_PROGRAM.md`'s
  M5-bis) — good provenance hygiene, worth using more, e.g. for the two dead report series (below).

**Automate next (proposal exists or is trivial; highest payoff/effort):**
1. Wire the four hooks in §3.2's table (`Stop`, `SessionEnd`, `PreToolUse` on `git commit`/registry
   edits, `SubagentStop`) — the scripts exist; effort is a `settings.json` change plus ~4 thin wrapper
   scripts.
2. Extend `registry_lint.py`'s retracted-claim/pointer-drift sweep to `LOOP_STATE.md`,
   `BOOST_PROGRAM.md`, and the `AgentSchedule` protocol docs — it would have caught Finding 10 (the
   "4 iter/4h" vs "3 iter/2h" self-contradiction) and would generalize the C4-inside-the-anti-C4-tool
   irony away.
3. Instrument LLM/API token spend into `RESOURCE_LEDGER.md` (or a sibling `tools/api_ledger.py`) —
   closes Finding 2. Without it, "~8× the API burn" (`DAILY_ROTATION.md:126`) can never become a
   MEASURED number, only an ESTIMATED one (see §5.2).
4. R6's still-unbuilt P4 (`ckpt_relay.py`), P5 (`results_ledger.py` / generated `GATES.md`), and P7
   (the `LOOP_STATE.md` live/archive split) remain exactly as valuable as they were on 07-25 — nothing
   in this pass changes that prioritization, it only confirms none of the three landed.
5. A fan-out governor (semaphore file + a `SubagentStart` hook) — Finding 6's 215-agent event is the
   concrete failure this would have prevented; today's `_common-protocol.md` budget caps *duration*
   per agent, not *concurrency* across agents.

**Delete / retire / consolidate (tax, low marginal value):**
- The two abandoned report series (`Daily Reports/` since 07-20, `Progress Reports/` since W33) —
  R6 already recommended this (P8); still not done. Retire with a one-line pointer to `Reports/`
  (itself now also dead — see below) rather than leaving three silently-abandoned directories for the
  next reader to have to independently discover are dead.
- The 3×/day `Reports/` cadence, now also dead 52 days — worth an explicit decision (revive at a lower
  frequency, or retire formally) rather than an unexplained silence.
- `settings.local.json`'s ~30 exact-string historical-command allowlist entries (e.g. one-off literal
  `git commit -q -m '...'` strings from specific past incidents) — dead weight once the `Bash(git *)`
  wildcard exists in `settings.json`; they don't generalize to a new commit message and just add
  reconciliation noise. A `repo_janitor.py`-style periodic prune is cheap.
- The RETRACTION_LOG top summary table's staleness (§3.1) — either fold C8/C9/C10/C13/C14/C16/C17
  back in, or replace the hand-maintained table with a 20-line generated tally (the grep in §3.1 is
  essentially the whole script).

---

## 5. Q4 — Token efficiency & model routing

### 5.1 Routing policy (proposed — none exists today, Finding 9)

Current pricing (PUBLISHED, Anthropic `claude-api` skill, cached 2026-06-24 — 3 months old as of this
report; I did not re-verify against a live fetch, flagged UNVERIFIED-recency):

| Model | ID | Context | $/MTok in / out |
|---|---|---|---|
| Haiku 4.5 | `claude-haiku-4-5` | 200K | $1 / $5 |
| Sonnet 5 | `claude-sonnet-5` | 1M | $2 / $10 |
| Opus 5 | `claude-opus-5` | 1M | $5 / $25 |
| Opus 5.5 (launching) | `claude-opus-5-5` | 1M | $4 / $20 |
| Fable 5.1 (most capable) | `claude-fable-5-1` | 1M | $10 / $50 |

| Role | Model | Effort | Why |
|---|---|---|---|
| Extraction/lint triage — parsing tool output, INTAKE audits, "did this file change" | **Haiku 4.5** | low | Mechanical; the judgment already lives in the tool (`registry_lint.py`, `fleet_probe.py`), the model narrates |
| Discipline-agent implementation & literature screens (the current daily loop) | **Sonnet 5** | medium–high | Matches this stream's own assigned tier; RE-Bench evidence (§6) supports agents at bounded, ≤1-day tasks |
| **Orchestrator relay / brief-writing** | **Opus tier (Opus 5 or 5.5), not Sonnet** | high | Finding 5: this is the programme's *own measured* highest-error layer (M2/M12), not the subagents — routing the cheapest model to the highest-error step is backwards |
| Architecture/causal judgment — gate verdicts, retraction root-causing, hierarchy decisions | **Opus 5 / 5.5** | high–xhigh | Low error tolerance; "why," not "what" |
| High-stakes claims crossing to the PI or external ("this decides a GPU-day") | **Fable 5.1** | max, sparingly | Matches CLAUDE.md's own rule that a GPU-day decision must be MEASURED/PUBLISHED, never INHERITED — reserve the most expensive model for the highest-consequence relay, not for volume |

### 5.2 Context budgets, caching, worked cost estimates (ESTIMATED — arithmetic shown)

**LOOP_STATE.md re-read tax.** At Sonnet 5 ($2/MTok in), one read of `CLAUDE.md` + `LOOP_STATE.md`
uncached ≈ (7,524 + 50,708)/1e6 × $2 ≈ **$0.116**. `LOOP_STATE.md:3-4` says the drumbeat previously
re-derived state "every 30 minutes" before this file existed; even a conservative 10 firings/day →
**~$1.16/day (~$35/mo)** for this one file, *before any work token*, and the file was growing ~9 KB/day
(§2.8) — the tax compounds daily. Splitting it per R6's P7 into a ~150-line capped live header
(~1,500 tokens — matching Anthropic's own published guidance that a condensed sub-agent summary should
run **1,000–2,000 tokens**, PUBLISHED, `anthropic.com/engineering/effective-context-engineering-for-ai-agents`)
plus an archive read only on demand: 10×1,500/1e6×$2 ≈ **$0.03/day** — a **~38× reduction**, and one
that stops compounding. Prompt caching (not evaluated here — UNVERIFIED whether any cache
`cache_control` breakpoints are used anywhere in this loop; nothing in the repo's tooling references
it) would cut the *repeat-read-within-session* cost further but does not fix the *growth* problem —
caching a 203 KB file cheaply is not a substitute for capping it.

**The 8× fan-out redesign, priced.** `DAILY_ROTATION.md:126` computes the run-count multiplier (~6/wk
→ 49/wk) but never converts it to a dollar figure — Finding 2 names why (nothing logs it). A bound,
shown explicitly as ESTIMATED: the one MEASURED mega-event (215 agents / 12.9M tokens, §1 Finding 6)
implies **~60,000 tokens/agent** average for a search-heavy session. Applying that as a rough per-run
proxy: **old regime** (6 runs/wk × ~60K tok) ≈ 360K tok/wk; at a blended ~$4.4/MTok (70/30 in/out on
Sonnet-class pricing) ≈ **~$1.6/wk**. **New regime** (49 runs/wk, but each run's budget was cut by
~2–3× — iterations 4→2, wall-clock 4h→75min, so call the per-run token cost ~⅓): 49 × 20,000 ≈ 980K
tok/wk ≈ **~$4.3/wk**. So the redesign's *real* multiplier may be closer to **~2.7×**, not the naive
**8×** the run-count alone suggests — *if* the budget cut actually holds in practice, which is exactly
what nobody can currently check (Finding 2). **This estimate rests on one analogy (the 215-agent
burst's average) standing in for a completely different workload (12-search daily screens) and should
be treated as illustrative of the arithmetic, not as a number to plan a budget against** — the fix is
to log real `response.usage` per run, not to trust this or any other back-of-envelope figure.

### 5.3 When *not* to fan out

Anthropic's own multi-agent write-up (PUBLISHED, fetched this session) is explicit: avoid fan-out for
"domains requiring shared context," "high interdependency tasks (**most coding tasks**)," and
real-time delegation; it fits "valuable tasks that involve heavy parallelization... and interfacing
with numerous complex tools." Read against TanitAD's own incident history, the 215-agent burst was
very likely the **right domain** (a literature/deep-research screen — squarely what the guidance
recommends fan-out for) at the **wrong scale** — 215 is over 20× Anthropic's own stated "10+ subagents"
threshold for "complex research," with no governor in between. The fix implied is not "fan out less
on research," it is "cap fan-out regardless of domain, and additionally keep single-owner/low-concurrency
discipline on anything touching `stack/` or the shared git index" — which is closer to what
`CLAUDE.md:372`'s "partition by directory/concern" language already gestures at, just never enforced
(§3.2).

---

## 6. Q5 — Designing an autonomous research loop, and where the frontier actually is

### 6.1 Frontier evidence, briefly

- **METR's revised time-horizon estimate** (PUBLISHED, `metr.org` — direct fetch blocked by this
  session's egress policy; figure corroborated across independent secondary sources including METR's
  own blog title "Time Horizon 1.1"): post-2023 doubling time is now **~130.8 days (4.3 months)**, ~20%
  faster than the original 7-month figure; if it holds, month-long autonomous tasks arrive **~2027**.
- **RE-Bench** (METR, arXiv:2411.15114, PUBLISHED): at a 2-hour budget, the best agents scored **4×**
  human experts; at 8 hours humans narrowly overtake; at 32 hours humans score **2×** the top agent.
  Agents are already ahead on **short, bounded** tasks and not yet on **long, open-ended** ones.
- **PaperBench** (OpenAI, arXiv:2504.01848, PUBLISHED): Claude 3.5 Sonnet (superseded several
  generations ago) scored **21.0%** replicating ICML papers end-to-end vs a **41.4%** human-PhD
  baseline on a 3-paper subset — a large, though dated, gap on full research replication specifically.
- **Reward hacking in autonomous research** (arXiv:2609.28614, PUBLISHED): 30.5%/74.6% hack rates,
  6.5% missed by panel review, evasion sophistication rising with rounds (§1 Finding 7) — the
  strongest available argument for keeping gate decisions human/pre-committed regardless of how
  capable agents get.
- **AlphaEvolve** (DeepMind, arXiv:2506.13131) and **Google Co-Scientist** (arXiv:2502.18864) both
  depend on the same precondition: a **trustworthy, automated evaluator** the search loop can trust
  blindly. TanitAD's `taniteval` + `run_gate.py` is an unusually strong instance of exactly that
  component — the missing piece for "AI that engineers AI" here is not the evaluator, it is the
  proposal/relay loop connecting evaluator output back to the next hypothesis without passing through
  the orchestrator's own unverified-premise failure mode (Finding 5).
- **OpenAI's own roadmap** (PUBLISHED, multiple outlets, Sept 2026): an "automated research intern...
  under human direction" was declared reached **2026-09-06**; "sustained independent work... little to
  no human intervention" is targeted for **March 2028**. TanitAD's D-018 (agents EXECUTE
  runs/intakes/monitoring; PI decides strategy/tactics/phase-transitions) already sits on the
  conservative, 2026 side of that industry line — this is a validation of the existing split, not a
  reason to move it.
- **LangProp** (arXiv:2401.10314) is the closest published precedent to "LLM engineers the driving
  policy": it optimizes *executable code* against a metric in a training loop, needing the LLM only at
  train time — directly analogous to what a TanitAD architecture-search loop would need to look like
  to stay cheap at deployment.

### 6.2 The loop, stage by stage — who owns it today

| Stage | Agent-owned today? | Frontier evidence | TanitAD status |
|---|---|---|---|
| Hypothesis / literature surfacing | **Yes** | Co-Scientist's Generation agent | Already daily (6 discipline agents) |
| Pre-registration (falsifier + thresholds) | **Yes**, gate-at-write-time recommended | AlphaEvolve's evaluator-first design | `Gates/*.card.json` exists; no schema check at registration (R6's unbuilt P6) |
| Bounded (≤1 day, 1 owner) implementation | **Yes** | RE-Bench: 4× human at ≤2h budgets | Matches the current daily-bounded-loop design — keep it bounded |
| Open-ended, multi-day, multi-file implementation | **Not yet reliably** | RE-Bench: humans overtake by 8h, 2× better at 32h | Correctly not attempted as agent-autonomous today |
| Training job launch/monitor | **Yes, mechanically** | — | `fleet_probe.py`, `supervise_run.sh` already do this |
| Four-family eval execution | **Yes, mechanically**; self-enforcing *the rule* is not | — | `CLAUDE.md:380`'s own note: "three separate reports went out ADE-only after this was requested" |
| Registry update | **Yes where pointer-covered** (14/∼hundreds), manual elsewhere | — | `registry_lint.py` exists; adoption partial |
| Restart/continue/kill gate decision | **No — correctly a human/pre-committed-rule decision** | Reward-hacking survey; OpenAI's 2026-vs-2028 line | `GATE_PROTOCOL.md` + D-018 already get this right |
| Claims crossing to publication/PI/external | **No** | PaperBench gap; CLAUDE.md's own evidence-class rule | Already an ESCALATE item under D-018 |

### 6.3 Exploiting task-horizon growth without losing rigor

The contrast worth naming explicitly: **METR's capability curve doubles in ~4.3 months; `LOOP_STATE.md`
was doubling in ~2–3 weeks** (§2.8's +9 KB/day on a ~150 KB base). The signal the programme should be
racing to keep up with is far slower than the noise it is currently accumulating. Three concrete
implications:

1. **Lengthen the per-run budget only in step with MEASURED returns at the current horizon** — not by
   assumption. `_common-protocol.md`'s 2-iteration/75-minute cap should move only when a discipline
   agent's own G-H "measured experiment" gate shows the extra time is landing more than it costs
   (which requires the token-telemetry from §4's automate-list item 3 to even evaluate).
2. **Spend newly-available horizon on the currently-scoped-out stage** (open-ended, multi-day,
   multi-file implementation) rather than on more parallel breadth — breadth is already at its stated
   ceiling (5–8 concurrent) and Finding 6 shows what happens past it.
3. **Do not move the gate-decision boundary as capability grows.** The risk that scales with capability
   is reward-hacking *sophistication* (the survey's own finding: evasion rounds 7→56 as rounds
   increase), not just raw error rate — a more capable agent is not thereby a safer one to hand a
   restart/continue/kill decision to.

---

## 7. Recommendations, ranked (cost × effect)

| # | Recommendation | Cost | Effect |
|---|---|---|---|
| **R1** | Wire the 4 already-built tools to hooks: `Stop` (idle-turn guard), `SessionEnd` → `session_guard.py`, `PreToolUse` on `git commit` → redirect to `safe_commit.py`, `SubagentStop` → manifest check | **S** (a `settings.json` edit + ~4 thin wrapper scripts; every underlying tool already exists) | Converts the programme's 4-times-repeated "NEVER IDLE" rule and its stranding-prevention contract from memory-dependent to structural |
| **R2** | Add LLM/API token-spend logging (extend `RESOURCE_LEDGER.md` or a sibling `tools/api_ledger.py`) | **S** | Closes Finding 2; the resource that blew a budget once (and is projected to grow ~8×) is currently un-measurable, only ESTIMATED |
| **R3** | Split `LOOP_STATE.md` into a capped live header + an append-only archive (R6's P7, still unbuilt) and add a `PostToolUse` size guard | **S/M** | Directly fixes Finding 4 (a self-contradicting "read this first" pointer) and the single largest recurring token cost identified in §5.2 (~38× on that one file alone) |
| **R4** | A fan-out governor: a semaphore file + `SubagentStart` hook capping concurrency, independent of the per-agent duration budget that already exists | **S** | Prevents a repeat of the 215-agent/12.9M-token event; makes the "5–8 concurrent" rule mechanical rather than aspirational |
| **R5** | Extend `registry_lint.py`'s retracted-claim/pointer-drift sweep to `LOOP_STATE.md`, `BOOST_PROGRAM.md`, and the `AgentSchedule` docs | **S** | Would have caught Finding 10; generalizes an existing, proven tool rather than building a new one |
| **R6** | Route the orchestrator's own brief-writing/relay step to an Opus-tier model at high effort, not the Sonnet tier used for subagent implementation | **S** (a routing-policy decision, no new code) | Directly targets the programme's own self-diagnosed highest-error layer (Finding 5, M2/M12) — currently nothing in the repo suggests this routing distinction is made at all |
| **R7** | Adopt a `prompt`-type hook at `Stop`/`SubagentStop` that checks evidence-class tagging and four-metric-family completeness before a report is filed | **M** | Mechanizes two rules that had to be restated because agents did not reliably self-apply them (`CLAUDE.md:180,380`) |
| **R8** | Backfill `registry_lint.py` pointer coverage beyond the current 14 rows | **M** (incremental, per `tools/README.md`'s own note that this can be done gradually) | Extends the one place "generated, not hand-copied" was actually achieved; the tool already exists and already supports incremental adoption |
| **R9** | Retire the two dead report series formally (strikethrough + pointer, matching `BOOST_PROGRAM.md`'s own good M5-bis pattern), and make an explicit call on the also-now-dead 3×/day series | **S** | Removes navigability tax for the next reader; zero risk (R6 already recommended this on 07-25) |
| **R10** | Build the still-missing `ckpt_relay.py` (P4) and `results_ledger.py`/generated `GATES.md` (P5, P6) from R6 | **M** each | Unchanged priority from 07-25 — still valuable, still unbuilt; listed last only because R1–R9 are cheaper and target failures *measured in this pass* specifically |

---

## 8. Open questions / UNVERIFIED

- **Whether a HANDOFF block exists anywhere marking the 08-04→09-25 pause deliberately** — I did not
  do an exhaustive per-discipline `STATE.md` sweep; `CONTINUATION_PROTOCOL.md §3.6` prescribes one and
  I found no evidence of one at the two most likely locations (`PROJECT_STATE.md` top, `LOOP_STATE.md`
  tail). Flagged, not asserted absent (per CLAUDE.md's own "absence found at one location" rule).
- **Actual per-discipline `STATE.md`/`KNOWLEDGE_BASE.md`/`BACKLOG.md` sizes** — an attempted
  measurement failed (a `find`/`xargs` pattern error, not re-run); the §5.2 cost estimate is therefore
  built only on the 5 core steering docs plus the one MEASURED mega-burst, not on discipline-level
  reading load, and likely **understates** true per-session context cost.
- **Whether prompt caching (`cache_control`) is used anywhere in the production loop** — nothing in
  `tools/`, `.claude/`, or the skill files references it; UNVERIFIED rather than confirmed-absent,
  since the harness-level default behavior is outside this repo's visibility.
- **Current (2026-09-25) Anthropic pricing** — the routing table in §5.1 uses the `claude-api` skill's
  cache, dated 2026-06-24 (3 months stale); I did not cross-check with a live fetch. Directionally
  reliable (the tiering and the routing logic do not depend on exact prices), but exact $/MTok figures
  should be re-verified before being used to size a real budget.
- **The exact per-run token cost of a discipline agent's daily loop** — §5.2's "~2.7× not 8×" estimate
  is explicitly a single-analogy ESTIMATE, not a measurement; R2 exists specifically because this
  cannot currently be checked.
- **METR's primary-source page** (`metr.org`) — blocked by this session's network egress policy; the
  4.3-month figure is corroborated by independent secondary sources (Wikipedia's METR page, AI
  Digest's time-horizons tracker, and the search-indexed METR blog title itself) but was not read
  directly.

---

## 9. Deliverable manifest

| Artifact | Location |
|---|---|
| This report | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/R5_agentic_harness.md` |

No code, tools, or repo edits were produced — this is a read-only review per the session brief ("Write
ONLY your own stream file... report bugs, don't fix them"). Nothing from this stream exists anywhere
other than the single file above; there is no pod- or worktree-only component to reconcile.

**Integration escalation:** none required for this file itself (it is the final deliverable of a
review stream). The *content* surfaces several items that should be escalated to whoever synthesizes
the 9 streams into the programme-wide review output: Finding 1 (zero mechanical enforcement) and
Finding 2 (no API-token ledger) are both cheap (S-effort) and address risks the programme's own docs
already flag as open ("the PI should decide on") — they are candidates for the synthesis's own
top-line recommendations, not just this stream's.
