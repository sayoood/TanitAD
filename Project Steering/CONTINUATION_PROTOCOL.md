# TanitAD — Continuation Protocol

**Purpose.** Guarantee that any working session — human, Claude Code, or scheduled agent — can stop at
any moment (context window full, crash, end of day) and the next session continues without loss of
context, direction, or results. This is the project's memory system.

---

## 1. The three-layer memory model

| Layer | Artifact | Update frequency | Content |
|---|---|---|---|
| **Constitution** | `Project Steering/Mission Plan.md` | Only by Sayed | Vision, goals, hypotheses, principles. Agents NEVER edit; they write proposals to `Project Steering/Proposals/`. |
| **State** | `PROJECT_STATE.md` (root) + per-area `STATE.md` | Every session | Where we are, what is next, what is blocked, session log. |
| **History** | `DECISIONS.md`, git log, `Project Steering/Progress Reports/`, research knowledge bases | Append-only | Why things are the way they are; results and findings. |

Rule of thumb: **state is overwritten, history is appended, the constitution is untouchable.**

## 2. Session start ritual (any agent or session)

1. Read `PROJECT_STATE.md` → current phase, focus, open questions.
2. Read the active phase plan (`Project Steering/Phase 0 Plan.md` while in Phase 0).
3. Read the `STATE.md` of the discipline you will work in.
4. Skim the last entries of `DECISIONS.md` (only entries newer than your knowledge).
5. `git pull` (if remote ahead) and `git log --oneline -15` to see recent work.
6. Only then start working.

## 3. Session end ritual (mandatory, even when interrupted)

1. Update `PROJECT_STATE.md`: §1 one-paragraph status, §3 next actions (checkboxes), §5 session-log row.
2. Update the discipline `STATE.md` you touched.
3. New decisions → append to `DECISIONS.md` (ADR format).
4. Results and findings → the correct home (see §5), never only in the chat transcript.
5. Commit with a conventional message (`area: what — why`), push to `origin main`.
6. If work is half-done: leave a `HANDOFF` block at the top of the touched `STATE.md`:
   what was being done, exact next step, files in flight, known pitfalls.

## 4. Context-window discipline (for Claude Code sessions)

- Front-load reading: state files first, code second, never bulk-read large docs already summarized in state files.
- Long explorations → delegate to subagents; keep only conclusions in the main context.
- When a session notices context pressure: finish the current atomic step, run the session end ritual
  immediately, and write the HANDOFF block. Never leave results only in conversation.
- Big immutable knowledge (research dossiers, regulation summaries) lives in files and is referenced by
  path — re-read only the section needed.

## 5. Where results live (single homes, no duplicates)

| Result type | Home |
|---|---|
| Research findings per discipline | `TanitAD Research Hub/<discipline>/Research/` (dated files + `KNOWLEDGE_BASE.md`) |
| Hypothesis status (H0–H15) | `TanitAD Research Hub/HYPOTHESIS_LEDGER.md` |
| Experiment results (training runs, gates) | `stack/experiments/<exp-id>/` (config + metrics.json + REPORT.md) |
| Benchmark numbers & leaderboard | `Benchmarks & Eval/LEADERBOARD.md` |
| Weekly synthesis | `Project Steering/Progress Reports/YYYY-Www.md` |
| Decisions | `DECISIONS.md` |
| Proposals to change the constitution | `Project Steering/Proposals/` |

## 6. Experiment record format (mandatory for every training/eval run)

`stack/experiments/<exp-id>/` contains:
- `config.yaml` — full reproducible config (git hash, seed, data version)
- `metrics.json` — final metrics incl. the four instrument rows I1–I4 (see D-004)
- `REPORT.md` — 10-line summary: hypothesis/gate targeted, result, verdict, next step
- Instrument rows FIRST. A run without I1–I4 rows does not exist for decision-making.

## 7. Naming and IDs

- Experiments: `p0-s<stage><nr>-<slug>` (e.g. `p0-sA03-bakeoff-residual`).
- Gates: D1…D8 as defined in `Project Steering/Phase 0 Plan.md` §4.
- Decisions: D-00x sequential.
- Research notes: `YYYY-MM-DD-<slug>.md`.

## 8. Weekly cadence

- Mon–Thu: discipline agents run (research + implementation), each updates its knowledge base and STATE.
- Fri: Opponent Analyzer + Orchestrator run; Orchestrator verifies all agents ran, aggregates, writes the
  weekly progress report, updates `PROJECT_STATE.md`, flags drift between plan and reality.
- Sayed steers the core MVP stream daily; agents feed proposals, never override.
