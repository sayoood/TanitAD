# Redesign-implementation audit (PI, 2026-08-27 night)

**Trigger (PI, verbatim intent):** the Research Lab is not running daily; the
folder redesign is incomplete (no FlyWheel folders; work still landing on the old
Research Hub); communication between the Master Mind, the agents, and the Lab's
results is not assured. **Every finding below is MEASURED tonight.**

## 1. Findings and same-night fixes

| finding (measured) | fix | state |
|---|---|---|
| **12 tracked files still under `TanitAD Research Hub/`** (the 08-24 label-extraction package) | `git mv` to the Lab tree, history preserved; empty Hub husk removed | ✅ tonight |
| **A live agent worktree carried a FULL Hub tree** and its tools wrote Hub paths (the EvalFlyWheel base) — the "something still working on the old hub" | its 62 staged files integrated with Hub→Lab re-path + key-union library merge (98 entries, 0 orphans); 62 doc citations normalized; the FlyWheel instructed: Hub is dead | ✅ `49eb9b852` |
| **Zero FlyWheel folders** despite §2 of the constitution | `FlyWheels/TanitAD_{Data,Training,Deploy,Eval}FlyWheel/` created, each with `incoming/` + a README carrying the comms contract and product ownership | ✅ tonight |
| **Research Lab not running daily** — last run artifact 2026-08-22; the constitution says the Master Mind triggers it ~08:00 daily. ⛔ Root cause: the trigger lived in prose, in no executable loop — **my miss** | the drumbeat's morning step now SPAWNS the Lab agent (not merely reports); first execution tomorrow ~08:00 | ✅ armed |
| **Comms unassured** — briefs written to folders were treated as commissioning (measured tonight: the EvalFlyWheel idled beside its brief until SendMessage'd) | **The contract, now in every FlyWheel README:** (1) durable channel = `incoming/<date>-<slug>/` work packages (§3 schema); (2) live channel = `SendMessage` between sessions; (3) ⛔ a brief in a folder is NOT a running agent — the commissioner spawns or messages one, always; (4) results return as RESULT.md in the same package + a SendMessage back when the peer is live | ✅ codified |

## 2. Deliberate non-residue (do not "clean" these)

`tools/kb_add.py` + `tools/tests/test_library.py` keep a **dual-name resolver**
(Lab preferred, Hub probed as fallback) — migration-tolerant by design, tested,
and the reason nothing broke mid-rename. Not Hub residue.

## 3. Still open (the audit's remaining work, next sessions)

1. **Agent↔agent comms**: FlyWheel sessions are mostly Remote-Control/offline —
   SendMessage reaches only live ones. The durable channel covers async, but a
   standing convention for WHICH FlyWheel sessions run WHEN needs the PI's
   session schedule (they start them by hand today).
2. **Lab-results routing**: the Lab's daily output should land as one
   `TanitAD Research Lab/<domain>/…` package AND be triaged into the next
   morning's report (added to the report template).
3. ⚠️ **Naming decision for the PI (flagged by the EvalFlyWheel):** the repo has
   BOTH `Benchmarks & Eval/` (top-level, singular — holds LEADERBOARD.md) AND
   `TanitAD Research Lab/Benchmarks & Evals/` (plural). They are DIFFERENT
   directories; a blanket singular→plural rename would break the former.
   Explicit decision, never a regex.
4. The nested `TanitAD Research Lab/Project Steering/` and `Opponent Analysis`
   naming vs the constitution's "Opponent+Benchmarks" — reconcile with the PI.
5. Sweep the other agent worktrees (5+ live) for Hub-era bases before their
   next integration — the zen-bose case will recur.
