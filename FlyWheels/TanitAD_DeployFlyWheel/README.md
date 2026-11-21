# TanitAD_DeployFlyWheel

**Owns:** P6 — TanitDeploy and the Thor pipeline (TANITAD_PROGRAMME.md §2).

**The contract (redesign completion, 2026-08-27):**
- **Briefs IN / results OUT** live in `incoming/<YYYY-MM-DD>-<slug>/` using the
  §3 work-package schema (SPEC/PLAN/tests/code/raw/RESULT/COMMS) — the durable
  channel that survives sessions.
- **Live coordination** is `SendMessage` between sessions (the Master Mind's
  session name is listed by `ListAgents`); a brief in a folder is NOT a running
  agent — the commissioner must also spawn or message one.
- Operating standard binds: STAGE never push · deliverable manifest · escalate
  integration. ⛔ The old `TanitAD Research Hub/` tree is DEAD — all writes go
  to `TanitAD Research Lab/` or here.
