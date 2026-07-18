# `tools/` — repo-level dev tooling (agent-facing, not `stack/` model code)

Cross-cutting scripts every TanitAD session/agent runs. These are **not** MVP model
code, so they live at the repo root (not under `stack/`) and are maintained directly
by the Tools & DevEnv agent — no intake round-trip. Stdlib-only, ASCII-clean stdout
(the Windows cp1252 console lesson), OS-agnostic (a `.py` core + a `.ps1` Windows
wrapper; on the pod call the `.py` directly).

| Tool | What it does | When to run |
|---|---|---|
| `session_guard.py` / `.ps1` | D-026 stranded-work guard: **blocks** on uncommitted hub deliverables; **warns** on unmerged `agent/*` branches vs tip and stale INTAKE verdicts. | **Session end**, every agent (protocol G-F). |

## session_guard

```bash
python tools/session_guard.py            # gate the current worktree
python tools/session_guard.py --strict    # branches + stale INTAKEs also block
python tools/session_guard.py --base origin/main   # tip = a different ref
python tools/session_guard.py --json      # machine-readable report
```

Windows: `.\tools\session_guard.ps1` (activates the off-Drive venv, same flags).

- **Exit 0** = clear to end the session (warnings may still print).
- **Exit 1** = a BLOCKING condition — uncommitted deliverable under `TanitAD Research
  Hub/`, `Project Steering/`, `PROJECT_STATE.md`, or `DECISIONS.md`. Commit or discard,
  then re-run until `RESULT: PASS`.
- **Exit 3** = not a git repo / git unavailable.

The "tip" defaults to `HEAD` (the worktree's current integration point) because
`origin/main` is intentionally diverged in this repo; pass `--base` to override.

Tests: `pytest tools/tests/` (15 falsifiers, each drives a throwaway git repo).
