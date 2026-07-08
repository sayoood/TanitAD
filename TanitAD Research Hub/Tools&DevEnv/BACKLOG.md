# Tools & DevEnv — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **Colab CLI burst harness** — make Google Colab usable by all agents from the command line
   (account in `Keys.txt`). Method: authenticate, run a smoke notebook that clones the repo,
   installs `stack/`, runs `pytest stack/tests -x -q` on a T4, and writes an artifact back.
   Deliverable: `Implementation/colab_burst/README.md` + runner script + measured session
   setup time. Expected: < 10 min from cold to tests green. Falsifier: CLI auth not scriptable
   ⇒ document the manual-once flow and the scripted rest.
2. **CARLA-on-pod harness prep (W31–32, D-014)** — dry-run the CARLA server install recipe in a
   throwaway container locally or on idle pod (headless, -RenderOffScreen); measure install size
   + boot time + a 100-tick rollout. Deliverable: pinned install script + measured numbers.
   This is the gate for D4–D6 closed-loop.

## P1

3. **MetaDrive supervised install** — BLOCKED on Sayed (~5–10 min supervised, PyPI py3.13
   incompatible → source install). Prepare the exact command sequence + verification script so
   the supervised window is minimal. Escalated via PROJECT_STATE §4.
4. **Pod bootstrap script v2** — one-command environment restore for a NEW pod (apt packages,
   venv, repo, epcache warm) — measured restore time; resilience for the "pod died, new ssh"
   scenario.
5. **Test-suite wall-clock profiling** — 139 tests; measure and cache the 5 slowest (fixtures?);
   target < 60 s local run to keep agents' G-E cheap.

## P2

6. **Windows/Linux path+encoding audit tooling** — the `|`-in-filenames and mojibake classes;
   a lint script for non-NTFS-safe names and non-UTF8 writes in the repo.

## Done / retired
- (2026-07-13-run) MetaDrive front-cam RGB + perturbation package shipped via intake; integrated.
- (2026-07-08) tmux removed from pod flow; detached setsid launcher + runner guard shipped (MVP).
