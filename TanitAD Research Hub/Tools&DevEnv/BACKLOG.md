# Tools & DevEnv — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. ~~Colab CLI burst harness~~ **DONE 2026-07-09 00:50 (MVP loop + Sayed's OAuth):** end-to-end
   validated — `colab run --gpu T4` = fresh T4, torch cu128 CUDA job, auto-release, **33 s
   cold-to-done, $0**. Agent pattern + etiquette: `Implementation/colab_burst/README.md`.
   (History below:)
   Facts established: the OFFICIAL `googlecolab/google-colab-cli` is GitHub-source only (not on
   PyPI), requires **py≥3.12**, and is **Linux/macOS only** → installed on **pod2** in a py3.12
   venv (`/opt/colabcli`, symlinked to `/usr/local/bin/colab`); agents on the Windows dev machine
   drive it via `ssh tanitad-pod2 colab ...`. Auth = one-time URL+code flow (Sayed's Google
   account, ~2 min, token persists) — instructions delivered to Sayed 2026-07-08. REMAINING once
   authed: end-to-end T4 smoke (`colab run` a script that clones repo + runs pytest), measure
   cold-to-green, write `Implementation/colab_burst/README.md` with the agent usage pattern.
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
