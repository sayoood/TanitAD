# `tools/` — repo-level dev tooling (agent-facing, not `stack/` model code)

Cross-cutting scripts every TanitAD session/agent runs. These are **not** MVP model
code, so they live at the repo root (not under `stack/`) and are maintained directly
by the Tools & DevEnv agent — no intake round-trip. Stdlib-only, ASCII-clean stdout
(the Windows cp1252 console lesson), OS-agnostic (a `.py` core + a `.ps1` Windows
wrapper; on the pod call the `.py` directly). `gpu_tripwire.py` is the one exception
to stdlib-only — it needs `torch` and the `stack/` package, by definition.

| Tool | What it does | When to run |
|---|---|---|
| `ci_gate.py` / `ci.ps1` | One-command test gate: fails on failure, **collection error**, slow test, wall blow-out, a missing/red tripwire node, a **suite below its manifest floor**, a total-collected count under `--min-total`, or a CUDA parity failure. | **Before every commit/push** (protocol G-E). |
| `gpu_tripwire.py` | CUDA device-parity probes on the real model (encode/imagine CPU-vs-GPU, I2 on device, backward-finite) + the batch-1 encode latency (I8 proxy). | Via `ci_gate --gpu-smoke`, or standalone on any GPU box. |
| `session_guard.py` / `.ps1` | D-026 stranded-work guard: **blocks** on uncommitted hub deliverables; **warns** on uncommitted `stack/`+`tools/` source, unmerged `agent/*` branches vs tip, and stale INTAKE verdicts. | **Session end**, every agent (protocol G-F). |
| `fleet_probe.py` | Fleet liveness by **discovery**, never by hardcoded log names: finds jobs in `ps`, binds each to its own log via the launcher's stdout redirect, cross-checks the GPU against the process table, catches freezes (`LOG_STALE`, `STEP_NOT_ADVANCING`) and measures disk with a real `dd`. Exit `0`/`1`/`2` = GREEN/AMBER/RED. | Before claiming the fleet is healthy; behind the `fleet-status` skill. |

Tests: `pytest tools/tests/` — **77 falsifiers, 16.5 s** (2026-07-21). Each drives a
throwaway git repo, a synthetic pytest project, or a captured `ps`/`nvidia-smi` payload
end-to-end; the CUDA-specific ones skip loudly on a CPU-only box.

## fleet_probe

```bash
python tools/fleet_probe.py                     # table, all four hosts, ~10 s
python tools/fleet_probe.py --json              # machine-readable
python tools/fleet_probe.py --hosts pod1 --no-dd
```

It exists because the previous monitor greped **hardcoded** run names
(`p0-sB01-realmix.log`, `arm_base.log`, `pgrep -fc train_worldmode[l]`) that had been
renamed away. A grep matching nothing prints nothing, and printing nothing was scored as
health — the fleet lost 2 of 4 GPUs behind dead trainers **four times** under a monitor
that reported no anomaly. The one rule the tool enforces everywhere:

> **Absence of evidence is an ALARM, not an all-clear.**

A verdict starts at `UNKNOWN` and needs positive evidence to reach GREEN; a running job
whose log cannot be discovered is **AMBER — "liveness UNVERIFIED"**, never green. Roles
matter: `role=burst` (the eval pod) is *supposed* to sit idle, `role=train` at 0 % is RED.

Two Windows traps are baked in, both measured the hard way (see the 2026-07-21 note):
remote bash payloads are sent as **LF bytes** (`text=True` would translate `\n` to CRLF
and every `fi` would arrive as `fi\r`), and on win32 it drives
`C:\Windows\System32\OpenSSH\ssh.exe` — git-bash's MSYS `ssh.exe` **deadlocks** under
`subprocess` pipes, and does so only against the *busy* hosts, which looks exactly like an
outage and is not one. Override with `--ssh`.

## ci_gate

```bash
python tools/ci_gate.py --rootdir stack                  # the standard gate
python tools/ci_gate.py --rootdir stack --gpu-smoke require
python tools/ci_gate.py --rootdir stack --json gate.json # for the orchestrator
python tools/ci_gate.py --rootdir stack -- -k comma2k19  # pytest passthrough
```

Windows: `.\tools\ci.ps1` (activates the off-Drive venv; `-GpuSmoke require`, `-Json`).

- **Exit 0** = GATE PASS. **Exit 1** = one or more gate reasons, all printed.
  **Exit 3** = pytest could not be launched at all.
- **Why a suite manifest and not just node tripwires:** a named-node tripwire only
  guards nodes somebody thought to name. `SUITE_MANIFEST` pins the load-bearing
  modules (instrument doctrine, calib trio, the three reference arms, eval/metric
  surfaces) to a collected-count **floor**, so a module that is deleted, renamed, or
  quietly halved fails the gate. Adding tests is always fine; removing them has to
  edit that dict on purpose. `--min-total` (default 390) is the same idea for
  wholesale loss, e.g. a broken `conftest.py` deselecting half the tree.
- **Budgets** (measured 2026-07-20): full suite **60.2 s / 531 tests** on the Drive
  tree, **39.3 s / 396 tests** in an off-Drive worktree; tall pole
  `test_replay_app_test_mode_and_regression_gate` 7.2–7.9 s. Defaults are 15 s
  per-test / 150 s wall — comfortably inside the 5-minute ceiling, so **no sharding
  is needed** at current suite size.

## gpu_tripwire

```bash
PYTHONPATH=stack python tools/gpu_tripwire.py            # human report
PYTHONPATH=stack python tools/gpu_tripwire.py --json g.json --require-cuda
```

The `stack/tests` suite is **100 % CPU-only** (`grep -rl cuda stack/tests` returns
nothing, measured 2026-07-20) while every trainer, eval and deploy tick runs on a
GPU — so device/dtype/NaN regressions were invisible to CI. Four probes close that:

| Probe | Asserts |
|---|---|
| `P1_encode_parity` | `WorldModel.encode` CPU vs CUDA, `max abs dev <= --tol` (1e-3) |
| `P2_imagine_parity` | operative predictor, every horizon, same tolerance |
| `P3_i2_on_device` | I2 batch-1 vs batch-B encoder consistency, **run on CUDA** (1e-4) |
| `P4_backward_finite` | one `loss.backward()` on CUDA; every gradient finite |

Measured on the local RTX 4060 (torch 2.11+cu128, fp32, 2026-07-20): all four pass
in **1.7 s**, worst deviation **9.5e-07**, batch-1 encode **1.26 ms**. No CUDA
visible → `--require-cuda` fails, otherwise a loud skip.

## session_guard

```bash
python tools/session_guard.py            # gate the current worktree
python tools/session_guard.py --strict    # branches, source + stale INTAKEs also block
python tools/session_guard.py --base origin/main   # tip = a different ref
python tools/session_guard.py --json      # machine-readable report
```

Windows: `.\tools\session_guard.ps1` (activates the off-Drive venv, same flags).

- **Exit 0** = clear to end the session (warnings may still print).
- **Exit 1** = a BLOCKING condition — uncommitted deliverable under `TanitAD Research
  Hub/`, `Project Steering/`, `PROJECT_STATE.md`, or `DECISIONS.md`. Commit or discard,
  then re-run until `RESULT: PASS`.
- **Exit 3** = not a git repo / git unavailable.
- The **source check** (`stack/`, `tools/`) warns rather than blocks — a mid-work tree
  is legitimately dirty — and lists untracked files separately, because an untracked
  module has no copy anywhere. It was added on 2026-07-20 after the shared Drive tree
  was found holding 40 uncommitted `stack/` paths, 22 of them untracked (12 test
  modules = 135 tests, 9 `tanitad/lake/*` modules) while the hub check said "clean".
- Status is read with `--untracked-files=all`: the git default collapses a wholly
  untracked directory to one `?? stack/` row, which would hide exactly those modules.

The "tip" defaults to `HEAD` (the worktree's current integration point) because
`origin/main` is intentionally diverged in this repo; pass `--base` to override.
