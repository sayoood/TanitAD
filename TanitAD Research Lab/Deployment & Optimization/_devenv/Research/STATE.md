# STATE — Tools&DevEnv

LAST_RUN: 2026-07-21 (W4, Monday) — branch `agent/tools-devenv-20260721`
  (worktree `C:/Users/Admin/wt-tools-0721`, off `8194807`)
QUALITY: full (G-A…G-I + G-T1 met; **2 measured experiments** — the live 4-pod fleet
  probe, 6 runs, and the rerun `.rrd` sink benchmark, 200 windows × 3 arms)
RESOURCE (G-I): dev-box CPU + **live network to all four pods** (6 whole-fleet probe
  runs incl. a 100 MB `dd` write test on each host), project venv
  `C:/Users/Admin/venvs/tanitad` (py 3.13.5, torch 2.11+cu128, rerun-sdk 0.34.1).
  **~3.1 h wall, $0.**
  Why not the eval pod: both experiments are *about* the fleet and the host toolchain,
  not about model compute. Experiment A measures the pods themselves — running it *on*
  a pod would measure the wrong thing; and its two hardest findings (an MSYS-ssh pipe
  deadlock, a `text=True` CRLF corruption) are **dev-box-specific** and only reproduce
  from this host. Experiment B is a serialization benchmark: rerun's sink path is CPU
  and disk, so an A40 would have idled. No GPU work was warranted, so none was taken —
  and pod2's A40 sat idle anyway, which is escalation #1.

## ESCALATION 1 — pod2 (A40) is idle with no trainer, and has been all day
`tools/fleet_probe.py` returned **RED `GPU_IDLE_NO_TRAINER`** on pod2 on every run
today (0 % util, 0 MiB, **no job process at all**; disk healthy, 208–474 MB/s). The
host is reachable and functional — it is simply doing nothing. **For Sayed /
orchestrator: fill it or stop paying for it.** The 2026-07-17 fleet review named
compute-starved runs the #1 quality ceiling; we are currently paying for compute and
not spending it. Live fleet at 2026-07-21:

| host | verdict | GPU | evidence |
|---|---|---|---|
| pod1 A6000 | GREEN | 91 %, 14,862 MiB | `flagship4b-v3enc-expA-nodrop-2k`, step **1050**, log 6 s fresh, disk 699 MB/s |
| **pod2 A40** | **RED** | 0 %, 0 MiB | **no job process** |
| pod3 A40 | AMBER | 97 %, 18,729 MiB | VLM labelling (Cosmos-Reason2-8B) alive, but its log emits no step → liveness partially unverified |
| eval A40 | GREEN | 0 %, 0 MiB | idle by design (`role=burst`) |

## ESCALATION 2 — the monitor's blind spot was structural, and it is fixed at the source
The 6-hourly monitor has missed a dead trainer **four times**. The root cause is not a
bad check, it is the *shape* of every check: `.claude/skills/fleet-status/SKILL.md`
grepped **hardcoded** run/log names (`p0-sB01-realmix.log`, `arm_base.log`,
`arm_kstep.log`, `pgrep -fc train_worldmode[l]`) belonging to runs that ended weeks ago.
**A grep that matches nothing prints nothing, and a monitor that prints nothing reports
no anomaly.** Every arm since has been renamed, so it was blind by construction.

Shipped: `tools/fleet_probe.py` (discovery-based; 20 falsifiers) **and the skill itself
rewritten to call it**, with the four-recurrence history in the file so the next editor
knows why hand-written greps are forbidden. The rule now enforced everywhere: **absence
of evidence is an ALARM, not an all-clear** — a running job whose log cannot be found is
AMBER, never GREEN.

## ESCALATION 3 — D-026 debt, re-measured this run (`session_guard`, my worktree)
Branches improved; **the intake ledger is at its worst ever**:

| Class | 2026-07-18 | 2026-07-20 | **2026-07-21** |
|---|---|---|---|
| unmerged `agent/*` branches | 9 | 11 | **7** |
| stale INTAKE verdicts (>3 d) | 5 | 8 | **14** |

Four are 12 days old (`lal-v2-anticipation`, `physicalai-r1-selection`,
`models-predictor-failfast`, and this discipline's own `testsuite-io-profiling`). Six
agents produce packages; nobody triages them. This is an orchestrator-policy gap and I
am explicitly *not* answering it with more tooling — `session_guard` has reported it
for four runs running.

## HANDOFF
1. **Every agent: `python tools/fleet_probe.py` before asserting the fleet is healthy.**
   Never hand-write a pod grep again; never `pgrep -f` a trainer name.
2. **Anyone driving ssh from Python on the dev box: use
   `C:\Windows\System32\OpenSSH\ssh.exe`, not git-bash's.** The MSYS client deadlocks
   under `subprocess` pipes and does so **only against the busy hosts** — it manufactures
   fake outages that look exactly like real ones. (Same payload: 2.0–2.2 s from a shell,
   >90 s timeout from Python; native OpenSSH 0.7–2.5 s on all four hosts.)
3. **Benchmarks & Eval / anyone holding archived `.rrd` files:** if you ever ran the
   replay app with `--rrd` *and* `--serve`, the artifact is a **stub** — 3,196 B instead
   of 10,593,180 B for the same input. Re-record with one sink.
4. **Orchestrator: triage `2026-07-21-rrd-dual-sink-guard/`** (5 falsifiers, standalone
   green). Behaviour-changing by design: `--rrd` + `--serve` starts failing loudly.

## Done this run
- **`tools/fleet_probe.py`** (took the top slot because the program's #1 risk moved to
  ops). Discovers jobs from `ps` grouped by `--out` (a 6-process fan-out collapses to one
  run), binds each to its own log via the launcher's stdout redirect walked up the ppid
  chain, cross-checks GPU vs process table (`ORPHANED_GPU_MEMORY`, `GPU_IDLE_NO_TRAINER`),
  catches freezes two ways (`LOG_STALE` 15 min, `STEP_NOT_ADVANCING` across probes), and
  measures disk with a real 100 MB `dd` — never `df`, which reports the MooseFS cluster
  and hides the per-pod quota. **Whole fleet in 9.7–11.3 s.** 20 falsifiers, 0.35 s.
- **`.claude/skills/fleet-status/SKILL.md` rewritten** to run the probe and forbid
  hardcoded greps.
- **rerun sink benchmark** (`Implementation/rrd_bench/`): 52,966 B/window at jpeg85,
  299 win/s; jpeg85 is **3.79× smaller than raw for 17 % less throughput**; and the
  mission-P1 dual-sink bug is **reproduced and quantified — a 3,314× silent data loss**
  (`serve_grpc()` replaces the file sink). Guard shipped via intake.
- **Three Windows/pod traps measured and encoded** (LF-bytes payloads, native OpenSSH,
  timeboxed per-dir `find`) — each cost real debugging time, each now has a falsifier or
  a docstring naming the symptom.
- **`GOALS.md` created** — it did not exist for the whole life of this discipline, which
  is itself a finding: three runs of tooling with no standing target. G1 fleet-liveness,
  G2 verified-viz, G3 zero-stranded-work, each with a falsifier and a measured first row.
- KB **+11 deltas**; BACKLOG re-prioritized (old P0#1 retired as **stale on three counts**,
  4 findings-driven items added); `tools/README.md` + suite now **77 falsifiers, 16.5 s**.
- Research note `2026-07-21-fleet-probe-and-the-rerun-dual-sink-loss.md`.

## Open threads / proposals to raise
- **The unskippable-gate gap is now the same gap three times** (`ci_gate`,
  `session_guard`, `fleet_probe`): all three are disciplines an agent must remember, none
  is executed automatically. A probe nobody runs is exactly as blind as the grep it
  replaced — GOALS G1's "detected within one 6-hour cycle" is **unprovable** until a cron
  runs it and pages on exit code 2. Now backlog P0#1.
- **The documented rerun tee deadlocks** — `rr.set_sinks(FileSink, GrpcSink(url))` after
  `serve_grpc()` hangs (killed at 120 s, no output; the sink connects back to its own
  in-process server). A real tee needs two `RecordingStream`s and an explicit `recording=`
  per log call. Backlog P0#2 with a pre-registered 5 % falsifier.
- **`rerun-sdk` is pinned in no requirements file** anywhere in the repo, though the whole
  viz backbone depends on it. Backlog P0#3.
- **The Viewer-MCP is still unwired** — until an agent can see its own render, "the
  overlay looks right" stays an assertion. Backlog P0#4; GOALS G2 is *at risk* because
  of it.
- **`gpu_tripwire` is still fp32+eager only** (carried from 2026-07-20) — Prod-Opt's
  CUDA-graph deploy tick and every bf16 training path remain unguarded.
- **`RESIM_ROADMAP.md` is still missing** — fourth run carrying this. Mission P1 says the
  TanitResim roadmap lives there. Honest status: I have now twice chosen a fleet/CI item
  over it because the operational risk was larger. If that is the wrong call it needs
  Sayed's word, not another quiet deferral.
- **TerraZero: still no public code** (5-min check). For whoever checks next: the GitHub
  org literally named `TerraZero` is an **unrelated third party** — Applied Intuition's
  page is `terra-applied.github.io`. An **AlpaSim E2E Closed-Loop Challenge 2026** exists
  as a possible external yardstick if the docker-host blocker ever clears.

## Prior handoff (2026-07-09, still open)
- **Sayed ~1 click:** pin `stack/` to Drive "Available offline" → removes the cold-I/O tax
  (off-Drive worktree 396 tests/39.0 s vs Drive 531/60.2 s). Tool ready
  (`profile_testsuite.py`).
- **A docker-capable GPU host** — still the single blocker on an AlpaSim closed loop; same
  infra class as the graphics-pod ask for CARLA pixels. Worth deciding once.
