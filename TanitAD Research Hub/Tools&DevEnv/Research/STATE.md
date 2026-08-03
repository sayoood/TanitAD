# STATE — Tools&DevEnv

LAST_RUN: **2026-08-03** (first DAILY run — the slot changed from Monday-weekly this day)
  — branch `agent/tools-devenv-20260803` (worktree `C:/Users/Admin/wt-tools-0803`, off `dfddd4e`)
QUALITY: full (G-A…G-I + G-T1 met; 2 measured experiments — the 8-host live fleet discovery
  run pre/post fix, and the `/proc/fd/1` log-binding before/after on the live flagship)
RESOURCE (G-I): **dev box (Windows) + 8 live SSH probes** to the real fleet. ~1.4 h wall,
  **$0 marginal**. Why not the eval pod: it is **unreachable** (`ssh rc=255`, consistent with
  `POD_SHUTDOWN_2026-08-02.md`), and this run's experiments are fleet-liveness probes and a
  stdlib test suite — an A40 answers neither. The one GPU-shaped item this run surfaced
  (Thor's GPU is invisible to `nvidia-smi --query-gpu`) needs a Jetson, not an A40.

## ⛔ DEBT CLEARED — and the 13 days cost something real

`agent/tools-devenv-20260721` sat **1 commit ahead of tip, 333 behind, for 13 days**, holding a
whole run: `tools/fleet_probe.py` (581 lines), 20 falsifiers, the **`fleet-status` skill rewrite**,
`GOALS.md`, the `2026-07-21-rrd-dual-sink-guard` intake package, `rrd_bench` results, a research
note. Cherry-picked onto `dfddd4e` as **`7f34086`** (2 additive conflicts, both sides kept).

⚠️ **The tip's `fleet-status` skill was still the ORIGINAL hardcoded-grep version.** The fix for
"the monitor reported GREEN because it found nothing, four times" existed on a branch and did not
exist in the fleet for 13 days.

**Still open for the orchestrator:** the `2026-07-21-rrd-dual-sink-guard` INTAKE has **no verdict**
(now 13 d). Older unfilled, carried from 2026-07-20: `2026-07-09-testsuite-io-profiling` (this
discipline's own), `lal-v2-anticipation`, `physicalai-r1-selection`, `models-predictor-failfast`.

## HANDOFF

### ⚠️ For the PI — two paid A40s are doing nothing, and I cannot stop them
MEASURED 2026-08-03 ~06:50 CEST: **pod2 and pod4 are both `RED GPU_IDLE_NO_TRAINER`** (0 % util,
no trainer process). `POD_SHUTDOWN_2026-08-02.md` records **$3.61** of credit and that stopping a
pod is a **console action only** — `runpodctl` on the pods is unauthenticated and there is no
RunPod key in `Keys.txt`. **pod2 is additionally `RED DISK_FULL`** (a real 100 MB `dd` refused), so
it could not resume a checkpoint even if refilled. pod1, pod3 and eval are unreachable.
**pod5 is fine and working:** `flagship-v5f-w120-30k` at step 1250, GPU 85 %.

### For every agent
- **`python tools/fleet_probe.py`** before claiming the fleet is healthy — it now discovers hosts
  from `~/.ssh/config`, so a pod provisioned today is probed today. Exit `0/1/2` = GREEN/AMBER/RED.
- Unchanged: `tools/ci_gate.py` before push, `tools/session_guard.py` at session end.
- **If you need to know where a running job writes, read `/proc/<pid>/fd/1`.** It is exact, free,
  and survives every rename. Do not grep for a log name.

### For Prod-Opt
**Thor's GPU is unmonitored.** `nvidia-smi --query-gpu=...` returns nothing on Jetson (it exposes
`tegrastats`). Honest AMBER while Thor idles; a real gap the moment Thor runs inference. → P0.5.

## Done this run

- **Debt: `7f34086`** — landed the 13-day-stranded fleet_probe run (see above). Merged suite
  **147 passed / 28.0 s**.
- **`fleet_probe` v2 (`9584405`) — membership is now DISCOVERED, not typed.** v1 killed hardcoded
  *log* names and shipped a hardcoded four-host *FLEET dict* — the same defect one level up.
  **MEASURED: the live ssh config holds 8 `tanitad-*` endpoints; the dict knew 4** — and the
  missing ones included **pod5, running `flagship-v5f-w120-30k` at that moment**. A v1 probe prints
  a complete-looking table with the working host absent. `ROLE_HINTS` now supplies only semantics;
  unhinted → `AMBER HOST_UNCLASSIFIED`, vanished alias → `AMBER ALIAS_GONE`, unreadable/empty
  config → **RED** (no fleet is UNKNOWN, not all-clear).
- ⭐ **`/proc/<pid>/fd/1` log binding.** v5f writes `/workspace/v5f_run.log` while its `--out` is
  `/workspace/experiments/flagship-v5f-w120-30k`, and `ssh -f` left no shell parent carrying a
  redirect — so the **live flagship read as `NO_LOG_BOUND`**, correctly (no evidence existed to
  bind on). The kernel had it all along. **Before → after, same host, same minute:
  `step=None log_age=n/a` → `GREEN step=1250 log_age=51s`, GPU 85 %.** Not a loosened check —
  added evidence; the falsifier pinning that the v1 AMBER was *right* is in the suite.
- **Two of my own v2 defects, found by running it, not reading it:** a failed `dd` at a **guessed**
  path was reported `RED DISK_FULL` on thor-wifi — which has no `/workspace` and **917 MB/s** of
  headroom (now `AMBER DISK_UNVERIFIED`; only a hinted `dd_path` may claim quota); and `thor-wifi`
  is the same Jetson on a second interface, which endpoint dedup cannot merge by construction.
- **Tests: `tools/tests/` 164 passed, 38.3 s.** The discovery falsifiers are **mutation-proven** —
  reintroducing v1's hardcoded membership fails 3 of them.
- Research note `2026-08-03-fleet-probe-v2-and-the-13-day-strand.md`; KB +6 deltas; BACKLOG
  re-prioritized.

## Open threads / proposals to raise

- **Expect the stale-name defect at a third level.** It has now appeared at log names (v1 fixed)
  and host names (v2 fixed). `JOB_RE` still encodes what a trainer *cmdline* looks like — a run
  launched through a wrapper this regex does not match is invisible the same way. Not speculative:
  it is the same shape twice already. → new backlog P0.6.
- **`fleet_probe` is still not scheduled.** Nothing runs it every 6 h; it is a discipline an agent
  must remember. Same unresolved gap as `ci_gate` (P0.2) — one hook wiring would close both.
- **`ci_gate` unskippable** (P0.2) and **`gpu_tripwire` bf16/CUDA-graph arm** (P0.3) unchanged
  from 2026-07-20; neither was reachable in a 1-day slot alongside the debt clearance.
- **RESIM_ROADMAP.md still missing** — mission P1 says the TanitResim roadmap lives there. Fourth
  run carrying this. It is now the top P0 for the next daily slot precisely because it keeps losing
  to more urgent work.

## Prior handoff (2026-07-09, still open)

- **Sayed ~1 click:** pin `stack/` to Drive "Available offline" → removes the cold-I/O tax.
  Off-Drive worktree 396 tests / 39.0 s vs Drive tree 531 / 60.2 s. Tool ready
  (`profile_testsuite.py`).
- CARLA camera pixels: graphics-capable pod recreation — NOT urgent; milestone 1 needs no pixels.
- **A docker-capable GPU host** → the only blocker on an AlpaSim closed loop *on a pod*.
  ⚠️ Partly overtaken: `1c4e1b0`/`63ae826` report AlpaSim/NuRec rendering **cracked on the Jetson
  Thor** (gsplat, 492 FPS) and `515190b` retracts the "AlpaSim is blocked" framing. Re-verify
  before quoting the old NO-GO.
