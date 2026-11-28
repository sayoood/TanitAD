# Run manifests — `supervise_run.sh <run>.env`

A manifest makes a training run **survivable**. `supervise_run.sh` relaunches a trainer that
auto-resumes from `ckpt.pt` whenever it dies, until the run's DONE sentinel appears, and emits an
external heartbeat so a stall or a death is detectable without an interactive session.

## Why these live in the repo and not only on the pod

They were pod-only, and `/workspace/ops/runs.d/` on the new pod was **empty** — so the programme's
headline 6-day run had **no supervisor at all**. A manifest that exists on one disk is the same
failure class as an artifact that exists on one disk.

## The contract

⚠️ **CORRECTED 2026-08-16 against `scripts/supervise_run.sh` itself — the old line here was wrong
in both directions.**

Required: **`RUN_ID` `OUT` `TRAIN_CMD`** (`supervise_run.sh:24-26`). ⚠️ **`WORKDIR` is NOT
required** — it defaults to `.` (`:27`), so a manifest that omits it runs the trainer from wherever
the supervisor was launched, which is a `ModuleNotFound` waiting to happen. **Set it anyway.**

Optional: `TRAIN_MATCH` `DONE_TOKEN` `HEARTBEAT` `HB_PERIOD` `MAX_BACKOFF` `TRAIN_LOG`
`INIT_BACKOFF` (`:30`) and ⛔ **`OPS_DIR`, which defaults to `/workspace/ops` (`:32`)**.

⛔ **`OPS_DIR` is the one that bites off a RunPod.** The heartbeat file and the single-instance
`flock` both live under it, so on **Thor** (where there is no `/workspace`) an unset `OPS_DIR`
silently puts the lock and the heartbeat on a path that does not belong to the box — the lock then
guards nothing and the heartbeat is invisible. **Set `OPS_DIR` explicitly on any non-RunPod host.**

⛔ **Always set `TRAIN_MATCH`** when a trainer may already be running hand-launched. The
single-instance `flock` only stops two *supervisors*; it cannot see a trainer started with `ssh -f`
or under a bare `while true` wrapper. With `TRAIN_MATCH` the supervisor **waits** and takes over
only once the existing process is gone. Waiting is always safe; double-launching never is — two
trainers fight over the GPU *and* over `ckpt.pt`.

## Deriving `TRAIN_CMD` — copy it, do not retype it

Read it **verbatim from `/proc/<pid>/cmdline`** (NUL-separated) and shell-quote each argument.
Retyping loses arguments that contain spaces — e.g. `--heldout-off-reason 'PI directive: …'`, which
is a *required* companion to `--no-heldout-gate` and is deliberately not a bare `--force` boolean.

```bash
python3 -c "import shlex; print(' '.join(shlex.quote(c) for c in \
  open('/proc/<PID>/cmdline','rb').read().decode().rstrip('\0').split('\0')))"
```

## Launch

⚠️ **Copy the manifest out of the repo FIRST.** The launch line below points at
`/workspace/ops/runs.d/`, and *that directory being empty is the exact failure this README was
written about* (see "Why these live in the repo"). The repo copies are in **`stack/ops/runs.d/`**.

```bash
mkdir -p "${OPS_DIR:-/workspace/ops}/runs.d" && \
cp stack/ops/runs.d/<run>.env "${OPS_DIR:-/workspace/ops}/runs.d/" && \
setsid nohup bash scripts/supervise_run.sh "${OPS_DIR:-/workspace/ops}/runs.d/<run>.env" \
  > /tmp/superv_<run>.log 2>&1 < /dev/null &
```

⚠️ **The supervisor's own log goes to `/tmp`, never `/workspace`.** A MooseFS write failure on the
log file raises `OSError: [Errno 5]` inside `print()` and kills the process **leaving no diagnostic
at all** — a logger writing to the failing filesystem cannot report that the filesystem is failing.
This single mechanism explained three separate "mysterious" deaths on 2026-08-03.

Verify after launch: the log must say **"trainer ALREADY RUNNING … NOT launching; waiting"**, and
the trainer's PID set must be *unchanged*.

⛔ **DO NOT VERIFY WITH `pgrep -f <trainer>` — corrected 2026-08-16.** This README used to say
exactly that, and it is the self-matching trap the rest of the programme bans: `pgrep -f` /
`ps | grep` put the searched token into the *searching* process's own command line, so the probe
matches itself. Measured three times here, most recently as a monitor that reported
`Traceback CUDA out of memory` for a run that was **healthy and three minutes in**. Read
`/proc/*/cmdline` instead, or use the ready-made probe:

```bash
python3 scripts/v6_chain.py verify --step S-T --root <experiments root>
```

## ⚠️ The v6 ladder generates its own manifests — do not hand-write them

```bash
python3 scripts/v6_chain.py manifests --dest stack/ops/runs.d \
  --root <experiments root> --workdir <stack path>
```

⛔ **One manifest PER STAGE, and every `TRAIN_CMD` is the TRAINER, never `v6_chain.py`** — a
supervised *chain* would replay stage 1 after a mid-ladder crash. Each manifest gets a run-scoped
`TRAIN_MATCH='train_v6_staged\.py.*<run_id>'`, so a supervisor waits for **its own** stage's trainer
and can never adopt a sibling's.

🟥 **KNOWN GAP (2026-08-16): the live `v6F-SW-30k` run on Thor has no manifest here.** Generating one
from this box would guess Thor's `$HOME` and a guessed manifest is worse than none — it is the
"lies about what is running" failure in a new costume. ⇒ It must be generated **on Thor**, with the
`TRAIN_CMD` read **verbatim from `/proc/25477/cmdline`** using the snippet above, and only while
that is safe to do. The three downstream stages' manifests come from `v6_chain.py manifests`.

⚠️ **Editing a manifest under a live supervisor changes NOTHING** — `supervise_run.sh` sources it
**once, at startup**, and replays the `TRAIN_CMD` it captured. To change a supervised run: edit →
kill the **SUPERVISOR** first → kill the trainer → start a fresh supervisor. And do not restart
immediately: the new supervisor races the old one's `flock`, prints *"another supervisor holds
…lock"* and dies, leaving **nothing running**.
