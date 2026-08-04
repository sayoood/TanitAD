# Run manifests — `supervise_run.sh <run>.env`

A manifest makes a training run **survivable**. `supervise_run.sh` relaunches a trainer that
auto-resumes from `ckpt.pt` whenever it dies, until the run's DONE sentinel appears, and emits an
external heartbeat so a stall or a death is detectable without an interactive session.

## Why these live in the repo and not only on the pod

They were pod-only, and `/workspace/ops/runs.d/` on the new pod was **empty** — so the programme's
headline 6-day run had **no supervisor at all**. A manifest that exists on one disk is the same
failure class as an artifact that exists on one disk.

## The contract

Required: `RUN_ID` `OUT` `WORKDIR` `TRAIN_CMD`. Optional: `TRAIN_MATCH` `DONE_TOKEN` `HEARTBEAT`
`HB_PERIOD` `MAX_BACKOFF` `TRAIN_LOG`.

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

```bash
setsid nohup bash scripts/supervise_run.sh /workspace/ops/runs.d/<run>.env \
  > /tmp/superv_<run>.log 2>&1 < /dev/null &
```

⚠️ **The supervisor's own log goes to `/tmp`, never `/workspace`.** A MooseFS write failure on the
log file raises `OSError: [Errno 5]` inside `print()` and kills the process **leaving no diagnostic
at all** — a logger writing to the failing filesystem cannot report that the filesystem is failing.
This single mechanism explained three separate "mysterious" deaths on 2026-08-03.

Verify after launch: the log must say **"trainer ALREADY RUNNING … NOT launching; waiting"**, and
`pgrep -f <trainer>` must return the *same* PID set as before.
