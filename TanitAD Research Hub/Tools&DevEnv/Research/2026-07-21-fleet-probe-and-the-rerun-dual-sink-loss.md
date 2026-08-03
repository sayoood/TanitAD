# 2026-07-21 — The monitor that reports GREEN because it found nothing, and the .rrd that is 3,314× empty

**Agent:** Tools & DevEnv (Monday) · **Branch:** `agent/tools-devenv-20260721`
(worktree `C:/Users/Admin/wt-tools-0721`, off `8194807`)
**Resource:** dev-box CPU + local network to the 4-pod fleet (6 live probe runs),
project venv `C:/Users/Admin/venvs/tanitad` (py 3.13.5, torch 2.11+cu128,
rerun-sdk 0.34.1). **~3.1 h wall, $0.**

Two experiments, both measured, both producing a shipped artifact and one
negative result each.

---

## 1. Experiment A — a fleet probe that cannot report a false GREEN

### The failure this targets

`PROJECT_STATE.md` (2026-07-20) names the program's top risk as operational:
at the 05:01 UTC probe **2 of 4 GPUs were idle behind dead trainers** — pod1's
REF-B v2 stopped at 22,600/30,000, pod2's flagship-v3enc crashed inside a
checkpoint write — and **the 6-hourly monitor ran at 04:55 and caught neither.
Fourth recurrence.**

I read the monitor (`.claude/skills/fleet-status/SKILL.md`) before writing any
code. The root cause is not a bug in a check; it is the *shape* of every check:

```
grep -o '"step": [0-9]*' /workspace/experiments/p0-sB01-realmix.log | tail -1
pgrep -fc "train_worldmode[l]"
grep ... /workspace/experiments/arm_base.log   # and arm_kstep.log
```

Three hardcoded targets. `p0-sB01-realmix` and the `arm_base`/`arm_kstep` pair
are runs that ended weeks ago; nothing on any pod is called `train_worldmodel`
today (pod1 runs `train_flagship4b.py`). **A grep that matches nothing prints
nothing, and a monitor that prints nothing reports no anomaly.** The monitor was
not wrong about the fleet — it was silent about it, and silence was scored as
health. That is why renaming a run breaks the monitor, and why it has broken
four times: every arm since has been renamed.

The structural fix is one rule: **absence of evidence is an alarm, not an
all-clear.**

### What was built — `tools/fleet_probe.py`

- **Discovers** jobs from `ps -eo pid,ppid,etimes,rss,args`, matching a broad
  job regex and grouping by `--out` (so a 6-process dataloader fan-out collapses
  to one run). No run name, script name or log path is configured anywhere.
- **Discovers** each run's log: first the stdout redirect recorded in the
  launcher's own cmdline (`bash -c '... > /tmp/x.out 2>&1 &'`), walking up to 8
  ppid links to find it, then any log under the run's `--out` dir, newest by
  mtime. A run with no discoverable log is **AMBER — "liveness UNVERIFIED"**,
  which is precisely the case the old monitor scored GREEN.
- **Cross-checks the GPU against the process table.** `ORPHANED_GPU_MEMORY`
  (>512 MiB resident, no owner process) is the dead-trainer signature;
  `GPU_IDLE_NO_TRAINER` fires on a `role=train` host at 0 %.
- **Detects freezes two ways:** `LOG_STALE` (process alive, log silent >15 min —
  the futex-deadlock class that `ps` alone cannot see) and
  `STEP_NOT_ADVANCING` (same step across probes >30 min apart, via a small
  state file).
- **Disk by real write, never `df`.** A 100 MB `dd ... conv=fsync` to
  `/workspace`, because `df` reports the multi-TB MooseFS cluster and hides the
  per-pod quota (CLAUDE.md; a full quota killed the flagship mid-checkpoint).
- Never uses `pgrep -f`/`pkill -f` — those self-match the probing ssh command.
- Verdicts start at `UNKNOWN`; only positive evidence reaches GREEN.

### Measured — live fleet, 2026-07-21

Whole fleet in **9.7–11.3 s** (6 runs). Final run:

| host | verdict | GPU | evidence |
|---|---|---|---|
| pod1 (A6000) | **GREEN** | 91 %, 14,862 MiB | `flagship4b-v3enc-expA-nodrop-2k`, pid 1400645, n=6 procs, **step 1050**, log_age 6 s, disk 699 MB/s |
| pod2 (A40) | **RED** | 0 %, 0 MiB | `GPU_IDLE_NO_TRAINER` — no job process at all, disk 208 MB/s |
| pod3 (A40) | **AMBER** | 97 %, 18,729 MiB | `vlm_semantic_labels.py` (Cosmos-Reason2-8B) pid 53351, log bound via ppid chain, log_age 378 s, `NO_STEP` |
| eval (A40) | GREEN | 0 %, 0 MiB | idle burst host — idle by design (`role=burst`) |

Every one of those lines is a fact the old monitor could not have produced:
pod1's run name did not exist when the skill was written; pod2's RED is exactly
the class that has been missed four times; pod3's AMBER is an honest "I could
not verify this", not a green.

**pod2 is idle right now** — a paid A40 doing nothing. Escalated below.

### Falsifiers — `tools/tests/test_fleet_probe.py`, 20 tests, 0.35 s

Most tests build a host that *looks* fine (ssh answered, a process exists, the
GPU is warm) and assert the probe still alarms. Two are inverted controls — a
healthy training host must reach GREEN, and a genuinely advancing step must not
be flagged — so a green run means something. `test_running_job_with_no_
discoverable_log_is_amber_not_green` is the direct regression test for the
four-time failure.

### Negative results / traps measured (each cost real time)

1. **CRLF into a remote bash is a lie about the fleet.** The first live run
   reported all four hosts down with `bash: line 21: syntax error: unexpected
   end of file`. Cause: a Windows checkout plus — the subtler half —
   `subprocess.run(..., text=True)`, whose stdin TextIOWrapper translates `\n`
   to `os.linesep` on write. Every `fi` arrives as `fi\r`. Fixed by encoding the
   payload to LF **bytes**; falsifier `test_crlf_payload_is_normalised_to_lf_bytes`.
2. **The MSYS ssh client deadlocks under `subprocess` pipes.** After the CRLF
   fix, pod1 and pod3 still reported `ssh timeout after 90 s` — while the
   *identical* payload run from a shell against the same hosts finished in
   **2.0–2.2 s**. Bisected to the transport: git-bash's
   `C:\Program Files\Git\usr\bin\ssh.EXE` hangs intermittently when fed from a
   native-Windows Python over pipes; `C:\Windows\System32\OpenSSH\ssh.exe` ran
   the same payload on all four hosts in **0.7–2.5 s**. It reproduced 100 % on
   the two *training* hosts and 0 % on the two idle ones — i.e. **it reads
   exactly like a fleet outage and is not one**. `fleet_probe.ssh_client()` now
   prefers native OpenSSH on win32 (`--ssh` overrides). Anyone who has ever seen
   "the pod is unreachable" from a Python tool on this box should re-check it.
3. **`find /workspace -maxdepth 3` times out on MooseFS** (>90 s on the busy
   pods). Replaced with per-directory `timeout 8 find ... -mmin -2880`. Same
   asymmetry: the naive form is blind precisely where it matters.

### Shipped

`.claude/skills/fleet-status/SKILL.md` **rewritten to call the probe** and to
forbid hand-written greps, with the four-recurrence history in the file so the
next editor knows why. This matters more than the script: the skill is what
actually runs every 6 hours.

---

## 2. Experiment B — the `.rrd` that is not empty and has no data

Backlog P0#1 asked for "episode → `.rrd`, measure size + write time (G-T1)".
**The backlog was stale in three ways, all corrected by measurement:**

- The rerun logging schema already exists and is mature (`tanitad/replay/rr_log.py`,
  417 lines, per-corpus camera projection, blueprint, legend, metric BEV grid).
- `rerun-sdk` is **already at 0.34.1** in the project venv — the item's
  "pin 0.34.1, migrate, budget 1–2 h for breaking API changes" was work that did
  not exist.
- But **`rerun` is pinned in no requirements file anywhere in the repo**
  (`grep rerun stack/requirements*.txt stack/pyproject.toml` → nothing). The
  viz backbone's only dependency floats. New backlog item.

So the real open question was mission P1's untouched bug: *"dual-sink
(serve+rrd) empty file"*.

### Measured — 200 windows × 3 arms × 256×256 RGB, dev box, 6.0 s total

`Tools&DevEnv/Implementation/rrd_bench/{bench_rrd.py,results.json}`

| case | bytes | B/window | windows/s |
|---|---:|---:|---:|
| `--rrd` only, jpeg85 (shipped default) | 10,593,179 | 52,966 | 299 |
| `--rrd` only, raw frames | 40,107,580 | 200,538 | 359 |
| **`--rrd` + `--serve`** | **3,196** | **16** | 284 |

**The bug is real and it is a 3,314× silent data loss.** Root cause, confirmed
against the SDK's own docstring: `rr.save()` installs the file sink and
`rr.serve_grpc()` — called four lines later in `RerunLogger.__init__` —
*"replaces existing sinks"*. Everything after that goes only to the gRPC
stream; the `.rrd` keeps the blueprint and the static BEV axes and nothing else.

**Why it survived a year of use: the file is not zero bytes.** Every "did the
artifact get written?" check passes. My own benchmark's first emptiness
criterion (`size < 1024`) also passed it — a 3,196-byte stub is "non-empty".
*Non-zero is not non-empty; the only valid emptiness test is against a
same-input single-sink baseline.*

Side result: **jpeg85 is a 3.79× size win for 17 % less throughput** — the
shipped default is the right one, now with a number behind it.

### Negative result — the obvious fix deadlocks

`rr.set_sinks(rr.FileSink(p), rr.GrpcSink(url))` after `serve_grpc()` is the
API-documented tee. Prototyped: **the process produced no output and was killed
at 120 s** — the GrpcSink connects back to the in-process server hosted by the
same thread. A working tee needs two `RecordingStream`s and an explicit
`recording=` on every `rr.log` in `rr_log.py`: a real refactor, backlogged with
a pre-registered falsifier (*dual-sink `.rrd` within 5 % of single-sink*).

### Shipped — intake `2026-07-21-rrd-dual-sink-guard/`

A fail-loud `check_sinks()` that refuses the combination before `rr.init`, with
the measurement and both workarounds in the error message, plus
`allow_stub_rrd=True` for anyone who genuinely wants only the live viewer.
**5 falsifiers, 0.01 s, standalone-green.** Silently shipping a stub is the
worse failure: a researcher opens the `.rrd` next week, sees an empty timeline,
and debugs the model instead of the sink.

---

## Actionable (G-B)

1. **For Sayed / orchestrator — pod2 (A40) has been idle with no trainer since
   at least 2026-07-21.** RED on every probe run today. Either fill it or stop
   paying for it; the fleet review's "compute-starved runs are the #1 quality
   ceiling" is currently being paid for and not spent.
2. **Every agent: run `python tools/fleet_probe.py` before claiming the fleet is
   healthy**, and never hand-write a pod grep again.
3. **Orchestrator: triage `2026-07-21-rrd-dual-sink-guard/`.** Anyone who has
   used `--rrd` together with `--serve` has a stub artifact; if any archived
   `.rrd` is suspiciously small, that is why.
4. **Prod-Opt / anyone driving ssh from Python on this box:** use
   `C:\Windows\System32\OpenSSH\ssh.exe`, not git-bash's. The MSYS client's
   pipe deadlock manufactures fake outages on exactly the busy hosts.
5. **Pin `rerun-sdk==0.34.1`** in the stack requirements — the viz backbone's
   only dependency is unpinned.

## Hypothesis ledger (G-D)

No H-row status changes. This run is instrumentation, not evidence about the
model — recorded here so the ledger is not padded.

## Production-readiness (D-029)

- `tools/fleet_probe.py` — **validated** (20 falsifiers + 6 live fleet runs
  across 4 hosts, both healthy and failing states observed). Gap to
  *production*: nothing executes it automatically. It is wired into the skill a
  human/agent invokes; a cron that runs it and pages on exit code 2 is the next
  level, and shares the "make it unskippable" work with `ci_gate` (backlog P0#2).
- `rrd_bench` — **prototype** (synthetic records, not a real cached episode).
  Gap: run it on one real `ep_*.pt` corpus episode to confirm B/window at true
  frame entropy.
- dual-sink guard — **validated**, pending orchestrator triage.
