# 2026-08-03 — fleet_probe v2: the host list was still hardcoded, and it hid the live flagship

**Agent:** Tools & DevEnv (first DAILY run; the slot changed from Monday-weekly on 2026-08-03)
**Branch:** `agent/tools-devenv-20260803` (worktree `C:/Users/Admin/wt-tools-0803`, off `dfddd4e`)
**Resource:** dev box (Windows) + 8 live SSH probes. ~1.4 h wall, **$0 marginal**.
**Readiness:** `tools/fleet_probe.py` v2 = **validated** (164 falsifiers + 8-host live run,
two live runs pre/post fix). Gap to *production*: nothing runs it on a schedule — the 6-hourly
monitor is still a human/agent discipline (backlog P0.2, unchanged).

---

## 0. First duty — the debt, cleared

`agent/tools-devenv-20260721` sat **1 commit ahead of tip and 333 behind, for 13 days**. That one
commit was a whole run: `tools/fleet_probe.py` (581 lines), 20 falsifiers, the `fleet-status`
skill rewrite, `GOALS.md`, an intake package (`2026-07-21-rrd-dual-sink-guard`), the `rrd_bench`
results and a research note. **In no other commit, on no other branch.**

Cherry-picked onto `dfddd4e` as `7f34086`. Two additive conflicts (`.gitignore`, `tools/README.md`),
both resolved by keeping both sides. Merged suite: **147 passed, 28.0 s** — the 20 fleet_probe
falsifiers coexist with the 127 that landed at tip meanwhile.

⚠️ **The `fleet-status` skill at tip was still the ORIGINAL hardcoded-grep version.** The fix for
the "monitor reported GREEN four times because it found nothing" defect existed on a branch and did
not exist in the fleet for 13 days. Stranding is not a bookkeeping problem.

## 1. The finding: v1 killed the defect at the log level and reproduced it at the host level

v1 discovers trainers from `ps` and logs from mtime + the launcher redirect — but its fleet was:

```python
FLEET = {"pod1": ..., "pod2": ..., "pod3": ..., "eval": ...}   # four hardcoded aliases
```

That is **the same defect one level up**: a name somebody typed, going stale silently.

**MEASURED 2026-08-03** (`~/.ssh/config`, ours): **8 `tanitad-*` endpoints exist, the dict knew 4.**

| endpoint | in v1's dict? | what it was actually doing |
|---|---|---|
| `69.30.85.106:22039` (`pod5`/`tanitad-new`) | ❌ **no** | ⭐ **running `flagship-v5f-w120-30k` right now** |
| `69.30.85.48:22192` (`pod4`/`tanitad-v2arch`) | ❌ no | ran `v1arch-on-v2bal` through the 08-02 shutdown (`POD_SHUTDOWN_2026-08-02.md`) |
| `192.168.178.194` / `.93` (`thor`, `thor-wifi`) | ❌ no | Jetson Thor, idle by design |
| `pod1`, `pod2`, `pod3`, `eval` | ✅ yes | 3 of 4 unreachable |

**So a v1 probe prints a complete-looking four-row table with the only host doing work absent from
it.** Not a false GREEN on a host — a host that does not exist as far as the monitor is concerned.
Strictly worse, and invisible to every check v1 has, because absence-of-a-row is not a finding.

### The fix, and its doctrine

Membership comes from `~/.ssh/config`; `ROLE_HINTS` supplies only **semantics** (what "healthy"
means for that host), **never membership**. Concretely:

- a host in the config with no hint is **still probed** and reported `AMBER HOST_UNCLASSIFIED` —
  probed, but "healthy" is undefined until someone classifies it;
- a hinted alias that **vanishes** from the config is `AMBER ALIAS_GONE` (pod released? drift?);
- an unreadable or `tanitad`-free config is **RED** — no fleet discovered is *UNKNOWN*, not
  all-clear. Exit code 2, loudly;
- aliases sharing an `(hostname, port)` endpoint are **one machine** (`pod4`/`v2arch`) and are
  probed once — probing twice would double-count an A40 and could report two verdicts for it.

## 2. Three defects the live run found in my own v2 — each fixed with its own falsifier

Running it was the test. Reading it would not have found any of these.

### 2.1 ⭐ `/proc/<pid>/fd/1` — the kernel knew all along

v2's first live run reported the flagship as
`AMBER NO_LOG_BOUND: scripts/train_flagship_v4.py (pid 19412) — liveness is UNVERIFIED`.

**Root cause, measured:** v5f runs with `--out /workspace/experiments/flagship-v5f-w120-30k` but
writes to **`/workspace/v5f_run.log`**. No prefix rule, basename rule or `--out`-dir rule can link
those two — and the launcher (`ssh -f`) left **no shell parent carrying a redirect**, so the ppid
walk had nothing to walk. The AMBER was *correct*: the probe had no evidence and refused to guess.

But the evidence existed the whole time:

```
/proc/19412/fd/1 -> /workspace/v5f_run.log
```

v2 now reads `/proc/<pid>/fd/1` for every pid and prefers it over every heuristic — it is the
kernel's own record of where the process writes, which is *direct evidence*, not inference.
`/dev/*`, pipes and sockets are ignored (a process on a tty has no log; binding one would invent
evidence). A log bound by `/proc` but outside the mtime-discovery window is
`AMBER LOG_AGE_UNKNOWN` — bound with certainty, freshness unknown — **not** a pass.

**MEASURED, before → after, same host, same minute:**

| | verdict | evidence |
|---|---|---|
| before | `AMBER NO_LOG_BOUND` | `step=None log_age=n/a` |
| after | **`GREEN`** | **`step=1250 log_age=51s`, GPU 85 %, 16 071 MiB** |

Note what this is *not*: I did not loosen a check to get green. I added evidence that was always
available, and the check passed on it. The falsifier
`test_without_proc_fd_the_same_job_is_correctly_UNVERIFIED` pins that the v1 AMBER was right.

### 2.2 A failed `dd` at a **guessed** path was a manufactured RED

v2's first run reported `RED DISK_FULL` on `thor-wifi`. Thor has **no `/workspace` at all** — and
917 MB/s of headroom on the path it does have. The RED came from a *default*, not a measurement.

`RED DISK_FULL` asserts a **cause** (the MooseFS per-pod quota). An unwritable *guessed* path
establishes only "unknown". Only a host with a hinted `dd_path` may now claim quota; an unhinted
one yields `AMBER DISK_UNVERIFIED: … headroom is UNKNOWN`. Absence of evidence stays an alarm —
but the honest one. This is the mirror image of the tool's founding rule, and it is worth naming:
**a monitor that manufactures alarms from guesses gets muted, which is how it ends up reporting
nothing at all.**

### 2.3 `thor-wifi` is the same Jetson on a second interface

Endpoint dedup cannot merge it — two interfaces have different IPs *by definition* — so it is
hinted explicitly. Left alone it read as a separate unclassified host with a phantom disk failure.

## 3. Live fleet state, MEASURED 2026-08-03 ~06:50 CEST (08 hosts, 28.1 s; 2 hosts, 4.6 s)

| host | verdict | evidence |
|---|---|---|
| **pod5** | **GREEN** | `flagship-v5f-w120-30k` step **1250**, log 51 s fresh, GPU **85 %** / 16 071 MiB, disk 388 MB/s |
| **pod2** | **RED** | `GPU_IDLE_NO_TRAINER` 0 % + **`DISK_FULL`** (100 MB write refused) |
| **pod4** | **RED** | `GPU_IDLE_NO_TRAINER` 0 %, 0 MiB; disk fine (433 MB/s) |
| pod1, pod3, eval | RED | ssh rc=255 `Connection refused` — consistent with `POD_SHUTDOWN_2026-08-02.md` |
| thor, thor-wifi | AMBER | idle by design (`role=edge`); disk 870–917 MB/s; `nvidia-smi --query-gpu` returns nothing on Jetson |

**⚠️ For the PI — two paid A40s (pod2, pod4) are at 0 % with no trainer.** The 08-02 shutdown note
records **$3.61 of credit** and says stopping pods is a **console action only** (`runpodctl` on the
pods is unauthenticated, and there is no RunPod key in `Keys.txt`). I cannot stop them. pod2 is
additionally out of `/workspace` quota, so it could not resume a checkpoint even if refilled.

**Second item for the PI:** `nvidia-smi --query-gpu` returns nothing on Thor (Jetson exposes
`tegrastats`, not the desktop query interface). Thor's GPU is therefore **unmonitored** by this
probe — an honest AMBER, and a real gap once Thor runs inference. → new backlog item.

## 4. What this changes for the programme

- **The `fleet-status` skill is now correct in the tree**, not on a branch. It calls the probe and
  forbids hand-written greps. (13 days late — see §0.)
- **Liveness of the live flagship is now positively verified**, with a step and a log age, by a
  binding that cannot go stale when the next arm is renamed. Every prior "v5f is training" claim in
  this window was INHERITED from a launch message; this one is MEASURED.
- **The stale-name defect class is now closed at both levels it has appeared at** (log names, host
  names). It will appear at a third — the honest expectation is that `JOB_RE` is next, since it
  still encodes what a trainer's *cmdline* looks like.

## 5. Evidence classes

| claim | class |
|---|---|
| 8 endpoints in the ssh config, v1 knew 4 | **MEASURED** — `parse_ssh_config` on the live file + v1/v2 runs |
| pod5 running v5f, step 1250, log 51 s | **MEASURED** — `tools/fleet_probe.py`, and `/proc/19412/fd/1` read directly over ssh |
| pod2/pod4 idle A40s; pod2 disk full | **MEASURED** — live probe, real 100 MB `dd` |
| pod4 ran `v1arch-on-v2bal` | **INHERITED** — `POD_SHUTDOWN_2026-08-02.md`; not re-verified (pod4 is idle now) |
| $3.61 credit remaining | **INHERITED** — same doc, 2026-08-02; the balance today is unknown to me |
| 164 falsifiers, 38.3 s | **MEASURED** — `pytest tools/tests/` |
| discovery falsifiers are real | **MEASURED** — mutation: reintroducing v1's hardcoded membership fails 3 of them |

## 6. Falsifier verdicts

| pre-registered falsifier | verdict |
|---|---|
| "the hardcoded host list is harmless because the fleet rarely changes" | ❌ **REFUTED** — 4 of 8 endpoints unknown to it, including the only working host |
| "the v1 `NO_LOG_BOUND` on pod5 is a probe bug" | ❌ **REFUTED** — it was correct; there was no evidence to bind on. New evidence, not a looser check, resolved it |
| "a failed `dd` proves a full quota" | ❌ **REFUTED** — thor-wifi: write failed, 917 MB/s available |
| "reintroducing hardcoded membership still passes the suite" | ❌ **REFUTED** — 3 falsifiers fail (mutation-checked) |
