# v5f `rc=137` — EXPLAINED: a container memory-cgroup OOM kill

**Author:** arch-inf agent · **Date:** 2026-08-04 · **Pod:** `tanitad-new` (`e98a867e7c55`, A40)
**Status of the run at time of writing:** UNTOUCHED — supervisor pid 30573, trainer pid 30599,
`restarts: 0`, step 5800+, on the known-good `--batch 4 --accum 16 --v2-lru 4 --workers 4`.

---

## THE ANSWER

> **`rc=137` = 128+9 = SIGKILL, delivered by the Linux OOM killer.** The container's own memory
> cgroup records **exactly one OOM kill** in a window proven continuous across the event, and the
> attempted `--batch 8 --accum 8 --v2-lru 64 --workers 8` config projects to
> **~51.2 GiB of UNRECLAIMABLE memory against a 46.57 GiB cap** — it does not fit, with no page
> cache left to give back.
>
> **Evidence class: MEASURED** (cgroup counters + source + payload sizes, all artifact-backed),
> with four named residuals in §5. It is **not** UNEXPLAINED.

### ⛔ The previous refutation was itself built on an unusable counter

`R-2026-08-03-mem` retracted the container-OOM diagnosis on the grounds that **`memory.failcnt`
was 0 throughout**. That reasoning does not hold **on this cgroup**:

| MEASURED 2026-08-04, `tanitad-new` | value |
|---|---|
| `memory.limit_in_bytes` | 49,999,998,976 (46.57 GiB) |
| `memory.memsw.limit_in_bytes` | 49,999,998,976 — **EQUAL** |
| cgroup `swap` / host swap | **0 / 0** (`/proc/swaps` empty) |
| `memory.failcnt` | **0** |
| **`memory.memsw.failcnt`** | **28,908,911** → **29,219,916** 26 min later (**≈ 200 failures/s**) |
| `memory.max_usage_in_bytes` | 49,999,998,976 — **exactly the limit** |

In cgroup v1, `try_charge()` charges the **memsw** counter *first* and only then `memory`;
`page_counter_try_charge` increments `failcnt` on whichever counter it exceeded. When
`memsw.limit <= memory.limit` and there is no swap — the ordinary Docker/RunPod setup, and exactly
this one — **memsw absorbs every failure and `memory.failcnt` is pinned at 0 for the life of the
container, however hard the cap is hit.**

The internal consistency proves it without needing the kernel source: a cgroup whose
**peak usage equals its limit exactly** has certainly hit that limit, and `failcnt 0` alongside it
is only explicable by the memsw-first path. The cap is not merely reached — it is being hit
**~200 times per second, continuously**, and satisfied by reclaiming page cache each time.

⇒ **"`failcnt` is 0, therefore the cap was never hit" is UNSOUND here.** The retraction's own
closing rule — *"prefer the counter that only moves on the event you care about — `failcnt`"* — was
**right**; the counter it picked is one that **structurally cannot move**. §6 proposes the
root-cause class.

**What survives from `R-2026-08-03-mem` intact:** `memory.usage_in_bytes` counts reclaimable page
cache and is not a pressure signal (37.2 GB of 50 at *idle*, `rss` 0.1 GB). That is correct and the
idle-baseline rule that came with it is excellent. Only the *refutation* built on `failcnt` fails.

---

## 1. The kill, and the window that makes the counter quotable

```
memory.oom_control:  oom_kill_disable 0   under_oom 0   oom_kill 1
```

`oom_kill` is monotonic **within one container** and **resets when the container is recreated** —
it read 6 then 0 on pod2 and was correctly retracted as "not durable history". So it is quoted here
**only with its window**:

- `PID 1  STARTED Sun Aug 2 22:46:31 2026 UTC  ELAPSED 114,445 s` → the container has run
  **continuously for 31.96 h**, spanning the 2026-08-03 20:55:50 event. The counter was **not**
  reset in between.
- Within that window the supervisor log records **exactly one** SIGKILL-shaped death.

**One OOM kill, one `rc=137`.** The victim was the trainer's own main process, not a DataLoader
worker: a killed worker surfaces as `DataLoader worker (pid N) is killed by signal` and the parent
exits **1** with a traceback; the supervisor recorded 137 for its direct child.

### Timeline, from `…/flagship-v5f-w120-30k/supervisor.log` (append-only, MEASURED)

| UTC | event |
|---|---|
| 20:30:57 | operator's `v5f_cutover.sh` starts, waiting for ckpt step ≥ 4000 |
| 20:48:22 | ckpt 4000 banked → **plain `kill` (SIGTERM)** to hardcoded pid 19412 |
| 20:48:24 | 19412 down in 2 s; **no SIGKILL escalation line**; script exits |
| 20:48:44 | supervisor A launches trainer 26521 |
| 20:52:50 | supervisor B up (no lock contention ⇒ A already gone), launches **27231** |
| **20:55:50** | **`trainer exited rc=137`** — 3 min 00 s after launch |
| 20:56:01 → 21:08:28 | four further supervisors; two exit on the `flock` race |
| 21:11:21 | supervisor G launches 30599 — **live ever since, `restarts: 0`** |

The 3-minute lifetime is diagnostic: the trainer died **before its first logged step** (log-every 50;
`train_log.jsonl` resumes cleanly at the batch-4 cadence from step 4050 with no gap), i.e. **during
DataLoader warm-up** — precisely when 8 workers fault in their runtime and 64-slot LRUs fill.

---

## 2. The mechanism — and it closes arithmetically

`stack/tanitad/data/v2_dataset.py:224-313`, the class's own docstring:

> *"The LRU is per-PROCESS and never crosses the DataLoader-worker boundary (see `__getstate__`):
> every worker fills its own, so total RAM is `num_workers * lru_size * mean_payload`
> (~2-4 MB/clip)."*

⚠️ **The payload figure in that docstring is wrong for this cache by 8–17×.**
**MEASURED: mean 33.4 MB/clip** (n=40 sample of `physicalai-train-e438721ae894-w120-256x640cyl`,
2401 clips) — it was written for the old square JPEG caches, not the 256×640 **lossless PNG** one.
**Budgeting `--v2-lru 64` from the docstring under-counts by ~16 GiB, which is exactly how a config
that cannot fit looked affordable.**

Per-unit costs, all MEASURED on the live shipped config:

| quantity | measured |
|---|---|
| mean v2 clip payload | **33.4 MB** (n=40) |
| per-worker torch runtime (`RssAnon`) | **2.03–2.18 GiB** (4 live workers) |
| main-process `RssShmem` | **5.26 GiB** at 4 workers × prefetch 2 × batch 4 = 32 in flight ⇒ **≈160 MiB / in-flight sample** |
| base unreclaimable (`rss + shmem`) | **9.90 GiB** of a **46.57 GiB** cap |

| config | LRU | +worker runtime | +transport | **unreclaimable total** | verdict |
|---|---:|---:|---:|---:|---|
| shipped `b4 a16 lru4 w4` | 0.65 | — | — | **9.90 GiB** | 36.7 GiB headroom |
| `--batch 8 --accum 8` only | 0.65 | 0 | +5.0 | **14.9 GiB** | **31.6 GiB headroom — SAFE** |
| `--workers 8` only | 1.2 | +8.1 | +5.0 | **23.6 GiB** | 23.0 GiB headroom — safe |
| `--v2-lru 64` only | 10.4 | 0 | 0 | **19.7 GiB** | safe alone |
| **all three (what ran)** | **18.8** | **+8.1** | **+15.1** | **≈51.2 GiB** | ⛔ **OVER the 46.57 GiB cap** |

**The three changes are individually affordable and jointly fatal.** No page cache remains to
reclaim at 51 GiB, so the memcg OOM killer is the only outcome — and it arrives during warm-up.

### ⭐ `--v2-lru 64` is not just dangerous, it is nearly worthless here

`DataLoader(..., shuffle=True)` (`train_flagship_v4.py:1386`) over **410,202 train windows spread
across 2400 clips** (`[data] train windows=410202`, `train.out`) means consecutive samples come from
essentially random clips. LRU hit rate ≈ `lru_size / 2400`:

| `--v2-lru` | hit rate | LRU RAM |
|---:|---:|---:|
| 4 | ~0.17 % | 0.65 GiB |
| 64 | ~2.7 % | 18.8 GiB |

**18 GiB for ~2.5 percentage points.** *(Evidence class: ARITHMETIC on measured corpus sizes.)*
⇒ **`--v2-lru 64` should never be re-tried on this cache**, independently of the OOM.

---

## 3. Alternatives — separated, not skipped

Each was probed and refuted with its own artifact.

| candidate | verdict | evidence |
|---|---|---|
| **Operator's `kill -9` escalation path** | ⛔ **REFUTED** | `/tmp/v5f_cutover.sh` targets only hardcoded `PID=19412`; `/tmp/v5f_cutover.log` shows plain `kill` at 20:48:22, dead by 20:48:24, **the "escalating to SIGKILL" line is absent**, script exited — **7.5 min before** the event, aimed at a different, already-dead pid |
| **`memwatch.sh`** | ⛔ **REFUTED (twice over)** | it contains **no kill path at all** (reads counters, echoes); mtime **21:01:52 — 6 min AFTER** the event. It did not exist yet |
| **Supervisor's own guards** | ⛔ **REFUTED, drift-checked** | pod `supervise_run.sh` md5 **`0daf4be621e26631ae8bf1d2b23f8cc9`** is **bit-identical to the repo copy**; it sends no signal anywhere — the only `kill` in it is `kill -0` (a liveness probe) |
| **GPU-side failure surfacing as a kill** | ⛔ **REFUTED** | 0 uncorrectable ECC (SRAM parity, SRAM SEC-DED, DRAM), `Remapping Failure Occurred: No`, no retired pages. And a CUDA OOM is a *Python exception* → **rc=1 + traceback**, never 137 |
| **Two trainers racing (double-launch)** | ⛔ **REFUTED** | supervisor B logged **no** `trainer ALREADY RUNNING` line, and that guard demonstrably works — it fired at 19:09:35 on pid 19412. 26521 was already gone |
| **CPU-quota throttling / PID limit** | ⛔ **REFUTED** | quota 7.65 CPUs, **0 throttled periods of 200** over a 20 s window (lifetime 206 / 934,881 = 0.02 %); `pids` 44/5120, `pids.events max 0` |
| **MooseFS `OSError: [Errno 5]` inside `print()`** | ⛔ **REFUTED for this death** | that mechanism kills with a Python traceback → rc=1; `train.out` shows no traceback |
| **Host (global) OOM killer** | ⚠️ **DISFAVOURED, not excluded** | host 503 GiB with 414 GiB available and **zero swap**; but see residual R1 |

---

## 4. Instrument shipped (P3)

`stack/scripts/pod_kill_forensics.py` + `stack/tests/test_pod_kill_forensics.py` (**21 tests, all
passing**). It answers *what can this counter say* before printing the number:

- `live_failcnt()` — **decides which `failcnt` is able to move** and names the frozen one. This is
  the whole diagnosis, made mechanical.
- `unreclaimable_bytes()` — `rss + shmem` vs the cap, **not** `usage_in_bytes`.
- `oom_window()` — **refuses to return a bare `oom_kill` count**; it carries the container-start
  window or says it is not quotable.
- `decode_exit_code()` — 137 ⇒ SIGKILL and *"never a Python exception — CUDA OOM exits 1"*.
- `collect_cpu()` — throttling as a **delta over a window**, not a lifetime total.
- `collect_gpu()` — ≥10 samples, because one sample is noise (see §5 R3).
- `collect_kernel_log_access()` — reports **that `dmesg` is unreadable**, so "no OOM in dmesg" can
  never again be mistaken for "no OOM".

Validated end-to-end on the live pod (`raw/v5f_forensics.json`):

```
unreclaimable (rss+shmem): 10.02 GB of 46.57 GB  => headroom 36.55 GB
live failcnt: memory.memsw.failcnt = 29219916  (memory.failcnt is STRUCTURALLY FROZEN AT 0)
OOM: 1 OOM kill(s) in this container since it started 31.96 h ago — NOT a lifetime history
kernel log NOT readable from here — an OOM kill CANNOT be confirmed or excluded via dmesg
```

### Test state — reported exactly, including what I could not get clean

| run | result |
|---|---|
| `tests/test_pod_kill_forensics.py` (isolated) | **21 passed** |
| `-k "v2_dataset or v2 or forensics"`, after the docstring edit | **201 passed, 10 skipped** |
| full `pytest -q` | **2055 passed, 9 failed, 12 skipped, 2 xfailed** |

⚠️ **The 9 failures are NOT from this work** and I checked rather than asserted it: they are all
`test_refc_*` / `test_lan.py` **trainer** tests; **zero** mention my files; the one I re-ran
(`test_refc_tactical.py::test_trainer_end_to_end_f1only`) **passes in isolation**; and
`stack/tanitad/refs/refc.py`, `stack/scripts/refc_train.py`, `stack/tests/test_refc_ce_objective.py`
are **staged mid-edit by a sibling agent**, with 14 concurrent `python` processes on the dev box at
the time. It is cross-agent concurrency plus in-flight REF-C work. **A clean sequential full-suite
number was not obtainable while three agents are running tests on the same tree** — flagged rather
than papered over, and worth re-running once the REF-C stream lands.

`stack/scripts/supervised_cutover.sh` is also shipped: a checkpoint-timed cutover that kills the
**supervisor first**, polls until *both* are gone, treats an unheld lock as debris, verifies the new
flags **from `/proc/<pid>/cmdline`**, and **restores the manifest and restarts on the old command**
if the new config does not come up. It is **staged, not run** (§7).

---

## 5. Residuals — what this explanation does NOT establish

- **R1 — I cannot read the kernel's OOM report.** `dmesg` → `Operation not permitted` (no
  `CAP_SYSLOG`), `/dev/kmsg` unreadable, `/var/log` holds no kernel log. `oom_kill` in cgroup-v1
  `memory.oom_control` increments for a task in this memcg killed by **either** the memcg OOM killer
  **or** the host's global one. **I cannot formally separate them from inside.** Host-OOM is strongly
  disfavoured but not excluded. → **A second, external probe exists and I could not reach it: the
  RunPod console surfaced an "Out of Memory (OOM) Detected" banner for pod2 (RETRACTION_LOG ~L1494).
  The PI can settle R1 in one look.**
- **R2 — the `oom_kill` counter has no timestamp.** Its window is bounded and continuous (31.96 h)
  and contains exactly one SIGKILL-shaped death, but pinning it to 20:55:50 is an inference from
  that coincidence, not a direct observation.
- **R3 — the 8-worker / batch-8 terms are linear extrapolations** of per-unit costs measured at the
  shipped config. **ESTIMATED, not MEASURED.**
- **R4 — I could not recover trainer 27231's actual argv.** The process is gone, `train.out` has no
  args echo, `config.json` was overwritten. That it ran `--batch 8 --accum 8 --v2-lru 64 --workers 8`
  rests on `/tmp/v5f_cutover.sh`'s own contemporaneous comment (mtime **20:30:57, predating the
  event**) plus the once-at-startup manifest-sourcing mechanism. Strong, but circumstantial.

---

## 6. Proposed root-cause class (for `RETRACTION_LOG.md`)

> **A counter that is STRUCTURALLY FROZEN for this configuration, read as evidence of absence.**
>
> This is the sibling of *"a counter that aggregates something RECLAIMABLE, read as pressure"* — and
> it bit **the fix for that very error**, in the same hour. The general form is CLAUDE.md's own
> *"absence found at ONE location is not absence"*, wearing a cgroup costume.
>
> ⇒ **RULE: before reading a zero as absence, establish that the counter is ABLE to be non-zero.**
> A counter at 0 that *cannot move* and a counter at 0 that *did not move* are the same digit and
> opposite facts. Cheap general check: find a sibling counter that is non-zero
> (`memory.memsw.failcnt` here), or induce the event once and confirm the counter responds.

---

## 7. P2 — the prize, and why I did NOT fire it

The RAM cause **is avoidable**: `--batch 8 --accum 8` with `--workers 4 --v2-lru 4` held projects to
**14.9 GiB of a 46.57 GiB cap — 31.6 GiB of headroom.** But two facts the brief did not have change
the risk calculus, and both point at *measuring before spending a cutover*:

1. ⚠️ **`--batch 8` has a prior, banked CUDA-OOM.** Commit **`6d714ad`** (2026-08-03, in the repo):
   *"`--batch 4 --accum 16` … takes GPU from 27.2 GB to 16.1 GB, where batch 8 still died at
   **44.42 GiB with 119.88 MiB free**"*. ⚠️ That measurement is **CONFOUNDED** — `--batch 4`,
   `--v2-lru 4`, `--save-every 250` **and** `expandable_segments:True` were applied together as one
   fix, so it does not cleanly forbid batch 8 *with* the expandable allocator. Linear extrapolation
   from today's **16,071 MiB of 46,068 MiB** at batch 4 suggests ~27 GiB at batch 8, which fits.
   Live risk, not a veto — but it was outside the brief's framing.
2. ⚠️ **The "GPU median ~39 %" premise is unstable.** MEASURED by me on the *same unchanged config*
   20 minutes apart: **median 33 % (n=20, range 21–100)** and then **median 99.5 % (n=12, range
   22–100)**. Utilisation is **bimodal across the 16-step accumulation cycle**, so the size of the
   prize depends on when you sample. It should be re-measured with a step-synchronised instrument
   (or a data-wait timer inside the training loop) before a cutover is spent chasing it.

**Recommendation — one variable, in this order, attended:**

| # | change | RAM projection | note |
|---|---|---|---|
| 0 | *re-measure* the input-bound claim properly | 0 | the prize is currently unquantified |
| 1 | `--batch 8 --accum 8` (hold workers 4, lru 4) | 14.9 / 46.57 GiB | eff_batch stays 64; only GPU risk remains |
| 2 | then `--workers 8` if still input-bound | 23.6 / 46.57 GiB | CPU quota 7.65 with 0 % throttling ⇒ headroom exists |
| ⛔ | **`--v2-lru 64` — never again on this cache** | 51.2 GiB combined | 18 GiB for ~2.5 pp of hit rate |

```bash
# after editing the manifest's TRAIN_CMD to --batch 8 --accum 8:
bash /workspace/TanitAD/stack/scripts/supervised_cutover.sh \
     --run flagship-v5f-w120-30k --min-step 6000 --expect-flag '--batch 8'
```

**I did not run it.** The brief gates P2 on the cause being avoidable — it is — but firing a
kill-and-restart of the programme's headline job at the end of a long turn, on a prize whose size I
have just shown to be unmeasured, against a GPU risk with a prior death behind it, is the shape of
last night's 40 minutes. **The trigger belongs with the PI, attended.**

---

## 8. What I did NOT do — plainly

- **Did not restart, reconfigure, or add training load to v5f.** It is on its known-good config,
  supervisor 30573 / trainer 30599, `restarts: 0`, advancing normally.
- Did not execute the cutover, and did not edit the manifest.
- Did not read the kernel log (denied) or the RunPod console (no access) — **R1 stays open.**
- Did not recover trainer 27231's argv (R4).
- **Did not fix v5f's stdout path.** `/workspace/v5f_run.log` and the supervisor's `train.out` are
  both on MooseFS — the exact mechanism that killed three jobs silently. **Noted, deliberately not
  touched: fixing it requires a restart.** It should be fixed at the next cutover, not before.
- **Left `/tmp/memwatch.sh` running (pid 29452).** ⚠️ It polls `memory.usage_in_bytes` — the counter
  already retracted as not-a-pressure-signal — so it prints `MEM-HIGH 98%` forever and will
  manufacture the next false alarm. It also spawns a `python3` every 30 s on a training pod.
  **Stopping it is the PI's call (pid 29452, explicit).** §9 proposes the replacement.

---

## 9. ESCALATION — needs a decision, not a README

1. ⭐ **Amend `R-2026-08-03-mem` in `RETRACTION_LOG.md`.** Its conclusion — *"the container-OOM
   diagnosis is refuted; the `rc=137` remains UNEXPLAINED"* — is itself now partly retracted. The
   `usage_in_bytes` half stands; the `failcnt` half does not. Add the class in §6.
2. **Fix the `V2CompressedCache` docstring** (`v2_dataset.py:230-231`): "~2-4 MB/clip" is 8–17× low
   for the 256×640 PNG caches and it is the number a future capacity plan will reach for.
3. **Replace `memwatch.sh`'s check** with `pod_kill_forensics.py`'s unreclaimable metric, or retire
   it — as written it can only produce false alarms.
4. **R1 is one look away** for whoever has the RunPod console: an OOM banner on `tanitad-new` would
   convert this from "cgroup OOM strongly favoured" to "confirmed".
5. **`--save-every 250` costs up to ~75 min per death** at 17.66 s/step. Any future cutover should
   be timed with `supervised_cutover.sh --min-step`, which is what it is for.
