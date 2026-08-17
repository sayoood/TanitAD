# Thor concurrency pilot — can the w120 extraction run alongside the live 30k training?

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` · **Status:** IN PROGRESS (during/after
measurement pending)

**PI question:** can the w120 Alpamayo extraction run on Thor *concurrently* with `v6F-SW-30k`, or
must it wait ~5.3 days for the GPU? The extraction needs no GPU, so concurrency recovers five days.

---

## 0. Headline — read this first

> ⛔ **THE ABORT CRITERION IN THE BRIEF IS STRUCTURALLY UNABLE TO FIRE, AND THE PILOT WOULD HAVE
> REPORTED "SAFE" NO MATTER WHAT HAPPENED.** `train_v6_staged.py`'s `step_s` is a **cumulative mean
> over every step since process start**, not a per-step time — the log says so itself. At the
> intended +5 % trip point the cumulative mean **never reaches 28.0 at any duration**. Corrected
> instrument: first-difference it (`stepwatch.py`). Everything below uses the corrected metric.

> ⭐ **THOR'S OWN THROUGHPUT IS 11.76 MB/s — 4.7× the 2.5 MB/s dev-box figure the plan was built on**
> (MEASURED, n=1 sustained 60 s, 707.8 MB). This is the number the brief flagged as most valuable
> and it changes the extraction plan independently of the concurrency answer.

> ⚠️ **PARITY CLAIM IN THE BRIEF IS WRONG: 201 of the 4,729 Alpamayo clips ARE already built in the
> parity corpus directory on Thor.** The brief states these clips are *not* in
> `physicalai-train-e438721ae894`. MEASURED overlap is **201**, not 0.

---

## 1. ⛔ The instrument defect (this is the main finding)

### 1.1 What the log actually reports

Every trainer line carries its own disclaimer:

```
"step_s": 26.4749,
"step_s_note": "elapsed/step over the 6300 steps THIS process ran
                (NOT accumulated over --log-every, and NOT divided by the resumed step number)"
```

`step_s` is **elapsed ÷ N**, where N is *every step this process has run* (currently 6,350). It is a
**cumulative mean**, and a cumulative mean over thousands of steps is almost perfectly insensitive to
a transient perturbation lasting tens of steps.

### 1.2 Proof that the criterion cannot fire

MEASURED at the pilot's start: N = 6,350 steps, elapsed = 168,109.9 s. Driving the cumulative mean to
the brief's 28.0 threshold needs `k` further steps at instantaneous rate `r`, where
`k·(r − 28) = 28·N − elapsed = 9,690`:

| instantaneous rate `r` | vs baseline | steps to trip `step_s > 28.0` | wall-clock |
|---|---|---|---|
| **27.7 s/step** | **+5 % (the intended trip point)** | **NEVER** (asymptote 27.7) | **∞** |
| 28.5 s/step | +8.1 % | 19,380 | 153 h |
| 30.0 s/step | +13.8 % | 4,845 | 40 h |
| 40.0 s/step | +51.7 % | 808 | 9.0 h |
| 53.0 s/step | +101 % (2× slower) | 388 | 5.7 h |

The pilot lasts ~1–2 h. **Nothing short of a total stall could have moved the briefed metric**, so a
"no effect detected" result from it would have carried no information.

### 1.3 The "26.47–26.66 all day" band is not variation either

That band is a **converging mean decaying monotonically**: over the 100 most recent pre-load points
`step_s` never once rose (verified: strictly non-increasing). It falls 26.5505 → 26.4749 across the
last 51 points purely because the process's slow start is being averaged away. Reading it as day
variation would set a false sense of the noise floor.

### 1.4 The corrected metric

Each line carries `step_s` and `N`, so `elapsed_i = step_s_i · N_i` is recoverable exactly and

```
r_inst(i) = (step_s_i·N_i − step_s_{i−1}·N_{i−1}) / (N_i − N_{i−1})
```

is the true mean per-step time over just that 50-step window. Implemented in `stepwatch.py`.

**Root-cause class:** *a counter that aggregates the wrong scope, read as if it answered the
question* — the same family as `df` on a pod (cluster not quota), Thor's `free`/`tegrastats`
(unified memory), and cgroup `memory.usage_in_bytes` (page cache read as pressure). All four are
already in `CLAUDE.md`; this is a fifth instance, and the first one **inside the trainer's own log**.
It also belongs to the C9/C13/C14 family: *an instrument structurally unable to report the answer it
is cited for.*

### 1.5 Corrected abort criteria actually armed

| trip | threshold | rationale |
|---|---|---|
| `SLOW2` | `r_inst > 27.69` on **two consecutive** points | baseline median × 1.05 — preserves the PI's +5 % intent |
| `SLOW1` | `r_inst > 30.00` on **one** point | fast tripwire: two consecutive points is 44 min of degraded training |
| `STALE` | trainer log older than 1500 s | normal cadence ~1318 s |
| `GONE` | PID 25477 or 42229 absent (`kill -0`, explicit PID) | never a pattern match |
| `DISK` | free < 300 GB | started at 366 GB |

Armed as a **self-acting Thor-side watchdog** (`pilot_watchdog.sh`), not a client-side poller,
because the trainer logs once per ~22 min and a laptop-side poller can be asleep when a criterion
trips. It signals **exactly one PID** — the build's, read from `build.pid`.

---

## 2. Baseline (pre-load) — MEASURED

`~/experiments/v6F-SW-30k/train_log.jsonl`, current process only (started step 6,250), steps
6,300–12,600, **127 logged points → 126 instantaneous**:

| metric | median | IQR | min | max |
|---|---|---|---|---|
| **`r_inst` (s/step, TRUE)** | **26.3672** | [26.3130, 26.4447] | 26.2422 | 27.2105 |
| `step_s` (cumulative mean — *not* usable) | 26.5827 | [26.5111, 26.8021] | 26.4740 | 27.5893 |
| `gnorm` | 529.34 | [297.40, 722.56] | 41.84 | 1366.70 |
| `loss` | 2.4375 | [2.0342, 3.0006] | 1.3347 | 5.4976 |

n = 126 instantaneous points spanning ~46 h of training — far beyond the ≥6 the brief required, and
obtained at **zero waiting cost** because the points were already logged before any load was added.
Baseline max (27.2105) sits **below** the SLOW2 trip (27.69), so the threshold is above all observed
pre-load variation.

---

## 3. ⭐ Thor's own throughput — MEASURED

The plan's 2.5 MB/s was the **dev box's** figure, n=1. Thor's own link, measured from Thor:

| probe | bytes | seconds | rate |
|---|---|---|---|
| HF camera chunk 0176 (bounded GET, `/dev/null`, idle-ish box) | 707,780,514 | 60.16 | **11.76 MB/s** |
| PyPI PyAV aarch64 wheel (independent host) | 33.6 MB | ~2.5 | ~13.6 MB/s |

**Thor is ~4.7× faster than the figure the extraction plan assumed.** Two independent hosts agree at
~12–14 MB/s, so this is a property of Thor's link, not of HF.

Chunk 0176 is 1,314,515,695 B (1.31 GB). Full-corpus extrapolation at 11.76 MB/s for 1,418 chunks at
~1.2 GB each ≈ **1.7 TB ≈ 41 h of download**, versus ~193 h at the assumed 2.5 MB/s.
⚠️ ESTIMATED — chunk sizes vary and only two were sized directly.

---

## 4. Corpus / parity facts — MEASURED

Densest-first ordering rebuilt from `clip_index.parquet` ∩ the Alpamayo clip list. It **reproduces
the brief's figures exactly**, which cross-validates both inputs:

- 4,729 Alpamayo clips (from `records.parquet`, `n_unique_clip_id = 4729`), **all 4,729 present** in
  the 306,152-row catalog, all `clip_is_valid = True`
- **1,418 chunks touched**, **max 76 clips/chunk, median 2** — matches the brief
- catalog splits: train 2,786 / val 1,250 / test 693
- top-10 densest chunks: **176(76), 178(60), 175(56), 182(52), 170(43), 295(41), 179(38), 181(37),
  180(37), 185(36) = 476 clips**

### 4.1 ⚠️ The parity non-overlap claim is wrong

The brief states these clips are **not** in `physicalai-train-e438721ae894`. MEASURED against the
built parity cache on Thor (`~/data/physicalai-train-e438721ae894-w120-256x640cyl`, 2,400
`.v2ep.pt`): **overlap = 201 clips**, e.g. `0089a096-68be-40df-8097-780bf1ae1c19`.

That 201 is exactly the size of the recently unified aug120 perception corpus, which is very likely
where the overlap comes from — but the number to act on is that **the sets are not disjoint**.

**What this does and does not mean.** It is **not** a parity violation: the pilot writes to a
separate `--out` and re-selects nothing, so `physicalai-train-e438721ae894` is untouched. It **is** a
correction that matters downstream, because (a) 201 clips would be built twice, and (b) anyone acting
on "these clips are not in the parity set" could later merge the new corpus into the parity corpus
believing it disjoint, which **would** break cross-arm comparability. The new corpus remains a
**separate labelled corpus, never an extension of the parity set.**

---

## 5. Environment facts found on the way (each cost a step)

- ⚠️ **PyAV was missing from ALL THREE pythons on Thor** — `/usr/bin/python3`, `tanitad-edge` and
  `tanitad-train` (three probes, per the absence rule). The extraction cannot run without it.
  Installed **into `tanitad-edge` only**, as `pip install --no-deps --only-binary=:all: av`:
  `--no-deps` so nothing can drag torch forward (the `uv pip` trap that has twice landed an
  unrunnable torch on pods), `--only-binary` so a missing wheel fails instead of compiling ffmpeg on
  a training box. Landed `av 18.1.0` from a prebuilt aarch64 wheel; **torch `2.13.0+cu130` and
  torchvision `0.28.0+cu130` verified unchanged after.**
- ⛔ **`tanitad-train` — the venv the live trainer runs from — was never written to.** It is also
  missing `pandas` and `pyarrow`, so it could not have run the builder anyway.
- **No raw PhysicalAI corpus on Thor.** No `camera/` tree, no `PhysicalAI*` directory; only the
  *built* caches under `~/data/`. Every byte of the extraction is a fresh download — which is why
  §3's throughput number governs the whole plan.
- Thor: 14 CPUs, load average **0.57** with the trainer running — the trainer is GPU-bound and the
  CPUs are ~4 % busy. This is the structural reason concurrency was worth testing.
- Free disk at start: **366 GB**.
- Target geometry taken from the existing corpus manifest, not guessed:
  `--hfov 120 --height 256 --width 640 --projection-mode cylindrical --codec png`
  (`frame_tag: 256x640f305.5775cyl`).

---

## 6. Method

Reused the banked builder `stack/scripts/v2_compressed.py build` unchanged — no new builder written.
Density was controlled by restricting `--sel` to the top-10 chunks, because `build()` iterates
`sorted(by_chunk)` and has no ordering flag.

Conservative by construction: `nice -n 19`, single process, `OMP_NUM_THREADS=3`,
`V2_TORCH_THREADS=3`, `PAI_DECODE_THREADS=3` (torch otherwise spawns ~113 threads per process, which
has previously dropped a GPU to 0–6 % `sm` for 50 minutes looking exactly like a hang).

HF token read **in place** from `~/.cache/huggingface/token` into the environment; never printed,
never passed in argv, never copied into the repo.

---

## 7. Results — during and after load

*(pending — filled in when the run completes)*

---

## 8. Deliverable manifest

*(see report)*
