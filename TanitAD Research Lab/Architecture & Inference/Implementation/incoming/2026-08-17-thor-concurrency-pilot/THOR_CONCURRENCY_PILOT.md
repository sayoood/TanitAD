# Thor concurrency pilot — can the w120 extraction run alongside the live 30k training?

**Date:** 2026-08-17/18 · **Branch:** `agent/arch-inf-20260803` · **Status:** COMPLETE

**PI question:** can the w120 Alpamayo extraction run on Thor *concurrently* with `v6F-SW-30k`, or
must it wait ~5.3 days for the GPU? The extraction needs no GPU, so concurrency recovers five days.

---

## 0. Headline — read this first

> ✅ **ANSWER: YES, RUN IT CONCURRENTLY.** The extraction slows training by a **measured +0.53 %**
> (95 % CI +0.28 % … +0.79 %) — real and statistically clear (p = 0.00064), and **the effect
> disappears the moment the load stops** (after vs before: −0.045 %, p = 0.713), which is what makes
> the attribution causal rather than coincidental. That is **~6× below** the PI's +5 % abort
> threshold: **40 minutes** of training time bought **5.3 days** of calendar. All 10 chunks completed,
> **476/476 clips verified by content**, no abort, trainer and snapshot daemon untouched throughout.

> ⛔ **THE ABORT CRITERION IN THE BRIEF IS STRUCTURALLY UNABLE TO FIRE, AND THE PILOT WOULD HAVE
> REPORTED "SAFE" NO MATTER WHAT HAPPENED.** `train_v6_staged.py`'s `step_s` is a **cumulative mean
> over every step since process start**, not a per-step time — the log says so itself. At the
> intended +5 % trip point the cumulative mean **never reaches 28.0 at any duration**. Corrected
> instrument: first-difference it (`stepwatch.py`). Everything below uses the corrected metric.

> ⭐ **THOR'S OWN THROUGHPUT IS 11.76 MB/s — 4.7× the 2.5 MB/s dev-box figure the plan was built on**
> (MEASURED, n=1 sustained 60 s, 707.8 MB). This is the number the brief flagged as most valuable
> and it changes the extraction plan independently of the concurrency answer.

> ⚠️ **PARITY CLAIM IN THE BRIEF IS WRONG, AND IT IS AN EVAL-LEAK RISK: 201 of the 4,729 Alpamayo
> clips are already in the parity TRAIN corpus** — the very cache the live trainer reads
> (`--v2-cache …physicalai-train-e438721ae894-w120-256x640cyl`, from `/proc/25477/cmdline`). The
> brief states these clips are *not* in `physicalai-train-e438721ae894`; MEASURED overlap is **201**,
> not 0. ⇒ Any eval split built from this corpus must exclude them (list banked) or it repeats the
> REF-A I-JEPA leak at 4.3 % scale.

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

### 1.4a How far it has already propagated — and the honest size of the damage

Swept the repo. `step_s` is quoted as the run's pace in at least three live documents:

| site | text | verdict |
|---|---|---|
| `Project Steering/Reports/2026-08-17-2319-program-report.md:21` | `\| marginal pace \| **26.47 s/step** \|` | ⛔ **mislabelled** — 26.47 is the *cumulative* mean; "marginal" is precisely what it is not. True marginal = **26.3672**. |
| `…/2026-08-17-st-launch-readiness/ST_LAUNCH_READINESS.md:4` | "reaches 30 000 in **5.47 days** (MEASURED: step 12 150, 26.4833 s/step)" | ETA built on the cumulative mean |
| `…/ST_LAUNCH_READINESS.md:268` | "the trainer's own per-process figure, not the `--log-every`-accumulated one" | half-right — correctly rules out the `--log-every` trap, still treats a per-process cumulative mean as current pace |

⚠️ **Do not overstate this.** For **ETA** purposes the error is small, because the cumulative mean has
nearly converged to the marginal rate: over the 17,350 steps remaining, 26.4749 vs 26.3672 differ by
**0.52 h across 5.3 days — 0.41 %**. Every ETA in those documents is therefore *substantially
correct*, and this is **not** a retraction of the ~5.3-day finish estimate.

⭐ **The damage is confined to one use — and it is exactly the use this pilot needed.** As a slowly-
moving summary of a whole run, `step_s` is fine. As a **detector of a change happening now** it is
inert, and that is what an abort criterion is. The lesson generalises past this pilot: *a statistic
that is adequate for reporting can be useless for control, and the two uses are easy to conflate
because they quote the same number.*

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
trips. It signals **exactly one PID** — the build's, read from `build.pid`. PIDs 25477 and 42229 are
only ever read with `kill -0`; no pattern match is used anywhere, because `pkill -f <trainer>`
self-matches the caller's own ssh command line and has killed sessions on this programme before.

**The abort path was TESTED, not assumed** — an untested safety mechanism is not one:

| fed `r_inst` | trips | note |
|---|---|---|
| 26.3606 (live baseline) | — | no false fire |
| 27.2105 (baseline max) | — | no false fire at the observed extreme |
| 27.6899 / 27.6901 | — / SLOW2 | boundary is exact |
| 30.01 | SLOW1 + SLOW2 | fast tripwire fires |
| −1.0 (sentinel: <2 rows) | — | insufficient data does not spuriously abort |

The build-exit path is proven by real execution: when attempt 1 died, the watchdog detected it within
60 s, wrote `ZZDONE|build_exited|` and stood down without touching the trainer.

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

⚠️ **This table is the baseline as first collected, and it is CONTAMINATED — see §7.2.** 18 of the
126 points are a post-resume warm-up transient at ~27.1 s/step, which inflates the mean, widens the
spread and makes the distribution bimodal. The **steady-state** baseline used for every comparison in
§7 is **n = 108, median 26.3591, IQR [26.3092, 26.3966], max 26.7032**. The armed abort thresholds
were left as set from this wider table, which only makes them more conservative.

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

**These 201 are genuine parity members, not later contamination of the directory.** Checked by mtime:
the overlapping files were written **2026-08-15 18:39–22:12** and the other 2,199 **18:38–22:16** —
one and the same build pass, from `only_clips: parity_train_clips.txt`. So the 201 are in the parity
train selection itself. (The count coincides with the recently unified 201-clip aug120 perception
corpus; the mtimes rule out aug120 having been written into this directory afterwards.)

**What this does and does not mean.**

- It is **not** a parity violation by this pilot. The build writes to a separate `--out` and
  re-selects nothing, so `physicalai-train-e438721ae894` is untouched. The new corpus stays a
  **separate labelled corpus, never an extension of the parity set.**
- ⛔ **The real hazard is evaluation contamination, and it points the other way.** This is **not
  hypothetical and not historical — it is the live run**. `v6F-SW-30k` (PID 25477) carries
  `--v2-cache /home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl` (read from
  `/proc/25477/cmdline`), which is **the exact directory the 201 overlapping clips live in**. The
  model now training has been seeing those 201 clips for ~5.3 days.

  So if the Alpamayo-labelled corpus is later used as a held-out or OOD evaluation set on the
  strength of "these clips are not in the parity set", **201 of its clips are train-contaminated for
  the flagship arm**. That is the **REF-A I-JEPA leak class** (~80 % of val inside train, which made
  that arm's val number unusable) at smaller scale — 201/4,729 = **4.3 %**. ⇒ **Exclude the 201 from
  any eval split built on this corpus, or report with and without them.** The ready-to-use list is
  banked as `alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt` (201 ids).
- Minor: those 201 clips would also be **built twice**, wasting ~7 GB and ~40 min of extraction.

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

### 6.1 ⚠️ A launch-path defect the pilot found (attempt 1 died after paying for the download)

The first launch **downloaded 536 MB and then died with zero clips built**:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/home/nvidia/w120pilot/pai_root/r0/r0_selection.parquet'
```

`physicalai._chunk_of_clip` (`physicalai.py:282`) reads `<root>/r0/r0_selection.parquet` to map
`clip_id → chunk`, which is how `intrinsics_for_clip` locates that chunk's `camera_intrinsics`
parquet. `build()` never creates it, and the failure lands in `_assert_geometry_deliverable` —
**after** `_ensure_ego` and the ~1.2 GB camera-chunk fetch.

This is the **`t1_eval` class from `CLAUDE.md`**: *an analysis-time dependency that fails after the
expensive part is already paid for*. The durable fix is a preflight existence check at startup so it
fails in milliseconds rather than after a chunk download. Worth noting that the builder's *own*
geometry assert exists for exactly this reason ("ABORT before hours of work, not after") — it just
sits downstream of a missing input it does not itself check.

⭐ **The crash was also load-bearing for correctness, not merely for liveness.** With no
`r0_selection.parquet`, `intrinsics_for_clip` warns once and falls back to the corpus median, which
"reverts the crop to geometric-center → horizon NOT rig-corrected". PhysicalAI front-wide has **two
rigs** (cy ≈ 543 rig A / ≈ 755 rig B), so a geometric-center crop is **~215 px wrong for rig B**. Had
the fallback path been reachable here, the pilot would have produced a **silently mis-cropped
corpus** — the expensive kind of wrong. Fix applied: the selection parquet (which already carries
exactly the `clip_id` + `chunk` columns required) is copied to `<root>/r0/r0_selection.parquet` by
the launcher.

Attempt 2 then ran clean, and its geometry check confirms the target frame is actually delivered:
`achieved_hfov_deg: 120.0`, `f_eff: 305.5775`, `frame_tag: 256x640f305.5775cyl` — byte-identical
framing to the existing w120 corpus. Rig A (n=56) fully observed; rig B (n=4) 8.9 % masked, which is
the cylindrical projection **masking rather than fabricating**, as intended.

---

## 7. Results

### 7.1 The extraction itself — COMPLETED, no abort

| | |
|---|---|
| clips built | **476 / 476** (`[build s0/1] DONE built=476`) |
| chunks | **10 / 10** (170, 175, 176, 178, 179, 180, 181, 182, 185, 295) |
| failures | **0** (`FAILED` count 0 across the whole log) |
| wall-clock | **8,348 s = 2 h 19 m** |
| cache written | **17.07 GB** (~35.9 MB/clip, PNG/lossless) |
| extraction rate | **3.42 clips/min** under concurrent load |
| free disk | 366 GB → **347 GB** low-water (abort floor was 300 GB — never approached) |
| abort | **NONE.** Trainer PID 25477 and snapshot PID 42229 alive at every one of the 141 watchdog polls |

The watchdog stood down cleanly on build exit (`ZZDONE|build_exited|2026-08-18T00:18:08Z`).

### 7.2 ⚠️ A second contaminated baseline, found while analysing — the post-resume WARM-UP

Before the comparison could be trusted, the baseline had to be cleaned a second time. Plotting the
126 pre-load points shows it is **bimodal**: a bulk at 26.24–26.44 and a distinct cluster at
27.04–27.21.

My first hypothesis was periodic checkpointing (`--save-every 250` against `--log-every 50` puts a
save in exactly one window in five). **That is wrong, and the data refutes it outright:** the 18 slow
points are **consecutive** — steps 6350 … 7200, with no periodicity at all (`step % 250 == 0` holds
for only 3 of the 18).

They are the **first 18 windows after the process resumed** (this process started at step 6250). The
run is a warm-up transient of ~900 steps at median **27.1156 s/step**, settling to **26.36** and
staying there. `--v2-lru 64` filling the episode cache is the obvious candidate, though the pilot did
not test that.

⭐ **This also explains §1.3.** The cumulative `step_s` decays monotonically precisely *because* this
900-step warm-up head is being slowly averaged out of it — one mechanism accounts for both the
decaying mean and the bimodal baseline.

⇒ **The warm-up is excluded from the baseline.** Comparing a steady-state loaded phase against a
baseline containing a resume transient would have inflated the baseline and *hidden* the effect —
the opposite error to the one in §1, and just as wrong. Note the direction: the sloppier analysis
would have made concurrency look *better*, not worse.

### 7.3 Step time — BEFORE / DURING / AFTER

Load window **1787003523 → 1787012995** (first launch to the end of the post-build verification —
the last moment any of my load ran). A logged point covers the 50 steps *before* it, so points whose
window straddles a boundary are excluded from both groups.

| phase | steps | n | median | IQR | min | max | Δ vs BEFORE |
|---|---|---|---|---|---|---|---|
| warm-up *(excluded)* | 6350–7200 | 18 | 27.1156 | [27.0421, 27.1666] | 26.9035 | 27.2105 | — |
| **BEFORE** (steady) | 7250–12600 | **108** | **26.3591** | [26.3092, 26.3966] | 26.2422 | 26.7032 | — |
| **DURING** | 12700–12900 | **5** | **26.4993** | [26.4989, 26.5138] | 26.4346 | 26.5652 | **+0.1402 s (+0.532 %)** |
| **AFTER** | 13050–13200 | **4** | **26.3474** | [26.3440, 26.3762] | 26.3365 | 26.4600 | **−0.0117 s (−0.045 %)** |

DURING: 26.4989 · 26.4993 · 26.5652 · 26.5138 · 26.4346
AFTER: 26.4600 · 26.3365 · 26.3483 · 26.3465

### 7.4 ⭐ The answer: a REAL effect, a small one, and it goes away

**Do not round this to "no effect" — and do not round it up either.** The AFTER phase is what turns
this from a correlation into an attribution:

| comparison | median shift | bootstrap 95 % CI (20 k resamples) | Mann-Whitney p |
|---|---|---|---|
| **DURING vs BEFORE** | **+0.532 %** | **[+0.282 %, +0.785 %]** | **0.00064** |
| **AFTER vs BEFORE** | **−0.045 %** | [−0.095 %, +0.384 %] | **0.713** |

**The slowdown appears when the extraction starts and disappears when it stops.** Load removed →
step time returns to baseline, the CI straddles zero and the test is comfortably null. That before /
during / after shape is what rules out the obvious alternative explanation — a slow drift in the
trainer that merely coincided with the pilot. Without the AFTER control the +0.53 % would have been
suggestive; with it, it is causal.

⚠️ **This mattered, and it nearly went the other way.** The FIRST clean after-point was **26.4600**
(+0.38 %), which looked as though the elevation had persisted and the effect was *not* the
extraction. Three further points (26.3365, 26.3483, 26.3465) settled it. **A single after-point
would have produced the wrong conclusion** — in either direction depending on which one landed.

**And the effect is decisively below the threshold that was asked about:**

| quantity | value |
|---|---|
| measured slowdown | **+0.53 %** (95 % CI +0.28 % … +0.79 %) |
| PI's abort threshold | +5 % |
| **margin** | **the CI's upper bound is 6.4× below the trip point** |
| cost over the remaining 17,100 steps | **+0.67 h** (~40 min) on a ~5.3-day run |
| benefit | **~5.3 days** of extraction not spent waiting for the GPU |

Trading **40 minutes** of training for **five days** of calendar is not a close call.

⚠️ **Caveats stated plainly.**
- n_during = **5**, one short of the 6 the brief asked for. The 10-chunk build ran 2 h 19 m and the
  trainer logs once per ~22 min, so **10 chunks cannot yield 6 clean fully-loaded windows** — the
  brief's two requirements are arithmetically incompatible and "10 chunks" was kept as binding.
  n_after = 4.
- All 5 DURING points fall inside the steady baseline's range (max 26.7032): a shift in central
  tendency, not an excursion into new territory.
- Measured for **one** extraction process at `nice -19` with 3 threads. It does **not** license
  running several in parallel — linearity was not tested, and the download rate already showed
  self-contention (§7.6).
- The +0.53 % is specific to this workload mix. Notably the post-build **verification** job (18 GB of
  file reads + PNG decode, 11 min) did **not** visibly slow the trainer, so not all CPU load costs
  the same; the mp4-decode + `grid_sample` path is the expensive part.

### 7.4a Training quality was not affected

| metric | BEFORE (n=108) | DURING (n=5) | AFTER (n=4) |
|---|---|---|---|
| `gnorm` median | 567.92 | 610.73 | 241.80 |
| `loss` median | 2.3772 | 2.0372 | 2.1566 |

`gnorm`'s baseline IQR is [297.40, 722.56] and per-point values swing 42–1367 across the run, so all
three medians sit **well inside** ordinary variation — including AFTER's 241.80, which is low but
unremarkable at n=4 given that spread. Loss likewise. **Nothing suggests the extraction perturbed
optimisation**; it took a small slice of wall-clock and nothing else.
⚠️ These are the trainer's own logged optimisation diagnostics, **not** an eval. This pilot makes no
claim about model quality in the four metric families — no eval was run and none was warranted.

### 7.5 GPU was never starved — with a matched control

`nvidia-smi utilization.gpu` sampled at 1 s, n=90 in each condition:

| condition | median | mean | frac ≥95 % | frac = 0 % |
|---|---|---|---|---|
| DURING load | 98.0 % | 84.06 % | 0.756 | 0.067 |
| **AFTER, no load (control)** | 97.0 % | 85.59 % | 0.767 | 0.033 |

**Indistinguishable.** The trainer is GPU-saturated in both conditions, which is the mechanism for
why the step-time effect is only half a percent: Thor has 14 CPUs, load average peaked at ~2.3, and a
`nice -19` CPU job cannot take work away from a GPU-bound step.

⭐ **The control also settled a loose end.** The dips to 0 % appear in the *unloaded* sample too
(3.3 % vs 6.7 %), so they are **intrinsic to the trainer** — step boundaries or checkpoint writes —
**not** caused by the extraction. Without this matched sample I would have had to leave them
unattributed, and they were the one observation that looked like starvation.
(`clocks.sm` reads `[N/A]` on Thor's tegra driver — another probe simply absent here.)

### 7.5a Output verified BY CONTENT (C77), all 476

Not existence — every payload was decoded through the real `load_compressed()`:

| | |
|---|---|
| checked / ok / failed | **476 / 476 / 0** |
| frame shapes | **`9x256x640` uint8 for all 476** (one shape, no drift) |
| stacked frames | 94,793 total, **199.1 per clip** |
| size | 18.33 GB, **38.52 MB/clip** |
| geometry tag | **`256x640f305.5775cyl`**, codec `png`, achieved hfov **120.0°** |

Every clip also passed finite-value checks on poses and actions and T-agreement across
frames/actions/poses/maneuvers. The geometry tag is **identical to the existing w120 parity corpus**,
so this cache is schema- and frame-compatible with it (while remaining a separate corpus — §4.1).

### 7.6 Throughput, and the reverse direction

| condition | rate |
|---|---|
| HF → Thor, trainer only | **11.76 MB/s** |
| HF → Thor, trainer **+ this extraction** | **8.58 MB/s** |

The extraction's own CPU work costs it ~27 % of its download rate. That is the extraction slowing
*itself*, not the trainer slowing it — and it is the figure to plan the full run with.

---

## 8. ⭐ What the full extraction actually costs — and why the tail should be capped

MEASURED inputs: chunk sizes 1.3145 GB (0176) and ~1.166 GB (0170) → **~1.22 GB/chunk**; Thor
throughput **11.76 MB/s** (trainer-only) / **8.58 MB/s** (trainer + this extraction competing);
output **~36–40 MB/clip** at PNG/lossless; extraction **~4.3 clips/min** under load.

The corpus is **brutally density-skewed** — 1,418 chunks for 4,729 clips, median **2 clips/chunk**.
Downloading a 1.22 GB chunk to extract 2 clips is the dominant cost, so *where you stop* matters far
more than how fast you go:

| chunks (densest-first) | clips | % of corpus | download | at 11.76 MB/s |
|---|---|---|---|---|
| 10 (**this pilot**) | 476 | 10.1 % | 12 GB | 0.3 h |
| 50 | 1,317 | 27.8 % | 61 GB | 1.5 h |
| **176** | **2,368** | **50.1 %** | **215 GB** | **5.2 h** |
| 300 | 2,922 | 61.8 % | 366 GB | 8.9 h |
| 624 | 3,785 | 80.0 % | 761 GB | 18.4 h |
| 946 | 4,257 | 90.0 % | 1,154 GB | 27.9 h |
| 1,418 (all) | 4,729 | 100 % | **1,730 GB** | **41.8 h** |

**The last 10 % of clips costs 472 further chunks — 576 GB and ~14 h — for 472 clips.** Half the
corpus is reachable in 12.4 % of the chunks.

⇒ **Recommendation: cap the extraction by density rather than running all 1,418 chunks.** Stopping at
~50 % of clips costs 5.2 h of download instead of 41.8 h, and stopping at 80 % costs 18.4 h.
⚠️ ESTIMATED — extrapolated from two directly-sized chunks; chunk sizes vary and only 0170/0176 were
measured. Storage at 90 % coverage would be ~162 GB against Thor's 364 GB free, which fits, but a
full run leaves less headroom than is comfortable on a box also holding a live training run.

⚠️ **Selection caveat:** taking the densest chunks is **not** a random sample of the corpus. Chunks
are geographic/temporal collection units, so a density cap biases the resulting corpus toward
whatever the dense chunks contain. If this corpus is to support a distributional claim, the cap must
be justified or the selection stratified — do not let a download-cost decision silently become a
dataset-composition decision.

---

## 9. Recommendation

1. ✅ **Run the w120 extraction concurrently with training.** Measured cost +0.53 % step time
   (CI +0.28 %…+0.79 %); measured benefit ~5.3 days. Keep the exact conditions that were tested:
   **one process, `nice -n 19`, `OMP_NUM_THREADS`/`V2_TORCH_THREADS`/`PAI_DECODE_THREADS` = 3.**
2. ⛔ **Do not parallelise it.** Linearity was not tested and the download already self-contends
   (11.76 → 8.58 MB/s). A second process is a new experiment, not an extension of this one.
3. ⭐ **Cap the extraction by density** (§8). Half the corpus costs 5.2 h of download; the full tail
   costs 41.8 h and 1.73 TB for the last 10 %. Decide the cap deliberately, and record that it is a
   dataset-composition decision, not just a cost one.
4. **Re-arm `pilot_watchdog.sh` for the full run.** It is workload-agnostic — it watches the trainer,
   not the builder — and it proved itself twice tonight (clean stand-down on build exit; boundary-
   tested trip logic).
5. ⛔ **Exclude the 201 parity-overlap clips from any eval split** built on this corpus (§4.1).
6. **Fix the launch path before the full run** (§6.1): `build()` should create or check
   `<root>/r0/r0_selection.parquet` at startup rather than failing after a 1.2 GB download.

## 10. Deliverable manifest

All paths relative to the repo root, **staged, not committed** (per the operating standard). Nothing
was pushed; no branch was switched; `MODEL_REGISTRY.md`, `train_v6_staged.py` and `v6_chain.py` were
not touched.

**Repo — `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-thor-concurrency-pilot/`**

| file | what it is |
|---|---|
| `THOR_CONCURRENCY_PILOT.md` | this report |
| `stepwatch.py` | ⭐ the corrected instrument — first-differences the cumulative `step_s` |
| `analyze_phases.py` | phase classification + Mann-Whitney; recovers wall-clock from `step_s·N` |
| `verify_out.py` | C77 content verification via real `load_compressed()` |
| `pilot_run.sh` | the launcher actually used (carries the `r0_selection.parquet` fix) |
| `pilot_watchdog.sh` | ⭐ self-acting abort watchdog; kills one explicit PID, never a pattern |
| `pilot_probe.sh` | one-shot opaque-marker status probe |
| `raw_baseline.txt`, `baseline.jsonl`, `baseline_curproc.jsonl` | pre-load series |
| `raw_final.txt`, `final.jsonl` | full series through step 13200 (the analysed data) |
| `load_window.txt` | exact load-window epochs |
| `trainer_cmdline.txt` | the protected run's full config, from `/proc/25477/cmdline` |
| `verify_476.json` | **476/476 content verification result** |
| `gpu_during.txt`, `gpu_after.txt` | matched 1 s GPU utilisation samples (n=90 each) |
| `alpamayo_clip_ids.txt` | the 4,729 Alpamayo clip ids |
| `chunk_order_densest_first.json` | all 1,418 chunks, densest-first |
| `pilot_sel_top10.parquet` | the exact 476-clip selection built (force-added; parquet is gitignored) |
| `parity_ls.txt`, `parity_mtimes.txt` | parity cache listing + mtimes (the overlap evidence) |
| **`alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt`** | ⛔ **the 201 leaked clip ids — actionable** |
| `out_geometry.json` | geometry manifest of the built corpus |
| `clip_index.parquet`, `alpamayo_sel_full.parquet` | inputs (gitignored, not staged; reproducible) |

**On Thor (`thor6`) — `/home/nvidia/w120pilot/`**

| path | what it is |
|---|---|
| `out/` | ⭐ **476 `.v2ep.pt` + `_geometry.json`, 18.33 GB** — the built corpus |
| `build.log`, `build.attempt1.log` | builder logs (attempt 1 = the `r0_selection` failure) |
| `watchdog.log`, `watchdog.attempt1.log` | 141 abort-probe records |
| `verify.json` | the 476/476 verification |
| `pilot_run.sh`, `pilot_watchdog.sh`, `pilot_probe.sh`, `verify_out.py` | md5-verified copies |
| `pai_root/` | transient download root (zips deleted per chunk by the builder) |

⚠️ **The 476-clip corpus lives ONLY on Thor** (18.33 GB — too large for the repo). It is **not** yet
banked elsewhere. If Thor is lost it costs 2 h 19 m to rebuild, and the recipe to do so is fully
staged (`pilot_run.sh` + `pilot_sel_top10.parquet`), so this is a recorded risk, not a stranded
artifact.

**Escalations (also filed as background tasks):**
1. `train_v6_staged.py` should log a true per-window step time — I am forbidden to touch that file.
2. The 201-clip exclusion must be applied wherever an Alpamayo eval split is defined.
3. `Project Steering/Reports/2026-08-17-2319-program-report.md:21` mislabels 26.47 as "marginal
   pace" (true marginal 26.3672). Labelling fix only — the ~5.3-day ETA stands.
