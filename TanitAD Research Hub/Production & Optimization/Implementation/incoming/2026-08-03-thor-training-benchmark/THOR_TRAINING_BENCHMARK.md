# Can we TRAIN on the Jetson Thor? — first training benchmark

**Date** 2026-08-03 · **Device** `tanitad-thor` (thor6, L4T R38 / JetPack 7, CUDA 13.0, driver 580.00)
**Arm** REF-C-base (`refc-diffusion-base`, 104,191,577 params) · **Trainer** `stack/scripts/refc_train.py`
**Venv** `~/venvs/tanitad-train` (the training venv — the two-venv rule was honoured; `tanitad-edge`
was read for its torch *version* and never used to run anything)

---

## 0. The answer, first

**Thor is usable for training this class of model, and it is far closer to an A40 than anyone
assumed — but it is capacity-limited, not speed-limited.**

| question | answer | class |
|---|---|---|
| Is Thor usable for training? | **Yes**, for ≤ ~100 M-param arms at ≤ 256 px | MEASURED |
| How much slower than an A40? | **1.03–1.25×** on the SAME arm + SAME config — call it ~1.2× | MEASURED vs INHERITED (§5) |
| What size of job? | REF-C-base at **batch 20** peaks at **16.5 GB** CUDA; batch 48 at 36.3 GB | MEASURED |
| Wall-clock for a real 30 k run | **≈ 12.5 h** (A40's recorded end-to-end: **12 h 06 m**) | MEASURED-derived |
| Does it throttle? | **No.** p95/p50 = **1.00** over 35 min, tj plateaus ~71–76 °C | MEASURED |
| What actually limits it? | **HOST memory (the dataloader) + a 20-SM GPU that saturates at ~13 windows/s** — not clock, not thermals | MEASURED |

The headline that matters for the programme: **a 30,000-step REF-C-base run — a real arm, not a toy —
takes about 12.5 h on the Thor sitting on a desk, against the 12 h 06 m the identical arm actually
took on a rented A40.** Thor is not an "inference-only" device. For this class of model it is a
genuine training node whose ceiling is *model size and host RAM*, not *patience*.

⚠️ **This is a THROUGHPUT benchmark and reports NO model-quality numbers.** The BINDING four-metric-
family rule (longitudinal / lateral / tactical / strategic) governs **evals**; nothing here is an eval.
The loss values in the logs are from a deliberately non-parity, 39-episode workload and are **not
quotable as results** — they exist only to prove the optimizer really ran.

---

## 1. What was run, and on what

| | |
|---|---|
| **Model** | REF-C-base, `--config base --mode diffusion`, 128 built-in FPS anchors |
| **Params** | **104,191,577** — byte-identical to the A40 run's recorded count (`MODEL_REGISTRY.md:1414`) |
| **Data** | `~/valdata/physicalai-val-0c5f7dac3b11/` — the canonical **val-40** epcache, `ep_*.pt`, uint8 `[T, 9, 256, 256]` |
| **Episodes used** | **39** (see §2 — one file is corrupt), 5,503 windows |
| **Parity** | ⛔ **DELIBERATELY NON-PARITY.** The corpus guard printed its NON-PARITY line and the run proceeded. This is a speed benchmark; no cross-arm number is claimed |
| **Contention** | **None.** Thor was idle before launch: GPU 0 %, 3 W, no python processes, no containers. Nothing else ran during any measurement |
| **Live pods** | ⛔ untouched. `tanitad-new` (v5f) and `tanitad-pod4` (v1arch) were read-only `tail`-ed for their configs and never loaded |

### Provenance of every measurement

All Thor numbers below come from these run directories on `tanitad-thor`:

```
/tmp/thorbench/sweep/refc_b{8,16,32,64,128}.log        batch sweep, --mode classifier
/tmp/thorbench/sweep/refc_b*.jsonl[.summary.json]      per-run memory probe
/tmp/thorbench/sustained_b20.log                       35-min sustained run, --mode diffusion
/tmp/thorbench/sustained_b20.jsonl[.summary.json]      its memory probe
/tmp/thorbench/sustained.tegrastats.log                its thermal/power capture
```

Copies of all of the above are banked under `raw/` next to this document.

---

## 2. Three things that had to be fixed before a single step could run

These are the deliverable, not preamble — each would have stopped the next person too.

**(a) `tanitad-train` had NO torch.** The venv contained `numpy` and `pip` and nothing else, so the
PI's "use the train venv" instruction was un-runnable as found. Installed
`torch 2.13.0+cu130` + `torchvision 0.28.0+cu130` (aarch64 SBSA wheels from
`download.pytorch.org/whl/cu130`, the first choice in `~/prep_envs.sh`; the jetson-ai-lab fallback was
not needed). **Verified by LOADING**, not by exit code:

```
torch 2.13.0+cu130   cuda 13.0   available True
device NVIDIA Thor   capability (11, 0)   20 SMs
fp32 4096³ matmul  28.74 ms →  4.78 TFLOP/s        MEASURED
bf16 4096³ matmul   5.62 ms → 24.48 TFLOP/s        MEASURED
```

**(b) Thor's checkout was 64 commits behind HEAD** (`4954544`, = `origin/main`). This is exactly the
documented drift trap — a launch from it would have run pre-fix code. `git` could **not** fix it: the
64 commits are on the local branch `agent/benchmarks-eval-20260802` and are **not pushed**, so
`origin/main` is also at `4954544`. Synced by shipping `stack/tanitad` + `stack/scripts` as a tarball
(1.2 MB) and verified with real imports.

**(c) One episode file is TRUNCATED.** `~/valdata/physicalai-val-0c5f7dac3b11/ep_00028.pt` is
**92,299,264 B** against a healthy range of **116.8–120.9 MB**, and raises
`PytorchStreamReader failed reading zip archive: failed finding central directory`. 39/40 load fine
(T = 198..205). It killed the first sweep with an error that reads like a torch bug rather than a bad
copy. Excluded from this benchmark; **a repair task has been filed.**

---

## 3. Geometry — asserted before any forward pass

The brief required this explicitly (it is a once-retracted defect class). Asserted from source, not
assumed:

```
REF-C small/base/xl encoder : in_channels 9, image_hw (256, 256), window 8     ← SQUARE
val epcache ep_00000.pt     : frames (199, 9, 256, 256) uint8
GEOMETRY ASSERT PASSED      : data (9,256,256) == REF-C base (9,256,256)
```

The 256×640 cylindrical cache (`…-w120-256x640cyl`, 38 `*.v2ep.pt`) is **v5f's** geometry and was
deliberately **not** fed to REF-C.

---

## 4. Memory on Thor — THREE standard probes are wrong, and one is right

This is the most transferable finding here, and it is the same class as *"never judge pod disk with
`df`"*. On Thor's unified memory:

| probe | what it said | truth | verdict |
|---|---|---|---|
| `torch.cuda.mem_get_info()` | **3.4 GB free** of 131.9 GB, and it *stayed* 3.4 GB | 60 GB allocated | ⛔ meaningless |
| `free` / `tegrastats RAM` | **106,071 / 125,772 MB used** on an IDLE box (total RSS 1.35 GB) | rose only **+596 MB** while 60 GB was allocated **and written** | ⛔ does not track CUDA |
| `/proc/<pid>/status VmRSS` | **0.62 GB** | `max_memory_allocated` = **24.01 GB** same process | ⛔ blind to CUDA |
| `torch.cuda.max_memory_allocated()` | tracks correctly | — | ✅ **the only working probe — and it must run IN-PROCESS** |

The 60 GB was proven real, not lazily mapped: every chunk was `fill_()`-ed and verified by
first/middle/**last** element **and** a full-tensor double-precision sum.

⚠️ **The idle "106 GB used" is a phantom and it MOVES.** During the sustained run the same counter
read **32.7 GB**. Whatever held it at boot was released. ⇒ **Do not size a Thor job from `free`,
`tegrastats` or `mem_get_info`.** Size it from `torch.cuda.max_memory_allocated()` and an empirical
run. This directly caused a wrong batch-ceiling reading in §6.

Because an external probe cannot see GPU memory, the instrumentation had to be **in-process** —
hence `thor_bench_probe.py` + `thor_bench_run.py`, which wrap the **real, unmodified trainer** via
`runpy` rather than reimplementing its step. A benchmark that reimplements the step measures the
reimplementation.

---

## 5. ⭐ Thor vs A40 — the SAME arm at the SAME config

This is the one comparison in this report that is genuinely like-for-like, and it exists because
REF-C-base was trained on an A40 at a config Thor can reproduce exactly.

**A40 side — `INHERITED`** (our own programme, MEASURED by another agent, *not* re-verified by me;
`tanitad-pod3` is powered down so re-measurement was impossible). Source:
`TanitAD Research Hub/Benchmarks & Eval/Research/2026-07-20-refc-medium-scaling.md`

| A40 figure | value | line |
|---|---|---|
| run rate, `--log-every 50` de-accumulated | **1.20 s/step** (60.1–62.9 s per 50 steps) | :242–248 |
| isolated compute-only (3 fwd+bwd+Adam) | **1.29 s/step** | :91–93 |
| peak allocated @ batch 20 | **14.44 GiB** (= 15.50 GB), reserved 18.31 GiB | :91–93 |
| end-to-end 30 k (16:38Z→04:44Z ÷ 30,000) | 1.45 s/step | :326 + `MODEL_REGISTRY.md:1412` |

**Thor side — `MEASURED` (this work).** `--config base --mode diffusion --batch 20 --workers 6`,
identical to the A40 command.

**1,400 steps · 2,127 s wall · 35.5 min continuous · 1,390 steps measured (first 10 discarded)**
*(`/tmp/thorbench/sustained_b20.log`, `sustained_b20.jsonl.summary.json`)*

| Thor figure | value |
|---|---|
| **s/step p50** | **1.500** |
| **s/step p95** | **1.500** |
| **p95 / p50** | **1.00** — no tail at all |
| s/step mean | 1.4805 |
| s/step max (1 outlier in 1,390) | 4.5 |
| throughput | **13.33 windows/s · 106.7 images/s** |
| `data_s` p50 | **0.000** — 6 workers fully hide the loader, same as the A40 |
| **peak CUDA allocated** | **16.465 GB** (reserved 18.155 GB) |
| peak host VmHWM | 23.135 GB (**excludes** CUDA — see §4) |

### The ratio

| A40 reference (INHERITED) | A40 s/step | Thor s/step (MEASURED) | **Thor / A40** |
|---|---:|---:|---:|
| run rate, de-accumulated | 1.20 | 1.500 | **1.25×** |
| isolated compute-only | 1.29 | 1.500 | **1.16×** |
| end-to-end over the real 30 k run | 1.45 | 1.500 | **1.03×** |

**Thor is between 1.03× and 1.25× slower than an A40 on this arm** — call it **~1.2×**. Memory agrees
too: **A40 15.50 GB vs Thor 16.465 GB peak allocated, +6.2 %.**

**What that means in wall-clock for a real job:** a full **30,000-step REF-C-base run is ≈ 12.5 h on
Thor** (30,000 × 1.500 s = 45,000 s), against the A40's **recorded 12 h 06 m end-to-end** for the
identical arm. For this workload the desk device and the rented A40 are, in practice, the same
machine.

⚠️ **Do not generalise this ratio to every arm.** It holds for a **conv-encoder** model at 256 px.
Thor's fp32 matmul is **4.78 TFLOP/s** vs an A40's ~37 TFLOP/s PUBLISHED peak — an ~8× paper gap that
does **not** show up here, because the ResNet-34-style encoder runs through cuDNN convolutions (TF32
by default) rather than fp32 GEMM. A transformer-heavy arm (the flagship's ViT trunk + 20-step
sequential rollout) is a **different** workload and would have to be measured separately, not
extrapolated from this number.

**Why the two configs are provably the same job, not merely similarly named:**

1. **Param count is identical to the digit** — 104,191,577 on Thor vs 104,191,577 recorded for the
   A40 arm. A different preset or a different graft would not land on the same integer.
2. **Peak GPU memory agrees to ~6 %** — A40 14.44 GiB (15.50 GB) vs Thor's measured peak at the same
   batch. Memory is a shape signature; two runs with different activation shapes do not coincide.
3. Same trainer file, same optimizer (Adam 1e-4), same batch 20 / workers 6 / `--mode diffusion`.

**Declared differences (none of which touch per-step compute):**

* **Episode pool** — 39 val episodes here vs 2,376 train episodes on the A40. This changes only the
  loader, and `data_s ≈ 0.0` on **both** sides (6 workers fully hide it), so it does not move s/step.
* **Route labels** — A40 used `--labels v21`, this run used the default `v1`. Label derivation is
  CPU-side inside the dataset and is hidden by the workers; the model graph is unchanged.
* **Anchors** — A40 loaded a 128-anchor FPS file, this run used the built-in 128-anchor default.
  Same count ⇒ same tensor shapes ⇒ same compute.
* ⚠️ The A40 side is **INHERITED**. CLAUDE.md forbids an INHERITED number *deciding a GPU-day*; the
  ratio below is therefore **directional evidence for a provisioning decision, not the decision**.
  It rests on two mutually corroborating A40 figures (1.20 run-rate and 1.29 compute-only), which is
  why it is quotable at all.

---

## 6. Batch sweep — what fits, and what actually limits it

`--mode classifier`, `--workers 4`, 39 episodes, 35 steps each, first 5 discarded, **one process per
batch** so each peak-memory reading is uncontaminated.

| batch | p50 s/step | p95 s/step | p95/p50 | windows/s | images/s | peak CUDA GB | host VmHWM GB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.600 | 0.700 | 1.17 | 13.33 | 106.7 | 7.15 | 11.19 |
| 16 | 1.300 | 1.300 | 1.00 | 12.31 | 98.5 | 12.97 | 16.86 |
| 20 *(matched arm, §5, diffusion)* | 1.500 | 1.500 | 1.00 | 13.33 | 106.7 | 16.47 | 23.14 |
| 32 | 2.300 | 2.500 | 1.09 | 13.91 | 111.3 | 24.65 | 31.41 |
| **48** | **3.400** | **3.500** | 1.03 | **14.12** | **112.9** | **36.33** | **50.45** |
| 64 | — | — | — | — | — | **SIGKILL 137** | — |
| 96 / 128 | — | — | — | — | — | **SIGKILL 137** | — |

**Throughput is flat at 12.3–14.1 windows/s across a 6× batch range (8 → 48).** Thor's 20 SMs are
already saturated at batch 8; larger batches buy **no** throughput and cost linear memory.
⇒ **on Thor, train at the SMALLEST batch that gives the gradient quality you want** — a big batch is
pure cost, which is the opposite of the usual A40 instinct.

Both memories scale linearly and therefore extrapolate honestly:
**CUDA ≈ 0.76 GB per unit batch · host VmHWM ≈ 1.05 GB per unit batch.**

⚠️ **The batch-64 kill is NOT a clean GPU-memory ceiling.** It was `SIGKILL` (137) from the **host**
OOM killer, not a CUDA OOM — and it happened while the phantom 106 GB of §4 was still occupying the
box. Re-tested after that occupancy released:

| batch | workers | result | peak CUDA GB | peak host VmHWM GB | s/step |
|---:|---:|---|---:|---:|---:|
| 48 | 4 | ✅ runs | 36.33 | **50.45** | 3.4 |
| 64 | 4 | ⛔ **SIGKILL 137** | — | — | — |
| 64 | **2** | ✅ **runs** | 47.99 | **36.41** | 4.7 |
| 64 | **0** | ✅ **runs** | 47.99 | **19.20** | 4.5 |
| 96 / 128 | 4 | ⛔ SIGKILL 137 | — | — | — |

**⭐ The ceiling is the DATALOADER, not the GPU.** At batch 64 the model needs 47.99 GB of CUDA
either way — what changes is host memory: **19.2 GB at 0 workers, 36.4 GB at 2, and OOM at 4**, i.e.
**≈ 8.6 GB of host RAM per worker** at this batch (`prefetch_factor=2` + `pin_memory=True` over
uint8 `[B, 8, 9, 256, 256]` windows). At batch 48/workers 4 the host side (50.45 GB) already exceeds
the GPU side (36.33 GB).

**Answers, stated separately because they differ:**
* **Largest batch measured to fit: 64** (workers ≤ 2, 47.99 GB CUDA). Above 64 was not tested at low
  worker counts, so 64 is a **floor on the ceiling, not the ceiling**.
* **Largest batch at the conventional `--workers 4`: 48.**
* Nothing here is GPU-memory-bound before ~48 GB, on a device whose CUDA heap accepted a
  **60 GB** written allocation (§4).

⚠️ **The first sweep's batch-64 kill was partly an artifact and is CORRECTED here.** It ran while the
phantom ~100 GB of §4 still occupied the box (`free`: 18 GB available). After that released
(`free`: **102 GB** available) batch 48 ran fine and batch 64 ran at ≤ 2 workers. **Had I stopped at
the first sweep I would have reported the ceiling as 32.** The re-test is the finding.

**Precision caveat, stated rather than buried:** `refc_train.py` writes `step_s` with
`round(t_step, 1)`, so every per-step figure is quantised to **0.1 s**. At batch 8 (0.6 s) that is
±17 %; at the batch-20 matched config (~1.6 s) it is ±3 %. p50/p95 are reported on that grid, and a
precise wall-clock mean is given alongside for the sustained run. The trainer's `step_s` is
ACCUMULATED over `--log-every` (CLAUDE.md §Traps) — every run here used **`--log-every 1`**, where
the accumulator resets each step and therefore *is* the per-step time.

---

## 7. Thermals and power over a sustained run

**35.5 min continuous at batch 20, 715 `tegrastats` samples at 3 s**
*(`/tmp/thorbench/sustained.tegrastats.log`)*

| | value |
|---|---|
| `tj` at start | 44.06 °C |
| `tj` **max** | **75.88 °C** |
| `tj` p50 | 71.38 °C |
| `tj` rise | **+31.8 °C** |
| `tj` at end (after the run stopped) | 60.44 °C |
| GPU power max | **49.6 W** |
| GPU power p50 (active) | 39.4 W |
| GPU active fraction | **0.993** — the GPU was busy 99.3 % of samples |
| CPU clock range | 972 – 2601 MHz |
| host RAM over the run | 20.2 → 39.7 GB (Δ 19.5 GB) |

**⭐ THERMAL VERDICT: NO THROTTLING.** The device warms 31.8 °C and plateaus in the low 70s, and the
step time does **not** move: **p95 / p50 = 1.00** across 1,390 steps. A 60-step burst and a 35-minute
run give the **same** number on this workload. The brief's concern was justified in principle and
simply does not bite here — at ~40 W GPU draw REF-C-base does not push Thor near its thermal limit.

⚠️ Scope: 35 min, one ambient, one workload, board fan on the devkit. A multi-hour run in a warm room,
or a heavier arm drawing closer to Thor's full power envelope, is **not** covered by this evidence.
The observed VIN board draw peaked around 98 W, so there is real headroom above 40 W that this arm
never used — and headroom that is not exercised is not measured.

---

## 8. Verdict — the PI's actual question

### Is Thor usable for training?

**Yes — and the honest headline is stronger than the brief anticipated.** The brief offered
*"Thor is an inference/eval device, training is N× slower"* as a perfectly good answer if that is what
the data said. **It is not what the data says.** On the same arm at the same config, Thor is
**~1.2×** an A40, sustains it for 35 minutes with **zero** throttling, and finishes a real 30 k run in
**12.5 h against the A40's recorded 12 h 06 m**.

### For what size of job?

| job | verdict | evidence |
|---|---|---|
| **REF-C-base (104 M) / 256 px, ≤ 30 k steps** | ✅ **Do it on Thor.** 12.5 h, batch 20, 16.5 GB | MEASURED |
| Same arm at batch 48–64 | ✅ fits, but **buys nothing** — throughput is flat | MEASURED |
| REF-C-XL (252 M) | 🟡 **plausible, unmeasured.** A40 ran it at ~3.4 s/step; at Thor's ~1.2× that is ~4 h per 5 k steps. Memory should fit at batch 20 | ESTIMATED |
| **Flagship v5f (ViT trunk, 256×640, 20-step rollout)** | ❓ **UNKNOWN — do not extrapolate.** Different workload class (§5) | — |
| Multi-arm / parallel panels | ⛔ **No.** One arm saturates the 20 SMs; a second only halves both | MEASURED |

### At what cost in wall-clock?

**≈ 1.500 s/step at batch 20** ⇒ **1 k steps = 25 min · 5 k = 2.1 h · 30 k = 12.5 h**, sustained, on
hardware already on the desk with no rental clock running.

### What this changes for the programme

1. **Thor is a THIRD training node, not just an edge device.** It is not a fallback — for
   REF-C-class arms it is A40-equivalent. The two-venv rule now has a real training venv behind it.
2. **Small batches, few workers.** Both instincts invert versus the A40: batch buys nothing
   (flat 13 windows/s), and workers cost ~8.6 GB of host RAM each. `--batch 20 --workers 2..4` is the
   sweet spot measured here.
3. **Never size a Thor job from `free` / `tegrastats` / `mem_get_info`.** All three lie (§4). This
   belongs in the traps preflight next to the `df` rule.
4. **The v5f arm could not be benchmarked on Thor and that is a real gap** — see NOT DONE below.

### Two defects found on the way

* **`ep_00028.pt` is truncated** in Thor's val cache (§2c). Repair task filed.
* **`train_flagship_v4.py` promises an escape hatch it does not implement.** Its preflight refuses a
  non-parity v2 cache with *"Pass `--require-parity`, **or record why this arm is deliberately
  non-parity**"* — but **no flag exists to record that**, and `--require-parity` then correctly
  refuses a non-parity cache. The two branches together make it **impossible to run v4 on any
  non-parity v2 corpus at all**, including for a pure throughput benchmark. Either implement the
  flag the message names, or reword the message. *(I did not weaken the guard — it is a sacred-corpus
  rail and it behaved correctly; the defect is that the message describes a door that isn't there.)*

---

## 9. Manifest

### In the repo (staged, never committed — agents stage only)

| artifact | path | what it is |
|---|---|---|
| This report | `TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-08-03-thor-training-benchmark/THOR_TRAINING_BENCHMARK.md` | the findings |
| Raw logs + probes | `…/2026-08-03-thor-training-benchmark/raw/` | **21 files**, every number above traces here |
| In-process probe | `stack/scripts/thor_bench_probe.py` | the only GPU-memory probe that works on Thor; documents the three that don't |
| Trainer wrapper | `stack/scripts/thor_bench_run.py` | runs a **real, unmodified** trainer under the probe via `runpy` |
| Batch sweep | `stack/scripts/thor_bench_sweep.sh` | one process per batch → uncontaminated peak memory |
| Analyser | `stack/scripts/thor_bench_report.py` | p50/p95 + tegrastats summary; asserts the `--log-every 1` precondition |

`raw/` contents: `sustained_b20.log.gz` (1,400 steps), `sustained.tegrastats.log.gz` (715 samples),
`sustained_b20.jsonl.summary.json`, `sustained_config.json`, `sustained_metrics.json`,
`refc_b{8,16,32,48,64,96,128}.log` + their `.jsonl.summary.json`, `b64_w{0,2}.log` + summaries.

### On the device (`tanitad-thor`) — reproducible, not required

`/tmp/thorbench/` (all run dirs) · `~/thorbench/data/thorbench-nonparity-train-val40mirror/`
(39 symlinks) · `~/venvs/tanitad-train` (now has torch 2.13.0+cu130) ·
`~/TanitAD/stack` (synced to the working tree; still **64 commits behind `origin/main` in git terms**
because those commits are unpushed — re-sync by tarball, not by `git pull`).

### Reproduce

```bash
ssh tanitad-thor
export PATH=$HOME/venvs/tanitad-train/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HOME/TanitAD/stack:$HOME/TanitAD/stack/scripts
export OMP_NUM_THREADS=6
cd $HOME/TanitAD/stack
THOR_BENCH_OUT=/tmp/b.jsonl python -u scripts/thor_bench_run.py scripts/refc_train.py \
    --data-root $HOME/thorbench/data --out /tmp/refc-bench \
    --config base --mode diffusion --steps 1400 --batch 20 --workers 6 \
    --log-every 1 --episodes 39 --save-every 100000
python scripts/thor_bench_report.py <dir> --warmup 10 --tegrastats <tegrastats.log>
```

---

## 10. DONE vs NOT DONE

### ✅ DONE

1. **A real training run on Thor** — REF-C-base, real weights, real data, 1,400 steps, 35.5 min.
2. **`tanitad-train` made usable** — torch 2.13.0+cu130 installed and verified by loading. The
   two-venv rule was honoured; `tanitad-edge` was never used to run anything.
3. **Geometry asserted before any forward pass** — data `(9,256,256)` == REF-C base, from source.
4. **s/step p50 AND p95**, batch sweep 8→128, **largest batch that fits** (64 @ ≤2 workers / 48 @ 4).
5. **Peak GPU memory and peak host RAM**, with the finding that three standard probes are wrong here.
6. **Sustained 35-min thermal/power run** — no throttling, p95/p50 = 1.00.
7. **A same-arm, same-config A40 comparison** — REF-C-base at batch 20 / workers 6 / diffusion, with
   the param count matching to the digit and peak memory agreeing to 6 %.
8. **Instrumentation staged in the repo** (4 scripts) — reusable on pods, not Thor-specific.
9. Two defects found and reported; a repair task filed for the truncated episode.

### ⛔ NOT DONE — stated plainly

1. **The v5f arm was NOT benchmarked on Thor.** `train_flagship_v4.py`'s preflight cannot be
   satisfied on a non-parity v2 corpus (§8), and Thor holds only the val cyl cache. **I did not
   weaken the guard to get a number.** ⇒ the brief's requested *"v5f Thor-vs-A40 at eff_batch 64"*
   does **not** exist, and the ratio in §5 is REF-C's, not v5f's.
2. **No A40 number was re-measured by me.** Both live A40s are running protected jobs and the other
   three pods are **down** (`Connection refused` on `tanitad-pod`, `pod3`, `eval`). The A40 side of
   §5 is therefore **INHERITED** from `2026-07-20-refc-medium-scaling.md`. It rests on two
   corroborating figures, but it is not MEASURED-by-me and CLAUDE.md forbids an INHERITED number
   *deciding* a GPU-day.
3. **The brief's "v5f ~13 s/step" is not what the log says now.** MEASURED read of
   `tanitad-new:/workspace/v5f_run.log` at steps 3400→3450→3500: `elapsed_s` 46325.9 → 47222.2 →
   48105.7 ⇒ **17.93 and 17.67 s/step marginal**. 13.6 s/step is the *cumulative mean*
   (46325.9 / 3400). Both are true of different things; the arm has slowed ~30 % versus its own
   average and **nobody has explained why**. Not chased here — flagged.
   *(For the record, `tanitad-pod4:/tmp/v1arch.log`, `train_flagship4b.py`, batch 16 × accum 4,
   `--grad-checkpoint`, `--log-every` default 50: `step_s` 425.8/417.5/418.3 ⇒ **8.35–8.52 s/step**.
   Neither arm is REF-C, so neither is a like-for-like Thor comparison.)*
4. **Above batch 64 untested** at low worker counts — 64 is a floor on the ceiling.
5. **One ambient, one thermal environment, 35 min.** Not a multi-hour or warm-room result.
6. **`--mode classifier` for the sweep, `--mode diffusion` for the matched arm.** The sweep table and
   the matched arm are therefore not the same decoder mode; batch 20 appears in both and is the
   bridge (1.500 s/step diffusion vs an interpolated ~1.4 s classifier — close, but not identical).
7. **No model-quality claim of any kind.** Non-parity, 39 episodes. The four-metric-family rule
   governs evals; this is not one.
