# Master Mind decisions, 2026-09-04 — refcv4b

## D1. DO NOT RESTART `refcv4b-b1-v72-40k`. The pace regression does not exist.

**I raised this as "~21 GPU-hours, the item costing money" and I was wrong.** The claim
inherited a comparison against an outlier without checking what the pod's baseline
actually is.

MEASURED (`TanitAD Research Lab/Tools & DevEnv/Research/2026-09-04-refcv4b-pace-regression/`):

| run | marginal median s/step | n |
|---|---|---|
| `refcv3-b1-v72-30k` seg 8 — same trainer, pod, cache, batch, workers, LRU; **completed all 40,284 steps** | **3.942** | 392 |
| `refcv4-…ABORTED-step6400` | **1.964** | 115 |
| **`refcv4b` (LIVE)** | **3.844** | 14 |

The 1.964 s/step run is **the only sub-2 s run in this pod's entire recorded history** and
it lasted 3 h 42 m. Against the real baseline the live run is **−2.5 %**, and its ~44.5 h
ETA beats refcv3's realised 45.6 h. There was never 21 GPU-hours to save.

⚠️ Two of my briefing's premises were also wrong: `--v2-lru 24`, `--batch 20`,
`--workers 6`, `--prefetch-factor 1`, `--sel-accel-max 2.0`, `--goal-str`,
`--ego-state-inject`, `--ego-dropout 0.5` are **not** differences between the runs — the
exhaustive diff finds 5 differing flags of 30, only two substantive. And this trainer logs
no `step_s` at all; `elapsed_s` is cumulative, so `WHY_ABORTED.md`'s 2.086 is a cumulative
figure whose matched marginal is 1.964.

**Root-cause class:** *a true measurement quoted outside its scope* — the `df` / `free` /
`step_s` family. The 2.016 figure was real; treating it as **the baseline** was the error.
⭐ The general form worth keeping: **before calling something a regression, establish what
the population looks like.** A single prior observation is not a baseline.

The aborted run's pace remains genuinely unattributed. Ranked #1 is CUDA caching-allocator
pressure at 94.2 % A40 occupancy (a cliff, not a slope), bounded by the fact that the
500-step eval block grew only 0.1 s where a doubled forward would have cost ~5 s — so the
extra 1.88 s/step is **not** in the no-grad forward path.

## D2. Adopt the zero-cost pace instrumentation at the NEXT launch, inert by default.

Log `torch.cuda.memory_stats()["num_alloc_retries"]` and `max_memory_allocated()` — pure
observation, no behaviour change. Put `cudnn.benchmark = True` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` behind **opt-in flags defaulting to
off**, so a supervisor relaunch of the live run stays bit-identical to its own first
1,400 steps. ⛔ A supervisor relaunch imports whatever is on disk; an allocator change
that is not opt-in would silently make the second half of a run a different experiment.

**Sequencing:** implementation waits until the pod-currency stream lands, to avoid two
writers on `refc_v3_train.py`.

## D3. Fix the vocabulary's low-speed friction load in the NEXT build, not by restart.

~16.27 % of the 117-anchor fan is undrivable at the window's own v0, but the oracle-best
anchor — the supervision target — exceeds μ = 0.7 on only 0.31 % of windows. Pool waste,
not a corrupted target. Fix: clamp `kappa` against the **realised** speed along the path.
Evidence: `.../Research/2026-09-04-refcv4b-flyability/RESULT.md`, instrument
`stack/tanitad/instruments/flyability.py` (14 tests).

## D4. Retire end-bearing from turn-coverage gates.

It under-reads capability on long paths (a fictitious 8.85 % hole) and over-reads it on
short ones (12 fictitious exceedances) — both in one 117-anchor set on one corpus. Use
terminal heading, and always report **demand** beside supply: GT heading demand exceeded
the vocabulary on **0 of 4,823 windows**.

## What is still open and belongs to the PI

The **route-label doctrine**: refcv3's route metric is unfalsifiable because its label is
an exact bijection of the fed nav token (141/141). The echo test cannot see it — the defect
is in the LABEL, not the input. The question is whether the leak guard should REFUSE such a
metric rather than score it WON. That is a doctrine call, not a measurement.
