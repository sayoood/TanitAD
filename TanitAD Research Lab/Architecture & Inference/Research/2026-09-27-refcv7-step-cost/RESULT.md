# refcv7 step cost: where the +3.5 s/step goes, and which levers are lossless

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-step-cost/`
**Tree:** clean `git archive` of `agent/arch-inf-20260803` tip `6fa5e8bb` (stack + taniteval + tools)
into `C:/lgt/rc7cost`. `tanitad.__file__` was checked to resolve inside that tree on every run.
**Hardware:** the dev box (RTX 4060 8 GB, i9-12900F). **No Thor number in this file is a
measurement of mine.** Thor was off limits, so every Thor figure is either INHERITED (named) or
ESTIMATED (derivation given).

## TL;DR

1. **The 10 cm map branch accounts for ~83-90 % of refcv7's extra step cost.** That covers the
   branch's forward, its checkpoint recompute and its backward at 1000x600, the extra stride-8
   trunk pass that feeds it, and its per-class signal, net of the 0.5 m map path that refcv7
   removed. On the dev box it is **83 % of the delta at b2 and 90 % on the linear b16
   projection** (MEASURED ablation ladder, replicated). Thor gives the same answer independently:
   the map-overfit harness measured the branch at **0.584 s/step at b4 on an exclusive GPU, and
   0.711 s with the stride-8 trunk layers trainable**. Scaled linearly to b16 that is ~2.8 s of
   the 3.47 s delta (INHERITED + ESTIMATED).
2. **The box-head changes cost ~9-14 %** of the delta (300 queries, deep supervision, focal loss,
   VIS-1, learned_ref and the detection census). **The priors cost ~5-6 %.** Within the box head,
   only deep supervision is visible above noise on the dev box.
3. **The data loader is not the bottleneck** (MEASURED CPU cost per sample; the Thor margin is
   ESTIMATED at 3x or more). Its only refcv7 cost is +0.13 s per sample of zlib decompression in
   the worker, which never reaches the GPU. **The exact Hungarian solver is not a lever either:**
   it takes about 1.6 ms per step at b2, gives the same assignment as scipy on 50/50 real cost
   matrices, and scipy saved nothing in an in-process A/B.
4. **Lossless levers, ranked by saving.** None of them changes training numerics.
   * **Conflict cadence.** The probe costs **+91-101 % of a step on every step it measures**
     (MEASURED; the INHERITED A9 figure was +79.8 %). At `--conflict-every 10` that is ~9-10 %
     of the step. Moving to **50** saves an **ESTIMATED ~0.6-0.7 s/step on Thor**. The trade-off
     is fewer instrument readings, not model numerics, so the PI or the prereg has to decide.
   * **Per-class map signal and box census on logged steps only.** In-process A/B: **-0.063 s at
     b2 (90 % CI [-0.069, -0.016]), -0.0195 s at b1 ([-0.028, -0.006])**. That is bit-identical
     training and identical log rows. ESTIMATED Thor saving: 0.2-0.7 s.
   * **Map grad-ckpt off.** **-0.019 s at b1, CI [-0.030, +0.009]: not separated.** It costs
     **+1.31 GB per sample** (MEASURED), about +21 GB at b16. ESTIMATED ~0.3 s on Thor.
   * **Measured at ~0:** `cudnn.benchmark`, scipy for the matcher, more loader workers.
   * **Not measurable here:** `torch.compile` of the map branch (no Triton on Windows). This is
     the biggest *unmeasured* candidate. Its numerics are equivalent but not bit-identical.
5. **The first two levers are bit-identical and together are worth an ESTIMATED ~0.8-1.4 s/step on
   Thor (~8-14 %).** That takes the run to about **8.5-9.1 s/step**, still above the PI's 8.0 line.
   * Adding ckpt-off (~0.3 s, +21 GB) and the stride-8 reuse (~0.5 s, a code change, equivalent
     numerics) takes the ESTIMATE to ~7.7-8.8 s/step.
   * The cadence lever would speed refcv6 as well. The refcv7-specific ~2.9-3.1 s of the 10 cm
     branch stays. Only compile (equivalent numerics) or numerics-changing levers can remove it:
     bf16 in the branch, a smaller extent or resolution, or narrower channels.

   **Confirm on Thor with `code/thor_confirm.sh` (<= 20 min) before anyone proposes a restart.**

## 0. Anchor numbers

| number | value | class |
|---|---|---|
| refcv7 step, Thor, b16, 416x1024, r101, exclusive GPU | **9.876 s/step** (marginal, steps 50->100, which includes 5 conflict steps) | INHERITED (brief) |
| refcv6 step, same trainer, trunk, batch and levers | **6.41 s/step** | INHERITED (brief) |
| delta | **+3.47 s/step (+54 %)** | derived |
| 10 cm branch alone, Thor, b4, exclusive GPU (map_hires_overfit, A15 config) | **0.584 s/step** with trunk detached; **0.711 s/step** with stride-8 trunk layers trainable | INHERITED: `2026-09-26-refcv7-map-hires/raw/gmo_early/g_map_overfit_A15.EARLY_NONBINDING.json` (`s8_detached`, `healthy`); A18 MAIN 0.6946 |

## 1. Method

`code/step_profiler.py` runs the **real trainer** (`refc_v3_train.main`) on the **canonical
154-token launch argv**. It edits nothing in the tree. Its instruments are attribute wrappers and
module forward hooks installed at `build_optimizer(model, args)`.

* **Argv** (`code/rc7_argv.py`): the canonical `refcv7-r101-s0.argv.json` with paths mapped by
  the launch gate's own `parse_path_map`/`map_path`/`flag_pairs`. Each rung is a set of flag
  edits on that argv. **R6 is derived by reversing `changes_vs_refcv6`, and checked against
  refcv6's own `config.json` argv: identical** except an explicit `--agent-queries 100`, which
  was the code default then.
* **Dev-box departures, stated in every `run_spec.json`:**
  * `--batch 1|2`.
  * `--trunk-compile` dropped (no Triton). The trunk is compiled on Thor in both arms, so the
    delta does not depend on it.
  * `--workers 0`. v2 cache providers lose `newest_frame_only` when pickled into a Windows
    **spawn** worker; see section 7.
  * `--episodes 12` and the **eval139 cache as the train set**. The dev box has 0/139 map-GT
    hits for the train-a6 clips, so a train-a6 run would skip the map read. This needs
    `--allow-eval-clips-in-train`, a named override, plus `--label-clock-max-unverified 0.05`.
    **Timing rig only; nothing it produces is a model result.**
  * No in-run eval.
* **Metric:** per-step **compute = wall - data_wait**. The dev box loads in the main process, and
  the harness syncs at the step boundary, so the subtraction is exact. **Marginal = steady median
  + (conflict-step extra)/10**, the same quantity the Thor probe reads over 50 steps.
* **Noise control.**
  * **Replicates.** Two identical R7 runs differed by **0.12 s/step** at b2. The first ran in
    the tail of another session's GPU/CPU work: IQR 1.87-2.03, against <= 0.04 for every other
    run. Each non-R7 rung was replicated and reproduced within **0.001-0.009 s**. The R7
    reference therefore pools 167 steady steps across 7 runs and is **1.994 s marginal at b2**.
    The perturbed first run is reported, not hidden (`raw/gpu_devbox/final_tables.json`).
  * **In-process A/B for every lever.** The lever alternates OFF/ON in 3-step blocks inside one
    process, so run-to-run noise cancels. The estimator is the median of 9 block-paired
    differences, with a bootstrap 90 % CI over pairs (`code/ab_summarise.py`).
  * ⚠️ This CI answers "would another draw of steps inside this run say this?" It says nothing
    about a Thor run. Thor is section 6.

## 2. Attribution: the ablation ladder (MEASURED, dev box RTX 4060)

Delta = R7 minus the rung, i.e. **what switching that refcv7 addition off saves**, in marginal
s/step (conflict amortised). R7 reference = pooled steady steps:

* b2: 167 steps over 7 runs, **1.994 s**;
* b1: 98 steps over 4 runs, **1.161 s**.

The b2 rows have 2 runs each. The b1 rows have 1 run, except R6 which has 2.

| refcv7 addition switched off (rung) | delta b1 | delta b2 | per-sample slope | linear b16 (dev box) | share of R7-R6 at b2 | at b16 |
|---|---|---|---|---|---|---|
| **everything, i.e. refcv6 (R6)** | 0.192 | **0.366** | 0.174 | 2.80 | 100 % | 100 % |
| **10 cm map back to the 0.5 m map** (`map_v6`: map-hires, near lift and refine, bev-source, crop; `--w-map 1` on `sam3_corpus`) | 0.146 | **0.305** | 0.159 | 2.54 | **83 %** | **90 %** |
| box head back to refcv6 (`box_v6`: 100 queries, no deep supervision, BCE, no VIS-1, learned queries) | 0.036 | **0.050** | 0.015 | 0.26 | 14 % | 9 % |
| priors off (residual prior, tac8, nav compliance, speed ceiling) | 0.007 | 0.017 | 0.010 | 0.16 | 5 % | 6 % |

Sub-splits, 1 run each at b2. Single-run noise on R7-like configs is ~+/-0.04, so these rank only:
**near lift and refine 0.035** (part of `map_v6`) · **no deep supervision 0.034** · VIS-1 off
0.000 · learned queries -0.001 · BCE -0.014 · **100 queries -0.050** (300 queries is not visible
on this rig).

**Inside the step** (`R7_timers_b2` vs `R6_timers_b2`; synchronised timers add ~4 %):

| b2, s/step | R7 | R6 | delta |
|---|---|---|---|
| backward (incl. checkpoint recompute) | 1.087 | 0.898 | **+0.189** |
| forward + loss | 0.723 | 0.584 | **+0.139** |
| of which 10 cm branch forward (encoder 0.032, decoder 0.024, lift 0.005, near 0.004) | 0.067 | 0 | +0.067 |
| of which stride-8 tap pass (eager, stem..layer2, newest frame) | 0.012 | 0 | +0.012 |
| of which 10 cm per-class signal (every step) | 0.023 | 0 | +0.023 |
| of which refined slot losses (both heads x 3 layers; matching 0.009, of which the Hungarian solver is **0.0016**) | 0.043 | ~0 | +0.043 |
| of which detection census (every step) | 0.007 | 0 | +0.007 |
| of which refcv6's 0.5 m lift and map branch (removed) | 0 | 0.010 | -0.010 |
| trunk main pass | 0.369 | 0.376 | ~0 |
| optimiser + clip | 0.041 | 0.041 | 0 |
| **conflict probe, per measured step** (two `autograd.grad` over the trunk) | **1.86-1.92** | **1.63-1.65** | +0.24 |

**Thor mapping (ESTIMATED).** Apply the dev-box shares (83-90 % map, 9-14 % box, 5-6 % priors)
to Thor's +3.47 s:

* **10 cm map branch ~2.9-3.1 s**
* **box head ~0.3-0.5 s**
* **priors ~0.2 s**

The independent Thor cross-check (INHERITED map-overfit, scaled linearly from b4) gives
**~2.3 s** for the branch plus **~0.5 s** for the trainable stride-8 tap: ~2.85 s before
subtracting the removed 0.5 m path. That agrees with the dev-box share. ⚠️ Linear b-scaling and
shares transferred across GPUs are assumptions. The 4060 and Thor have the **same memory
bandwidth** (272 vs 273 GB/s, PUBLISHED specs), which matters because the branch is dominated by
GroupNorm, GELU, interpolate and grid-sample work over 1000x600x32 and 400x240x64 activations.
Thor's compiled trunk makes the map's *relative* share larger there than on the dev box, which is
consistent with Thor's +54 % against the dev box's +22 % (b2) / +25 % (b16 linear).

The shares overlap by a few per cent: single removals sum to 1.02x (b2) / 1.06x (b16) of the
full R6 delta.

### 2b. Why the 10 cm branch costs what it costs: device-independent op counts

`code/opcount.py` is a TorchDispatchMode over the real step, run on CPU at b1: steady step 2,
`--trunk-chunk-ckpt 1`, no bf16. A count is a property of the program, not of the device.
Backward ops are attributed through the autograd node's creating scope; recompute is detected as
grad-enabled-inside-backward. Output: `raw/opcount_devbox/*`, `code/summarise_opcount.py`.

| per sample (b1) | 10 cm branch (R7) | R6 perception group (0.5 m lift + BEV encoder + map head + box decoder) | R7 perception group (box decoder + BEV pool) | trunk groups, R7 minus R6 (= the stride-8 tap pass) |
|---|---|---|---|---|
| kernels: fwd / bwd / recompute | 120 / 72 / 103 | 128 / 240 / 0 | 162 / 403 / 0 | +194 / +294 / +198 |
| FLOPs | **429 GF** | 74 GF | 14 GF | **+121 GF** |
| bytes touched | **14.5 GB** | 2.9 GB | 0.8 GB | +4.2 GB |

**The branch is bandwidth-shaped, not launch-shaped.**

* At b16 it touches **~233 GB per step**. On a 273 GB/s memory system that is a >= 0.85 s floor
  before a single FLOP counts. Thor's inherited ~2.3 s means ~100 GB/s achieved.
* That is exactly the profile `torch.compile` fusion (GroupNorm + GELU + elementwise) and bf16
  attack. It is why compile is the largest open lossless-equivalent lever, and bf16 the largest
  lossy one.

**The box head is sync-shaped, not FLOP-shaped.** refcv7's losses, matching, census and per-class
signal perform **~195 host syncs per sample** (`.item()`, `int(t)`, `.cpu()`), against ~63 for
refcv6's path. That is ~2,000 more per b16 step, each of which drains the GPU queue.

⚠️ On a CPU run a sync on a CPU-resident tensor is counted too. The ~790 "outside scopes" syncs
are AdamW's per-parameter step `.item()` on CPU step tensors and are **not** device syncs on the
GPU. Treat sync counts as upper bounds.

## 3. Ranked lossless levers (training numerics unchanged)

| # | lever | MEASURED on the dev box | numerics | memory | cost to apply | ESTIMATED on Thor (b16) |
|---|---|---|---|---|---|---|
| 1 | **`--conflict-every 10 -> 50`** (or 100; or off) | probe extra per measured step = **+100 % / +95 % of a steady step** (b2, two runs), **+91 %** (b1); amortised 0.185 s/step at 10 (b2). `conflict_off` vs the pooled R7 (cross-run): -0.187 s/step | **bit-identical training** (the detector is observational; `tests/test_refcv6_grad_conflict.py`) -- fewer instrument readings | -1.1 GB peak with it off (b2) | argv change -> new gate token + restart | **~0.6-0.7 s/step** at 50 (0.7-0.8 at 100, 0.8-0.9 off), from r = 0.8-1.0 x a ~9.0 s steady step. **Instrument-density decision: PI / prereg** |
| 2 | **per-class 10 cm signal + box census on logged steps only** | in-process A/B: **-0.063 s @ b2 [-0.069, -0.016]**, **-0.0195 s @ b1 [-0.028, -0.006]**; per-class alone -0.023 @ b2 [-0.057, -0.003]; census alone -0.021 @ b2 [-0.044, +0.005] | **bit-identical training** (`per_class_signal` is `no_grad`, uses no RNG, and only log rows read it); log rows **identical** at logged steps | transient -0.3..0.6 GB (the [B,8,1000,600] one-hot temporaries) | code, 8 edits in `refc_v3_train.py` (`:4240` / `:4296` / `:4350` / `:5096` / `:5519` / `:5546` / `:9413`). **Proposal, not applied:** `code/apply_lever2.py` is byte-exact and CRLF-preserving; it REFUSES any base but the tip blob `203b437f` and asserts the result blob `fd874293`. **Smoke-tested on the dev-box GPU** (`raw/patch_smoke/`): the patched trainer runs, and its logged rows at steps 5/10/12 carry the **identical 598-key sets** (442 `map_hires_*`) as the unpatched control. D3 passes. | **0.2-0.7 s/step** (per-sample saving 0.015 s from timers to 0.044 s from the A/B slope, x16) |
| 3 | **`--map-hires-grad-ckpt off`** | in-process A/B at b1: **-0.019 s [-0.030, +0.009] -- not separated**; b2 **cannot be measured on 8 GB** (7.78 GB peak -> WDDM spill, 6.96 s/step, INVALID) | equivalent by construction (the recompute is the same deterministic forward) | **+1.31 GB per sample MEASURED** (4.51 -> 5.82 GB at b1) => **~+21 GB at b16** (analytic 21.3 GB, map-hires package) | argv + SPEC A7 item 4 amendment | **~0.3 s** (12 % of the branch x 0.146 s/sample x 16). Admissible only if the probe's `cuda_max_mem_gb` + 21 GB stays well under 128 GB unified -- **read it before deciding** |
| 4 | reuse the MAIN trunk pass's stride-8 map instead of the separate eager `s8_from_normalised` pass | tap forward 0.012 s @ b2 (+ its backward) | **equivalent, not bit-identical** (compiled main pass vs eager tap) | small | code change in `timm_trunk` / `refc.py` | up to **~0.5 s** (INHERITED map-overfit: trainable s8 adds 0.127 s @ b4) |
| 5 | `torch.compile` of the map branch (encoder / decoder / near) | **NOT MEASURABLE here: no Triton on Windows.** The harness lever `--ab compile_map` exists and its plumbing is validated with `--compile-backend eager` | equivalent, not bit-identical (fused kernels) | likely lower | code (or harness lever on Thor; `--compile-backend eager` ran 24 steps on the dev box without error) | **unknown -- the largest candidate**: the branch is GN/GELU/elementwise-bound, which Inductor fuses |
| 6 | `--cudnn-benchmark` | A/B **+0.004 s [-0.071, +0.011]: nothing** | equivalent (algorithm choice) | - | argv | UNVERIFIED on Thor (dilated convs) -- cheap to A/B there |
| 7 | scipy `linear_sum_assignment` for `hungarian` | A/B **+0.004 s [-0.025, +0.019]: nothing**; solver = 1.6 ms/step @ b2; synthetic 300x16: 0.24 ms vs 0.015 ms | **same assignment 50/50 on REAL cost matrices** (`raw/hungarian/hungarian_real_bank.json`; median 3 targets, max 11) and 280/280 synthetic | - | dependency (**Thor venv has no scipy**, INHERITED) | < 0.06 s. **Not worth it** |
| 8 | more / fewer loader workers | loader supplies a batch every ~1.0 s with 6 workers on this CPU (section 4) | - | - | argv | ~0 while the Thor worker is < ~9x slower than an i9 P-core |

**NOT lossless (listed because they would move numerics):**

| lever | cost |
|---|---|
| bf16 autocast in the 10 cm branch | changes numerics |
| fewer queries (300 -> 100) | measured -0.050 s @ b2, i.e. no saving on this rig, and changes the head |
| drop deep supervision | 0.034 s @ b2 |
| smaller extent or coarser resolution | changes the product |
| `--conflict-mode subtract` | one backward instead of two, but it is a *different statistic*, not interchangeable in a B2 verdict |

## 4. Data loader (MEASURED, CPU)

`--mode loader` captures the real `V3Dataset` at DataLoader construction and times `ds[i]` with
every item helper wrapped (`raw/loader/`).

* **Cold regime:** includes this box's slow exFAT D: reads, 4.93 / 4.26 s per sample. cProfile
  puts 9.7 of 12.5 s in `BufferedReader.read`. **Not a Thor number.**
* **Warm regime** (`*_warm8`): OS file cache warm, every in-process LRU cleared before each
  sample, which is the training regime.

| per sample, n = 32 | refcv7 | refcv6 | delta |
|---|---|---|---|
| total | **0.373 s** | **0.251 s** | +0.122 s |
| frames: v2ep load 0.019 + 32 PNG decodes ~0.17 | 0.217 | 0.228 | ~0 |
| map target | **0.154** (decompress 120.6 MB `fine_codes` to use one 0.6 MB frame) | 0.024 (0.5 m `cart_frac`) | **+0.130** |
| collate 16 | 0.214 | 0.207 | ~0 |

**Do 6 workers keep up?**

* One worker builds a batch of 16 in ~6.2 s (refcv6: 4.2 s), so six workers supply one batch
  per ~1.0 s against a 9.9 s step.
* Thor's Neoverse-V3AE per-core speed relative to an i9 P-core is **UNVERIFIED**. At an assumed
  1.5-3x slower, supply is ~1.5-3 s/batch.
* NVMe reads (~99 MB/sample, ~1.6 GB/step) are not in the warm number.
* ⇒ **Not the bottleneck.** The only refcv7 loader cost is the 120 MB zlib inflate per sample.
  A per-frame-chunked map store would remove it, but that only matters if the run ever becomes
  loader-bound.

## 5. Where the GPU idles (the "97 %, 97 %, 16 %" sample)

A dev-box torch.profiler trace was taken: `R7_torchprof_b2`, 6 steps, exported trace analysed by
`code/trace_analyse.py` into `raw/gpu_devbox/R7_torchprof_b2/trace_idle.json`.

* **Per step:** kernels busy **1.49 s** of a **~2.0 s** compute step (~75 %). Idle time
  excluding the data wait:
  * **~0.43 s in sub-50 µs gaps**, a launch-bound regime of many tiny kernels;
  * 0.06 s in the backward scope;
  * 0.02 s in the planner decoder;
  * ~0.015 s in the Python-looped slot losses, matcher and census.
* **Kernel time by launching thread:** 0.52 s main, 1.15 s autograd.
* Nothing on the dev box produces a sustained 16 % phase inside a training step apart from the
  data wait. That wait is a dev-box artefact (`--workers 0`).
* On Thor a single 16 % sample is consistent with any of:
  * a conflict step's CPU-side setup;
  * the every-500 checkpoint save or eval;
  * a sync-bound Python-loop phase, which is 8x larger at b16.

  **UNVERIFIED.** The Thor confirmation run records `data_wait` per step, which settles the
  loader question directly.

## 6. Thor confirmation in <= 20 min, at the next restart window

`code/thor_confirm.sh <launch-tree> <harness-dir> <out-dir> [compile] [ckpt]`. It refuses unless
the GPU is empty, and writes only into `<out-dir>`. Every job runs at b16, `--keep-compile`, with 6
workers (Linux fork, so the spawn defect in section 7 does not apply), on the canonical argv:

1. **`--ab percls_census --ab-block 2`, 30 steps, warm 8.** Gives the lever at b16 on Thor, the
   conflict-step extra (steps 10 and 20), and `data_wait` per step (the loader question). ~8 min.
2. **R7 `timers` mode, 22 steps.** Gives the Thor attribution: `mod:_map_hires*` forward and recompute,
   `trunk_s8_tap`, `refined_slot_losses`, `map_per_class_signal`, backward, optimiser. ~7 min.
3. **Loader bench on CPU** (`--loader-clips 8`, 32 samples, warm cache). ~3 min.

If the window allows, pass `compile` for `--ab compile_map --compile-backend inductor` and/or
`ckpt` for `--ab ckpt_off`. For `ckpt`, check the probe's `cuda_max_mem_gb` + ~21 GB first.
Ship `code/` md5-verified, per the programme's file-ship rule; never git on a pod.

**Pass criteria (pre-stated):**

* A lever is worth a restart only if its paired A/B saving at b16 is separated (CI excludes 0)
  **and** larger than Thor's inference-free run-to-run floor.
* Map branch plus s8 below 70 % of the timers' R7-minus-R6-like terms would **refute** the
  attribution above.
* A median `data_wait` above 0.2 s/step would refute section 4.

## 7. UNVERIFIED items, side findings, incidents

* **UNVERIFIED:**
  * every Thor figure except the INHERITED anchors;
  * the Thor/i9 CPU ratio;
  * linear b16 scaling;
  * cudnn and compile on Thor;
  * Hungarian tie behaviour beyond the 50 captured real matrices;
  * the op-count b2 pair and hence device-independent slopes: only b1 is summarised in 2b.
* **Side finding (a real defect, dev-box scope):** `tanitad/data/v2_dataset.py`
  `V2CompressedCache.__getstate__` omits `newest_frame_only`. Any **spawn**-started worker
  (Windows, macOS) dies with `AttributeError`. Linux fork on Thor is unaffected. A one-line fix
  plus a pickle round-trip test; **not applied here** (no repo code changes).
* **Side finding:** the per-class 10 cm signal and the box census are computed **every step**,
  while only 1 step in 50 logs them. That is lever 2.
* **Incident (disclosed):** at ~21:02 a CPU smoke launched with `CUDA_VISIBLE_DEVICES=""`. Windows
  drops empty environment variables, so the child saw the GPU and hit `torch.cuda` init for ~5 s
  while another session's job was on it. No kernels ran; it crashed at `get_device_name`. The
  harness now uses `-1` and `--require-cpu`, which asserts `device_count() == 0` before anything
  else.
* **Resource timeline:** the GPU was held by the other session's chain past 23:00. That chain was
  `m5_gpu_chain.py` with a `--deadline 04:27`, and PID 38376 stayed on the GPU until ~01:55.
  GPU work ran **01:57-04:41** Berlin. Host RAM repeatedly fell below the 6 GB floor, and the
  runner's watchdog killed 13 attempts of my own CPU jobs to keep it (`raw/*/ladder_log.jsonl`).

## 8. Deliverable manifest

### In this package (staged by the Master Mind; nothing committed or pushed by me)

**Harness (`code/`):**

| file | role |
|---|---|
| `step_profiler.py` | modes `plain` / `timers` / `torchprof` / `loader` / `opcount`; `--ab` in-process A/B; `--lever` |
| `rc7_argv.py` | canonical argv -> rung argv; R6 reversal check |
| `run_ladder.py` | gated sequential runner: GPU-exclusive, RAM floor watchdog, retry |
| `opcount.py` | device-independent counter |
| `summarise_ladder.py`, `final_tables.py`, `ab_summarise.py`, `summarise_opcount.py`, `trace_analyse.py` | analysis |
| `hungarian_synth.py`, `hungarian_bench.py`, `map_gt_read_profile.py` | micro-benches |
| `thor_confirm.sh` | the <= 20 min Thor script |
| `pathmap_devbox_eval139_as_train.txt`, `plan_*.json`, `drive_all.sh` | exact job definitions |
| `apply_lever2.py` + `lever2_pairs.json` + `lever2_logonly.READ_ONLY.patch` | lever 2 proposal |

**Evidence (`raw/`):**

| path | contents |
|---|---|
| `gpu_devbox/<job>/` | `profile_*.json`, `steps.jsonl`, `run_spec.json`, `log.txt`, `ab_summary.json` |
| `gpu_devbox/final_tables.json`, `gpu_devbox/summary.json` | all numbers in sections 2-3 |
| `loader/` | section 4 |
| `hungarian/` | synthetic + 50 real cost matrices |
| `opcount_devbox/` | section 2b; `R7_opcount_b1_steps2` is the 2-step first pass |
| `patch_smoke/` | the lever-2 smoke |

### Only on the dev box (NOT in the repo, and not needed for the verdict)

| path | contents |
|---|---|
| `C:/lgt/rc7cost` | clean tree `6fa5e8bb` |
| `C:/lgt/rc7cost_l2` | same tree + lever 2 applied |
| `C:/lgt/rc7cost_runs/*/<job>/run/` | per-job trainer run dirs incl. `ckpt.pt` (timing-rig checkpoints, worthless as models) |
| `C:/lgt/rc7cost_runs/gpu/R7_torchprof_b2/trace.json` | 552 MB chrome trace; its analysis is banked |

**No repo code was changed.** The lever-2 patch and the `__getstate__` fix (section 7) are
proposals for the Master Mind.
