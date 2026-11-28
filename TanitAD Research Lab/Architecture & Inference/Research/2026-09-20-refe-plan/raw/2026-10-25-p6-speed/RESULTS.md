# P6 step-time study: results (banked incrementally)

**Target (PI):** cut P6 from ~58.9 s/step to ≤ 29.5 s/step on the same A40, with the recipe unchanged.
**Code:** f2f604f `refe/` (a pod copy is at `/root/tad/speed/refe`). **Torch for every number below:** system `python3` 3.12.3 + **torch 2.8.0+cu128**, cuDNN 91002, numpy 2.5.2. `venv-teacher` has not arrived yet, so nothing has been confirmed under it.
**Harness:** `bench_p6.py` builds the real REFe through `refe/model.py` + `load_dinov3` exactly as `train.py:run()` does for P6's argv, and drives train.py's online-path micro-batch sequence. That sequence is D4 deferred backward with the same host syncs, sha checks and diagnostics. Labels come from a fixed table, so the CPU labeller is costed separately.

---

## ⭐ FINAL SUMMARY (2026-10-25). Every GPU number: system python3, **torch 2.8.0+cu128**, A40 power-capped at 300 W

**Verdict: the 50 % target is NOT reached, and with the recipe unchanged it cannot be reached on this A40.**
* **Physics.** The trunk's GEMMs (cuBLAS 105–114 TFLOPS) plus FlashAttention-2 need **0.617 s per micro-batch** at the speeds this GPU actually reaches. Today's micro-batch is **0.895 s** (bench) / **0.903 s** (banked run). So the most any recipe-preserving lever could buy is ≤ 31 %, and only if every memory-bound kernel and the heads cost nothing.
* **Measured.** 12 recipe-preserving levers, each within the 0.47 % cross-process noise of the base (table below). **Best measured combined recipe-preserving speedup: 0 % on the GPU.**
* **Largest open recipe-preserving item:** the banked run's labeller stall, **1.77 % of the step** (median `stall_frac`, n = 2,608 windows). It is an argv-only fix (`--label-workers`), ESTIMATED ≤ 1.8 %, not yet measured as a lever.
* **What reaches 50 %:** only a PI decision.
  * **(a)** A recipe change. Measured: **384×704 input −47.5 %**, **LoRA only in the top 6 blocks −41.5 %**, 448×832 −27.3 %, LoRA top-12 −27.7 %, bf16 residual −4.0 %. Combinations are in queue 5.
  * **(b)** A faster GPU (ESTIMATED A100 ≈ 2.1–2.5×).
  * Either way, a GPU cut above ~30 % is then **CPU-bound** on the 7.65-CPU quota unless the labeller gets ~40–50 % cheaper per set. Its profiler is ready; it is BLOCKED on `metric_cache` + `venv-teacher` arriving on the pod.

### Ranked lever table

| rank | lever | class | s / micro | s / step | Δ step | equivalence evidence | lands how |
|---|---|---|---|---|---|---|---|
| — | **base** (P6 argv) | — | **0.8950** (0.8950 / 0.8963 / 0.8992 in 3 processes) | **57.31** (banked run: 57.80 median) | — | floor: grad rel-L2 2.3e-5, loss Δ 0 | — |
| 1 | `--label-workers 8` (one job round per micro-batch) | preserving | n/a (not in the bench) | ≈ −1.0 s (ESTIMATED from the banked stall) | ≤ −1.8 % | labels are per proposal and independent of the worker count; G-LIVE-LAB re-proves it | **argv only**, not in the identity ⇒ rides the resume; needs a gate re-mint |
| 2 | `--img-u8` (uint8 frames, `/255` on the device) | preserving | 0.8999 | 57.63 | 0 (GPU) | **EQUIVALENT** at the floor; the conversion is **bit-identical** over all 256 values | `img_u8.patch` (train.py + G-DVB row + test pin); CPU-only value, see `HANDOFF.md` step 2 |
| 3 | `--order early` (no host bubble) | preserving | 0.9034 | 57.85 | 0 | **EQUIVALENT** at the floor | not worth a gate cycle (bubble = 1.8 ms) |
| 4 | compiled heads / max-autotune / coordinate-descent / fused QKV / cuDNN SDPA / n2-once / expandable segments | preserving | 0.893–0.904 | 57.2–57.9 | 0 (noise) | heads/max-autotune: forward numerics change, grad rel-L2 4e-2 | **not recommended** |
| — | `--res 384x704` | **RECIPE** | 0.4698 | 30.10 | **−47.5 %** | different inputs | PI decision + declared change |
| — | `--lora-from 18` | **RECIPE** | 0.5236 | 33.54 | **−41.5 %** | blocks 0–17 LoRA frozen | PI decision + declared change |
| — | `--lora-from 12` | **RECIPE** | 0.6469 | 41.43 | **−27.7 %** | blocks 0–11 LoRA frozen | PI decision + declared change |
| — | `--res 448x832` | **RECIPE** | 0.6511 | 41.70 | **−27.3 %** | different inputs | PI decision + declared change |
| — | `--bf16-resid` | **RECIPE** (precision) | 0.8591 | 55.02 | −4.0 % | queue 4 measures it against the ULP floor | PI decision |

---

## INTERIM (banked about 60 min in, 2026-10-25)

### Baseline: today's P6 config (compile backbone, bf16, tf32, batch 4, deferred backward)

| source | s / micro-batch | s / step (64 micro + opt) |
|---|---|---|
| **banked P6 run** `rb2603/.../metrics.jsonl`, steps 1–2602, median Δelapsed | 57.80/64 = **0.903** | **57.80** (p10 57.48, p90 58.37) |
| **bench, rep 1–3 (same process)** | **0.8990 / 0.8950 / 0.8948** → median **0.8950** (spread 0.47 %) | 57.31 (incl. 0.033 s opt step) |

The bench reproduces the real step to within 1 % (0.895 vs 0.903 s). The 0.9 % that remains is the real run's data-loader, pin-memory and labeller CPU contention. Peak memory: bench 40.97 GB, run 41.26 GB.

### Where a micro-batch goes (CUDA events, median, bench base)

| phase | s | share |
|---|---|---|
| H2D (float32 images, 94 MB) | 0.0047 | 0.5 % |
| **trunk forward** (ViT-L, 16 images × 1,925 tokens) | **0.3536** | 39.5 % |
| heads forward + losses | 0.0117 | 1.3 % |
| GPU idle, F(k) end → B(k−1) start (host bubble) | 0.0018 | 0.2 % |
| heads backward (eager) | 0.0366 | 4.1 % |
| **trunk backward** | **0.4851** | 54.2 % |

**The trunk is 93.7 % of the micro-batch.** Host syncs, H2D and the D4 bookkeeping add up to < 1 %.

### What the A40 can do on these exact shapes (`ceiling_p6.py`, torch 2.8.0+cu128)

| kernel class | measured on the A40 | × 24 layers, fwd + bwd |
|---|---|---|
| trunk GEMMs (30,800 rows; qkv, proj, fc1, fc2, and the dgrad GEMMs) | **105–114 TFLOPS** (cuBLAS, bf16) | **0.339 s** |
| SDPA flash (FA2), [16, 16, 1925, 64] | fwd 2.78 ms (87 TFLOPS), fwd+bwd 11.59 ms | **0.278 s** |
| SDPA cuDNN | fwd 2.39 ms (faster), fwd+bwd 12.56 ms (bwd slower) | 0.302 s |
| SDPA mem-efficient | fwd+bwd 25.0 ms | 0.600 s |

⇒ **GEMM + attention floor for the trunk: 0.617 s.** The measured trunk is 0.839 s. The ~0.22 s gap is memory-bound work: LayerNorm, rope, GELU, LoRA, residual adds and LayerScale. The residual stream is **fp32**, because `torch.cat` promotes the bf16 patch tokens to the fp32 cls/register parameters.

### Kernel-class profile of the base (torch.profiler, 2 micro-batches, self device time per micro-batch; `logs/prof_base.txt`)

| class | ms / micro | share |
|---|---|---|
| GEMM kernels (cuBLAS/CUTLASS, trunk + heads) | **369** | 42 % |
| attention kernels (FA2: bwd 158, fwd 66, + heads' MHA) | **247** | 28 % |
| inductor-fused elementwise (LayerNorm, GELU, rope, residual, LayerScale, LoRA adds, casts) | **223** | 25 % |
| copies / casts / other / norm | 39 | 5 % |

GEMM and attention run at the speed `ceiling_p6.py` measures in isolation. **The only slack is the 223 ms of memory-bound fused kernels**, and inductor has already fused them. GEMM-epilogue fusion (max-autotune) recovered only 7.6 ms of it: on the A40, inductor's Triton GEMM templates hit the 101 KB shared-memory limit and fall back to cuBLAS (`maxat.log`: *"out of resource: triton_mm Required: 131072 Hardware limit: 101376"*).

### ⛔ Interim verdict on the 50 % target

**A recipe-preserving 50 % cut is below the A40's physical floor for this model.** 29.5 s/step means 0.46 s per micro-batch. The trunk's GEMMs plus FlashAttention alone, at the speeds cuBLAS and FA2 actually achieve on this GPU, need **0.617 s**. Even with all memory-bound work fused away, the heads at zero, and no host overhead, the best recipe-preserving step is about **0.62–0.65 s/micro, i.e. a 28–31 % cut at the very most**. Realistically the cut is smaller (see the levers below). Reaching 50 % **requires a recipe change**, which is a PI decision. Those levers are being measured now (list below).

### Levers measured so far (bench, torch 2.8.0+cu128, median of 3 reps × 12 micro-batches; each lever alone)

**Noise floor.** The same base config in three processes gave **0.8950 / 0.8963 / 0.8992 s/micro (spread 0.47 %)**. Within a run, rep-to-rep spread is ≤ 0.6 %. The GPU runs **power-capped**: SM clock 1,560 of 1,740 MHz at 291 W of 300 W, throttle reason `SW_POWER_CAP`.

| lever | s / micro | Δ vs base 0.8950 | phase that moved | verdict |
|---|---|---|---|---|
| base (repeat, 2nd process) | 0.8963 | +0.1 % | — | noise floor |
| `--n2-once` (RegisterCompress LayerNorm computed once) | 0.8993 | +0.5 % | b_heads 36.6→34.8 ms | **no gain** |
| `--img u8` (uint8 H2D, exact `/255` on the GPU) | 0.8999 | +0.5 % | h2d 4.7→1.8 ms | **no GPU gain**; worth it only for CPU (see below) |
| `--order early` (B(k−1) launched before the props sync) | 0.9034 | +0.9 % | bubble already 1.8 ms | **no gain** |
| `--compile parts` (heads compiled too) | 0.8946 | −0.0 % | b_heads 36.6→31.2 ms, f_heads 11.7→13.4 ms | **no gain** |

| `--fused-qkv` (one GEMM for q/k/v base) | 0.8979 | +0.3 % | — | **no gain** (isolated GEMM is slower, 1.82 vs 1.74 ms) |
| `--sdpa cudnn` | 0.9004 | +0.6 % | — | **no gain** |
| `--compile-mode max-autotune-no-cudagraphs` | 0.8902, **repeat 0.8968** | −0.5 % / +0.2 % | f_trunk 353.6→346.0 ms (run 1) | **no gain**: the repeat erased it. Also **not equivalent** at the floor (below) |
| `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` + max-autotune | 0.8932 | −0.2 % | — | **no gain** |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 0.9035 | +0.9 % | — | **no gain** |
| `--batch 2` (per-sample efficiency trend) | 0.4770 (= 0.2385 / sample) | **+6.6 % per sample** vs batch 4 | — | larger micro-batches help, but **batch 8 does not fit**: two live graphs (D4) × 8 = ~82 GB > 46 GB. Activation checkpointing would cost ≈ +35 % compute |
| `--compile none` | OOM | — | — | `--compile` is load-bearing for **memory**: eager does not fit two live batch-4 graphs |

**Each lever is within ±1 % of the base**, as the floor analysis predicts. All queue-1 and queue-3 levers are in the table above. Queue 2 produced the equivalence dumps and the recipe-changing timings (sections below).

**For the MM's launch timing:** nothing measured so far is worth holding P6. No recipe-preserving lever has shown a gain outside noise, and the floor analysis bounds what any of them can buy at ≤ 31 %, realistically a few %. ⇒ **P6 can be relaunched now on the unchanged recipe** unless the PI wants to decide the recipe-changing list first (below, being measured).

### Which kinds of lever can ride on P6's resume (from source, f2f604f)

* **Identity** (`train.py:995 IDENTITY_ARGS`) covers `batch, accum, amp, tf32, fused_adam, compile` plus the bank, calib and trunk sha256s. It does **not** cover `workers, label_workers, label_chunks, label_timeout_s, label_python, metric_cache`.
  ⇒ `--batch/--accum/--fused-adam/--compile` changes are refused on resume unless named in `--declare-change`. The labeller and loader knobs ride on a resume freely.
* **Launch gate**: the PASS token binds commit + argv sha256 + data manifests. ⇒ **any** code change or argv change needs a fresh gate pass. A new trainer flag needs a `DVB_KINDS` entry plus an observed `dvb_rows` row (`gate_checks.py:394`, enforced at `:588-603`; `test_launch_gate` pins the table).

---

## RECIPE-CHANGING levers: PI decision, never a free win (bench timings, torch 2.8.0+cu128, median of 3 reps)

Each one changes what is trained. Speedups are against the base 0.8950 s/micro. A step = 64 micro-batches + 0.033 s optimiser.

| lever | what it changes in the recipe | s / micro | s / step | cut | peak mem |
|---|---|---|---|---|---|
| `--bf16-resid` | trunk residual stream and LayerScale in bf16 (today fp32): a **precision** departure beyond `--amp bf16` | 0.8591 | 55.0 | **−4.0 %** | 35.3 GB |
| `--res 448x832` | input resolution 512×960 → 448×832 (1,456 tokens/cam, −24 %) | 0.6511 | 41.7 | **−27.3 %** | 31.9 GB |
| `--lora-from 12` | LoRA in blocks 0–11 **frozen**; no backward below block 12 | 0.6469 | 41.4 | **−27.7 %** | 21.9 GB |
| `--lora-from 18` | LoRA trained in the top 6 blocks only | 0.5236 | 33.5 | **−41.5 %** | 12.4 GB |
| `--res 384x704` | input resolution → 384×704 (1,056 tokens/cam, −45 %) | 0.4698 | 30.1 | **−47.5 %** | 24.0 GB |

Combinations are queued (queue 5); rows are added below when they land.

⚠️ **Mid-run faithfulness.** P6 resumes at step 2,603 with trained LoRA in **all 24 blocks**, at 512×960, with an fp32 residual. Every row above changes the run **at the resume**, so it becomes a **declared mid-run recipe change**. It is not a continuation of the pre-registered paper-faithful run, and the identity check refuses it unless it is declared.

### ⛔ The CPU wall: any cut above ~30 % is labeller-bound

From the banked run (`metrics.jsonl` labeller windows, n = 2,608): **`cpu_s_per_set` median 0.856**, 4 sets per micro-batch, `stall_frac` median 1.77 % (p90 2.32 %). The pod's CPU quota is **7.65 CPUs** (`cpu.cfs_quota_us` 765000 / 100000). The CPUs needed for labelling alone are 4 × 0.856 / (s per micro):

| s / micro | labelling CPUs | left for the main process + 2 loader workers (needs ~1.5–2) |
|---|---|---|
| 0.903 (today) | 3.79 | 3.86 ✅ |
| 0.647 (`lora-from 12`) | 5.29 | 2.36 ⚠️ marginal |
| 0.524 (`lora-from 18`) | 6.53 | 1.12 ⛔ |
| 0.470 (`res 384x704`) | 7.29 | 0.36 ⛔ |

⇒ **The labeller has to get ~40–50 % cheaper per set before any GPU cut above ~30 % turns into wall-clock time.** The labeller profile (`prof_labeller.py`) is ready. It is **BLOCKED**: `/workspace/data/refe_navtrain/metric_cache` and `/workspace/venv-teacher` have not arrived on the pod yet.

### The only clean route to 50 % with the recipe unchanged: faster hardware (PI spend decision)

The step is 70 % GEMM + attention at cuBLAS/FA2 speed, and 25 % memory-bound. ESTIMATED from spec ratios (not measured):
* **A100-80GB**: bf16 312 vs 150 TFLOPS, HBM 2.0 vs 0.70 TB/s → ≈ 2.1–2.5× on this step.
* **H100**: ≈ 4–5×.

The labeller must still keep up (same CPU arithmetic, with that pod's own quota).

---

## EQUIVALENCE (one real optimiser step on a fixed batch)

**Setup.** Step-2603 LoRA + heads + AdamW moments (`rb2603/.../ckpt_last.pt` via `ckpt_io.load_partial` and `opt.load_state_dict`). Fixed 16 samples = 4 micro-batches × 4 in the D4 deferred order, then clip + AdamW. The dump holds per-micro losses, the accumulated pre-clip gradient (all 18,998,850 trainable values, flattened) and the parameter delta. Table: `logs/equiv_table.jsonl`.

| arm vs base_a | grad cosine | grad rel-L2 | loss max \|Δ\| | delta cosine | verdict |
|---|---|---|---|---|---|
| **base_b (floor)** | 0.99999999975 | **2.3e-5** | **0.0** | 0.99999988 | the base is not bit-identical run to run: FA2 backward accumulates dQ with atomics. The forward **is** deterministic (loss Δ exactly 0) |
| `--img u8` | 0.99999999975 | 2.2e-5 | 0.0 | 0.99999988 | **EQUIVALENT** (at the floor; the inputs are bit-identical) |
| `--order early` | 0.99999999968 | 2.5e-5 | 0.0 | 0.99999991 | **EQUIVALENT** (at the floor) |
| `--n2-once` | 0.9999999303 | 3.7e-4 | 0.0 | 0.9999997 | forward identical, backward ~16× the floor (different accumulation order) |
| `--compile parts` | 0.99921 | **4.0e-2** | 9.1e-4 | 0.99974 | **NOT equivalent at the floor**: the forward numerics changed |
| `--compile-mode max-autotune` | 0.99919 | **4.0e-2** | 2.3e-3 | 0.99975 | **NOT equivalent at the floor**: the forward numerics changed |
| `--bf16-resid` (RECIPE) | — | — | — | — | queue 4 re-dumps it with per-group and WTA-winner fields |

⚠️ **Why a forward-numerics change jumps to 4e-2 while the floor is 2e-5 (HYPOTHESIS, being tested in queue 4).** The trajectory loss is **winner-takes-all**: an ULP-level change in a proposal can flip the argmin, and the gradient then lands on a different proposal. Queue 4 re-runs the dumps with the WTA winner per sample and the per-parameter-group difference. It also adds a **calibration arm** that perturbs only the input images by (1 + 1e-6). If an ULP perturbation alone produces ~4e-2, then any kernel change, cuDNN upgrade or GPU swap moves the gradient this much, and "not equivalent" here means "not bit-reproducible", not "a different recipe". Neither lever is recommended either way: their speed gain is zero.

---

## RESUME / LAUNCH-GATE landing rules per lever (from source, f2f604f)

* **Resume identity** (`train.py:995 IDENTITY_ARGS`, checked at `:1455`) covers:
  * `batch, accum, amp, tf32, fused_adam, compile, lr, epochs, steps, seed, yaw_wrap, nav_mode, labeller, label_backend, …`
  * the content sha256s of the targets, scorer, calib and frozen trunk.

  It does **NOT** cover `workers, label_workers, label_chunks, label_timeout_s, label_python, metric_cache, images, weights` or the logging and checkpoint cadence.
* **`--declare-change` is FORBIDDEN in any launch argv** (`launch_gate.py:55 FORBIDDEN`). So an identity-key change cannot go through the P6 gate at all; it needs a gate-profile change (PI).
* **The P6 profile PINS** `--batch 4 --accum 64 --label-chunks 2 --epochs 25 --lr 2e-4 --amp bf16 …` (`launch_gate.py:280 required_flags`). Any change to those is a profile change (PI).
* **Any argv or code change ⇒ a fresh PASS token.** The token binds commit + argv sha256 + data manifests.
  * A new trainer flag ⇒ a `DVB_KINDS` entry **and** an observed `dvb_rows` row (`gate_checks.py:394/:486`), plus the flag-count pin `test_launch_gate.py:322` (`len(dests) == 61`).

| lever | identity | gate | verdict |
|---|---|---|---|
| `--label-workers N` | not covered | not pinned (`label_workers` is a `built` DVB row) | **rides the resume**; argv change ⇒ gate re-mint (G-LIVE-LAB re-measures) |
| `--label-chunks` | not covered | **pinned to 2** | profile change |
| `--workers` (loader) | not covered | `built` row, not pinned | rides the resume; gate re-mint |
| `--img-u8` (patch) | not needed (bit-identical) | new flag ⇒ DVB row + pin 61→62 (all in `img_u8.patch`) | code commit + gate re-mint; rides the resume |
| `--batch/--accum` | covered | pinned | impossible anyway (batch 8 needs ~82 GB) |
| `--fused-adam`, compile mode | covered / would be new | `--declare-change` forbidden | not worth it (0 % measured) |
| resolution / LoRA depth / bf16 residual | new identity keys | new flags + declared change, forbidden in-profile | **new pre-registration / PI decision** |
