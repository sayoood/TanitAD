# Flagship v1 on Jetson AGX Orin and AGX Thor — hardware reality, proven inference levers, and the step-count question

- **Agent / date:** Production & Optimization (research half), 2026-07-20 (written 2026-07-21 wall-clock)
- **Scope:** DESK RESEARCH ONLY. No pod was touched, no GPU job was run. The measurement half
  (CUDA-graph capture of the 20-step rollout on the A40 eval pod) is owned by a sibling agent and is
  deliberately **not** duplicated here.
- **Subject:** `flagship4b-speedjerk-30k` (flagship v1, the deployed arm), **planning tick** =
  `encode(8-frame window) → 20 SEQUENTIAL predictor steps @10 Hz → per-step metric Δpose → SE(2) accumulate`.
- **Provenance discipline:** every number below carries a tag —
  **[M]** MEASURED by us · **[P]** PUBLISHED (cited) · **[E]** ESTIMATED (my inference, assumptions named).
  An untagged number is a bug in this note.

---

## 0. TL;DR — the three findings that change the deployment plan

1. **The rollout is not just launch-bound, it is about to become BANDWIDTH-bound, and that floor
   is nearly the same on Thor as on Orin.** At batch 1 each of the 20 sequential steps streams the
   *entire* operative predictor (91.4 M params **[M]**) from DRAM — the weights do not fit in any L2
   (Orin L2 = 4 MB **[P]**). Rollout weight traffic per tick = **7.31 GB fp32 / 3.66 GB fp16 / 1.83 GB int8**
   **[E]**. Divided by device bandwidth **[P]** that is a hard floor of **17.9 ms (Orin, fp16)** vs
   **13.4 ms (Thor, fp16)** for the rollout alone — Thor's memory bandwidth is only **1.33×** Orin's
   (273 vs 204.8 GB/s **[P]**), *not* the headline 7.5×. **Thor only beats Orin decisively on this
   workload if we use FP8/FP4** (which halves/quarters the traffic): Thor-fp8 ≈ 6.7 ms vs Orin-fp16
   ≈ 17.9 ms **[E]**. Buying Thor and running fp16 buys ~1.3×, not 7.5×.

2. **The hypothesis "launch-bound ⇒ CUDA Graphs matter MORE on Jetson" is right about the end state
   and wrong about the order.** In a naive **fp32** port the Orin GPU is ~7× slower in arithmetic
   (5.32 vs 37.4 peak fp32 TFLOPS **[P]**) while its CPU launch path is only ~1.5–2.5× slower **[E]**
   — so the kernels lengthen faster than the launches do and the workload becomes *less* launch-bound,
   not more. Only once you do the thing Orin forces you to do anyway — a **TensorRT fp16/int8 engine
   on the tensor cores**, ~8–16× the fp32 rate — does the GPU work collapse and the tick become *more*
   launch-bound than on the A40. **Order the levers: precision first (a no-op on A40, a 7–16× lever on
   Orin), CUDA graphs second.** Corollary: our measured `amp16 > tf32` slowdown **[M]** must **not** be
   carried to Jetson — it measures `torch.autocast` per-op cast overhead in an eager chain, not fp16
   tensor-core arithmetic.

3. **Cutting the number of sequential steps is the only lever that attacks the floor itself, the
   literature supports it, and we already own the trained weights to test it.** Everything kernel-level
   (graphs, fusion, TRT) is bounded below by 20 × (one predictor pass). The AD literature is
   unusually clean here **[P]**: **DiffusionDrive** (CVPR 2025 — REF-C's own reference architecture)
   measured **1 / 2 / 3 decoder steps → 87.9 / 88.1 / 88.1 PDMS on NAVSIM (flat)**, and against a
   20-step baseline went **130.0 ms / 84.6 PDMS → 7.6 ms / 88.1 PDMS** — *17× faster and more
   accurate*; UniAD and VAD decode a 3 s plan **non-autoregressively at 0.5 s spacing**; and
   nuScenes/NAVSIM already **plan at 2 Hz and track to 10 Hz with an LQR + bicycle model**, so a
   sub-10 Hz waypoint output is the field convention, not a concession. **And verified in our code:
   the deployed v1 predictor was built with `horizons=(1, 2, 4)` (`config.py:279-280`,
   `flagship4b_config()` overrides only depth) — trained k=2 and k=4 heads are already in the
   checkpoint and are computed and thrown away 20× per tick.** A 10-step (5 Hz) rollout needs a
   recalibration of the ~13 M-param step readout against a *frozen* predictor — hours, no encoder
   training. **Two-sided falsifier: it might come out better, since it feeds its own predictions back
   10 times instead of 20.** Counter-evidence is in §4.2 Tier 4 and it is real — hierarchical/jumpy
   world models have failed to beat flat baselines in RL, via "model exploitation", which is exactly
   our open-loop→closed-loop failure mode. **So the eval must be two-sided AND closed-loop.**

**Escalation (integration):** §6.3 lists three code-level defects/micro-levers found by reading
`stack/tanitad/models/predictor.py` and `metric_dynamics.py` that are **prerequisites for clean CUDA-graph
capture**, not optional polish. They need an owner. Also: `taniteval/results/eff_flagship-30k.json` — the
raw JSON every efficiency number in the registry cites — **is not in the repo** (see §7 manifest).

---

## 1. What we actually measured, and what it decomposes into

### 1.1 The measured tick (source: `taniteval/results/eff_flagship-30k.json`, 2026-07-20, A40 exclusive)

| precision | plan_step p50 / p99 | encoder mean | rollout mean | achieved |
|---|---|---|---|---|
| fp32 | 103.42 / 146.60 ms | 27.91 ms (25.9 %) | **90.37 ms (83.7 %)** | 3.72 TFLOP/s |
| tf32 | 93.76 / 102.71 ms | 15.35 ms (16.3 %) | **87.16 ms (92.4 %)** | 4.26 TFLOP/s |
| amp16 | 104.49 / 113.13 ms | 15.77 ms (15.0 %) | **101.65 ms (96.7 %)** | 3.82 TFLOP/s |

All **[M]**. Total 402 GFLOPs/tick, peak 1217 MB, weights 1141.5 MB, per-step marginal 4.35–5.08 ms.

*Two harness caveats worth carrying, both self-documented in `taniteval/taniteval/efficiency.py`:*
- The **percentages are against `mean_ms`, not `p50`** (`_shares`, `efficiency.py:532`). encoder% +
  rollout% > 100 % is therefore expected and the harness says so explicitly
  (`stage_sum_note`, `efficiency.py:594`): *">100 % ⇒ the stages overlap CPU-launch with GPU-execute
  inside the full step (launch-bound)"*. The overshoot is itself evidence, not an error.
- FLOPs are `FlopCounterMode` (conv/matmul/SDPA only, elementwise and norms excluded) — a
  matmul-and-conv **lower bound** (`efficiency.py:178`).

### 1.2 The decomposition that makes the diagnosis unambiguous **[E]**

Analytic FLOPs from the architecture (24·N·d² per transformer block + 4N²d attention):

| stage | shape | FLOPs **[E]** | share of FLOPs | share of time (fp32) **[M]** | achieved rate **[E]** | % of A40 fp32 peak |
|---|---|---|---|---|---|---|
| encoder | 12 blocks, d768, N=256 tokens, ×8 window positions (+patch embed) | **374.5 GFLOP** | **94 %** | 26 % | 13.42 TFLOP/s | **35.9 %** |
| rollout | 10 blocks, d768, N=8 tokens, ×20 steps | **22.7 GFLOP** | **5.7 %** | **84 %** | **0.251 TFLOP/s** | **0.67 %** |
| total | | 397.2 GFLOP | | | | |

**397.2 GFLOP [E] vs 402 GFLOPs [M] — agreement within 1.2 %**, which validates the split.

> **The headline: the rollout does 5.7 % of the arithmetic in 84 % of the time, at 0.67 % of the
> machine's peak. The encoder does 94 % of the arithmetic in 26 % of the time, at 36 % of peak.**
> The encoder is a normal, reasonably efficient compute kernel. The rollout is a 20-deep chain of
> ~150–250 tiny dependent kernels per step whose cost is almost entirely *not* arithmetic.

Per-kernel arithmetic: a d768 predictor GEMM at N=8 is ~9.4 MFLOP — microseconds of GPU work behind
~20 µs of PyTorch eager dispatch. NVIDIA's own breakdown of a framework kernel launch **[P]** is
*language transitions 10–100 µs, runtime 5–20 µs, driver 5–15 µs, hardware submission 1–5 µs*
(21–140 µs total, [CUDA Graph Best Practice for PyTorch](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/introduction.html)).
Our implied ~20 µs/op sits squarely in that band. **This is the textbook CUDA-graph case.**

### 1.3 The bandwidth floor nobody has quoted yet **[E]**

At batch 1 there is no weight reuse: every predictor call reads all its parameters from DRAM
(91.4 M params ≫ Orin's 4 MB L2 **[P]**). `rollout_decode` (`metric_dynamics.py:220-244`) calls
`predictor(win_s, win_a)` **once per step with the full 8-token window** — verified by reading the
code — so this is 20 complete parameter sweeps, not an incremental decode.

**A KV cache does not help us.** The window *slides* (oldest latent is dropped, `metric_dynamics.py:241`),
so cached keys/values are invalid, and in any case a cache would cut token count (FLOPs) not weight
traffic (the binding term).

Weight traffic per tick: encoder 87.1 M + 20 × 91.4 M predictor:

| precision | traffic/tick | A40 (696 GB/s) | **Orin (204.8 GB/s)** | **Thor (273 GB/s)** |
|---|---|---|---|---|
| fp32 | 7.66 GB | 11.0 ms | 37.4 ms | 28.1 ms |
| fp16 | 3.83 GB | 5.5 ms | **18.7 ms** | **14.0 ms** |
| int8 / fp8 | 1.91 GB | 2.7 ms | 9.4 ms | 7.0 ms |
| fp4 (Thor only) | 0.96 GB | — | — | 3.5 ms |

Rollout only (drop the encoder term): A40 fp32 **10.5 ms** vs measured **90.37 ms** → we are **8.6×
above the bandwidth floor**, which is exactly what "launch-bound with headroom" looks like. On Orin
in fp16 the floor is **17.9 ms**; on Thor in fp8, **6.7 ms**.

> **Consequence: a perfect CUDA graph + perfect TRT engine on Orin still cannot put the rollout below
> ~18 ms in fp16.** After that, the only remaining levers are *fewer steps* or *a smaller predictor*.
> That is why §4 matters more than §3.

---

## 2. Hardware reality — Orin vs Thor vs the A40 we measured on

### 2.1 Verified vendor figures **[P]**

| | **A40** (our eval pod) | **Jetson AGX Orin 64 GB** | **Jetson AGX Thor T5000** |
|---|---|---|---|
| GPU | Ampere GA102, 84 SM, 10,752 CUDA cores, 336 3rd-gen TC | Ampere, **16 SM (8 TPC)**, 2048 CUDA cores, **64 3rd-gen TC** | **Blackwell**, **10 TPC**, 2560 CUDA cores, **96 5th-gen TC** |
| GPU max clock | ~1.74 GHz boost | **1301 MHz** (MAXN) | **1575 MHz** (MAXN) / 1386 MHz (120 W default) |
| Peak FP32 | **37.4 TFLOPS** | **5.32 TFLOPS** | not published |
| Peak TF32 TC | **74.8 TFLOPS** | supported (sm_87) | supported |
| Tensor-core AI | 149.7 FP16 / 299.3 INT8 dense TOPS | **170 sparse INT8 TOPS** (GPU) → 85 dense INT8 / **42.5 dense FP16 TFLOPS [E]** | **2070 FP4 sparse TFLOPS**, **1035 FP8 TFLOPS** |
| Precisions | fp32/tf32/fp16/bf16/int8 | fp32/tf32/fp16/bf16/**int8** — **no fp8, no fp4** | + **FP8 and NVFP4**, **next-gen Transformer Engine that dynamically switches FP4↔FP8** |
| Memory | 48 GB GDDR6, **696 GB/s** | 64 GB 256-bit LPDDR5, **204.8 GB/s**, **unified (CPU+GPU, zero-copy)** | 128 GB 256-bit LPDDR5X, **273 GB/s**, unified |
| L2 | 6 MB | **4 MB** | not published |
| CPU (the kernel launcher) | server x86 host | **12× Arm Cortex-A78AE, max 2201.6 MHz** | **14× Arm Neoverse-V3AE, max 2601 MHz** |
| DLA | — | **2× NVDLA v2.0 @ 1.6 GHz** (~105 of the 275 total TOPS) | **none** — NVDLA workloads move to GPU or PVA |
| MIG | no | no | **yes** |
| Power | 300 W | 15 / 30 / 50 W + **MAXN (up to 60 W)** | 70 / 90 / **120 W (default)** / MAXN, envelope 40–130 W |

Sources: [A40 datasheet figures](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf) ·
[Jetson AGX Orin Technical Brief v1.2](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) ·
[Jetson Linux Developer Guide r36.4.3 — Orin power modes](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html) ·
[Jetson Thor product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) ·
[Introducing NVIDIA Jetson Thor (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) ·
[Jetson Linux Developer Guide r38.4 — Thor power modes](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html) ·
[RidgeRun Jetson AGX Thor overview](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_AGX_Thor).

**Power modes — the number that matters is the GPU TPC count, not the watts.** Orin's official
nvpmodel table **[P]**: MAXN = 12 CPU @2201.6 MHz + **8 TPC @1301 MHz**; 50 W = 12 CPU @1497.6 + 8 TPC
@**816 MHz**; 30 W = 8 CPU @1728 + **4 TPC** @612 MHz; 15 W = 4 CPU @1113.6 + **3 TPC** @408 MHz.
**15 W mode disables 5 of 8 TPCs and drops the GPU clock 3.2×** — i.e. ~8× less GPU than MAXN, and it
also cuts the CPU to 4 cores at half clock, which hurts the launch path *and* the compute path at once.
Thor **[P]**: MAXN = 14 CPU @2601 + 10 TPC @1575 MHz; 120 W (default) = 14 CPU + 10 TPC @1386 MHz;
90 W/70 W = 12 CPU + **6 TPC** @1530 MHz. NVIDIA explicitly warns MAXN is *"an experimental mode…
we don't recommend running heavy workloads for prolonged periods"* and that hardware throttling engages
above TDP **[P]** — **so any latency we quote must name its nvpmodel mode, exactly as our tick numbers
name their precision.**

**Two discrepancies I am flagging rather than resolving:**
- Thor tensor-core generation: NVIDIA's own blog says **"96 fifth-generation Tensor Cores"**; RidgeRun's
  wiki says "4th Gen". NVIDIA is the primary source → **5th gen**.
- Thor's FP8 = 1035 TFLOPS is from RidgeRun, not an NVIDIA page, and it is almost certainly the
  *sparse* figure (it is exactly half the NVIDIA FP4-sparse number). Dense FP8 ≈ 518 TFLOPS **[E]**.
- Orin's dense FP16/INT8 rates are **[E]**, derived from NVIDIA's published *170 sparse INT8 TOPS*
  under the standard 2× sparsity convention. Several secondary sites garble this into "170 dense INT8
  TOPS and 85 FP16 TFLOPS"; that is internally inconsistent with 16 SM × 1.301 GHz.

### 2.2 How a launch-bound, batch-1, 402-GFLOP workload actually scales — hypothesis tested

**The hypothesis under test (Sayed):** *because the bottleneck is kernel-launch and dependency latency
rather than FLOPs, and because Jetson's ARM CPU is much weaker at launching kernels than a server x86
host, the workload gets relatively worse on Orin than a naive FLOPs/bandwidth ratio predicts — therefore
CUDA Graphs matter MORE on Jetson.*

**Verdict: the conclusion is right, the reasoning has a hole, and the hole reverses the order of work.**

*Where the hypothesis is right:*
- The launch path **is** CPU-serial work and Orin's CPU **is** weaker. Orin tops out at
  **2201.6 MHz on a Cortex-A78AE** **[P]** — a 2020-class in-order-ish mobile core — versus a server
  x86 host. NVIDIA's own PyTorch CUDA-graph guide states graphs help most when *"pairing older CPUs with
  modern GPUs"* and in *"complex software stacks with substantial dispatch overhead"* **[P]**.
- PyTorch eager dispatch (Python + ATen) is the single largest launch term (10–100 µs **[P]**), and it
  is pure CPU work that scales ~inversely with single-thread performance.
- Thor is materially better here than Orin: **Neoverse-V3AE @2601 MHz, 14 cores** **[P]** — a far wider,
  newer core. If the deployment is launch-bound, **Thor's CPU is a bigger part of its advantage than
  its GPU is.**

*Where the hypothesis has a hole:*
1. **Jetson is a unified-memory SoC.** NVIDIA's launch-cost breakdown assigns 1–5 µs to *"hardware
   submission (PCIe overhead, GPU scheduling)"* **[P]**; on an iGPU there is no PCIe. So the launch cost
   does **not** scale with CPU clock alone — part of it gets *cheaper*. Net growth Orin-vs-x86 is
   plausibly **1.5–2.5×** **[E]**, not 7×. *Unknown, would need measurement* — NVIDIA has publicly
   declined to publish Jetson launch-latency reference data, telling a customer on their own forum
   *"we don't have such data for Jetson"* and offering desktop Ampere numbers instead **[P]**
   ([forum thread](https://forums.developer.nvidia.com/t/orin-cuda-graph-latency-is-too-long/309359)).
2. **The kernels lengthen faster than the launches do.** In fp32 the Orin GPU is ~7.0× slower on peak
   arithmetic and ~3.4× slower on bandwidth **[P]**. Launch-boundedness is the *ratio* of launch cost to
   kernel duration; multiplying the numerator by ~2 and the denominator by ~3–7 makes the workload
   **less** launch-bound on Orin, not more. A naive fp32 port of today's PyTorch tick would be
   arithmetic-starved *first*: **402 GFLOP against Orin's 5.32 TFLOPS fp32 peak is a 75.6 ms floor at
   100 % of peak** **[E]** — i.e. the fp32 encoder alone (which achieves 36 % of peak **[M]**) projects
   to **~196 ms** on Orin **[E]**, blowing the 100 ms budget before the rollout starts.
3. **But that regime is one you must leave anyway.** Moving to fp16/int8 tensor cores multiplies Orin's
   arithmetic ceiling by 8–16× (5.32 → 42.5 → 85) **[E from P]** while leaving the CPU launch cost
   untouched. **After the precision move, the tick is more launch-bound on Orin than it ever was on the
   A40, and the hypothesis holds exactly as stated.**

**Therefore the ordering is:**

| # | lever | effect on A40 | effect on Orin |
|---|---|---|---|
| 1 | **fp16/int8 TRT engine (tensor cores)** | ~nil (measured: `amp16` is *slower*) **[M]** | **7–16× on the arithmetic ceiling, 2–4× on the bandwidth floor** **[E]** — mandatory |
| 2 | **full-tick CUDA graph** | large (rollout is 8.6× above its bandwidth floor) **[E]** | **larger still, after step 1** **[E]** |
| 3 | fewer sequential steps / smaller predictor | linear on the floor | linear on the floor |

**Do not carry `amp16 > tf32` to Jetson.** That measurement is `torch.autocast` inserting per-op
`.half()/.float()` casts into an *already* launch-bound eager chain on a GPU whose fp32 path is 7× faster
than Orin's. A TensorRT fp16 engine stores fp16 weights, emits no casts, halves DRAM traffic, and runs on
tensor cores. Different intervention, different device, different regime — the A40 result has **no
predictive value** for it.

### 2.3 Orin vs Thor for *this* workload, honestly

NVIDIA's *"7.5× higher AI compute"* **[P]** compares Thor's **2070 FP4-sparse TFLOPS** to Orin's
**275 INT8-sparse TOPS** — different number formats. On a like-for-like basis third parties put it at
**3–4×** **[P]**, and for our tick the relevant ratios are:

| ratio Thor ÷ Orin | value | relevance to our tick |
|---|---|---|
| memory bandwidth | **1.33×** (273 / 204.8) **[P]** | **binding for the 20-step rollout at batch 1** |
| CPU max clock × core width | 2601 vs 2201.6 MHz, V3AE vs A78AE **[P]** | binding for the launch path (bigger than 1.18× in practice **[E]**) |
| tensor-core arithmetic at matched precision (fp16) | ~6× **[E]** | binding for the encoder |
| lowest usable precision | fp8/fp4 vs int8 **[P]** | **the real Thor lever** — halves/quarters rollout DRAM traffic |
| DLA | 2 → 0 **[P]** | irrelevant (see §3.7) |

**Read:** Thor's advantage on the *encoder* is large and immediate. Thor's advantage on the *rollout* is
only ~1.3× unless we move to FP8/FP4, in which case it becomes ~2.7×. **If the deployment target is
Thor, FP8 is not an optimisation, it is the reason to buy Thor.**

---

## 3. The proven-technique list for exactly this bottleneck

### 3.1 The lever table (liftable into the master optimization doc)

Evidence class: **A** = measured by us on this model · **B** = published, measured on a *similar*
workload (batch-1 transformer / launch-bound chain) · **C** = published but on a *dissimilar* workload
(CNN, training, large batch) — do not assume transfer · **D** = vendor claim / mechanism only.

| # | lever | mechanism | expected gain (and on WHAT it was measured) | evidence | accuracy risk | impl. cost |
|---|---|---|---|---|---|---|
| **L1** | **Full-tick CUDA-graph capture** (encode + 20 steps + readout as ONE graph) | Replaces ~3–5 k per-kernel CPU launches with one graph launch (~10 µs **[P]**). CUDA ≥12.6 makes straight-line graph launch **constant-time**: 2 µs + ~1 ns/node vs 2 µs + 200 ns/node before **[P]** — our rollout *is* straight-line | **2.57×** on our *select* pass, 4060, batch 1 **[M]** (⚠ NOT the 20-step rollout). 1.70× Mask R-CNN / 1.12× BERT, MLPerf training, many GPUs **[P, class C]**. **Our own bound: A40 rollout is 8.6× above its bandwidth floor [E] ⇒ headroom for ~3–8× on the rollout** | **A** (bound) + **B** | **None.** Graph replays the identical kernels: our own measurement was rel-err 2.8e-7, decoded-waypoint shift **0.00 m** **[M]** | Medium — needs static shapes/addresses (§6.3 blockers) |
| **L2** | **TensorRT engine + layer fusion** (unrolled 20-step rollout) | Fuses LayerNorm/GELU/bias/residual into GEMM epilogues; picks tactics per shape; removes the framework entirely | TRT vs PyTorch **+17.7 %** on Jetson Orin NX at batch 2, YOLO-class CNN **[P, class C — do not transfer]**. **No published batch-1 sequential-transformer-rollout number found — unknown, would need measurement** | **D**/C | Low if fp32/fp16 engine; verify per-layer | Medium-High (build on target, see §5) |
| **L3** | **fp16 precision** (TRT engine, not autocast) | Halves weight traffic (the rollout floor) *and* moves onto tensor cores (8× fp32 rate on Orin) | **1.38×** on our encoder, 4060 **[M]**. ~2× with negligible accuracy loss is the standard Jetson result **[P, class C]**. Rollout bandwidth floor 37.4 → **18.7 ms** on Orin **[E]** | **A**+B | Low. Our ONNX/TRT-relevant graph already exports parity-clean at fp32 (max\|Δ\| 8.8e-6 / 1.2e-5 **[M]**) | Low |
| **L4a** | **WEIGHT-ONLY int8/fp8 on the predictor** | Halves/quarters the *weight* traffic — which is 100 % of our binding term — without touching activations | Orin rollout floor → **9.4 ms**; Thor-fp8 → **7.0 ms** **[E]** | **B** — the weight/activation asymmetry replicates across 4 literatures; **BitTP** (a trajectory-regression transformer) went **1.58-bit weight-only ADE/FDE 0.35/0.62 → 0.30/0.49, i.e. BETTER than BF16** **[P]** | **Moderate.** Must still be gated (§3.5.5) | Medium |
| **L4b** | **Activation quantization (W8A8 → W4A4)** | Further arithmetic gains | — | **B (negative)** — DINO-WM itself: W8A8 near-lossless, **W4A4 collapses to ≈0 success**; BitTP +activation blew ADE up **5×** **[P]** | **HIGH — see §3.5.** 20 autoregressive steps compound error into metres, and our margin over the kinematic floor is only ~0.05 m | High |
| **L5** | **Encoder caching** (encode only the newest frame; reuse 7 embeddings) | The window slides by 1; 7/8 of the encoder work is redundant | **103.42 → 84.74 ms** on A40 **[M]**; `encode_1frame` = 4.71 ms vs `encode_window` 27.91 ms **[M]**. On Orin the absolute saving is ~7× larger **[E]** | **A** | **None if the embeddings are bit-identical** — must verify no window-position/pos-embed coupling | Low-Medium |
| **L6** | **Fewer sequential steps** (10 @5 Hz via the already-trained k=2 head, or a distilled few-step predictor) | Attacks the serial chain itself — linear on *both* the launch count and the bandwidth floor | 20→10 halves the 17.9 ms Orin fp16 rollout floor to **8.9 ms [E]**. **DiffusionDrive, NAVSIM: 20 steps 130.0 ms / 84.6 PDMS → 2 steps 7.6 ms / 88.1 PDMS — 17× faster AND more accurate [P]**; 1/2/3 steps all 87.9/88.1/88.1 (flat) | **B** (strong, our domain) — see §4.2 | **Real, must be re-gated two-sided (open- AND closed-loop).** Counter-evidence exists (§4.2 Tier 4) | **Medium** — readout recalibration only, frozen predictor (§4.4) |
| **L7** | **Smaller predictor** (distil 91.4 M → ~25 M) | The rollout floor is *linear* in predictor params, because they are re-read 20× | 3.7× on the rollout floor **[E]** | **D** | Real — re-gate | High |
| **L8** | **Kernel fusion (hand or TRT)** | Removes *global-memory round trips*, which CUDA graphs do **not**: NVIDIA is explicit that *"kernels inside a graph run separately and pass intermediate results through global memory"* **[P]** | 3× on a `sum(abs(x))` bandwidth microbench, RTX 4090 **[P, class C]** | **D**/C | None (numerically equivalent fusions) | High if hand-written; free inside TRT |
| **L9** | **DLA offload** | — | **REFUTED, see §3.7** | — | — | — |
| **L10** | **Batching** | — | **N/A for deployment.** Measured best 34.8 windows/s @ batch 32 **[M]**, but the vehicle has batch 1 | — | — | — |

### 3.2 CUDA Graphs — the mechanism and why it is the right lever here

CUDA graphs "reduce overheads by providing a mechanism to launch multiple GPU operations through a
single CPU operation" **[P]** ([Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)).
The canonical NVIDIA demo — **20 short kernels per timestep**, structurally identical to our 20-step
rollout — moved per-kernel time from 3.8 µs to 3.4 µs against a 2.9 µs kernel duration on a V100 **[P]**.
That modest ratio is because their kernels were 2.9 µs of *real* work; ours are ~µs of work behind ~20 µs
of PyTorch dispatch, so our headroom is much larger.

Two facts make graphs *better* for us than the generic case:
- **Straight-line topology.** Our rollout is a pure chain (every node has exactly one dependent). CUDA
  12.6+ launches straight-line graphs in **constant time** — measured 300 µs → 2.5 µs for a 2000-node
  graph, and 2 µs + 200 ns/node → 2.5 µs + ~1 ns/node, on an Intel Xeon Silver 4208 + RTX 3060 **[P]**
  ([Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)).
  A ~3–5 k-node capture of our whole tick would launch in ~µs.
- **TensorRT composes with it.** TRT's own best-practices doc says CUDA graph capture is *"particularly
  useful when enqueue and launch times take longer than actual GPU executions"*, naming *"small GEMM
  sizes or mostly element-wise operations, where CUDA kernels take 5-15 microseconds to launch per
  kernel"* **[P]** — a description of our rollout. Pattern: `cudaStreamBeginCapture` →
  `context->enqueueV3(stream)` → `cudaStreamEndCapture`; call `enqueueV3` once first to flush deferred
  shape updates; one execution context per captured graph **[P]**.

**Constraints (all **[P]**, all satisfiable by us):** static shapes, static memory addresses, no
data-dependent control flow, no CPU sync inside the capture region, no dynamic allocation. **Our tick is
fully static** (W=8, K=20, 256 px) — but three code sites currently violate the "static addresses / no
per-call allocation" rule; see §6.3.

**`torch.compile(mode="reduce-overhead")` vs manual capture.** `reduce-overhead` is inductor + CUDA
graphs and needs **Triton**. On our Windows dev box it fails outright (`TritonMissing`) and the
Triton-free `backend="cudagraphs"` is **~20× SLOWER** (117.8 ms vs 6.08 ms eager) because per-call dynamo
guards swamp a tiny model **[M]**. On Jetson, Triton for aarch64 generally has to be **built from
source** **[P]** and torch's CUDA backend is tuned for dGPUs. **Recommendation: target the deployment
runtime with hand-rolled `torch.cuda.CUDAGraph` capture or TRT+`cudaStreamBeginCapture`; treat
`torch.compile` as a dev convenience, never as the deployment path.** (This is a re-confirmation of the
2026-07-18 finding, now with the Jetson toolchain reason attached.)

### 3.3 fp16 — the one precision claim that IS safe to make

fp16 is the only precision step that is simultaneously (a) a tensor-core enabler on Orin, (b) a 2×
cut in the binding DRAM traffic, and (c) numerically benign for a network trained in fp32/bf16 with
LayerNorm. It is also the only one we have *already* measured on this model's encoder (1.38× on the
4060 **[M]**). **Do fp16 unconditionally; treat int8/fp8 as a separate, gated project (§3.5).**

### 3.4 Kernel fusion for small transformer blocks

NVIDIA is explicit that fusion and graphs solve **different** problems: *"Graphs provide microsecond
host-side savings; fusion eliminates the actual memory work"* **[P]**
([Kernel Fusion in NVIDIA CUDA](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead)).
For us fusion matters *after* graphs, because it is the only lever besides precision that touches the
DRAM floor. In practice we get it for free from TensorRT (LayerNorm/GELU/bias/residual epilogues); a
hand-written fused block is not worth it until TRT's per-layer report shows a specific unfused chain.

**TensorRT fused multi-head attention (FMHA)** is the one fusion worth chasing explicitly for the
encoder (N=256 tokens). Support and constraints **[P]**
([Working with Transformers, TRT 10.x](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-with-transformers.html)):
head size 16–256 divisible by 8 on SM75–SM90 (Orin is sm_87); **8–128 divisible by 8 and no
sequence-length restriction on Blackwell SM100/SM110** (Thor); FP8 requires head size divisible by 16
and *"is expected to outperform FP16 MHA when head size ≥ 128 or sequence length ≥ 128"*. **Our
d_model 768 / 12 heads = head_dim 64 — inside every window, including Thor's tighter 8–128 range.**
Note the corollary: FP8 MHA will help the **encoder** (seq 256) and will **not** help the **predictor**
(seq 8) — where attention is negligible anyway (0.2 % of predictor FLOPs **[E]**).

### 3.5 Quantization of a 20-step autoregressive metric regressor — the risk nobody has bounded

*This is the section where I am least willing to give you a number.*

**Bottom line up front: the literature contains no measurement of quantized-vs-fp32 error across the
steps of an open-loop autoregressive latent rollout with a metric-regression decoder. Nothing published
substitutes for the horizon-stratified experiment at the end of this section.** What the literature
*does* establish is a set of consistent, cross-domain warnings, and one asymmetry strong enough to be
a design rule.

#### 3.5.1 The one asymmetry that replicates everywhere: **activations, not weights**

Four independent literatures land on the same split **[P]**:

| study | workload | weight-only | + activation quantization |
|---|---|---|---|
| **BitTP** ([arXiv:2605.29705](https://arxiv.org/abs/2605.29705)) — *the most on-point single data point: a trajectory-prediction transformer, a regression+rollout task* | ETH/UCY pedestrian ADE/FDE | **1.58-bit weight-only matched or BEAT BF16**: 0.35/0.62 → **0.30/0.49** | **catastrophic: 0.35/0.62 → 1.84/3.12** (5× blow-up) |
| **World-model quantization study** ([arXiv:2602.02110](https://arxiv.org/abs/2602.02110)) — quantizes **DINO-WM itself** | Wall (nav) / PushT (manip) planning success | per-group weight-only recovers where per-tensor W4 collapses | **W8A8 near-lossless** (Wall 0.94–0.98); **W4A4 collapses** (Wall ≈0.00–0.08); 3-bit never recovers — *"quantization noise overwhelms the learned transition dynamics"* |
| **NVIDIA TAO** (PeopleNet detection) | mAP | — | naive INT8 PTQ **78.37 % → 59.06 %**, recovered to **78.06 % with QAT**; NVIDIA explicitly recommends **excluding regression layers** from quantization |
| **Qualcomm FP8-vs-INT8** ([arXiv:2303.17951](https://arxiv.org/abs/2303.17951)) | ImageNet ViT vs DeepLabV3 segmentation | — | see 3.5.3 — the format that wins **flips** between classification and dense continuous output |

**Design rule that follows: quantize weights harder than activations.** Weights are also exactly where
our bandwidth problem lives (§1.3) — the 7.31 GB of rollout traffic is *parameters*, not activations.
**Weight-only quantization buys us the entire bandwidth win at a fraction of the accuracy risk.** This
is the single most actionable finding in the quantization literature for us.

#### 3.5.2 Why the generic "INT8 is within 1 %" claim does not apply to us

1. **We are a regression head, not a classifier.** A classifier's argmax absorbs quantization noise;
   our output is a metric Δpose in metres feeding an SE(2) accumulation, with no absorbing
   nonlinearity. Evidence is suggestive but not universal: a spacecraft-pose study
   ([arXiv:2407.06170](https://arxiv.org/abs/2407.06170)) found the **position (regression)** head
   degraded far more than **orientation (soft-classification)** at INT8 (+12.7 % vs +1.5 % error) —
   **but this reversed at more aggressive bit-widths**, so "regression is always worse" is *not* a safe
   law. The object-detection PTQ papers that motivate themselves on exactly this split (Reg-PTQ,
   CVPR 2024; HQOD, [arXiv:2408.02561](https://arxiv.org/abs/2408.02561)) never publish the
   cls-head-vs-box-head numbers that would settle it. **This is a genuine hole in the field.**
2. **Error compounds 20 times.** The most explicit published statement of accumulation is from
   diffusion PTQ — **PTQD** ([arXiv:2305.10657](https://arxiv.org/abs/2305.10657), NeurIPS 2023):
   *"quantization noise may accumulate, resulting in a low SNR during later denoising steps."*
   **PTQ4DM** ([arXiv:2211.15736](https://arxiv.org/abs/2211.15736), CVPR 2023) quantifies the cost of
   ignoring it: calibrating at a single timestep gives **FID 49.37** vs **23.96** with timestep-spread
   calibration (FP baseline 21.63, ImageNet-64 DDIM-250). **TFMQ-DM**
   ([arXiv:2311.16503](https://arxiv.org/abs/2311.16503), CVPR 2024 Highlight) and **EDA-DM**
   ([arXiv:2401.04585](https://arxiv.org/abs/2401.04585)) independently confirm activation statistics
   shift substantially across steps. ⚠ **Caveat I am keeping honest:** most of these measure
   *distribution mismatch across steps*, not literally *error growth as a function of step count*; no
   paper produced a clean "error vs number of steps" curve. And diffusion denoising is not physical-time
   rollout — the mechanism (iterated refinement with shifting statistics) is analogous, the task is not.
3. **The one paper that quantizes our literal model family cannot answer our question.** The DINO-WM
   quantization study's horizon axis is **CEM/MPC replanning iterations** — a closed loop where each
   iteration gets a fresh chance to correct — **not** sequential autoregressive latent steps in one
   open-loop rollout. Its apparent "longer horizon rescues 4-bit" pattern is most likely a replanning
   artefact and **our open-loop, SE(2)-composed 20-step rollout has no such rescue mechanism.** What
   *does* transfer is its negative result: joint W4A4 and 3-bit collapse and do not recover.
4. **Our own numbers show how little headroom there is.** Flagship v1's ADE@2s is **0.4522 ± 0.0312 m**
   against a kinematic floor of 0.5005 m and CV 0.8248 m **[M, registry]**. The whole margin over the
   best trivial baseline is **~0.05 m**. A quantization regression of 0.05 m — invisible in any
   published "<1 % accuracy loss" framing — **erases the model's entire claim to beat the floor.**

#### 3.5.3 Format choice: FP8 is not automatically safer than INT8 for *our* output type

The best independent vision benchmark of the two formats (Qualcomm,
[arXiv:2303.17951](https://arxiv.org/abs/2303.17951)) **[P]** found they trade wins by task shape:
**ViT classification favoured FP8-E4M3** (−0.19 pp vs −1.33 pp for INT8, ImageNet), but
**dense-prediction segmentation collapsed under FP8** (DeepLabV3/Pascal VOC: **−34.98 pp** E4M3,
**−66.38 pp** E5M2) while **INT8 stayed near-lossless (−1.67 pp)**. Our stack contains *both* shapes —
a ViT encoder (favours FP8) and a dense continuous-output decoder (the case where FP8 failed). **So
"Thor has FP8, therefore quantization is safe" is not a supported inference. Mixed precision by module,
chosen by measurement.**

**NVFP4/FP4: treat as untested for us.** Every concrete NVFP4 accuracy figure I could find — including
NVIDIA's own — is an **LLM reasoning/code/math benchmark** (MMLU-Pro, GPQA, AIME). **No source tests
FP4 on a ViT, on dense prediction, or on a regression task.** *Unknown, would need measurement.*

#### 3.5.4 ViT-specific quantization hazards (the encoder)

Well documented and consistent **[P]**: **LayerNorm** per-channel range ratios up to **622:1** (Swin-B,
vs ~5:1 for a ResNet — FQ-ViT, [arXiv:2111.13824](https://arxiv.org/abs/2111.13824)); **post-Softmax**
distributions where **99.2 % of the mass is below 0.3**, which breaks uniform quantizers (RepQ-ViT,
[arXiv:2212.08254](https://arxiv.org/abs/2212.08254)); and GELU/post-LayerNorm outlier channels
(PTQ4ViT, [arXiv:2111.12293](https://arxiv.org/abs/2111.12293)). The good news for a frozen-style
encoder: RegCache ([arXiv:2510.04547](https://arxiv.org/abs/2510.04547)) shows naive PTQ of a frozen
pretrained vision encoder collapsing, but PTQ4ViT/RepQ-ViT recovering to within ~0.3–1.7 points of
FP32. ⚠ **Every one of those numbers is single-pass classification accuracy — none was measured
through a downstream multi-step autoregressive predictor.**

#### 3.5.5 The recommendation, and the experiment that settles it

**Strategy, in descending order of confidence:**
1. **fp16 everywhere first** (§3.3) — this alone captures the 2× bandwidth win at essentially no risk.
2. **Weight-only INT8/FP8 on the predictor blocks**, keeping activations, the `step_readout`, and the
   ViT at fp16. This is the best-evidenced configuration in the whole search (§3.5.1) and it targets
   exactly the traffic that binds us.
3. **Per-channel / per-group calibration**, never per-tensor, for anything transformer-shaped **[P]**.
4. **If weight-only INT8 shows measurable degradation, go straight to QAT** rather than iterating on
   PTQ calibration tricks. That pattern (PTQ collapses → QAT recovers most of it) repeats in every
   task-adjacent source that tested both: TAO detection 78.37 → 59.06 → 78.06; CILRS driving on CARLA
   NoCrash 82 % → 34 % → 62 % ([arXiv:2505.15304](https://arxiv.org/abs/2505.15304)); D4RL
   HumanoidWalk 550 → 205 → 535.
5. **Never** joint W4A4 or 3-bit — collapse is documented on our literal model family.

**The experiment (nothing published substitutes for it):** run the deployed checkpoint open-loop, no
replanning, on the fixed val set at fp32 and at each candidate precision, and **plot per-step Δpose
error AND the SE(2)-accumulated error at every one of the 20 steps** — not just ADE@2s.
- **Gate:** paired episode-cluster bootstrap (`taniteval/ci.py`, per CLAUDE.md — never
  `overlapping_holdout_se`). **Falsifier: any ADE@2s degradation whose paired CI excludes 0, or any
  point-estimate degradation > 0.02 m, kills that precision for the rollout.**
- **The diagnostic that isolates the compounding claim:** ADE at 0.5 / 1 / 1.5 / 2 s is already
  reported for fp32 **[M, registry]**. **If quantization error compounds, the degradation ratio must
  GROW with horizon.** Flat across horizons ⇒ a per-step bias, correctable by recalibrating the step
  readout. Superlinear growth ⇒ keep the predictor's residual/state-carrying path at higher precision
  regardless of budget.
- **Ablate the three axes separately** — weight-only vs weight+activation; predictor vs encoder;
  `step_readout` precision independent of the backbone. The detection-quantization literature has
  consistently declined to run that last split; we should not repeat the omission.

### 3.6 What quantization the *hardware* will and will not give us

- **Orin has no FP8 and no FP4** — its lowest tensor-core precision is INT8 **[P]**. So on Orin the
  choice is fp16 or the high-risk INT8 path; there is no low-risk middle.
- **Thor has FP8 and NVFP4 with a Transformer Engine that dynamically switches FP4↔FP8** **[P]**. FP8
  (E4M3) keeps an exponent and is far better behaved for a regression network than INT8's fixed scale.
  **This is a real, concrete reason to prefer Thor for this specific model** — it offers a 2× traffic
  cut at materially lower numerical risk than Orin's only option.

### 3.7 DLA — verified NOT usable for our model

**Refuted, as suspected.** The decisive fact is in NVIDIA's own DLA operator matrix **[P]**
([NVIDIA/Deep-Learning-Accelerator-SW operators README](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/operators/README.md)):
**`MatMul` is supported only when "the second input must be a constant."** Attention's two core
operations — `Q·Kᵀ` and `P·V` — have *two dynamic inputs* by construction. There is no way to express
self-attention within that constraint. Secondary evidence agrees that transformer/attention layers do
not run on DLA as of JetPack 6.2 and that DLA additionally forbids dynamic shapes and plugins **[P]**.
Everything else follows: a ViT + a 10-block transformer predictor would fall back to the GPU for every
attention layer, and the GPU↔DLA hand-offs would cost more than the DLA saves.

**And on Thor the question disappears entirely: Thor has no DLA** — NVIDIA directs former NVDLA
workloads to the GPU or PVA **[P]**. **Do not spend any further effort on DLA.**

---

## 4. The question that decides the architecture — can we cut 20 sequential steps?

*(Literature ranking below; the mechanisms and our own arithmetic first.)*

### 4.1 Why this dominates every kernel-level trick

| variant | rollout weight traffic **[E]** | **Orin fp16 floor** | **Thor fp8 floor** | launches to hide |
|---|---|---|---|---|
| 20 steps (today) | 3.66 GB | **17.9 ms** | 6.7 ms | ~3–5 k |
| 10 steps @5 Hz + interpolate | 1.83 GB | **8.9 ms** | 3.4 ms | ~1.5–2.5 k |
| 4 steps @2 Hz + interpolate | 0.73 GB | **3.6 ms** | 1.3 ms | ~600–1000 |
| 20 steps, predictor distilled 91.4 M → 25 M | 1.00 GB | **4.9 ms** | 1.8 ms | ~3–5 k |
| 10 steps **and** 25 M predictor | 0.50 GB | **2.4 ms** | 0.9 ms | ~1.5–2.5 k |

**Step reduction and predictor shrinkage are multiplicative, and both are linear in the floor.** No
kernel-level lever can go below the top row. This is why §4 is worth more than §3.

### 4.2 The literature, ranked by evidence strength

**Tier 1 — direct measurement in our own domain (AD planning, nuScenes/NAVSIM/nuPlan)** — all **[P]**:

| # | claim | evidence |
|---|---|---|
| **T1.1** | **Coarse planning + downstream tracking is already the field convention, not a novel risk.** nuScenes prediction: 6 s horizon at **2 Hz = 0.5 s spacing**. NAVSIM: 4 s horizon, **simulated at 10 Hz via an LQR controller + kinematic bicycle model tracking a single planned trajectory**. | [nuScenes prediction challenge](https://eval.ai/web/challenges/challenge-page/591/overview) · NAVSIM, [arXiv:2406.15349](https://arxiv.org/abs/2406.15349) |
| **T1.2** | **One-shot decoding of a multi-second trajectory is the dominant SOTA paradigm in AD, not a fringe idea.** UniAD's plan query attends once to BEV features and decodes **6 waypoints over 3 s (0.5 s spacing)** non-autoregressively — planning L2 avg 1.03 m, collision 0.31 % (nuScenes, CVPR 2023 Best Paper). VAD: **29.0 % lower collision, 2.5× (Base) / 9.3× (Tiny) faster** than prior methods. | UniAD [arXiv:2212.10156](https://arxiv.org/abs/2212.10156) · VAD [arXiv:2303.12077](https://arxiv.org/abs/2303.12077) |
| **T1.3** | **⭐ Cutting iterative steps on an AD trajectory decoder cost literally zero accuracy in the one place it was measured directly.** DiffusionDrive, same 60 M model, NAVSIM: **1 / 2 / 3 denoising steps → 87.9 / 88.1 / 88.1 PDMS — flat.** Against a vanilla 20-step diffusion baseline: **20 steps, 130.0 ms, 7 FPS, 84.6 PDMS → 2 steps, 7.6 ms, 45 FPS, 88.1 PDMS** (RTX 4090). **Fewer steps was both 17× faster AND more accurate.** | DiffusionDrive, CVPR 2025, [arXiv:2411.15139](https://arxiv.org/abs/2411.15139) — *this is REF-C's own reference architecture (D-030)* |
| **T1.4** | Reducing the cost of an autoregressive world-model chain in AD has been attempted and yields **1.6–2×** — well short of a step-count collapse, and by adaptive skipping rather than a smaller fixed step count. | DISK, [arXiv:2602.00440](https://arxiv.org/abs/2602.00440), NuPlan+NuScenes |
| **T1.5** | **Autoregressive decoding does not convincingly beat one-shot even where it is most defensible.** AMP (autoregressive) vs MTR++ (one-shot) on Waymo: minADE 0.6012 vs **0.5906**, minFDE 1.1918 vs 1.1939, mAP 0.4334 vs 0.4329; on the Interaction split one-shot leads on 4 of 6 metrics. AMP's own authors concede a "small gap". | MotionLM (ICCV 2023) · AMP [arXiv:2403.13331](https://arxiv.org/abs/2403.13331) |

**Tier 2 — direct measurement, adjacent domain** — all **[P]**:

- **Distillation works, but naive step-cutting does not.** Consistency Policy
  ([arXiv:2405.07503](https://arxiv.org/abs/2405.07503), RSS 2024), Robomimic ToolHang success:
  **DDPM 27 steps 0.79 · DDiM 9 steps (naive cut, no distillation) 0.14 · CP 1-step (distilled) 0.70 ·
  CP 3-step 0.77.** Real-robot latency 192–198 ms → 21–22 ms (~9×). **The lesson is precise: the
  distillation training, not the step count, is what preserves accuracy — a naive stride will look far
  worse than a distilled few-step student.**
- **Direct multi-step beats recursive composition *conditionally*, not universally.** The most rigorous
  recent statement: single-step models win when **well-specified**; direct multi-step wins under
  **model misspecification / partial observability**
  ([arXiv:2504.01766](https://arxiv.org/abs/2504.01766), 2025). Econometrics reached the identical
  conditional conclusion 18 years earlier (Chevillon 2007), and the NN5 competition review found
  *"Multiple-Output strategies are the best performing approaches"* (Ben Taieb et al. 2012). See also
  Asadi, *Combating the Compounding-Error Problem with a Multi-step Model*
  ([arXiv:1905.13320](https://arxiv.org/abs/1905.13320)), and Informer (AAAI 2021), whose one-shot
  decoder exists explicitly to avoid autoregressive *"cumulative error"*.
- **Jumpy / temporally-abstract latent prediction is a real, long-standing family**: TD-VAE
  ([arXiv:1806.03107](https://arxiv.org/abs/1806.03107), ICLR 2019), Clockwork VAE
  ([arXiv:2102.09532](https://arxiv.org/abs/2102.09532), NeurIPS 2021), Compositional Planning with
  Jumpy World Models ([arXiv:2602.19634](https://arxiv.org/abs/2602.19634), **+200 % relative on
  long-horizon tasks**), VLWM ([arXiv:2606.21775](https://arxiv.org/abs/2606.21775), +13 %). **All in
  RL / video / robotics — none in driving.**

**Tier 4 — the counter-evidence, which I am NOT going to bury:**

- **Hierarchical/temporally-abstract world models failed to beat flat baselines** in a systematic
  study: at k=8 (Nav2d, PointMaze, Reacher) hierarchical ≈ flat *at best, never better*; **at k=4 on
  HalfCheetah the flat baseline clearly won**, the largest gap in the paper. Diagnosed root cause:
  **"model exploitation"** — coarser abstraction gives the planner more room to exploit world-model
  inaccuracies (Schiewer/Subramoney/Wiskott, *Scientific Reports* 2024,
  [arXiv:2406.00483](https://arxiv.org/abs/2406.00483)). **[P]**
- **This maps directly onto our known weakness.** Our own measured open-loop → closed-loop collapse
  (**0.452 m → 1.685 m**, divergence >5 m in 22.2 % **[M, registry]**) is exactly the regime where
  "model exploitation" bites. **A jumpier model could look fine open-loop and fail closed-loop, so the
  step-reduction eval MUST be two-sided: open-loop ADE *and* closed-loop divergence.**

### 4.3 Verdict on "can we defensibly cut 20 steps to 10 or 4?"

**Yes for the DECODE, conditionally for the STATE PROPAGATION, and the distinction is the whole answer.**

| lever | what it cuts | evidence | verdict |
|---|---|---|---|
| **Strided decode + interpolation** (predict at 5 Hz / 2 Hz, interpolate to the 10 Hz controller) | the *output* rate | **T1.1 field convention + T1.2 SOTA + T1.3 measured-free** | **Defensible today.** Lowest risk lever in this note; it is what the rest of the field already ships |
| **Direct multi-horizon head** (one pass → many Δposes) | serial steps entirely | **T1.2/T1.3 strong in AD; T1.5 says autoregression isn't buying accuracy anyway** | **Defensible, but it is a research-direction decision** — it trades away the action-conditioned recursion that makes this a *world model*, and interacts with the v3 planning pivot (D-033) |
| **Distilled few-step predictor** | the latent chain itself | **Mechanism proven for denoising (Tier 2), unproven for physical-time latent rollout; nearest AD attempt (T1.4) got only 1.6–2×** | **Plausible, highest ceiling, highest cost.** Do NOT naively stride without distilling — Consistency Policy measured naive-cut 0.14 vs distilled-1-step 0.70 |
| **Jumpy / hierarchical latent** | the latent chain itself | **contested even within RL** (Tier 2 positive vs Tier 4 negative) | **Weakest evidence. Not a deployment lever.** |

**The single unknown that decides it, and which no paper can answer for us:** is our 20-step rollout
currently limited by **compounding error** (⇒ fewer/direct steps help, possibly a lot) or by
**within-step underfitting** (⇒ more, smaller steps are right and cutting them costs accuracy)? Per
[arXiv:2504.01766](https://arxiv.org/abs/2504.01766) this is *exactly* the variable that flips the
answer, and it is a property of our model, not of the literature.

### 4.4 The cheapest possible version of this experiment — testable on the current checkpoint

**⭐ We already own the asset — verified in code.** The operative predictor builds **one head per entry
in `cfg.horizons`** (`predictor.py:83-84`) and the deployed preset sets
**`horizons=(1, 2, 4)`** — `flagship4b_config()` derives from `base250cam_config()`, which builds
`PredictorConfig(d_model=768, depth=12, n_heads=12, window=8, horizons=(1, 2, 4), action_dim=2)`
(`config.py:279-280`), and `flagship4b_config()` overrides **only the depth** (12 → 10,
`config.py:327`). Meanwhile `rollout_decode` consumes **only `[1]`** (`metric_dynamics.py:236`).

> **A trained k=2 head (⇒ a 10-step, 5 Hz rollout) and a trained k=4 head (⇒ a 5-step, 2.5 Hz rollout)
> already exist inside the deployed v1 checkpoint and have never been used at inference.** They are
> computed and discarded on every one of the 20 steps (§6.3).

**Honest scoping — this is cheap, not free.** Two variants must not be confused:

- **(A) Strided ROLL — the real lever.** Roll with the k=2 head, 10 times, each covering 0.2 s. This
  halves the rollout cost. **But `StepDisplacementReadout` decodes Δpose from a latent *pair* and was
  calibrated on 1-step (0.1 s) transitions** (`metric_dynamics.py` module docstring), so it would
  systematically mis-scale a 0.2 s transition. **The readout must be recalibrated** — a ~13 M-param
  grounding head against a *frozen* predictor and *frozen* encoder. Hours, not days; no encoder
  training; no parity risk (same corpus, same key). Also re-index `future_actions` so the action
  conditioning matches the new stride.
- **(B) Strided DECODE only.** Roll 20 times, decode every other transition. Truly zero training —
  and **worth nothing for latency**, because the roll is the cost, not the decode. Useful only as a
  numerical sanity check.

**Protocol for (A):** same 40 val episodes, `taniteval/ci.py` **paired** episode-cluster bootstrap,
report **both** open-loop ADE@0.5/1/1.5/2 s **and** closed-loop divergence (per §4.2 Tier 4).
**Two-sided falsifier — cutting steps could plausibly come out BETTER**, since a 10-step rollout feeds
its own predictions back 10 times instead of 20, and our closed-loop number says this model's error does
grow when it consumes its own output.

### 4.5 What our own conventions already permit

The 2 s horizon is scored at 0.5 s granularity in the registry (ADE@0.5/1/1.5/2 s **[M]**) and the
planning gate is ADE@2s. **Nothing in our scoring requires 10 Hz waypoint output** — 10 Hz is the
*control* rate, and per T1.1 the field standard is to plan at 2 Hz and let a downstream controller
track. This is a much smaller change to the deliverable than it first appears.

---

## 5. Deployment toolchain reality

### 5.1 The path, end to end

```
flagship4b-speedjerk-30k ckpt.pt (PyTorch, fp32)
  └─ torch.onnx.export  ── ALREADY MEASURED CLEAN [M]: opset 17 legacy AND dynamo,
     │                     encoder+readout max|Δ| 8.8e-6, predictor 1.2e-5, NO unexportable ops
     │                     (MHA / FiLM / causal triu / AvgPool all export)
     ├─ (Orin) JetPack 6.2 : L4T 36.4.3, Ubuntu 22.04, CUDA 12.6, TensorRT 10.3, cuDNN 9.3   [P]
     │                        or JetPack 7.x (Orin family is supported)                       [P]
     ├─ (Thor) JetPack 7.0 : CUDA 13.0, TensorRT 10.13, Ubuntu 24.04, kernel 6.8              [P]
     │         JetPack 7.2 : L4T 39.2, CUDA 13.2.1, TensorRT 10.16.2, cuDNN 9.20.0            [P]
     ├─ trtexec --fp16 (or torch-tensorrt)  → .engine   ← MUST BE BUILT ON THE TARGET
     └─ runtime: enqueueV3 wrapped in cudaStreamBeginCapture/EndCapture (one CUDA graph)
```

### 5.2 The things that will bite this specific model

1. **⚠ Engines are not portable. Build on the target.** *"Without hardware compatibility mode, TensorRT
   engines are not portable across different GPU architectures"* **[P]**
   ([Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)).
   `kSAME_COMPUTE_CAPABILITY` / `kAMPERE_PLUS` exist but cost performance, and *whether either is
   supported on Jetson is UNVERIFIED*. **Plan for an on-device build step; do not plan to ship an
   engine built on the A40.** Practical consequence: **we need physical Orin/Thor hardware in the loop**
   — the A40 cannot stand in for the engine build the way it can for a latency microbench.

2. **⚠ The 20-step loop.** TensorRT requires *"the shape of Loop carried dependencies must be the same
   across all loop iterations"* and *scan output length cannot be dynamic* **[P]**; CUDA-graph capture
   explicitly fails on *"loops, conditionals, and data-dependent shapes"* **[P]**. Our trip count is a
   **fixed 20**, so both options are open:
   - **(a) Unroll into one engine.** One `enqueueV3`, cross-step fusion possible, single graph.
     Cost: ~20× the graph size and build time; the ONNX file becomes large.
   - **(b) One-step engine, enqueued 20× inside a single captured CUDA graph.** Small engine, fast
     build, but no cross-step fusion. *This is the lower-risk first build.*
   **Do not export the loop as an ONNX `Loop` node.**

3. **⚠ FMHA fusion is not guaranteed for ViTs.** Open, unresolved NVIDIA issue: **DINOv2 ViT-L/14,
   TensorRT 10.8, PyTorch 2.7.1, opset 23 — the exported graph keeps separate `MatMul`/`Softmax` and no
   fused-MHA layer appears** **[P]** ([NVIDIA/TensorRT #4537](https://github.com/NVIDIA/tensorrt/issues/4537)).
   Since our encoder is a ViT, **verify fusion explicitly** with
   `trtexec --verbose --dumpLayerInfo` and look for the fused attention layer; if absent, restructure the
   exported attention to match the documented pattern (scale applied to Q or K *before* the attention;
   mask feeding an `Add`/`Where`/`Sub`) **[P]**. Our head_dim of 64 satisfies every documented head-size
   constraint on both sm_87 and Blackwell.

4. **⚠ Windows export gotcha (already ours):** the dynamo exporter prints emoji and dies with
   `UnicodeEncodeError` under cp1252 — export must run with `PYTHONUTF8=1` **[M, 2026-07-08]**.

5. **torch-tensorrt on Jetson** **[P]** ([Torch-TensorRT in JetPack](https://docs.pytorch.org/TensorRT/getting_started/jetpack.html)):
   a prebuilt wheel exists for **JetPack 6.2** paired with **torch 2.8.0 / torchvision 0.23.0** from the
   JPL index. Building from source needs `bazelisk-linux-arm64` + `ninja-build` and **OOMs on Orin**
   without `build --jobs=2` in `.bazelrc` or a swapfile. **Recommendation: use `trtexec` on the ONNX we
   already export cleanly — it has fewer moving parts than torch-tensorrt and our ONNX path is already
   parity-proven [M].**

### 5.3 Unblocking the dev box (`import tensorrt` → ModuleNotFoundError)

A working setup requires, in order of preference:
1. **Build on the target device.** Since engines are not portable anyway (§5.2.1), the dev box does not
   *need* TensorRT — it needs ONNX (which we have) and an Orin/Thor to build on. **This reframes the
   blocker: it is a hardware-access problem, not a Python-package problem.**
2. **If a dev-box TRT is still wanted** (for op-coverage checking only, never for a shippable engine):
   Linux x86 with `pip install tensorrt` + `onnxruntime-gpu` (CUDA-12 EP), or the
   `nvcr.io/nvidia/tensorrt` NGC container. **On Windows the CUDA/TRT execution providers are the
   recurring failure** — our own record shows ORT exposing only CPU/Azure EPs **[M, 2026-07-18]**.
   A Linux box or WSL2 with the CUDA toolkit is the realistic route.
3. `torch.compile` on the dev box additionally needs a **Windows Triton wheel** matching torch
   2.11+cu128 — still outstanding from the 2026-07-18 note.

---

## 6. Concrete next actions

### 6.1 For the measurement half (sibling agent) — what this analysis predicts

Stated so it can be falsified rather than confirmed after the fact:

- **Prediction:** a full-tick CUDA graph on the A40 takes the rollout from **90.37 ms** into the
  **12–30 ms** band and the tick from **103.42 ms** to **~40–55 ms** **[E]**.
- **Hard lower bound:** the rollout **cannot** go below **~10.5 ms** on the A40 (fp32 weight-streaming
  floor **[E]**). If a measurement claims better, the harness is wrong (or the weights are being cached
  in a way I have not accounted for).
- **Falsifier for the whole launch-bound diagnosis:** if the graphed rollout stays **above ~45 ms**,
  launch overhead is *not* the dominant term and the next suspect is the per-step allocation churn in
  §6.3, then the `step_readout`.

### 6.2 Recommended deployment sequence (do them in this order)

1. **fp16 TensorRT engine** (mandatory on Orin; the single largest lever there). Gate: paired
   episode-cluster bootstrap on ADE@2s vs fp32.
2. **Full-tick CUDA graph** over the TRT context. Gate: rel-err and decoded-waypoint shift vs eager
   (our 4060 precedent: rel-err 2.8e-7, wp-shift 0.00 m **[M]**).
3. **Encoder caching** — already measured worth 18.7 ms on the A40 **[M]**, worth ~7× more on Orin **[E]**.
4. **The strided-rollout experiment (§4.4)** — uses weights we already trained; halves the rollout
   floor; the AD literature says it may cost nothing (§4.2 T1.3). Two-sided, closed-loop-inclusive gate.
5. **WEIGHT-ONLY int8/fp8 on the predictor blocks**, activations and `step_readout` left at fp16 —
   this captures the full bandwidth win at the low-risk end of the evidence (§3.5.1/§3.5.5).
6. **Activation quantization only if 1–5 still miss the budget, and only behind the horizon-stratified
   gate of §3.5.5.** Never joint W4A4 or 3-bit.

### 6.3 ⚠ ESCALATION — three code sites that block clean graph capture (need an owner)

Found by reading the code, not by running it. Each is a **prerequisite** for L1, not polish:

| site | issue | why it matters |
|---|---|---|
| `stack/tanitad/models/predictor.py:112` | `torch.triu(torch.ones(w, w, device=…))` **builds the causal mask on device on every call** — 20× per tick for a compile-time constant | Per-call allocation; a graph-capture hazard and 40 wasted kernel launches per tick |
| `stack/tanitad/models/predictor.py:118-122` | `forward` computes **every** horizon head (`for k in cfg.horizons`), but `rollout_decode` consumes only `[1]` (`metric_dynamics.py:236`). **Verified: v1's operative predictor has `horizons=(1, 2, 4)`** (`config.py:279-280`, unchanged by `flagship4b_config()`) | **2 of 3 `Linear(768→2048)` heads computed and discarded 20× per tick** ≈ **252 MB of wasted DRAM reads/tick in fp32 [E]** plus 40 wasted launches. Free win — **and those same discarded heads are the trained asset that makes the §4.4 strided-rollout experiment cheap.** Do not delete them; stop *evaluating* them |
| `stack/tanitad/models/metric_dynamics.py:241-242` | Two `torch.cat`s per step slide the window, allocating fresh `[B,8,S]` / `[B,8,A]` tensors — **40 allocations per tick** | Violates the "static memory addresses" requirement of graph capture **[P]**. Fix = a preallocated ring buffer written in place |

`predictor.py:85` also carries `self.out_proj` marked *"reserved: feed predictions back"* — never called
at inference; harmless at runtime, but it should be excluded from the deployment export.

---

## 7. Bounds, gaps, and what I could not answer

- **The raw JSON is not in the repo.** Every measured tick number here traces to
  `taniteval/results/eff_flagship-30k.json`, which the registry cites — but `taniteval/results/` **does
  not exist in the working tree** (checked). The numbers in this note are taken from the registry's
  efficiency block and the task brief, which agree exactly. **This is a reproducibility gap, not a
  numerical doubt.** Flagged in the manifest.
- **A registry/brief discrepancy I did not resolve:** the registry lists `operative 96,609,283` params;
  the brief gives the predictor as **91.4 M**. The difference is plausibly the multi-horizon heads. All
  bandwidth floors above use **91.4 M**; using 96.6 M raises them **~5.7 %** and changes no conclusion.
- **No published measurement of CUDA-graph speedup on Jetson for a transformer rollout exists** that I
  could find. NVIDIA declined to publish Jetson launch-latency reference data on their own forum **[P]**.
  **Unknown, would need measurement.** The experiment: time a 20-iteration chain of d768 GEMMs eager vs
  graph-captured on both an A40 and an Orin at MAXN, and report the *ratio of ratios*. That single
  microbench settles §2.2 definitively and needs no checkpoint.
- **No published batch-1 sequential-transformer-rollout TensorRT speedup** was found. Every TRT number
  I saw was CNN/large-batch/throughput. **Do not let a CNN speedup into the master doc.**
- **No published quantized-vs-fp32 error curve across the steps of an open-loop autoregressive latent
  rollout with a metric-regression decoder exists — for any architecture.** The single paper that
  quantizes DINO-WM measures **closed-loop CEM replanning iterations**, a metric that structurally
  cannot be projected onto our open-loop case. **Unknown, would need measurement** — the experiment is
  specified in §3.5.5.
- **The regression-vs-classification quantization-sensitivity question is genuinely unsettled in the
  field.** Only two studies isolate output type on a shared backbone, they partially disagree at
  extreme bit-widths, and the object-detection PTQ literature that could settle it declines to publish
  the split. Do not quote "regression is more fragile" as established.
- **No source tests FP4/NVFP4 on a ViT, on dense prediction, or on a regression task.** Every published
  NVFP4 accuracy figure is an LLM benchmark.
- **Two literature items are flagged low-confidence by the search and should be re-verified before
  citation:** the DESTINE "1 Hz optimal coarse rate" ablation ([arXiv:2310.07438](https://arxiv.org/abs/2310.07438),
  secondary-source only) and StepbaQ (NeurIPS 2024, arXiv ID unconfirmed).
- **Thor availability, price, thermals in a vehicle, and automotive qualification** were out of scope.
- **I did not verify** whether `kAMPERE_PLUS` hardware-compatibility mode is supported on Jetson, or
  Thor's L2 size, or Thor's dense (non-sparse) FP8 figure from an NVIDIA primary source.

---

## Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| This note | `repo:TanitAD Research Hub/Production & Optimization/Research/2026-07-20-orin-thor-deployment-and-inference-levers.md` (staged) | no — content is derivable from the cited sources |
| Analytic FLOPs / bandwidth-floor model (§1.2, §1.3, §4.1) | inside this note | **YES — nowhere else** |
| Code-site findings (§6.3) | inside this note; the code is in `repo:stack/tanitad/models/{predictor,metric_dynamics}.py` | **YES — the analysis is only here** |
| `taniteval/results/eff_flagship-30k.json` (the source of every tick number) | `tanitad-eval:/root/taniteval/results/` — **NOT in the repo** | **YES — pod only. Rescue it.** |

No source file was modified. No pod was touched. No commit, no push.
