# Flagship v1 — inference optimization: the measured baseline, the lever ladder, and the Orin/Thor path

**Live document.** Refresh in place; do not fork a v2. Owner: Production & Optimization.
Opened 2026-07-20 on Sayed's instruction ("deep analysis, **proven** measures, think Orin and Thor,
document and conserve").

> **Evidence discipline — every number below carries a class, and an unclassified number is a defect.**
> `M-OURS` measured by us on this exact workload · `M-OURS-ALT` measured by us on a *different*
> workload (transfer not established) · `PUB` published elsewhere, cited · `PROJ` arithmetic from our
> own measurements, not itself measured · `UNTESTED`.
> This exists because a "deploy tick 11.16 ms / 89.6 Hz" propagated through three documents for two
> days as a property of a model, when it was a property of a *different operation on different
> hardware at a different checkpoint on a different corpus* (MODEL_REGISTRY §1.2, corrected 07-20).

---

## 1. What we are optimizing — the definition, stated once

**The planning tick** is the only latency that matters for deployment, because it is the path that
produces the trajectory the leaderboard scores:

```
encode(8-frame window, ViT)  ->  20 SEQUENTIAL predictor steps @10 Hz
                             ->  per-step metric Δpose  ->  SE(2) accumulate  ->  trajectory
```

It is **intent-free** — the tactical/strategic path is not in it (that costs a further ~8.5–10.3 ms
if switched on, `M-OURS`). Model: `flagship4b-speedjerk-30k`, 263.4 M params, encoder ViT d768×12
(87.1 M, 9-ch 3-frame stack, 256 px, patch 16 → 256 tokens), operative predictor d768×10 (91.4 M).

**Not to be confused with the decision tick** (`encode(1 frame) + select_K9`), which is a different,
much smaller operation. Both are legitimate; they are not interchangeable. See MODEL_REGISTRY §1.2.

---

## 2. The measured baseline — `M-OURS`

One A40, batch 1, exclusive GPU (`contamination_check.valid == true`), 200 iters after 30 warmup,
`torch.cuda.Event` timing, synchronize-bracketed, step 29,999, physicalai val.
Raw: `taniteval/results/eff_flagship-30k.json`. Harness: `taniteval/taniteval/efficiency.py`.

| precision | tick p50 | tick p99 | encoder | rollout | rollout share | achieved TFLOPs | 10 Hz @p99 |
|---|---:|---:|---:|---:|---:|---:|:--:|
| fp32 | **103.42** | 146.60 | 27.91 | 90.37 | 83.7 % | 3.72 | ❌ |
| tf32 | **93.76** | 102.71 | 15.35 | 87.16 | 92.4 % | 4.26 | ❌ |
| amp16 | **104.49** | 113.13 | 15.77 | 101.65 | 96.7 % | 3.82 | ❌ |

Per rollout step **4.35–5.08 ms**. Encoding one frame instead of the window: **4.71 ms** (fp32).
Total **402 GFLOPs**/tick. Peak memory **1217 MB** (weights 1141.5 MB, activations 75.5 MB).
Batched throughput best **34.8 windows/s** @ batch 32.

**The flagship misses the 10 Hz control budget at p99 in every precision we have measured.**

---

## 3. The diagnosis — launch/serialisation-bound, not arithmetic-bound

Four independent lines of evidence, all `M-OURS`:

1. **Utilisation.** 3.7–4.3 achieved TFLOPs on a device whose fp32/tf32 peak is an order of
   magnitude higher — roughly 5–10 % of the machine.
2. **Precision does not help.** `amp16` is *slower* than `tf32` (104.49 vs 93.76). Autocast inserts
   per-op casts into a chain of small dependent kernels; the casts cost more than the arithmetic they
   save. **Precision cannot fix a launch-bound pass** — this is the single most useful negative result
   in the table.
3. **The comparison arm proves the mechanism.** REF-C-XL does **1.75× the FLOPs** (702 vs 402) and
   runs **2.3–4.0× faster**, achieving **15.9–25.2 TFLOPs**. Its work is one big conv pass plus a
   *batched* 256-anchor fan; ours is 20 dependent steps. Same GPU, same protocol, opposite regime.
4. **The tail.** flagship p50→p99 spreads **42 %** (103.42→146.60); REF-C spreads **1 %**
   (44.28→44.64). Wide tails are the signature of host-side launch jitter, not of arithmetic.

**Consequence for where effort goes.** The rollout is 84–97 % of the tick and the encoder 15–26 %;
with encoder caching (L2) the encoder falls to ~5.6 %. **Encoder optimization is not the lever.**
Token pruning, ToMe, a smaller ViT — all attack the wrong 5 %. Every serious lever must attack the
**serial chain of 20 predictor steps**.

---

## 4. The lever ladder

Ordered by (expected gain × confidence) ÷ cost. Numbers marked `PROJ` are arithmetic on our own
measurements and are **not yet proven** — the measurement campaign opened 2026-07-20 is closing them.

### L1 — Capture the rollout in a CUDA graph ⭐ the decisive lever

**Mechanism.** Record the launch sequence once, replay in a single call. Removes per-kernel launch
overhead *and*, if the graph spans all 20 steps, the 19 inter-step CPU round-trips.

**Evidence.** `M-OURS-ALT` (2026-07-18, RTX 4060, step-6500, comma2k19, fp32/tf32-off, 200 reps):

| pass | eager | CUDA-graph | speedup |
|---|---:|---:|---:|
| **`predict_1pass`** — ONE batch-1 predictor forward | 6.08 ms | **2.36 ms** | **2.57×** |
| `select_K9` — batch 9 | 5.94 ms | 4.45 ms | 1.33× |

Stated mechanism in that note: *"the gain scales with launch-boundedness — batch dilutes it."*

**Why the transfer prior is strong.** Our rollout is **20 sequential batch-1 `predict_1pass` calls** —
the maximally-launch-bound case the 2.57× was measured on, not the diluted one. Static shapes, fixed
step count, no data-dependent control flow: an ideal capture target.

**Accuracy.** Free, and provably so — the graph replays *the same fp32 kernels*. Precedent: rel-err
2.8e-7, max|Δ| 7.6e-6, cosine 1.0, agreement 100 %, decoded-waypoint shift **0.00 m**. Anything
materially worse indicates a broken capture, not a precision trade.

**Projection** (`PROJ`, fp32): rollout 90.37 → 35.16 ms ⇒ tick **103.42 → 48.21 ms**.
**Secondary benefit, hypothesised:** graph replay is far more deterministic than eager launching, so
the 42 % p50→p99 spread should collapse. *For a control loop the tail is the spec* — this may matter
more than the mean.

**Route.** Manual `torch.cuda.CUDAGraph`. `M-OURS`: on our Windows dev box
`torch.compile(mode="reduce-overhead")` fails (`TritonMissing`) and `torch.compile(backend="cudagraphs")`
was **~20× SLOWER** (dynamo guard/re-trace overhead swamps a small predictor). Linux pods can test
both; do not assume `torch.compile` wins.

### L2 — Rolling encoder cache

**Mechanism.** Every tick re-encodes all 8 window frames; 7 were already encoded on previous ticks.
Cache them, encode only the new frame. Pure bookkeeping.

**Gain.** `PROJ` from harness stage timings: saving **23.20 ms** ⇒ tick **103.42 → 84.74 ms** (−18 %).
tf32 saving 12.16 ms; amp16 11.28 ms.

**Accuracy.** Must be **exact** — a deterministic encoder on unchanged frames returns identical
tokens. Any deviation is a bug. **Risk to watch:** this is only valid if the model truly re-encodes
*unchanged* frames; if any preprocessing is tick-dependent, the cache is invalid. Verify, don't assume.

### L3 — True fp16 weights (not autocast)

**Mechanism.** `model.half()` — fp16 weights with no per-op casting, unlike autocast.

**Evidence.** `M-OURS-ALT`: encoder 9.34 → 6.78 ms (1.38×) on the 4060; predictor barely moved
(5.81 → 5.99). The 07-18 note's orthogonality claim: **encoder compute-bound → precision; predictor
launch-bound → CUDA graph.** Our A40 data is consistent (encoder 27.91 → 15.35 under tf32).

**Expected.** Real win on the encoder, ~nothing on the rollout. Combined with L2 the encoder is only
~5.6 % of the tick, so **L3's value largely evaporates once L2 lands** — it matters mainly on Orin,
where fp16 also halves weight-memory traffic on a bandwidth-constrained device.

**Accuracy.** Not free: 07-18 measured agreement 96.9 % (2 flips / 64), waypoint shift 0.7 cm mean /
1.9 cm max — above the ≥95 % deploy bar, and the flips came from fp16, not from the graph.

### L4 — Composition

07-18 found fp16 + CUDA-graph composed **additively with no interference** (measured 11.16 vs
projected 11.21, 0.4 %). Whether that holds when the graph spans 20 steps instead of one pass is
`UNTESTED`. Best-case composition `PROJ`: **~29.5 ms (≈34 Hz)** from the fp32 baseline with L1+L2.

### L5 — Shorten the serial chain itself (architectural)

The only lever that attacks the cause rather than the overhead. 20 steps @10 Hz over 2 s → could a
10-step @5 Hz rollout with interpolation, a temporally-strided rollout, or a distilled multi-step
predictor preserve accuracy? **Latency side is being measured; the accuracy side is `UNTESTED` and
needs the canonical harness — a shorter rollout changes what the model predicts.** Ranked last on
confidence, first on ceiling: it is the only lever whose gain is not bounded by launch overhead.

### L6 — TensorRT

The native Jetson path; also brings layer fusion and quantization. **On Jetson this is L1's
prerequisite, not its rival** (§6.1: precision first, graphs second). Blocked on *hardware access*,
not on a package — engines are not portable across GPU architectures, so this cannot be unblocked on
the dev box at any price. `UNTESTED`.

### L7 — Stop computing the two unused horizon heads ⭐ free, verified, unclaimed

`predictor.py:118-121` computes heads for **all** of `horizons = (1, 2, 4)` on every forward, and
`rollout_decode` consumes only `[1]`. That is **2 wasted heads × 20 steps per tick**, ~252 MB of
needless DRAM reads. Verified directly: `PredictorConfig.horizons = (1, 2, 4)` (`config.py:33`,
re-set at `:280`); `flagship4b_config()` overrides only `depth`, so the tuple survives;
`predictor.py:84` builds one `nn.Linear` per horizon.

Inference-path only — no retraining, no accuracy risk (the discarded outputs are discarded). Small on
the A40, **larger on Jetson** where DRAM traffic is the binding term (§6.2). Being measured.

### L8 — Strided roll on the ALREADY-TRAINED k=2 / k=4 heads ⭐⭐ the strategic lever

**The finding of the night.** Because `horizons = (1, 2, 4)`, the deployed checkpoint **already
contains trained 2-step and 4-step prediction heads** — we have simply never used them. A 2 s horizon
currently costs **20 sequential k=1 steps**; the same horizon is reachable in **10 steps (k=2)** or
**5 steps (k=4)** from weights we already own.

This is the only lever that attacks the serial chain *and* the §6.2 bandwidth floor, which no
kernel-level trick can cross. Combined with L1 it is the difference between a 10 Hz miss and headroom.

**Supporting evidence `[P]`:** DiffusionDrive — REF-C's own reference architecture — measured
1/2/3 decoder steps → 87.9/88.1/88.1 PDMS on NAVSIM (flat), and **20 steps / 130.0 ms / 84.6 PDMS vs
2 steps / 7.6 ms / 88.1 PDMS — 17× faster *and* more accurate**. UniAD/VAD decode 3 s plans
non-autoregressively at 0.5 s spacing; nuScenes/NAVSIM plan at 2 Hz and track to 10 Hz with LQR +
a bicycle model.

**Honest scoping — this is not free.** The ~13 M-param step readout was calibrated on **0.1 s**
transitions, so a strided roll needs recalibration against a *frozen* predictor (hours, no encoder
training). And counter-evidence exists: hierarchical world models have failed flat baselines via
"model exploitation". **The gate must therefore be two-sided AND closed-loop** — open-loop ADE alone
must not decide this, since open-loop does not predict closed-loop here (0.452 → 1.685).

---

## 5. Anti-levers — things that look like wins and are not

| Anti-lever | Why it fails | Class |
|---|---|---|
| **Autocast fp16 (`amp16`)** | *Slower* than tf32 (104.49 vs 93.76). Per-op casts in a launch-bound chain cost more than they save | `M-OURS` |
| **Optimizing the encoder** | 15–26 % of the tick, ~5.6 % after L2. Attacking it cannot fix a 10 Hz miss | `M-OURS` |
| **`torch.compile(backend="cudagraphs")`** | ~20× slower on a small predictor — guard/re-trace overhead dominates | `M-OURS-ALT` |
| **Bigger batches** | Deployment is batch 1. Batching helps offline eval (34.8 windows/s @32) and *dilutes* the CUDA-graph win | `M-OURS` |
| **Quoting the decision tick as the deploy number** | 11.16 ms excludes the entire rollout — the thing we score | `M-OURS` |
| **Offloading to Orin's DLA** | DLA takes `MatMul` only when *"the second input must be a constant"*; attention's `Q·Kᵀ` and `P·V` both have two dynamic inputs. **Thor has no DLA at all** | `PUB` |
| **Buying Thor to fix this tick, then running fp16** | Thor's bandwidth is **1.33×** Orin's, not the headline 7.5× compute ratio. On a weight-streaming rollout that buys ≈1.3×. FP8/FP4 is the actual lever | `PUB`+`E` |
| **Activation quantization** | Weight-only is the replicated-safe half, and weights are 100 % of our binding term anyway — activation quant adds risk for no bandwidth gain | `PUB` |
| **Expecting a kernel trick to beat the bandwidth floor** | 7.31 GB/tick of weight traffic floors the rollout at 17.9 ms (Orin fp16). Only fewer steps or a smaller predictor go below it | `E` |

### 5.1 The finding that reaches beyond latency — CEM planning is infeasible on this rollout

A rollout-based planner costs `n_candidates × horizon × 4.52 ms`. **8 candidates × 20 steps =
723 ms/tick — 7× over the 10 Hz budget** (`M-OURS`, derived from the measured per-step cost).

This is not a v3.5 detail: it applies to **P2/CEM planning over the v1 world model in general** — the
direction D-033 pivoted to and the substrate v3 is built on. The imagine-and-select thesis is
computationally viable only if the candidate count stays very small **or** the per-step cost collapses
(L1) **or** the chain shortens (L8). At 4.52 ms/step CEM is out of reach; at L1's projected
1.76 ms/step, 8 candidates cost 282 ms — still over, but a 4-candidate variant fits.

Conversely, **anchored decoding is cheap**: REF-C's entire 256-anchor fan plus 2 denoise passes is
**8.8 ms** (classifier pass 3.0 ms, each denoise 2.9 ms). "Diffusion is expensive" was never true here.

---

## 6. Orin / Thor

Full analysis: `Research/2026-07-20-orin-thor-deployment-and-inference-levers.md` (769 lines,
every number tagged `[M]`/`[P]`/`[E]`).

### 6.1 My launch-bound hypothesis was right about the end state and WRONG about the order

I predicted that because Jetson's ARM CPU launches kernels more slowly, the workload would be *more*
launch-bound on Orin and CUDA Graphs would matter more. **Partly refuted `[E]`:** in a naive **fp32**
port, Orin's GPU is ~7× slower in arithmetic while its CPU launch path is only ~1.5–2.5× slower
(unified memory also removes the PCIe submission term, which NVIDIA prices at 1–5 µs). Kernels
therefore lengthen *faster* than launches do, and the tick becomes **less** launch-bound, not more.
402 GFLOP against Orin's 5.32 TFLOPS fp32 peak is a **75.6 ms floor at 100 % of peak**; the fp32
encoder alone projects to ~196 ms.

**Corrected ordering — on Jetson: precision FIRST, graphs SECOND.** Only after a TensorRT fp16/int8
engine collapses the GPU work does the tick become launch-bound again, and *then* CUDA Graphs bind
harder than on the A40. On the A40 the launch-bound diagnosis and L1's priority are unchanged.

⚠️ **Do not carry `amp16 > tf32` to Jetson.** That result measures `torch.autocast`'s per-op casts in
an eager chain. A TensorRT fp16 *engine* is a different intervention entirely, and the anti-lever in
§5 does not generalise to it.

### 6.2 The bandwidth floor — and why Thor is not the answer people assume

At batch 1, each of the 20 sequential steps streams the **entire 91.4 M-param predictor** from DRAM
(91.4 M ≫ Orin's 4 MB L2). That is **7.31 GB of weight traffic per tick** in fp32. Against published
bandwidth `[P]`:

| device | bandwidth | rollout floor (fp16) `[E]` |
|---|---:|---:|
| AGX Orin | 204.8 GB/s | **17.9 ms** |
| AGX Thor | 273 GB/s | **13.4 ms** |
| Thor + FP8 | — | **≈6.7 ms** |

**Thor's memory bandwidth is only 1.33× Orin's — not the headline 7.5× compute ratio.** On *this*
workload, **buying Thor and running fp16 buys ≈1.3×.** Thor wins decisively only through **FP8/FP4**,
which halves or quarters the traffic. Procurement-relevant, and it should not be learned after purchase.

**No kernel-level trick can go below that floor.** Only **fewer steps** (L8) or a **smaller predictor**
can — which is what promotes L8 from a nice-to-have to the strategic lever.

### 6.3 The decomposition that settles the diagnosis `[E, reconciles to the measured total within 1.2 %]`

> The **encoder** does **94 % of the FLOPs in 26 % of the time at 36 % of A40 peak**.
> The **rollout** does **5.7 % of the FLOPs in 84 % of the time at 0.67 % of peak**.

Two orders of magnitude apart in efficiency, inside one tick.

### 6.4 Toolchain reality

- **DLA is refuted `[P]`**, from NVIDIA's own operator matrix: DLA supports `MatMul` only when *"the
  second input must be a constant"*, and attention's `Q·Kᵀ` and `P·V` both have two dynamic inputs.
  **Thor has no DLA at all.** Stop spending effort here.
- **TensorRT engines are not portable across GPU architectures `[P]`.** This reframes our
  `import tensorrt` → ModuleNotFoundError blocker: it is a **hardware-access problem, not a Python
  packaging problem.** We need an Orin/Thor in the loop to build the engine. No dev-box fix exists.
- **ONNX/ViT pitfall `[P]`:** fused-MHA is not guaranteed — open NVIDIA issue #4537 (DINOv2 ViT-L/14,
  TRT 10.8) shows the export keeping separate MatMul/Softmax. Verify with `trtexec --dumpLayerInfo`.
  Our `head_dim = 64` satisfies every documented fused-MHA constraint on both sm_87 and Blackwell.
- **Quantization `[P]`:** the asymmetry that replicates across four literatures is **weight-only is
  safe, activation quantization is not** — and weights are **100 % of our binding term**, so
  weight-only captures the entire bandwidth win at the low-risk end. ⚠️ **Nothing published measures
  error accumulation in an open-loop autoregressive metric-regression rollout** (the one paper
  quantizing DINO-WM measures closed-loop CEM replanning, which does not project onto our case).
  Q7 stays open and must be measured, not assumed.

### 6.5 Prerequisites for clean CUDA-graph capture — verified in our code, needs an owner

CUDA graphs require **static memory addresses**. Three sites violate that (all confirmed by direct read):

| site | problem | why it bites |
|---|---|---|
| `metric_dynamics.py:241-242` | 2 allocating `torch.cat`s inside the rollout loop → **38 per tick** | capture fails, or bakes in pointers invalid on replay — the *silent wrong answer* failure mode |
| `predictor.py:112` | rebuilds the causal mask with `torch.triu(torch.ones(...))` **every call** | allocation in the hot loop |
| `predictor.py:118-121` | computes **all 3 horizon heads, discards 2**, 20× per tick (~252 MB wasted DRAM reads) | pure waste — and DRAM traffic is the binding term on Jetson (§6.2) |

---

## 7. Measurement protocol — so numbers stay comparable

Non-negotiable, because a benchmark that drifts is worse than none:

1. **Exclusive GPU.** `contamination_check.valid == true` or the row is not published.
2. **Identical precision flags across every arm.** Silently differing autocast is the classic way to
   publish a fake speedup.
3. `torch.cuda.Event` timing, synchronize-bracketed, warmup discarded, ≥200 iters.
4. **Report p50 *and* p99, and always against the 10 Hz budget.** A control loop lives on its tail.
5. **State the tick definition** with every number (planning vs decision), plus hardware, checkpoint
   and corpus. Five dimensions varied silently once already.
6. **Every latency claim carries an accuracy delta** — max|Δ|, cosine, agreement, waypoint shift.
   A fast wrong answer is not an optimization.

---

## 8. Open measurements — the register

| # | Question | Status |
|---|---|---|
| Q1 | Does the 2.57× CUDA-graph win transfer to the 20-step rollout? Does one graph over 20 steps beat 20 replays of a 1-step graph? | in flight (registry R12) |
| Q2 | Does encoder caching deliver the projected 84.74 ms end-to-end, bit-identically? | in flight |
| Q3 | Does graph replay collapse the 42 % p50→p99 spread? | in flight |
| Q4 | Do L1+L2(+L3) compose additively at 20-step scale? | in flight |
| Q5 | Latency curve at rollout k = 20 / 10 / 5 | in flight (latency only) |
| Q6 | Accuracy cost of a shortened rollout — **now split**: (a) truncating k=1, (b) **strided roll on the existing k=2/k=4 heads** (L8) | **not started** — needs step-readout recalibration + a two-sided **closed-loop** gate |
| Q7 | Quantization error accumulation over 20 autoregressive steps. ⚠️ **Literature does not answer this** — nothing published measures open-loop autoregressive *metric-regression* rollouts | **not started** — must be measured |
| Q8 | Measured Orin and Thor ticks | **blocked — hardware, not toolchain.** Engines are not portable across GPU architectures; no dev-box workaround exists at any price |
| Q9 | Do the three §6.5 static-address violations block capture, and does fixing them change the L1 result? | in flight |
| Q10 | Does L7 (drop 2 unused horizon heads) show measurably on the A40, and how much more on a bandwidth-bound device? | in flight |

---

*Cross-refs: `Project Steering/MODEL_REGISTRY.md` §1.2 (tick definitions), §6 reading 3 (latency as
the flagship-vs-REF-C tiebreaker), §7 R12 · `Research/2026-07-18-predictor-cudagraph-and-numerics-sweep.md` ·
`Research/2026-07-18-combined-tick-and-atomic-archive.md` · `Implementation/predictor_latency/` ·
`Implementation/combined_tick/` · `taniteval/taniteval/efficiency.py`.*
