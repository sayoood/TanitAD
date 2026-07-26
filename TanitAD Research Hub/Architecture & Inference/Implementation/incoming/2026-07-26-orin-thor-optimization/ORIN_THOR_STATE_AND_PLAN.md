# Orin / Thor deployment — state assessment, the real budget, and a ranked plan

- **Date:** 2026-07-26 · **Stream:** Orin/Thor inference optimization (memory + compute) · **Model:** `flagship4b-speedjerk-30k` (flagship v1, deployed)
- **Hardware in this session:** **NO Orin, NO Thor.** Every device number below is a **floor or a bracket**, labelled `ESTIMATED`, computed from `MEASURED` architecture quantities and `PUBLISHED` vendor specs. No desktop-GPU millisecond is presented as an embedded millisecond.
- **Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another doc, not re-verified) · `ESTIMATED` (arithmetic on a MEASURED/PUBLISHED input, operation stated) · `HYPOTHESIS`.
- **New measurement this session:** the **per-component** memory + compute budget — params, FLOPs, activation bytes and DRAM weight traffic broken out by encoder / predictor / heads / planner, plus arithmetic intensity per component. This had never been measured; the program had only a 401.9 GFLOP whole-tick total, and has been reasoning about the wrong currency because of it.

> **Headline.** The deployed tick is **not** the object the program has been optimizing. Measured: it is **74.24 GFLOP**, not 401.9 (that figure is the *uncached* 8-frame encode). Its FLOPs are **63 % encoder**; its **DRAM bytes are 95.5 % rollout**. On Orin the rollout sits **28× below the machine balance** and on Thor **129×** below it — so the binding term on both chips is the operative predictor's **weight streaming**, and *every* lever that trades FLOPs (fusion, KV-caching, crop reduction, window shortening) is worth **≈ 0 %**. 10 Hz clears on both chips at FP16 across the entire uncertainty bracket. Thor's headline 7.5× compute advantage collapses to **1.33×** unless it is spent on **FP8/NVFP4 weight compression** — which is exactly the one question that is fully answerable *without* Thor, and is therefore the next experiment.

---

## 1. State assessment — what is done, what is unverified, what is only planned

Two prior intakes exist and are decision-grade for what they measured. Read them first: `…/incoming/2026-07-22-orin-thor-deployment/` (export + TRT-FP16 + plan) and `…/incoming/2026-07-23-orin-int8-benchmark/` (the INT8 verdict, real weights). This section is the honest gap.

### 1.1 DONE and VERIFIED (`MEASURED`, artifact on disk)

| # | Item | Evidence | Where |
|---|---|---|---|
| D1 | Static-shape ONNX export of the **deployed** arch (encoder `[1,9,256,256]→[1,2048]`, predictor `states[1,8,2048], actions[1,8,3]`), opset 17, no unexportable ops | torch-vs-ORT parity **1.25e-6 / 1.9e-6** (random init, A40) and **1.996e-6 / 1.490e-6** (**real ckpt**, A6000) | `2026-07-22…/artifacts/export_report.json`; `2026-07-23…/orin_int8_benchmark.json` |
| D2 | ONNX → **TensorRT-FP16 engine builds**, no plugin needed | enc **1.205** / pred **0.666** ms (A40, random init); enc **1.1315** / pred **0.5837** ms (A6000, real weights) | `trt_fp16_report.json`; `orin_int8_benchmark.json` |
| D3 | **TRT fuses our MHA** — the open NVIDIA #4537 "ViT MHA won't fuse" risk does not bite us on SM 8.6 | no standalone Softmax, Myelin/foreign block, both graphs, **both runs** | same |
| D4 | **CUDA-graph capture of the 20-step rollout succeeds with EXACT equivalence** (0.0 m), reproduced on two independent A40s | 96.40 → 27.87 ms (3.46×); eval pod 95.03 → 28.73 ms (3.31×) | `bench_latency_report.json`; `taniteval/results/eff_levers_flagship-30k.json` |
| D5 | **Composed L4 planning tick 18.7476 ms p50 / 18.7641 p99** (fp16 weights + pruned horizon heads + cached 1-frame encode + whole-rollout graph) | A40, batch 1, exclusive under `gpu_lock.sh`, contamination-checked | `eff_levers_flagship-30k.json` → `levers.all_levers` |
| D6 | **INT8 weight+activation is REFUTED on BOTH gates.** Latency: enc +2.1 %, pred **−2.1 %** (a wash, no win). Accuracy: encoder `readout_head` collapses to **cos 0.566**; 20-step rollout **+0.0215 m**, past the 0.02 m falsifier, with the compounding signature (27× growth 0.5 s→2 s) | A6000, real ckpt step 29999, entropy-calibrated, 256 real calib samples, 880 windows | `2026-07-23…/DEPLOYMENT_READINESS_INT8.md` §2–4 |
| D7 | **INT8 weight-only PASSES the accuracy gate** — blanket cos 0.99947 (enc) / 0.9999997 (pred); rollout **+0.0065 m**, 3× margin under the falsifier | same run, PyTorch fake-quant, real weights | same, §3–4 |
| D8 | **Param count 263,442,838** reproduced **three** times independently (registry; pod1 strict-load, zero missing/unexpected keys; this session) | exact integer match | `budget_report.json` `params.reproduces_registry_total_model = true` |
| D9 | ⭐ **NEW — the per-component budget** (params, FLOPs, activation bytes, DRAM traffic, arithmetic intensity), and an **independent reproduction of the registry's 401.922 GFLOP uncached tick to 4 decimal places** (401.9215) on a different host and torch version | this session | `artifacts/budget_report.json`, `budget_probe.py` |
| D10 | ⭐ **NEW — lever quantification without silicon**: encoder FLOP(N) curve (R² 1.000, n=7, window 64–1024 tokens), KV-cache ceiling + its code-level blocker, fan-batching intensity curve, quantization-format numerics for FP8/NVFP4/MXFP4/INT4 | this session | `artifacts/lever_report.json`, `lever_probe.py` |

### 1.2 IMPLEMENTED but UNVERIFIED — the honest middle

| # | Item | Why it is not a result |
|---|---|---|
| U1 | **The TensorRT engine itself** | Built only on **A40 (SM 8.6)** and **A6000 (SM 8.6)**. `PUBLISHED`: TRT engines are **not portable across GPU architectures**. Nothing has ever been built for SM 8.7 (Orin) or Blackwell (Thor). What transfers is the *path*, not the engine and not one millisecond of it. |
| U2 | **"≈14 ms/tick" TRT rollout** (`DEPLOYMENT_PLAN` §3.2) | `ESTIMATED` from a **single-call** 0.666 ms × 20; it omits the step-readout, the window slide and any unhidden launch. Never a measured 20-step tick. ⚠️ It is also one rounding away from colliding with the **retracted 14.331 ms** decision-tick figure — never quote it bare. |
| U3 | **Strided k=2 / k=4 rollout** (10 / 5 predictor calls for the same 2 s) | Latency `MEASURED` (graphed 57.53 → 42.88 → 35.69 ms). Accuracy is **explicitly UNMEASURED and not claimable** — the artifact says so in its own `ACCURACY` field. This is the single largest un-gated lever we own. |
| U4 | **Weight-only INT8/FP8 as a real TRT engine** | Never built. The 07-23 weight-only numbers are a **PyTorch fake-quant simulation** (the right tool for accuracy, not a latency number). TRT's calibrated path gives W+A only; QDQ-scoped weight-only is real export surgery. |
| U5 | **fp16 + fp32 SE(2) accumulate** | `MEASURED` in the torch harness (deviation 0.0241 → 0.0127 m) but never inside an engine, and never re-scored on the canonical 40-episode val set at engine precision. |
| U6 | **`BENCHMARK_PLAN.md` Gate A on target** | The harness is specified and the A40/A6000 half is built. The decision-grade half (per-layer kernel selection) is arch-specific and cannot run. |
| U7 | **v4 deployment delta** | `INHERITED` from `V4_FLAGSHIP_DESIGN.md` + a REF-C decoder timing. v4 is still training (~8 h from 30k as of this session). Its 9.78 M diffusion planner has **never** been FLOP-counted or byte-counted on the deploy path. |

### 1.3 PLANNED ONLY — no measurement exists, on any hardware

| # | Item | Blocker |
|---|---|---|
| P1 | **Any Orin or Thor tick, of any kind** | no silicon |
| P2 | The **shippable engine** (on-device build + calibration) | engines not portable (`PUBLISHED`) |
| P3 | **FP8** and **NVFP4** — latency *and* accuracy | latency needs Blackwell; **accuracy does not** (see §5) |
| P4 | Orin **INT8 kernel selection** (the 2.7× trap) | arch-specific — and now largely moot, INT8 W+A is refuted on accuracy anyway |
| P5 | **nvpmodel power mode × sustained thermal clock** | never named in any artifact. A Jetson number without its power mode is as unquotable as a tick without its definition. |
| P6 | **Achieved DRAM efficiency on Jetson** | the single parameter that collapses the 3.3×-wide bracket in §3 |
| P7 | Jetson **launch latency** (settles precision-first vs capture-first ordering) | NVIDIA has declined to publish it |
| P8 | **DLA** | 🔴 **REFUTED, closed.** DLA `MatMul` requires the second input to be constant; attention's `Q·Kᵀ` and `P·V` both have two dynamic inputs. **Thor has no DLA at all.** Stop spending effort here. |

### 1.4 ⚠️ Naming defect, before it becomes the fifth tick value

The brief calls **18.75 ms** "the deployed decision tick". Per `MODEL_REGISTRY.md` §1.2 that number is the **optimised *planning* tick** (lever `all_levers`). The registry reserves "**decision tick**" for `encode(1 frame) + select_K9`, which has three of its own values (11.16 ms **retracted**; 17.75 ms fp32 @ step 6,500; **14.331 ms** traced 2026-07-26 to an RTX 4060 / fp32-eager / comma2k19 n=30 / **`base250cam` random-init** measurement — *not* the deployed checkpoint). **Throughout this document, 18.75 ms means the optimised planning tick — the one that produces the scored ADE@2s.** Naming it "decision tick" would create a fifth collision in a family that has already produced one retraction.

---

## 2. The real memory + compute budget — MEASURED, per component

Method: `budget_probe.py` instantiates the **exact deployed architecture** (`flagship4b_config()` + `adapt_config_action_dim(cfg, 3)`) and measures params, FLOPs (`torch.utils.flop_counter.FlopCounterMode`, MHA fast path disabled during the count — the same convention as `taniteval.efficiency._flops`), and peak allocator bytes at the deployed batch/window. Weights are random-init: **all four quantities are weight-independent architecture reads**, and the probe proves it by reproducing the registry's param count **exactly** and the registry's uncached-tick GFLOPs to 4 decimal places.

**The tick being budgeted** = the deployed L4-composed planning tick: *encode ONE new 9-channel frame (7 window states cached) → 20 sequential predictor steps → 20 step-readout decodes → SE(2) accumulate.*

### 2.1 Parameters — and a registry label correction

| component | params `MEASURED` | on the deployed scored path? |
|---|---:|:--|
| encoder (ViT d768 × 12, 9-ch, 256 px, patch 16 → 16×16) | **87,022,848** | ✅ |
| readout (`SpatialGridReadout`, 4×4 grid × 128 → state 2048) | **98,432** | ✅ |
| operative predictor (d768 × 10, window 8, horizons 1/2/4, action_dim 3) | **91,361,280** | ✅ |
| `grounding.step['op']` (the step readout) | **2,107,395** | ✅ |
| **deployed operative path total** | **180,589,955** | **68.6 % of the model** |
| tactical_pred | 26,535,424 | ✗ (hierarchy) |
| tactical_policy | 22,736,141 | ✗ (hierarchy) |
| strategic_policy | 8,385,027 | ✗ (hierarchy) |
| imagination (H15) | 22,055,683 | ✗ (training) |
| inv_dyn | 5,248,003 | ✗ (training) |
| grounding tac/str heads | 8,954,892 | ✗ (training) |
| **model total** | **263,442,838** | = registry exactly ✅ |

> ⚠️ **Registry component labels bundle a sibling module. The totals are correct; two labels are not.**
> `MEASURED`: registry "encoder 87,121,280" = encoder (87,022,848) **+ readout** (98,432). Registry "operative 96,609,283" = predictor (91,361,280) **+ inv_dyn** (5,248,003). Both differences are exact to the parameter.
> **This matters for deployment**, because `inv_dyn` is a *training-time* grounding probe that never runs in the rollout. Using the registry's labelled 96,609,283 for the bandwidth model **over-states the rollout's DRAM traffic by 5.7 %**. (`DEPLOYMENT_PLAN` §4.1 already used the correct 91.4 M; its §1 table quotes the 96.6 M label. Both appear in one document.)

### 2.2 Memory — parameter memory dominates activation memory by 27×–900×

| | fp32 | fp16 | int8/fp8 | nvfp4 |
|---|---:|---:|---:|---:|
| **deployed operative path** (180.59 M) | 722.4 MB | **361.2 MB** | 180.6 MB | 90.3 MB |
| full model (263.44 M, all brains) | 1053.8 MB | 526.9 MB | 263.4 MB | 131.7 MB |

**Peak activation at the deployed batch/window (`MEASURED`, batch 1):**

| stage | activation |
|---|---:|
| encoder, 1 frame → state | **13.37 MB** |
| encoder, 8-frame window (the *uncached* path) | 76.55 MB |
| **the entire 20-step rollout_decode** | **0.404 MB** |
| predictor, 1 call | 0.320 MB |
| strategic + tactical hierarchy, 1 call | 0.336 MB |

Cross-check: the A40 composed lever measured **9.2 MB** activation for the whole fp16 tick — consistent with 13.37 MB at fp32 halved, plus bookkeeping. Independent, two hosts.

**What dominates memory:** **parameters, overwhelmingly.** At fp16 the deployed path is 361.2 MB of weights against **13.4 MB** of peak activations — **27×**; for the rollout stage alone it is 3,654 MB of weight *traffic* against 0.404 MB of activations — **~9,000×**. **Capacity is a non-issue**: 361.2 MB is **0.56 %** of Orin's 64 GB and 0.28 % of Thor's 128 GB. *Nobody should spend an hour on activation memory or on model-capacity fit.*

### 2.3 Compute per tick — and the correction the program needs

| stage | GFLOP `MEASURED` | share of deployed tick |
|---|---:|---:|
| encoder, **1** new frame (deployed, cached) | **46.81** | **63.1 %** |
| 20 × operative predictor | **27.34** | **36.8 %** |
| 20 × step readout | 0.084 | 0.1 % |
| **deployed tick total** | **74.24** | 100 % |
| *(uncached 8-frame encode variant)* | *401.92* | *reproduces registry 401.922* ✅ |
| strategic + tactical hierarchy, if run | 1.85 | +2.5 % |
| imagination (H15), if run | 58.69 | *(off the deployed path)* |

> 🔴 **Correction to a claim in circulation.** "The encoder is **94 %** of tick FLOPs" is true **only of the uncached 8-frame tick**. Under the deployed L2 encoder cache — which is *in* the composed 18.75 ms lever — the encoder is **63.1 %** and the tick is **74.24 GFLOP, not 401.9**. Quoting 401.9 GFLOP or "94 % encoder" for the deployed configuration over-states the workload by **5.4×** and points every optimization at the wrong component.

### 2.4 DRAM weight traffic per tick — the currency that actually binds

At batch 1 the working set (≥175 MB fp16) is far larger than Orin's **4 MB** L2, so every call re-streams its weights from DRAM.

| | fp32 | **fp16** | int8/fp8 | nvfp4 |
|---|---:|---:|---:|---:|
| encoder (×1, cached) | 348.5 MB | **174.2 MB** | 87.1 MB | 43.6 MB |
| predictor (×20) | 7,308.9 MB | **3,654.5 MB** | 1,827.2 MB | 913.6 MB |
| step readout (×20) | 168.6 MB | 84.3 MB | 42.2 MB | 21.1 MB |
| **tick total** | **7,826.0 MB** | **3,913.0 MB** | 1,956.5 MB | 978.3 MB |
| *of which: the 2 unused horizon heads* | *252.0 MB* | *126.0 MB* | *63.0 MB* | *31.5 MB* |

**Share of tick DRAM bytes (fp16): encoder 4.5 % · rollout 95.5 % · the predictor alone 93.4 %.**

### 2.5 ⭐ What actually dominates — the inversion, and the roofline placement

| component | share of tick **FLOPs** | share of tick **bytes** | arithmetic intensity (fp16) |
|---|---:|---:|---:|
| encoder (1 frame) | **63.1 %** | 4.5 % | **268.7 FLOP/byte** |
| rollout (20 × predictor + readout) | 36.9 % | **95.5 %** | **7.34 FLOP/byte** |
| — predictor, per single step | — | — | **7.48 FLOP/byte** |
| whole deployed tick | 100 % | 100 % | 18.97 FLOP/byte |

Machine balance (peak dense FP16 ÷ peak bandwidth), `ESTIMATED` from `PUBLISHED` specs: **Orin 207.5 FLOP/byte · Thor 947.8 FLOP/byte.**

> **The two halves of the tick sit on opposite sides of Orin's roofline and need opposite optimizations.**
> - The **encoder** at 268.7 FLOP/byte is (just) **compute-bound on Orin** and **memory-bound on Thor**.
> - The **rollout** at 7.34 FLOP/byte is **28× below Orin's balance and 129× below Thor's** — it is a pure bandwidth problem on every device we might ship on, by a margin no kernel work can close.
>
> **The single dominant term in the entire deployment is the operative predictor's weight streaming: 91,361,280 params × 2 bytes × 20 sequential steps = 3.65 GB per tick, 93.4 % of all DRAM traffic.** Anything that does not reduce *that number of bytes* is, to first order, worth zero. This is the finding the program was missing, and it reverses the intuitive ranking of half the candidate levers (§4).

**The hierarchy is nearly free** — a directly relevant corollary. Running strategic + tactical every tick costs **1.85 GFLOP (2.5 %)** and **115.3 MB (2.9 %)** of tick bytes; at their configured cadences (tactical every 5 ticks, strategic every 20) the amortized cost is **≈ 20.6 MB/tick = 0.5 %**. See §4 item 11 and the escalation in §7.2.

---

## 3. Target envelopes and gap analysis

### 3.1 The devices — `PUBLISHED` specs, with the derivation chain stated

| | **A40** (our only proxy) | **Jetson AGX Orin 64 GB** | **Jetson AGX Thor T5000** |
|---|---|---|---|
| GPU arch | Ampere GA102, SM 8.6 | Ampere GA10B, **SM 8.7** | **Blackwell** + Transformer Engine |
| headline AI perf | — | **275 TOPS INT8 (sparse)**, whole module `PUB` | **2070 TFLOPS FP4 (sparse)** `PUB` |
| GPU-only tensor peak | 149.7 TFLOPS FP16 `PUB` | **170 sparse INT8 TOPS** `PUB` | 2070 sparse FP4 TFLOPS `PUB` |
| → dense FP16 | 149.7 | **42.5** `EST` (÷2 sparse, ÷2 int8→fp16) | **258.75** `EST` (÷2, ÷2, ÷2) |
| → dense INT8 | 299.3 | **85** `EST` | 517.5 `EST` |
| → dense FP8 | ❌ none | ❌ **no FP8 datapath on Ampere** | **517.5** `EST` |
| → dense NVFP4 | ❌ | ❌ | **1035** `EST` (÷2 sparse) |
| memory | 48 GB GDDR6 | **64 GB, 256-bit LPDDR5** `PUB` | **128 GB, 256-bit LPDDR5X** `PUB` |
| **bandwidth** | 696 GB/s `PUB` | **204.8 GB/s** `PUB` | **273 GB/s** `PUB` |
| L2 | 6 MB | **4 MB** | not published |
| power | 300 W TDP | **15–60 W** `PUB` | **40–130 W** `PUB` |
| DLA | — | 2× NVDLA v2.0 (~105 sparse TOPS) — **cannot run our attention** 🔴 | **none** |

Sources: [Jetson AGX Orin product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) · [Jetson Thor product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) · [Jetson modules overview](https://developer.nvidia.com/embedded/jetson-modules) · [Jetson AGX Orin Series Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf). Everything marked `EST` is arithmetic on a `PUBLISHED` figure with the operation named — never a second vendor claim.

### 3.2 Calibrating the bracket — what the A40 actually achieved

The A40's `MEASURED` composed tick (18.7476 ms) moves **3,913 MB** of fp16 weights ⇒ an achieved **209 GB/s**, which is **30.3 %** of its own stage-wise roofline. That is *not* a DRAM-efficiency measurement: at 696 GB/s this workload never becomes bandwidth-limited, which is exactly the registry's independent diagnosis ("achieved 3.7–4.3 TFLOPs ⇒ **launch/serialisation-bound**"). **Consequence: 18.75 ms cannot be scaled to Jetson by a bandwidth ratio.** It is a measurement of a different bottleneck.

So every device tick below is a **bracket**: the lower end is the stage-wise roofline floor (unreachable by construction — 100 % of peak); the upper end applies the A40's measured 30.3 % utilisation as a deliberately pessimistic ceiling. Both ends `ESTIMATED`.

### 3.3 Gap analysis — does it fit, at what precision, at what rate?

| device / precision | encoder stage binds on | rollout stage binds on | **tick bracket** | **Hz bracket** | 10 Hz across the *whole* bracket? |
|---|---|---|---:|---:|:--:|
| **Orin FP16** | compute (1.10 ms) | **bandwidth (18.26 ms)** | **19.4 – 63.8 ms** | **15.7 – 51.7** | ✅ **yes** |
| Orin INT8 🔴 refuted | compute | bandwidth (9.13) | 9.7 – 31.9 ms | 31.3 – 103.3 | ✅ but **INT8 W+A fails the accuracy gate — not admissible** |
| Orin FP8 / NVFP4 | — | — | — | — | ❌ **no datapath on Ampere** |
| **Thor FP16** | **bandwidth (0.64 ms)** | **bandwidth (13.70 ms)** | **14.3 – 47.3 ms** | **21.2 – 69.8** | ✅ **yes** |
| Thor FP8 | bandwidth | bandwidth (6.85) | 7.2 – 23.6 ms | 42.3 – 139.5 | ✅ (accuracy ungated) |
| Thor NVFP4 | bandwidth | bandwidth (3.42) | **3.6 – 11.8 ms** | 84.6 – 279.1 | ✅ (accuracy ungated, and **predicted to fail** — §5) |

**Verdict — it fits, and the risk is headroom, not feasibility.**

1. **The model fits both chips with enormous margin on capacity** (0.56 % of Orin's memory at fp16) and **clears 10 Hz at FP16 across the entire bracket on both.** The feasibility question is closed as far as arithmetic can close it. What is genuinely open is *headroom*: **1.6× to 5.2×** on Orin. Only silicon narrows that.
2. **Thor's 7.5×-compute headline buys 1.33× at FP16** — exactly the bandwidth ratio (273 / 204.8), because at FP16 the tick is **50× bandwidth-bound on Thor** (vs 10.9× on Orin). Buying Thor and running FP16 is buying compute we cannot use. **Thor pays for itself only if its precision advantage is spent on *bytes*: FP8 (2×) or NVFP4 (4×).** That is a single, testable question — §5.
3. **On Orin there is no low-risk middle.** FP16 is the floor: no FP8, no FP4, and INT8 is refuted on accuracy (`MEASURED`, D6). **Orin's deployment precision is FP16, full stop**, and its tick floor of 19.4 ms is a hard physical bound no kernel work crosses.
4. **Power, `ESTIMATED`:** at 10 Hz the tick sustains **39.1 GB/s** of DRAM traffic (19 % of Orin's peak) ⇒ **≈ 2.19 W** of DRAM energy alone at 7 pJ/bit (published LPDDR5 range ≈ 5–8 pJ/bit; midpoint used) — meaningful against Orin's **15–60 W** module envelope, negligible against Thor's 40–130 W. NVFP4 would cut it to 0.55 W. **Not measured; the assumption is stated so it can be falsified.**

---

## 4. Ranked optimization list

Ranked by **effect on the binding term** (DRAM bytes for the rollout, FLOPs for the encoder) × confidence ÷ effort. "Δ tick floor" is the lever applied **alone** to the Orin FP16 stage-wise floor of **19.36 ms** (`projection.py`, `ESTIMATED`). Lever *ordering* on real silicon must be re-measured — on the A40 capture comes first; on Orin precision comes first.

| # | Lever | Δ Orin fp16 tick floor | Memory saved | Accuracy risk | Effort | Pre-registered check |
|---|---|---:|---|---|---|---|
| **1** | **Strided rollout on the already-trained k=2 head** — 10 predictor calls instead of 20 for the same 2 s | **−47.2 %** (19.36 → **10.23 ms**) | −1,827 MB/tick traffic | **Unknown — the gap.** The `step_readout` is calibrated on 0.1 s transitions; a k=2 roll decodes 0.2 s transitions | **Low–med** (no retrain; readout recalibration is the work) | Gate B on the canonical 40 val episodes: ADE@0.5/1/1.5/2 s, **paired episode-cluster bootstrap** (`taniteval/ci.py`, never `overlapping_holdout_se`). **Falsifier:** paired CI on ΔADE@2s excludes 0, **or** any point >0.02 m, **or** degradation ratio grows with horizon. Latency already `MEASURED` (graphed 57.53 → 42.88 ms) |
| **2** | **FP16 end-to-end + fp32 SE(2) accumulate** | −50 % vs fp32 (rollout bytes 7,477 → 3,739 MB) | **halves all weight memory** | **Low, `MEASURED`**: ADE Δ **+0.000076 m**; waypoint deviation 0.0241 → **0.0127 m** with fp32 accumulate | **Low** — already built as an A40/A6000 engine | Already gated. ⚠️ Record: the registry's fp16 "max abs dev 0.024 m" is a **waypoint deviation, not an ADE delta** — the ADE delta is 7.6e-5 m. Do not read 0.024 against the 0.02 m falsifier |
| **3** | **FP8-E4M3 weight-only on the predictor blocks — Thor only** | Thor 14.33 → **7.17 ms** (−50 %) | −50 % weight bytes | **Medium — the best-evidenced remaining precision lever.** `MEASURED` this session: FP8-E4M3 per-channel weight rel-L2 is **3.44×** INT8's on Gaussian but **0.99–1.25×** on heavy-tailed/outlier weights. INT8 weight-only `MEASURED` +0.0065 m = a **pass with 3× margin** | Med (accuracy testable **now**; latency needs Thor) | §5. Predicted ΔADE@2s ∈ **[+0.006, +0.022] m**. Same falsifier as #1. Keep the encoder `readout_head` **out** of any activation quantization (`MEASURED` collapse, cos 0.566) |
| **4** | **NVFP4 weight-only on the predictor — Thor only** | Thor 14.33 → **3.58 ms** (−75 %) — **the single largest lever in the program** | −75 % weight bytes | **High, and predicted to FAIL.** `MEASURED`: NVFP4 (block-16, E4M3 scale) weight rel-L2 is **12.4×** INT8's on Gaussian, **3.9–4.6×** on heavy-tailed. Linear propagation from the +0.0065 m INT8 anchor ⇒ **+0.025 to +0.081 m** | Med to test, high to ship | §5. **Falsifier for the pessimistic prediction:** if NVFP4 lands **under +0.02 m with a horizon-flat profile**, the propagation model is wrong, Thor's 4× lever is live, and **the chip choice changes** |
| **5** | **Structured pruning of the predictor** | width d768→d512 **−49.3 %**; →d384 −67.0 %; depth 10→6 **−33.4 %** | proportional | **High** — a different model; every accuracy claim re-opens | **High** (GPU-days: retrain + full re-gate) | Only if 1–4 miss the target. Full gate protocol (`GATE_PROTOCOL.md`), not a spot check |
| **6** | **L7 — drop the 2 unused horizon heads** | −3.2 % (126 of 3,913 MB) | −126 MB/tick | **None** — `MEASURED` exact, 0.0 m deviation | **~zero** — already inside the composed L4 lever | Already gated. Just make sure the shipped graph exports **one** head |
| **7** | **Encoder crop change (either direction)** | narrowing: **−1.3 % max**; widening to the full front field: **+0 % to +8.6 %** | 0 MB — **crop moves DRAM weight traffic by exactly zero** | n/a for latency | Low | **See the interaction note below** |
| **8** | **CUDA-graph capture** | proven; **do not re-litigate** | — | none (exact) | done | On Orin re-measure the *ordering* (precision first). Single-step capture replayed loses **7.7 µs/step** vs whole-rollout — use it if the runtime cannot capture a loop |
| **9** | 🔴 **Operator fusion beyond TRT's** | **0.0 %** | — | — | — | **REFUTED as a latency lever.** The rollout's entire activation footprint is **0.404 MB against 3,739 MB of weight traffic (0.011 %)**. Fusion cannot move the binding term. TRT already fuses our MHA (`MEASURED`, two hosts) |
| **10** | 🔴 **KV / feature caching across the temporal window** | **0.0 %** | — | — | — | **REFUTED twice over — see below** |
| **11** | **Batch / window restructuring** | window 8→4 cuts per-step FLOPs 2× ⇒ **0.0 %** of the tick. **But the fan is a capacity win** | — | — | — | **See the fan note below** |
| **12** | 🔴 **INT8 weight+activation** | — | — | — | — | **REFUTED on both gates** (`MEASURED`, D6). Do not propose again without a new mechanism |

**⚠️ Item 7 — the crop interaction, quantified.** The ViT trunk's parameters are **token-count independent**, so a crop change is a pure FLOP/activation lever and moves weight traffic by zero. `MEASURED` FLOP(N) = **0.173408·N + 3.6864e-5·N²**, R² **1.000**, n = 7, fit window **64–1024 tokens** (all scenarios below are *inside* the window — no extrapolation); the quadratic (attention) term is only **5.2 %** of encoder FLOPs at the deployed N = 256, so cost is near-linear in token count.

| crop scenario | tokens | encoder GFLOP | Δ Orin fp16 tick floor |
|---|---:|---:|---:|
| **deployed** (256 px, 16×16) | 256 | 46.81 | — |
| halve the crop, same angular resolution | 128 | 22.79 | **−1.3 %** |
| 0.75× linear downscale | 144 | 25.73 | −1.3 % |
| widen to the full front field, **token-budget-matched** (rescale to 256) | 256 | 46.81 | **0.0 %** |
| widen to the full front field at 0.75× angular resolution | 336 | 62.43 | +1.9 % |
| widen to the full front field at **same** angular resolution | 600 | 117.32 | **+8.6 %** |

> **The crop is not a deployment lever in either direction, and the widening under consideration elsewhere should not be slowed on latency grounds.** Narrowing saves **at most 0.25 ms (1.3 %)** — a *hard* bound, because below ~128 tokens the encoder hits its own weight-streaming floor (0.851 ms) and further FLOP cuts buy literally nothing. Widening to the whole front field costs **at most +8.6 %** of the tick floor, and **nothing at all** if the token budget is held constant (i.e. widen the field and rescale to 256 px — same cost, wider view, lower angular resolution).
> ⚠️ **Two live values for the retained FOV, and they disagree.** The H2 stream (2026-07-25) uses **51.4°** of a 120.5° camera; `Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md` marks failure mode 2 **CONFIRMED** and says the real f-theta focal makes the retained field **33.1°**, not 51.4°. Deployment-neutral (cost depends on tokens, not degrees) but it changes the widening ratio from 2.34× to **3.67×** linear ⇒ ~939 tokens ⇒ ~**+18 %** of the tick floor — still cheap. **Escalated (§7.2); the two streams must reconcile.**

**⚠️ Item 10 — why KV caching is dead, twice.**
1. **Mechanically invalid as written.** `OperativePredictor.forward` (`stack/tanitad/models/predictor.py`) computes `x = self.in_proj(states) + self.pos[:, :w]` — an **absolute** window position embedding — and the deploy rollout **slides** the window every step (`rollout_decode`, `metric_dynamics.py`: `win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)`). Every retained state's positional index shifts by one each step, so its K/V change. A cache requires a relative/rotary rewrite — **i.e. a retrain**.
2. **Worthless even if free.** The ceiling is `MEASURED`: perfect incremental decode drops the predictor from **1.3672 → 0.1789 GFLOP/step (7.64×)** — and removes **zero bytes**. Applied to Orin FP16: **0.00 % of the tick floor.** It is a large win in the currency that does not bind.

**⭐ Item 11 — the fan is nearly-free capacity we already pay for.** `MEASURED`: predictor arithmetic intensity rises **linearly in K** — 7.5 (K=1) → 30.0 (K=4) → 59.9 (K=8) → **238.9 FLOP/byte (K=32)** — because the *same* 182.3 MB of fp16 weights serves every candidate. This is the mechanism behind the already-`MEASURED` "marginal candidate ≈ 0.3 ms" (K=8 fan 20.82 ms p50) and it explains it from first principles rather than as a surprise. **On a bandwidth-bound rollout, imagine-and-select is close to free — an argument to *use* the three-planner hierarchy, not to fear it.** Note the corollary: only at **K ≈ 28** does the rollout finally reach Orin's machine balance, i.e. a K=32 fan is roughly where the hardware would first be used efficiently.

---

## 5. The cheapest discriminating experiment — no Orin, no Thor required

> **Run the precision-format Gate B on the real checkpoint for FP8-E4M3 and NVFP4 weight-only, and settle whether Thor is worth buying.**

**Why this one, above everything else.** §3.3 established that Thor's entire advantage over Orin at FP16 is **1.33×** — the bandwidth ratio — and that the *only* way to convert its 7.5× compute headline into tick savings is **weight compression** (FP8 = 2×, NVFP4 = 4× on the binding term). That is simultaneously (a) the largest remaining lever in the program, (b) the input to a **hardware purchasing decision**, and (c) a question whose *accuracy* half is entirely determined by architecture and weights — **not by silicon**. We can answer the half that decides the direction, today, for a couple of GPU-hours.

**Protocol.** Extend `2026-07-23-orin-int8-benchmark/bench_p1_accuracy.py`'s `_fakequant_` with two formats (implemented and unit-checked in this session's `lever_probe.py`: `fq_fp8_per_channel`, `fq_block_fp4`):
- **FP8-E4M3, per-output-channel scale**, weights only, predictor blocks;
- **NVFP4**: E2M1 elements, **block-16** with an **E4M3 block scale**, weights only, predictor blocks;
- carry **INT8 weight-only** as the positive control (it must reproduce **+0.0065 m** or the harness is wrong — a built-in falsifier for the instrument itself);
- keep the encoder `readout_head`, all activations, and the SE(2) accumulate at FP16/FP32 (`MEASURED` hazard, cos 0.566).

Score the **20-step rollout on the canonical 40 val episodes** (`physicalai-val-0c5f7dac3b11`) — not the train-cache proxy the 07-23 run had to use — reporting ADE@0.5/1/1.5/2 s and the **paired episode-cluster bootstrap** (`taniteval/ci.py`). **Never `overlapping_holdout_se`.**

**Cost:** ~1–2 GPU-hours on the **free eval pod**, or on pod1's existing `/workspace/int8_bench/` tree, which already holds the real ckpt, the venv and the harness. No training pod is touched.

**Pre-registered predictions — both outcomes committed in advance:**

| format | weight rel-L2 vs INT8 (`MEASURED`, this session) | **predicted ΔADE@2s** | adoption falsifier | what a surprise would mean |
|---|---:|---|---|---|
| INT8 w-only (control) | 1.00× | +0.0065 m (must reproduce) | — | harness is broken |
| **FP8-E4M3 w-only** | 0.99–3.44× | **+0.006 → +0.022 m** — *straddles the line, genuinely uncertain* | paired CI on ΔADE@2s excludes 0 **and** Δ > 0.02 m; **or** the degradation ratio grows with horizon | a clean pass makes FP8 the Thor deployment precision |
| **NVFP4 w-only** | 3.85–12.4× | **+0.025 → +0.081 m** ⇒ **predicted to FAIL** | same | **a pass falsifies my linear-propagation model, unlocks a 4× lever, and changes the chip choice** |

**Runner-up, if a second experiment is affordable:** the **strided k=2 Gate B** (§4 item 1) — the largest no-new-hardware, no-retrain lever we own (−47 %), and the only one whose accuracy is *explicitly* recorded as unmeasured in its own artifact.

---

## 6. What needs hardware, and what does not

**Genuinely blocked on Orin/Thor silicon — escalate, do not fabricate:**
1. **Any real tick.** The §3.3 bracket is **3.3× wide** (19.4–63.8 ms on Orin). Only silicon closes it. The specific unknown is **achieved DRAM utilisation** — one measurement collapses the whole bracket.
2. **The shippable engine.** TRT engines are not portable across GPU architectures (`PUBLISHED`); the build + INT8/FP8 calibration must happen on the target, and calibration was already `MEASURED` at 4–5× the FP16 build time on a *fast* x86 host — expect far worse on Orin's ARM cores.
3. **FP8 / NVFP4 *latency*** and the Transformer Engine's dynamic FP4↔FP8 switching (Blackwell only).
4. **Kernel selection and MHA-fusion tactics on SM 8.7 / Blackwell** — fusion is arch-specific; we have verified it only on SM 8.6, twice.
5. **`nvpmodel` power mode × sustained thermal clock**, and the real power draw against the ≈2.19 W DRAM estimate. *A Jetson latency without its power mode is as unquotable as a tick without its definition.*
6. **Jetson launch latency** — settles precision-first vs capture-first ordering on-device.

**Does NOT need hardware — and most of it is not done:**
1. ✅ **Every accuracy gate.** FP8, NVFP4, strided-k2, any pruning — accuracy is a function of architecture and weights, not silicon. *This is where all remaining cheap value is.*
2. ✅ The complete **memory/compute/roofline budget** — done here.
3. ✅ **ONNX export + TRT build path + MHA fusion existence** — done, twice, one with real weights.
4. ✅ The **encoder `readout_head` activation hazard** — found on real weights; it will bite any future activation-quantization attempt on any device.
5. ✅ **v4's deploy delta** — sizeable now: its operative predictor is v1 verbatim (96.6 M by the registry's label = **91.36 M predictor + 5.25 M inv_dyn**), so §2.4's traffic table applies unchanged; the new term is the ~9.78 M anchored-diffusion planner, worth **≈ 39 MB/tick (1.0 % of tick bytes)** at 2 denoise passes. ⚠️ Its **FLOPs have never been counted** (the 256-anchor pass is batched and likely compute-bound, so the bytes view understates it) — a 10-minute probe once v4 lands.

---

## 7. Escalations

### 7.1 🔴 A RETRACTED claim is still live in the master lever ladder — flagged 4 days ago **in a document**, and still there

`TanitAD Research Hub/Production & Optimization/FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md` **§5.1, line 216** still reads:

> *"CEM planning is infeasible on this rollout … 8 candidates × 20 steps = **723 ms/tick — 7× over the 10 Hz budget**"* — tagged **`M-OURS`** (i.e. measured).

It is **RETRACTED** (`RETRACTION_LOG.md` 07-21, class **C3**: "mechanism instead of measurement"; the fan was `MEASURED` at **20.82 ms** at K=8). The companion doc it was flagged alongside (`Benchmarks & Eval/Research/2026-07-20-inference-efficiency-v1-vs-refc.md`) **has been fixed**; this one has not — and it is the *live master ladder*, carrying a `MEASURED` tag that makes it maximally quotable.

Two things make this worth raising rather than filing:
- **The mechanism is backwards, not merely the number.** §5.1 says the fan is infeasible because cost scales as `n_candidates × horizon × per_step`. `MEASURED` this session: on a bandwidth-bound rollout, **arithmetic intensity rises linearly in K** (7.5 → 238.9 FLOP/byte, K=1→32) because the same 182.3 MB weight read serves every candidate. The fan is not expensive-and-we-mismeasured-it; it is **structurally almost free**, and only at K≈28 does the hardware even become efficiently used.
- **This is the same failure shape CLAUDE.md warns about twice** — an unmeasured cost argument used to scope the hierarchy down (cf. the confounded "strategic choice is a ~2 % lever"), and a fix that was written *into a doc* rather than escalated (cf. the orthogonality instrument, unmerged 10 days). **The 2026-07-22 intake did exactly the documented-not-escalated thing, and 4 days later the claim is still live.** Raising it in-channel instead. Suggested replacement text is in §2.5 + §4 item 11 of this document; I have not edited another stream's live doc.

### 7.2 🟡 The encoder's retained FOV has two live values, 4 days apart

**51.4°** (`2026-07-25-h2-e0-e1/PRE_REGISTRATION.md`, from `calib.py F_REF=266`) vs **33.1°** (`Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md`, failure mode 2 marked **CONFIRMED**, real f-theta focal 925.9 px vs the assumed 554.3 px). Deployment-neutral — encoder cost depends on token count, not degrees — but it moves H2's premise from "we discard **57 %** of the front field" to "**73 %**", and the full-field widening cost from +8.6 % to ~+18 % of the Orin tick floor (still cheap either way). The two streams should reconcile before H2's E0 partition is read.

### 7.3 🟡 Registry component labels bundle sibling modules (totals are correct)

`MEASURED` exactly: registry "encoder 87,121,280" = encoder + **readout**; "operative 96,609,283" = predictor + **inv_dyn**. The bandwidth-relevant predictor figure is **91,361,280**; using the label over-states rollout DRAM traffic by 5.7 %. `DEPLOYMENT_PLAN` already uses both numbers in one document (§1 table vs §4.1). Suggest the registry split the two labels.

### 7.4 🟡 Naming — 18.75 ms is the *planning* tick, not the "decision tick"

See §1.4. The decision-tick family already produced one retraction (11.16 ms) and one traced mis-attribution (14.331 ms, `base250cam` random-init on an RTX 4060). One careless sentence creates a fifth. Also: `DEPLOYMENT_PLAN` §3.2's `ESTIMATED` "**≈14 ms/tick**" TRT projection is a rounding away from colliding with the retracted 14.331 ms — recommend it never be quoted without its "ESTIMATED, A40, not a measured 20-step tick" qualifier.

---

## Deliverable manifest

| # | Artifact | Where it lives | Only copy? |
|---|---|---|---|
| 1 | `ORIN_THOR_STATE_AND_PLAN.md` — this document | `repo: TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-orin-thor-optimization/` | no — derivable from the three JSONs below |
| 2 | `budget_probe.py` — per-component params / FLOPs / activation / traffic probe | same folder | yes |
| 3 | `lever_probe.py` — encoder FLOP(N) sweep, KV-cache ceiling, fan intensity, quantization-format numerics, roofline | same folder | yes |
| 4 | `projection.py` — A40-calibrated Orin/Thor brackets + per-lever tick deltas | same folder | yes |
| 5 | `artifacts/budget_report.json` — **MEASURED** budget (reproduces registry params exactly and the 401.922 GFLOP uncached tick to 4 dp) | same folder | yes |
| 6 | `artifacts/lever_report.json` — **MEASURED** lever quantification + PUBLISHED device table | same folder | yes |
| 7 | `artifacts/projection_report.json` — **ESTIMATED** brackets and lever deltas | same folder | yes |

**Nothing is staged or committed** (per brief; the orchestrator stages). Nothing needs merging into `stack/` — the probes are self-contained and read the live `tanitad` package. **No pod was touched: every measurement in this document ran on the dev box (CPU/local GPU) and emits only device-independent quantities — no latency was measured or reported from it.**

**Cross-refs:** `2026-07-22-orin-thor-deployment/{DEPLOYMENT_PLAN,BENCHMARK_PLAN}.md` · `2026-07-23-orin-int8-benchmark/DEPLOYMENT_READINESS_INT8.md` · `Project Steering/MODEL_REGISTRY.md` §1.2 · `taniteval/results/eff_levers_flagship-30k.json` · `Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md` · `Project Steering/RETRACTION_LOG.md` (07-21, C3).
