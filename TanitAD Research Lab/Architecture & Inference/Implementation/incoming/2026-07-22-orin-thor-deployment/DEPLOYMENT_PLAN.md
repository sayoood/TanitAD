# Flagship on Jetson Orin & Thor — the staged deployment plan

- **Date:** 2026-07-22 · **Owner:** deploy-prep stream (this intake) · **Model:** `flagship4b-speedjerk-30k` (flagship v1, deployed) + v4 delta (§7)
- **Status:** research + export-recipe + benchmark-plan. The Orin/Thor chips are **not on hand**; every number is tagged **measurable-now** or **hardware-blocked**.
- **Evidence class on every number** — `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another doc, not re-verified) · `ESTIMATED` · `HYPOTHESIS`. A GPU-day decision may rest only on MEASURED/PUBLISHED (CLAUDE.md §Operating standard).

> **This document does NOT restart the inference analysis — it consolidates and extends three prior notes that are already decision-grade, and adds the two things they explicitly left open: (a) a static-shape ONNX export of the *exact deployed architecture*, and (b) a TensorRT-FP16 engine build on an A40 proxy.** Read those three first; this plan cites them rather than repeating them:
> - `Production & Optimization/Research/2026-07-20-orin-thor-deployment-and-inference-levers.md` — the 769-line desk analysis (hardware reality, bandwidth floor, quantization risk, toolchain). **Source of the per-chip facts in §2.**
> - `Production & Optimization/Research/2026-07-20-flagship-v1-inference-levers-measured.md` — the A40 measurement campaign (CUDA-graph capture, composed tick, CEM fan). **Source of the tick numbers in §4.**
> - `Production & Optimization/FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md` — the live master lever ladder.

---

## 0. TL;DR — the deployment plan in eight lines

1. **The deploy target is the planning tick**: `encode(1 new 9-ch frame) → slide 8-state window → 20 sequential operative-predictor steps → per-step Δpose → SE(2) accumulate`. It, not the "decision tick", produces the ADE the leaderboard scores.
2. **On the A40 the eager tick misses 10 Hz (100.29 / 113.98 ms p50/p99, MEASURED); the composed-lever tick meets it with 5.3× headroom (18.75 ms, MEASURED).** The decisive lever is **CUDA-graph capture of the rollout** (bit-identical, 0.0 m deviation).
3. **On Jetson the lever ORDER inverts: precision FIRST (a TensorRT FP16/INT8 engine), CUDA-graph SECOND** — because a naive fp32 port is arithmetic-starved on Orin (75.6 ms floor at 100 % of a 5.32-TFLOPS peak), not launch-bound.
4. **Per-chip precision map (§2, PUBLISHED):** Orin (Ampere, SM 8.7) → **FP16 baseline, INT8 only behind a per-layer benchmark — NO FP8, NO FP4**. Thor (Blackwell) → **FP16/FP8, and NVFP4 for the 4× memory win (Thor-only)**.
5. ⚠️ **The INT8 trap that makes this a plan and not a switch:** ViT INT8 on Orin can run **~2.7× SLOWER than FP16** on non-optimal kernels. The recipe therefore *mandates a per-layer FP16-vs-INT8 benchmark before any INT8 commit* (§2.4). INT8 is a hypothesis until measured, never a default.
6. **MEASURED this intake (§3, A40 proxy):** the exact deployed architecture **exports to static-shape ONNX cleanly** (encoder + predictor, parity ≤1.9e-6), **builds to a TensorRT-FP16 engine** (enc 1.205 ms / pred 0.666 ms per call), and **TRT fuses our MHA** — the NVIDIA #4537 "ViT MHA won't fuse" risk does **not** bite us on SM 8.6.
7. **Hardware-blocked, honestly:** real Orin/Thor throughput, the on-device engine build (engines are **not portable across GPU architectures**, PUBLISHED), and any NVFP4 number (needs Blackwell silicon). The A40 is a *proxy*, not Orin (§6).
8. **v4 changes almost nothing for deployment (§7):** its operative predictor is **v1 verbatim** (same 20-step rollout, same levers, same precision map); it adds a ~9.8 M anchored-diffusion planner whose cost is the **denoise-step count** (~2.9 ms/pass on A40, MEASURED for the identical REF-C decoder).

---

## 1. What we are deploying — the exact object

**Deployed v1 = `flagship4b-speedjerk-30k`, step 29,999** (MODEL_REGISTRY §1.2). Operative (intent-free) path only — the tactical/strategic brains are off the scored path.

| component | shape / config | params | source |
|---|---|---|---|
| encoder (ViT) | in_ch 9, 256 px, patch 16 → 16×16, d768 × depth 12, 12 heads | 87,121,280 | REGISTRY §1.2 ✅ |
| operative predictor | d768 × depth 10, 12 heads, window 8, horizons (1,2,4), **action_dim 3** (speed channel) | 96,609,283 | REGISTRY §1.2 ✅ |
| step readout (grounding) | latent-pair → metric Δpose, calibrated on 0.1 s transitions | 2.11 M | 07-20 efficiency note |

**The static deploy shapes (for a static-shape engine):**
- **Encoder graph:** `frames [1, 9, 256, 256] → state [1, 2048]` — the *per-frame* encoder. This is the cache-friendly deploy shape: at 10 Hz, 7 of the 8 window frames were already encoded last tick, so deployment encodes **one** new 9-channel frame and reuses 7 cached states (the L2 encoder-cache lever).
- **Predictor graph:** `states [1, 8, 2048], actions [1, 8, 3] → (z_h1, z_h2, z_h4)`. Trip count is a **fixed 20** — capture as one CUDA graph, or unroll into one engine; **never** an ONNX `Loop` node (PUBLISHED: TRT loop-carried shapes must be static; §5.2 of the desk note).

**The speed channel is a hard contract:** `v0 = poses[t,3] / 10.0`, `SPEED_SCALE = 10.0`, appended as the 3rd action channel (REGISTRY §1.2). `action_dim=3` at instantiation gives operative 96,609,283 (vs 96,607,490 at action_dim=2 — a 1,793-param delta, latency-irrelevant but faithfulness-relevant, so the export uses action_dim=3).

---

## 2. Per-chip target + quantization recipe — LOCKED with citations

### 2.1 The chip → precision map (PUBLISHED, verified against vendor primaries)

| | **A40** (our proxy) | **Jetson AGX Orin 64 GB** | **Jetson AGX Thor T5000** |
|---|---|---|---|
| GPU arch / SM | Ampere GA102, **SM 8.6** | Ampere, **SM 8.7** | **Blackwell**, SM 11.x |
| tensor-core precisions | fp16/bf16/tf32/int8 | fp16/bf16/tf32/**int8** — **NO fp8, NO fp4** | + **FP8 (E4M3/E5M2) and NVFP4**; Transformer Engine dynamically switches FP4↔FP8 |
| memory bandwidth | 696 GB/s | **204.8 GB/s** | **273 GB/s** |
| L2 | 6 MB | **4 MB** | n/p |
| JetPack / TRT | (x86, TRT via pip) | **JetPack 6.2**: L4T 36.4.3, CUDA 12.6, **TensorRT 10.3**; or JetPack 7.x | **JetPack 7.0**: CUDA 13.0, **TensorRT 10.13**; **7.2**: CUDA 13.2.1, **TensorRT 10.16.2** |
| lowest usable rollout precision | int8 | **int8** (the floor — no lower-risk middle) | **fp8 / nvfp4** |

Primary sources (all in the desk note §2.1, re-verified): [Jetson AGX Orin Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) · [Jetson Thor product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) + [Introducing Jetson Thor (blog)](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) · [Jetson Linux Dev Guide r36.4.3 (Orin)](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/) / [r38.4 (Thor)](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/) · [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).

**Why FP8 is not on Orin:** FP8 tensor cores appear at Ada (SM 8.9) / Hopper (SM 9.0) and Blackwell; **Ampere (SM 8.0/8.6/8.7) has no FP8 datapath** — TensorRT will refuse an FP8 flag on Orin. Orin's only sub-FP16 tensor-core precision is INT8. **On Orin the choice is FP16 or the high-risk INT8 path; there is no low-risk middle.** [PUBLISHED — NVIDIA arch datasheets + TRT support matrix.]

**Why NVFP4 is the Thor-only lever:** NVFP4 (4-bit) quarters the rollout's *weight* traffic — which is 100 % of our binding term (§4.3) — and it exists only on Blackwell's 5th-gen tensor cores + Transformer Engine. **If the deployment target is Thor, FP8/NVFP4 is not an optimisation — it is the reason to buy Thor** (Thor's fp16 bandwidth is only **1.33×** Orin's, so fp16-on-Thor buys ~1.3×, not the headline 7.5×; §4.3).

### 2.2 The recipe, in descending order of confidence (matches the desk note §3.5.5)

1. **FP16 everywhere first.** The one precision step that is simultaneously a tensor-core enabler on Orin, a 2× cut in the binding DRAM traffic, and numerically benign for a fp32/bf16-trained LayerNorm net. **Do it unconditionally.** [Encoder 3.81× under fp16 weights, MEASURED — 07-20 levers note §4.]
2. **Keep the SE(2) accumulation in fp32.** It halves fp16's waypoint deviation (0.0241 → 0.0127 m) for ~zero cost — it is ~20 trivial ops. [MEASURED — 07-20 levers note §4.]
3. **Weight-only INT8/FP8 on the predictor blocks**, keeping activations, the `step_readout`, and the ViT at fp16. This is the best-evidenced config in the whole search and it targets exactly the weight traffic that binds the rollout. [PUBLISHED — weight/activation asymmetry replicates across 4 literatures; desk note §3.5.1.]
4. **Per-channel / per-group calibration, never per-tensor**, for anything transformer-shaped. [PUBLISHED.]
5. **If weight-only INT8 degrades, go straight to QAT** (the PTQ-collapses→QAT-recovers pattern repeats in every task-adjacent source: TAO 78.4→59.1→78.1; CILRS 82→34→62 %).
6. **NEVER joint W4A4 or 3-bit** — collapse is documented on our literal model family (DINO-WM quantization study: W8A8 near-lossless, W4A4 → ≈0 success). [PUBLISHED.]

### 2.3 The horizon-stratified quantization gate (the experiment nothing published substitutes for)

Our margin over the kinematic floor is only **~0.05 m** (v1 ADE@2s 0.4522 vs floor 0.5005). A quantization regression invisible in any "<1 % accuracy" framing erases the model's entire claim to beat the floor. So INT8/FP8 for the rollout is gated on:

- Run the checkpoint open-loop at fp32 and at each candidate precision on the fixed val set; **plot per-step Δpose error AND the SE(2)-accumulated error at every one of the 20 steps** — not just ADE@2s.
- **Gate:** paired episode-cluster bootstrap (`taniteval/ci.py`, never `overlapping_holdout_se`). **Falsifier: any ADE@2s degradation whose paired CI excludes 0, or any point degradation > 0.02 m, kills that precision for the rollout.**
- **The compounding diagnostic:** if quantization error compounds, the degradation ratio must GROW with horizon (ADE@0.5/1/1.5/2 s). Flat ⇒ per-step bias, fix by recalibrating the readout. Superlinear ⇒ keep the state-carrying path at higher precision regardless of budget.

### 2.4 ⚠️ THE INT8 TRAP — this is why the recipe mandates a per-layer benchmark (RETRACTION-worthy if ignored)

**INT8 does not imply speedup on a ViT on Orin — this is documented, not hypothetical.** Multiple NVIDIA primary sources report TensorRT **INT8 running SLOWER than FP16 for transformer/ViT graphs**, and name the mechanism: in INT8 the fused attention path is **broken into many plain matmul ops** (plus QDQ reformatting overhead), which is slower than the FP16 fused-MHA kernel. [PUBLISHED — [NVIDIA/TensorRT #993 "Int8 mode is slower than fp16"](https://github.com/NVIDIA/TensorRT/issues/993) · [#2067 "Qdq model int8 is much slower than fp16 on Orin"](https://github.com/NVIDIA/TensorRT/issues/2067) · [NVIDIA forum: pytorch_quantization transformer INT8 slower than FP16](https://forums.developer.nvidia.com/t/hugging-face-transformer-models-pytorch-quantization-ptq-quantization-int8-is-slower-than-fp16/195453) · [Orin Nano INT8-slower-than-FP16 thread](https://forums.developer.nvidia.com/t/tensorrt-int8-inference-is-slower-than-fp16-in-models-with-conditional-flow/294653).] The specific **~2.7× magnitude** is from the deploy brief [INHERITED] — but the **direction and mechanism are now PUBLISHED**, which is all the recipe needs: INT8 is a HAZARD to disprove per-layer, not a speedup to assume.

**Consequence, binding on the recipe:** INT8 is adopted **only** after a **per-layer `trtexec --dumpProfile` / builder-inspector benchmark shows FP16-vs-INT8 per-layer latency AND the §2.3 accuracy gate both favour INT8** — layer by layer, not model-wide. The default assumption is **FP16**; INT8 is a per-layer *hypothesis*. A model-wide "INT8 = faster" assumption is exactly the kind of unmeasured mechanism the RETRACTION_LOG class C3 is written for.

The per-layer benchmark harness is specified in `BENCHMARK_PLAN.md` (this folder) and is **measurable now on the A40 proxy** (the kernel-selection behaviour differs on Orin, but the harness, the accuracy gate, and the FP16 baseline all transfer).

### 2.5 What the hardware will NOT give us (refuted levers, PUBLISHED)

- **DLA is out.** NVIDIA's own operator matrix: DLA takes `MatMul` only when *"the second input must be a constant"*; attention's `Q·Kᵀ` and `P·V` both have two dynamic inputs — self-attention cannot be expressed on DLA. **Thor has no DLA at all.** Stop spending effort here.
- **FMHA fusion is not guaranteed for ViTs.** Open NVIDIA issue #4537 (DINOv2 ViT-L/14, TRT 10.8) shows the export keeping separate `MatMul`/`Softmax` with no fused-MHA layer. **Our head_dim = 64 satisfies every documented fused-MHA constraint on both SM 8.7 and Blackwell, but fusion must be VERIFIED, not assumed** (`trtexec --dumpLayerInfo`). This is measurable now on the A40 (§3).

---

## 3. MEASURED — static-shape ONNX export + TensorRT-FP16 on the A40 proxy

*Run on `tanitad-pod3` (A40, SM 8.6), under `gpu_lock.sh acquire deploy-prep`, torch 2.8.0+cu128. The deployed v1 ckpt lives only on pod2 (v4 training) and the eval pod (AlpaSim) — both off-limits — so the export uses the **exact deployed `flagship4b` architecture (action_dim 3) with random init**. This is faithful: graph fidelity and latency are architecture- not weight-determined (registry §1.2; 07-20 levers note §8). A real-weight re-export is a one-line `--ckpt` swap.*

### 3.1 ONNX export result — ✅ MEASURED (both graphs build clean)

Run 2026-07-22 on pod3 (A40, torch 2.8.0+cu128, opset 17, static shapes, `action_dim=3`, model 263.44 M = registry `total_model 263,442,838` exactly). Raw: `artifacts/export_report.json`.

| graph | static shape | export | parity max\|Δ\| (torch vs ORT-CPU, n=5) | onnx MB |
|---|---|---|---|---|
| encoder_readout | `[1,9,256,256] → [1,2048]` | ✅ **ok** | **1.25e-6** (PASS, tol 1e-4) | 348.4 |
| predictor | `states[1,8,2048], actions[1,8,3] → (z1,z2,z4)` | ✅ **ok** | **1.9e-6** (PASS, tol 1e-4) | 315.8 |

**Both graphs of the *deployed* architecture export cleanly at opset 17 with no unexportable ops** (MHA / FiLM / causal-triu all export via the legacy TorchScript exporter; one benign `TracerWarning` on the `window==8` assert, which bakes in as the intended static constant). Parity is float-noise (≤1.9 µm on a 30 m trajectory). This extends the 2026-07-08 export (`Production & Optimization/Implementation/onnx_export/parity.json`, step 6500, `base250cam` action_dim 2, CPU: encoder 8.8e-6 / predictor 1.2e-5) to the deployed `flagship4b` action_dim-3 config on the A40. **The static-shape ONNX foundation for a TRT engine is proven.** [MEASURED — this run + prior.]

### 3.2 TensorRT-FP16 engine on the A40 (proxy) — ✅ MEASURED (builds; MHA fuses; fast)

Built via the TensorRT Python API (`OnnxParser` → `build_serialized_network` with `BuilderFlag.FP16`) on **TensorRT 10.16.1.11** (cu12 — the *same major* as Thor's JetPack 7.2 TRT 10.16.2). Raw: `artifacts/trt_fp16_report.json`. (The bare `pip install tensorrt` first pulled TRT **11.1**, a CUDA-13 build that fails on the pod's CUDA-12.8 driver — corrected to `tensorrt-cu12>=10,<11`; logged so the on-device install picks the JetPack-matched TRT.)

| graph | build | build time | engine MB (fp16) | **FP16 latency p50 / p99** | **MHA fused?** |
|---|---|---|---|---|---|
| encoder `[1,9,256,256]→[1,2048]` | ✅ ok | 37.5 s | 177.4 | **1.205 / 1.211 ms** | ✅ **fused** (105 layers, no standalone softmax, Myelin block) |
| predictor `states[1,8,2048],actions[1,8,3]` | ✅ ok | 30.9 s | 180.7 | **0.666 / 0.672 ms** | ✅ **fused** (94 layers, no standalone softmax) |

**Three MEASURED findings that de-risk the Jetson path:**
1. **ONNX→TRT-FP16 builds cleanly for BOTH our ViT encoder and our predictor** — no unsupported ops, no plugin required. The static-shape ONNX path is TRT-ready.
2. ⭐ **TRT FUSES our encoder's MHA** (no standalone `Softmax` layer; a fused Myelin/foreign attention block). **The open NVIDIA #4537 risk — "DINOv2 ViT MHA does not fuse" — does NOT bite our encoder on TRT 10.16 / SM 8.6.** Our head_dim = 64 satisfies the fused-MHA constraints, and here it is confirmed by inspection, not assumed. (Re-verify on the Orin SM 8.7 / Thor Blackwell tactic; fusion is arch-specific.)
3. **The FP16 engine is fast:** the predictor single-call runs **0.666 ms** on the A40 (vs torch fp16 4.12 ms — **6× faster**; TRT fuses the whole block and drops framework dispatch). A 20-step TRT-FP16 rollout is therefore ≈ 20 × 0.666 ≈ **13 ms** + a ~1.2 ms cached-encoder pass ≈ **~14 ms/tick** [ESTIMATED from the measured per-call TRT latency — NOT a measured 20-step tick; it omits the step-readout + window-slide and any inter-step launch not hidden by a graph]. Consistent with, and slightly under, the torch composed 18.75 ms (§4), via a different (TRT) route.

⚠️ **This is an A40 (SM 8.6) PROXY engine — NOT portable to Orin (SM 8.7) or Thor (Blackwell).** *"Without hardware-compatibility mode, TensorRT engines are not portable across different GPU architectures"* (PUBLISHED). Every latency here is an **A40** number. What the A40 build establishes and *does* transfer: the ONNX→TRT path builds, no op needs a plugin, and MHA fusion is achievable for our graph. The on-device engine build + its latency stay hardware-blocked (§6).

### 3.3 Torch A40 latency reference — ✅ MEASURED (independent reproduction on a second A40)

Run 2026-07-22 on pod3 (A40), `artifacts/bench_latency_report.json`. A **predictor-only rollout proxy** (20 predictor forwards; no step-readout/window-slide) — NOT the full tick. Its job is to independently reconfirm the *mechanism* on a different A40 than the eval pod:

| measurement | p50 (ms) | p99 (ms) |
|---|---:|---:|
| predictor 1-call, fp32 | 4.96 | 5.08 |
| predictor 1-call, fp16 | 4.12 | 4.36 |
| 20-step rollout, **eager** fp32 | 96.40 | 102.55 |
| 20-step rollout, **CUDA-graph** (single-step captured, replayed 20×) | **27.87** | **27.88** |

**Independently reproduces every structural claim in §4:** (a) **CUDA-graph capture SUCCEEDS** first-try on a second A40 — the "capture is blocked" worry is doubly refuted; (b) the graph is the **rollout** lever — **96.40 → 27.87 ms, 3.46×** (matches the eval pod's rollout-stage 95.03 → 28.73 ms / 3.31× to within 3 %); (c) the tail **collapses** (p99−p50 = 0.014 ms — deterministic replay); (d) **fp16 barely moves the predictor** (4.96 → 4.12 ms, 1.2×) vs the graph's 3.46× — precision is not the rollout lever. Headline *tick* numbers stay cited from §4 (this proxy omits the encoder + readout). [MEASURED.]

---

## 4. The rollout tick under CUDA graphs — MEASURED (07-20 levers note; registry §1.2)

*Source: `taniteval/results/eff_levers_flagship-30k.json`, A40, batch 1, exclusive under `gpu_lock.sh`, 200 iters × 5 replicates, `contamination_check.valid` sampled before/after/between every variant. All MEASURED.*

| lever | tick p50 (ms) | ×p50 | max abs dev | 10 Hz @ p99? |
|---|---:|---:|---:|:--:|
| eager fp32 | 100.29 | 1.00 | — | ❌ (113.98 p99) |
| **L1b CUDA-graph the 20-step rollout** | 57.18 | **1.75** | **0.0 m (exact)** | ✅ |
| L2 encoder cache alone | 95.11 | 1.05 | 1.9e-6 m | ❌ |
| L3 fp16 weights alone | 98.47 | 1.02 | 0.024 m | ❌ |
| L7 drop 2 unused horizon heads alone | 100.47 | 1.00 | 0.0 m | ❌ |
| **L4 = L1+L2+L3+L7 composed** | **18.75** | **5.35** | 0.024 m | ✅ **53.3 Hz** |

**The captured tick meets 10 Hz at p99 with 5.3× headroom (18.76 ms), and its p50/p99 gap essentially vanishes (18.75 → 18.76) — deterministic, which is what a control loop needs.** Key facts that carry to Jetson as *structure* (not as numbers):

- **CUDA-graph capture succeeds with EXACT equivalence.** The standing worry that ~38 window-slide `torch.cat`s + the per-call causal-mask rebuild block capture was **refuted**: allocations *inside* a capture are served from the graph's private pool and replay at the same addresses. Zero build errors on every variant. [MEASURED.] The real hazard is the opposite — a capture that succeeds and returns *stale* numbers — pinned by a test that a graph fed a different input returns a different output.
- **The CPU round-trips are NOT the cost:** one graph over 20 steps beats 20 single-step replays by 7.7 µs/step. So a runtime that cannot capture a 20-iteration loop loses ~0.3 % — single-step capture, replayed, is the robust deploy default.
- **Levers are SEQUENCED, not additive.** L2/L3/L7 are worth ~1.0× *before* L1 and 24 / 32 / 0.6 ms *after* it. **Capture first, then precision and caching.** Any Jetson port must re-measure the ordering (on Orin, precision comes first — §2 / desk note §2.2).
- ⭐ **CEM/imagine-and-select is NOT latency-blocked.** An 8-candidate fan costs **20.82 ms p50 / 23.72 ms p99** (K=32: 28.41 ms); marginal candidate ≈ 0.3 ms — provided you **encode once and broadcast** (re-encoding per candidate costs +5.6 ms at K=8). This **refutes** the retracted `n_candidates × horizon × per_step = 723 ms` arithmetic. [MEASURED — 07-20 levers note §6b; RETRACTION_LOG 07-21.]

### 4.1 The bandwidth floor no kernel trick can cross (ESTIMATED, from measured params)

At batch 1 each of the 20 steps streams the entire 91.4 M-param predictor from DRAM (≫ Orin's 4 MB L2). Weight traffic 7.31 GB fp32 / 3.66 GB fp16 / 1.83 GB int8 per tick. Divided by device bandwidth:

| precision | rollout floor — Orin (204.8 GB/s) | Thor (273 GB/s) |
|---|---:|---:|
| fp16 | **17.9 ms** | 13.4 ms |
| int8 / fp8 | 9.4 ms | 7.0 ms |
| nvfp4 (Thor only) | — | **3.5 ms** |

**A perfect CUDA graph + perfect TRT engine on Orin still cannot put the fp16 rollout below ~18 ms.** Below that, the only levers are **fewer sequential steps** (the already-trained k=2/k=4 heads → 10/5 steps, gated on step-readout recalibration + a two-sided closed-loop eval) or a **smaller predictor**. [ESTIMATED from MEASURED params — desk note §1.3/§4.]

---

## 5. The staged deployment sequence

| stage | action | precision | measurable now? |
|---|---|---|---|
| **S0** | ONNX export, static shapes, deployed `flagship4b` arch | fp32 graph | ✅ **§3 (this intake)** |
| **S1** | TensorRT **FP16** engine (Orin: mandatory first lever; A40: a proxy build) | fp16 | ✅ build path on A40; ❌ Orin engine (on-device) |
| **S2** | full-tick / single-step-replayed **CUDA graph** over the TRT context | — | partial (A40 torch proxy ✅; TRT+graph on-device ❌) |
| **S3** | **encoder cache** (encode 1 new frame, reuse 7) — worth 24 ms *after* S2, ~0 before | — | ✅ mechanism measured |
| **S4** | keep SE(2) accumulate in **fp32** | fp32 tail | ✅ measured |
| **S5** | **weight-only INT8/FP8** on predictor blocks — behind the §2.3 gate AND the §2.4 per-layer benchmark | int8/fp8 | ⚠️ A40 proxy for build; Orin kernel-selection differs |
| **S6** | (Thor only) **NVFP4** predictor weights | nvfp4 | ❌ needs Blackwell silicon |
| **S7** | (fallback if S2 unavailable on target) strided k=2/k=4 rollout — gated on readout recalibration + closed-loop | — | latency ✅; accuracy gate not started |

**Ordering rule:** on the A40, capture-first (§4). **On Orin, precision-first** — a naive fp32 port is arithmetic-starved (402 GFLOP against Orin's 5.32 TFLOPS fp32 peak = 75.6 ms floor at 100 % of peak; the fp32 encoder alone projects to ~196 ms). Only after the FP16/INT8 engine collapses the GPU work does the tick become launch-bound and the CUDA graph bind harder than on the A40. **Re-measure the order on the target; do not assume it.** [ESTIMATED/PUBLISHED — desk note §2.2.]

---

## 6. Deployment-readiness checklist — measurable NOW vs hardware-blocked

### ✅ Measurable now (A40 / dev-box proxy) — done or in this intake

| item | status |
|---|---|
| Static-shape ONNX export of deployed arch (encoder + predictor) | ✅ §3.1 |
| ONNX parity (torch vs ORT) | ✅ §3.1 |
| ONNX→TRT-FP16 engine **builds**? + FP16 latency reference | ✅ §3.2 — builds; enc 1.205 ms / pred 0.666 ms (A40 proxy) |
| Fused-MHA present in the TRT engine? (#4537 risk) | ✅ §3.2 — **fused** (both graphs; risk retired on SM 8.6) |
| Per-layer FP16 profile (seeds the INT8 benchmark) | ✅ §3.2 (engine-inspector layer info; extend to INT8 per BENCHMARK_PLAN) |
| CUDA-graph capture of the rollout, exact-equivalence + tick | ✅ §4 (07-20 measured) |
| CEM/fan latency (K=1..32) | ✅ §4 (07-20 measured) |
| Bandwidth-floor model (Orin/Thor) | ✅ §4.1 (estimated from measured params) |
| Per-layer FP16-vs-INT8 benchmark **harness** | `BENCHMARK_PLAN.md` (spec ready; INT8 calibration = the run) |

### ❌ Hardware-blocked (needs real Orin/Thor silicon — escalate, do not fabricate)

| item | why blocked |
|---|---|
| Real Orin / Thor **throughput** (the actual vehicle tick) | no chip on hand; A40 ≠ Orin (see gap below) |
| The **shippable engine build** | TRT engines are **not portable across GPU architectures** (PUBLISHED) — must build on the target device |
| Any **NVFP4** accuracy or latency number | needs Blackwell silicon; no source tests FP4 on a ViT / regression task at all |
| Orin **INT8 kernel-selection** (the 2.7×-slower trap) | kernel tactics are per-arch; the A40 proxy cannot reproduce Orin's INT8 path |
| Jetson **launch-latency** (settles capture-vs-precision order on-device) | NVIDIA has declined to publish Jetson launch-latency data |

### ⚠️ The proxy gap, stated honestly

The A40 is a **300 W datacentre GA102 at SM 8.6**; Orin is a **~60 W SM-8.7 SoC with ~1/7 the fp32 arithmetic and ~1/3.4 the bandwidth**, unified memory (no PCIe submission term), and a weaker ARM launch path. **No A40 absolute number transfers to a vehicle.** What transfers is *structure*: the launch-bound diagnosis, the bandwidth-floor arithmetic, the exact-equivalence of graph capture, the ONNX exportability, and the sequenced-not-additive lever behaviour. Every latency in §3/§4 is an **A40** number; the Orin/Thor columns in §2.1/§4.1 are vendor-spec-derived floors, not measured ticks.

---

## 7. v4 implications (brief — v4 just started training, do not over-analyze)

v4 = `flagship-v4-30k`, ≈ **247.9 M** (~30 M *smaller* than v1), joint world-model + three-planner diffusion stack (`V4_FLAGSHIP_DESIGN.md`). For deployment:

1. **The expensive object is unchanged.** v4's operative predictor is **v1 verbatim** — d768 × 10, action_dim 3, **96,609,283 params** (`V4_FLAGSHIP_DESIGN.md §3.1`). The 20-step rollout, the CUDA-graph lever, the bandwidth floor (§4.1), and the entire §2 precision map **apply to v4 unchanged**.
2. **The new deployment term is the diffusion operative planner** — the in-repo `FlagshipV15Head`/`V15Decoder` (= REF-C's `AnchoredDiffusionDecoder`), ~9.78 M + ≤0.81 M factorised heads. Its cost is the **denoise-step count**: the *identical* REF-C decoder measures **classifier pass 3.03 ms + 2.89 ms per denoise pass** on the A40 (256 anchors, MEASURED — 07-20 v1-vs-REF-C note §2.2). So a 2-pass operative planner ≈ **8.8 ms** on top of the rollout tick — parallel-in-modes, not serial-in-time.
3. **The denoise count is the tunable knob.** DiffusionDrive (REF-C's reference arch): 1 / 2 / 3 steps → 87.9 / 88.1 / 88.1 PDMS (flat) — **fewer denoise steps is nearly free** (PUBLISHED). Deployment should size the operative planner at the smallest denoise count that holds accuracy (likely 1–2), directly trading ~2.9 ms/pass against the tick budget.
4. **Net:** v4's deploy budget ≈ v1's rollout tick + ~3–9 ms diffusion planner; 30 M fewer params slightly cuts encoder/weight traffic but the dominant rollout term is identical. **No new precision or export risk beyond v1's** — the planner is the same anchored-diffusion family already proven cheap.

---

## 8. Escalations, gaps, and next actions

1. ✅ **The INT8-slower-than-FP16 trap (§2.4) is now PUBLISHED-backed** (NVIDIA/TensorRT #993, #2067, forum threads — mechanism: INT8 breaks fused attention into many matmuls). Only the exact **2.7× magnitude** remains INHERITED from the brief; the direction/mechanism no longer are. The recipe's per-layer benchmark gate stands regardless.
2. 🟡 **The deployed v1 ckpt is only on pod2 (off-limits) + eval (off-limits).** The export/latency here are weight-independent and faithful, but a real-weight re-export + a canonical 40-episode fp16/int8 re-score (the §2.3 accuracy gate) need the ckpt reachable on a free pod. **Escalation:** stage a copy of `flagship4b-speedjerk-30k/ckpt.pt` to a free pod (or HF) when pod2 is not mid-checkpoint.
3. 🟡 **Stale retracted claim still live in two docs:** `FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md §5.1` and `2026-07-20-inference-efficiency-v1-vs-refc.md §7.2` still print "8 candidates × 20 steps = 723 ms/tick" as if current. It is **RETRACTED** (measured 20.82 ms; RETRACTION_LOG 07-21). Flagged for the owner to fix (not edited here — those are another stream's live docs).
4. **On-device build kit.** When Orin/Thor arrives: JetPack 6.2 (Orin) / 7.x (Thor), `trtexec --fp16 --dumpLayerInfo --dumpProfile` on the §3 ONNX, then the §2.3 accuracy gate + §2.4 per-layer INT8 benchmark, then CUDA-graph wrap of `enqueueV3`.

---

## Deliverable manifest

| # | artifact | where it lives | only copy? |
|---|---|---|---|
| 1 | `DEPLOYMENT_PLAN.md` — this plan | `repo: …/incoming/2026-07-22-orin-thor-deployment/` (staged) | no — derivable from the 3 cited notes |
| 2 | `BENCHMARK_PLAN.md` — per-layer FP16-vs-INT8 harness spec + on-device runbook | same folder (staged) | the spec is only here |
| 3 | `export_and_bench.py` — static-shape ONNX export of deployed arch (+ latency scaffold) | same folder (staged) + `tanitad-pod3:/root/` | no |
| 4 | `bench_latency.py` — A40 torch predictor/rollout/graph latency probe | same folder (staged) + pod3 | no |
| 5 | `trt_build.py` — ONNX→TRT-FP16 engine build + latency + MHA-fusion inspector | same folder (staged) + pod3 | no |
| 6 | `artifacts/export_report.json` — MEASURED export + parity (encoder/predictor) | same folder (staged) + pod3 | the A40 measurement |
| 7 | `artifacts/bench_latency_report.json` — MEASURED torch A40 latency (graph reproduction) | same folder (staged) + pod3 | the A40 measurement |
| 8 | `artifacts/trt_fp16_report.json` — MEASURED TRT-FP16 build (builds, fused, latency) | same folder (staged) + pod3 | the A40 measurement |
| — | ONNX graphs (348/316 MB) | `pod3:/workspace/deploy_onnx/{encoder_readout_f4b,predictor_f4b}.onnx` | **pod-only** (too large to stage; regenerable via `export_and_bench.py`) |

**Integration note:** this is a self-contained intake (research + export recipe + benchmark plan). Nothing needs merging into `stack/`. The one cross-stream action is escalation #3 (a retracted 723 ms claim still live in two Production & Optimization docs) — flagged, not edited.

**Cross-refs:** desk note `Production & Optimization/Research/2026-07-20-orin-thor-deployment-and-inference-levers.md` · levers note `…/2026-07-20-flagship-v1-inference-levers-measured.md` · master `…/FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md` · `MODEL_REGISTRY.md §1.2` · `V4_FLAGSHIP_DESIGN.md §3.1`.
