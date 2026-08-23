# Per-layer FP16-vs-INT8 benchmark — flagship-v1 encoder + operative predictor

- **Date / agent:** 2026-07-23, deployment/inference stream (`orin-int8`), pod1 `tanitad-pod` (NVIDIA RTX A6000, SM 8.6, Ampere — proxy for Jetson AGX Orin's SM 8.7, same tensor-core generation as the 2026-07-22 A40 intake).
- **Mission:** the one MANDATED follow-up the 2026-07-22 `2026-07-23-orin-thor-deployment` intake flagged but did not run: *"a per-layer FP16-vs-INT8 benchmark, because the published ViT-INT8 trap is that INT8 can silently wreck encoder accuracy on some layers."* This closes it — and finds the trap bites on **latency**, not (mainly) on the transformer blocks' accuracy.
- **Model:** `flagship4b-speedjerk-30k` (flagship v1, deployed, `Sayood/tanitad-flagship-4b-speedjerk`, ckpt **step 29999** confirmed from the loaded checkpoint), `action_dim=3`, 263,442,838 core-model params (MEASURED, matches `MODEL_REGISTRY.md` §1.2 exactly via `tanitad.eval.ckpt_compat.build_world_from_ckpt` strict-load — see §5 for a correction to the prior intake's construction path).
- **Evidence classes used throughout:** **MEASURED** (this run, A6000) · **PUBLISHED** (cited) · **ESTIMATED** (scaled from MEASURED by a stated assumption) · **INHERITED** (the 2026-07-22 intake, not re-verified here beyond the FP16 reproduction check in §2).

---

## 0. Headline — three findings, most important first

1. **⭐ Blanket INT8 does not clear Gate A (latency) on this hardware, at all.** Real TensorRT-INT8 (entropy-calibrated, weight+activation, real data) is **2.1 % faster** than FP16 on the encoder (1.108 vs 1.1315 ms) and **2.1 % SLOWER** than FP16 on the predictor (0.596 vs 0.5837 ms) — **MEASURED**. This is not noise dressed up: the per-layer profile shows *why* — INT8 adds a real reformatting/re-quantization tax at the network boundary (encoder: 1.1 %→2.9 % of total time in `Reformatting CopyNode` ops) and shifts where time sits inside attention (predictor: attention-related layers 22.0 %→32.2 % of total time) without shrinking the total. **This is the PUBLISHED trap (NVIDIA/TensorRT #993, #2067 — "INT8 breaks fused attention into more ops, plus QDQ reformatting") reproduced on our own architecture, mechanistically, not just as a headline number.**
2. **Accuracy: the transformer blocks are essentially immune to INT8; one specific head is not.** Every encoder transformer block (0–11) and every predictor block (in_proj, act_emb, blocks 0–9, heads) tolerates **weight-only** INT8 with isolated cosine ≥ 0.9999990 (encoder) / ≥ 0.9999997 (predictor) — indistinguishable from FP32. The encoder's **`readout_head`** (the final spatial-pool→state projection, `SpatialGridReadout.proj`) is the one real hazard: weight-only INT8 is fine (cos 0.9995) but **weight+activation INT8 alone collapses it to cosine 0.566** — and this single block accounts for essentially the *entire* blanket-encoder W+A failure (blanket W+A cos 0.568 ≈ the isolated readout-head number). **No transformer block drives the encoder's INT8 accuracy risk — the un-normalized post-pool projection does.**
3. **The 20-step rollout compounds a per-step near-miss into a real one.** Per single predictor call, weight+activation INT8 looks almost perfect (cosine ≥ 0.9999 on every isolated block). Rolled out 20 steps with the real grounding decoder on 880 held-out windows, the downstream ADE proxy shows int8-weight-only **passes** cleanly (ADE@2s +0.0065 m, well under the 0.02 m falsifier) but int8-weight+activation **fails** (+0.0215 m, past the 0.02 m falsifier line) — **and the degradation ratio grows sharply with horizon (27×, 0.5 s→2 s)**, the exact compounding signature the pre-registered reading was watching for.

**Net call: FP16 stays the default for the whole model. Real, calibrated, weight+activation INT8 buys ~nothing on this hardware and costs real accuracy on the rollout and on one specific encoder head. If a future toolchain supports true weight-only (QDQ-scoped) INT8, the transformer blocks are excellent candidates and the readout head is not — see the per-layer map in §3.**

---

## 1. What is MEASURED here, and what it extends

`export_and_bench.py` / `trt_build.py` (2026-07-22, pod3, A40, **random-init weights**, torch 2.8.0+cu128) proved the ONNX→TRT-FP16 path builds and gave the FP16 baseline (encoder 1.205 ms / predictor 0.666 ms) and confirmed MHA fusion. This run extends it with the two things that intake explicitly could not do:

- **Real deployed weights** (the actual `ckpt.pt`, step 29999, pulled from HF `Sayood/tanitad-flagship-4b-speedjerk` onto pod1 — this also closes that intake's escalation #2, "stage a copy of the ckpt to a free pod," now done).
- **The mandated per-layer FP16-vs-INT8 benchmark itself** — both gates (A: latency, B: accuracy), per-block, plus a downstream rollout proxy.

**FP16 baseline reproduction check** (different pod, different GPU SKU, real weights vs random-init, single-head vs 3-head predictor export — four differences at once, so treat as directional, not identical):

| | 2026-07-22 (A40, random-init) | 2026-07-23 (A6000, real weights, this run) |
|---|---:|---:|
| encoder FP16 p50 | 1.205 ms | **1.1315 ms** |
| predictor FP16 p50 | 0.666 ms | **0.5837 ms** |

Both are lower here, consistently, which is exactly what "real weights, single-output-head graph, same architecture generation" predicts (no red flag) — **MEASURED, both rows**.

**Environment note (torch version gap, fixed, not worked around by lowering rigor):** pod1 runs **torch 2.4.1+cu124** (vs pod3's 2.8.0+cu128). torch 2.4.1's ONNX exporter has no symbolic for `aten::_native_multi_head_attention` at opset 17 (`nn.MultiheadAttention` export fails outright). Rather than gamble on upgrading torch/CUDA on a shared pod (a cu128 wheel needs a newer driver than pod1's 550.127.08 — real risk of breaking other work on the pod), every `nn.MultiheadAttention` was replaced, **for export only**, with a hand-decomposed attention module that reads the **identical parameter tensors** (not copies) and computes the same math via primitive ops (`Linear`/`matmul`/`softmax`). **Verified before trusting the export:** decomposed-vs-original max abs delta = **1.49e-6 (encoder) / 1.43e-6 (predictor)** on random probe inputs — float noise. ONNX-vs-eager parity after export: **1.996e-6 (encoder) / 1.490e-6 (predictor)**. Both PASS.

**Correction to the 2026-07-22 intake (harmless, but worth fixing):** that intake's `export_and_bench.py` builds the model via `cfg.predictor.action_dim = 3` (a direct post-hoc mutation) rather than `tanitad.eval.ckpt_compat.build_world_from_ckpt`'s `dataclasses.replace`, which also widens `tactical_pred.action_dim`. The mutation under-counts params by exactly 512 (263,442,326 vs the registry's canonical 263,442,838) because it never touches `tactical_pred`. **MEASURED here**: `build_world_from_ckpt` + strict `load_state_dict` on the real checkpoint reproduces **263,442,838 exactly**, with zero missing/unexpected keys. This does not affect either intake's actual exported encoder/predictor graphs (neither touches `tactical_pred`), so no numbers there need correcting — flagging it purely so nobody reuses the shortcut and gets a silently-wrong param count elsewhere.

---

## 2. Gate A — real TensorRT latency, FP16 vs INT8 (MEASURED, A6000)

INT8 built via TensorRT's calibrated (implicit-quantization) path: `BuilderFlag.INT8 | BuilderFlag.FP16` (FP16 fallback for unquantized layers, per the runbook), `IInt8EntropyCalibrator2`, **256 real calibration samples** from `physicalai-train-e438721ae894` (encoder: single real frames; predictor: real (state, action) windows harvested by running the real fp32 encoder — both per BENCHMARK_PLAN.md §2). TensorRT 10.16.1.11 (same version the 07-22 intake used).

| graph | precision | p50 | p99 | mean | engine size | build time | MHA fused? |
|---|---|---:|---:|---:|---:|---:|:--:|
| encoder | FP16 | 1.1315 ms | 1.3118 ms | 1.1372 ms | 176.8 MB | 26.7 s | ✅ |
| encoder | **INT8** | **1.108 ms** | 1.2761 ms | 1.1117 ms | 179.4 MB | 124.1 s | ✅ |
| predictor | FP16 | 0.5837 ms | 0.6021 ms | 0.5849 ms | 176.2 MB | 34.8 s | ✅ |
| predictor | **INT8** | **0.596 ms** | 0.6089 ms | 0.5977 ms | 174.6 MB | 125.8 s | ✅ |

**Encoder INT8/FP16 = 1.021× (a 2.1 % win). Predictor INT8/FP16 = 0.979× (a 2.1 % LOSS).** Neither is the "7–16×" or even "2×" the desk research's *theoretical* TOPS ratios would suggest (Ampere INT8:FP16 dense TOPS ≈ 2:1) — because the workload is tiny (batch 1, ≤350 MB of weights) and dominated by kernel-launch/reformatting overhead, not arithmetic. INT8 calibration alone also takes **~4–5× longer to build** (124–126 s vs 27–35 s) — relevant for planning an on-device Jetson build/calibration cycle, which will be far slower again on Orin's weaker CPU.

**Mechanism, from the per-layer execution profile** (`IProfiler`, `ProfilingVerbosity.DETAILED`, 30-rep average per layer):

| graph | precision | total (Σ per-layer ms) | `Reformatting CopyNode` share | attention-related share |
|---|---|---:|---:|---:|
| encoder | FP16 | 1.3193 ms | 1.1 % | 35.8 % |
| encoder | INT8 | 1.2847 ms | **2.9 %** | 35.7 % |
| predictor | FP16 | 0.8526 ms | 2.2 % | 22.0 % |
| predictor | INT8 | 0.8653 ms | 2.2 % | **32.2 %** |

The encoder's INT8 build inserts an explicit `Reformatting CopyNode for Input Tensor 0 to /encoder/patch/Conv` (0.027 ms) that doesn't exist in FP16 — quantize/dequantize tax right at the input, one of the two published mechanisms. The predictor shows the other: attention's *share* of total time grows under INT8 even though nothing there was supposed to get slower — consistent with "INT8 attention decomposes into more, less-fusable ops" (TensorRT #993/#2067). **Both engines report `fused` (no standalone Softmax) by the coarse fusion check — the #4537 "MHA doesn't fuse" risk still does not bite us — but "fused" is not the same as "fused equally efficiently," and the profile shows it isn't.**

⚠️ **Per-block hardware latency is only PARTIALLY attributable.** TensorRT's Myelin fuser merges most of each transformer block into named-but-opaque kernels (`__myl_Fc_myl3_N`, `_gemm_mha_v2_myl3_N`, `__myl_AddCastMean...myl3_N`) that mix pieces of adjacent ops; only the MLP's second matmul, the fused QKV projection, and the attention output projection keep their `/encoder/blocks.N/...` scope names in the profile. Where visible, **per-block latency is uniform within ~10 % across all 12 encoder blocks** (MLP-2 matmul: 0.0226–0.0238 ms; QKV: 0.0157–0.0175 ms; out-proj: 0.0104–0.0121 ms) in both precisions — no single block dominates or is a latency outlier. **A true complete per-block hardware breakdown would need 12 separate single-block sub-engines (not built here — time-boxed out); the per-block ACCURACY map in §3, built in PyTorch by isolating one block at a time, is the reliable per-block deliverable, and it is complete for every block.**

---

## 3. Gate B — per-block accuracy map (MEASURED, PyTorch weight/activation fake-quantization, real weights + real data)

Method: for each named block, **isolated** (that block only, per-output-channel symmetric INT8 fake-quantization of every internal `Linear`/`Conv2d`/`MultiheadAttention` weight; every other block stays FP32) vs **blanket** (every block quantized at once). `int8_wo` = weight-only; `int8_wa` = weight-only + a per-tensor dynamic activation fake-quant at that block's boundary. 96 real held-out frames / (state, action) pairs from `physicalai-train-e438721ae894` (a different slice than TRT calibration and than the downstream-proxy episodes — no reuse across measurements that matter for a claim). Threshold: the mission's own bar, **cosine ≥ 0.999**.

### 3.1 Encoder (patch-embed → 12 blocks → readout)

| block | isolated INT8 weight-only cos | isolated INT8 W+A cos | verdict |
|---|---:|---:|---|
| patch_embed | 1.000000 | 0.999679 | **INT8-safe (both)** |
| enc_block_0 | 0.9999997 | 0.9999917 | **INT8-safe (both)** |
| enc_block_1 | 1.000000 | 0.9999997 | **INT8-safe (both)** |
| enc_block_2 | 1.000000 | 0.9999997 | **INT8-safe (both)** |
| enc_block_3 | 1.000000 | 0.9999995 | **INT8-safe (both)** |
| enc_block_4 | 1.000000 | 0.9999992 | **INT8-safe (both)** |
| enc_block_5 | 1.000000 | 0.9999967 | **INT8-safe (both)** |
| enc_block_6 | 1.000000 | 0.9999968 | **INT8-safe (both)** |
| enc_block_7 | 1.000000 | 0.9999965 | **INT8-safe (both)** |
| enc_block_8 | 1.000000 | 0.9999954 | **INT8-safe (both)** |
| enc_block_9 | 1.000000 | 0.9999959 | **INT8-safe (both)** |
| enc_block_10 | 1.000000 | 0.9999826 | **INT8-safe (both)** |
| enc_block_11 | 0.9999990 | 0.9994242 | **INT8-safe (both — but the closest call of the 12)** |
| **readout_head** | 0.9994738 | **0.5660150** | **weight-only safe; ACTIVATION QUANTIZATION MUST STAY FP16 — the one real hazard** |

**Blanket (all 12 blocks + readout at once):** FP16 cos 0.99999988 · **INT8 weight-only cos 0.9994700 (SAFE, ≥0.999)** · **INT8 weight+activation cos 0.5675597 (FAILS — and it is almost entirely the readout head: isolated 0.566 vs blanket 0.568, i.e. the 12 transformer blocks' own W+A error barely adds anything on top).**

**Why the readout head specifically:** it is `SpatialGridReadout.proj`, a plain `Linear(768→32)` reading straight off `AvgPool2d` output with **no LayerNorm in front of it** — every transformer block's Linear/MHA sees a freshly-LayerNorm'd, well-scaled input; this one doesn't. This is exactly the documented ViT-quantization hazard class (FQ-ViT: LayerNorm-adjacent activations are well-behaved; RepQ-ViT/PTQ4ViT: un-normalized post-pool / outlier-channel activations are not) — reproduced concretely, on our own model, at our own readout head, not just cited from the literature.

### 3.2 Predictor (in_proj → act_emb → 10 causal blocks → per-horizon heads)

| block | isolated INT8 weight-only cos | isolated INT8 W+A cos | verdict |
|---|---:|---:|---|
| in_proj | 1.000000 | 0.9999999 | **INT8-safe (both)** |
| act_emb | 0.9999997 | 0.9998741 | **INT8-safe (both — the lowest of the predictor's blocks)** |
| pred_block_0…9 (all 10) | 1.000000 (every block) | 0.999996 – 1.000000 | **INT8-safe (both), every block** |
| pred_heads | 1.000000 | 0.9999994 | **INT8-safe (both)** |

**Blanket:** FP16 cos 1.0 · **INT8 weight-only cos 0.9999997 (SAFE)** · **INT8 weight+activation cos 0.9998685 (nominally ≥0.999 at the single-step level — but see §4: this is exactly the case the 20-step rollout turns into a failure).**

**Predictor verdict: no block is an accuracy hazard in isolation, at the single-step level.** The risk here is purely the autoregressive compounding in §4, not any one layer.

---

## 4. Downstream proxy — 20-step rollout ADE, held-out train-cache episodes (MEASURED)

**Scope and an explicit limitation, stated up front:** 880 windows from **40 held-out episodes** (`physicalai-train-e438721ae894`, episodes 2000–2039 — disjoint from the 150 episodes used for TRT calibration and the 96 samples used for §3's accuracy sweep) — the real `HierarchicalGrounding.step["op"]` decoder (loaded from the real checkpoint, zero missing/unexpected keys), the real `rollout_decode` + `accumulate_se2` (kept FP32 throughout, per the deployment recipe). **This is the TRAIN corpus, not the canonical `physicalai-val-0c5f7dac3b11` taniteval val set** (that lives on `tanitad-eval`, explicitly off-limits to this pod1-only stream) — every number below is a **directional proxy**, point-estimate only (no bootstrap CI — that treatment is for a canonical val run), never to be quoted as a registry ADE. Only the **predictor's** precision varies below; the encoder stays FP32 throughout (isolates the rollout/compounding question specifically).

| precision | ADE@2s (0–2s mean) | Δ vs FP32 | falsifier (>0.02 m or CI-excludes-0)? | degradation ratio, 0.5s→2s |
|---|---:|---:|:--:|---:|
| fp32 (reference) | 0.42626 | — | — | — |
| fp16 | 0.42634 | **+0.000076 m** | ✅ pass (negligible) | 0.10× (flat/noise) |
| int8 weight-only | 0.43279 | **+0.006521 m** | ✅ pass (well under 0.02 m) | 47× |
| **int8 weight+activation** | **0.44773** | **+0.021463 m** | 🔴 **FAILS (past 0.02 m)** | **27×** |

**The compounding signature is exactly what the pre-registered reading was watching for:** weight+activation INT8's per-step decoded Δpose error is indistinguishable from FP32 at 0.5 s (Δ +0.0019 m) but grows to +0.0517 m by 2 s — the ratio *grows with horizon*, not flat, which per BENCHMARK_PLAN.md §Gate-B means "keep the state-carrying path (here: the predictor's activations) at higher precision," not "recalibrate the readout and move on." **Weight-only INT8, by contrast, stays well clear of the falsifier despite a similar-shaped but much smaller growth curve — its absolute magnitude never threatens the 0.02 m line.**

---

## 5. The mixed-precision recommendation (the deliverable)

**Both outcomes were pre-registered; this is the one that landed, and it is not "INT8 is fine":**

> **FP16 stays the default for the entire model on Ampere-class hardware (A6000 proxy for Orin).** A calibrated, TensorRT weight+activation INT8 build (a) buys **no net latency win** — 2.1 % faster on the encoder, 2.1 % SLOWER on the predictor, both close enough to call it a wash — and (b) **fails the accuracy gate**: the encoder's readout head collapses (cosine 0.566) and the predictor's 20-step rollout crosses the pre-registered 0.02 m falsifier (+0.0215 m) via genuine error compounding. **Neither gate clears for blanket INT8.**

If INT8 is revisited later (e.g. a toolchain that supports true weight-only/QDQ-scoped quantization, which this run's TensorRT calibrated path does not give us — see §6 scope note):

| component | INT8-safe (weight-only) | Must stay FP16 |
|---|---|---|
| **Encoder** | patch-embed, all 12 transformer blocks | **`readout_head`'s activations** (its weights alone are fine, cos 0.9995) |
| **Predictor** | in_proj, act_emb, all 10 causal blocks, per-horizon heads — **every block, weight-only AND weight+activation, at the single-step level** | **The rollout as a whole, if activations are quantized** — no single block is the problem; the 20-step autoregressive chain is. Weight-only survives the full rollout; weight+activation does not. |

**One line for Sayed:** *the ViT-INT8 trap we were warned about is real on our model, but it shows up as a **latency non-win** first and an **accuracy failure concentrated in one un-normalized head plus 20-step compounding** second — not as scattered failures across the transformer blocks, which are all excellent INT8 candidates on the accuracy axis alone.*

---

## 6. Orin / Thor — ESTIMATED (scaled from this run's MEASURED A6000 latencies)

**Method (exactly as instructed):** `latency(device, precision) = latency_MEASURED(A6000, matched precision) × [TOPS_dense(A6000) / TOPS_dense(device, precision)]` — inversely proportional to published dense tensor-core TOPS/TFLOPS, a rough **compute-bound upper bound**. Sources: A40/A6000 (same GA102 die, 84 SM, 336 3rd-gen TC) 149.7 FP16 / 299.3 INT8 dense **[PUBLISHED, A40 datasheet]**; Orin 42.5 FP16 / 85 INT8 dense **[ESTIMATED — derived in the 2026-07-20 desk note from NVIDIA's 170 sparse-INT8-TOPS *GPU-only* figure (excludes the ~105 TOPS of DLA, which cannot run our attention layers — §3.7 of that note, refuted)]**; Thor 255 FP16 **[ESTIMATED, desk note's own "~6× Orin at fp16" ratio]** / 518 FP8 **[ESTIMATED, RidgeRun 1035 sparse ÷2]** / 1035 NVFP4 **[ESTIMATED, NVIDIA 2070 FP4-sparse ÷2]**.

| | encoder (1 call) | predictor (1 call) |
|---|---:|---:|
| **A6000 MEASURED — FP16 / INT8** | 1.1315 / 1.108 ms | 0.5837 / 0.596 ms |
| Orin ESTIMATED — FP16 / INT8 | 3.986 / 3.902 ms | 2.056 / 2.099 ms |
| Thor ESTIMATED — FP16 / FP8 / NVFP4 | 0.664 / 0.327 / 0.164 ms | 0.343 / 0.169 / 0.084 ms |

⚠️ **The compute-bound scaling above predicts an Orin INT8 win (3.902 < 3.986 ms) that our own MEASUREMENT directly contradicts (§2: INT8 barely beat FP16 on the encoder and lost to it on the predictor, on the SAME architecture family).** Orin's INT8 kernel-selection is documented (§2.4 of the 2026-07-22 intake, PUBLISHED GitHub issues) to be **at least as prone** to this trap as datacenter Ampere. **Read the Orin-INT8 estimate as a theoretical ceiling only — the honest expectation, given what we measured, is that real Orin INT8 will track close to or slightly behind Orin FP16, not meaningfully ahead of it.**

**Full-tick estimate** (`encoder(1 cached frame) + 20 × predictor`; "graph-projected" applies the 3.46× CUDA-graph rollout speedup **MEASURED on A40/A6000-class silicon in this repo** — a structural assumption carried to Jetson, since CUDA-graph-on-TRT was not built for Orin/Thor here, hardware-blocked, no chip on hand):

| device / precision | naive (ms) | graph-projected (ms) | vs 100 ms / 10 Hz budget |
|---|---:|---:|:--:|
| Orin FP16 | 45.1 | **15.9** | ✅ both |
| Orin INT8 (see caveat above) | 45.9 | 16.0 | ✅ both, but expect no real gain over FP16 |
| Thor FP16 | 7.5 | **2.6** | ✅ both |
| Thor FP8 | 3.7 | 1.3 | ✅ both |
| Thor NVFP4 | 1.9 | 0.7 | ✅ both |

**Cross-check against the existing, more physically-appropriate bandwidth-floor model** (2026-07-20 desk note §4.1 — the predictor/rollout is DRAM-bandwidth-bound, not compute-bound, so a TOPS-ratio scaling is the wrong physical model for it specifically): **Orin fp16 rollout floor 17.9 ms / int8 9.4 ms; Thor fp16 13.4 ms / int8-or-fp8 7.0 ms / nvfp4 3.5 ms** — all **lower** than this note's naive `20 × predictor` estimate (Orin fp16 ≈ 41.1 ms), because compute-bound scaling over-estimates a memory-bound component. **For the predictor/rollout specifically, trust the bandwidth-floor numbers over the TOPS-ratio numbers in this note; for the encoder (94 % of tick FLOPs, genuinely compute-bound per that same desk note), this note's TOPS-ratio numbers are the more appropriate estimate.** Every number in this section is **ESTIMATED** — no Orin/Thor silicon is on hand; this remains hardware-blocked exactly as the 2026-07-22 intake stated.

---

## 7. Escalations

1. ✅ **Closed**: the 2026-07-22 intake's escalation #2 ("stage a copy of `flagship4b-speedjerk-30k/ckpt.pt` to a free pod") — done here, pod1, staged from HF, strict-load verified against the registry's exact param count.
2. 🟡 **New, minor**: `export_and_bench.py` (2026-07-22 intake)'s `cfg.predictor.action_dim = 3` shortcut under-widens `tactical_pred` by 512 params vs the registry canonical count (§1). Harmless to that intake's actual results (neither exported graph touches `tactical_pred`) — flagged so nobody reuses the shortcut and gets a silently-wrong total elsewhere. Not fixed in that file (another stream's artifact); a one-line fix (`build_world_from_ckpt` or `adapt_config_action_dim`) if anyone touches it next.
3. 🟡 **Scope boundary, stated honestly, not silently narrowed:** the INT8 engines here are TensorRT's **calibrated (implicit-quantization) weight+activation** path — the real, standard, "does this help at all" hardware test, and the one the trap warnings are actually about. A true **weight-only** TensorRT build (QDQ nodes scoped to weights only) was not attempted — a materially bigger export-graph-surgery lift — so the weight-only numbers in §3/§4 are a **PyTorch fake-quantization simulation** (real weights, real data, isolated per block), not a second real TensorRT engine. It is the right tool for the accuracy question (which is where weight-only vs weight+activation actually diverges) and is not a substitute for a real weight-only TRT latency number, which nobody has built for this model yet.
4. 🟡 **Environment drift, worth a nightly-drift-style note:** pod1's torch (2.4.1+cu124) is older than pod3's (2.8.0+cu128, used for the 2026-07-22 intake) and cannot export `nn.MultiheadAttention` to ONNX at all without the decomposition worked around here. Any future export work on pod1 (or any pod with torch <~2.6) needs the same `DecomposedMHA` pattern (or a torch upgrade, not attempted here — risk: cu128 wheels may need a newer driver than pod1's 550.127.08, untested).

---

## Deliverable manifest

| # | artifact | where it lives | only copy? |
|---|---|---|---|
| 1 | `DEPLOYMENT_READINESS_INT8.md` — this report | `repo: …/incoming/2026-07-23-orin-int8-benchmark/` (staged) | no — derivable from the JSON + scripts |
| 2 | `orin_int8_benchmark.json` — full MEASURED report (setup, per-block accuracy sweep ×2 components, downstream ADE proxy, ONNX export report, MHA-decomposition parity, all 4 TRT engine builds + latency + fusion + per-layer profiles, synthesized mixed-precision recommendation + Orin/Thor estimates) | same folder (staged) + `pod1:/workspace/int8_bench/orin_int8_benchmark.json` | the pod copy is regenerable by re-running the scripts below |
| 3 | `bench_p1_accuracy.py` — PyTorch per-block accuracy sweep + downstream ADE proxy | same folder (staged) + `pod1:/workspace/int8_bench/` | no |
| 4 | `bench_p2_trt.py` — real-weight ONNX export (+ MHA decomposition) + TRT FP16/INT8 build + calibration + latency + per-layer profiling | same folder (staged) + pod1 | no |
| 5 | `bench_p3_synthesize.py` — mixed-precision recommendation + Orin/Thor scaling synthesis (CPU-only) | same folder (staged) + pod1 | no |
| — | real ckpt (`ckpt.pt`, 3.3 GB), venv (torch/tensorrt/onnx/onnxruntime/huggingface_hub), ONNX graphs (~695 MB), clean `tanitad` package copy | `pod1:/workspace/int8_bench/` | **pod-only** (too large to stage; ckpt is regenerable from HF `Sayood/tanitad-flagship-4b-speedjerk`, ONNX/venv regenerable by re-running the scripts) |

**Integration note:** self-contained research + benchmark intake, nothing needs merging into `stack/`. The one cross-stream fact worth carrying into `MODEL_REGISTRY.md`'s deployment section or a future registry refresh: **INT8 is not recommended for flagship-v1 on Ampere/Orin-class hardware** — FP16 stays the deployment precision; this reverses no prior claim (the 2026-07-22 intake never asserted INT8 was safe, it correctly deferred the question to this benchmark) but it does settle the open question that intake raised.

**Pod state:** `orin-int8` gpu_lock released; `nvidia-smi` confirms **0 MiB / 0 %** on pod1 at hand-off. Work directory `/workspace/int8_bench/` left in place (ckpt + venv + ONNX, ~5–6 GB) for reproducibility — not on the pod's disk-pressure-sensitive `/workspace/experiments/` path, and `flagship4b-v3enc-30k`'s checkpoints (the ones flagged DO NOT RECYCLE in the registry) were never touched.
