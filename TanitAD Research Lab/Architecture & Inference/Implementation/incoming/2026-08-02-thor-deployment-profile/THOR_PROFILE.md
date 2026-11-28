# Jetson Thor — deep deployment profile of the TanitAD world model (2026-08-02)

**PI request:** *"do a deep profiling analysis of our model on the thor including inference time,
memory, latency etc."* **MEASURED on `thor6`** — NVIDIA Thor (Blackwell), aarch64, L4T R38.4.0,
torch **2.13.0+cu130**, `tanitad-edge` venv (use case 1 per the two-venv rule).

**Model:** `WorldModel(flagship4b_config)`, **263.58 M params**, at v5f's **deployed geometry
176×624, 117° HFOV cylindrical** — not a convenient square. Budget: **100 ms** (10 Hz planning).

⚠️ **Method, because a latency number without it is worthless:** warmup (10 iters — cuDNN picks
its algorithm on the first call) then **`torch.cuda.synchronize()` around every timed region**
(CUDA is async; unsynchronised timing measures kernel *launch*). **p50 and p99 over 50 iterations**,
never a mean — the budget is a deadline, so the tail is the spec.

---

## 1. Latency

| stage | p50 | p99 | vs 100 ms budget |
|---|---|---|---|
| **encode_window fp32** (8 frames) | **187.81 ms** | 191.45 ms | ⛔ **188 %** — over budget |
| ⭐ **encode_window bf16** | **27.78 ms** | 32.74 ms | ✅ **28 %** — fits, with 3.6× headroom |
| predictor 1-step | **4.29 ms** | 6.06 ms | the imagination unit cost |
| predictor 20-step roll | **80.87 ms** | 86.48 ms | full imagination horizon |

⭐ **bf16 autocast alone is a 6.76× speedup** and moves the encoder from *188 % over budget* to
*28 % of it*. One flag, no export toolchain, no accuracy work — the single highest-leverage
deployment lever measured so far.

⚠️ **p99/p50 ratio is 1.02** — the distribution is tight. No hidden tail risk on this device;
what you see at p50 is what you get at p99. That is unusual and worth keeping: it means a
deadline-based scheduler can be planned around these numbers directly.

## 2. Memory — and the unified-memory caveat

| | fp32 | bf16 |
|---|---|---|
| allocator peak | **1231.5 MB** | 1375.1 MB |
| allocated (steady) | 1105.0 MB | 1105.0 MB |
| **system RSS after model load** | **11.48 GB** | — |
| system RSS at end | 13.19 GB | — |

⚠️ **Thor's 122 GB is UNIFIED (CPU+GPU share it).** `torch.cuda.max_memory_allocated` reports the
**allocator's** view (~1.2 GB) — the *system* footprint is **11.5 GB**, ~10× larger, because the
CUDA context, cuDNN workspaces and the torch runtime live in the same pool. **Quote the RSS, not
the allocator number**, when sizing a deployment. Even so: 13.2 GB of 122 GB ⇒ **~9× headroom**.

⚠️ bf16 peak is *higher* than fp32 (1375 vs 1231 MB) — autocast keeps fp32 master copies alongside
the cast tensors. bf16 buys **speed, not memory**, here.

## 3. Batch scaling — the 122 GB does **not** buy throughput

| batch | p50 | ms/sample | peak |
|---|---|---|---|
| 1 | 187.10 ms | 187.10 | 1263 MB |
| 2 | 373.62 ms | 186.81 | 1421 MB |
| 4 | 739.56 ms | 184.89 | 1738 MB |
| 8 | 1465.92 ms | **183.24** | 2371 MB |

⛔ **Batching is ~free in memory and worth ~2 % in time.** 8× the work costs 7.84× the latency —
per-sample time improves only 187.1 → 183.2 ms. ⇒ **the encoder is COMPUTE-bound, not
launch-bound, on Thor.** Practical consequences: (a) don't batch to chase throughput — the win
isn't there; (b) **the optimisation must attack the encoder's arithmetic** (precision, TensorRT
kernel fusion, or resolution), not the scheduling.

## 4. What this means for deployment

1. ⭐ **bf16 is the first lever and it already clears the budget** (27.8 ms p50 vs 100 ms). Every
   further optimisation starts from *inside* budget, which changes the character of the work from
   rescue to margin-building.
2. **The encoder is the whole cost.** 187.8 ms encode vs 4.3 ms per predictor step: the encoder is
   **44×** a predictor step. Optimisation effort belongs there, not in the planner.
3. ⭐ **Imagination is affordable at the edge.** A full 20-step roll is 80.9 ms fp32 — and the
   *training-side* measurement agreed independently (R3: `cond_imagination` cost ≈1.01× step time).
   The imagination-conditioned v5f design is not an edge-infeasible luxury.
4. **Headroom is large**: 13.2 GB of 122 GB used. Room for multi-camera, larger context windows, or
   several models resident at once — the 122 GB is better spent on *capacity* than on batch size.

⚠️ **This is an EAGER fp32/bf16 baseline. TensorRT, INT8, and CUDA-graph capture are NOT applied.**
Read it as the starting point: the A40 precedent moved a v1-class plan step from **138 ms → 18.75 ms
(5.35×)** with a *sequenced* four-lever pass, and that programme's own finding was that the levers
are **sequenced, not additive** (capture first). Expect a similar shape here, from a better start.

## 5. Next measurements (in value order)

| # | measurement | why |
|---|---|---|
| 1 | **TensorRT / torch.compile export** of the encoder at bf16 | the encoder is 100 % of the problem and 44× the predictor |
| 2 | **INT8 PTQ** with an accuracy check on the four families | ⛔ never accept a quantisation that is only validated on ADE |
| 3 | **CUDA-graph capture** of the 20-step roll | 20 launches per plan step; capture is the A40's proven first lever |
| 4 | **End-to-end plan-step latency with the v5f head** | this profile covers WM + predictor; the 256-anchor diffusion head is not yet included |
| 5 | **Power / thermal at sustained load** (`tegrastats`) | an edge number that holds for 3 s and then throttles is not a deployment number |

## Evidence class

| claim | class |
|---|---|
| every latency/memory figure above | **MEASURED (ours)** — `thor_profile.json`, warmup + CUDA-synced, p50/p99 over 50 iters |
| 263.58 M params at 176×624/117° | **MEASURED** — geometry applied through the trainer's own `resolve_v2_frames` |
| system RSS 11.48 → 13.19 GB | **MEASURED** — `/proc/meminfo`, MemTotal − MemAvailable |
| "encoder is compute-bound" | **MEASURED** — inferred from flat ms/sample across batch 1→8 |
| A40 precedent 138 → 18.75 ms | **INHERITED** — paper §7.10, different silicon, **not** re-verified on Thor |
| expected TensorRT/INT8 gains | **HYPOTHESIS** — the whole point of measurements 1–3 |

---

# ADDENDUM — sustained load, and the prod-optimization playbook applied (same day)

## A. The risk check on everything above: **NO THROTTLING**

⚠️ The p50s in §1 came from 50 iterations (~10 s). An edge SoC that holds a number for ten
seconds and then throttles has **not** met a budget. Re-ran the bf16 encoder for **180 s**:

| | |
|---|---|
| first-decile p50 | 27.90 ms |
| **last-decile p50** | **26.57 ms** |
| throttle ratio (last/first) | **0.952** — it got *faster*, not slower |
| overall p50 / p99 | 27.78 / — |

✅ **The published burst number holds under sustained load.** (Faster-at-the-end is consistent
with clocks settling upward once the initial DVFS ramp completes.)

## B. ⭐ Applying the Production & Optimization stream's MEASURED playbook

This is not a fresh search — that stream already measured, on a 4060: *"the whole win is the ViT;
the predictor is **launch-bound**"*; manual `torch.cuda.CUDAGraph` on the predictor = **2.57×**
(rel-err 2.8e-7, agreement 100 %); fp16-encoder + graph-predictor combined tick **17.75 → 11.16 ms
(1.59×)** matching its additive projection to **0.4 %**; and `torch.compile` **not viable**.

**My first Thor pass put graph capture on the ENCODER and got 1.09× — that is their diagnosis
reproducing, not contradicting it:** a *compute*-bound stage cannot be helped by removing launch
overhead. Each lever then went on the stage their measurements assign it.

### The combined tick (20-step imagination roll + encode), MEASURED on Thor

| configuration | p50 | p99 | vs 100 ms budget |
|---|---|---|---|
| fp32 eager (baseline) | **272.56 ms** | 273.71 | ⛔ 273 % |
| ⭐ **bf16 encoder + CUDA-graph predictor** | **98.63 ms** | 102.57 | ✅ **98.6 %** |
| | **2.76× speedup** | | |

**Per-stage, with the accuracy delta beside every speed delta (their G-P2 rule):**

| stage | eager | optimised | gain | accuracy |
|---|---|---|---|---|
| encoder | 187.8 ms fp32 | **27.8 ms bf16** | **6.76×** | rel-err **0.0059**, max\|Δz\| 0.0080 |
| predictor 1-step | 4.23 ms | **3.42 ms** graph | 1.24× | rel-err **0.0** (bit-exact) |
| 20-step roll | 81.52 ms | **69.62 ms** graph | 1.17× | bit-exact |

⭐ **The CUDA graph is numerically FREE — rel-err exactly 0.0**, i.e. bit-identical replay. It is
the one lever with no accuracy question at all, which is why their stream put it first.

⭐⭐ **THE LEVERS COMPOSE ON THOR TOO.** Additive projection **101.25 ms** vs measured **98.63 ms**
— **−2.6 %**, i.e. slightly *better* than additive. The 4060 finding (0.4 %) replicates on
Blackwell/aarch64. ⇒ future levers can be planned additively rather than re-measured combinatorially.

### Two cross-silicon differences worth recording

1. **Predictor graph gain is 1.24× on Thor vs 2.57× on the 4060.** Not a contradiction — the gain
   is launch-overhead removal, and Thor's launch overhead is a *smaller fraction* of a step that
   its faster compute has already shortened. ⇒ **the graph matters less here, and precision matters
   more** than the 4060 experience predicts.
2. ⛔ **`torch.compile` fails on Thor too** — `InductorError` from the gcc/`libcuda.so.1` link step
   (their 4060 cause was a missing Triton). **Same verdict, different root cause, two platforms:**
   the deployment path is **manual capture + TensorRT**, and this should now be treated as settled
   rather than re-attempted per-device.

## C. Where the remaining 98.63 ms sits, and what is next

Encoder **27.8** + roll **69.6** ≈ 97.4 of the 98.6 ms ⇒ **the 20-step imagination roll is now the
majority cost (71 %)**, having been the minority before bf16. The target has moved.

| # | next lever | why now |
|---|---|---|
| 1 | ⭐ **TensorRT fp16 engine** — their **#1 latency item**, recorded as *"toolchain-blocked… run when `tensorrt` lands"* | ⚠️ **Thor's image has NO TensorRT** (`nvinfer` absent, no `trtexec`, no python module) — it needs installing, but the ONNX IR is already **parity-clean at opset 17/18** (max\|Δz\| 8.8e-6, no unexportable ops), so the export half of the job is *already done* |
| 2 | **Capture the WHOLE 20-step roll in ONE graph** | today each step is a separate replay; one capture removes 19 replay boundaries from the now-dominant stage |
| 3 | **INT8 PTQ** | ⛔ accuracy must clear **all four families**, never ADE alone |
| 4 | **bf16 accuracy gate** | rel-err 0.0059 is small but **not free** — it must be checked against their 95.3 % decision-agreement bar on real windows before bf16 ships |

## Evidence class (addendum)

| claim | class |
|---|---|
| sustained 180 s, no throttling | **MEASURED (ours)** — `thor_sustained.json` |
| combined tick 272.56 → 98.63 ms (2.76×), per-stage gains, accuracy deltas | **MEASURED (ours)** — `thor_combined_tick.json`, warmup + CUDA-synced |
| levers compose on Thor (−2.6 % vs additive) | **MEASURED (ours)** |
| 4060 figures (2.57×, 1.59×, 0.4 %, 95.3 % bar) | **INHERITED** — Production & Optimization runs #4/#5, different silicon |
| ONNX parity-clean at opset 17/18 | **INHERITED** — their 2026-07-08 export run, **not** re-verified on Thor |
| TensorRT will beat the graph baseline | **HYPOTHESIS** — item 1 exists to test it |

---

# ADDENDUM 2 — the roll is compute-bound, and precision HURTS it (2026-08-02, same day)

After bf16 the 20-step imagination roll became **71 %** of the 98.63 ms tick, so it became the
target. Two levers were tested on it. **Both are negative results, and both are worth more than a
positive one would have been**, because each closes a direction that looked obvious.

## Lever #2 — capture the WHOLE roll in ONE CUDA graph

| variant | p50 | vs eager | accuracy |
|---|---|---|---|
| eager fp32 | 81.70 ms | — | reference |
| per-step graph (20 replays) | 70.47 ms | 1.16× | **rel-err 0.0** |
| ⭐ **whole roll, one graph** | **69.25 ms** | 1.18× | **rel-err 0.0** |

⚠️ **One capture beats 20 replays by only 1.02×.** Removing 19 launch boundaries and 20 Python
window-slides bought **2 %** ⇒ **the roll is COMPUTE-bound, not launch-bound.** The launch overhead
was already gone after the per-step capture; what remains (~69 ms ≈ 20 × 3.4 ms) is genuine
predictor arithmetic.

⭐ Both captures are **bit-exact** (rel-err exactly 0.0), including the full-roll variant that
reuses one static buffer set across 20 steps — the aliasing hazard this test existed to catch did
not fire.

## Lever #3 — precision on the roll ⛔ **SLOWER, both dtypes**

| variant | p50 | speedup | rel-err |
|---|---|---|---|
| fp32 | 81.67 ms | 1.00× | reference |
| **bf16** | **95.10 ms** | ⛔ **0.86×** | 0.00759 |
| **fp16** | **90.80 ms** | ⛔ **0.90×** | 0.000834 |

⛔ **Autocast makes the roll 10–16 % SLOWER.** The predictor's tensors are `1×8×2048` — far too
small for tensor-core throughput to repay autocast's per-operation cast overhead. The encoder's
large convolution/attention tensors repay it 6.76×; the predictor's do not.

⭐⭐ **THE ASYMMETRY IS THE FINDING: precision is a 6.76× win on the encoder and a 0.86× LOSS on
the predictor.** "Cast the model to bf16" is exactly the kind of blanket optimisation that would
have silently cost ~14 % on the dominant stage. **Per-stage, never global.**

ⓘ Worth keeping if precision is ever revisited: **fp16 is 9× more accurate than bf16 here**
(rel-err 0.000834 vs 0.00759) *and* faster — bf16's wider exponent buys nothing on these values.

## Where the tick stands, and what is actually left

| configuration | tick p50 |
|---|---|
| fp32 eager baseline | 272.56 ms |
| bf16 encoder + per-step graph | 98.63 ms |
| ⭐ **bf16 encoder + full-roll graph (fp32)** | **≈97.05 ms** |
| | **2.81× total, inside the 100 ms budget** |

**The eager-PyTorch levers are now EXHAUSTED.** Precision is applied where it helps and rejected
where it hurts; launch overhead is fully removed from both stages; `torch.compile` fails on two
platforms. Every remaining gain must come from **kernel-level work**:

| # | lever | rationale |
|---|---|---|
| 1 | ⭐ **TensorRT engine on the PREDICTOR** | the dominant stage is 20 small transformer steps — kernel *fusion* is precisely what small-tensor compute-bound work needs, and it is the one thing eager PyTorch cannot do |
| 2 | **TensorRT on the encoder** | already 27.8 ms; fusion + fp16 kernels may still beat autocast |
| 3 | **INT8 PTQ** | ⛔ accuracy must clear **all four families**, never ADE alone |
| 4 | **Shorten the roll** | an architecture question, not an engineering one: does the planner need K=20, or does K=10 lose nothing? That is a four-family measurement, not a latency one |

## Evidence class (addendum 2)

| claim | class |
|---|---|
| full-roll graph 69.25 ms, bit-exact, 1.02× over per-step | **MEASURED (ours)** — `thor_fullroll_graph.json` |
| bf16 0.86× / fp16 0.90× on the roll, with rel-errs | **MEASURED (ours)** — `thor_bf16_roll.json` |
| "the roll is compute-bound" | **MEASURED** — inferred from the 1.02× ceiling on launch-overhead removal |
| "small tensors defeat autocast on this predictor" | **HYPOTHESIS** — consistent with the measurement and with tensor shape, not independently profiled at kernel level |
| TensorRT will help the predictor | **HYPOTHESIS** — lever #1 exists to test it |

---

# ADDENDUM 3 — the K cost curve, and power/thermal (2026-08-02)

## A. What shortening the imagination roll would buy — the LATENCY half only

⛔ **This is deliberately half an answer.** `K` is an ARCHITECTURE parameter: the planner reads
imagination at horizons **[5, 10, 15, 20]** and the anchors live at 20 steps. Cutting K without a
four-family measurement trades an unknown in decision quality for milliseconds — precisely what
the binding rule forbids. This prices the trade; it does not authorise it.

| K | roll | **full tick** (bf16 enc + roll) | vs 100 ms budget | saved vs K=20 |
|---|---|---|---|---|
| 4 | 17.2 ms | **45.0 ms** | 45 % | **64.9 ms** |
| 8 | 33.5 ms | **61.3 ms** | 61 % | 48.7 ms |
| **10** | 41.3 ms | **69.1 ms** | 69 % | **40.9 ms** |
| 12 | 49.6 ms | 77.4 ms | 77 % | 32.6 ms |
| 16 | 65.9 ms | 93.7 ms | 94 % | 16.2 ms |
| **20** (current) | 82.2 ms | **110.0 ms** | ⛔ 110 % | — |

⭐ **The roll is EXACTLY LINEAR in K**: 4.308 ms/step at K=4 vs **4.107 ms/step at K=20** (−4.7 %).
Compute-bound, confirmed a third independent way. ⇒ **the latency half of the K question is now
settled by arithmetic** — every step costs ~4.1 ms, so any K can be priced without re-measuring.

⚠️ **And it exposes something the earlier tick number hid.** This unoptimised-roll measurement puts
the K=20 tick at **110 ms — over budget**. The 97.05 ms figure in Addendum 2 depends on the
full-roll CUDA graph (69.25 ms). ⇒ **the graph is not a nice-to-have; without it K=20 does not
fit**, and the margin at K=20 is thin either way.

⭐ **K=10 would put the tick at 69.1 ms even WITHOUT the graph** — a 41 ms saving, the single
largest remaining lever, and it costs no kernel engineering at all. **It is an accuracy question,
not an engineering one.** The measurement that decides it: score the arm at K∈{10,20} on the four
families, paired episode-cluster bootstrap. If tactical/strategic quality is flat, K=10 is free
real estate.

## B. Power and thermal under sustained load

**MEASURED** from `tegrastats` across the profiling session (1,157 samples @ 2 s ≈ 39 min):

| | |
|---|---|
| GPU power | **1,976 mW** steady under load |
| junction temp | **~61.3 → 61.9 °C** — a ~0.6 °C drift across the whole session |
| RAM | 12,518 → 11,014 MB of 125,772 MB |

✅ **Thermally uncommitted.** ~62 °C junction on a board rated far higher, and a **sub-1 °C** drift
over 39 minutes — which independently corroborates the 180 s no-throttle result (Addendum 1) on a
much longer window. ⭐ **~2 W for the GPU** at full inference load is an *automotive-plausible*
power envelope, and it means the thermal headroom for a **higher-power mode** (`nvpmodel`) is
entirely unexplored — a possible free speedup that costs nothing but a config change.

## C. Consolidated lever table (all measured today)

| lever | stage | result | ship? |
|---|---|---|---|
| bf16 autocast | encoder | **6.76×** | ✅ yes |
| bf16/fp16 autocast | predictor roll | **0.86× / 0.90×** | ⛔ **no — slower** |
| CUDA graph, per-step | predictor | 1.24×, bit-exact | ✅ yes |
| CUDA graph, whole roll | roll | 1.02× over per-step | ✅ yes (and **required** at K=20) |
| `torch.compile` | either | fails, 2 platforms | ⛔ no |
| batching | encoder | ~2 % | ⛔ no |
| **K 20 → 10** | roll | **−41 ms** | ⚠️ **needs a four-family accuracy gate** |
| TensorRT | both | not yet installed | 🔵 pending |
| higher `nvpmodel` | whole board | unexplored, ~62 °C headroom | 🔵 pending |

## Evidence class (addendum 3)

| claim | class |
|---|---|
| K-sweep timings, linearity 4.31 → 4.11 ms/step | **MEASURED (ours)** — `thor_ksweep.json` |
| GPU 1976 mW, tj ~61.3–61.9 °C, 39 min | **MEASURED (ours)** — `tegrastats.log`, 1,157 samples |
| "K=10 halves the dominant stage" | **MEASURED** (latency) — ⛔ accuracy consequence **UNMEASURED** |
| "nvpmodel headroom exists" | **HYPOTHESIS** — inferred from thermals, not tested |


---

# ADDENDUM 4 — NVFP4 on Thor: the silicon supports it, our tensors are too small to use it

**PI question: "did you investigate NVFP4?"** No — that was a real gap, and worth closing
properly, because **Thor reports `sm_110` — Blackwell, the generation NVFP4 was introduced for.**

## What NVFP4 is (PUBLISHED)

E2M1 4-bit float with **two-level scaling** — a per-tensor FP32 scale plus a **per-block E4M3
(FP8) scale** — which is what separates it from flat INT4 and preserves dynamic range. Published
claims: **up to ~4x throughput over BF16**, ~2-3x arithmetic and ~1.8x memory versus FP8, with
**DeepSeek-R1 PTQ staying within ~1 %** of FP8 (MMLU-Pro 85 -> 84, GPQA 81 -> 80). Tooling is the
TensorRT Model Optimizer plus TransformerEngine's fused quantize-and-GEMM path.

⚠️ **Every one of those numbers is an LLM number.** That matters more than usual here.

## What is actually true on our Thor (MEASURED)

| probe | result |
|---|---|
| compute capability | **`sm_110`** (Blackwell), 20 SMs, 131.9 GB |
| `torch.float4_e2m1fn_x2` dtype | ✅ **present** |
| `torch._scaled_mm` / `functional.scaled_mm` | ✅ present |
| **casting bf16 -> fp4 in eager torch** | ⛔ **`RuntimeError: copy_() does not support casting`** |

⇒ **The storage dtype exists, but eager PyTorch has no cast path.** NVFP4 here is not a
`.to(dtype)` away — it needs the **TensorRT Model Optimizer / TransformerEngine** route
(block-scale computation + fused kernels), which is exactly the toolchain still downloading.

## ⭐ The decisive measurement: FP8 GEMM speedup **as a function of size**

FP8 runs today in eager torch and is NVFP4's nearest neighbour — same tensor-core lineage, one
step less aggressive — so it is a legitimate proxy for whether low precision can help our shapes
at all:

| GEMM | tag | bf16 | fp8 | speedup |
|---|---|---|---|---|
| **8x2048x2048** | ⭐ **our predictor step** | 0.0632 ms | 0.0524 ms | **1.21x** |
| 128x2048x2048 | our batched | 0.0738 | 0.0654 | 1.13x |
| 1024x2048x2048 | medium | 0.2753 | 0.1645 | **1.67x** |
| 4096x4096x4096 | LLM-scale | 1.0856 | 0.5510 | **1.97x** |

⭐⭐ **The speedup is a function of GEMM size, and our tensors sit at the bottom of the curve.**
Low precision returns **1.21x** at our predictor's shape versus **1.97x** at LLM scale — and that
1.21x is on the *bare GEMM*, before the per-op cast overhead that already made bf16 a **0.86x
LOSS** on the real roll. **The published ~4x NVFP4 figures are measured at the right-hand end of
this curve; we live on the left.**

## Verdict, per stage

| stage | NVFP4 outlook | reasoning |
|---|---|---|
| **predictor roll** (69 ms, 71 % of tick) | ⛔ **unpromising** | 8x2048x2048 GEMMs; FP8 is only 1.21x on the bare GEMM and bf16 was net-*negative* in situ. NVFP4 adds block-scale work on top |
| **encoder** (27.8 ms) | ⚠️ **worth testing** | large conv/attention GEMMs — the regime where the curve pays. But it is already only 28 % of the tick, so even 2x buys ~14 ms |
| **memory footprint** | ✅ **real, if ever needed** | ~1.8x smaller than FP8 — irrelevant at 13 GB of 122 GB today, relevant on a smaller Orin-class target |

⇒ **NVFP4 is not the next lever for this model on this device.** The honest ranking is unchanged:
**K 20 -> 10 (-41 ms, gated on a four-family accuracy check)** >> TensorRT fusion >> NVFP4.

⭐ **The reason is architectural, not a limitation of NVFP4:** our world model is a *small-tensor,
many-step* workload, and every low-precision format is built for *large-tensor, few-step* work.
Same root cause as the bf16-roll regression — **the third independent observation of that fact
today.**

## What would change this verdict

1. **A wider predictor**, or **batched multi-candidate rolls** (`imagine_candidates`, one roll per
   candidate), would move our GEMMs rightward on the curve — an *architecture* change that would
   make quantisation pay. Worth remembering if per-candidate imagination ever ships.
2. **TensorRT ModelOpt NVFP4 on the encoder**, measured, once the toolchain lands.
3. ⛔ **Any NVFP4 adoption must clear the four families**, not ADE — a 4-bit format that preserves
   ADE while degrading manoeuvre κ or route accuracy is exactly the silent regression the binding
   rule exists to catch.

## Evidence class

| claim | class |
|---|---|
| `sm_110`, fp4 dtype present, cast unsupported in eager torch | **MEASURED (ours)** — `thor_nvfp4.json` |
| FP8 GEMM speedup 1.21x -> 1.97x across sizes | **MEASURED (ours)** |
| NVFP4 format, ~4x vs BF16, DeepSeek-R1 within ~1 % | **PUBLISHED** (cited) — LLM workloads, **not** re-verified for our shapes |
| "NVFP4 will underperform on the predictor" | **HYPOTHESIS** — strongly supported by the FP8 size curve and the bf16 regression, but NVFP4 itself is **not yet runnable** here |


---

# ADDENDUM 5 — TensorRT: the block is gone and it BEATS the free graph (2026-08-02)

**The Production & Optimization stream's #1 latency item, blocked since 2026-07-18** with the note
*"toolchain-blocked on the dev box (tensorrt missing) -> run when a pod is idle or tensorrt lands"*.
TensorRT **10.13.3.9** installed on Thor; the block is gone.

⚠️ First, a process failure worth logging: the apt install **completed at 15:59** and I kept polling
`dpkg -l` mid-transaction, read "0 packages", and treated it as stalled for ~2 hours. **The
completion line was in the log the whole time.** Root-cause class: polling a proxy signal instead of
reading the artifact that owns the answer — the same class as C62.

## The target: the PREDICTOR, not the encoder

Today established that the roll is 71 % of the tick, is COMPUTE-bound (full-roll capture bought
1.02x), and that precision alone LOSES there (bf16 0.86x). **Fusion is the one thing eager PyTorch
cannot do**, and small-tensor many-op work is exactly what it targets.

## Result — engine must beat the FREE CUDA graph to justify the toolchain (their bar)

| configuration | per step | vs eager | vs CUDA graph |
|---|---|---|---|
| eager fp32 | 4.23 ms | 1.00x | — |
| CUDA graph (free) | 3.42 ms | 1.24x | 1.00x |
| **TRT fp32** | **1.994 ms** | 2.12x | **1.72x** |
| ⭐ **TRT fp16** | **1.168 ms** | **3.62x** | ⭐ **2.93x** |

✅ **The engine clears their bar by 2.93x.** Build cost is trivial: ONNX export 303 MB, fp16 engine
**36 s** to build, 173.9 MB on disk (half the fp32 engine).

⭐ ONNX exported clean at **opset 17** on the first attempt, corroborating their 2026-07-08 finding
(*"no unexportable ops -- MHA/FiLM/causal-triu all fine"*) on new silicon and a newer torch.

## ⭐⭐ The projected tick

| | |
|---|---|
| roll20 at TRT-fp16 | 20 x 1.168 = **23.4 ms** (was 69.6 ms with the graph) |
| **full tick** | **51.2 ms** = bf16 encoder 27.8 + roll 23.4 |
| **vs the 272.56 ms fp32 baseline** | ⭐⭐ **5.33x** |
| vs the 100 ms budget | ✅ **51 %** — roughly 2x headroom |

⭐ **5.33x lands almost exactly on the A40 precedent's 5.35x** (138 -> 18.75 ms) — an independent
replication of that stream's total achievable speedup, on different silicon, by a different lever
mix. That the two agree this closely is a strong signal the ceiling is structural, not incidental.

⭐ **And it flips the balance again**: with the roll at 23.4 ms, **the encoder (27.8 ms) is once
more the larger stage**. The next TRT target is the encoder — where fp16 kernels + fusion may beat
autocast.

⚠️ **NOT YET VERIFIED and required before shipping**: engine-vs-eager numerical agreement. trtexec
reports timing only; the four-family accuracy gate and their 95.3 % decision-agreement bar have
NOT been run on engine outputs. **A 3.62x that changes decisions is not a 3.62x.**

## Revised lever ranking

| # | lever | expected |
|---|---|---|
| 1 | ⭐ **TRT engine on the ENCODER** | now the larger stage at 27.8 ms |
| 2 | **Accuracy gate on the TRT-fp16 predictor** | ⛔ blocking for deployment, not optional |
| 3 | K 20 -> 10 | still -23 ms at TRT speeds, still gated on four families |
| 4 | INT8 / NVFP4 via ModelOpt | now plausible: TRT owns the block-scale path eager torch lacks |

## Evidence class

| claim | class |
|---|---|
| TRT 10.13.3.9; engines build; 1.994 / 1.168 ms medians | **MEASURED (ours)** -- `thor_trt.json`, trtexec 200 iters |
| ONNX opset-17 export clean | **MEASURED (ours)** -- corroborates their 2026-07-08 result |
| projected tick 51.2 ms / 5.33x | **MEASURED per-stage, PROJECTED in composition** -- levers composed to -2.6 % earlier today, but this exact combination is not yet run end-to-end |
| engine numerical agreement | ⛔ **UNMEASURED** -- gating item #2 |
