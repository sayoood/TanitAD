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
