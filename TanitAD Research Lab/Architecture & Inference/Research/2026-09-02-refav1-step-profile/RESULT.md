# RESULT — refav1 step profile: the 30-step operative rollout IS the step

**Owner:** Deploy & Optimization FlyWheel · **Date:** 2026-09-02 · **Instrument:** `stack/scripts/refa_v1_profile.py` · **Raw:** `raw/profile.json` (every arm), `raw/derived.json`, `raw/trace.json` (rollout fwd+bwd K=15, 1 step, 17 MB), `raw/trace_forward_nograd_bs1_fp32.json` (whole forward, 12 MB), `raw/top_kernels.md`, `raw/snapshot_manifest.json`, `raw/run1–7.log`, `raw/offline_attr.log`, `raw/tools/` (the table/derive/finalize scripts that generated every number in this file)
**Scope:** ⛔ **For the NEXT refav1 run. Nothing here is to be applied to the live Thor run; Thor was not contacted.** Every number is dev-box (RTX 4060, 8 GB, Windows) unless marked Thor; dev-box → Thor scaling is **not linear** (Thor's 20 SMs saturate at batch 8; its fp32 / TF32 / bf16 / fp64 rate ratios are not an RTX 4060's).

## Headline

1. **Top-1 cost: the 30-step operative token-field rollout, forward + backward — ~83 % of the step at batch 1 (MEASURED) and ~92 % at the live batch 8 (ESTIMATED from measured linear batch-scaling).** Everything else — tactical, strategic, brains, target path, instruments, optimizer, loader, H2D — is the remaining ~8–17 %.
2. **It is COMPUTE-bound, not launch-bound and not loader-bound**, by the pre-registered rules: GPU busy **98.1–98.3 %** on a profiled real fwd+bwd of the rollout (MEASURED, two runs) and **97.2 %** on the whole no-grad forward; the rollout's time grows **8.22×** from batch 1 → 8 (forward, MEASURED) and **2.08×** per batch doubling for fwd+bwd (MEASURED); a manual CUDA-graph replay of the rollout step is **1.01× (bs 1) / 0.99× (bs 8)**, bit-identical (MEASURED) — there is no launch overhead to remove on this GPU; the loader's live-state cost is **~0.22 s per batch of 8** (MEASURED, warm page cache), overlapped behind the previous step's backward.
3. **The lever the profile supports is precision on the rollout's GEMMs: bf16 autocast makes the rollout fwd+bwd 3.0× faster (MEASURED, bs 1, K=30: 542 ms bf16 vs 1,629 ms fp32-ESTIMATED; K=15: 278 vs 808 both MEASURED = 2.91×) and its forward 3.07× at batch 8 (MEASURED); TF32 alone gives 1.54× (MEASURED, forward, bs 8).** ESTIMATED Thor step if bf16 landed: **≈ 8 s/step (2.6×) → the 21,109-step epoch in ≈ 1.9 days instead of 5.05** — conditional on Thor's bf16:fp32 throughput ratio being ≥ the 4060's, which this box cannot measure (see §6). A bf16 run is a **numerics change** and needs its own control (§6b).
4. **The participation eigendecomposition is real but small: 73.5 ms per step in fp64 (MEASURED), ≈ 0.4 % of a 20.66 s step; the three `_chan_std` instruments add 3.0 ms.** Every-50-steps is correct hygiene, not a speed lever. The fp32 variant reads the identical value (24.1514) in 14.8 ms.
5. ⚠️ **The full live step does not fit an 8 GB card at ANY batch or precision** (the rollout keeps all 30 steps' activations for backward — truncated BPTT bounds the *gradient path*, not the *stored graph*: ~7 GB at batch 1 fp32, ESTIMATED ~56 GB at batch 8 on Thor's unified memory). The fraction table therefore comes from **isolated per-component fwd+bwd timings (wall-clock valid) plus an activation-offloaded full step (kernel-time valid, wall-clock not)**, cross-checked against each other. **Escalation (Master Mind):** the top lever (bf16, or TF32 as the conservative half) must be measured on Thor with the same script (10-min job, `tanitad-train` venv) before the next launch line is written; this box cannot supply the Thor ratio.

## 1. What was profiled (snapshot, data, config)

| item | value |
|---|---|
| code | HEAD `3f9b09df` snapshot (`git archive`), verified per file by LF-normalised blob hash: `refa_v1.py` **7bab551e**, `refa_v1_train.py` **62fc0999**, `refav1_loader.py` **a3a6f7be** (+10 load-bearing deps, all match — `raw/snapshot_manifest.json`). ⚠️ G:/mirror working trees moved DURING this work (another agent: `speed_channel`, `tmix_groups`, `ema_targets`, `ema_decay`, `ema_decay_end`; blobs `8264ded8` / `d94a1632` / `5b34f495`) — **all default OFF, so HEAD's default step is also the next run's unless a flag is switched on; EMA targets ON would add an un-profiled target-encoder pass per step.** The live Thor process runs the relaunch commit's files (INHERITED: Thor's files were shipped by md5, not verified here). |
| config | the live launch line: bs 8 (profile batch 1–2, see §2), `--lru 64`, `--target-space frozen`, `--detach-aux-targets`, `--bptt-truncate 15`, `--min-participation 0`, AdamW lr 3e-4 / adapter ×0.1 / wd 0.01, clip 1.0, **fp32 with TF32 OFF** (PyTorch default; the trainer sets nothing) — 175,164,468 params (operative 80.0 M, tactical 54.9 M, tactical policy 21.7 M, strategic policy 8.0 M, tac_pool 4.2 M, adapter 2.8 M). |
| data | 6 REAL DINOv3 episodes from the 2026-08-31 probe cache (`C:/Users/Admin/refav1_probe/dinov3cache`, fp16, T=101 each) cast to `float8_e4m3fn` with the shipping builder's own cast (`dinov3_fp8_encode_ship.py:195`) into `C:/Users/Admin/refav1_probe/profcache_fp8` — **66,193,268 bytes per episode, byte-identical in size to a genuinely shipped file (`ship/1697c853….pt`)**; v2eps from `refav1_probe/eps`; labels + nav = the v7.2 train blob (md5 `0ff902130ce76886b8a925eceed9e3a5`; 1/6 episodes join, 10/222 windows in-band — the loader's `-100` path exercised as live). ⇒ **the loader-decode/dequant fraction IS measured** (no synthetic fallback was needed). |
| box | RTX 4060 8 GB (24 SMs), i9-12900F, 31.8 GB RAM, NVMe ~2.0 GB/s cold read, torch 2.11.0+cu128, Windows 11. Allocator capped at 0.80 × VRAM = 6.40 GB. |
| N | warm-up 5 + measured 10 for the step arm; 2 + 5 (medians) for sweeps/components; 2 + 10 for instruments. All medians; `all_ms` lists in the raw. |

⚠️ **Two platform traps that would have produced confident wrong numbers, both caught by a control:**
* **Windows/WDDM over-commits VRAM into host RAM silently.** Without a cap PyTorch "allocated" **22.49 GiB on an 8 GiB card** before failing (first, uncapped attempt 20:32 — its log was overwritten by the capped rerun; the capped `fit` rows in `profile.json` show the cap binding at 6.34–6.40 GB); every timing in that state pages over PCIe. Fix: `torch.cuda.set_per_process_memory_fraction(0.80)` so over-commit is a real OOM (`--mem-fraction`). Same family as `df` / Thor `free` / cgroup `usage_in_bytes`: a counter that answers a different question.
* **An MSYS-style `/c/...` PYTHONPATH is silently not a Windows path** (`C:\c\Users\...`), so the venv's editable install routed `import tanitad` to **G:** despite the mirror being "first" on the path. The script pins `--stack-root` and **asserts** which tree was imported.

## 2. What fits (the `fit` arm) — nothing, and why that is itself a finding

| precision | bs 8 | bs 4 | bs 2 | bs 1 |
|---|---|---|---|---|
| fp32 (live) | OOM @ 6.34 GB | OOM @ 6.39 | OOM @ 6.37 | **OOM @ 6.38** |
| bf16 autocast | OOM @ 6.35 | OOM @ 6.38 | OOM @ 6.38 | **OOM @ 6.40** |

MEASURED, `raw/profile.json:fit`. The operative rollout stores every step's block activations for backward (`rollout()` keeps all 30 outputs in the graph; the `bptt_truncate` detach only cuts the *carried* state's gradient path). Isolated, the rollout fwd+bwd at bs 1 fp32 peaks at **2.29 / 3.55 / 4.81 / 6.06 GB for K = 5 / 10 / 15 / 20** (MEASURED) → **≈ 0.25 GB per rollout step per batch row**, i.e. ≈ 7.5 GB at K=30, bs 1, and **ESTIMATED ≈ 60 GB at bs 8 on Thor** (linear in batch, MEASURED bs 2 K=5: 3.96 GB vs 2.29). bf16 halves it (K=30 bs 1: **5.55 GB**, MEASURED).

## 3. The fraction table (dev box, bs 1, fp32 — the live precision)

Isolated fwd+bwd per component on a real batch (`raw/profile.json:components_fp32_bs1`), CUDA-event timed, medians of 5. The op-rollout at K=30 does not fit; it is fit on the K-ladder — **total = 54.7·K − 11 ms, R² 1.0000, n 4, window K ∈ [5, 20], evaluated 1.5× beyond the window** (per-step cost 52.3 / 53.7 / 53.8 / 54.1 ms at K = 5 / 10 / 15 / 20: flat). Its forward alone at K=30 is MEASURED (no-grad sweep): 480 ms, consistent with the fit's 16.4 ms/step forward.

| phase | fwd ms | bwd ms | total ms | share | class |
|---|---|---|---|---|---|
| **operative rollout K=30 (incl. `to_enc` + mse)** | 490 | 1,139 | **1,629** | **83.4 %** | ESTIMATED from the ladder (fwd MEASURED 480) |
| tactical: `_tac_field` init + rollout K=10 + mse | 22 | 54 | 77 | 3.9 % | MEASURED |
| instruments: participation eig (fp64) + 3× `_chan_std` | 76.5 | — | 76.5 | 3.9 % | MEASURED standalone (`eig` arm) |
| optimizer: AdamW step + `clip_grad_norm_` | 58 + 10 | — | 68 | 3.5 % | MEASURED |
| target path (std+adapter on the 30-step future 18, `_tac_field` ×30 15, subspace 0.4, ext 1) — grad-recorded, then detached | 34 | 0 | 34 | 1.7 % | MEASURED (no-grad variant: 34 ms, same) |
| H2D (`.to(device)`, ~95 MB pageable at bs 1) + loader (LRU-hot 100 ms / 8) | — | — | ≈ 25 | 1.3 % | MEASURED (loader) / run3 span (H2D) |
| brains (strategic + tactical policies, nav) | 8 | 12 | 19 | 1.0 % | MEASURED |
| strategic subspace + rollout (2 + 2 ext) + mse | 5.5 | 9.3 | 15 | 0.8 % | MEASURED |
| encode window (std + adapter on 4 frames) | 2 | 4 | 6 | 0.3 % | MEASURED |
| proposal + heads + label CE | 1 | 1 | 2 | 0.1 % | MEASURED |
| **sum** | | | **≈ 1,952** | 100 % | |

**Cross-check: §3b attributes the kernel time of the exact trainer step (activations paged to host RAM) to phases — its per-phase *kernel* time is admissible, its wall-clock is not — and agrees: rollout 84 % of kernel time.**

### 3a. Batch scaling — what changes at the live batch 8

| quantity | bs 1 | bs 2 | bs 4 | bs 8 | scaling | class |
|---|---|---|---|---|---|---|
| op rollout forward, no-grad, fp32 | 480 | 951 | 1,953 | 3,947 ms | **8.22×** | MEASURED (`sweep`) |
| op rollout fwd+bwd, K=5, fp32 | 261 | 543 | OOM | OOM | 2.08× per doubling | MEASURED (`components_fp32_bs2`) |
| whole forward, no-grad, fp32 | 633 | 1,169 | 2,244 | 4,493 ms | 7.10× | MEASURED |
| target path (adapter future + tac_field ×30) | 33 | 69 | | | 2.1× | MEASURED |
| tactical fwd+bwd | 77 | 125 | | | 1.6× | MEASURED |
| instruments, optimizer, brains | 76.5 / 68 / 19 | 76.5 / 69 / 20 | | | flat | MEASURED |

ESTIMATED dev-box step at bs 8 fp32: rollout 1,629 × 8.2 ≈ **13.4 s**; tactical ≈ 0.3 s; target path ≈ 0.27 s; loader (live state) 0.22 s + H2D ≈ 0.1 s; instruments 0.08 s; optimizer 0.07 s; rest ≈ 0.1 s → **≈ 14.5 s/step, rollout share ≈ 92 %.** Against Thor's MEASURED 20.66 s that puts Thor at ≈ 1.4× an RTX 4060 in fp32-without-TF32 — the plausible range for its CUDA-core fp32 rate, and consistent with the step being GEMM-bound on both.

### 3b. Kernel-time attribution of the full live step (run3, offloaded, bs 1 fp32)

Kernel time of the **exact trainer step** (loader → H2D → forward → backward → clip → AdamW; live loader; bs 1 fp32; activations paged to host so it fits) attributed to the forward phase that created each op — backward nodes are mapped through autograd sequence numbers (36,805 forward ops mapped); 3 profiled steps (`profile.json:step_fp32_bs1.profiler_trace_offline`; the 256 MB trace is kept locally). Total kernel time **1777 ms/step**; GPU-busy 72.5 % (NOT a launch-bound reading: the pageable offload traffic starves the GPU by construction — see the no-offload profiles below). 229 ms/step carried no launch record and is reassigned by kernel identity (cutlass `simt_sgemm` ×360/step = 30 steps × 6 blocks × 2 → rollout; the single fp64 `d884gemm` → participation covariance; `_foreach_*` → AdamW; `raw/offline_attr.log`).

| phase | fwd kernel ms | bwd kernel ms | reassigned ms | total ms | share of kernel time |
|---|---|---|---|---|---|
| operative rollout K=30 (fwd + bwd) | 436 | 891 | 167 | 1494 | 84.1 % |
| instruments: participation eig (fp64) | 47 | 0 | 52 | 99 | 5.6 % |
| optimizer: AdamW + clip_grad_norm | 51 | 0 | 10 | 60 | 3.4 % |
| tactical: tac_field init + rollout K=10 | 22 | 35 | 0 | 57 | 3.2 % |
| target path: std+adapter on future/ext, tac_field x30 (grad-recorded, detached) | 46 | 0 | 0 | 46 | 2.6 % |
| op loss: to_enc + mse | 7 | 4 | 0 | 12 | 0.7 % |
| encode window: std + adapter | 2 | 3 | 0 | 4 | 0.2 % |
| brains (strategic + tactical policies) | 1 | 1 | 0 | 2 | 0.1 % |
| forward glue (chan_std reductions, mean, stack, slices) | 0 | 1 | 0 | 2 | 0.1 % |
| strategic: subspace + rollout | 1 | 1 | 0 | 2 | 0.1 % |
| heads / proposal / label CE | 0 | 0 | 0 | 0 | 0.0 % |
| h2d / loader (kernels only) | 0 | 0 | 0 | 0 | 0.0 % |
| other: P/backward | 0 | 0 | 0 | 0 | 0.0 % |
| still unattributed | | | | 0 | 0.0 % |

**Every forward phase under the profiler (no grad, bs 1 fp32, `raw/trace_forward_nograd_bs1_fp32.json`, `profile.json:fwdprof_fp32_bs1_trace`).** Device time = kernels launched inside the marker (nested markers overlap: `P/encode` ⊃ `P/std_window` + `P/adapter_window`). `P/forward` 623 ms device span; kernel time 608 ms/step; **GPU-busy 97.2 %**; 3640 kernels/step, median 24 us.

| marker | device ms | CPU ms | calls | share of forward |
|---|---|---|---|---|
| `P/op_rollout` | 471.0 | 0.0 | 1 | 75.6 % |
| `P/participation_eig` | 80.3 | 0.0 | 1 | 12.9 % |
| `P/tac_rollout` | 17.2 | 0.0 | 1 | 2.8 % |
| `P/adapter_future` | 17.1 | 0.0 | 1 | 2.7 % |
| `P/tac_field_targets` | 15.4 | 0.0 | 30 | 2.5 % |
| `P/brains` | 8.6 | 0.0 | 1 | 1.4 % |
| `P/to_enc` | 4.2 | 0.0 | 1 | 0.7 % |
| `P/std_future` | 2.5 | 0.0 | 2 | 0.4 % |
| `P/encode` | 1.7 | 0.6 | 1 | 0.3 % |
| `P/adapter_window` | 1.6 | 0.0 | 1 | 0.3 % |
| `P/loss_mse` | 1.3 | 0.0 | 4 | 0.2 % |
| `P/str_rollout` | 1.1 | 0.0 | 1 | 0.2 % |
| `P/adapter_str_ext` | 1.1 | 0.0 | 1 | 0.2 % |
| `P/tac_field_init` | 0.5 | 0.0 | 1 | 0.1 % |
| `P/str_subspace` | 0.4 | 0.0 | 3 | 0.1 % |
| `P/participation_ratio` | 0.3 | 0.0 | 1 | 0.0 % |
| `P/heads` | 0.1 | 0.0 | 2 | 0.0 % |
| `P/std_window` | 0.1 | 0.0 | 1 | 0.0 % |

**Rollout fwd+bwd at K=15 under the profiler (bs 1 fp32, no offload; the 30-step rollout is two of these).** Kernel time **812 ms/step** (event-timed: 808 ms — profiler overhead is negligible), **GPU-busy 98.1 %** (1 step, `raw/trace.json`, 17.1 MB) and **98.3 %** (3 steps, trace kept locally at 53.9 MB); 5341 kernels/step, median kernel 17 us; 30 % of kernels are shorter than 10 us but they hold only 5.8 % of kernel time — the launch-bound signature is absent.

Top kernels of the rollout fwd+bwd (K=15, self device time per step, `raw/top_kernels.md`):

| # | kernel | calls/step | ms/step | % of kernel time |
|---|---|---|---|---|
| 1 | `ampere_sgemm_128x64_tn` | 391 | 190.7 | 23.5 |
| 2 | `ampere_sgemm_64x64_nt` | 285 | 145.7 | 17.9 |
| 3 | `_Z41fmha_cutlassB_f32_aligned_64x64_k128_sm80N22PyTorchMemEffAttention23Attentio` | 90 | 134.0 | 16.5 |
| 4 | `_ZN7cutlass7Kernel2I43cutlass_80_simt_sgemm_256x128_8x4_nn_align1EEvNT_6ParamsE` | 180 | 83.2 | 10.2 |
| 5 | `ampere_sgemm_128x64_nn` | 210 | 82.4 | 10.1 |
| 6 | `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_addIfEESt5arr` | 1776 | 56.1 | 6.9 |
| 7 | `_Z40fmha_cutlassF_f32_aligned_64x128_rf_sm80N22PyTorchMemEffAttention15Attention` | 90 | 49.7 | 6.1 |
| 8 | `ampere_sgemm_128x128_nt` | 105 | 16.8 | 2.1 |
| 9 | `_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_26GeluBackwardCUDAKernel` | 105 | 12.3 | 1.5 |
| 10 | `_ZN2at6native53_GLOBAL__N__87a90936_20_layer_norm_kernel_cu_3ff0b71f39layer_norm` | 195 | 6.9 | 0.8 |
| 11 | `_ZN2at6native13reduce_kernelILi128ELi4ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0` | 451 | 5.8 | 0.7 |
| 12 | `_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_22gpu_kernel_impl_nocastIZZZNS0` | 270 | 4.3 | 0.5 |
| 13 | `_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIP` | 272 | 3.7 | 0.4 |
| 14 | `Memcpy DtoD (Device -> Device)` | 270 | 3.1 | 0.4 |
| 15 | `_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_18GeluCUDAKernelImplERNS` | 105 | 2.8 | 0.3 |

GEMM + attention kernels in the top-20: **707 of 812 ms (87 %)**; the rest (residual adds, GELU/GELU-backward, LayerNorm, reductions, copies) is the ceiling `torch.compile`'s fusion could touch.

**How the full-step table was obtained, and what it cost.** The full fwd+bwd step fits no batch on 8 GB (§2). `torch.autograd.graph.save_on_cpu` with pinned memory fails at ~7 GB with a CUDA-runtime `out of memory` (`raw/run2.log`); pageable offload runs (9.89 s/step wall over 10 measured steps, `profile.json:step_fp32_bs1.wall`, wall-clock inadmissible by construction) but holds **18.7–22.7 GB of host working set** (≈ 3× a step's 7.5 GB of saved activations) and a 10-step profiler pass on top drove the host to 1.1 GB free / commit at the pagefile limit (`raw/run3.log`–`run5.log`). The 3-step pass exported its trace before the process was stopped; the table above is that trace, attributed offline with the shipped script's own `trace_stats()` (`raw/offline_attr.log`).

## 4. Why it is compute-bound (the pre-registered rules, applied)

| rule | measurement | verdict |
|---|---|---|
| loader-bound if live − pre-staged ≥ 20 % | loader live-state ≈ 0.22 s per batch of 8 (8 `torch.load` misses: 13.9 ms fp8 file + 8.1 ms v2ep per window, warm page cache; slice + fp8→fp32 dequant 5.9 ms/window) vs a ≥ 14.5 s step; NVMe cold reads 2.0–2.3 GB/s → worst case +0.4 s. The host is free during the backward (the only host syncs are the forward's four `float()` calls), so the loader overlaps the previous step's backward; only H2D (~0.1 s at bs 8, pageable) is exposed. | **NO** (≤ 1–3 % even if fully exposed; ESTIMATED exposure < 1 %) |
| launch-bound if GPU-busy < 70 % and bs 1→8 < 4× | GPU busy 98.1–98.3 % (rollout fwd+bwd), 97.2 % (whole forward); rollout 8.22× (fwd) / 2.08× per doubling (fwd+bwd); graph replay 1.01× / 0.99× | **NO** |
| compute-bound if GPU-busy ≥ 85 % and ≈ linear | 98.1–98.3 % and 8.22× | **YES** |

Where the compute goes (ESTIMATED from the architecture, to explain the measurement): per token per block 2·(3+1+8)·1024² ≈ 25.2 MFLOP of GEMM + 2.6 MFLOP of attention; × 640 tokens × 6 blocks × 30 steps ≈ 3.2 TFLOP forward at bs 1 (+ `mix`/`head`/`to_enc` ≈ 0.15) → ≈ 10 TFLOP fwd+bwd per step per batch row. The MEASURED 1,629 ms means ≈ 6 TFLOPS achieved on the 4060 in fp32 (its non-tensor-core peak is ~15) — GEMM-bound, which is exactly the regime where TF32/bf16 tensor cores pay and CUDA graphs do not.

## 5. The hidden costs the brief asked about

* **Participation eigendecomposition** (`refa_v1.py:1098-1104` → `spectral.covariance_eigs`): the covariance and `eigvalsh` run in **float64** on the GPU on the first 4,096 of 153,600 rows (bs 8) — **73.5 ms/step MEASURED** on the 4060 (fp64 = 1/64 rate here), CPU 64 ms, an fp32 variant 14.8 ms with the same value. Fixed cost, batch-independent: ≈ 0.4 % of 20.66 s. Not the hidden cost it might have been. ⚠️ It also returns a Python `float` → a **host sync every step**, as do the three `_chan_std` instruments (3.0 ms) — four syncs per step; harmless for throughput here because they sit after the rollout is enqueued.
* **The target path is grad-recorded for nothing**: `tgt = adapter(std(future))` builds a graph over [B,30,640,1024] whose every consumer is `.detach()`ed (`detach_aux_targets=True`, `target_space=frozen`). MEASURED: same speed with or without grad (17.8 vs 17.6 ms at bs 1), so **no time to win, but 0.22 GB of graph per batch row at peak** (1.45 vs 1.23 GB) — ~1.8 GB at bs 8. A `torch.no_grad()` there is a memory tidy-up for the next run, not a speed lever.
* **`_tac_field` on the targets is 30 separate cross-attention calls** (`refa_v1.py:1007`): 15 ms at bs 1, 33 ms at bs 2 — ≈ 1 % of the step; batchable into one call ([B·30, 64] queries) but not worth the diff by itself.
* **The loader's LRU is effectively always cold on Thor**: `--lru 64` over 4,713 episodes gives a per-window hit probability of 64/4,713 ≈ 1.4 %, so the live loader is the miss path measured above (8 file loads per batch, ≈ 0.8 GB of reads). It is cheap because the fp8 file is one tensor (`torch.load` 13.9 ms warm) — and hidden behind the backward.

## 6. Levers — what the profile supports and what it does not

| lever | supports | does NOT support / caveats |
|---|---|---|
| **(a) prefetch / dequant thread** | nothing to hide: loader ≈ 0.22 s (1.5 % at bs 8) and already overlapped; H2D ≈ 0.1 s exposed (pinned + `non_blocking` would hide it: ≈ 0.5 %) | — |
| **(b) bf16 autocast on the rollout** | **the top lever.** MEASURED here: rollout fwd+bwd 2.91× (K=15) / 3.0× (K=30 vs ladder); forward 3.07× at bs 8; whole-forward 2.58× at bs 8; halves activation memory (5.55 GB vs > 6.4 at K=30 bs 1). Full-step arithmetic at bs 8 (ESTIMATED): 14.5 s → ≈ 5.3 s on this box (2.7×) | (i) Thor's bf16 : fp32 ratio is NOT measured — Blackwell tensor cores vs its CUDA-core fp32 make ≥ 3× plausible, but the Thor lever-ladder (`products/P6-TanitDeploy/2026-08-23-thor-lever-ladder/`) measured TF32 at only +22 % under graphs on a launch-bound tiny tick (not transferable; different regime); (ii) **numerics**: the same ladder MEASURED TF32 perturbing every learned tensor (z_op 5.6e-2) — a bf16 arm is a *different run* and needs a 500-step loss/gnorm/participation control against fp32 before it is trusted; `GradScaler` is not needed for bf16 but the LayerNorm/attention paths stay fp32 under autocast, which is what the 3× already includes |
| **TF32 (the one-liner sub-lever)** | rollout forward 1.54× at bs 8, whole forward 1.53× (MEASURED); no dtype change; ESTIMATED Thor 20.66 → ≈ 14 s | same numerics caveat, smaller; Thor's own TF32 gain on a GEMM-bound step is ESTIMATED, not measured |
| **(c) CUDA graphs (manual `torch.cuda.CUDAGraph`)** | capture works and is bit-identical (max diff 0.0) | **no gain**: 1.01× (bs 1) / 0.99× (bs 8) fp32 — GPU busy 98 %, nothing to remove. Naive per-step capture under autocast is *slower* (0.79× / 0.48×) because the weight casts get baked into the replay. Thor's ARM host launches slower, but ≈ 8k kernels/step of ms-scale GEMMs hide it; only a Thor profile showing GPU-busy < 85 % would reopen this |
| **(d) `torch.compile` (Thor only)** | untestable here (memory note: Triton is installable on Windows but not in the tanitad venv; inductor's gains would be elementwise fusion) | on Thor it is blocked by a missing `python3-dev` (`Python.h`), a one-line repair the PI has not authorised (`thor-lever-ladder` H-DEPLOY-5); inductor's own log says Thor's 20 SMs are too few for `max_autotune_gemm`. Ceiling: the non-GEMM elementwise share of kernel time (see `raw/top_kernels.md`) — ESTIMATED 10–20 %, well below the precision lever |
| **(e) participation eig every 50 steps** | correct hygiene: 73.5 + 3.0 ms per step MEASURED | ≈ 0.4 % of the step — not a speed lever; the fp32 variant (14.8 ms, identical value) is the cheaper fix if per-step monitoring is wanted |

### 6a. ESTIMATED Thor s/step if the top lever landed

Thor MEASURED 20.66 s/step (bs 8, fp32, TF32 off). Rollout share at bs 8 ≈ 0.92 (ESTIMATED, §3a). If Thor's bf16 gain on the rollout equals this box's 3.0×: **20.66 × (1 − 0.92·(1 − 1/3.0)) ≈ 8.0 s/step (2.6×)** → 21,109 steps ≈ **1.9 days** (was 5.05). With TF32 only (1.54×): **≈ 14.0 s/step** (1.5×) → 3.4 days. Both ESTIMATED; the only admissible way to turn them into a number is to run `refa_v1_profile.py --arms sweep,components --k-ladder 15,30` on Thor in the `tanitad-train` venv (≈ 10 min, no training interference if scheduled between the live run's checkpoints — a PI/Master-Mind decision, since Thor is training).

### 6b. What the next launch line should carry (for the Master Mind to decide, not applied here)

1. `torch.autocast("cuda", dtype=torch.bfloat16)` around the forward (or `torch.set_float32_matmul_precision("high")` as the conservative half) — **with a pre-registered 500-step A/B against fp32 on loss, gnorm, `participation`, `tgt_std_*`** before either is adopted; the Thor ladder's TF32 finding says the learned tensors move.
2. Keep the participation monitor but at `log_every` cadence, or switch it to the fp32 variant.
3. `torch.no_grad()` around the target path (memory, not time).
4. Nothing on the loader.

## 7. Limits of this measurement

* Absolute seconds are the 4060's; Thor numbers are ESTIMATED with the arithmetic shown, and the bf16/TF32 ratios on Thor are unmeasured.
* The full step could not run within VRAM at any batch; the fraction table is a component sum (validated by the offloaded step's kernel attribution, §3b) and the bs-8 figure is a linear extrapolation on MEASURED scaling (8.22× forward, 2.08× per doubling fwd+bwd).
* The K=30 fp32 rollout cost is a 1.5×-beyond-window extrapolation of an R² = 1.0000 line (n 4); the K=30 forward is measured and agrees.
* Under the profiler the K=15 rollout fwd+bwd reads 812 ms of kernel time vs 808 ms event-timed — CUPTI overhead is negligible here; a first pass that counted Kineto's GPU-side copies of the `P/*` annotations as kernels read 1,031 ms and was corrected (`_ka_dump` now excludes them).
* The desktop compositor held the GPU at 25–55 % "utilisation" at idle clocks before the runs; under load the card boosted to 2.8 GHz and the medians' spread is < 3 %.
* Loader page cache was warm for the miss-cost numbers; cold NVMe reads were measured separately (2.0–2.3 GB/s) and bound the worst case.
* In-flight code edits (EMA targets, speed channel, tmix groups) are outside this profile.
