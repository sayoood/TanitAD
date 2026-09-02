# SPEC — refav1 step profile: where do the 20.66 s go?

**Owner:** Deploy & Optimization FlyWheel · **Date:** 2026-09-02 · **Status:** pre-registered before any number was computed.
**Scope:** the NEXT refav1 run. Nothing here is applied to the live Thor run; Thor is never contacted.

## Question

The live refav1 arm trains on Thor at **20.66 s/step** (MEASURED by the Master Mind from the running
process; bs 8, `--lru 64`, `--bptt-truncate 15`, `--target-space frozen`, `--detach-aux-targets`), one
epoch = 21,109 steps = 5.05 days. Nobody has measured where a step's time goes. This package measures the
**fraction** of a training step spent in each phase on the dev box (RTX 4060, 8 GB) and classifies the
step as loader-bound, launch-bound or compute-bound.

## What is measured (instrument: `stack/scripts/refa_v1_profile.py`)

The exact trainer step (`refa_v1_train.py:325-362`: loader `batch()` → `.to(device)` → `forward` (incl.
loss assembly and instruments) → `zero_grad` → `backward` → `clip_grad_norm_` → `AdamW.step`), on the
model/trainer/loader **as they are** (snapshot blob hashes recorded in `RESULT.md`; no edit to those files).
Markers are `torch.profiler.record_function` ranges + CUDA events placed by instance-level wrappers in the
profiler script only.

Data: 6 real DINOv3 episodes from the 2026-08-31 probe cache (`C:/Users/Admin/refav1_probe/dinov3cache`,
fp16), cast to `float8_e4m3fn` with the shipping builder's own cast (`dinov3_fp8_encode_ship.py:195`) — the
byte-format the live loader opens. Labels/nav: the v7.2 train blob (1 of the 6 episodes joins).

Arms: `fit` (max batch per precision) · `step` (wall-clock per phase + profiler kernel time, three loader
states) · `sweep` (no-grad forward vs batch 1/2/4/8 and precision fp32/tf32/bf16) · `components`
(isolated fwd+bwd per component, K-ladder for the operative rollout) · `eig` (participation
eigendecomposition + per-channel std instruments standalone) · `graph` (CUDA-graph replay of one operative
step vs eager) · `loader` (LRU-hot / LRU-cold / raw miss cost).

## Pre-stated decision rules

| verdict | rule (all MEASURED on the dev box) |
|---|---|
| **loader-bound** | wall(live loader) − wall(pre-staged batch) ≥ 20 % of wall(live loader) |
| **launch-bound** | GPU-busy fraction (union of kernel time / wall) < 70 % during measured steps, AND the operative rollout's no-grad time grows < 4× from batch 1 to batch 8 (sub-linear = latency-bound) |
| **compute-bound** | GPU-busy ≥ 85 % AND time grows ≈ linearly with batch (≥ 6× from 1 to 8) |
| top-1 cost | the phase with the largest share of **kernel time** (backward attributed to its forward phase via the component arm; the trace attribution is a cross-check) |

Between 70 % and 85 % GPU-busy the verdict is "mixed" and both levers are reported with their measured
headroom rather than a single label.

## Lever criteria (what the profile can and cannot support)

(a) **prefetch/dequant thread** — supported iff the loader's *exposed* time (live − pre-staged) is > 5 % of
the step; the LRU-cold arm is the live state (hit-rate 64/4713 ≈ 1.4 %).
(b) **bf16 autocast** — supported iff the fp32→bf16 no-grad rollout speed-up is > 1.5× on this GPU AND the
GEMM kernels dominate the top-20; Thor's tensor-core ratio differs, so the Thor number is ESTIMATED.
(c) **CUDA graphs** — supported iff (graph replay vs eager) removes > 10 % of the operative step's forward
wall AND the GPU-busy fraction is < 85 %; forward-only capture is what is measured here.
(d) **torch.compile** — Thor-only option; NOT measurable on this box (no Triton on Windows). Reported as
untested.
(e) **participation eig every 50 steps** — supported iff the standalone eig + `_chan_std` cost is > 1 % of
the step.

## Known limits (stated up front)

* Dev-box→Thor scaling is **not linear**: Thor's 20 SMs saturate at batch 8 and its fp32/bf16/fp64 rate
  ratios differ from an RTX 4060. Every Thor s/step in `RESULT.md` is ESTIMATED with the reasoning shown.
* The full step at bs 8 does not fit in 8 GB; the profile batch is whatever `fit` reports, and the
  batch-scaling sweep is what bridges it.
* `torch.compile` is untestable here (Windows, no Triton).
* Absolute seconds on this box are not Thor's; fractions are the deliverable.
