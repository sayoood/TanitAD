# 2026-07-10 — INT8 weight-quant curve (localizes the sensitivity to the readout) + clean-GPU latency + windowing fail-loud (review #3)

Production & Optimization, Saturday run #3. One measured optimization experiment (backlog
**P1.6** INT8/FP8 curves + **P1.4b** clean-GPU latency) and one compliance review (**#3**:
the data-window index, backlog P1.8). Every optimization claim reports the accuracy delta next
to the speed/efficiency delta (G-P2). Hardware: local **RTX 4060** (declared Orin latency proxy,
I8), batch 1, **GPU idle/exclusive**, ~5 min wall-clock, **$0** (no cloud spend). Checkpoint:
`ckpt_full.pt` (step-6500), 64 real comma2k19 windows — the SAME windows as the 2026-07-09
fp16/bf16 run, so the numbers compose.

---

## 1. INT8 weight-only quantization curve — the ViT tower is *not* the red line; the readout is

**Method** (`Implementation/int8_quant/int8_quant_curve.py`, results
`int8_quant_step6500.json`). Per-output-channel **symmetric int8 fake-quant** of each Linear/Conv
weight (`scale = max|w_row| / 127`), activations kept fp32 → isolates the **weight-quant** accuracy
term so it composes with the separately-measured fp16 **activation** term. This is exactly the
rounding NVIDIA ModelOpt applies before TensorRT fuses the QDQ nodes — the honest precursor to a
real INT8 engine (TRT/ModelOpt toolchain still absent on the dev box). Accuracy is scored in the
**decision space** (per the 2026-07-09 precedent that cosine ≈ 1.0 is too coarse): one FIXED fp64
ridge probe decodes a K=9 sustained-steer imagine-and-select fan; we report selection-agreement %,
decoded-waypoint shift in **metres**, and encoder/predictor rel-err vs the fp32 reference.
INT8 **latency** is NOT claimed (needs fused int8 TRT kernels); the measurable INT8 efficiency
delta today is the **weight-memory footprint** (4× on the quantized subtree).

**Result (accuracy delta ‖ efficiency delta):**

| Group (weights → int8) | selection agreement | flips /64 | wp-shift mean / max | enc rel-err | weight saved | verdict |
|---|---|---|---|---|---|---|
| `int8_encoder` (ViT tower only) | **98.4 %** | 1 | **1.6 cm** / 12.2 cm | 7.0e-4 | **228 MB** (4×) | **SAFE** |
| `int8_predictor` (op + tactical) | **95.3 %** | 3 | 10.9 cm / 36.8 cm | 6.0e-4 | **324 MB** (4×) | **SAFE** (= fp16 bar) |
| `int8_heads` (readout+preds+inv_dyn+imag) | **48.4 %** | 33 | **1.67 m** / 1.95 m | 1.67e-2 | 390 MB (4×) | **UNSAFE** |
| `int8_all` (whole model) | **48.4 %** | 33 | 1.67 m / 1.96 m | 1.67e-2 | 619 MB (4×) | **UNSAFE** |

All arms finite (no overflow — precision-limited, as with fp16/bf16). Total fp32 linear/conv weight
footprint = **824.9 MB**.

**The headline — sensitivity localizes to the readout, not the ViT.**
`int8_all` is *identical* to `int8_heads` (48.4 %, 33 flips) while `int8_encoder` alone is 98.4 %
and `int8_predictor` alone is 95.3 %. So the whole-model collapse comes **entirely from the
heads-minus-predictor set {readout, inv_dyn, imagination}**. The tell: the *encoder-state* rel-err
jumps from 7.0e-4 (encoder-only) to **1.67e-2** (heads) — and the only state-*producing* module in
"heads" is the **readout** that projects the ViT features to the 2048-d planning state everything
conditions on. Rounding the readout weights shifts that state globally, so the decoded fan moves at
every window (33/64 flips, 1.67 m). Conclusion: **the readout / state-projection is the INT8
weight red line; the ViT tower and the predictor are INT8-safe.**

**This refines (does not just confirm) the KB "ViT INT8 trap" thesis.** The literature rule "keep
the vision tower FP16, quantize heads first" is a rule about **activation** quantization (ViT
activations have heavy-tailed outliers that INT8 activation-scaling clips). It is **the wrong
prior for weight-only quant**: here the ViT *weights* tolerate INT8 best (98.4 %, 1.6 cm) and the
*heads* (specifically the readout) are the sensitive part. **Falsifier verdict:** the "quantize
heads first" heuristic is REFUTED for weight-quant on this model; the per-module curve overrides
the heuristic (G-P1: measured, not assumed).

**Deployment rule (updated).** INT8-quantize the **ViT encoder + predictor** weights, keep the
**readout (+ inv_dyn / imagination) ≥ fp16**. Efficiency delta: mixed INT8(encoder+predictor)
reclaims **228 + 324 = 552 MB** = 67 % of the 824.9 MB weight footprint at 4×, while the two
quantized subtrees each stay ≥ 95 % decision-agreement (accuracy delta reported alongside, G-P2).
Compose with fp16 activations (95.3 % safe, 2026-07-09) for the full mixed-engine target.

**Open falsifier for the real engine (P1.4a).** fp16-activations and int8-weights are each ~5 %
individually; their **joint** flip rate is untested. Pre-registered acceptance bar for the eventual
ModelOpt/TRT engine: fp16-act + int8-weight(encoder+predictor) must hold **≥ 95 % agreement / ≤ ~4 cm
wp-shift** on these 64 windows, with the readout kept ≥ fp16. If the two error terms compound below
that, drop the predictor to fp16 too and INT8 only the encoder (98.4 %, 228 MB — the safest single win).

---

## 2. Clean-GPU latency — closes P1.4b (the 2026-07-09 contention caveat)

Measured on the **idle/exclusive** 4060 (no CarlaUE4/python resident), batch 1, strict numerics.

| precision | decision tick p50 | Hz | speedup vs fp32 | peak VRAM |
|---|---|---|---|---|
| fp32 | **15.76 ms** | 63.5 | 1.00× | 1.10 GB |
| fp16 | **13.40 ms** | 74.6 | **1.18×** | 1.65 GB† |

**Falsifier verdict — CONFIRMED:** clean fp32 tick **15.76 ms ≈ the 15.07 ms baseline** (2026-07-08,
also exclusive GPU) → the **33.5 ms** read on 2026-07-09 was **GPU contention**, exactly as flagged.
The operative-rate requirement (10–20 Hz) is met with **3–4× headroom** before any TensorRT/quant.
Accuracy delta for the fp16 arm: agreement **95.3 %** (2026-07-09) — fp16 is decision-safe (G-P2).
† **VRAM per-precision withheld as a clean number:** the fp16 arm was measured with the fp32
reference model still resident (needed for the INT8 sweep), so its 1.65 GB **double-counts** the
fp32 weights — the standalone fp16 footprint is *lower* than fp32, not higher. Same double-count
caveat as 2026-07-09; a clean per-precision VRAM delta still needs a single-model-resident run.

---

## 3. Compliance review #3 — silent short-episode drop → fail-loud (intake shipped)

Reviewed the shared window-index construction (data cluster, continuing review #1). **Finding
(silent-wrong/no-data class, F-5/F-6/F-7 ops-fragility):** all three window datasets —
`stack/tanitad/data/_contract.py:120-121`, `toy_driving.py:131-132`, `comma2k19.py:278-279` —
build `t_max = T - window - max_horizon; range(t_max)`. An episode with `T < window+max_horizon+1`
contributes **zero windows and is silently dropped** (no counter/warn/log; `comma2k19` even wraps
`max(0,t_max)`, guarding the negative-range but keeping the drop just as silent). Two real failure
modes: (a) a `window`/`max_horizon`/corpus change silently shrinks and long-episode-biases the train
set with **no signal**; (b) all-too-short → `len(ds)==0` → the trainer **spins on `StopIteration`**
or crashes deep in training with a cause-hiding message.

**Fix (intake `2026-07-10-contract-windowing-failloud/`):** `build_window_index` counts drops,
**warns** when any episode is dropped, and **raises a clear `ValueError`** if the index would be
empty — naming `window`, `max_horizon`, the required min length, and the lengths seen. Non-dropped
behaviour is **byte-for-byte identical** (the conservative `range(T-window-max_horizon)` convention
is preserved for parity across all three datasets). **10 tests pass standalone** (torch-only, no
`tanitad`/CUDA/data; G-E). Proposed target: a shared `stack/tanitad/data/_windowing.py` used by all
three sites. INTAKE.md has the full risk/rollback. (Note: I fixed a bug in the prior draft's own
boundary test — it expected a *sole* too-short episode to return an empty index, but the fail-loud
design correctly *raises*; the test now exercises the drop with a surviving episode.)

---

## Ledger / gate impact

- **G-H (measured experiment):** ✅ INT8 curve + clean latency, decision-grade numbers, falsifiers
  ruled on (heads-first heuristic refuted; contention caveat confirmed).
- **G-P1 / G-P2:** ✅ review names real file:lines + ships 10 passing tests; every speed/efficiency
  delta carries its accuracy delta (agreement %, wp-shift metres).
- **No hypothesis status change** (G-D N/A) — production-hardening + efficiency inputs (CNCE), not a
  capability claim. LEADERBOARD efficiency block gets the INT8 weight-footprint row + clean latency.
- **Boundary:** nothing written to `stack/` directly; the fix is an intake package (D-011).

## Sources (G-A)

- `Implementation/int8_quant/int8_quant_curve.py`, `int8_quant_step6500.json`, `run.log` (this run).
- `Implementation/half_precision/half_precision_step6500.json` (fp16/bf16 accuracy, 2026-07-09).
- `Implementation/incoming/2026-07-10-contract-windowing-failloud/` (intake + tests).
- Stack sites: `stack/tanitad/data/_contract.py:120`, `toy_driving.py:131`, `comma2k19.py:278`
  (grep-confirmed this run).
