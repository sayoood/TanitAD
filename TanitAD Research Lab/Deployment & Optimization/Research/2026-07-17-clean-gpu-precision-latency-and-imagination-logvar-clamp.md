# Prod-Opt run #3 (2026-07-17) — clean-GPU precision/latency (P1.4b) + imagination logvar clamp (review #3)

- **Agent:** production-optimization-agent (Saturday), run #3
- **Hardware:** local RTX 4060 (declared Orin latency proxy, I8), **exclusive/uncontended
  this run** (3 % idle util before start, no CarlaUE4, no resident python). Clocks
  NOT admin-pinnable on this box (`nvidia-smi -lgc` → permission denied) — mitigated
  by p50/p95 over 100 reps; the dominant 2026-07-09 confound was *contention*, now removed.
- **Cost:** $0 (local). Wall-clock: ~6 min GPU experiment + review. Loop iterations: 1
  (both gates met first pass). Web searches: 0 (execution run, not a literature run).
- **QUALITY:** full — one measured experiment with decision-grade numbers (G-H, G-P2) +
  one compliance-review intake with 17 tests (G-P1). Readiness: **validated** (measured
  on the Orin proxy; gap to *production* = TRT engine on real Orin, toolchain-blocked).

## 1. Measured experiment — P1.4b: clean-GPU batch-1 precision/latency (G-H, G-P2)

Re-ran the fp32/fp16/bf16 batch-1 decision-tick sweep on the **now-exclusive** 4060
(`half_precision_latency_accuracy.py`, step-6500 `ckpt_full.pt`, 64 real comma2k19 val
windows, one fixed fp64 imagine-and-select probe). The 2026-07-09 run withheld absolute
latency/Hz + per-precision VRAM because the GPU was CarlaUE4-contended (99 % util, fp32
tick read 33.5 ms). This run answers the withheld questions.

| Precision | Decision tick p50 | Hz | speedup | encode p50 | K9-select p50 | agreement | wp-shift mean | verdict |
|---|---|---|---|---|---|---|---|---|
| **fp32** | **14.79 ms** | **67.6** | 1.00× | 8.98 | 5.81 | 1.000 (ref) | — | reference |
| **fp16** | **10.67 ms** | **93.7** | **1.39×** | **4.69** | 5.99 | **0.953** (3/64) | **3.9 cm** | **DEPLOY** |
| bf16 | 10.64 ms | 94.0 | 1.39× | 4.42 | 6.22 | 0.672 (21/64) | 47.7 cm | **REJECT** |

Peak VRAM: fp32 **1.102 GB** (clean standalone). fp16/bf16 rows read 1.65 GB but that is a
**harness artifact** — the script keeps the fp32 reference model co-resident for the
accuracy delta, and 261 M params × 2 B ≈ 0.52 GB is exactly the delta, i.e. it is
`fp32-resident + fp16-model`, not fp16's standalone footprint. Only the fp32 standalone
number is trustworthy here (same caveat the KB already flagged 2026-07-09). Clean
per-precision VRAM needs a **one-process-per-precision** harness → new backlog item P1.4c.

### Findings
1. **fp32 tick 14.79 ms reproduces the clean 15.07 ms baseline** (2026-07-08) → the
   2026-07-09 **33.5 ms was pure CarlaUE4 contention, not a real regression**. Absolute
   latency/Hz are now decision-grade. ~68 Hz fp32 / ~94 Hz fp16 — both far above the
   10–20 Hz operative requirement **before any TRT/quantization** (3.4–4.7× headroom).
2. **fp16's entire 1.39× win is the ViT ENCODER** (8.98 → 4.69 ms, ~1.9×). The predictor
   and K9-select passes barely move (5.81 → 5.99 ms) — at batch-1 they are launch/memory-
   bound, not compute-bound, so half-precision buys nothing there. **Production implication:
   the latency lever is encoder precision.** The planned INT8/FP8 work (P1.6) targeting the
   predictor/heads will yield ~0 batch-1 latency — the time is in the encoder, which must
   stay ≥fp16 (bf16 unsafe, see #3). INT8 on the encoder is the only further latency win and
   is the known "native-TRT ViT INT8 trap" → OwLite/ModelOpt, ViT tower FP16. This re-orders
   P1.6: **quantize for VRAM/energy, not for batch-1 latency.**
3. **Precision policy reproduced exactly on the clean run** (G-P2): fp16 decision-SAFE
   (95.3 % agreement, 3/64 flips, 3.9 cm mean / 19 cm max waypoint shift, enc rel-err
   7.8e-4); bf16 UNSAFE (67.2 %, 21/64 flips, 47.7 cm mean / **3.58 m max** shift, rel-err
   7.2e-3). Both finite (precision- not range-limited). Agreement numbers match 2026-07-09
   to the digit (95.3/67.2) across a different probe fit → **the policy is reproducible**.
   Speed alone is a trap: bf16 is the *same* 1.39× as fp16 but flips 1 in 3 maneuvers.
   **Deploy TRT-fp16, never bf16.** The pre-registered TRT-fp16 acceptance bar stands
   (≥95 % agreement, ≤~4 cm wp-shift on these 64 windows) for the eventual engine.

Source: `Implementation/half_precision/half_precision_clean_20260717.json`.

## 2. Compliance review #3 — `tanitad/models/imagination.py` numerics (G-P1)

The H15 imagination field's per-cell `logvar_head` is an **unbounded** `nn.Linear` feeding
two un-clamped `exp()` sites. Both are live production hazards:

- **`imagination_nll` (`imagination.py:135`)** `exp(-logvar)` overflows fp32 at
  logvar < -88.7 → **+inf loss → NaN gradients**. MEASURED: `logvar=-100 → loss=inf`.
  Called in `train_worldmodel.py:338` (the **flagship arm**), `train_flagship4b.py:164`,
  `finetune_traj.py:217` → one over-confident cell can NaN-kill a training run (the
  F-5/F-6/F-7 silent-death class the pod monitor keeps catching).
- **`replay/arms.py:284`** `(0.5*logvar).exp()` (the OKRI/LOPS/H2-trigger uncertainty
  export) overflows on unbounded **positive** logvar → NaN metric / NaN safety trigger.

**Fix (intake `2026-07-17-imagination-logvar-clamp`, 17 tests):** clamp logvar to `[-10, 10]`
at the head output (protects all consumers) + a defensive clamp inside `imagination_nll`.
Behaviour-preserving in the healthy range (allclose to the stack formula for logvar∈[-3,3])
→ a trained checkpoint is numerically unchanged; only pathological values are trimmed.
`clamp` zeroes gradient at the bound = "stop pushing logvar to ±inf". Witness tests confirm
the unpatched stack `imagination_nll` overflows and that the safe form matches it in-range.
Baseline `pytest stack/tests -k "imag or h15 or d9"` = 11 passed pre-merge.

## 3. Actionable recommendations

- **Deploy fp16 on the decision path; keep the ViT tower ≥fp16; never bf16** (P3, measured).
- **Re-order P1.6 quantization:** target VRAM/energy, not batch-1 latency — the predictor/
  heads are already latency-floored; the encoder holds the time and must stay ≥fp16.
- **Merge the logvar clamp** before any long imagination-heavy run — it removes a live
  NaN-a-training-run mode in the flagship path (P3 / ops-fragility).
- **P1.4c (new):** one-process-per-precision VRAM harness for a clean fp16 standalone
  footprint (the co-resident fp32 reference inflates the current fp16/bf16 VRAM rows).

## 4. Ledger / hypothesis impact

No hypothesis status change (P8). Evidence rows: this is a production/numerics run — the
efficiency numbers (68 Hz fp32 / 94 Hz fp16, CNCE inputs) feed `Benchmarks & Eval/
LEADERBOARD.md`; H5 (efficiency) gets a clean-GPU latency datapoint, no status upgrade.
