# p0-sA01 — 261 M local pipeline validation (RTX 4060) — REPORT

**Purpose.** Memory-fit + full-loss-stack validation of TanitAD-4B-M (261 M) on the dev GPU.
Not a training run (300 steps, batch 8, lr still in warmup at end).

## Results

- **Fits and runs**: 261 M fp32, batch 8 @ 128 px on the 8 GB 4060; all five losses active
  (pred, tactical, SIGReg, inv-dyn, H15) and finite; loss 5.71 → ~2.4, SIGReg 49.6 → ~17.
- **I4 = 2.95** (imag-relative): correctly blocks predictive claims — barely-warmed-up model. Expected.
- **I2 FAILED as recorded (8.8e-4 > 1e-4) — and that is the headline finding:**

## Finding F-1: TF32/cuDNN kernel selection breaks batch-1↔batched encoding consistency at scale

No batch-statistics layer exists in the model (LayerNorm only), yet batch-1 and batched encodings
deviated ~1e-3-class. Root cause verified by controlled experiment: TF32 + cuDNN autotuning pick
different kernels (different reduction orders, ~1e-3 precision) per batch size.
Measured: unpinned TF32 deviation **3.76e-4**; under `strict_numerics()` (TF32 off, cuDNN autotune
off) **4.07e-7** → passes.

**Consequence (doctrine update):** the entire measurement path — probe fitting, gate evaluation,
deployment inference checks — runs inside `tanitad/instruments/numerics.py: strict_numerics()`.
Training keeps fast kernels. The I2 check itself now pins numerics. This is the ALPS-4B BatchNorm
lesson recurring in a milder, scale-dependent form: **the smoke test passed (3.6e-7) and only the
261 M model exposed it** — instrument rows must be re-validated at every scale point.

## Verdict

Pipeline validated for the A40 run (p0-sB01). No architecture change indicated. Instrument hardened.
