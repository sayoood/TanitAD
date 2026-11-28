# PREREAD_PRH_TAC: is the seed-shared adapter subspace task-shaped? (O2's committed next step; frozen 2026-11-11 ~12:00, before any TAC statistic)

**Trigger.** PRH-SEED v2 read **O2 PLATONIC-OTHER** with all controls passing:
* metric CC32, calibration sha256 `4a3ca969…`;
* P1 +0.0169 [.0155, .0186];
* P2b −0.0464 / −0.0458, i.e. the trained adapters sit closer to PCA than to SLOW;
* source: `RESULT_PRH_SEED_V2.json`.

The v1 card commits O2's next step: *"the overlap of TR with the linear-regression subspace of the 60-d tactical code"*. This read does that, with zero training, on CPU.

## Representations
Everything is identical to v2: data, eval-141, split by clip-id parity, frames, the metric **CC32**, and the episode-cluster bootstrap (B 2,000). The new representation is **TAC**.

**TAC** is the ridge regression (λ = 1e-3 × mean eigenvalue of `nm`'s covariance, fit half) of the ground-truth tactical targets on `nm`, restricted to the frames that carry them. Its fitted outputs give a representation of dimension `d_TAC`.

**Targets** (the trainer's tactical code definition):
* the 22 goal bits;
* the lateral one-hot (5);
* the longitudinal one-hot (7).

Rules for the targets:
* A frame enters only if its lateral and longitudinal labels are single-class (partial-label sets excluded).
* The 26 constraint cells are excluded: they are masked and partly undefined.
* The label source is the one the eval harness joins (`join_eval141.npz` / the eval grid). The implementer names the file:line and the frame-to-window rule.

If fewer than 32 non-degenerate TAC dimensions exist, CC32 is computed as the mean of all available canonical correlations, and the count is reported.

## Prediction and outcomes (both committed)
* **Q1:** for EACH trained seed, `S(TRs,TAC) − S(RIs,TAC) > 0`, separated.
* Beside rows (no bar):
  * `S(TRs,TAC) − S(TRs,SLOW)`;
  * `S(PCA,TAC)`;
  * the fraction of TAC's out-of-sample R² recovered by a ridge readout from TR versus from RI (the same ridge, fit half → score half).

| outcome | condition | committed next step |
|---|---|---|
| **TAC-LEAN** | Q1 separated positive at both seeds | Training shapes the shared adapter toward the tactical task. SLOW is therefore a **world-model-path-only** candidate: a second adapter that feeds the WM, never the readers. Write the L2 card as "a SLOW WM adapter beside M2b's" for a box-2 ladder slot (the PI's call). |
| **NO-TAC-LEAN** | otherwise | The shared subspace is variance-driven (PCA-like), not task-driven, so the trained adapter adds little beyond DINOv3's high-variance directions. Write the L2 card as "SLOW against a PCA-initialised adapter as the WM latent" (box-2 slot, the PI's call). |

Either outcome leads to the L2 card; the outcome decides its control arm.

## Controls
* **C-SELF:** `S(TAC,TAC)` = 1.
* **C-SHUF:** TAC permuted across score clips gives S below the v2 FLOOR.
* **C-LABEL-SHUF:** TAC refit with targets permuted across fit-half clips must make Q1 non-separated. This is the deliberate regression: a TAC built from scrambled labels must not look task-shaped.

## Scope
A representation probe, not a driving eval: no tier, the four families do not apply. Labels are read only to define TAC. Nothing is trained, and parity is untouched.
