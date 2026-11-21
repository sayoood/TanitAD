# INTAKE — imagination-NLL logvar clamp (silent-NaN fail-safe) + d9 determinism

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-11-imagination-nll-logvar-clamp/`
- **Author agent / date:** Production & Optimization agent, 2026-07-11
- **Proposed target:** `stack/tanitad/models/imagination.py` (`imagination_nll`, `d9_rows`)
- **Hypothesis / WP served:** H15 (imagination) / H2 (modality steering trigger) / gate D9 — production hardening (P3), not a capability claim.

## What & why (≤10 lines)

Two same-file numerics fixes found in compliance review #3 of `tanitad/models/imagination.py`.
**(1) The anchor — silent NaN.** `imagination_nll` (imagination.py:135) computes `torch.exp(-logvar)`
with `logvar` from an **unbounded** Linear head (imagination.py:110). `exp(-logvar)` overflows to
`+inf` at `logvar < -88.72` (fp32) / **`< -11.09` (fp16, the deployment/autocast precision)**, so the
loss goes non-finite. It is wired straight into the live trainer (`train_worldmodel.py:338` →
`loss` → `backward()` → `opt.step()`) with **no nan/inf guard** (verified 330-358): one bad cell
NaNs every gradient (clip_grad_norm can't recover), `opt.step` NaNs every weight, and the atomic
save then **persists a corrupted resume point**. Reachability is **measured**, not hypothetical: the
NLL's own per-cell optimum is `logvar*=ln(err2)`, so any cell predicted better than `err2=1.53e-5`
has its optimum already past the fp16 boundary — plain SGD reaches non-finite loss in **45 steps**
(fp16, err2=1e-7). Fix: clamp `logvar` to `[-8, 8]` before the exp (identity in-band → zero
behaviour change where well-conditioned; caps the pathological tail). **(2) determinism** — `d9_rows`
(imagination.py:154) shuffles with an unseeded `randperm`, making the D9 chance-floor `shuffled_cosine`
non-reproducible; fix accepts an optional `generator` (default `None` = unchanged). Full evidence:
`../../Research/2026-07-11-imagination-nll-logvar-clamp.md`, experiment `../imagination_nll_overflow/`.

## Evidence & tests

- Tests included: `tests/test_imagination_nll_hardened.py` — **10 passed** (torch-only, no tanitad/CUDA/data).
  Cover: in-band parity vs the current formula = **0.0 exactly**; original overflows / hardened finite
  (fp16 @ −20, fp32 @ −100); gradient finite at extreme logvar; finite over `logvar∈[-160,160] ×
  err2∈[0,20]` both dtypes; clamp-parameter monotonicity; weighting semantics unchanged; d9 seeded
  reproducibility + `generator=None` fallback + no-hidden-cells NaN guard.
- Cross-checked the drop-in against the **actual** `stack` `imagination_nll`: in-band `max|Δ| = 0.0`.
- Measured numbers (experiment `../imagination_nll_overflow/logvar_overflow.json`, CPU, seed 0, $0):
  overflow thresholds fp32 −88.72 / fp16 −11.09 (measured == `−ln(finfo.max)`); gradient `2.4e8`@−20,
  `2.8e34`@−80; SGD reaches non-finite in 45 (fp16) / 356 (fp32) steps; clamp keeps loss+grad finite
  and converges; in-band parity `max|Δ| = 0.0`.

## Risk & rollback

- Blast radius: one function body + one signature (`imagination_nll` gains `logvar_clamp=8.0`; `d9_rows`
  gains `generator=None`) in `stack/tanitad/models/imagination.py`. Both defaults are backward-compatible;
  no other call site changes. In-band training is bit-identical (parity 0.0), so no retraining implied.
- Integration note: apply the two edits in-place (do NOT add a new module) to keep the change minimal.
  Optionally thread a seeded `generator` from the gate runner into `d9_rows` for reproducible D9 rows.
- Rollback: revert the two-line diff; the clamp/generator are additive so revert is clean and safe.
- Deferred (out of scope, honest): a `softplus` precision reparameterisation would keep gradient
  outside the band (a hard clamp has zero gradient there, so a runaway logvar can *park* just past the
  edge — loss stays finite/bounded). Larger semantic change; revisit only if a run shows edge-parking
  degrades D9 calibration.
