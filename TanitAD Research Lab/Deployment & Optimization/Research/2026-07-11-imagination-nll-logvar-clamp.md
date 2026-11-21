# 2026-07-11 — Imagination-NLL `exp(-logvar)` overflow: a live silent-NaN path (measured, fixed) + recovering the unmerged 07-10 run

Production & Optimization, Saturday run #4. One compliance review (**#3 continuation**:
`tanitad/models/imagination.py`, backlog **P1.7**) shipped as an intake package with a **measured
experiment** (D-020 G-H): the exact `logvar` overflow thresholds, a **reachability verdict**, and the
parity of the fix. Hardware: **CPU, deterministic (seed 0), ~1 s, $0** — chosen deliberately because
the local 4060 was **100 % util / contended** this run (a clean GPU optimization experiment would be
invalid; the accuracy-vs-latency optimization experiment of record this week is the INT8 curve below,
already measured on an idle GPU 2026-07-10). G-P2 note: today's "efficiency" delta is a **safety
envelope** (a NaN that silently corrupts the checkpoint), so the delta reported next to it is the
**parity/accuracy** of the fix (in-band `max|Δ| = 0.0`), not a speed number.

---

## 0. Run context — the 07-10 prod-opt work was NOT lost; it is committed but UNMERGED

The orchestrator's W29 report flagged "Prod-Opt stranded `int8_quant` + `contract-windowing-failloud`
uncommitted in main" (a D-026 failure mode). **Verified this run:** that work is fully committed on
branch **`prod-opt-20260711`'s ancestor `1a5754d`** (`hub(prod-opt): INT8 weight-quant curve + clean
latency + windowing fail-loud …`), authored 2026-07-10 — it is simply **unmerged into `main`**. The
untracked copies still sitting in the `main` working tree are **byte-identical residue** of that commit
(`diff` = 0 on the script, the windowing module, and the test file) and are a **merge hazard**: `git
merge` refuses when untracked files would be overwritten. **Action for the orchestrator:** merge
`agent/prod-opt-20260711` (which contains `1a5754d` as a linear ancestor — this run branched from it, so
both the 07-10 and 07-11 prod-opt work land in one merge), then `git clean` the identical residue from
`main` first if the merge complains. Nothing was re-done; today's run stacks on top.

For the record, the 07-10 experiment (`Research/2026-07-10-…md`) stands and is the **optimization
experiment of record** for the INT8/latency duty: per-module INT8 **weight**-quant localizes the
sensitivity to the **readout** (int8_encoder 98.4 % / 1.6 cm SAFE, int8_predictor 95.3 % SAFE,
int8_heads 48.4 % / 1.67 m UNSAFE), **refuting** the "quantize heads first" heuristic for weight-quant;
clean-GPU latency fp32 15.76 ms / fp16 13.40 ms (1.18×) closed the P1.4b contention caveat.

---

## 1. Compliance review #3 (continuation) — `imagination_nll` can silently NaN the whole model

**Finding (silent-wrong / F-ops-fragility class), `stack/tanitad/models/imagination.py:135`:**

```python
nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)     # exp(-logvar) UNCLAMPED
```

`logvar` is produced by an **unbounded** Linear head (`ImaginationField.logvar_head`,
imagination.py:110 — no output activation, not zero-initialised). `torch.exp(-logvar)` is evaluated
**before** the `*err2`, so it overflows to `+inf` purely as a function of `logvar`:

| dtype | overflow boundary (measured) | = theory `−ln(finfo.max)` |
|---|---|---|
| fp32 | **logvar < −88.72** | −88.723 |
| fp16 | **logvar < −11.09** | −11.090 |

fp16 is **~8× closer to zero** — and fp16 is this program's declared **deployment** precision (and a
common autocast-training mode). Below the boundary the loss is non-finite, and **there is no nan/inf
guard between the loss and the optimizer step** (verified `train_worldmodel.py:330-358`): `loss_h15`
→ `loss` (line 346) → `(loss/accum).backward()` (348) NaNs every gradient → `clip_grad_norm_(…,1.0)`
(349) cannot recover (NaN in → NaN out) → `opt.step()` (350) writes NaN into **every parameter** →
the **atomic checkpoint save** (356-358) then persists a **corrupted resume point**. One bad cell,
one bad step, silent whole-model corruption — exactly the F-5/F-6/F-7 ops-fragility class this stream
tracks.

**Reachability is MEASURED, not asserted** (`Implementation/imagination_nll_overflow/`,
`logvar_overflow.json`). The heteroscedastic NLL's own per-cell optimum is `logvar* = ln(err2)`, so:

- any cell predicted better than **`err2 = 1.53e-5`** has `logvar*` **already past the fp16 boundary**
  → plain gradient descent *toward the optimum* crosses into `+inf`;
- reproduced with plain SGD on one well-predicted cell: **non-finite in 45 steps** (fp16, err2=1e-7,
  logvar hit −11.25) and 356 steps (fp32, err2=1e-40, logvar −88.96);
- even **before** the hard overflow the gradient `0.5·(1 − exp(−logvar)·err2)` explodes:
  **`2.4e8` at logvar=−20, `2.6e21` at −50, `2.8e34` at −80** → `clip_grad_norm_` then collapses the
  whole step onto this one exploding direction (destabilisation without a NaN, too).

This is not a contrived edge: driving `logvar → −∞` on easy cells to shrink the `0.5·logvar` term is
the textbook heteroscedastic-regression instability (β-NLL, Seitzer et al. 2022).

**Why it matters beyond "training crashes":** `logvar` **is** the H15 epistemic signal — it is the
LOPS input and the **H2 modality-steering trigger** ("a sensor may only be powered down when the
imagination's uncertainty in its field of view is low"). A NaN/`inf` here doesn't just stop training;
it poisons the exact self-monitoring value the safety story depends on.

### Fix (intake `2026-07-11-imagination-nll-logvar-clamp/`)

Clamp `logvar` to `[−logvar_clamp, +logvar_clamp]` (default **8.0**) before the exp:

```python
logvar = logvar.clamp(min=-logvar_clamp, max=logvar_clamp)   # fail-safe: exp(-logvar) finite fp16 & fp32
```

- **Finite** over `logvar ∈ [−160, 160] × err2 ∈ [0, 20]` in **both** fp32 and fp16 (measured).
- **Identity in-band** → **in-band parity vs the current formula = `0.0` exactly** (and `0.0` vs the
  *actual* stack function, cross-checked): wherever `logvar` is well-conditioned, training is
  bit-identical — no retraining implied, no capability change.
- 8.0 covers the realistic optimum range (`logvar*=ln(err2)` for `err2 ≥ 3.4e-4`) with margin; symmetric
  so the degenerate "infinite-uncertainty" direction is bounded too.
- **Second fix, same file (determinism / seed discipline):** `d9_rows` (imagination.py:154) shuffles
  with an **unseeded** `randperm` → the D9 chance-floor `shuffled_cosine` is non-reproducible run to
  run (a determinism gap on a value feeding a **gate**). Added an optional `generator` (default `None`
  = current behaviour); `hidden_cosine` / `calibration_gap` never depended on the shuffle.
- **Tests: 10 passed** standalone (torch-only). **Honest caveat (P8):** a hard clamp has zero gradient
  outside the band, so a runaway `logvar` can *park* just past the edge (loss stays finite/bounded) — the
  minimal parity-preserving fix; a `softplus` precision reparameterisation (gradient everywhere, changed
  semantics) is deferred unless a run shows edge-parking hurts D9.

**Falsifier verdict — CONFIRMED reachable.** Pre-registered falsifier: "if plain SGD on a well-predicted
cell did not reach a non-finite loss in fp16, the overflow would be unreachable and the clamp
unnecessary." It reached non-finite in 45 steps → the guard is warranted.

---

## 2. Ledger / gate impact

- **G-H (measured experiment):** ✅ overflow thresholds + reachability + fix parity, decision-grade,
  falsifier ruled on. GPU-contention-immune (CPU, $0) — an honest experiment while the 4060 was busy.
- **G-P1:** ✅ review names `imagination.py:135` / `:110` / `:154` and the live wiring
  `train_worldmodel.py:338-358`, and ships **10 passing tests** + a measured JSON.
- **G-P2:** ✅ the safety-envelope delta (silent-NaN removed) carries its accuracy delta (in-band
  parity `0.0`). The week's speed-vs-accuracy optimization delta is the 07-10 INT8 curve (accuracy %
  next to weight-MB), now stacked in this branch and surfaced for merge.
- **No hypothesis status change** (G-D N/A) — production hardening (P3), not a capability claim. But it
  protects the H15/H2/LOPS signal integrity; noted on the ledger as an instrument-hardening row.
- **Boundary (D-011):** nothing written to `stack/` directly; the fix is an intake package.

## 3. Backlog movement

- **P1.7 imagination_nll clamp → DONE** (this intake). Retired from the "logged review-#2 findings" list.
  The other half of the old P1.7 (`tactical_pred` fail-fast) is **subsumed**: `tactical_pred` is an
  `OperativePredictor` instance (`fourbrain.py:49`), the same class whose `assert w==window`
  (`predictor.py:73`) is already fixed by the pending review-#2 intake `2026-07-09-models-predictor-failfast`
  — one fix covers both instances. No separate package needed; flagged for the orchestrator to confirm
  when that intake integrates.
- **Next up (P1.8):** compliance review #3 proper on `stack/scripts/` + the training loop —
  and specifically **add the missing nan/inf guard on `loss` before `backward`/`opt.step`** in
  `train_worldmodel.py` (defence-in-depth: even with the clamp, any future non-finite loss source should
  fail loud / skip the step, not silently corrupt the checkpoint). This run localised the need; the guard
  is its own small intake.

## Sources (G-A)

- `Implementation/imagination_nll_overflow/logvar_overflow_sweep.py`, `logvar_overflow.json` (this run).
- `Implementation/incoming/2026-07-11-imagination-nll-logvar-clamp/` (intake + 10 tests + INTAKE.md).
- Stack sites: `stack/tanitad/models/imagination.py:110,135,154`; `stack/tanitad/train/train_worldmodel.py:338-358`;
  `stack/tanitad/models/fourbrain.py:49`, `predictor.py:73` (grep/read-confirmed this run).
- Prior: `Research/2026-07-10-int8-quant-curve-and-windowing-failloud.md` (unmerged `1a5754d`);
  `Implementation/half_precision/half_precision_step6500.json` (fp16 precision policy, 2026-07-09).
- β-NLL heteroscedastic instability: Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty
  Estimation with Probabilistic Neural Networks", ICLR 2022 (the `logvar → −∞` incentive).
