# D3 decomposition — independent instrument audit + Compounding Ratio adoption

**Agent:** Benchmarks & Eval (Thursday) · **Date:** 2026-07-10 · **Base commit:** `859caa8`
**Role:** independent-test (Mission Plan; agent duty #5 — recompute a Wednesday gate claim on a fresh
seed I fully control). **Compute:** local CPU, numpy-only, deterministic (seed 20260710), <2 s, **$0**.

## 0. TL;DR

I audited the D3 decomposition result (commits `9bbf4ca`/`c0b22b7`, ckpt step-14k) that a gate/arch
decision could rest on. Verdict, differentiated per claim:

| D3 claim | Verdict | Basis |
|---|---|---|
| "rel error FALLS with k ⇒ direct heads don't compound" | **FALSIFIED as stated (artifact)** | synthetic model that provably compounds superlinearly still yields falling rel_k |
| "recursion 2–4× worse than direct" | **CONFIRMED real** | denominator-free = the accepted Compounding Ratio; CR 4.00 / 3.72 |
| K-step arm "imag_rel 8.13→1.03 (1-step)" | **directionally real, headline confounded ≤2×** | cross-model rel_k inflated by encoder drift-scale; honest read = within-arm CR 3.90→0.385 |

Deliverable: a hardened, artifact-immune instrument (`i4_compounding.py`, 7 analytic tests green) that
adds the **Compounding Ratio (CR)** + absolute-error/drift companions so no D3 decision rests on the
rel_k-vs-k slope again. This is the I4 analogue of the D-004 I2 BatchNorm lesson: measure the thing,
not a ratio that silently hides it.

## 1. What the D3 instrument actually computes

`stack/scripts/d3_decompose.py:81` (verbatim):
```
rel_k = median( ||pred_k - true_k|| / ||true_k - z_t|| )
```
The **denominator `‖true_k − z_t‖` is the persistence drift** — how far the latent has moved from
"now" over k steps. It is *not* constant in k. On highway near-constant-velocity motion it grows
roughly linearly (or faster) with the horizon. So `rel_k` is a ratio of two horizon-growing
quantities, and its slope in k mixes model quality with scene drift.

Reported step-14k numbers (`stack/experiments/p0-d3-decompose/d3_decompose_14k.json`):

| corpus | direct_k1 | direct_k2 | direct_k4 | recursive_1step_x4 |
|---|---|---|---|---|
| comma2k19 (highway) | 12.112 | 8.381 | 5.123 | 20.513 |
| physicalai (urban) | 6.881 | 4.002 | 2.550 | 9.473 |

The "rel FALLS with k" pattern is the 12.1 → 8.4 → 5.1 (and 6.9 → 4.0 → 2.55) descent.

## 2. Audit — synthetic ground truth (`../Benchmarks & Eval/Implementation/i4_horizon_normalization_audit/`)

I rebuilt the exact instrument in numpy and drove it with latent trajectories I control:
`z_k = z_0 + k^{drift_exp}·v + noise`, model prediction `= true_k + abs_scale·k^{err_exp}·(unit dir)`.
Result JSON: `i4_horizon_audit_result.json`.

**Claim 1 — the "falls with k ⇒ no compounding" read is an artifact.**
- *Linear compounding* (abs err 0.30→0.60→1.20, genuinely doubling per horizon) on linear drift →
  rel_k **flat** at 0.297 / 0.299 / 0.300. A model that plainly compounds shows *no rise* in rel_k.
- *Superlinear compounding* (abs err 0.30→0.74→1.82, err∝k¹·³) on slightly superlinear drift
  (drift∝k¹·⁵) → rel_k **FALLS** 0.297 → 0.261 → 0.227. **A model whose absolute error compounds
  worse-than-linearly reproduces the exact falling shape the D3 note reads as "no compounding."**
  ⇒ the rel_k-vs-k slope cannot distinguish "heads are fine" from "heads compound but drift grows
  faster." Falsifier for "falls⇒fine" met.

**Claim 2 — "recursion 2–4× worse" is REAL and artifact-immune.**
`recursive_1step_x4` and `direct_k4` predict the **same target `true_4` relative to the same `z_t`**,
so they **share the identical denominator** `‖true_4 − z_t‖`. Their rel-ratio therefore equals their
absolute-error ratio exactly:
- comma2k19: 20.513 / 5.123 = **4.00** · physicalai: 9.473 / 2.550 = **3.72**.
This is precisely the **Compounding Ratio CR** of the world-model literature (SkyJEPA arXiv 2606.23444;
Robotic World Model arXiv 2501.10100): *CR = e_rollout / e_teacher-forced at a shared horizon;* CR≈1
= no compounding, CR>1 = compounding. CR 3.7–4.0 at step-14k = **strong genuine compounding of the
recursive path** — the K-step-training prescription is correct.

**Claim 3 — cross-MODEL rel_k comparison is confounded (collapse masquerade).**
Two models with **identical absolute error** but encoder drift halved: rel_k1 0.297 → 0.576,
**inflation ×1.94** from geometry alone. So comparing rel_k *across arms* (base vs K-step, which have
different encoders and hence different `z_t`/`true_k`) can swing ~2× without any change in predictive
quality. The K-step-arm headline "imag_rel 8.13→1.03 (1-step)"
(`stack/experiments/p0-kstep-arms/`) mixes two encoders' denominators and must carry that caveat.

## 3. The honest, decision-grade statement of the K-step win

Use **within-arm CR** (denominator-free, both terms from the same model), computed from the arm JSONs
at step-10500:
- base arm: recursive 14.495 / direct_k4 3.712 = **CR 3.90** (strong compounding).
- K-step arm: recursive 1.113 / direct_k4 2.891 = **CR 0.385** (<1 → the recursive rollout path is now
  *locally easier* than the direct head — the signature of a model trained with K-step rollout;
  matches the literature's CR<1 interpretation).

CR 3.90 → 0.385 is the artifact-immune way to state "K-step training kills recursive compounding."
It does not depend on the cross-model denominator that inflates the "8.13→1.03" headline.

## 4. Increment shipped (G-E, G-B2, G-H)

Intake `Implementation/incoming/2026-07-10-i4-compounding-instrument/` — `i4_compounding.py`
(`rel_triplet` exposes num/den; `compounding_ratio`/`cr_from_*`; `compounds()` verdict from the
absolute-error curve) + `test_i4_compounding.py` (**7 analytic-GT tests, 0.25 s, green**) + INTAKE.
Proposed: `d3_decompose.analyze()` emits `{rel_k, abs_err_k, drift_k}` + a top-level CR; add CR to the
D3 gate runner `extra_metrics`. Additive, changes no metric value or threshold.

## 5. Literature grounding (loop SEARCH, since 2026-07-09)

- **Compounding Ratio is standard metrology** (SkyJEPA 2606.23444; Robotic World Model 2501.10100):
  `CR_k = e_rollout / e_TF`; near-1 healthy, >1 compounding, <1 rollout locally easier. Directly names
  and justifies the instrument I adopt — impact: D3/I4 doctrine, H1/H5.
- **Horizon-dependent error crossover** (WM survey 2605.00080; HWM discussion): for short horizons
  (≤1 s) low-level 1-step WMs win; for ≥1.5 s a single high-level direct prediction beats autoregressive
  rollout of a low-level model (reduced accumulation). Grounds "K must match the decode horizon"
  (Arch 07-09) and our direct-vs-recursive framing — impact: H1/H5, D3.
- **NAVSIM v2 / EPDMS** (2506.04218; HF navhard space AGC2025): navhard = 450 Stage-1 / 5462 Stage-2
  nuPlan; EPDMS adds explicit lane-keeping + extended-comfort terms and *filters penalties also
  incurred by the human reference driver* — the same "compare against a reference, not an absolute"
  design our CR uses. No single-number June-2026 refresh beyond the April snapshot on the LEADERBOARD.

## 6. Hypotheses touched

- **H1 (hierarchical latent dynamics predictive):** evidence-of-instrument, no status change (P8). The
  D3 compounding signal (CR 3.7–4.0) is real; the "no-compounding" reassurance was an artifact — so H1
  is *less* supported by D3 than the raw note suggested, but the K-step arm's CR 0.385 is the fix path.
- **H5 (K-step rollout / consistency helps):** strengthened qualitatively — within-arm CR 3.90→0.385
  is a clean, artifact-immune win signature. Still not decision-grade (mid-training, single seed, no CI).
- No ledger status upgrade (D-004: gates BLOCKED/mid-training; instrument-hardening only, P8).

## 7. Falsifier / honesty (P8)

My own Claim-1 falsifier was pre-registered: "if a genuinely-compounding synthetic model cannot
produce a falling rel_k, the artifact hypothesis is wrong." It *did* produce it (superlinear case) →
hypothesis stands. Claim 2 could have shown recursion≈direct (no compounding) — it did not; the
compounding is real. I did **not** re-run the model on the GPU (checkpoint inference is the Wednesday/
loop path); my audit is of the *instrument and the reported ratios*, which is the metrology this
discipline owns — the absolute per-window errors behind the ratios were not exported, so the CR
reconstruction from rel values assumes the shared-denominator identity (proven exact in test 3–4).

## 8. Backlog delta

- Retire P0#? none. Add **P0: propose CR into `d3_decompose` + gate `extra_metrics`** (this intake) and
  **P1: re-run D3 with abs_err/drift exported** so CR is measured directly, not reconstructed.
- Standing: ≥3-seed SC-01 CARLA re-run (unchanged, still blocked on the CARLA-on-pod camera path).
