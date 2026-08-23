# Benchmarks & Eval — 2026-07-11: D1 ADE statistical-power audit + probe-capacity methodology review

**Agent:** Benchmarks & Eval (Thursday-scheduled; fired Sat 2026-07-11, day-of-week drift noted).
**Role exercised:** independent-test / gate-result audit (Mission Plan) + weekly research focus item
"statistical power (n per claim)".
**Base commit:** `0284a5c`. **Worktree:** `agent/bench-eval-20260711`.
**Cost:** $0 (local RTX 4060, 80.8 s encode + bootstrap). **G-H:** met (measured numbers below).

---

## 0. TL;DR (decision-grade)

The gate ladder has been reading a **D1 "regression" of ADE@1s 5.18 m (step-14k, 9 val eps) → 11.52 m
(step-21k, 4 val eps)** as a signal, while flagging it "small-sample noise, directional not decision-grade"
(PROJECT_STATE, 2026-07-10). **Nobody had quantified the estimator's sampling variance.** I did, on the
real step-6500 checkpoint + comma2k19 val cache.

**The D1 ADE@1s estimator's own noise band at n=4 val episodes (±4.5 m bootstrap CI half-width; 7.3 m
range across split seeds alone) is LARGER than the 3.17 m falsifier band (= half the reported 5.18→11.52
swing). The step-21k D1 "regression" is statistically indistinguishable from a hard-route-heavy sampling
draw. It is NOT a decision-grade regression.** The program should stop treating the 21k D1 number as
evidence of anything until D1 is read at ≥~20 val episodes and reported as mean±CI over seeds, not a
single-seed point.

---

## 1. What the shipped D1 estimator actually does

`run_d1` (`stack/tanitad/eval/gates.py:214`) reports `ade@1s` as the mean over **all val windows** from a
**single fixed seed=0** 80/20 episode split. Windows within one route are strongly correlated (same road,
lighting, speed profile), so the effective sample size is ~the number of **val episodes** — 9 at step-14k,
**4** at step-21k — not the window count. A single fixed seed means the number depends on *which* routes
happen to land in val.

## 2. Experiment (real, local, $0)

`Implementation/d1_power_audit/d1_ade_power_audit.py` — ckpt `ckpt_full.pt` (step-6500, the only trained
ckpt local), 40 comma2k19 val episodes, stride 12 → **755 windows / 40 routes**, ridge α=1e-3 (run_d1
default), ADE@1s ego-frame identical to `evaluate_checkpoint.py`. `strict_numerics()` on. Two variance
sources that differ between the 14k (n=9) and 21k (n=4) reads:

- **A. Val-set sampling variance** — fix a probe (trained on a disjoint 20-route pool), bootstrap the val
  set of size n over routes (3 000 reps), report the 95 % band of pooled window-ADE.
- **B. Shipped-estimator swing** — call the *actual* `run_d1` over the pool, vary the split seed 0..49, at
  `val_frac` 0.2 (~8 val eps) and 0.1 (~4 val eps) → the dispersion of the number a gate run reports.

## 3. Results (measured)

**Per-route ADE@1s dispersion (20 held-out routes, fixed probe):** min **2.31 m** / max **18.75 m**,
mean 8.73 m, SD 5.04 m, **coefficient of variation 0.577**. Routes differ in decode difficulty by ±58 %.

**A — fixed-probe val bootstrap (95 % CI half-width):**

| n val eps | mean ADE | 95 % CI | half-width |
|---|---|---|---|
| **4** | 8.71 | [4.53, 13.55] | **±4.51 m** |
| **9** | 8.68 | [5.70, 11.96] | **±3.13 m** |
| 20 | 8.75 | [6.63, 10.85] | ±2.11 m |

**B — shipped `run_d1` swing across 50 split seeds (identical estimator the program reads):**

| val_frac | ~val eps | mean | min | max | **range** | SD |
|---|---|---|---|---|---|---|
| 0.2 | 8 | 5.28 | 2.76 | 8.22 | **5.46 m** | 1.27 |
| 0.1 | **4** | 5.02 | 2.14 | 9.42 | **7.28 m** | 1.74 |

## 4. Interpretation

- **Falsifier verdict: the regression FAILS to clear the noise band.** Half the reported swing is 3.17 m.
  At n=4 the estimator's own 95 % CI half-width is **4.51 m** and the single-seed swing is **7.28 m** —
  both exceed 3.17 m. Pre-registered falsifier tripped → **the 5.18→11.52 m D1 move is not
  decision-grade.**
- **11.52 m is a normal n=4 draw, not a regression.** It sits inside the n=4 bootstrap CI upper bound
  (13.55 m); with per-route ADE reaching 18.75 m, four hard-route-heavy draws trivially average >11 m.
- **Even the n=9 (step-14k) read is marginal** (half-width ±3.13 m ≈ the falsifier band). The *first*
  point on the D1 trend was already inside its own error bars.
- **The single fixed seed is the silent hazard.** At the sizes we run, the *same checkpoint* reports
  anywhere in a 5–7 m window purely from the split seed. A gate that swings 7 m on a random seed cannot
  anchor a "gate movement" claim.
- **Honesty (P8):** measured at step-6500, not 14k/21k. This characterises a property of the *estimator*
  (between-route dispersion at small n), driven by route-difficulty heterogeneity, not the checkpoint. The
  CoV 0.58 lets it transfer across the mean-ADE level. The fixed-probe level (~8.7 m) sits above the
  shipped run_d1 level (~5 m) because run_d1 refits the probe on 80–90 % of routes each seed (more train
  data → better probe); the *dispersion*, not the level, is the finding.

## 5. Independent-test audit of `d1_probe_capacity.py` (the loop's D1 "step 1", `0284a5c`)

The loop shipped a probe-capacity discriminator (ridge×3 + MLP on 14k vs current latents) to decide
"info LOST vs less LINEAR". Script only — **no results committed yet**. Methodology review:

1. **Same small-sample fragility (primary finding).** It uses `[:12]` episodes/corpus, a parity split →
   ~6 val eps/corpus, and compares **single-split** val-ADE between two checkpoints. My audit shows ADE@1s
   at n≈6 has a CI half-width >3 m. **A ckpt-to-ckpt ADE delta below ~3–4 m from this script is inside the
   noise band and cannot support the "info lost vs less linear" verdict.** Recommend: bootstrap over
   routes (or ≥10 seeds) and report the MLP-vs-ridge *gap* with a CI, not point ADEs.
2. **Corpus mixing.** Parity split on the concatenated comma+physicalai index mixes corpora into both
   splits. The two corpora have very different ADE scales (D3 JSON: comma direct_k1 12.11 vs physicalai
   6.88 m), so the mixed val-ADE also moves with the comma/physicalai ratio of the val draw — a third
   variance source. Recommend per-corpus reporting.
3. **MLP fit budget (minor).** 60 epochs, no val-based stopping. Because *both* checkpoints get the
   identical budget it is a fair *relative* comparison, but if the MLP saturates below the information
   ceiling for both it cannot separate "linear-readout mismatch" from "info present but MLP-inaccessible".
   Recommend logging train-loss convergence as an admissibility check.

**Net:** the discriminator is a good idea, but as written its verdict is not decision-grade for the same
reason the raw gate isn't. Bootstrap + per-corpus + convergence-check would fix it. (Feedback for the
loop; I did not edit `stack/`.)

## 6. Actionable recommendations

- **R1 (validation-strategy, my discipline owns).** Extend the discipline's existing "≥3-seed mean±CI"
  rule — currently applied only to closed-loop CARLA — **to the open-loop decode gates D1/D3**. A D1/D3
  number is a weak claim unless reported as mean±CI over ≥5 split seeds (bootstrap preferred). Single-seed
  `run_d1` point reads are hereby deprecated for "gate movement" claims. (Adopted below; LEADERBOARD
  footnote added.)
- **R2 (sizing).** For a **decision-grade** D1 read able to resolve a ~6 m change, use **≥20 val
  episodes** (CI half-width ±2.1 m). At the current 4–9 val eps the gate is descriptive only.
- **R3 (to the loop / Architecture).** Re-run the step-14k/21k D1 comparison with (a) a fixed, shared val
  set of ≥20 routes across both checkpoints and (b) bootstrap CIs, before any D1 trend is reported. Until
  then, drop the "D1 5.18→11.52 regression" line from the gate narrative — it is a sampling artifact.
- **R4 (proposal to Architecture, non-blocking).** Add an optional `seeds`/`bootstrap` path to `run_d1`
  that returns `ade@1s_mean`, `ade@1s_ci95`, `n_val_eps`. Sketch in the intake INTAKE.md.

## 7. Hypothesis ledger

No H-status change. This is instrument hardening on the D1 gate (H-agnostic). The finding *tempers* every
prior D1-based read: D1 has never been decision-grade at the val sizes used. Recorded as an evidence-quality
note, no upgrade/downgrade (P8).

## 8. Backlog / next

Done this run: G-H power audit (intake pkg, 4 sanity tests green). Next: R3 re-run needs the 14k+21k
checkpoints (pod/loop) — filed as backlog #1. WP.29 paragraph extraction still pending (P1).

**Files:** `Implementation/d1_power_audit/{d1_ade_power_audit.py, d1_ade_power_audit.json, tests/, INTAKE.md}`,
this note, KB delta, LEADERBOARD D1 power footnote, BACKLOG, STATE.
