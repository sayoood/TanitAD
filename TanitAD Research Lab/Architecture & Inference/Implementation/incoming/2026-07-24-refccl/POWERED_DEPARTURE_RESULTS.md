# MORE-POWERED DEPARTURE EVAL — does the naive departure win survive at n=40? → NO (BOUND, decisively)

**MEASURED on `tanitad-eval`, `gpu_lock departure-power`, 2026-07-24.** 2-fold cross-fit of the naive
recovery recipe (steps 1500 / lat 1.75 / clean 0.30 / λ_dev 0.5 / lr 1e-4, decoder-only) to reach the
**maximally-powered n=40 held-out** (the existing ckpts, trained on 0:28, cap at n=12). foldA trained on
eps 0:20 → eval 20:40; foldB trained on 20:40 → eval 0:20; **pooled → all 40 eps held-out**, each by a model
that never trained on it. Scored under `corridor_departure_rate` AND the fair `band_ade2d(1.0)`, paired
episode-cluster bootstrap. Raw: `powered_departure.json`, `powered_departure.log`, `powered_departure_eval.py`.

## Result — **n=40 eps / 881 windows, power 1.83× vs n=12**

| metric | base | FT (cross-fit) | paired Δ(base−ft) [CI] |
|---|---|---|---|
| corridor_departure_rate@1.75 | 0.0134 | **0.0436** (3.3× MORE) | **−0.0302 [−0.0595,−0.0088] SEPARATED** |
| band_ade2d(1.0) (fair metric) | 0.1763 | 0.5419 | −0.3655 [−0.482,−0.262] SEP |
| closed_ade2s (exact L2) | — | — | −0.6614 [−0.831,−0.498] SEP |

**The n=12 "departure win" REVERSES at n=40:** naive dCDR **+0.0089 S** (D2 eval, n=12) / +0.0083 n.s.
(re-score, n=12) → **−0.0302 S** (cross-fit, n=40). At full power the recovery FT **departs MORE**, not less.

## PRE-REGISTERED VERDICT: **BOUND — decisively.** The departure win was n=12 (favorable-split) noise.

The pre-registered WIN (dCDR CI∌0 positive AND band_ade2d CI∋0) is **not met** — dCDR is CI-separated
**negative** (departs more) and band_ade2d is CI-separated-worse. **The closed-loop lever is NOT a real net
win on road-keeping.** The direction closes honestly.

## ⚠️ Honest confound (stated, not hidden) — and why it doesn't rescue the lever
The cross-fit trains each fold on **20 eps** vs the original naive's **28** — so n (12→40) AND per-fold
training data (28→20) both changed. The fold-FTs are **notably worse** than the 28-ep original (ft_dep 0.044
vs the original naive's 0.0085 at n=12; band_ade2d 0.54 vs 0.28), so part of the reversal is the smaller
per-fold training set, not pure statistical power. **BUT this does not rescue the lever, for two reasons:**
1. The cross-fit is the **standard unbiased full-corpus estimator**; its estimate of the recipe's held-out
   departure effect is **negative and separated**. The n=12 **+0.008** was a **single favorable split**
   (train 28 / eval 28:40); averaged over both splits the effect is negative.
2. A lever whose departure benefit **requires ≥28 training eps AND a favorable held-out split to appear at
   all**, and **evaporates/reverses under a standard cross-fit**, is **not robustly promotable** — data-
   sensitivity of this degree is itself a strike against it. *(A cleaner isolation — re-train the 28-ep naive
   with a different single held-out split — would separate "power" from "fold-size"; but no split makes a
   non-replicating, easily-reversed +0.008 into a robust win.)*

## The closed-loop-improvement direction — closed honestly (the full arc)
| stage | claim at the time | corrected read |
|---|---|---|
| D2 (n=12) | recovery-aug **halves** held-out departures (+0.0089 S) | ⚠️ n=12 favorable-split; does NOT replicate at n=40 cross-fit |
| RefcCL | encoder-in-loop doesn't dissolve the trade (safe canary) | holds — encoder safely fine-tunable |
| LOWOOD-CL | on-policy objective doesn't escape (worse) | holds — on-policy signal impoverished |
| tolerance re-score | the ADE "cost" was ~74–95 % a knife-edge-L2 artifact | holds — fair metric forgives most ADE cost |
| **powered eval (n=40)** | — | ⭐ **the departure BENEFIT is not robust — reverses at full power** |

**Net, honest:** the recovery-augmentation lever's ADE-cost was largely a metric artifact **and** its
departure-benefit is not robust (n=12 noise). On **road-keeping (A)**, in-envelope recovery augmentation is
**NOT a net win** at full statistical power. The direction closes. **⚠️ CORRECTION for the RETRACTION_LOG
(class C5 — effect off an underpowered single split):** the D2/RefcCL "held-out departure reduction"
headlines were **n=12-fragile**; the powered n=40 cross-fit does not support them. Any forward use of
"recovery-aug reduces departures" must cite the n=40 cross-fit (BOUND), not the n=12 single split.

## Durable, un-retracted deliverables from the whole direction
- **Method + machinery** (renderer-free ∧ non-self-referential ∧ data-efficient recovery; the low-OOD
  on-policy training harness; the tolerance-band metric; the encoder-integrity canary) — sound and reusable.
- **REF-C's encoder is safely fine-tunable** (RefcCL canary holds at a material move) — de-risks future work.
- **Two measurement lessons of program value:** (i) **exact-path L2-ADE mis-scores benign recovery** — use a
  tolerance-band metric for closed-loop; (ii) **n=12 held-out is underpowered for ~1 pp departure effects** —
  use full-corpus cross-fit. Both should gate every future closed-loop claim.
- **Diagnosis for the next bet:** road-keeping (A) via recovery-aug is exhausted; the open questions are the
  **reactive-agent collision (B)** renderer paths and a **map/tolerance-aware instrument** — the standing
  program directions, now cleanly justified rather than premature.

**Honest bounds:** n=40 via cross-fit (confound above); within-instrument RELATIVE; low-OOD lane-keeping, not
a safety rate.
