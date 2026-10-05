# SPEC_R1 addendum A2 (DRAFT for the Master Mind to register) — the next levers after H1–H5 missed the gate

*Drafted 2026-10-05 ~01:40 UTC by the R1 agent after H0–H5 (seed 0 for H5) were scored and before H5c / H0n finished.
POST-HOC relative to SPEC_R1: every arm below is motivated by those numbers. Nothing here re-judges a registered arm,
and nothing runs before the Master Mind registers this file (hash + time in `raw/SPEC_SHA256.txt`).*

## What motivated it (MEASURED, `arms/score.json` on Thor; RESULT_R1.md §4)

* No arm clears the R1 gate. Turn direction-correct H0 0.815 → H3 0.891 → H4 0.905 → H5 0.909 (seed 0); heading-15
  stays 0.54–0.57; every TRAINED arm (H1 included, old labels) worsens straight ADE by +0.07 to +0.20 m.
* H2's tactical layer learns the dense labels on held-out eval (lateral accuracy 0.47 → 0.82) without moving the pick.
* H5's tactical condition does not control the fan: forced TURN_L / TURN_R / LANE_KEEP → the pick follows on 0.274 /
  0.437 / 0.428 (bar 0.95); the zero-init condition grew to weight norm 0.376 in 1,320 steps.

## A2-1 — run the registered TRAIN-FIT diagnostic (SPEC_R1 §7) now

Every arm H0, H1, H2, H2s, H3, H4, H5, H5c on 1,000 of its OWN training windows (numpy `default_rng(0)` choice of the
7,000), sampler seed 0, the same head path, batch 1 (`r1_arms.py --eval-split train`). Measures: turn direction-correct,
heading-15 (GT-turn windows among the 1,000), straight ΔADE vs H0 (paired episode-cluster bootstrap over train clips).
Reading: SPEC_R1 §7 verbatim. Diagnostic only, no bar. Heads are staged in `/dev/shm` (not counted against disk).

## A2-2 — H5g: the learned tactical condition, amplified at inference (no retraining)

* **Rule.** At inference the tactical-condition term is scaled by w: `cond = cond_proj(m) + ego/target terms +
  w · tac_cond(v)`; w = 1 is H5 exactly. Grid **w ∈ {1, 3, 10}**, applied identically in the predicted-posterior mode and
  under forcing. Same heads as H5 (and H5c for the control); nothing is retrained.
* **Choice of w (no eval selection).** On the A2-1 TRAIN-FIT windows: the smallest w whose pooled forced-direction
  controllability (TURN_L ∪ TURN_R) ≥ 0.95; if none, w = 10 (the largest tested) is reported as the arm.
* **Reported arm H5g** (H5 heads, chosen w) on the 4,634 scored eval windows, sampler seeds 0 and 1: SPEC_R1 §6's
  controllability bar (≥ 0.95 per forced class, both seeds) AND the R1 route gate, four families, vs H0.
* **Control H5cg** (H5c heads, the same w): must NOT pass the controllability bar (SPEC_R1 §6's H5c clause).
* Reading: H5g passes controllability and H5cg fails ⇒ the condition is learned but under-weighted, and refcv8's
  conditioning needs a stronger path or guidance; both fail ⇒ the zero-init additive condition did not learn a usable
  direction in 1,320 frozen-trunk steps (the warm-started full run must give it a stronger path, e.g. per-anchor
  conditioning); H5cg passes ⇒ the instrument is broken.

## A2-3 — H1n: is the straight cost the turn-enriched training sample?

* **Arm.** H1's recipe exactly (old labels, per-clip nav, E9 CE, 1,320 steps, lr 5e-5, batch 32, training seed 0) on a
  window subsample of the SAME train cache with the natural turn share: all 5,273 non-turn windows + a seeded
  (`default_rng(0)`) 585 of the 1,727 GT-turn windows (585 / 5,858 = 10.0 %; eval139's natural share is 9.7 %,
  2,317 / 23,772). One lever vs H1: the turn share of the training windows (24.7 % → 10.0 %).
* **Bar (attribution).** H1n's straight ΔADE vs H0 ≤ +0.05 m with CI upper ≤ +0.10 m on both sampler seeds ⇒ the
  enrichment carried H1's straight cost; otherwise the cost is the fine-tune itself (lr / steps / overfit), and A2
  names that as the next lever. Turn direction-correct is reported beside it (expected to drop toward H0).

## Budget

A2-1 ≈ 25 min of Thor GPU (one lock acquisition); A2-2 ≈ 2 × 3 w × (eval + forcing) ≈ 3 × 20 min; A2-3 ≈ 15 min train +
10 min eval. All under `thor_gpu.lock`, each pass ≤ ~40 min. Disk: +0.12 GB (H1n heads moved off Thor after its eval).
