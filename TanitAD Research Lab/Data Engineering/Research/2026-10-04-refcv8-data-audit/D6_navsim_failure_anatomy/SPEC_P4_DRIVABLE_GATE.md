# SPEC P4 — an inference-time drivable-area gate from refcv7's OWN map head (zero training, NavSim-legal)

*Master Mind, 2026-10-04 ~23:15 Europe/Berlin. Pre-registered BEFORE any gated plan, mask or score exists. Follows from
P1' (`raw/d6_p1x.json`, all controls K1–K6 incl. K4s PASS, MEASURED on refcv7 step 50,400 navhard two-stage).*

## 0. Why (MEASURED, P1')

* **S1** (n 1,140): DAC = 0 although the PDM-Closed reference is clean. The 117-candidate fan holds a FULL-clean
  candidate on **0.890** [0.862, 0.917] of them. refcv7's own scorer pick is clean on **0.065**; a uniformly random fan
  member is clean on 0.320. The best clean candidate's median rank is 5, and it is in the top 5 on 0.509.
* **S2** (n 914): NC = 0. A FULL-clean candidate exists on 0.720; **0.930** of those are slower than the pick.
* Registered reading (SPEC_P1P2 §6): generation is not the defect (F ≥ 0.70). On DAC, a mild DAC-aware re-ranking is
  sufficient (top-5 ≥ 0.5). On NC, the defect is a speed commit.

Lever (c) of the registered table is an inference-time drivable gate from the BEV map head. It is the cheapest test of
the DAC half: no training, no new input, vision-only. Because it uses the model's OWN map prediction from the cameras,
it is admissible on the leaderboard-legal row.

## 1. Hypothesis

**H-D6-P4-GATE:** re-ranking refcv7-50,400's emitted fan with a gate built from its own predicted drivable area raises
navhard two-stage EPDMS, by removing plans the model itself predicts leave the drivable area.

## 2. Arms (one checkpoint: refcv7 step 50,400, `seam` inputs exactly as the official 50,400 navhard scoring)

| arm | pick rule | role |
|---|---|---|
| **G0** | the deployed pick, unchanged | identity control: per-token sub-scores must equal the banked official frame (max abs diff 0.0 on every column) |
| **G1** | among the candidates that PASS the gate (below), the one the DEPLOYED selection rule ranks best; if none pass, the deployed pick | treatment |
| **G1-DER** | as G1, but each scene's predicted drivable mask is taken from ANOTHER scene (fixed derangement, seed 20261004) | deliberate regression: must NOT clear the bar |
| **G-ORC** | as G1, but the gate uses the GT drivable area from the NavSim metric cache | PRIVILEGED ceiling, diagnostic only, never a lever claim |

**The gate (fixed now).**
* A candidate PASSES iff every check point lies on a BEV cell whose predicted drivable probability is ≥ 0.5.
* Check points: the ego footprint's 4 corners and its centre (NavSim ego dimensions) at t = 0.5, 1.0, …, 4.0 s.
* A check point outside the map head's output window is NOT checked (unknown passes). The share of unchecked points is
  reported.
* The mask is the model's own map-head output for that scene's input: the drivable class, at the head's native
  resolution, in the head's ego frame. The executor documents the exact tensor and the frame transform with file:line.
* No threshold, check point or horizon may be changed after any G1 number is seen. If the drivable class does not exist
  in the head, or the head cannot be run on the NavSim inputs, the test is NOT RUNNABLE and is reported as such. It is
  never adapted silently.

## 3. Measures and estimator

* On all 5,912 navhard two-stage tokens: official EPDMS (the devkit's aggregate), the DAC-zero rate, the NC-zero rate,
  EP, and the share of scenes whose pick the gate changed.
* Per stratum: S1, S1b, S2 and the remaining scenes.
* Scoring: every changed pick is scored exactly by the devkit in the local metric cache, re-using the P1' per-candidate
  scores where they already exist (`raw/p1x_scored_*.jsonl`), and scoring the rest.
* Estimator: the paired log-cluster bootstrap over the same tokens (the P2 estimator, B 2000).
* Seed floor: |A1 − A1_s1| on the same tokens (banked: step-50,400 navhard R7_A1 and R7_A1_s1, both PASS).

## 4. Bars (literals, both outcomes committed)

* **PASS (G1):** EPDMS(G1) − EPDMS(G0) ≥ **+0.020**, with the paired CI excluding 0, AND the delta > 2 × the seed-floor
  delta on the same tokens, AND the DAC-zero rate falls by ≥ 3.0 pp with its CI excluding 0. G1-DER must NOT pass (its
  EPDMS delta ≤ the seed floor). G0 must reproduce the banked frame exactly.
* **FAIL:** any PASS clause fails while G0 and G1-DER behave.
* **VOID:** G0 does not reproduce exactly, or G1-DER passes. The instrument is then broken and nothing is read.

**What each outcome means, committed now.**
* PASS ⇒ the model's own map knows where it may drive and the selector ignores it. Two consequences:
  * the gate enters refcv8's evaluation as an opt-in inference row on BOTH NavSim rows (legal and privileged);
  * a trained DAC-aware selector term (drivable compliance against SAM3 drivable GT, as a sub-score critic) becomes a
    named refcv8 lever.
* FAIL with G-ORC passing ⇒ the gate idea works but the predicted map is the bottleneck. The WP-D map levers move up in
  the refcv8 priority, and the trained critic remains the lever.
* FAIL with G-ORC also failing ⇒ a drivable gate on this fan does not reach the official score (e.g. the clean candidates
  fail other sub-scores, or the cut is too coarse). The next lever is the trained listwise selector with sub-score
  critics (X1), not a gate.

## 5. Scope and stamps

* One checkpoint, one inference seed (0) for the gate arms. The seed floor is the banked A1_s1.
* T2-class closed-loop NavSim metric, LOCAL scoring only: no leaderboard submission (PI ruling 2026-10-04).
* Not a refcv8 number. It is a lever test on refcv7-50,400.
