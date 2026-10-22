# Plan: from P2 (88.82) to above DriveZero (94.55) on navtest (2026-10-22, MM; PI: "follow your recommendation")

**Target:** surpass DriveZero Table A13 (DINOv3 ViT-L, 4 cameras, navtrain only, no SimScale), **PDMS 94.55** (PUBLISHED, p. 31). This is the like-for-like row (`eval/PREREG_P6.md`).
- **Tier:** T1, NAVSIM v1.1 PDMS, a pseudo-simulation of an open-loop plan against logged agents. Every read reports the four families (STRATEGIC PAUSED).
- **Seeds:** a claim needs a training-seed replicate (`H-ESTIM-SEED-1`).
- Our harness scores the logged human at 94.55 [93.80, 95.24] (REVIEW_7 §0.1), so the paper's row sits at the human ceiling of this metric.

## 1. Where we stand (MEASURED)

| | PDMS | NC | DAC | EP | TTC |
|---|---|---|---|---|---|
| P2 seed 0 (`LEADERBOARD.md` §0E.3) | **88.82** [87.86, 89.77] | 98.81 | 96.87 | **82.34** | 95.32 |
| P2 seed 1 | 89.15 [88.21, 90.07] | 98.77 | 97.23 | 82.99 | 95.13 |
| **DriveZero ViT-L, Table A13: the target** (PUBLISHED p. 31; sub-scores INHERITED from `P6_CONFIDENCE.md`) | **94.55** | 98.80 | **99.25** | **92.72** | 95.74 |

**The gap to the target is 5.73 points. P2 seed 0 is −10.4 on EP and −2.4 on DAC; it is level on NC and TTC.**

**What limits us, in order of measured size:**
1. **Selection, not proposals.**
   - The best of our 64 proposals reads **97.07**: our fan already holds a better-than-paper plan.
   - Perfect NC/DAC/TTC heads alone would pick 95.16. The single-head oracle gains are DAC +5.23, TTC +3.98, NC +1.42 (REVIEW_7 §0.3, on the earlier model).
2. **EP is a symptom of 1:** the pick sits at median rank 49 of 64 by path length. EP recovers when safety and TTC discrimination improve (REVIEW_7 §0.4).
3. **Generalisation to unseen drive sessions** (`D-REFE-SESSION-EXPOSURE-1`, today):
   - navtest shares 0 of its 44 sessions with navtrain's 162.
   - The search on P2's scorer gained +0.56 on trained-on sessions and lost −0.16 on navtest (C2).
   - ⇒ any lever that leans harder on the scorer must be tuned on unseen sessions. Until a trunk reserves some, the only such data is navtest.

## 2. The sequence

| step | what | why it is next | cost | gate / committed reading |
|---|---|---|---|---|
| **S1 (running)** | **P6**, the paper-faithful scorer: PDM targets on the student's own proposals, 100 % coverage, lag 0, full bank from step 0 | the root cause with the strongest evidence (REVIEW_7 §0.5) | resumes ~2026-10-23 from epoch 6/25; ~19 epochs, the rate is measured on resume | `PREREG_P6` (in force) + the dated pod-move amendment, before the read. **ONE navtest look.** `P6_CONFIDENCE`: beating 88.25 is likely; reaching 94.55 is LOW |
| S2 (0 GPU, on P6's outputs) | the selection rule and progress calibration on P6's scorer | PREREG_P6's committed next arm if EP is the largest gap | dev box CPU | its own prereg |
| **S3** | **P7, the next trunk**, with three changes: **P7.1** a training-only map-supervised drivable-area/occupancy aux head (DAC: the largest single-head oracle); **a session reserve of ~16 navtrain sessions** (`D-SESSION-RESERVE`, DECIDED 2026-10-22); the 3-C bank backed up on HF from the start | DAC is the paper's clearest sub-score lead (99.01 vs 96.9). The reserve makes unseen-session tuning possible without navtest | one A40, ~7 days (the PI authorises the compute when P6 reads) | its own prereg; the reserve's sessions drawn by a seeded rule recorded **before** any P7 data exists |
| **S4** | **P7.2, the scorer ensemble** (training seeds), plus **pessimistic selection and calibration** tuned on P7's reserved sessions | it attacks the error exploitation C2 measured. P7's reserve gives it a fit set navtest never touches | inference only | prereg; fitted on the reserve, judged by one navtest look |
| S5 | re-test the **search** on P7's ensemble scorer (C2's design) | the search failed only because the scorer's errors were exploitable; on a calibrated ensemble it may pay | dev box GPU | its own registration and its own look |
| S6 (later) | Thor latency, p99 ≤ 100 ms, 4 cameras (`D-LATENCY-BUDGET`) | the PI ruled results first, optimisation later | Thor | measured and reported; decides nothing until then |

**Measures carried by every read:**
- PDMS with the paired log-cluster bootstrap.
- The sub-scores NC / DAC / EP / TTC / C.
- The four families: STRATEGIC PAUSED; TACTICAL proxy; LATERAL DAC/DDC and curvature/yaw-rate; LONGITUDINAL EP/TTC.
- Best-of-64 and the head oracles: the selection headroom.
- The seen-session vs unseen-session split of any selection gain.

## 3. Not doing, and why

- **More search on P2:** closed by C2.
- **Pessimism tuned on P2:** Stage 0 A1 was NULL-UNDERPOWERED, 21 sessions are too few, and navtest is the only unseen pool for P2.
- **Splitting navtest into fit and judge halves:** rejected (one-look rule).
