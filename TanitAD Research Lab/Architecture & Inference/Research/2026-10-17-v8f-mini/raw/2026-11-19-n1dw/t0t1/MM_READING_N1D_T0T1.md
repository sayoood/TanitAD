# MM reading: PREREG_N1D_T0T1. N1D's roundabout decision is lost in DECODING, not over the rollout (2026-11-19)

**Inputs**
- Card: `PREREG_N1D_T0T1.md` (sha256 5ef1ba4f), frozen before any number.
- Result: `RESULT_T0T1.json` (sha256 a69bd4a0); printout `READ_T0T1.txt` (sha256 a9386844); code in `code/`.

**Evidence and estimator**
- MEASURED from the banked NAVRBD D1-HO T1-kin dumps; zero training, CPU.
- Population: 5,736 windows / 497 held-out episodes.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only; training variance is carried by the two seeds.
- Tiers:
  - T0 here is the **tick-0 open-loop plan**, O's 30-knot plan at t0 from the logged history.
  - T1 is the rolled track: m = 10, causal bucket.

## Outcome as committed: DECODE-LOSES at both seeds

| arm | T0 TAC-LAT | T1 TAC-LAT | T0 - T1 | probe F-PLAN (tree) - T0 |
|---|---|---|---|---|
| N1D_s0 | 0.6266 | 0.6196 | +0.0070 [-0.0155, +0.0274], not separated | +0.0762 |
| N1D_s1 | 0.6349 | 0.6499 | -0.0150 [-0.0348, +0.0026], not separated | +0.0588 |

- **No rollout loss:** T0 - T1 is not above 0 by 0.0383 at either seed. The plan O decodes at t0 is already as wrong on the roundabout decision as the rolled track.
- **The loss is in decoding:** the plan-head input carries 0.70 / 0.69 (PROBE-N1D-FEAT), and the decoded plan delivers 0.63 / 0.63. These are point differences between two estimators, printed as such.
- **Beside (N1R and the channel):**
  - N1R's T0 - T1 is +0.0197 [-0.0026, +0.0405] (s0) and -0.0134 [-0.0334, +0.0059] (s1); neither is separated.
  - The channel's effect exists already at T0: N1D - N1R is +0.0164 [+0.0087, +0.0241] (s0) and +0.0155 [+0.0043, +0.0264] (s1), both separated.
  - The effect does not fade over the rollout. The fade term, (N1D T0-T1) - (N1R T0-T1), is -0.0127 [-0.0253, -0.0005] at s0 (separated: the gain is larger at T1) and -0.0016 at s1 (not separated).

## Controls: all pass
- **K1:** the self-pair reads 0.0, with CI [0, 0].
- **K-RE:** all four T1 accuracies equal the banked NAVRBD values, absolute difference 0.0.
- **K-CONST:** 0.338215.
- **K-T0:** the tick-0 plan equals the extractor's plan, 0.0 m on 5,736 / 5,736 windows, for both N1D seeds. N1R's tick-0 plan is read the same way, but its extractor parity was not measured.

## What it means, and the next lever (RULE ZERO)
- Across both reads, the roundabout decision reaches O's plan-head input (PROBE-PRESENT-UNUSED) and is lost in the plan head's decoding into the track, not over the rollout. So the lever is decoder-side.
- **N1DW (frozen, in build) is exactly that lever:** it weights the trajectory terms the plan head is trained on, on roundabout samples. A null on N1DW now points to the card's next decoder-side lever: an auxiliary route-decision head on O's plan feature that conditions the decoder (its own card and a PI go).
- The rollout-side levers (an RB weight on DAgger rows, a re-plan consistency term) are not indicated by this read.
- The SHUF permutation null for PROBE-N1D-FEAT is still running on the pod. It is read beside and moves no band.
