# VERDICT_FULLN (Master Mind, 2026-11-09)
Card `PREREG_V8F_DECN.md` (`cfa5bb1f`) with its amendment 1; the probe its RISES trigger commits, streamed read-only from box 2 one clip at a time, on the deciding inputs. Tier **T0**, a readout diagnostic, never a driving result. MEASURED: `RESULT_FULLN.md`, `runs/*.json`. Pooled held-out R2, no interval, the same 17 held-out episodes (one digest over every N1053 and N4565 unit); differences under 0.05 carry no weight. LONGITUDINAL, LATERAL and TACTICAL have no metric here (a readout R2 only); STRATEGIC is PAUSED (PI item 25).

## Validity
Planted signal passes on every unit. The shuffled-target control is at or below 0.05 on 31 of 32 rows. The one row above it (R2 pooled token, s1, MLP m = 4, speed: shuffled +0.055) leaves the deciding maximum under amendment 1, so that cell reads 0.801 (MLP m = 1), in place of the excluded 0.844.

## The read: best held-out R2 over the readers (optimistic, a maximum over four readers)
| target (bar) | input | s0: N53 / N1053 / N4565 | s1: N53 / N1053 / N4565 |
|---|---|---|---|
| speed (0.70) | R0 M2b z1 | 0.420 / 0.695 / **0.778** | 0.387 / 0.686 / **0.702** |
| speed (0.70) | R2 pooled token | 0.498 / 0.721 / **0.840** | 0.391 / 0.736 / **0.801** |
| yaw rate (0.50) | R0 M2b z1 | 0.423 / 0.570 / **0.510** | 0.268 / 0.441 / 0.463 |
| yaw rate (0.50) | R2 pooled token | 0.502 / 0.451 / **0.540** | 0.329 / 0.507 / 0.324 |

## Outcome at N4565, against the card's bars (the labels are this verdict's; the card defines its outcome labels at N1053)
* **Speed: reaches the bar at both seeds on both inputs.** R2 is the firm one (0.840 / 0.801). R0 at s1 clears by 0.002, under the 0.05 that carries weight. The rise from N53 to N4565 is +0.32 to +0.41 on every deciding cell.
* **Yaw rate: does not reach the bar at both seeds on either input** (R0 0.510 / 0.463, R2 0.540 / 0.324). More data stopped lifting it: from N1053 to N4565 the change is -0.184 to +0.089, in both directions. It needs the MLP at every N (ridge at N4565 at or below 0.127). The yaw-rate target is the action at t, which the frames up to t have not yet seen.
* **The pooled token reads speed better than z1 at both seeds** (0.840 / 0.801 against 0.778 / 0.702). The level-1 latent the world model predicts carries less of it than the token it is projected from.

## What this does and does not settle
The vision chain reads speed at the 0.70 bar once all trained clips are used, so DECINFO's NOT-READ-FROM-VISION was a read at 53 episodes. That bar is far below what a planning cost needs. PT3M's ceilings (`plantest3/mlp4/RESULT_PT3M.md`) put the absolute read error at about 2.6 m/s at R2 0.69 on SCORE windows, against 0.69 m/s for HOLD's speed error at 2.4 s. ESTIMATED, assuming the error scales with the square root of 1 - R2: R2 0.84 would be about 1.9 m/s, and matching HOLD would take R2 near 0.98.

## Limits
* A maximum over four readers on 17 held-out episodes, with no interval.
* R2 at N4565 mixes more data and a wider population, as N53 to N1053 did.
* The MLP's early stop falls at 13 to 28 epochs of 60; these values may be below what a better-tuned reader would read.

## Next (Rule Zero)
Speed: the readers enter the planner through PT3M, running under its frozen card; the N4565 readers are the next step that card names if the reader's precision is the limit. Yaw rate: more data at this scale is not the lever. The measured-speed world-model card stays queued behind the band retrain (PI, 2026-11-09), and the measured current yaw rate and speed are admissible inputs at inference (R-EGO).
