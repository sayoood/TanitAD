# RESULT_FULLN: the deciding inputs at every trained clip (N4565)
Card `../../../PREREG_V8F_DECN.md` (`cfa5bb1f`, amendment 1). Tier **T0**, a readout diagnostic, never a driving result. MEASURED: `runs/*.json` here, N53 and N1053 from `../runs/`. Pooled held-out R2, no interval, the same 17 held-out episodes; differences under 0.05 carry no weight. Best over four readers, so optimistic. LONGITUDINAL, LATERAL, TACTICAL have no metric here; STRATEGIC is PAUSED (PI item 25).

## Controls
* Same held-out digest over every N1053 and N4565 unit: True.
* Shuffled targets at or below 0.05: 31 of 32 rows at N4565; rows above: R2 M2b_s1 mlp m=4 speed: +0.055 (amendment 1: a failing row leaves the deciding maximum).
* Planted signal: all pass.

## Best held-out R2 per deciding cell
| target (bar) | input | seed | N53 | N1053 | N4565 | reader at N4565 | rise N1053 to N4565 |
|---|---|---|---|---|---|---|---|
| speed (0.70) | R0 M2b z1 | s0 | 0.420 | 0.695 | 0.778 | mlp m=4 | +0.084 |
| speed (0.70) | R0 M2b z1 | s1 | 0.387 | 0.686 | 0.702 | mlp m=1 | +0.016 |
| speed (0.70) | R2 pooled token | s0 | 0.498 | 0.721 | 0.840 | mlp m=4 | +0.120 |
| speed (0.70) | R2 pooled token | s1 | 0.391 | 0.736 | 0.801 | mlp m=1 | +0.065 |
| yaw (0.50) | R0 M2b z1 | s0 | 0.423 | 0.570 | 0.510 | mlp m=4 | -0.059 |
| yaw (0.50) | R0 M2b z1 | s1 | 0.268 | 0.441 | 0.463 | mlp m=4 | +0.022 |
| yaw (0.50) | R2 pooled token | s0 | 0.502 | 0.451 | 0.540 | mlp m=4 | +0.089 |
| yaw (0.50) | R2 pooled token | s1 | 0.329 | 0.507 | 0.324 | mlp m=4 | -0.184 |

## Every reader at N4565
| target | input | seed | ridge m=1 | MLP m=1 | ridge m=4 | MLP m=4 |
|---|---|---|---|---|---|---|
| speed | R0 | s0 | +0.715 | +0.709 | +0.755 | +0.778 |
| speed | R0 | s1 | +0.660 | +0.702 | +0.696 | +0.688 |
| speed | R2 | s0 | +0.725 | +0.785 | +0.748 | +0.840 |
| speed | R2 | s1 | +0.707 | +0.801 | +0.736 | +0.844 |
| yaw | R0 | s0 | -0.088 | +0.496 | -0.088 | +0.510 |
| yaw | R0 | s1 | +0.084 | +0.320 | +0.127 | +0.463 |
| yaw | R2 | s0 | -0.067 | +0.425 | -0.029 | +0.540 |
| yaw | R2 | s1 | +0.053 | +0.212 | +0.088 | +0.324 |

## Outcome per target

* **speed: READ AT FULL N.** Clears the bar at both seeds: R0 yes, R2 yes.
* **yaw: BELOW THE BAR AT FULL N.** Clears the bar at both seeds: R0 no, R2 no.
