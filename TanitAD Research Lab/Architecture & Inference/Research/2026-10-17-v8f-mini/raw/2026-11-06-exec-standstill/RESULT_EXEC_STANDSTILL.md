# RESULT_EXEC_STANDSTILL: the executor's standstill heading does not drive v8F's launch crawl (Master Mind, 2026-11-06)

**Card:** `PREREG_EXEC_STANDSTILL.md`, frozen at 13:31:52, sha256 `c782928e...`, unchanged.
**Reading rules:** `DEFINITIONS_H.md`, written at 13:45:54, before any contrast; the results were written at 14:12:56.
**Tier:** T1-kin (own executed motion, logged cameras), eval-141.
**Estimator:** paired episode-cluster bootstrap, B = 2000, seed 0. It answers *another draw of episodes*. The training-seed spread
`|dE_s1 − dE_s0|` is 0.0002. The planner is deterministic.
**Evidence class:** MEASURED, `raw/RESULT_TABLES.md` and `raw/EXEC_STANDSTILL_READ.json`.

## Gates: all PASS, each with its RED arm (`raw/GATES.json`, `raw/GATE_ID_FULL.json`)
* **G-ID.** At ε = 1e-3, the full eval-141 grid is strictly bit-identical to the banked C0 rollouts, at both seeds. RED arm: ε = 0.1
  differs on 52 of 61 launch windows.
* **G-MOV.** hold equals base exactly on 44 of 44 all-moving windows.
* **G-PHYS.** 0 heading changes on 5,200 sub-ε frames. RED arm: 2,892 violations.
* **G-P10.** The tick-0 plan is identical on all windows. RED arm: 123 of 123 break it.

## Primary (card arm): E = M1 crawl share of C0 minus P10; dE = E_hold − E_base
| seed | E_base | E_hold | dE [95 % CI] | share explained |
|---|---|---|---|---|
| s0 | +0.0532 [+0.0355, +0.0741] | +0.0538 [+0.0359, +0.0743] | **+0.0026 [+0.0003, +0.0057]**, separated-POSITIVE | −4.9 % |
| s1 | +0.0594 [+0.0366, +0.0861] | +0.0587 [+0.0367, +0.0851] | +0.0024 [+0.0000, +0.0059], not separated | −4.0 % |

* **Outcome, literal:** **UNCOMMITTED.** The card names no outcome for a separated-positive dE, and `DEFINITIONS_H.md` fixed that reading
  before the data.
* **Read in substance:** **[HA] is not supported.** A physically consistent executor does not shrink the crawl excess at either seed. At
  s0 it moves it by +0.0026 in the wrong direction, about 5 % of E_base.
* **DAG-OACC (beside, never deciding):** NO-EFFECT. dE is +0.0007 at s0 and +0.0016 at s1, neither separated.
* **Also beside:** on the 240 launch-class windows, C0's own M1 excess over P10 is +0.263 / +0.280 under base and +0.280 / +0.288 under
  hold. The crawl is concentrated at launch and does not respond to the heading input.

## Beside: what the physically consistent executor changes (C0 − P10, dE per row; instrument contrasts, not a model change)
| row | dE at s0 / s1 |
|---|---|
| ADE6 | −0.130 / −0.102 m, both separated: C0's margin over P10 grows |
| heading MAE | −0.44 / −0.66 °, separated |
| cross-track MAE | −0.16 / −0.14 m, separated |
| yaw-rate MAE | −1.39 / −1.67 °/s, separated. Under hold, C0's yaw-rate row is not separated-worse than P10 (+2.06 [−0.25, +4.44] and +2.04 [−0.28, +4.63]) |
| S-POS@80 | −1.08 / −1.81 m, separated |
| accel MAE | +0.09, not separated / +0.21, separated: C0's acceleration margin shrinks |
| TAC-LAT agreement | +0.028 / +0.035, separated |

Lateral rows are reported per start class in `raw/RESULT_TABLES.md`, never pooled.

## Next lever (Rule Zero)
* **The crawl is not an executor-heading artifact.** It stays where report v41 §7.35 placed it: the speed side of the ego history. The
  next lever is the frozen T2 card, which tests the camera clock. Its stage 2 runs as frozen, under the `base` executor. Its package is
  being built (AGENT I).
* **`hold` is not adopted.** It is physically consistent and improves C0's lateral standing against P10. Adopting it for the lateral
  rows needs its own card, because the outcome here is UNCOMMITTED, not 3.
* **No blast-radius list.** That list belongs to outcomes 1 and 2 only.
