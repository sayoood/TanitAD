# VERDICT_PT3N (Master Mind, 2026-11-09)
Card `PREREG_V8F_PT3N.md` (`de94f72c`). Tier **T0**, an oracle-goal diagnostic, never a driving result. MEASURED: `RESULT_PT3N.md`, `RESULT_PT3N.json`, `runs/`. Estimator on every interval: paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000; it answers "another draw of episodes" and neither training seed nor planner seed. Planner-seed replicate is 0.0000 on all eight rows, so this planner's inference noise is below the third decimal.

## Validity
**Valid: all six controls pass.** Decoder tie to DECN difference 0.00000 (bar 0.01); N53 reference difference 0.000000 (bar 0.001); harness reproduction identical on one full unit (largest absolute difference 0.0); split guard with its mutation; clips; PLANTEST3's `tune_choice.json` sha256 unchanged, run errors 0. The sentence in the middle of `RESULT_PT3N.md` that points to `RESULT_PLANTEST3_controls.md` is carried over from PLANTEST3's report text; the validity block at the top of that file is the one that applies to this read.

## The read (card's table, per form and family, both seeds required)
| form | family | HOLD | G-TRUE s0 / s1 | G-TRUE minus HOLD s0 | s1 | label |
|---|---|---|---|---|---|---|
| plain | LONGITUDINAL (m/s^2) | 0.3348 | 0.3337 / 0.3396 | -0.0011 [-0.0069, 0.0049] | +0.0048 [-0.0022, 0.0120] | NO-GAIN, HOLD-COLLAPSE both seeds |
| plain | LATERAL (rad/s) | 0.0256 | 0.0285 / 0.0280 | +0.0029 [0.0017, 0.0043] worse | +0.0024 [0.0013, 0.0038] worse | NO-GAIN |
| shrink | LONGITUDINAL | 0.3348 | 0.3308 / 0.3319 | -0.0040 [-0.0082, -0.0001] better | -0.0029 [-0.0073, 0.0014] | seeds disagree, HOLD-COLLAPSE both seeds |
| shrink | LATERAL | 0.0256 | 0.0274 / 0.0270 | +0.0018 [0.0009, 0.0032] worse | +0.0015 [0.0005, 0.0028] worse | NO-GAIN, HOLD-COLLAPSE both seeds |

G-TRUE beats G-RAND on all eight rows (separated-better), so the plan does respond to the goal. It does not beat HOLD. TACTICAL: not computed (n = 0, this read has no decision output). STRATEGIC: PAUSED (PI item 25). Beside, no bar: mean ADE at 2.4 s of G-TRUE is 0.247 to 0.263 m against HOLD 0.2284 m, so on that mean every plan is further from the logged path than HOLD is.

**The one row that clears HOLD (shrink, LONGITUDINAL, s0) is not a PLANNABLE read.** It clears at one seed, by an interval that ends at -0.0001, and the plan sits 0.048 prior-std units from HOLD, under the 0.10 below which the card says a row does not count as planning.

DECODER-WEAK stays raised on the whole read: ridge held-out R2 speed 0.680 / 0.647 (bar 0.70), yaw rate -0.043 / +0.105 (bar 0.50). KIN-CEILING does not beat HOLD in either family (LONGITUDINAL 3.41 / 3.39 against 0.335 m/s^2; LATERAL 0.050 against 0.0256 rad/s).

## What the larger decoder did
PT3N minus PLANTEST3 on the same windows, G-TRUE, is separated-better on all eight rows: -0.0028 to -0.0124 m/s^2 and -0.0007 to -0.0023 rad/s. No bar was set on this difference. Part of it is the plan moving closer to HOLD: the distance from HOLD falls on all eight rows (for example plain LONGITUDINAL 0.102 to 0.071 at s0, 0.127 to 0.090 at s1), and the size of G-TRUE minus G-RAND shrinks on all eight (plain LONGITUDINAL s0 -0.0595 to -0.0424). PLANTEST3's NO-GAIN is therefore not explained by its 53-to-70-episode decoder alone: with 1,053 fit episodes the outcome label is unchanged.

**Lambda is chosen at the edge of its grid.** TUNE picks lambda = 1 on all eight (form, seed, family) cells, as it did in PLANTEST3, and lambda = 0 reads 1.31 to 1.38 m/s^2 on TUNE, about four times the lambda = 1 value there (0.32). The decoded cost, left alone, moves plans away from the logged actions; the regulariser pulls them back.

## Limits
* One change from PLANTEST3 (the decoder's fit set), one linear single-frame reader. TD3 and DEC-PATH were not run; HIER-ADDS is NO by construction.
* An oracle goal (the true encoded future), so a pass here would be a necessary condition and never a driving result.
* The card's table does not name a one-seed PLANNABLE. Both seeds are required and the row carries HOLD-COLLAPSE, so it does not fire the PLANNABLE trigger. The next line reads it that way, and this reading was made after seeing the row.

## Next (the card's triggers, Rule Zero)
No row is PLANNABLE at both seeds. KIN-CEILING does not beat HOLD, and DECODER-WEAK is raised, so the committed next arm is **a nonlinear multi-frame readout in the cost, fitted at N1053** (DECN's MLP, four frames, reads yaw rate 0.570 / 0.441). It runs under its own frozen card, `PREREG_V8F_PT3M.md`. Every outcome: the harness is read on the ladder world models when they finish; PLANTEST and PLANTEST2 stay required gates.
