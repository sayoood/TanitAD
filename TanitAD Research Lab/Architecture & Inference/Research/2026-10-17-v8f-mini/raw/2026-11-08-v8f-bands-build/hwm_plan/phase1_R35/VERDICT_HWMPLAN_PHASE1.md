# VERDICT HWM-PLAN phase 1 (R35): PLANNER-LIMITED; HIER-ADDS in LONGITUDINAL under the DECODE-WEAK flag; LATERAL HIER SEEDS-DISAGREE

Card `PREREG_V8F_HWMPLAN_DRAFT.md` (sha256 `ac20d690`), frozen by `FREEZE_HWMPLAN.md` (`ae2bbe40`), amended by `AMEND_HWMPLAN_1.md` (`dcbd55a9`, world model R35). Run `phase1_R35.sh`, marker `PHASE1_R35.done` rc=0 at 2026-11-14 05:13 (1 worker, about 2 h 10 min after launch). Report `run_hwm.py report`, both seeds, rc 0: `RESULT_HWMPLAN_R35_s{0,1}.{json,md}`.

**Tier T0 (oracle goal = encoded true-future latent; zero training; frozen readers and planner): a world-model diagnostic, never driving performance.** HOLD and PRIOR rows are T1-kin references. Estimator: paired episode-cluster bootstrap (`taniteval/ci.py`) over 994 SCORE windows in 71 episodes (PLANTEST2 split; equal to the ladder harness's R35 c1 set). This answers "another draw of episodes". The inference-seed replicate (seed 1 against seed 0, same checkpoint) is the floor; two training seeds. MEASURED.

## Deciding rows (full 6 s band; positive = worse; s0 / s1)
| contrast | acceleration MAE (m/s^2) | yaw-rate MAE (rad/s) |
|---|---|---|
| HWM-ORACLE − HOLD | +0.5450 [+0.4907, +0.5977] / +0.4541 [+0.3914, +0.5138] | +0.0671 [+0.0615, +0.0724] / +0.0634 [+0.0583, +0.0686] |
| FLAT-ORACLE − HOLD | +0.7892 / +0.7154 (both separated worse) | +0.0737 / +0.0739 (both separated worse) |
| HWM-ORACLE − FLAT-ORACLE | −0.2442 [−0.3002, −0.1895] / −0.2613 [−0.3244, −0.1987] | −0.0067 [−0.0137, +0.0001] / −0.0104 [−0.0185, −0.0029] |
| HWM inference-seed replicate | −0.0109 / −0.0073 | +0.0001 / +0.0008 |

## Outcomes as committed
* **PLANNER-LIMITED** at both seeds. HWM-ORACLE minus HOLD is separated-worse on both deciding quantities, so the T0 ceiling is absent.
* **HIER-ADDS in LONGITUDINAL.** The contrast is separated-better at both seeds, at about 22x and 36x the replicate difference. **LATERAL: SEEDS-DISAGREE** (s0 not separated, s1 separated-better).
* **DECODE-WEAK flag: ON at both seeds.** The median relative macro-decode residual is 0.916 / 0.917 (q10 0.85, q90 0.99) against the bar of 0.1. Under the card, the 2-6 s band is then a property of the decode, not of the level-2 plan. The LONGITUDINAL HIER gain is mostly in that band: HWM minus FLAT in 2-6 s is −0.3203 / −0.3061, against −0.0921 [−0.1642, −0.0203] / −0.1717 [−0.2510, −0.0947] in 0-2 s. MEASURED beside, from the banked plans: HWM's decoded knots 11-30 sit 1.11 / 0.99 prior-std from HOLD in acceleration, against FLAT's 1.59 / 1.50. A decode that cannot reach the level-2 macro-actions, started from HOLD, keeps the plan nearer HOLD, and HOLD is the better reference here. So the 2-6 s gain is not credited to the hierarchy.
* **Level 1 under level-2 sub-goals (0-2 s; knots 1-10 are level 1's own plan, not decoded).** In LONGITUDINAL, HWM beats FLAT, separated at both seeds and above the replicate (−0.0013 / −0.0136). In LATERAL it is −0.0034 (none) / −0.0156 (better), so the seeds disagree. Even so, HWM is worse than HOLD there by +0.9907 / +0.8917 m/s^2.
* HOLD-COLLAPSE: off (plans sit 1.1-1.6 prior-std from HOLD). BAND-SPLIT: off. COVERAGE: n/a (T3 = 2: all 1,976 windows carry the goal frame).
* **TACTICAL** (plan-derived, one geometry rule, per band; full 6 s kappa against the logged trajectory): HWM lateral 0.104 / 0.104, longitudinal 0.156 / 0.180; FLAT 0.100 / 0.109 and 0.135 / 0.172; HOLD 0.481 and 0.361. Goal and anchor selection: n = 0. **STRATEGIC: PAUSED (PI item 25).**

## Master Mind context
* Direct higher-level steps reduce the deficit against HOLD by about a third (full-band acceleration +0.79 → +0.55 at s0, +0.72 → +0.45 at s1). They do not remove it. In the 0-2 s band, where no decode is involved, HWM is still about 0.9 m/s^2 worse than holding. So composing 0.2 s steps (WMCEIL) is part of the planner's error, not all of it.
* A residual of 0.92 after 300 AdamW iterations on a 6-number problem looks like level-2 macro-actions the encoder cannot produce from any bounded primitive sequence (off its image), not an under-optimised decode. This is a hypothesis; the cheapest test is next.

## Next lever (Rule Zero; committed rows)
DECODE-WEAK commits a learned macro-action decoder (a head on the frozen model) before the 2-6 s band is read again. HIER-ADDS keeps the hierarchy. AMEND_HWMPLAN_1 makes phase 1b (the window-aware planner for R35H) admissible. **First, the cheapest discriminating check (instrument, minutes, dev-box CPU):** decode macro-actions that are known to be in the image (macro-encoder outputs of real TRAIN primitive triples) with the same `decode_macro`. If their residual is low, the planned macros are off-image: the lever is the stitch lever `l1_full` (the card's G3 arm, which removes the decode from the 2-6 s path) or an image constraint at level 2, and a learned decoder cannot help. If their residual is also high, the decode itself is the defect and the learned decoder of the DECODE-WEAK row is the lever. PLANNER-LIMITED stands for the 0-2 s band either way. Its committed path is the cost in decoded action-relevant quantities, then world-model action sensitivity.

Plans and windows: PRIVATE on HF, see `HF_BANK_HWMPLAN_PHASE1.json`.
