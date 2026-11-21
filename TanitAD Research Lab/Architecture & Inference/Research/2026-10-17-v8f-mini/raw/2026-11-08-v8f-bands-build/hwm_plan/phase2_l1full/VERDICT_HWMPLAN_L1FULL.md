# VERDICT HWM-PLAN G3 (l1_full on R35): L1-FULL-HURTS in LONGITUDINAL; LATERAL SEEDS-DISAGREE; PLANNER-LIMITED on the decode-free path

Card `PREREG_V8F_HWMPLAN_DRAFT.md` (`ac20d690`), `FREEZE_HWMPLAN.md` (`ae2bbe40`), `AMEND_HWMPLAN_1.md` (`dcbd55a9`), `AMEND_HWMPLAN_2.md` (`b9a7602a`, written before this arm was scored). Run `phase2_l1full.sh`, marker `PHASE2_L1FULL.done` rc=0 at 2026-11-14 07:42 (2 workers, 1 h 51 min). Read by `contrast_l1full.py`, which imports the report's metric and CI functions. Before the arm was read, it reproduced all 75 phase-1 rows exactly at both seeds, and it reads exactly 0.0 [0, 0] when an arm is compared with itself. Outputs: `CONTRAST_L1FULL_R35_s{0,1}.json`.

**Tier T0 (oracle goal; zero training): a world-model diagnostic, never driving performance.** The estimator is the paired episode-cluster bootstrap over the same 994 SCORE windows in 71 episodes (windows sha256-equal to phase 1's), B 2,000. It answers "another draw of episodes". The inference-seed replicate is the floor. MEASURED.

## Rows (A = HWM-ORACLE `l1_full`, B = HWM-ORACLE `decode` from phase 1; positive = worse; s0 / s1)
| contrast | band | acceleration MAE (m/s^2) | yaw-rate MAE (rad/s) |
|---|---|---|---|
| A − B | 0-2 s | +0.0273 [−0.0160, +0.0724] / −0.0011 [−0.0357, +0.0345] | +0.0019 / +0.0020 (none) |
| A − B | 2-6 s | **+0.3184 [+0.2541, +0.3817] / +0.2285 [+0.1562, +0.2992]** | +0.0113 [+0.0023, +0.0198] / −0.0008 [−0.0090, +0.0074] |
| A − B | full 6 s | +0.2213 [+0.1758, +0.2673] / +0.1520 [+0.1030, +0.2012] | +0.0082 [+0.0020, +0.0144] / +0.0002 [−0.0058, +0.0061] |
| A − HOLD | full 6 s | +0.7664 [+0.6892, +0.8408] / +0.6061 [+0.5128, +0.6970] | +0.0752 / +0.0636 (separated worse) |
| A replicate | 2-6 s | −0.0006 / −0.0000 | +0.0013 / +0.0010 |

## Outcomes as committed
* **L1-FULL-HURTS (LONGITUDINAL), both seeds.** `l1_full` minus `decode` is separated-worse in the 2-6 s band, far beyond the replicate, and not separated in 0-2 s. **LATERAL: SEEDS-DISAGREE** (HURTS at s0, NEUTRAL at s1).
* **PLANNER-LIMITED holds on the decode-free path.** HWM-ORACLE with `l1_full` is separated-worse than HOLD in every band and family. Over the 2-6 s band it sits next to the flat planner (A − HOLD +0.6406 / +0.4638 against FLAT − HOLD +0.6425 / +0.5414 in phase 1; that comparison is across reads, not paired).
* TACTICAL (plan-derived kappa against the logged trajectory, full 6 s): A lateral 0.063 / 0.156, longitudinal 0.203 / 0.116; B 0.104 / 0.104 and 0.156 / 0.180. Goal setting n = 0. **STRATEGIC: PAUSED (PI item 25).**

## Master Mind context
* This settles phase 1's DECODE-WEAK question. Planning the 2-6 s knots with level 1 against level-2 sub-goals reads like the flat planner there. The decode path read 0.23-0.32 m/s^2 better only because its decode fails (off-image macros, `AMEND_HWMPLAN_2`) and leaves those knots near HOLD. So the hierarchy's measured gain over flat on R35 is the 0-2 s LONGITUDINAL effect alone: −0.0921 / −0.1717 m/s^2, both seeds, above the replicate. It is real and small. The plan stays about 0.9-1.0 m/s^2 worse than holding in that same band.
* Across every gradient planner read on this rig (FLAT1, TD3, HWM `decode` and `l1_full`; three z1 bases; readers; measured speed as input), planning toward the true future through the learned world model is worse than holding. The planner lowers its own cost on every window. A level-2 plan leaves the image of the encoder by a factor of about 7. Both point the same way: the optimiser finds actions the world model scores well and the logged drive does not take. **The world model's response to actions is the largest measured limit**, larger than the goal, the cost basis, the readers or the hierarchy's stitching.

## Next lever (Rule Zero; committed rows)
* The card's L1-FULL-HURTS row keeps the decode and commits the level-2 re-plan from level 1's end state (the BAND-SPLIT lever). Zero training, CPU. Because the decode check shows level-2 plans leave the image, that re-plan needs the level-2 image constraint, which has no card row, to be informative. Both are named here and get their own amendment before any window is scored.
* The card's PLANNER-LIMITED path ends at **world-model action sensitivity** (inverse-dynamics weight, the action-diverse sampler of `PREREG_V8F_DIVRS`). Its earlier step, the decoded action cost, was spent at FLAT1 by PT3M / PT3F. This is the lever with the largest measured effect. It is a training arm, so it needs box-2 time and the PI's go.

Plans: PRIVATE on HF, see `HF_BANK_HWMPLAN_L1FULL.json`.
