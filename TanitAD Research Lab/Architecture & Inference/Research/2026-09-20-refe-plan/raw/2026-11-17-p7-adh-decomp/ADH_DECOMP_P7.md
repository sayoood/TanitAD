# ADH_DECOMP_P7 — why P7's pick follows the command no better than a random proposal; does a command-consistent re-rank help? (0 GPU)

Evidence class: **MEASURED (ours)**, tier **T1** (NAVSIM v1.1 plan, self-action open loop; command is an input). Estimator for every interval: **paired log-cluster bootstrap, 136 logs, B 10,000, seed 20260927** (the `proxy_eval.boot` stream, via the pinned `command_adherence.py` functions imported unchanged). It answers "another draw of LOGS" only (no training-seed or inference variance). Inputs: `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full{,.proposals}.npz`, csv `.../score/refe_stage2_p7_full/refe_stage2_p7_full.csv`. All numbers: `adh_decomp_p7.json` (sections q1, q2_q3, q4). n = 12,146 tokens (all), 4,076 (turn-active, LEFT u RIGHT), 136 logs.

## Q1 Reproduction — exact
Pick adherence turn-active **0.8805 [0.8555, 0.9025]**, all tokens 0.8623 [0.8398, 0.8837]; floor (mean of 64 proposals) turn-active **0.8841 [0.8645, 0.9020]**, all 0.8386. Bit-equal to `command_adherence_p7_full.json`. Gates: dump `proposals[pick]` == seam poses bitwise; class of dump pick == class of seam. No mismatch.

## Q2 Ceiling / label noise — the headline finding
Classing the LOGGED HUMAN future (export `human_future_poses`, same labeller, `plan_classes`) against the command:
- **turn-active 0.8810 [0.8557, 0.9026]**, kappa 0.776; all tokens 0.8758 [0.8604, 0.8912], kappa 0.762.
- 3x3 (command rows LEFT/STRAIGHT/RIGHT; cols plan class L/S/R): LEFT [2262, 226, 13]; STRAIGHT [547, 7046, 477]; RIGHT [4, 242, 1329]. Recall LEFT 0.904, STRAIGHT 0.873, RIGHT 0.844.
- **P7 pick minus human, turn-active: -0.0005 [-0.0161, +0.0136] (not separated).** On all tokens pick is -0.0135 [-0.0258, -0.0016] below human.
=> On the turn-active headline the pick is AT the human ceiling. The command/classing pair itself disagrees with what a human did on ~12 % of turn-active tokens (485 tokens: 226 LEFT->STRAIGHT, 242 RIGHT->STRAIGHT, 17 opposite). 0.88 is not a P7 defect; it bounds every lever below.

## Q3 Decomposition (turn-active n 4,076; pick adheres 3,589)
- Best-of-64 adherent proposal exists on 4,028 (**0.9882**); SELECTION miss (adherent exists, pick not) **439**; GENERATION miss (none adherent) **48**. All tokens: 1,296 selection / 377 generation (oracle 0.9690).
- Per class (n; pick; oracle; selection miss; generation miss): LEFT 2,501; 0.9128; 0.9864; 184; 34. STRAIGHT 8,070; 0.8530; 0.9592; 857; 329. RIGHT 1,575; 0.8292; 0.9911; 255; 14.
- Share of the fan that adheres: mean 0.884, median 1.0 (turn-active). Pick adheres 0.941 when share >= 0.5, 0.216 when share < 0.5; in selection misses the mean adherent share is 0.49. So the scorer follows the fan majority; misses sit where the fan is split.
- So the fan is not the lever for turns (generation miss 1.2 %); the scorer is, subject to the Q2 ceiling.

## Q4 Re-rank (adherent proposal with highest scorer aggregate; else keep pick)
Control: pick replicated EXACTLY (planner.py `aggregate`, rule `navsim_v1`, V1_W (5,5,2), sigmoid logits, argmax; per-token torch float32): **12,146 / 12,146** equal to `pick` (1 token has a tied max). 
- Adherence turn-active **0.9882 [0.9798, 0.9946]**; delta vs pick **+0.1077 [+0.0880, +0.1298]** (separated); kappa delta +0.202. All tokens 0.9690, delta +0.1067 [+0.0899, +0.1250]. Per class pick->rerank: LEFT 0.913->0.986, STRAIGHT 0.853->0.959, RIGHT 0.829->0.991. Equals best-of-64 by construction.
- **Tokens changed: 1,296 (turn-active 439: LEFT 184, STRAIGHT 857, RIGHT 255 over all).**
- PDMS of the CHANGED tokens' current picks (csv `score`): mean **0.893** (n 1,296; 75.7 % >= 0.9; 4.2 % zero) vs 0.913 on unchanged and 0.911 overall; turn-active changed 0.872. We replace mostly GOOD plans, slightly worse than average. The re-rank gives up scorer aggregate on changed tokens: 0.913 -> 0.744 (mean).
- Descriptive strata (human future NOT an input; only read after the selection was fixed): of the 439 changed turn-active tokens, **268 are where the human did not follow the command** (current picks' PDMS 0.910, i.e. good plans overwritten; adherence there 0.35->0.90 = following the noisy label); **171 are where the human did follow it** (current PDMS 0.813, genuine misses). Plan class matching the human class turn-active falls 0.911 -> 0.891. 
- **Permuted-command control** (seed 20261117, 2,798 tokens changed): adherence to the TRUE command falls to 0.682 turn-active (-0.199 [-0.221, -0.176]) and 0.747 all tokens: does not exceed the pick. Passes (a broken re-rank would not fall).
- Seam: `rerank_p7_seam.npz`, same keys as `stage2_p7_full.npz` (token, fingerprint, poses [12146,8,3], sampling, arm; arm renamed `REFe_p7_rerank_cmd`), same token order; 1,296 rows differ, the other 10,850 bit-equal (checked). sha256 `ced64bce101379f82c5ca1efb27610a6586c528816c3db93c022075ee1d81fee`.
- **A7 repair: NOT re-applied, deliberately.** The dump proposals are built with `planner.executed(traj, j)` for every j (`refe_navtest_seam.py:294-295`), i.e. A7 is applied on the 20-pose grid before `to_navsim`; they are already the executed form (dump[pick] == seam bitwise confirms it). Re-applying on 8 poses (h[-1]=h[-2]) would be wrong and would alter poses.

## Read of the lever (not a verdict on PDMS; the MM scores the seam)
Adherence rises to the oracle, but ~61 % of the extra turn-active switches (268/439) are tokens where the human itself did not follow the command, and the scorer ranked the replaced plans higher on its own PDMS proxy. Expect PDMS flat to down; the seam is the experiment. A less noisy lever to test next: re-rank only when the scorer margin is small, or only when the fan majority AND scorer agree on non-adherence (a threshold sweep over the 439; 0 GPU) — not run here.

## Q5 Admissibility
The re-rank reads only (a) the navigation command (`commands_and_logs` -> `driving_command`, an inference input per the PI ruling), (b) the planner's own 64 proposals' classes (`cls64`, from the proposals themselves via the labeller) and (c) the planner's own scorer logits (`A`). In `scripts/s3_rerank.py`, `rerank(cmd_in)` takes only `cmd_in`, `cls64`, `A`, `pk`; `exp["human_future_poses"]`, the csv PDMS and the human class are loaded only AFTER `sel` is fixed and feed only blocks named DESCRIPTIVE/PDMS; no situation-classifier output and no ego-derived label enter. (The labeller classes the PROPOSALS, a function of the plan, not of ego state or GT.) The permuted-command control shows the selection depends on the command only.

## Files (hold = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp/`)
`ADH_DECOMP_P7.md`, `adh_decomp_p7.json`, `adh_q2_q3.json`, `adh_q4.json`, `rerank_p7_seam.npz` (+ `.sha256`), `rerank_selection_idx.npz`, `scripts/{common,s1_repro,s2_human_decomp,s3_rerank}.py`, `HANDOFF.md`. Instrument code was exported (not edited) from mirror FETCH_HEAD into scratch (deleted).
