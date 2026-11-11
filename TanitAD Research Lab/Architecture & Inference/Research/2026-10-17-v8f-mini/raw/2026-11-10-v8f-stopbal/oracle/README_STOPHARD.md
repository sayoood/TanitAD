# STOPHARD read: implementation notes and numbers (no verdict; the Master Mind writes it)

Card: `PREREAD_V8F_STOPHARD.md`, sha256 `bc9ac911...fbe9` re-hashed before any work (equal); the hash is asserted at every run of `stophard_read.py` and pinned in `test_stophard.py::test_card_hash_unchanged`. Tier: T1-kin first tick, zero training, inference-time transform of T's own output; never driving performance. CPU, threads 2, `CUDA_VISIBLE_DEVICES=-1`. Tree `F:/TanitAD-scratch/stopbal_read/tree` (the run log shows tanitad loaded from there).

## Files
* `stophard_read.py`: transform, controls, estimator calls, outcome rule. Stage `run` banks one npz per arm to scratch; stage `analyse` reads them and writes `RESULT_STOPHARD.json`.
* `test_stophard.py`: 24 tests, all pass (MEASURED, CPU): card hash, decision rules against the STOPBAL source text, threshold edge (0.5 fires, one ulp under does not), HARD / HGOAL / HLON differ only in named entries and only where the decision fires, IDENT bit-identical, no-label signature, input not mutated, counts, red arms (forgets goal entry; touches a lat or constraint cell; hardens the wrong lon class; fires everywhere; label-driven decision; goal written in lon-only mode; reachable threshold read as IDENT; an unfired window changed), outcome table, gain and cost rules, no-floor mutant, estimator red arm (identical vectors: 0.0, not separated), fail-closed.
* `RESULT_STOPHARD.json`: every cell, control, count, floor, flag and outcome label.

## Implementation choices (the card is not reinterpreted)
* The model pass is `stoporc_read.run_arm`, imported and NOT edited. Its two module hooks (`MODES`, `force_codes`) are pointed at this file's transform before the call, so STOPORC's batch build, windows, checkpoint-hash check and dump tie-out are reused as is. The hook receives the GT allowed lon set and ignores it; `hard_code(own, mode)` takes no other input (asserted by signature in a test).
* Decisions (verified against `theory/stopbal/read/stopbal_read.py`, `P_STOP_MIN = 0.5`, `STOP_GI = 6`, `LON_STOP_I = 2`, `goal_stop = gp[:, STOP_GI] >= P_STOP_MIN`, `lon_stop = lp.argmax(1) == LON_STOP_I`): goal fires where own `code[6] >= 0.5`; lon fires where own `argmax(code[27:34]) == 2`. HARD sets `code[6] = 1.0` and the lon part to one-hot STOP on those windows. Nothing else changes.
* UNREACH (IDENT of control 1): the same transform with goal threshold 2.0 and the lon rule off; its code is bit-equal to OWN and its plan is run through `forward_diag`.
* Sets: S = 117 stop-point windows (25 episodes); A = all has-and-present windows (2,351 = 117 + 2,234). Cells: all 12 band x metric cells for ADE / FDE, speed and acceleration MAE, heading and yaw-rate MAE; S and A cells are `tactical_11_30:ade_m`.
* Floor: per variant, set and cell, |delta(NULL_rep) - delta(NULL_s0)| (point), read at seed 0, applied to both seeds. Gain = separated, lower error, |point| above the floor; cost = separated, higher error, |point| above the floor.
* Lever to variant: BAL-G with HGOAL, BAL-L with HLON. Outcome per lever needs both seeds; the NULL arms and NULL_rep are in the JSON under the same rules and are not in the outcome.

## Controls (MEASURED, `RESULT_STOPHARD.json` `arms.*.controls`, all 7 arms; all pass)
1. IDENT (unreachable): code bit-equal OWN, plan max abs 0.0 m, `forward_diag(OWN code)` vs OWN plan 0.0 m, all 12 cells equal, on every arm. Own plan vs dump `ol` within 1e-4 on every arm.
2. HARD / HGOAL / HLON differ from OWN only in the named entries (0 entries outside), only on windows where the decision fires; every window whose code is unchanged has a plan equal to OWN (max abs 0.0 m).
3. Decision counts, stop-point windows / control windows (2,234), equal the STOPDEC npz (per window) and `RESULT_STOPBAL.json` `M.hit.n_pos` / `M.fpr.n_pos` where that file carries the arm (NULL_rep is not in it; STOPDEC npz only):

| arm | goal fires (stop / ctl) | lon fires (stop / ctl) | both on stop | either on stop |
|---|---|---|---|---|
| BAL-NULL_s0 | 7 / 12 | 7 / 25 | 4 | 10 |
| BAL-NULL_s1 | 14 / 19 | 17 / 30 | 13 | 18 |
| BAL-NULL_rep | 15 / 35 | 23 / 50 | 15 | 23 |
| BAL-G_s0 | 26 / 62 | 14 / 23 | 14 | 26 |
| BAL-G_s1 | 34 / 113 | 17 / 28 | 17 | 34 |
| BAL-L_s0 | 11 / 9 | 34 / 103 | 11 | 34 |
| BAL-L_s1 | 13 / 17 | 46 / 160 | 13 | 46 |

## Numbers (tactical ADE, knots 11-30, m; variant minus OWN; [95% CI]; * separated; floor in m; g = gain, c = cost by the card's rule)
Estimator answers "another draw of episodes" only.

| arm | variant | S: stop-point windows | A: all has-and-present windows |
|---|---|---|---|
| BAL-G_s0 | HGOAL | -0.0520 [-0.094, -0.019]* floor 0.022 g | +0.0059 [+0.001, +0.012]* floor 0.001 c |
| BAL-G_s1 | HGOAL | -0.0415 [-0.081, -0.009]* floor 0.022 g | +0.0096 [+0.002, +0.020]* floor 0.001 c |
| BAL-L_s0 | HLON | -0.0877 [-0.156, -0.034]* floor 0.021 g | +0.0076 [+0.000, +0.015]* floor 0.007 c |
| BAL-L_s1 | HLON | -0.0814 [-0.179, -0.009]* floor 0.021 g | +0.0173 [+0.006, +0.030]* floor 0.007 c |

Beside (HARD, both parts): S: G_s0 -0.0549*, G_s1 -0.0439*, L_s0 -0.0889*, L_s1 -0.0764 (not separated). NULL arms, HARD on S: s0 -0.0141, s1 +0.0007, rep -0.0518* (floors 0.038). OWN mean tactical ADE on S: 4.84 to 5.11 m on the five non-rep arms, NULL_rep 4.055; on A 2.93 to 3.02 m.
TACTICAL family: n = 0 (T's decision is the transformed input). STRATEGIC: PAUSED (PI item 25). The other families (FDE, speed and acceleration MAE, heading and yaw-rate MAE, both bands) are in the JSON per arm, variant and set.

## Outcome labels computed by the code
BAL-G (HGOAL): `HARD-TRADES` (S gain at s0 and s1; A cost at both). BAL-L (HLON): `HARD-TRADES` (S gain at s0 and s1; A cost at both). Controls held (`controls_ok` true; stop-point count 117 on every arm).

## Temporary data
Scratch `F:/TanitAD-scratch/stophard_build/` (seven per-arm npz banks, smoke npz, run log; 55 MB): re-creatable by `stophard_read.py --stage run`; scratch npz sha256 are in `RESULT_STOPHARD.json` `banked`; deleted at the end of the job.
