# w_field by the draft's gradient-norm balance — measured on P7, CPU (2026-11-17)

**Frozen value: `--w-field 2.0`** (the smallest candidate with r(w) >= 0.25; r(2) = 0.394). Never tuned on PDMS: the rule reads no PDMS, no navtest label, no score.
Procedure: exactly `PREREG_AUX_FIELD_LOSS_DRAFT.md` "Weight rule" + its 2026-10-13 clarification, run by the committed tool `refe/weight_rule.py` (sha256 `81f0ef0d8d14…` at tip `6b789b7e`), driven by `code/fieldp7_wr.py`.
Seed 20261012, 8 micro-batches of `--batch 4` (P7's micro-batch), gradients pooled over the 8 (what `--accum` hands the optimiser), `r(w) = ||grad_traj_head (w L_field)|| / ||grad_traj_head l_traj||`, `traj_head` = 81,212 parameters.
Init = P7 `model_final.pt` (`2613fc17…`), strict warm start through the real `train.run`. `l_traj` = `wta_loss` with P7's `--yaw-wrap off`.
Artifact: `raw/weight_rule_p7_cpu.json` (sha256 `90ae1b30…`), log `raw/weight_rule_p7_cpu.log`. MEASURED.

| w | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| r(w) | 0.0493 | 0.0985 | 0.197 | **0.394** | 0.788 | 1.576 | 3.152 | 6.305 | 12.61 |

Pooled `||grad l_traj||` = 7.560, pooled `||grad L_field||` (w = 1) = 1.490, r(1) = 0.1970. K2 (r(64) < 0.25) NOT triggered; K2b (r(1/4) > 1.0) NOT triggered. r is exactly linear in w (the loss enters as `l_traj + l_score + w L_field`), so the table is r(1) x w.

Per micro-batch (positions in the seeded permutation; `micro_batch_ids` with log/token/rank are in the JSON):
| mb | l_traj | L_field (w=1) | ‖g l_traj‖ | ‖g L_field‖ | r1 |
|---|---|---|---|---|---|
| 0 | 0.1362 | 0.01589 | 17.80 | 1.800 | 0.1011 |
| 1 | 0.1355 | 0.02915 | 16.39 | 3.568 | 0.2176 |
| 2 | 0.0993 | 0.02671 | 20.18 | 4.960 | 0.2458 |
| 3 | 0.0874 | 0.01481 | 11.75 | 0.905 | 0.0771 |
| 4 | 0.0955 | 0.00195 | 15.08 | 0.126 | 0.0083 |
| 5 | 0.1371 | 0.00994 | 6.80 | 1.170 | 0.1719 |
| 6 | 0.1771 | 0.01304 | 24.12 | 1.418 | 0.0588 |
| 7 | 0.1137 | 0.01277 | 13.58 | 0.793 | 0.0584 |

Per-micro r1: min 0.0083, median 0.1011, max 0.2458 (diagnostics only; the pooled ratio decides). Label coverage of the 8 batches 1.000; in-extent corner share 1.000; **violating share 0.0197** (2 % of footprint corners of the 64 proposals are outside the drivable area on this pool).

## Deviations from "the arm's own bank" — MM to accept or order a re-measure (UNVERIFIED on the real bank)
1. **Pool.** The draft says 8 micro-batches of the arm's own bank (P7's reserve-free 3-C bank, 150,603 tuples, on the pod). That bank and its pixels are pod-only; the pod is off limits. The 32 rows are drawn (seed 20261012) from the **local rank-0 navtrain bank** `E:/Projects/TanitAD/data/refe_m5/bank_train` (1,600 rows, 162 logs, all with real pixels, all in the pinned label store, all in the local yaw table). Same corpus (navtrain), same label store, same model; a different row set (and rank-0 teacher rows only).
2. **Command.** `--nav-mode command` with `goal_rule` commands (the rows carry no NAVSIM command), where P7's bank carries NAVSIM's own.
3. **Yaw table.** P7's pod-built table (sha `3a0873cc…`) is not local; the local partial devbox table (`517c6413…`, 44,304 tokens, covers all 1,600 rows) holds the same `ego_pose.angular_rate_z` source.
4. **Scorer** = zero placeholder (no local scorer targets); irrelevant to `traj_head` gradients. **cv2 shim** (PIL bilinear), see REACHABILITY.
5. **Memory trick.** Every parameter outside `traj_head` was set `requires_grad=False` before `measure()`; the gradient w.r.t. `traj_head` does not depend on that (the committed tool is otherwise unchanged).
6. **Sensitivity.** w = 1 gives r = 0.197 (< 0.25), so the choice of 2 rests on r(1) lying between 0.125 and 0.25. Per-micro spread is wide (0.008 to 0.246); 8 micro-batches is the draft's n.
**Committed handling (proposed amendment, MM to ratify before landing):** the value 2.0 is FROZEN now. The launch gate re-measures r(w) on the arm's own bank and REPORTS it; it may only veto (K2 / K2b), never re-pick w. A re-pick after seeing the real bank would be tuning.
