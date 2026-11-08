# FC_FINDING: frame-control failure in the H-SEL-1 confirmation dump (stream YSEL2-FC)
Evidence class: MEASURED, CPU only, no model, no GPU. Scripts and data in this folder (`fc_census.py`, `raw_all.json`, `fc_census.json`). Nothing in the pinned package, card, markers or `dump/chunks/` was edited. No scoring output was opened.

## 1. Root cause (the export, not the DB sampling)
- Check: `selc_dump.py:236-244` compares `sc.get_ego_future_trajectory(0, 4.0, 8)` (ego frame) to the W3 export `human_future_poses`.
- nuPlan sampling: `nuplan_scenario.py:332-343` calls `sample_indices_with_time_horizon(8, 4.0, 0.05)` (`scenario_utils.py:4-29`): step = 81//8 = 10 rows, so rows +10,+20,...,+80 after the anchor, then `get_sampled_ego_states_from_db` (`nuplan_scenario_queries.py:450`, `ORDER BY timestamp`, `row_num`). This is a ROW offset, i.e. a nominal 0.5 s grid. On the failing tokens the DB rows ARE on that grid (db_dt 0.5,1.0,...,4.0 within 1 ms).
- The export (OpenScene `navsim_logs/test/<log>.pkl`, ~2 Hz frame list) is the 8 NEXT FRAMES of that list. In two logs that list is MISSING a frame (inter-frame gap ~1.0 s instead of 0.5 s). From the gap onward the export's step k is the pose at (k+1)*0.5 s, not k*0.5 s: a one-step time shift, so error 2.2 to 5.1 m (vehicle moving ~4.4 to 10 m/s).
- Proof: re-pairing each export step with the DB pose at the META timestamp of that frame (DB 4.5 s horizon, slot = round(meta_dt/0.5)-1) gives max error <= 3.7e-15 m on all 17 failing tokens (`explained_err_m_vs_meta_timed_db_slot`). Export poses are exact; only the time labels of the grid differ. Anchor timestamps: meta == DB on all 5877 tokens.
- Contrast: the seam doc (`refe_navtest_seam.py:43-44`) reported 0.0000 m on 25/25 tokens, a sample that contained no gap token.

## 2. Census (all 75 logs, 5877 tokens, 4 workers, wall 147 s)
- Fail (> 1 mm): **17 tokens (0.29 %) in 2 logs**, both among the 61 pending; 0 in the 14 passed logs. Unknown/NaN: 0. Unexplained: 0.
- `2021.06.03.12.02.06_veh-35_00233_00609`: 9 tokens (ids in `fc_census.json`). Max error 5.1237 m (frame dropped at step 5 to 6) or 2.2031 m (dropped at step 1 to 4).
- `2021.06.03.17.06.58_veh-35_02943_03220`: 8 tokens, all max 2.4174 m (gap at step 1 to 8).
- Full list with per-step errors and gap positions: `fc_census.json`. Pending-log tokens checked: 5194.
- The run would have crashed on the 2nd bad log too, so a plain re-run would die twice more (once per bad log).

## 3. Does it touch what the experiment measures?
- Model inputs: `planner.infer(ego, img)` (`selc_dump.py:225-229`) uses ego state + images + driving command from `exp[tok]` (`set_driving_command`); no future. Proposals, logits, pick: unaffected.
- `gt_future` (`selc_dump.py:74-87`, DB 20-sample future with true timestamps): unaffected, feeds `gt_valid`, cruise mask and `v4_gt` (`selc_analyze.py:43-46,225-244`).
- Scoring: `selc_score.py` has no reference to `human_future_poses`, `human8` or `frame_ctrl_m` (grep: 0 hits), i.e. PDMS columns do not depend on it.
- `frame_ctrl_m` is record-only: stored at `selc_dump.py:242`, enforced only at `:277`; no `selc_analyze/gates/rule/chain` hit.
- ONLY consumer of the export future: `human8` -> `ade()` at `selc_analyze.py:123-137` (secondary `d_ade_human_m_cruise`, `ade_human_m_cruise_base`). Up to 17 tokens (only those in the cruise mask) have a time-shifted human reference there. Primary H-SEL-1 PDMS read is untouched.

## 4. Two options for the Master Mind (not applied)
**Option A: change the check's reference (keep all 5877 tokens).** Dated amendment before any fresh score is read: the frame control compares the export to the DB future sampled at the export's own OpenScene timestamps (4.5 s/9-sample DB read, slot from `meta_dt`). Verified here: 5877/5877 <= 1 mm (17 gap tokens at 4e-15 m), a stronger statement than the card's. Consequence: the pre-registered token set, strata and power are unchanged; the gate text in section 6.6 is reworded, `selc_dump.py` and thus `CODE_PIN.json` need re-pinning (the 14 finished chunks store `frame_ctrl_m` from the old code, all <= 1 mm, so they stay valid). The secondary human-ADE keeps 17 time-misaligned reference rows unless `human8` is re-timed or those tokens masked in that secondary only.
**Option B: exclude the 17 named tokens (list in `fc_census.json`) by dated amendment.** Keep the 1 mm check as written; the dump skips these tokens. Consequence: N falls 5877 to 5860 (0.29 %), the strata/power calculation and `tokens_confirm.json` provenance change (a re-pin of tokens and code anyway); the 2 bad logs then lose 17 of their tokens (not whole logs) and the primary and secondary reads both run on the clean set. Cost: a reviewer can say tokens were dropped after the design, mitigated by the exclusion being by an input-quality rule fixed before any fresh score, and non-outcome-dependent (it is a property of the export file).
Both need a code change and a re-pin; neither can change a score. Recommendation (non-binding): A, since it keeps the pre-registered population and the root cause is a mislabeled time grid.
