# HANDOFF: BATTERY stream (2026-11-08)

State: core statistics DONE and tested (19 passed). Nothing running. Nothing staged or committed (artifacts tree only).

Use: `CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe scorer_battery.py --out <json>` (about 1 min) runs u400, u400-without-lane-factor and the four regression arms. For a new arm (e.g. an S1 or C0 checkpoint's logits on the C3 table): `import scorer_battery as sb; T,_ = sb.load_table(); boot = sb.make_boot(T); res, tok = sb.run_arm(T, boot, sb.arm_from_logits("S1", logits))`, then `sb.pair_vs(boot, T, tok_C0, tok_S1)` for paired deltas on the same draws. Logits must be [1058,64,7] in the C3 token order (`T["token"]`). For E2 (a new token set) `load_table()` needs a sibling that builds the same dict (token, log, stratum, cruise, pdms, sub, valid, logits, v4_all, v0, v4_gt, gt_valid, traj, gt_pose20); the stats take that dict and nothing else.
Tests: `CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_scorer_battery.py -p no:cacheprovider --basetemp=<existing parent>/x` (about 1 min; needs torch).

Open items, in order:
1. Lane truth: see LANE_TRUTH.md (reachable on CPU, not built; needs a teacher-rollout queue for the 1,058 tokens).
2. Mutation checks are input mutations per statistic (shifted pick, reversed score, swapped pred/true, scrambled ratio, offset plan, swapped stratum weights). Source-line mutation (delete the emitting line, require RED) was not done.
3. For P6/S1 reads wire `PREREG_SCORER1` M1/B1/G3 (mean over TTC and NC of the Spearman gap, slow-share drop, regret delta) on top of `pair_vs`; all inputs exist in the JSON structure.
4. Tactical family is n/a at this tier; closed-loop families belong to the gym scorecard.
5. Temp left behind: pytest `--basetemp` folder `F:/tmp_battery_pt` (my own, a few KB; the auto-delete was refused by the safety check, so it still exists). Delete it by hand: `rm -r F:/tmp_battery_pt`. `__pycache__` in this folder is also scratch.

Deletions logged: none.
