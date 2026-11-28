# Landed by the Master Mind 2026-10-25 ~22:35 Berlin (from the dev box `C:/Users/Admin/refcv8_pf/gtmap_oracle/`)

* `run/P4.json`: the four raw clip UUIDs (the two `no_gt_clips`, listed twice) are replaced in THIS copy by
  `sha12:<sha256(clip_id)[:12]>`; the dev-box original is unchanged. No number in the file is affected.
* Left on the dev box: `run/p4_s{0,1}.npz`, `run/dump_s{0,1}/`, the roll logs `run/roll_s{0,1}.log` (they print clip-id
  prefixes), the launch tree `tree_fb47a58/` and its tar, the aborted runs (`run_aborted_budget_run2/`, `chain.log.run1_abort_util`,
  `chain.log.run2_haltbudget`) and the smoke dirs. The delta cache was on `F:/p4_gtmap/`.
