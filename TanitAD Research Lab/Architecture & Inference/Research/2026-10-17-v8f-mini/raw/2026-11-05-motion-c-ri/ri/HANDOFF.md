# HANDOFF: REINDEXED tool built, tested, smoked, chain launched 2026-11-05T11:52:59+01:00. A fresh agent finalises it after the chain's marker.

Package `E:/Projects/TanitAD-artifacts/dagland/motion_c/ri/`. Scratch `F:/TanitAD-scratch/motion_c/ri/` (`tree_ri/` = stack/ + taniteval/ copied from `recover_read/tree_dag`, plus the tool and the test; `out/` = the 4 rollouts; `smoke/`). Nothing is in git (the Master Mind lands).

## State
* **Tool** `code/v8f_t1_eval_ri.py` sha256 `0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7` = the FRAMEFREEZE tool `0721fc01...` + `code/apply_ri_patch.py` (each replacement matches once; regenerates the same sha). New `--scene reindexed`: rows i > t0 of the item's frame history take the logged cached frame at `min(i, tau*)` (tau* = arc-length-nearest logged index to the projection of the model's own executed (x, y); running maximum over the item's rows); rows i <= t0 stay logged bit for bit (byte-compared). Header block `scene_reindex` (model top level, `floor` beside it). `logged`, `frozen`, `hold_now` and flag absent are bit-identical to the source tool.
* **Tests** `logs/pytest_ri.log` **12 passed** (KRI1 identity, KRI2 half speed, KRI3 ahead, KRI4 bit-identity CLI + unit, KRI5 past-row guard, mutations, REACH, parser/refusals, pure projection rules). `logs/mutations.log`: m1 apply line deleted -> KRI2 RED; m2 `min(i, .)` dropped -> KRI3 RED; m3 past mask dropped -> KRI5 RED and the receipt's own past-row check RED.
* **Smoke** `logs/smoke_ri.out` SMOKERI OK (C0 s0, 32 windows, 8 threads); `logs/smoke_composite_s1.out` (s1, `--feasible kbox` composite, 6 windows).
* **Chain** `ri_c0_chain.cmd` -> `ri_c0_chain.sh`, launched via WMI (cmd PID 35452), 12 threads, one rollout at a time, order `RIC0_s0`, `RIC0_s1`, `RIC0C_s0`, `RIC0C_s1` (RIC0C = `--scene reindexed --feasible kbox`), all with `--smooth sg72 --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu`. Out `F:/TanitAD-scratch/motion_c/ri/out/<tag>` (+ `done_<tag>.txt`). Then `code/ri_receipts.py` gates the 4 receipts -> `raw/RI_RECEIPTS.json`. RIC0_s0 progress log read 3.5 s per 32 windows at 11:53:35 (about 4.5 min of rollout plus the analysis); expect ~6 min per rollout, all four and the receipts done ~12:20 (the first `ZZRI-RUN-DONE` line confirms).
* **Markers** `raw/RI.done` (all four rollouts + all receipt checks passed) / `raw/RI.fail` (reason inside). Logs `logs/chain.out` (`ZZRI-*`), `logs/run_<tag>.out`, `logs/receipts.out`.

## Check (ONE command, no sleep loop)
`ls raw; tail -n 5 logs/chain.out`. `.fail`: read the reason (a rollout crash: `logs/run_<tag>.out`; a receipt check: `raw/RI_RECEIPTS.json` `failures`). RESUME = launch `ri_c0_chain.cmd` again via WMI `Win32_Process.Create` (finished runs are skipped; an incomplete out dir is moved to `_incomplete_*`; a live chain refuses a second start via `raw/RI.running`).

## Next step for the reader (not done here)
Read `out/<tag>/t1v8f_<tag>.json` + `ticks/` as the `scene reindexed` rows of RECOVER-FT; rerun `python -I code/make_manifest.py` after the marker.

## For the Master Mind (integration)
Land `code/`, `tests/test_v8f_t1_ri.py` (belongs in `stack/tests/`), the tool (belongs in `stack/scripts/` beside the FRAMEFREEZE tool; the test loads that tool by file path, sha pinned). Disclosed limits: longitudinal only (lateral offset of the projection is reported: smoke p90 0.9 m, max 9.2 m); dynamic agents shown at the human's time; T2 not provisioned. The largest smoke plan difference (132.2 m, one 3 m/s window at tick 2) is between two off-distribution plans of the C0 model, not a tool effect.
