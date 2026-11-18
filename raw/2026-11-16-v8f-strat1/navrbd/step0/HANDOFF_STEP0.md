# HANDOFF_STEP0 (2026-11-18): NAVRBD step 0, held-out floor F_HO. Stopped by the MM for UM_s0. No pod command was issued after the stop message.

## State
Built and verified (MEASURED, CPU, zero training):
* navtok table of the 10k join by the unedited navtok `build_table.build` (`code/build_navtok_x.py`). Proof: the same path on eval141 equals the pinned `navtok_table.npz` eval split bit for bit (cls_id, dist_m, flag, age_s, clip/off/n, vocab). `out/navtok_proof_eval141.npz.json`.
* Text-nav, nav_rb state table and RB grid by the tree's own builders (`code/build_inputs_x.py`, tree = pod `/root/strat1/navrb2/tree`, builder and eval-wrapper sha256 identical to the local e71782a3-equivalent tree). Proof on eval141 on the pod: eval_nav, eval_nav_d, eval_rowclamp and eval_nav_rb equal the pinned files bit for bit (`/root/strat1/navrbd_step0/out/proof/PROOF_eval141.json`).
* 10k inputs, `/root/strat1/navrbd_step0/out/inputs/` (INPUTS_SHA256.json there): navtext_nav_x.npz 33442548..., navtext_nav_rb_x.npz ce8f6135..., eval_grid_x.json f5771c0a..., navtok_table_x.npz a5c04fa2..., join_X.npz f383ab78... (equals the local copy), tracks 7eb96925.... Grid = 7,594 windows / 545 episodes (asserted), G-CLOCK-NAV and G-CLOCK-RB pass, 36,951 RB steps (equals RB10K_COUNTS), regression arm FAILS as required.
* Checkpoints from private HF `Sayood/tanitad-v8f-strat1-runs`, sha256 equal to the hub LFS sha: N1R_s0 35dc943e..., N1R_s1 32b0336d.... Local copies `F:/TanitAD-scratch/strat1/navrbd/step0/in/ckpt/` and pod `/root/strat1/navrbd_step0/ckpt/`.

## Finding that needs an MM ruling
The T1 tool refuses a window without a complete 6 s future (gt_ok False). Of the 7,594 RB grid windows, 1,858 fail it (0 beyond clip length). Feasible grid: **5,736 windows / 497 episodes** (`out/inputs/eval_grid_x_feasible.json`, sha ca289d0f...; `FEASIBLE_GRID.json`). The card's 7,594 / 545 cannot be dumped by the pinned tool; D1-HO would be on 5,736 / 497 unless the MM rules otherwise.

## Partial outputs: must be redone
The two dumps (N1R_s0, N1R_s1) were started detached and killed by the MM; `/root/strat1/navrbd_step0/out/t1/` is partial (no DONE file) and must be deleted before the re-run. `out/smoke/` is a 20-window code check, never a result. Pod cache view `cache_view/eval` is a symlink to `/root/data/cache_x_dscale/train`.

## Exact re-run (after UM_s0 finishes; check `nvidia-smi` and `ps` for v8f_mini_train first)
```
rm -rf /root/strat1/navrbd_step0/out/t1 /root/strat1/navrbd_step0/out/smoke     # own temp, literal paths
ssh <pod> "cd /root/strat1/navrbd_step0 && nohup setsid bash code/dump_ho.sh N1R_s0 < /dev/null > logs/dump_N1R_s0.out 2>&1 &"   # own ssh command
ssh <pod> "cd /root/strat1/navrbd_step0 && nohup setsid bash code/dump_ho.sh N1R_s1 < /dev/null > logs/dump_N1R_s1.out 2>&1 &"   # own ssh command
```
Sentinels `out/t1/DONE_<ARM>` / `FAIL_<ARM>`. Cost MEASURED 0.15 s per window (2 threads), so about 14 min per arm, both in parallel.

## Still to do
1. Pull `out/t1/dump_NAVRBD-HO-*_m10_causal_heads` to F:.
2. Compute TAC-LAT per window locally: the plan track `cl` through `read_otband.labels` (as `obd_geom`, `labels` node only) against the label from the GT track `g`; accuracy = (pred == label). Dumps loaded with `strat1_read.load_dump_n(dir, 497)`.
3. D1-HO per arm, descriptive accuracy and class shares, F_HO = |s1 - s0| with the paired episode-cluster bootstrap (`taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0), K1 self-pair (exactly 0.0), K-CONST majority check; write `out/HO_FLOOR.json` and its sha256.
4. Delete own temp: `F:/TanitAD-scratch/strat1/navrbd/step0/out/navtok_proof_eval141.npz` after the report is banked; pod folder stays until step 0 ends.

## Deletions log
None this session. Local download of the two ckpts (86,976,357 bytes each) remains in `step0/in/ckpt/` (re-pullable).
