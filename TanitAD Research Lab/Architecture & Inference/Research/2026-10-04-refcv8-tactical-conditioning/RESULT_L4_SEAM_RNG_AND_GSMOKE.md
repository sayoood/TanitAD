# WP-B L4 and the L3 G-SMOKE reads (2026-10-05)

## 1. The L3 full-size G-SMOKE (Thor, tree_L3 = 4785e5e, 2026-10-05 00:22–00:35 UTC)

MEASURED. Artifacts are in `/home/nvidia/refcv8_run/runs/refcv8-wpb-smoke-L3/`. The run is a 30-step warm start from
refcv7-50,400 (md5 d5f104ee) with `--r8-alloc-emit-start 10` and `--smoke-seed-ego-frames`.

| read | result |
|---|---|
| `vis1_rekey_check.json` | **PASS**: the seeded first batch held 16 ego-stripped windows, all 16 re-keyed; `n_ego_rows_removed` (train) = 693; step 30 reached |
| S1 reached end + warm start | **PASS**: 1,131 keys loaded, 34 new refcv8 keys, optimiser fresh |
| S2 r8 terms live and finite | FAIL, **for the missing eval row only**. `r8_missing_or_nonfinite` is empty on every training row |
| S3 grad-share | FAIL on linearity: `gs_trunk_lin_rel_err` = **5.9e-3** against the 1e-4 bar under `--trunk-bf16`. The fp32 groups read bev025 9.3e-5 and bev_pool 6.0e-5 |
| S4 seams took gradient | **PASS**: 34 of 34 refcv8 tensors moved |
| S5 cost | **PASS**: 9.2 s/step against 10.5. Only ONE clean interval (steps 10→20) |
| p7 X1 / X2 / X3 / X5 | PASS. X4_rows fails for the missing eval row |

**First full-scale gradient shares** (step 30 of a warm start; these are readings, not a lever claim): agent 0.537,
box3d 0.162, map_hires 0.0011, refcv8 0.00077, traj 0.00016, tac_v6 −0.00082. They follow the ordering of P-GRAD's refcv7-50,400 shares.

**Why the eval row is missing (MEASURED, kernel log).** A GLOBAL OOM fired at 02:35:03 local, in the in-run eval phase. The eval runs in the main process with batch 16; its anon RSS was 18.5 GB. At the same time WP-RL was resident while yielding the
lock, at ~16 GB main RSS plus 4 workers. The kill also took desktop daemons. The GPU lock serialises the GPU, not host RAM.

## 2. Defect: the refcv8 seams consumed the global RNG. Fixed in `refc_v3.py`.

* **Observed** (MEASURED, dev-box CPU, 1 step each, same seed, same first batch; the fed-speed CRC was 4015187787 on both arms):
  V0 against V-R8, the agent terms matched on 36 of 36 keys. The 10 cm map terms differed on 184 of 354 keys (drivable 0–20 m intersection 2,061 against 58), and box3d on 18 of 53.
* **Mechanism.** `enable_refcv8` built the seams inside `RefCV3Model.__init__` from the GLOBAL stream. The trainer builds
  the perception branch and the map branch AFTER the model, so V-R8's map and box3d heads started from a different
  init. That is a second variable in every from-scratch refcv8-vs-refcv7 comparison, and it failed the ladder's registered I-0 bar.
  A warm start is unaffected, because the checkpoint overwrites those heads.
* **Fix.** The seams are built inside `torch.random.fork_rng`, from a dedicated stream (`cfg.refcv8.seed + R8_INIT_SEED_OFFSET`).
* **After the fix** (MEASURED, same setup): I0-a PASSES on 354 of 354 map keys. Agent (36/36) and box3d (53/53) are also identical, and MM-2 passes.
* Pinned by `stack/tests/test_refcv8_seam_rng.py`. Its RED arm is a fork that does not restore.

## 3. The gradient-share fp32 replay (S3 option (i), pending the MM's ruling)

Under a bf16 trunk, the gradient of the sum is not the sum of the per-term gradients to better than about bf16 epsilon. The trunk-group
linearity control therefore cannot meet 1e-4. `grad_share.fp32_replay` reads the statistic on a replay of the SAME step instead:
* every bf16 lever is switched OFF for the replay;
* the global RNG is forked, the dedicated generators are restored (and any the replay itself created are dropped), and the buffers are restored;
* the trainer takes the replay before its training forward.

Pinned by `stack/tests/test_refcv8_grad_share_fp32.py`. On CPU bf16, the direct reading misses 1e-4 and the replay meets it.
The training step after a replay is bit-identical, loss and every `.grad`.
**Not yet measured:** the replay at full size on the GPU. It needs the G-SMOKE re-run.

## 4. Gates (dev box, the MM's rule)

| landing | tip | candidate | candidate-only failures |
|---|---|---|---|
| L3 | 3cd48fd: 7 failed / 2,526 passed | 7 / 2,555 | none |
| L4 | 4785e5e: 7 failed / 2,555 passed | 7 / 2,564 | none |
