# E2-RAW stage B: the E2 rule on the RAW step-6,000 weights of K2DC and K0 -- verdict NO-GO

Pre-registered: `../PREREG_E2RAW_2026-11-16.md` (landed 232f8738, before any raw number). Stage A (map, clause a): `../README_STAGE_A.md`.
Stage B ran the E2 roll (`r9e2_roll.sh` with `R9E2_CKPT_NAME=ckpt.pt`, landed 1c48ee8f) on Thor, seeded with stage A's raw DEV dumps and
FIT-threshold EVALs (so clause (a) is scored at each arm's own FIT thresholds, as pre-registered); S6c and the verdict ran on the dev box.
The read's own verdict file is quoted below, unedited (`read/E2_VERDICT.md`). One training run per arm (`H-ESTIM-SEED-1`): a separated
interval answers another draw of EPISODES only. STRATEGIC: as the read reports it.

## The read's verdict (quoted)

# E2 verdict: NO-GO  (K2 vs K0)

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (b) box AP@2 m head output no NMS: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **PASS** | T1 self-action open loop (S6c-DEV) | - |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

GO iff K2 beats K0 in point estimate on the map primary, box AP@2 m and the planner's S6c sidewalk rate, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4).
NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

## Context

* The original E2 verdicts (K2M1N20, K2DC on `model_ema.pt`) are NO-GO and STAND as written. This is a separate, pre-registered read;
  a GO here goes to the PI, who decides whether E3 fires (its approval is tied to E2 GO).
* Paired diagnostic of clause (a) (`../paired_diag/FLOOR.md`, decides nothing): K2DC raw - K0 raw nf_swedge_prec_lv_5_20 +0.0182 [-0.0044, +0.0450] (not separated); 5-10 m +0.0348 [+0.0049, +0.0707] and 0-5 m +0.0727 [+0.0218, +0.1258] separated better (paired episode-cluster bootstrap, B 2000, 68 episodes, T0).
* Provenance: Thor `runs/E2RAW_K2DC/stageB/` (`logs/ROLL_STATUS.txt`), read tree `read_tree_9697fa4df004_k2dc_raw1c48ee8f` (K2DC's
  E2 read tree + the landed roll 3816129936e8 / env d4bdcf07cf33); every raw dump records weights_are_ema false, step 6000.
  The read files are byte-exact copies of the dev-box pull, which the S6c driver verified by sha256 against Thor.
