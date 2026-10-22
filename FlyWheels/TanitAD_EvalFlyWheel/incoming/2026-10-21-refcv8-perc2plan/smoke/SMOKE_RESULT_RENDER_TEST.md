# PERC2PLAN-R8 — does refcv8's planner USE its perception inputs?

* Rendered 2026-10-21 19:52:59 +0200 from `smoke/SMOKE_ANALYSIS_s1.json`; prereg `PREREG_PERC2PLAN.md` sha256 `d1a2eb53d4488dde404b4c30b84ddb9d35bfa0aead8564fa07d729d4e2b68568`.
* **Tier: T1 self-action OPEN LOOP on logged eval139 frames (the SHIP-R8 tier); ONE training seed.**
* **Interval:** paired episode-cluster bootstrap (refav1_arm._paired_families), B 200 -- answers 'another draw of EPISODES' only; inference-run variance is read from the second inference seed (floor).
* Model: refcv8 EMA-48,852, per-frame nav feed, eval139 (4,754 windows / 139 episodes), os-only arm set, inference seeds 0 and 1, Thor.

## Verdicts (mechanical, PREREG §4)

| arm | headline | S2 verdict | S6 verdict | S2 ADE Δ s0 | S2 ADE Δ s1 | S6 ADE Δ s0 | S6 ADE Δ s1 |
|---|---|---|---|---|---|---|---|
| AGT_ZERO | **INCOMPLETE** | INCOMPLETE (needs both seeds + floor) | INCOMPLETE (needs both seeds + floor) | n/a | -0.0022 [-0.0034, -0.0009] | n/a | -0.0213 [-0.0266, -0.0160] |
| AGT_SHUF | **INCOMPLETE** | INCOMPLETE (needs both seeds + floor) | INCOMPLETE (needs both seeds + floor) | n/a | -0.0054 [-0.0083, -0.0025] | n/a | -0.0126 [-0.0130, -0.0121] |
| BEV_ZERO | **INCOMPLETE** | INCOMPLETE (needs both seeds + floor) | INCOMPLETE (needs both seeds + floor) | n/a | +0.4486 [+0.1829, +0.7143] | n/a | +2.3502 [+2.2562, +2.4442] |
| BEV_SHUF | **INCOMPLETE** | INCOMPLETE (needs both seeds + floor) | INCOMPLETE (needs both seeds + floor) | n/a | +0.1623 [+0.0812, +0.2434] | n/a | +1.2695 [+1.2360, +1.3029] |
| BOTH_ZERO | **INCOMPLETE** | INCOMPLETE (needs both seeds + floor) | INCOMPLETE (needs both seeds + floor) | n/a | +0.5322 [+0.2246, +0.8397] | n/a | +2.3627 [+2.2285, +2.4969] |

ADE Δ = arm − BASE at the same inference seed, metres, 95 % paired episode-cluster CI.

### Inference-seed floor (BASE_s1 − BASE_s0)


### Separated components at BOTH seeds

* AGT_ZERO S2: worse None · better None
* AGT_ZERO S6: worse None · better None
* AGT_SHUF S2: worse None · better None
* AGT_SHUF S6: worse None · better None
* BEV_ZERO S2: worse None · better None
* BEV_ZERO S6: worse None · better None
* BEV_SHUF S2: worse None · better None
* BEV_SHUF S6: worse None · better None
* BOTH_ZERO S2: worse None · better None
* BOTH_ZERO S6: worse None · better None

## All four families, per seed (arm − BASE)

### s1 — n_windows S2 6, S6 6, episodes 2

**S2**

| family.metric | BASE mean | AGT_ZERO | AGT_SHUF | BEV_ZERO | BEV_SHUF | BOTH_ZERO |
|---|---|---|---|---|---|---|
| ADE.ade_m | 0.115803 | -0.0022 [-0.0034, -0.0009] **sep** | -0.0054 [-0.0083, -0.0025] **sep** | +0.4486 [+0.1829, +0.7143] **sep** | +0.1623 [+0.0812, +0.2434] **sep** | +0.5322 [+0.2246, +0.8397] **sep** |
| ADE.fde_m | 0.231274 | -0.0057 [-0.0079, -0.0035] **sep** | -0.0096 [-0.0105, -0.0087] **sep** | +0.7512 [+0.4103, +1.0921] **sep** | +0.3229 [+0.1403, +0.5056] **sep** | +0.8339 [+0.4698, +1.1981] **sep** |
| longitudinal.LON_speed_mae_mps | 0.117942 | -0.0023 [-0.0048, +0.0003] | -0.0032 [-0.0078, +0.0014] | +0.3648 [+0.1677, +0.5620] **sep** | +0.0078 [-0.0586, +0.0742] | +0.3899 [+0.1618, +0.6180] **sep** |
| longitudinal.LON_along_mae_m | 0.087893 | +0.0015 [-0.0007, +0.0037] | +0.0008 [-0.0023, +0.0040] | +0.4631 [+0.2115, +0.7146] **sep** | -0.0021 [-0.0155, +0.0113] | +0.5372 [+0.2344, +0.8400] **sep** |
| longitudinal.LON_accel_mae_mps2 | 0.212365 | -0.0061 [-0.0136, +0.0014] | -0.0077 [-0.0174, +0.0021] | +0.2746 [+0.2405, +0.3086] **sep** | +0.0592 [-0.0569, +0.1753] | +0.2682 [+0.2207, +0.3158] **sep** |
| lateral.LAT_cross_mae_m | 0.045569 | -0.0027 [-0.0041, -0.0013] **sep** | -0.0041 [-0.0076, -0.0006] **sep** | +0.0166 [+0.0032, +0.0299] **sep** | +0.1932 [+0.0929, +0.2935] **sep** | +0.0291 [+0.0024, +0.0559] **sep** |
| lateral.LAT_heading_mae_deg | 0.384392 | -0.0347 [-0.0462, -0.0232] **sep** | -0.0623 [-0.0891, -0.0355] **sep** | +0.7347 [+0.2246, +1.2449] **sep** | +1.7417 [+0.6952, +2.7882] **sep** | +0.7981 [+0.1985, +1.3977] **sep** |
| lateral.LAT_yaw_rate_mae_radps | 0.009166 | -0.0002 [-0.0003, -0.0001] **sep** | -0.0002 [-0.0005, +0.0001] | +0.0231 [+0.0013, +0.0376] **sep** | +0.0284 [+0.0100, +0.0468] **sep** | +0.0253 [+0.0035, +0.0398] **sep** |
| lateral.LAT_curvature_mae_1pm | 0.001943 | -0.0000 [-0.0001, +0.0001] | -0.0000 [-0.0001, +0.0001] | +0.0046 [+0.0006, +0.0073] **sep** | +0.0046 [+0.0017, +0.0074] **sep** | +0.0051 [+0.0015, +0.0075] **sep** |
| lateral.LAT_yaw_rate_mae_masked_radps | 0.009166 | -0.0002 [-0.0003, -0.0001] **sep** | -0.0002 [-0.0005, +0.0001] | +0.0231 [+0.0013, +0.0376] **sep** | +0.0284 [+0.0100, +0.0468] **sep** | +0.0253 [+0.0035, +0.0398] **sep** |
| tactical.TAC_traj_lat_correct | 1.0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1667 [-0.3333, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| tactical.TAC_traj_lon_correct | 0.833333 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.1667 [+0.0000, +0.3333] | +0.0000 [+0.0000, +0.0000] | +0.1667 [+0.0000, +0.3333] |

**S6**

| family.metric | BASE mean | AGT_ZERO | AGT_SHUF | BEV_ZERO | BEV_SHUF | BOTH_ZERO |
|---|---|---|---|---|---|---|
| ADE.ade_m | 1.252861 | -0.0213 [-0.0266, -0.0160] **sep** | -0.0126 [-0.0130, -0.0121] **sep** | +2.3502 [+2.2562, +2.4442] **sep** | +1.2695 [+1.2360, +1.3029] **sep** | +2.3627 [+2.2285, +2.4969] **sep** |
| ADE.fde_m | 3.637143 | -0.0806 [-0.1213, -0.0400] **sep** | -0.0710 [-0.0763, -0.0658] **sep** | +4.3839 [+4.2591, +4.5088] **sep** | +2.7432 [+2.2799, +3.2065] **sep** | +4.7635 [+4.3189, +5.2080] **sep** |
| longitudinal.LON_speed_mae_mps | 0.617655 | -0.0074 [-0.0096, -0.0051] **sep** | -0.0084 [-0.0145, -0.0023] **sep** | +0.4086 [+0.1521, +0.6651] **sep** | +0.2243 [+0.1557, +0.2929] **sep** | +0.3933 [+0.1131, +0.6736] **sep** |
| longitudinal.LON_along_mae_m | 1.185439 | -0.0134 [-0.0166, -0.0102] **sep** | -0.0002 [-0.0114, +0.0110] | +2.1350 [+1.8295, +2.4405] **sep** | +0.6444 [+0.2462, +1.0427] **sep** | +1.9740 [+1.4584, +2.4897] **sep** |
| longitudinal.LON_accel_mae_mps2 | 0.350609 | -0.0029 [-0.0052, -0.0006] **sep** | -0.0081 [-0.0097, -0.0064] **sep** | +0.0275 [-0.0074, +0.0624] | +0.1011 [+0.0720, +0.1302] **sep** | +0.0365 [-0.0111, +0.0841] |
| lateral.LAT_cross_mae_m | 0.203539 | -0.0062 [-0.0172, +0.0047] | -0.0116 [-0.0221, -0.0010] **sep** | +0.7645 [+0.0973, +1.4317] **sep** | +1.0435 [+0.6743, +1.4128] **sep** | +0.8974 [+0.1481, +1.6468] **sep** |
| lateral.LAT_heading_mae_deg | 1.500406 | -0.0270 [-0.0921, +0.0381] | -0.0355 [-0.1123, +0.0413] | +4.0257 [+0.7779, +6.1910] **sep** | +4.6120 [+3.5735, +5.6505] **sep** | +5.0122 [+1.1300, +7.6004] **sep** |
| lateral.LAT_yaw_rate_mae_radps | 0.02307 | -0.0003 [-0.0010, +0.0004] | -0.0005 [-0.0013, +0.0003] | +0.0217 [+0.0062, +0.0321] **sep** | +0.0241 [+0.0185, +0.0296] **sep** | +0.0275 [+0.0119, +0.0378] **sep** |
| lateral.LAT_curvature_mae_1pm | 0.004405 | -0.0000 [-0.0001, +0.0001] | -0.0000 [-0.0001, +0.0001] | +0.0036 [+0.0014, +0.0051] **sep** | +0.0039 [+0.0030, +0.0049] **sep** | +0.0050 [+0.0029, +0.0064] **sep** |
| lateral.LAT_yaw_rate_mae_masked_radps | 0.02307 | -0.0003 [-0.0010, +0.0004] | -0.0005 [-0.0013, +0.0003] | +0.0217 [+0.0062, +0.0321] **sep** | +0.0241 [+0.0185, +0.0296] **sep** | +0.0275 [+0.0119, +0.0378] **sep** |
| tactical.TAC_traj_lat_correct | 1.0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.3333 [-0.3333, -0.3333] **sep** | -0.1667 [-0.3333, +0.0000] |
| tactical.TAC_traj_lon_correct | 0.666667 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.3333 [+0.0000, +0.6667] | +0.0000 [-0.3333, +0.3333] | +0.3333 [+0.0000, +0.6667] |

Plan displacement vs BASE (mean / p95, m) · sel_idx changed · tac_decoder_v6 argmax changed:

* S2|AGT_ZERO: 0.0068 / 0.0158
* S2|AGT_SHUF: 0.0142 / 0.0293
* S2|BEV_ZERO: 0.5222 / 1.4081
* S2|BEV_SHUF: 0.2841 / 0.7675
* S2|BOTH_ZERO: 0.5923 / 1.475
* S6|AGT_ZERO: 0.0431 / 0.1126
* S6|AGT_SHUF: 0.0576 / 0.0817
* S6|BEV_ZERO: 3.1101 / 5.9549
* S6|BEV_SHUF: 2.1317 / 3.9932
* S6|BOTH_ZERO: 2.8608 / 5.846
* sel_idx changed: {'AGT_ZERO': 0.0, 'AGT_SHUF': 0.0, 'BEV_ZERO': 0.8333, 'BEV_SHUF': 0.8333, 'BOTH_ZERO': 0.8333}
* tacv6: {'AGT_ZERO|nav_true.tacv6_lat_logits': 0.1667, 'AGT_ZERO|nav_true.tacv6_lon_logits': 0.1667, 'AGT_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 5.784205913543701, 'AGT_SHUF|nav_true.tacv6_lat_logits': 0.1667, 'AGT_SHUF|nav_true.tacv6_lon_logits': 0.8333, 'AGT_SHUF|nav_true.tacv6_goal_logits|max_abs_logit_diff': 7.593480587005615, 'BEV_ZERO|nav_true.tacv6_lat_logits': 0.1667, 'BEV_ZERO|nav_true.tacv6_lon_logits': 0.0, 'BEV_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 7.690232276916504, 'BEV_SHUF|nav_true.tacv6_lat_logits': 0.0, 'BEV_SHUF|nav_true.tacv6_lon_logits': 0.3333, 'BEV_SHUF|nav_true.tacv6_goal_logits|max_abs_logit_diff': 8.358832359313965, 'BOTH_ZERO|nav_true.tacv6_lat_logits': 0.1667, 'BOTH_ZERO|nav_true.tacv6_lon_logits': 0.6667, 'BOTH_ZERO|nav_true.tacv6_goal_logits|max_abs_logit_diff': 9.517660140991211}

**Strategic family:** NOT COMPUTABLE by this instrument (n = 0) — _paired_families carries no strategic component; the nav-obedience block exists only in the full-arm-set acceptance stage, not on an os-only roll.

## Controls

* VOID gates all pass: **False**; failed: ['C1_identity_os_bitexact', 'C1_identity_plan_full_bitexact', 'C3_s1_os_bitexact']
* C2 (BASE_s1 vs banked dev-box battery s1, ≤ 1e-5): **C2-FAIL (cross-device reproduction not established)**
* s1: C2 {"binding": true, "n_windows": 6, "max_abs_m": 0.5854377746582031, "frac_windows_bitexact": 0.0, "frac_windows_le_1e-5": 0.0, "pass": false, "g_equal": true} · C3 {"aligned": false, "os_bitexact": false, "plan_full_bitexact": false, "max_abs_m": null} · S6 GT independent max |Δ| 0.0
  * C4/C5/C6: C5_aligned_BASE=True, C5_aligned_AGT_ZERO=True, C5_aligned_AGT_SHUF=True, C5_aligned_BEV_ZERO=True, C5_aligned_BEV_SHUF=True, C5_aligned_BOTH_ZERO=True, C4_BASE=True, C4_BASE_donor_same_clip=0, C4_AGT_ZERO=True, C4_AGT_ZERO_donor_same_clip=0, C4_AGT_SHUF=True, C4_AGT_SHUF_donor_same_clip=0, C4_AGT_SHUF_donor_same_clip_zero=True, C4_BEV_ZERO=True, C4_BEV_ZERO_donor_same_clip=0, C4_BEV_SHUF=True, C4_BEV_SHUF_donor_same_clip=0, C4_BEV_SHUF_donor_same_clip_zero=True, C4_BOTH_ZERO=True, C4_BOTH_ZERO_donor_same_clip=0, C6_self_pair_zero_S2=True, C6_self_pair_zero_S6=True

### CPU smoke (dev box, 6 windows — controls only, not a result)

* C1 as registered (bit-exact vs the unmodified child): **FAIL as registered (bit-exact); PASS at the brief's tolerance (max |d| <= 1e-5)** — max |Δ| os 4.84e-08 m, plan_full 3.81e-06 m, sel_idx identical True
* Attribution: the HOOKS are a bit-exact identity at BASE (hooked == un-hooked wrapper, bitwise); the residual 3.8e-6 m is the WRAPPER PROCESS vs the directly-run child (same code, same seed, same device, both deterministic run-to-run) -- a CPU float-reduction difference from a different process history, not the intervention. Every arm and BASE go through the SAME wrapper, so arm - BASE isolates the hook mode.
* Zero is exactly zero at the hook: {'AGT_ZERO': [True, None], 'BEV_ZERO': [None, True], 'BOTH_ZERO': [True, True]}
