# E2-WISE WOFF (WIRE-OFF): the E2 rule on K2DC with its two planner wires zeroed, K2DC vs K0 -- verdict NO-GO

Pre-registered before any E2-WISE number: `../../PREREG_E2WISE_2026-11-17.md` (landed ad9f1cbe) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM1.md`
(landed 5f647742) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM4.md` (landed 81d409d9). Weights: `scripts/wire_off.py` (Addendum 4): K2DC's `model_ema_tail5k6k.pt` with exactly 8 tensors set to 0 (the 6 `core.decoder.layers.*.box_gate` + `_perception.bev_pool.map_chan_proj.proj.weight` / `.bias`), every other tensor bit-equal to the tail;
K0 has none of these tensors, so its file is its tail, byte-equal. No interpolation and no init checkpoint. Zero training. DEV = eval139 even half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval
answers another draw of EPISODES only. WIRE-OFF file md5: K2DC ec11ab226390f528effcb7afa6d15266, K0 f0ae6f32ebfe0fa2187f060d06aef2d5 (every dump.json
records it, `stageA/dump_headers/`).

## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)

| arm | E2-TAIL | this read |
|---|---|---|
| K2DC | 0.4850 | **0.4850** [0.3996, 0.5632] |
| K0 | 0.4665 | **0.4665** [0.3852, 0.5375] |

* Perception check (ADDENDUM 1, required by Addendum 4): map points equal E2-TAIL to 1e-9: YES; box AP@2m points equal E2-TAIL to 1e-6: NO (K2DC 0.307658 vs 0.307623, K0 0.307665 vs 0.307665). DIAGNOSED under Next: the check's premise was wrong (W-MAP-2 also feeds the box head), not the tool.

## Notes on this read

* Weights (`wire_off.py`, provenance in `stageA/weights/`): K2DC n_zeroed 8 (Addendum 4: 8), untouched bit-equal 1193; K0 n_zeroed 0,
  a byte copy of its tail (md5 equal). Zeroed tensors and their tail values: `_perception.bev_pool.map_chan_proj.proj.bias` -0.0005, +0.0001 ...; `_perception.bev_pool.map_chan_proj.proj.weight` +0.0014, +0.0041 ...; `core.decoder.layers.0.box_gate` -0.0252; `core.decoder.layers.1.box_gate` -0.0077; `core.decoder.layers.2.box_gate` +0.0063; `core.decoder.layers.3.box_gate` -0.0039; `tac_decoder_v6.layers.0.box_gate` -0.0056; `tac_decoder_v6.layers.1.box_gate` -0.0071.
* Clause (b) is judged under the PI's 2026-11-17 ruling (landed 50856a00): clause (b) paired box AP@2m K2 - K0 -0.0000 [-0.0076, +0.0080] (paired episode-cluster bootstrap, B 2000), not separated -> PASS.

## Lateral seed-0 gaps vs E2-TAIL (paired K2DC - K0, T1, 70 episodes; Addendum 4's secondary readout)

| row | E2-TAIL | WIRE-OFF |
|---|---|---|
| LATERAL / `LAT_cross_mae_m` | +0.0126 [+0.0063, +0.0194] **WORSE** | +0.0124 [+0.0062, +0.0192] **WORSE** |
| LATERAL / `LAT_cross_abs_final_m` | +0.0288 [+0.0151, +0.0435] **WORSE** | +0.0285 [+0.0150, +0.0431] **WORSE** |
| LATERAL / `LAT_heading_mae_deg` | +0.1096 [+0.0442, +0.1813] **WORSE** | +0.1100 [+0.0439, +0.1827] **WORSE** |
| LATERAL / `LAT_yaw_rate_mae_radps_valid` | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0022 [+0.0009, +0.0036] **WORSE** |

## Stage B -- the read's verdict (quoted, unedited from `stageB/read/E2_VERDICT.md`)

# E2 verdict: NO-GO  (K2 vs K0)

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (b) box AP@2 m head output no NMS: K2 not separated worse than K0 (paired, episodes) | **PASS** | T0 (label level) | - |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **PASS** | T1 self-action open loop (S6c-DEV) | - |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

GO iff K2 beats K0 in point estimate on the map primary and the planner's S6c sidewalk rate, K2 is not separated worse than K0 on box AP@2 m, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4) (clause (b) amended by the PI 2026-11-17: not separated worse).
NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

* (d) seed 0: separated WORSE: LONGITUDINAL/LON_dk_headway_min_m, LONGITUDINAL/LON_dk_min_ttc_s, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct
* (d) seed 1: separated WORSE: LONGITUDINAL/LON_dk_min_ttc_s, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m

## (d), row by row: E2-TAIL vs this read (paired K2DC - K0, episode-cluster bootstrap, T1, 70 episodes)

Every row separated worse at either seed in either read. WORSE = separated in the bad direction.

| family / row | tail s0 | woff s0 | tail s1 | woff s1 |
|---|---|---|---|---|
| LONGITUDINAL / `LON_dk_headway_min_m` | -0.0369 [-0.0976, +0.0224] | -0.0527 [-0.1083, -0.0087] **WORSE** | -0.0089 [-0.0553, +0.0431] | -0.0149 [-0.0610, +0.0353] |
| LONGITUDINAL / `LON_dk_min_ttc_s` | -0.3953 [-0.9620, -0.0229] **WORSE** | -0.4293 [-0.9979, -0.0592] **WORSE** | -0.3532 [-0.8438, -0.0070] **WORSE** | -0.3598 [-0.8595, -0.0082] **WORSE** |
| LATERAL / `LAT_heading_mae_deg` | +0.1096 [+0.0442, +0.1813] **WORSE** | +0.1100 [+0.0439, +0.1827] **WORSE** | +0.0804 [+0.0175, +0.1515] **WORSE** | +0.0838 [+0.0210, +0.1542] **WORSE** |
| LATERAL / `LAT_yaw_rate_mae_radps_valid` | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0016 [+0.0004, +0.0030] **WORSE** | +0.0017 [+0.0004, +0.0031] **WORSE** |
| LATERAL / `LAT_cross_mae_m` | +0.0126 [+0.0063, +0.0194] **WORSE** | +0.0124 [+0.0062, +0.0192] **WORSE** | +0.0091 [+0.0024, +0.0166] **WORSE** | +0.0095 [+0.0028, +0.0169] **WORSE** |
| LATERAL / `LAT_cross_abs_final_m` | +0.0288 [+0.0151, +0.0435] **WORSE** | +0.0285 [+0.0150, +0.0431] **WORSE** | +0.0203 [+0.0053, +0.0370] **WORSE** | +0.0211 [+0.0064, +0.0376] **WORSE** |
| TACTICAL / `TAC_traj_lat_correct` | -0.0071 [-0.0146, -0.0004] **WORSE** | -0.0071 [-0.0146, -0.0004] **WORSE** | -0.0071 [-0.0147, +0.0000] | -0.0067 [-0.0142, +0.0000] |

## Secondary readout: K2DC seed-0 `LAT_cross_mae_m` (decides nothing alone)

* K2DC 0.1159 m (tail 0.1160), K0 0.1034 m (tail 0.1034); gap +0.0124 (tail +0.0126). Both arms above their tail values: NO.
* STRATEGIC: the read prints PAUSED (its PI item-25 text). The PI ended that pause 2026-11-16; in these arms the strategic modules are
  equal to the init (S_plan excludes them), so there is no trained strategic row to compare.

## Next (Rule Zero)

* ADDENDUM 1 CHECK FAILS AS WRITTEN (required by Addendum 4): the map points equal E2-TAIL to 1e-9 and K0's box points are equal, but K2DC's box AP@2m moved 0.307623 -> 0.307658 (+0.000035). DIAGNOSED from source (MM, 2026-11-18, tip 1783717d): Addendum 4's premise "the map primary and box AP do not read either wire" is FALSE for W-MAP-2. `map_chan_proj` is added to the pooled BEV (`stack/tanitad/models/refcv6_perception_branch.py:357`, `out = out + self.map_chan_proj(map_chan)`), that pooled BEV is `bev_feats`, and the box head's anchors come from `self.box_heat(bev_feats)` (`:720`); the branch's own guard lists box3d among the readers of that BEV. So zeroing the projection moves box detection by construction: a premise error in the addendum, not a tool or read defect (map points and K0 bit-equal; the weights control read 8 / 0 tensors zeroed). Logged as R-E2-WOFF-1 in RETRACTION_LOG.md. READING (post-hoc, labelled so): the intervention removed BOTH wires from EVERY consumer, planner and box head alike, and (d) did not move: seed-0 cross-track +0.0124 (tail +0.0126), heading +0.1100 (+0.1096), yaw rate +0.0022 (+0.0022). The lateral deficit is not carried by these 8 tensors at inference = Addendum 4 outcome 3 in substance. With PLAN-RESET (+0.0113) and WISE-BOTH (+0.0098) it sits in what K2DC's training did to the shared trunk; the next lever is a retrain (K2DC02, share target 0.02; or the planner L2-SP anchor), a PI decision. K2DC02's launch-gate records were started on the idle Thor at 12:48Z (records only, no training). Clause (b) under the PI's 2026-11-17 ruling: clause (b) paired box AP@2m K2 - K0 -0.0000 [-0.0076, +0.0080] (paired episode-cluster bootstrap, B 2000), not separated -> PASS.
* E2, E2-RAW and E2-TAIL stand as written.

## Provenance

* Stage A files pulled from Thor `/home/nvidia/r8ns/r9e2/runs/E2WISE_WOFF/stageA/`, sha256-equal. Rasters, perc_det.pkl and
  the WIRE-OFF .pt files are not banked (re-creatable: wire_off.py on Thor's runs/r9e2-*/model_ema_tail5k6k.pt, md5 above). Stage B read files are
  byte-exact copies of the dev-box S6c driver's pull (sha256-verified against Thor `runs/E2WISE_WOFF/stageB/`).
