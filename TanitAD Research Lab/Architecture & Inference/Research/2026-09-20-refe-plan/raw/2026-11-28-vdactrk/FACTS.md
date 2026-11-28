# VDACTRK facts (read-only verification, 2026-11-28). PK = TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan

## 1. aux_drivable_head.py - VERIFIED
- Head: 40x32 learnable BEV queries, 2 m cells, x[-8,72) y[-32,32), rear-axle t0 frame (x fwd, y left); 2 x CrossBlock(256,8,ctx 256); per cell a mask logit + signed distance 6 m*tanh(.). "2,440,450 parameters" (docstring).
- Input: ONE tensor, `visual_ctx` [B,7680,256] (tap "visual", model.py:759 `visual_ctx = self.scene_proj(visual)`) or `scene_ctx` [B,64,256]. No ego/goal/nav/map.
- Loss: BCE(logit, 4x4-block mask fraction) + w_sdf*Huber(sdf/6, target/6, beta 0.1); label-less frame = 0 (docstring); HC adds boundary-band weight bw.
- Field -> verdict: `feed_scalars` (docstring/lines ~119-140): bilinear sample of predicted SDF at footprint corners (FL.footprint_corners, NAVSIM-representation poses) over 40 steps t=0.1..4.0 s -> min corner SDF, mean centre SDF, share of corner-steps <0. H1 (feed off, default) = scorer unchanged; H2 feeds the 3 scalars via additive Linear(3,256). Out-of-extent corner = SDF +6 m (unknown).
- NOTE: 40-step (0.1 s) poses expected; trk20 is 20 steps at 0.2 s -> needs interpolation or the 40-step tracker states (card computes [64,41,11]).

## 2. Label store - VERIFIED
- aux_label_store.py header: raster x[-8,72) y[-32,32) 0.5 m cells [160,128], rear-axle t0 frame, int8 SDF unit 0.1 m clip +-12.7, mask = sdf>0; polygons = navsim_dac.drivable_polygons (ROADBLOCK+INTERSECTION+CARPARK within 100 m of ego centre = rear axle + 1.461 m).
- Disk (NOT where code defaults say; DEFAULT_STORE = D:/Projects/TanitAD-artifacts/aux_drivable/label_store, D:/Projects/TanitAD-artifacts does not exist now): `E:/Projects/TanitAD-artifacts/aux_drivable/label_store` manifest: 103,139 keys, 1,192 logs, 351,383,740 B, store_sha256 4a17db67...287a; navtest overlap token 0 / log 0 (leakage block). `.../label_store_navtest`: 12,146 keys, 136 logs, 100 % coverage, sha256 985a3bc7...0d7c1 (source db, written 2026-10-13). Per-city navtest: LV 4064, BOS 3731, PIT 2437, SG 1914.
- Shard files (.aux1) not listed under `ls label_store` top level in my probe (showed failures_all.jsonl, manifest.json, missing_tokens.txt, run_meta_all.json); manifest.files has 1,192 entries. I did not verify shards exist on disk - verify before relying.
- 97.7 %: SPEC_AUX_DRIVABLE_HEAD.md:22 "whole-proposal verdict 97.7 % / 98.1 %; corners 99.03 % / 99.12 %; 42,640 corners / 260 verdicts, 20 samples", MEASURED. Small sample (260 verdicts, 20 samples). No larger artifact found.

## 3. C1 - VERIFIED (mostly)
- PREREG_P5C.md (raw/2026-11-11-p5c-readiness): C1 = P7 final + 2,000 steps lr 5e-5, WTA target = Tc20s table (traj-label-table); line 115 "aux head OFF: no --aux-drv". Note C1 is NOT "P7 + aux off" for labels: the one lever is the Tc20s label table; C0 is the matched raw-label arm.
- C1 ckpt F:/safehead_cache/ckpt/C1/model_final.pt: 1,288,515,000 B; navtest dump meta ckpt_sha256 f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4 (matches). Scan of data.pkl key names: zero keys containing aux/drv/drivable. So NO drivable head weights.
- visual_ctx: model.py:759 [B,P,dec_width], P = 7680 tokens x 256 (per head docstring). Computed in the model forward from the backbone; C1 uses its normal cameras (not re-checked).

## 4. trk20 - VERIFIED, admissibility ESTABLISHED BY CARD (not independently audited)
- npz keys: token, k_ship, k_id, k_trk, k_red, id_maxdiff, red_d0, t_trk, s_plan[12146,64,6], s_trk[12146,64,6], trk20[12146,64,20,3], ship8[12146,8,3], s0[12146,11], meta. Written 2026-11-28T02:59Z; summary: n 12146, id_match 12146, id_maxdiff_max 0.0, switch 7489, red_tokens 1247, red_switch 748.
- Card PREREG_TRKSCORE.md (raw/2026-11-26-trkscore), section 2.1: `trk20[j,i]` = simulated state at t=0.2(i+1) (state index 2(i+1)) in t0 rear-axle frame; 3 channels = (x, y, heading), heading on the planned branch h = h_plan + wrap(h_sim - h_plan). Tracker = navsim@3e8291b transform_trajectory(props8[j], 8x0.5 s) -> 40x0.1 s -> PDMSimulator.simulate_proposals (LQR + kinematic bicycle), output [64,41,11] world-frame.
- Determinism: card line 25 "The seam is deterministic: registry 14.6 ... max |dpose| 0.0, pick agreement 1.0"; summary id_maxdiff_max 0.0.
- Admissibility (card "Admissibility", lines 45-47): "At inference the rule reads only the agent's own plan and its ego status at t0: pose, velocity, acceleration and steering angle"; "No map, agents, future, PDM-Closed trajectory or metric cache is read". Start state = planner's own EgoState at t0. (Card's claim; CTRL-PARITY checks 50 tokens.)

## 5. H-VDAC-P6-1 S1 - VERIFIED
- Artifacts: vdac/a5_read/evidence/fresh/run.log; hb_read/READ_HB.md:55; hc_read/READ_HC.md:55-56,85.
- HA-s0: S1-P1 IoU[0,40) 0.7802 [0.7587,0.7974] (bar 0.80 FAIL); S1-Q 0.8332 [0.7951,0.8691] (bar 0.90 FAIL); C-SHUF 0.4468, C-FLIP 0.7056; true-field 2 m AUC 0.9408 (sigma_b 6.5905 m); per-city IoU LV 0.8914 BOS 0.7394 PIT 0.7234 SG 0.6960.
- HB-s0 (LoRA copy of trunk): IoU 0.8113 holds, S1-Q 0.8376 [0.7963,0.8744] fails. HC-s0 (1 m cells, boundary-weighted): IoU 0.7757 fails, S1-Q 0.8346 [0.7919,0.8751] fails. READ_HC.md:85 "S1-Q is flat across all three levers: HA 0.8332, HB 0.8376, HC 0.8346. No paired S1-Q delta is separated."
- S1-Q (PREREG_VDAC_P6.md:72) = pooled AUC of -d_k for NAVSIM DAC=0 over **P6's proposals** on H48 sets (48 held-out navtrain logs) with labels; d_k = signed-distance footprint verdict from the head. Population is P6 / navtrain H48, NOT C1 / navtest. Ceiling 0.94 (true field), tracked variant S1-QT unavailable (missing navtrain metric-cache ego states).
- Recipe: S1c = 10,000 updates, batch 8, visual tap, depth 2, feed off, frozen P6 trunk, 1.2753 s/update (A40), 3.2 L40-h per head; S1 total 7.0-7.2 L40-h (DESIGN_VDAC_P6.md:19,32,193-194). Head training data = navtrain bank labels minus H48 (the 8,489-set FIT pack is the S0 population, not the head training set; I did not find lr value, checked lines only partially). Fit pack: F:/tanitad_scratch/p6fit/fit_pack8.npz (S0 only).

## 6. Navtest inputs for a head - VERIFIED: NO banked C1 visual_ctx for navtest
- F:/safehead_cache/navtest/*.npz (136 files, one per log): keys cmd[n,4], ego[n,7], log, logits[n,64,6], meta, pick, props[n,64,20,3], props8[n,64,8,3], s[n,64,256], split, token. `s` = input of model.score_head (sh_dump_nt.py:7,96: post-decoder scorer features [1,64,256]), NOT visual_ctx. meta ckpt_sha256 = f09949e2... (C1), nav_mode command, yaw_table navtest.
- visual_ctx dumps in E:/Projects/TanitAD-artifacts/aux_navtest_forward: chunk6_ctx (6 files, [7680,256] float32, ~7.9 MB each), chunk6v2_ctx; c3_full_ctx has 2116 files of shape (64,7) (not visual_ctx). These are pilot/C3-era/other-checkpoint; none is C1-on-navtest. 12,146 x 7.9 MB = ~96 GB if C1 visual_ctx were dumped fp32 (ARITHMETIC) -> must re-run C1 forward per token (sh_dump_nt ~1.28 s/token = ~4.3 h, card ETA) with a hook, or run the head inline.
- Heads trained on P6's visual_ctx will not transfer to C1's trunk; a C1-specific head (or P7 trunk if C1 = P7 + 2,000 steps, features differ slightly) needs training. Head checkpoints: not located on disk by me (pod).

## Blockers / reshaping
1. Head must read C1's (or P7's) visual_ctx; S1 heads were on P6 -> retrain ~3.2 L40-h.
2. S1-Q 0.83-0.84 < 0.90 bar on P6; no S1 head evidence on C1 / navtest.
3. Label store path moved (D: -> E:); shards unverified on disk.
4. trk20 is 20 steps @0.2 s; head feed expects 40 steps @0.1 s.
5. navtest C1 visual_ctx not banked.
