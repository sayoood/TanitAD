# PREREG_V8F_LATSEM: what each world-model level encodes, what it imagines, and whether its imagined future follows the action (Master Mind, 2026-11-07, written before any LATSEM statistic exists)

## Why
* The PI asked whether the semantics of each abstraction level can be shown, as H-JEPA does (arXiv 2610.06805, banked; its figures use post-hoc decoders, probes and latent-distance maps). WMACT (`PREREG_V8F_WMACT.md`, `d21b14cc`) measured skill and action use in latent units only. Latent units say nothing about driving meaning.
* This read gives each level's latent a driving meaning with zero training: linear probes, nearest-frame retrieval and counterfactual actions. Tier T0, a WM diagnostic, never a driving claim. The planner reads none of this.

## Data (fixed)
* `theory/wmact/latents.pt`: M2b s0 / s1 level-1 latents of the eval-141 windows. A window's frame f is cache step g = t - 15 + f. Restricted to the 139 episodes with a frame file in `v2ep-eval139-416x1024cyl/` (2 have none). Frames are shown at 416x1024; the model encoded the 256x640 view of the same camera.
* Frame bank: every distinct (episode, g) with frame_ok. Levels 2 and 3 per frame: z2 = E2(z1), z3 = E3(z2) (pointwise, w = 1). Level l's step is s = 1, 2, 4 frames (0.2, 0.4, 0.8 s).
* Labels per frame, from `native_tracks.pt` row 2g: speed v; heading. From the window's raw actions: accel a and yaw rate w. Future: dv2 = v(row 2g+20) - v(2g) and dpsi2 = wrapped heading change over the same 2.0 s.

## R1: probes (what each level ENCODES)
* Ridge, lambda 10 on features standardised on the fit fold, with intercept. Two folds by episode: fold = int(sha256(episode)[:8], 16) % 2. Fit on one fold, predict the other, both ways.
* R2_q = 1 - sum (pred - y)^2 / sum (y - fit-fold mean)^2, targets v, a, w, dv2, dpsi2, levels 1 to 3, seeds s0 / s1.
* **ENCODES q at level l:** R2_q > 0 separated at both seeds.
* **Temporal decomposition** (the paper's claim that higher levels drop fast detail and keep slow state): DD = [R2_a(L3) - R2_a(L1)] - [R2_v(L3) - R2_v(L1)], paired. **TD-CONSISTENT** if DD < 0 separated at both seeds; **TD-INCONSISTENT** if DD > 0 separated at both; otherwise **TD-NOT-RESOLVED**.

## R2: retrieval (what each level IMAGINES)
* Windows with every frame ok. Level l, horizon k (L1 1/5/10, L2 1/3, L3 1/3); the true future is frame k*s. Query = the WMACT rollout from the start state under the true actions. The nearest bank frame by squared L2 in level l's latent, from OTHER episodes only. q_ret = that frame's label.
* SEM-SKILL_q = 1 - sum (q_ret(query) - q*)^2 / sum (q_ret(start state) - q*)^2, q in {v, w}.
* **IMAGINES SPEED at (l, k):** SEM-SKILL_v > 0 separated at both seeds. Level 1 decides with the shrunk rollout (c_k from `RESULT_WMACT_SHRINK.json`, fitted on TRAIN, as `AMENDMENT_WMACT_1` commits); the raw L1 rollout is beside. The oracle query (the true future latent) is beside as the ceiling.

## R3: counterfactual actions (does the imagined future FOLLOW the action)
* Constant raw actions over the window: brake (-3.0, 0), hold (0, 0), accel (+1.5, 0), w+ (0, +0.25), w- (0, -0.25). Rolled out per level from the true start state (macro-actions of the same sequence above level 1), read at the longest horizon (2.0 / 1.2 / 2.4 s). Decoded by the R1 probe of the other fold.
* CF-LON_l = mean [v_hat(accel) - v_hat(brake)]; CF-LAT_l = mean [w_hat(w+) - w_hat(w-)].
* **ACTION-SEMANTIC-LON / -LAT at level l:** > 0 separated at both seeds; **ACTION-BLIND** if not separated at either; otherwise MIXED. Beside: the kinematic reference (4.5 m/s^2 times the horizon, speed floored at 0; 0.5 rad/s for w).

## Estimator
* Episode-cluster bootstrap, B 2,000, seed 0 (`wmact_lib.boot_weights`), ratios of episode sums; paired where two arms share frames or windows. This answers "another draw of EPISODES"; two training seeds stand in for "another TRAINING RUN"; inference is deterministic.

## Controls (must read their known values, or the read is INVALID)
1. Every copy of one (episode, g) across windows has identical z1 and raw actions (max abs difference 0).
2. Per-frame z2 / z3 equal `HJEPAStack.encode`'s levels at the matched positions to 1e-5.
3. The start-state query gives SEM-SKILL exactly 0.0; two identical action sequences give CF exactly 0.0.
4. A probe fitted on labels permuted across episodes is not separated above 0 on any row.

## Pictures (decide nothing; windows chosen by true labels only)
* Four windows: STOP (largest speed drop over frames 0 to 12), LAUNCH (largest gain), TURN (largest heading change), CRUISE (speed nearest the median with heading change under 0.05 rad and speed change under 0.5 m/s). Per level: now, the true future and the retrieved frame. Plus imagined frames under the five actions for LAUNCH and TURN, and per-level PCA maps coloured by v and w. Seed s0.

## Log (dated, append-only after the freeze)
* 2026-11-07 09:56:59 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 4882 bytes above this log, sha256 `0b0c2f73c4e5db9c72f25ee9a1df1b2b883988879c410dadfd4b6b1d1e7b1823`, before any LATSEM statistic exists.
* 2026-11-07 10:03:08 (dev-box clock, from `date`), Master Mind. Run 1 read INVALID on control 1 alone (z1 copies differ by up to 1.2e-6, one to two float32 rounding steps; raw actions exactly 0). `AMENDMENT_LATSEM_1.md` (prefix 1375 bytes, sha256 `ddb07ff809b5060e83b043b07e645ce0965a5f464dcde474c886155696d9580a`) sets control 1 for z1 to at most 1e-5, written before any statistic was looked at. Run 2 follows.
* 2026-11-07 10:06:21 (dev-box clock, from `date`), Master Mind. Run 2: controls pass; outcomes in `latsem/RESULT_LATSEM.json`. R2 has no headroom (the oracle query is not separated above 0), so `AMENDMENT_LATSEM_2.md` (prefix 1642 bytes, sha256 `83446cc661ce44f638dceb152d2ed1703b1e77604515967b5541d49b4a3d3393`) adds SPEED-SKILL through the R1 probe, written before that statistic exists. Run 3 follows.
