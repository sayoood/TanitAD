# RESULTS_COMBO_30k -- ENS-K -> RETIME at EMA-30k, eval-fitted gains (EXPLORATORY, post-hoc arm)

Label (PREREG_COMBO section 5, K=4, sets A={0,1,2,3} and B={4,5,6,7}): **STACKS -- EXPLORATORY, EVAL-FITTED GAINS, single training seed, S6 K=4 UNTESTED, transfer unmodelled.**
Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0 ("another draw of EPISODES" only; the disjoint draw sets are the inference-run replicate; not another training run).
**Bar of record = single-draw `os` (BAR-R8-E1 vs `ha0_ext_pose`). This is a NEW arm: ENS costs up to xK planner draws, RETIME costs zero GPU. Nothing here converts a single-draw FAIL into a PASS.**

Provenance: `PREREG_COMBO.md` sha256 `a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287` (pinned in the script; the file's `Written` line was corrected 07:36 -> 07:34 BEFORE the sha was pinned and before any code or number existed).
Script `code/combo_ens_retime_r8.py` sha256 `83ff0a56c16277cc...`; gains `raw/a0k0_retime/FROZEN_GAINS_RETIME_ema30k.json` sha256 `4ecde275554b1aaa...` (applied UNCHANGED, never refitted; S2 g = [0.9968, 0.5834, 0.4260, 0.3825]).
Machine output: `COMBO_step30000ema_unc_eval.json` / `.md` (judge by the JSON). Run twice (the second after adding the additivity block and K=2 label wording): **0 differing leaves** outside timestamps/script sha/new keys (the bootstrap is seeded -> deterministic).
Command: `sh code/run_combo_ens_retime_r8.sh step30000ema_unc eval` (282 s CPU, `OMP_NUM_THREADS=6`, memory-gated).

## Controls (all PASS; the analysis prints VOID otherwise)
| control | read |
|---|---|
| C-MASK | sha `4ebc6b86...`, 4,159 / 4,754 windows, 138 episodes |
| C-ALIGN / C-ID / C-DIST | all arrays aligned to the dumps (incl. seed-0/1 dump `os` == panel `os` bit for bit); seeds 0..7 identical model-free arms; min pair plan L2 0.0908 m (8 distinct draws) |
| C-R0926 | `ha0_ext_pose - ha0_ext` +0.0055 [+0.0036, +0.0076] (literal reproduced); pose-echo mean ADE 0.2858 m |
| **C-G1** | gain 1 reproduces the plan **bit for bit** for 4A, 4B, 2A, 2B, 8 and for single draws 0/1 (max abs 0.0) |
| **C-G0** | gain 0 vs the independent loop walker: max error 2.0e-13 m (tol 1e-6) |
| **C-K1** | K=1 gain 1 == banked single-draw cells: vs echo -0.0085 [-0.0229,+0.0065] / -0.0076 [-0.0216,+0.0073]; vs pose -0.0136 [-0.0286,+0.0018] / -0.0127 [-0.0272,+0.0024] |
| C-K1R | K=1 with the frozen gains == the independent `FORWARD_RETIME_step30000ema_unc.json`: vs pose -0.0320 [-0.0427,-0.0211] / -0.0315 [-0.0420,-0.0207] |
| C-ENS | ENS-2 {0,1}, ENS-4A, ENS-4B alone == `KDRAW_E1.json` pose cells (-0.0189, -0.0225, -0.0210) |
| C-ONPATH / C-TRI | combo points on the ENS polyline (max 7e-15 m); ADE(ENS) <= mean of draws on every window |
| **deliberate regression** | `COMBO_TEST_MUTATION=skip_retime` on the REAL data -> **VOID (C-K1R)** (`MUTATION_skip_retime_real_data_VOID.md`); on the planted fixture the same mutation turns STACKS into NO-STACK with ST_ens exactly 0 (pytest, 17 passed; 5 scratch mutants of the instrument each turn it RED) |

## E1 (ADE 0-2 s, ceiling-compliant S2) -- primary reference `ha0_ext_pose` (D-R8-E1-REF), `ha0_ext` beside
| arm | set A (pose) | set B (pose) | set A (echo) | set B (echo) | mean ADE A / B |
|---|---|---|---|---|---|
| single-draw `os` (seed 0 / 1) | -0.0136 / -0.0127 | | -0.0085 / -0.0076 | | 0.2722 / 0.2731 |
| ENS-4 | -0.0225 [-0.0373,-0.0071] | -0.0210 [-0.0363,-0.0056] | -0.0174 [-0.0314,-0.0021] | -0.0158 [-0.0304,-0.0006] | 0.2633 / 0.2648 |
| RET-alone (mean of the 4 single-draw retimes) | -0.0322 [-0.0428,-0.0215] | -0.0318 [-0.0424,-0.0210] | -0.0271 | -0.0266 | 0.2536 / 0.2541 |
| **COMBO-4 = retime(ENS-4)** | **-0.0353 [-0.0458,-0.0244]** | **-0.0348 [-0.0454,-0.0240]** | -0.0302 [-0.0405,-0.0194] | -0.0296 [-0.0401,-0.0189] | 0.2505 / 0.2511 |
| beside: COMBO-G-4 (gated gains) | -0.0400 [-0.0506,-0.0289] | -0.0395 [-0.0507,-0.0285] | -0.0348 | -0.0344 | 0.2459 / 0.2463 |
E1-PASS (both sets separated, beyond 2 x floor 0.00053 m): **PASS vs pose, PASS vs echo**. K=2 ({0,1}, {4,5}): COMBO-2 -0.0338 / -0.0335 vs pose; K=8: -0.0356.

## STACKING (paired, ADE 0-2 s; separated better AND beyond 2 x max(floors) on BOTH sets => STACKS)
| K=4 | set A | set B |
|---|---|---|
| COMBO - ENS-4 (RETIME on top of the ensemble) | -0.0128 [-0.0197,-0.0064] | -0.0138 [-0.0208,-0.0074] |
| COMBO - RET-alone (ENS on top of RETIME) | **-0.0030 [-0.0033,-0.0028]** | **-0.0030 [-0.0032,-0.0028]** |
Floors (m): COMBO 0.00053, ENS 0.00155, RET-alone 0.00048 -> 2 x max = 0.0031 (vs ENS) and 0.0011 (vs RET-alone); both cells clear them on both sets. K=2: -0.0020 / -0.0020 (ENS on top of RETIME); K=8: -0.0035.
Best single lever on both sets = RET-alone (0.2536 / 0.2541 vs ENS-4 0.2633 / 0.2648).

## HOW LARGE IS THE STACK (read this before quoting the label)
The label is statistical and is correct; the **marginal ENS contribution is small**. Mean ADE, compliant S2 (set A | set B): single-draw `os` 0.2719 | 0.2731; ENS-4 -0.0086 | -0.0083; RETIME alone -0.0183 | -0.0191;
additive prediction -0.0269 | -0.0273; **observed COMBO-4 -0.0214 | -0.0220 = 79 % | 81 % of additive**. ENS on top of RETIME keeps only **35 % | 36 %** of its standalone gain (-0.0030 m, 1.2 % of the ADE) for up to x4 planner draws;
RETIME on top of ENS keeps 70 % | 72 % of its standalone gain. Dose response (ENS on top of RETIME): K=2 -0.0020, K=4 -0.0030, K=8 -0.0035 (saturating). The two levers overlap by roughly two thirds on the ENS side; the PI's x4 compute is buying 0.003 m.

## ACC-LAT, NR, transfer
* **ACC-LAT HOLDS, non-vacuous, both surfaces tested.** S2 K=4 (COMBO - ENS-4, five LATERAL rows, sets A and B): no row separated worse (heading -0.0022 / -0.0015, curvature -0.0000, yaw rate -0.0002, cross -0.0008, |cross| final -0.0028 / -0.0026); ST_ens separated better on both sets (it moved).
  **S6 K=2 (draws {0,1} -- the only S6 ensemble that exists; ONE set, no replicate; n 3,251 w / 138 ep):** ADE COMBO - ENS-2 -0.2194 [-0.3172,-0.1189] (moved); heading +0.0076 [-0.0871,+0.1351], curvature +0.0000, yaw rate -0.0004 [-0.0010,+0.0002], cross -0.0136, |cross| final -0.0381: none separated worse. **S6 at K=4 is UNTESTED** (no S6-grid rolls of seeds 2..7 exist; GPU).
* **NR vs refcv7-50,400: PASS in ADE / LONGITUDINAL / LATERAL on both sets**, against both refcv7 single draws and against the refcv7 2-draw mean (no NR2 caveat). TACTICAL printed (trajectory-derived lon decision better, separated), STRATEGIC NOT APPLICABLE (n = 0).
* **Transfer (gains fitted on single-draw plans, applied to an averaged one) is immaterial at 30k:** COMBO - RT-THEN-ENS (retime each draw, then average) = 0.0000 [-0.0000, +0.0000] on every set; the averaged plan's arc length is 0.99999 of the mean draw's (draws differ by ~0.09 m on 10-20 m plans), so the two orders coincide. The HARMS mechanism the prereg worried about is not present here. Mode-disagreement subset (1.6-1.7 % of windows): COMBO - ENS curvature +0.0002 [-0.0006,+0.0014] / +0.0002 [-0.0004,+0.0011], not separated.

## Limits (binding on any quote)
1. **Eval-fitted gains**: fitted on ALL 139 eval episodes (seeds 0/1 pooled); every cell here is in-sample for the gain vector. Size of that optimism on single draws: frozen -0.0320 vs cross-fitted RETIME_XF -0.0321 (vs pose, seed 0; INHERITED from `RESULTS_BRIEF.md`) -- negligible for a 4-scalar vector, but the clean answer is the train-fitted gains (queued). Draws 2..7 were not in the fit; they share the eval windows/GT.
2. Single training seed; T1 open loop; EMA-30k is a milestone, not the registered checkpoint; S2 only for K=4.
3. The narrow stacking CIs (+-0.0003 m) mean the effect is systematic across episodes (models fixed); they say nothing about another training run.
4. The script's K=2 label is printed as descriptive ("stacking cells stack"; NR is not computed at K=2). It decides nothing.

## What the final chain must call (PREREG section 7), unchanged
`sh <PKG>/code/run_combo_ens_retime_r8.sh step48852ema_unc train` -> `raw/combo_ens_retime/step48852ema_unc_train/COMBO_step48852ema_unc_train.{json,md}`. It needs (it fails loud naming the first missing one, never substitutes):
(1) `KD_STEP=48852 sh code/kdraw_queue_r8.sh` -> `E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s{2..7}/KDRAW_VERIFY.json ok` (+ `ckpt_ema_48852.pt`; ~4.6 h GPU, MEASURED 2,730.6 s/seed at 30k);
(2) the final battery trees under `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/` (dump_s0/s1, panel_s0/s1 with `b_refcv7_50400_s*`, s6_s0/s1);
(3) `raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json` + `.sha256` (queued train-fit chain; fitted on 30k plans -> applying to 48,852 is a second transfer, named in the final report). The same script also takes `step30000ema_unc train` the moment (3) exists (clean-gains 30k milestone read).

## Next lever (Rule Zero) and blockers
This arm did not fail; there is nothing to rescue. The open decisions are: (a) the clean-gains read, BLOCKED on the train-fit chain (waiting for the GPU queue, `ZZTFWAITZZ`) -- unblocks by itself, then one command; (b) the registered read, BLOCKED on the 48,852 checkpoint and ~4.6 h of GPU K-draw rolls (PI/GPU scheduling); (c) PI decision: whether up to x4 planner draws for ~0.003 m (1.2 % ADE) on top of a free RETIME is worth deploying -- the data above says RETIME alone carries ~86 % of the combo's gain over single-draw `os`;
(d) S6 K=4 ACC-LAT needs S6-grid rolls of seeds 2..7 (GPU) -- only if (c) is yes.
