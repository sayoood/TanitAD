# refcv9 L1 — progress head on refcv8's `z_tac`: RESULT

*Architecture & Inference agent, 2026-10-25 13:00 Berlin. Prereg `PREREG_L1.md` sha256 **41ba4b2ff2443520c334a6f713acf1dcc1894adfbd3e134f2d91de8d6d91c8cf**
(sealed 10:10, before any 48,852 `z_tac` existed); Amendment 1 **5193dc29…** (K4 at seed 0 re-referenced; sealed before any fit);
Amendment 2 **62674610…** (braking AUC on S6, Master Mind request; sealed before any fit); Amendment 3 **f38506f0…** (follow-up arms,
sealed after O2 and before computing them). Deviations: `DEVIATIONS.md`.*

**Tier** T1 self-action open loop on logged eval139 frames; re-timed arms are post-hoc emulations of the in-forward layer on the POST-cap path
(cap-binding difference UNMEASURED). **Estimator** paired episode-cluster bootstrap B 2000 over episodes (`pair_read_r8.paired_cell` →
`taniteval.ci`): answers *another draw of episodes* only; one training seed; inference seeds 0 and 1 reported separately.
**n** S2 ceiling-compliant 4,159 windows / 138 episodes; S6 3,251 / 138; fit = A6 train139 compliant rows (S2 4,211 / 137 ep; S6 3,301 / 137 ep).
**Inference-seed floor** (OS s0 − s1, S2 ADE) −0.0021 [−0.0060, +0.0017].

## Verdict (mechanical, `code/l1_verdict.py` → `result/VERDICT.json`): **O2 — the frozen latent adds nothing over the scalars at this head class and fit size**

Every control passed: K1 (H0 reproduces the banked RETIME_TRAINFIT cell, S2 −0.0343 / −0.0330, S6 −0.8669 / −0.8662), K2, K3 (CONST eval R² ≤ 0),
K4 (bank plan = os-only child, 0.0 m at s0 per Amendment 1; = banked dump_s1, 0.0 m at s1), K5, K6 (z_tac bit-identical across inference
seeds, max |Δ| 0.0), K7 (H2s reproduces the 30k probe: λ 10, train R² 0.506 / 0.099 / 0.066 / 0.103), V-m (H2m not below H2s).

| S2, E1 = ADE − `ha0_ext_pose` | s0 | s1 |
|---|---|---|
| OS (refcv8) | −0.0069 [−0.0246, +0.0112] | −0.0048 [−0.0210, +0.0124] |
| **H0** refcv8 + shipped stage | −0.0343 [−0.0474, −0.0216] | −0.0330 [−0.0452, −0.0206] |
| **H2s** scalar floor | −0.0371 [−0.0498, −0.0259] | −0.0363 [−0.0486, −0.0256] |
| **H2** z_tac + scalars (k 32, λ 1) | **+0.0264 [+0.0052, +0.0476] SEP worse** | +0.0269 [+0.0059, +0.0481] SEP worse |
| H2n annex MLP, L1 loss (67,332 params) | −0.0082 [−0.0244, +0.0100] | −0.0078 [−0.0238, +0.0102] |
| H1 (b) linear-conditioned gain | −0.0262 [−0.0411, −0.0113] | −0.0246 [−0.0385, −0.0101] |
| CONST | −0.0296 | −0.0287 |

P1 FAIL, P2 FAIL (H2 − H0 +0.0606 [+0.0449, +0.0774] / +0.0599 SEP), **P3 FAIL** (H2 − H2s +0.0635 [+0.0433, +0.0851] / +0.0633 SEP — worse,
not better), P4 pass (trivially), G FAIL. Secondary: H2n − H2s +0.029 SEP (worse); H2s − H0 −0.0029 / −0.0034 n.s.; H1 − H0 +0.008 SEP (worse).

**Four families, H2 − H0 (S2, both seeds):** ADE +0.061 SEP · LONGITUDINAL speed MAE +0.082, along MAE +0.065, |along| at 2 s +0.173,
within-0.5 m/s −0.074, accel MAE +0.116 (all SEP worse) · LATERAL heading / curvature / yaw n.s., cross-track MAE +0.002 SEP, |cross| 2 s
+0.004–0.005 SEP (the path is kept; re-timing moves the point along it) · TACTICAL traj-lon correct −0.078 SEP, traj-lat 0.000 / −0.001 ·
distance-keeping (headway / TTC) NOT computed (no lead-agent track in this harness, n = 0) · STRATEGIC n = 0 (`--no-strategic`).
**S6 (secondary, E3):** H0 −0.867 / −0.866, H2s −0.884 / −0.877, H2 +0.144 / +0.151 (H2 − H0 +1.01 SEP).

**Why (MEASURED):** H2's episode-held-out fit INSIDE train is good (OOF R² 0.54 / 0.25 / 0.27 / 0.33 vs scalars 0.51 / 0.10 / 0.07 / 0.10) and
collapses on eval (R² 0.41 / −0.27 / −0.49 / −0.55). Train139 is in-sample for refcv8; its z_tac carries clip-memorised progress
information that episode-CV inside train cannot expose. The marginal distributions barely differ (train-vs-eval domain AUC **0.547**), so
it is a conditional, not a covariate, shift.

## Braking AUC (Amendment 2; L2's `l2_roc.py` reused byte-identical; S6 compliant, 639 decel / 1,856 cruise, 136 ep)

Controls C1 (GT 1.0), C2 (const 0.5), C3 (FD/MD reproduce ZERO_GPU), C4 (L2 AUCs and the stage − r7 CI reproduce exactly), C5 (L1 OS = L2 os,
0.0) all PASS. AUC refcv8 + stage 0.7445 / 0.7361, refcv7 0.7266 / 0.7182 (reproduced). **H2 0.6667 — AUC-DOWN vs stage**
(−0.078 [−0.119, −0.038] / −0.069 [−0.111, −0.030]), AUC-SAME vs refcv7 and vs H2s. **H2s 0.657 — AUC-DOWN vs stage** (−0.088 / −0.079).
H2n 0.706 (−0.038 / −0.030). H0 = stage exactly. ⇒ committed reading: a guard failure on S6 for H2 (blocks any L1 ship; does not relabel O2).
The scalars alone discriminate braking worse than refcv8's own speed choice — vision IS in refcv8's plan; the head did not extract it.

## Follow-up (Amendment 3, pre-registered after O2, before computation; `result/followup/FOLLOWUP.json`)

* **MEM-YES.** Cross-fitted on EVAL episodes (5 folds, a window is never scored by a head that saw its episode), z_tac + scalars beats
  scalars-only: **CF2 − CF2s S2 −0.0044 [−0.0076, −0.0011] / −0.0045 [−0.0076, −0.0011]** (2.1× the inference-seed floor), **S6 −0.129
  [−0.207, −0.054] / −0.129 [−0.208, −0.055]**, both seeds separated; eval R² gain +0.02 / +0.03 / +0.04 at 1 / 1.5 / 2 s. ⇒ z_tac on UNSEEN
  clips carries linearly decodable progress information beyond the scalars; L1's failure is the in-sample fit split.
  Braking AUC (beside): CF2 0.760 vs stage +0.015 / +0.024 n.s.; CF2s 0.697.
* **STACK-NO.** H3 (scalar head + the plan's own residual, fit on train) − H0 +0.0034 [+0.0004, +0.0066] / +0.0037 SEP worse; the plan
  coefficient β inherits the selector's in-sample optimism (same mechanism).
* EXPLORATORY, unregistered (computed after the CF2 and H0 points were seen): CF2 − H0 S2 −0.0105 [−0.0176, −0.0037] / −0.0110
  [−0.0179, −0.0046]; S6 −0.206 / −0.199 SEP. Part of it is in-distribution fitting (CF2s − H2s −0.0032 SEP), so the clean latent share is
  CF2 − CF2s above. It sizes the prize; it is not a lever claim.

## Next lever (Rule Zero)

1. **A clean calibration split** for the head: V-C fresh PhysicalAI clips outside the 4,711 (corpus build) or V-A (203 refcv8-only clips,
   clean only under a refcv7 warm start). **BLOCKED**: both need Thor / a corpus pull — a PI decision (annex §3). Unblocks: PI picks V-A or V-C;
   the head fit + scoring here then reruns on CPU in ~5 min from the banked z_tac (only the clean split's z_tac needs a ~0.8 s/window roll).
2. **Joint training in refcv9** with stop-grad on z_tac (the annex's c4 arm), with an explicit train/held-out read of the head every eval
   step — the in-sample optimism measured here is the risk to watch; needs PI compute (Thor) and a replicate (B9-2).
3. Done in this run, 0-GPU, ready for (1)/(2): the in-forward `ProgressRetime` torch port (`code/progress_retime_torch.py`) reproduces the
   shipped stage on 48,852 pf to 7.6e-6 m (4,754 windows, both seeds), g = 1 → plan exactly, gradient reaches the head output only,
   a +0.05 table mutation goes RED (`logs/prt_check_s{0,1}.log`).

## Cost

GPU MEASURED 2.27 h (smoke 1 min, eval s0 3,684 s, eval s1 2,329 s, train 2,139 s) on the dev-box RTX 4060, peak CUDA 1.22 GiB; CPU fits ~10 min.
Disk: 78 MB on C: (z_tac bank fp32, 3 × ~9.4 MB compressed; fp16 price 14.7 MB). Nothing written to E:, F:, G:. No git. Thor untouched.
