# PREREG_L1C — the progress head fitted on a CLEAN calibration split of fresh PhysicalAI clips (Master Mind, 2026-10-25 ~13:40 Berlin)

Sealed before any fresh clip is selected, downloaded or rolled. PI decision in chat 2026-10-25 ~13:35 CET: *"Pull fresh PhysicalAI
clips (Recommended)"* (~300 clips outside the 4,719-clip corpus, built on the dev box, used as a calibration split; training corpus and
eval set unchanged, parity kept). Parent: PREREG_L1 (sha256 41ba4b2f…) and its Amendment 3 (f38506f0…), result `H-R9-L1-PROGRESS-1`
(landed 0ee472f): fitted on refcv8's own training clips the head fails (O2); cross-fitted on eval episodes it beats the scalar floor.

## 1. The calibration split (fixed before selection)
* Universe: PhysicalAI-AV clips NOT in the programme's 4,719-clip v7 universe, carrying the SAME camera rig as the corpus
  (`camera_front_wide_120fov`, and the corpus's rig family by principal point cy — see memory/registry "two rigs by cy"), and passing
  the SAME r0 selection filters the corpus used (`physicalai_r0.py`), applied unchanged.
* Selection: order the eligible clips by sha256(clip_id) ascending; take the first 300 whose build completes. No clip is chosen or
  dropped by anything computed from refcv8. Clips that fail to build are logged by sha12 with the reason, and the next in order is taken.
* Every input the eval139 roll reads is produced for the fresh clips by the SAME builder that produced eval139's (episode build, v2
  cache, per-frame nav, the fed speed ceiling `v_lim`, agent joins, clock, extrinsics), and the roll uses the eval139 code path
  (`l1_ztac_roll.py --mode eval`, refcv8 EMA-48,852, md5 b4364029…) with only the input paths swapped. Inference seed 0.
* **Fallback branch, committed now:** if `v_lim` cannot be produced for the fresh clips by the corpus's own pipeline, BOTH heads (H2c,
  H2cs) drop the three `v_lim` features in fit AND in scoring, the fit uses all fresh windows (compliance needs `v_lim`), and the eval
  population stays the compliant one. The branch taken is stated in the result; nothing else changes.

## 2. Arms (scored on eval139 S2 compliant, 4,159 w / 138 ep, and S6 3,251 w / 138 ep, inference seeds 0 and 1)
| arm | definition |
|---|---|
| **H2c** | PREREG_L1's H2 recipe (10 scalars ‖ PCA_k(standardised z_tac), k and λ by 5-fold EPISODE CV) fitted on the FRESH split only |
| **H2cs** | PREREG_L1's H2s recipe (10 scalars, λ by CV) fitted on the FRESH split only — the scalar floor on the same rows |
| H0 | refcv8 + shipped stage (banked; reproduces −0.0343 / −0.0330, K1) |
| CF2 | Amendment 3's eval cross-fit (banked; reproduced, not refit) — the reference the clean fit is expected to approach |
| H2cm | H2c applied with eval z_tac deranged across episodes (deliberate regression) |

## 3. Controls (must read their known values, or the run is void)
K1 H0 = −0.0343 / −0.0330 to 1e-4 · K2 CF2 − CF2s reproduces −0.0044 / −0.0045 to 1e-4 · K3 a constant head reads eval R² ≤ 0 ·
K4 the fresh roll's identity checks (bank plan = the roll's own os plan, 0.0 m) · V-m H2cm does NOT beat H2cs separated at either
seed · D-fresh (diagnostic, no bar): domain AUC fresh-vs-eval139 z_tac (PCA-32 logistic, episode CV); above 0.70 is flagged, because
then the fresh clips are not the eval distribution and a fail is not about the head.

## 4. Bars (paired episode-cluster bootstrap over episodes, B 2000; answers "another draw of EPISODES" only; one training seed)
* **P1** H2c − H2cs separated < 0 at BOTH inference seeds, |Δ| > the inference-seed floor (−0.0021) — vision adds on a clean fit.
* **P2** H2c − H0 separated < 0 at BOTH seeds — the clean head beats the shipped stage.
* **G** for H2c − H0: no four-family row separated WORSE at both seeds; braking AUC (L2's `l2_roc.py`, unchanged) not separated below H0.
* Four families reported for H2c − H0 and H2c − H2cs (distance keeping and strategic state n = 0 with the reason, as in L1).

## 5. Outcomes, committed
* **O1 (P1 + P2 + G pass):** SHIP candidate for refcv8 (a stage upgrade; it needs the registered SHIP-R8 battery on the new stage and
  the PI's go) and the refcv9 design is confirmed: progress head + a reserved clean calibration split.
* **O2 (P1 passes, P2 or G fails):** vision is real on a clean fit but not enough to beat the stage at this head class; next lever =
  a nonlinear head on the same split, then more clips.
* **O3 (P1 fails):** the cross-fit result does not transfer to a separate calibration set. Read D-fresh first: if flagged, the
  split is off-distribution (a data finding, not a head finding); if not, the CF2 gain was eval-specific and L1 is closed for refcv8.
