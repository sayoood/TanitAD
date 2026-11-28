# VERDICT_EGOWM: NO-EFFECT for E-KIN and for E-NOW (Master Mind, 2026-11-11 ~01:15)

Card `PREREG_V8F_EGOWM.md` (frozen 2026-11-09 19:36:33, prefix sha256 `ddd6fc9b...f446e7`). Tier **T0** (world-model diagnostic; adapter, readers and planner frozen at M2b, so driving cannot change). All numbers MEASURED: `read/results/arm_{E-KIN,E-NOW}_s{0,1}.json`, `read/results/arm_E-KIN_rep_s2.json`, control `ladder_read/results/arm_R35H_s{0,1}.json`, decision `read/VERDICT_EGOWM_IA0.json` (written by `read/verdict_egowm_ia0.py`, landed at `ee1ba225` before any seed-1 or replicate read existed). Estimator of each skill interval: episode-cluster bootstrap over the eval clips, B 2,000, seed 0 (it answers "another draw of episodes"; training variance is the replicate floor below). LONGITUDINAL and LATERAL decide only through the planner half (below); TACTICAL n = 0 (no decision output, card); STRATEGIC **PAUSED** (PI item 25).

## The read: IA-0 skill against Copy-Last at the band ends
| run | L1 2.0 s | L2 6.0 s | L3 12 s |
|---|---|---|---|
| R35H s0 / s1 (control) | 0.2594 / 0.2673 | 0.5458 / 0.5552 | 0.6721 / 0.6872 |
| E-KIN s0 / s1 | 0.2664 / 0.2788 | 0.5495 / 0.5673 | 0.6860 / 0.6825 |
| E-KIN rep s2 (floor) | 0.2782 | 0.5607 | 0.6954 |
| E-NOW s0 / s1 | 0.2586 / 0.2799 | 0.5427 / 0.5648 | 0.6767 / 0.6791 |
Every run BEATS Copy-Last and is HEALTHY at every level (IMAG1B's two-part rule; intervals in the result files). Every checkpoint is at step 19,164.

## The card's rule
Floor per level = the largest |skill difference| among E-KIN s0, s1, rep s2: **L1 0.0124, L2 0.0178, L3 0.0129**. An effect counts only if arm minus R35H (same seed) exceeds the floor of its level at both seeds 0 and 1.
| arm minus R35H, s0 / s1 | L1 | L2 | L3 |
|---|---|---|---|
| E-KIN | +0.0070 / +0.0115 | +0.0038 / +0.0122 | +0.0139 / -0.0047 |
| E-NOW | -0.0008 / +0.0126 | -0.0031 / +0.0096 | +0.0046 / -0.0081 |
No cell exceeds its floor at both seeds. Two cells exceed it at one seed and go the other way at the other (E-KIN L3 s0 +0.0139 against 0.0129; E-NOW L1 s1 +0.0126 against 0.0124); the card counts neither.

## Outcome: **NO-EFFECT** for E-KIN and for E-NOW
The deciding row is L1 at both seeds. Both SPEED-HELPS and SKILL-ONLY need it, so the planner half cannot change the outcome. HYPOTHESIS refuted, as the card words it: *"the predictor cannot tell how fast the scene will change"* is not what limits k-step skill on this rig. Measured speed given as an input (integrated over the actions, or held at NOW) moves skill by less than the gap between two training seeds of one arm. The card's "taken as actions" question (E-KIN against E-NOW) is moot: the two differ by less than the floor.

## Beside, no bar
* **Planner half** (committed, reported when it finishes): `read/run_lp_speed.py` units for the four arms, `report_speed.py` to `read/RESULT_EGOWM_PLAN.json`; R35H's own SCORE stage is rerunning (`ladder_plantest/run_lp_w3par.py`, IMPL_ADDENDUM_ORDER).
* Memory: two seed-1 reads were first refused by the 6 GB guard (`read/run_ia0_four_memguard.log`) and rerun with nothing else changed (`read/run_ia0_s1.log`).

## Next (Rule Zero, the card's committed row)
NO-EFFECT: *"the lever `PREREG_V8F_DECINFO.md` names (spatial pool or adapter)"*. That card is `wmpool/PREREG_V8F_WMPOOL.md` (draft landed `32c7910f`, frozen with this verdict): attention over a 2 x 5 pre-pooled adapter grid in place of the token mean into z1, three runs. Its Why states that DECINFO / DECN give the lever little support on speed. Box-2 slot after STOPBAL: the PI's ordering call.
