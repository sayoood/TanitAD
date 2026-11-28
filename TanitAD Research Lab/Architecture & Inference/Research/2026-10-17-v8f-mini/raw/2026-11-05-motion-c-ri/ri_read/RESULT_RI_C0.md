# RI-C0 read: the C0 base under reindexed frames reads CLOCK-INSENSITIVE (2026-11-05)

Card `PREREG_RI_C0.md` (sha256 `942b6391…`, frozen before any RI rollout). Tier **T1-kin** (self-action open loop on logged
cameras; `--scene reindexed` = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap,
B = 2000, seed 0 (gapdecomp.decompose via the RECOVER-NEXT reader loaded by file path); it answers another draw of EPISODES only.
Evidence class MEASURED (ours): `raw/RI_C0_READ.json`, rollouts `F:/TanitAD-scratch/motion_c/ri/out/RIC0{,C}_s{0,1}`, tool
`motion_c/ri/code/v8f_t1_eval_ri.py` sha256 `0417c838…`. Reused, read only: the C0 / C0C / FZC0 arrays of the RECOVER-NEXT reader.

## Verdict
**CLOCK-INSENSITIVE** at both seeds: M1 crawl share RIC0 − C0 is not separated (s0 +0.0001 [−0.0005, +0.0007], s1 +0.0002
[−0.0005, +0.0008]) and the M1 excess against P10 is the same under reindexed and logged frames (ratio 1.002 / 1.003).
The base does not use the camera clock, as the frozen features carrying no motion predicts (§7.31). Logged-frame T1-kin reads of
frozen-feature arms stand; (c) and (a) are still decided on `reindexed` (DESIGN_C.md section 4), because their levers can read it.

## Numbers (T1-kin, n = 2,399 windows of eval-141 per seed)
| quantity | s0 | s1 |
|---|---|---|
| M1 excess vs P10, reindexed / logged | +0.0533 / +0.0532 | +0.0596 / +0.0594 |
| ADE6 RIC0 − C0 (m) | +0.0018 [+0.0005, +0.0032] | +0.0012 [−0.0001, +0.0027] |
| along-track MAE RIC0 − C0 (m) | +0.0031 [+0.0014, +0.0049] | +0.0014 [+0.0002, +0.0029] |
| rows of RIC0 − C0 separated (of 19) | ADE6, along-track MAE, DK-TG, heading MAE, TAC-LON | along-track MAE |
| composite M1 vs P10, RIC0C / C0C | +0.0591 [+0.0403, +0.0812] / +0.0587 [+0.0401, +0.0812] | +0.0619 [+0.0395, +0.0889] / +0.0604 [+0.0383, +0.0868] |
| composite ADE6 vs P10 (m), RIC0C / C0C | −1.1190 / −1.1204 | −1.0066 / −1.0086 |
| Read A of the composite under RI (D1, D2, D3) | True, False (M1 alone), True | True, False (M1 alone), True |

The separated RIC0 − C0 rows are millimetre-scale (largest +0.0031 m). The composite is NOT DOMINANT under RI exactly as on logged
frames: D2 fails on M1 alone.

## Receipts and checks
* Non-vacuous: per rollout 7,197 tick items; rows after t0 reindexed (own progress behind the human's) 7,573 to 8,028, clamped
  (own progress ahead) 7,530 to 8,302; past rows 52,778, bit-identical to logged in every rollout.
* The P10 floor's kinematic rows are bit-identical between the RI and the logged runs at both seeds (0 of 2,399 windows differ on
  each of the 16 non-S-POS rows), so the RI − logged contrast carries no thread-count noise on the floor. The floor's S-POS rows
  differ (1,792 to 2,106 windows; max 8.9 m at s0, 28.1 m at s1), because those rows score S's checkpoints and S reads the frames.
  Disclosed: the STRATEGIC comparator of Read A depends on the frames.
* Tests of the tool: 12 passed, three mutations RED (`motion_c/ri/logs/`). Lateral offset is ignored by the projection (smoke:
  p90 0.9 m, max 9.2 m).
