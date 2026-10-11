# RESULT -- does refcv7-r101-s0 @ 50,400 use its PAST camera frames? (EvalFlyWheel, 2026-10-12)

**PREREG sha256 (recorded BEFORE any scored number existed; one typo fixed before the first roll, this is the final):**
`ac66a1c0e1da79ac77af65817a2406f98ca1924fa6196b123f77d7bc3e8581d9` (`PREREG.md`). Tier: **T1 self-action OPEN LOOP**, logged eval139 frames,
status UNRULED for an action-free model -- never closed loop, never driving performance. Evidence class: MEASURED (artifacts in `raw/`).
Checkpoint `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt` (md5 b418d0fc... per r7base RESULT). Tip 48435a3. Dev-box RTX 4060: **fits, peak 1.1-1.4 GiB** allocated at batch 1, 416x1024, the battery's own bf16 trunk -- no fp16 fallback or window cut needed for memory. Thor untouched.

## Verdict (as pre-registered): **O3 -- MIXED / INDETERMINATE. Not O1, not O2.**
On G3 (1,253 S6-eligible windows, 139 episodes) every NAVSIM-style arm has all five primaries INDETERMINATE: no primary is DEGRADED, none is
proven EQUIVALENT. The PREREG rule says what O3 means: *report which primaries moved; the next lever is the full-S6 pass, not an interpretation.*
That pass is **already running** (see `HANDOFF.md`); this document is the G3 verdict only and is not to be upgraded by it retroactively.

## What was run (G3, S6 horizons 1..6 s; S2 = the same windows' first 4 slots)
Arms, same checkpoint/windows/nav/v0/ego-history/labels; only the camera pixels change (details `PREREG.md` s2):
A REAL (seed 0) - A-rep (REAL, inference seed 1 = noise floor) - B HOLD (each row's K=3 stack <- 3 copies of that row's newest frame; isolates the K-fusion) -
C NAVSIM-ST (all 10 slots <- t0) - D NAVSIM-NT (slots <- nearest 2 Hz frame; **84 of 1,253 windows had the t0-1.0 s frame clamped to the episode's first frame**) - E BLIND (grey; 12 episodes).
Grid: every 3rd S6-eligible window per episode (1,253 of 3,668; list sha256 `dd6a9063...`, raw clip ids not landed). Not the K=8 EVAL-DIAG grid.

### Controls (read before the arms)
* **E BLIND went RED**: S6 ADE +5.163 m [+3.231, +7.201] vs REAL (paired, n 108 windows / 12 episodes) -- the instrument can see images (`raw/g3_blind_control.json`).
* Determinism: REAL rolled twice, same seed/windows -> bit-identical (18/18). Against the BANKED `dump_s1` (aligned RNG prefix, episodes 0-1, 69 windows) 58/69 windows are bit-equal; the 11 others differ by discrete selection flips with the sampler stream intact (`raw/controls.json`). A subsampled grid cannot reproduce the banked per-window draws (sequential RNG stream), so REAL was **re-rolled**, not borrowed from the banked dump.
* The brief's "inference-seed replicate" is A-rep. Question each interval answers: **episode-cluster paired bootstrap = would another draw of EPISODES say this?; A-rep delta = would another INFERENCE run say this?; neither answers TRAINING-seed variance (one checkpoint).**

### S6 paired deltas, arm - REAL (negative = arm better for errors; n 1253 windows / 139 episodes; SEP = CI excludes 0)
| family | cell | REAL | A-rep (noise floor) | B HOLD | C ST | D NT |
|---|---|---|---|---|---|---|
| ADE | ade_m | 2.6886 | +0.049 [+0.008, +0.093] SEP | +0.016 [-0.032, +0.071] | +0.040 [-0.022, +0.102] | +0.039 [-0.005, +0.089] |
| LON | along-track MAE m | 2.3155 | +0.0436 [+0.0009, +0.0854] SEP | +0.0136 [-0.0219, +0.0505] | +0.0384 [-0.0082, +0.0841] | +0.0278 [-0.0125, +0.0688] |
| LON | speed MAE m/s | 0.9714 | +0.0140 [-0.0024, +0.0307] | +0.0031 [-0.0099, +0.0157] | +0.0159 [-0.0012, +0.0337] | +0.0071 [-0.0075, +0.0222] |
| LON | accel MAE m/s2 | 0.4398 | +0.0038 [-0.0016, +0.0092] | +0.0022 [-0.0019, +0.0063] | +0.0064 [+0.0011, +0.0122] SEP | +0.0032 [-0.0013, +0.0076] |
| LON | **LEAD-VIOL share_of_at_risk** (progress ratio ~1.10 in every arm) | 0.0596 | +0.0000 [-0.0170, +0.0154] | +0.0000 [-0.0102, +0.0102] | +0.0000 [-0.0136, +0.0141] | +0.0000 [-0.0102, +0.0102] |
| LAT | cross-track MAE m | 0.8345 | +0.0201 [-0.0024, +0.0490] | +0.0091 [-0.0288, +0.0510] | +0.0064 [-0.0397, +0.0505] | +0.0249 [-0.0031, +0.0608] |
| LAT | heading MAE deg | 3.4063 | +0.111 [-0.018, +0.277] | +0.118 [-0.051, +0.327] | +0.114 [-0.062, +0.319] | +0.165 [+0.014, +0.369] SEP |
| LAT | yaw-rate MAE (valid) rad/s | 0.0287 | +0.0005 | +0.0006 | +0.0006 | +0.0010 [+0.0002, +0.0020] SEP |
| TAC | lon-decision acc (traj-derived) | 0.6018 | +0.0016 [-0.0159, +0.0184] | -0.0056 [-0.0175, +0.0064] | -0.0104 [-0.0271, +0.0056] | -0.0064 [-0.0184, +0.0056] |
| TAC | lat-decision acc | 0.8795 | -0.0024 [-0.0104, +0.0056] | +0.0016 [-0.0056, +0.0088] | +0.0024 [-0.0048, +0.0096] | +0.0032 [-0.0048, +0.0104] |
| STRAT | -- | NOT APPLICABLE, n = 0 (strategic layer OFF, PI R5) | | | | |
Full tables incl. S2, FDE, curvature/target-speed cells (per-arm unpaired) and distance keeping: `raw/tables_g3.md`, `raw/g3_cross_paired_s{6,2}.json`, `raw/g3_lead_viol.json`.
LEAD-VIOL S6: 17 violations in REAL and in every arm (392 lead windows); the S2 block has 2 violations in REAL, 1 in B/C/D -- **underpowered, not a finding**.

### PREREG s5 classification (`raw/g3_verdict_s6.json`)
B HOLD, C ST, D NT: **MIXED** -- P1-P5 all INDETERMINATE (CIs reach beyond the 2 % margin of 0.054 m ADE, and |delta| <= the inference-seed replicate's own +0.049 m). A-rep itself has a "separated" ADE/along-track delta (H-ESTIM-SEED-1 in the flesh: a separated paired CI is not an effect).

## What it does and does not support (read as bounds, not as the verdict)
* MEASURED bounds on this checkpoint/grid: the harshest NAVSIM construction (ST, no motion in any image slot) moves S6 ADE by **+0.040 m (+1.5 %), 95 % upper bound +0.102 m (+3.8 %)**; NT +0.039 (upper +0.089, 3.3 %); HOLD +0.016 (upper +0.071, 2.6 %). Every point estimate is at or below one inference-seed replicate. LEAD-VIOL does not move. So there is **no sign of a large history-mismatch penalty (O2 is not supported)**, but the 2 % equivalence margin is **not** cleared (O1 is not licensed) at n = 1,253.
* Only separated cells beyond the floor-class are small: ST accel MAE +1.5 %, NT heading +4.8 % / yaw-rate +3.5 % (S6) and NT S2 ADE +0.004 m -- the NT/ST rows are where any real, small history reliance would live. Post-hoc reading; not promoted.
* HYPOTHESIS (not tested): ego history (REAL in every arm) already carries the motion the image stack would; a joint ablation is a different experiment.
* Scope: T1 open loop on PhysicalAI; one checkpoint; NAVSIM's own bridge also interpolates ego history (not applied here); **this does not measure NAVSIM PDMS**.

## Implication for refcv9's input (conditional on the full-S6 result)
No evidence for the 2 Hz-hold augmentation lever (O2). The NAVSIM-native contract (K=1, rows at 0.5 s, W=4; 10 -> 4 trunk frames, ~56 % FLOPs ESTIMATED) is **plausible but unproven**: it stays a refcv9 candidate pending the full-S6 pass (3,668 windows ~ 1.7x narrower CI). If that pass leaves ST/NT INDETERMINATE the remaining discriminator is a training-side A/B, not more eval.

## Deliverable manifest
`stage/<repo path>/temporal_history_2026-10-12/`: `PREREG.md`, `RESULT.md`, `HANDOFF.md`, `code/{make_g3_windows,hold_roll,hist_analyze,blind_control,make_tables}.py`, `raw/*` (above). Top-level: `C:/Users/Admin/qland/hold_temporal/RESULT.md`, `msg.txt`.
Off-git, dev box only: roll dumps `E:/hold_temporal/work/dumps/{real_s0,real_s1,hold,st,nt,blind}`, analysis `E:/hold_temporal/work/analysis`, tip clone `E:/hold_temporal/tip`, battery code copy `E:/hold_temporal/work/battery`. Full-S6 extension: `E:/hold_temporal/work/{dumps/f_*,analysis_full}`.
Suggested register line (for the Master Mind): H-HIST-1 -- G3 verdict O3/INDETERMINATE; bounds above; extension pending.
