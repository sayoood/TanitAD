# HANDOFF_LATSEM2 (agent, 2026-11-07 13:20, dev box CPU; nothing staged, committed or pushed)

## State
LATSEM2 ran once (run 1, 210 s, B 2,000 seed 0). **Read INVALID: controls 3 and 4 failed.** Controls 1 and 2 hold (exactly 0.0). Card prefix 3,769 bytes `1325f21c` untouched; log lines (a) 13:08:14 (IMPL_CHOICES, `b17688dc`) and (b) 13:17:27 (result) appended. Nothing was re-drawn or tuned to pass.

## Evidence for the Master Mind's amendment of controls 3 and 4 (all in `RESULT_LATSEM2.json`)
* **C3** (S1, 54 rows, one permutation draw each, "not separated above 0"): one row fails, `M2b_s0/L2/turn`, balanced accuracy minus majority +0.056 [+0.006, +0.110]. 20 extra draws per row (beside): classification rows separated above 0 in 18 of 360 draws (5.0 %), continuous rows 0 of 720. 18 classification rows therefore fail one draw with probability about 60 %; the committed bar cannot pass reliably, because balanced accuracy minus a majority baseline has a null centred on 0 while junk R2 is strongly negative.
* **C4** (label-shuffled S3, "not separated below 0"): 3 of 6 primary rows separated below 0 (s0 L2 -0.35, s0 L3 -0.80, s1 L3 -0.52 m); 20 extra draws: 25 of 120 rows below 0, 50 above. Cause: accelerate and brake move the imagined latent along a nearly fixed direction, so any probe direction reads a consistent nonzero difference; the episode bootstrap holds the probe fixed. The real S3 (+2.4 to +7.6 m) exceeds every shuffled draw (|point| at most 1.6 m) in all 6 rows, so a permutation reference on the point estimate (real outside the range of R shuffled draws) would pass it. Same for C3: compare the real S1 value with R permuted fits, not a one-draw interval.
* Any amendment is post hoc (numbers seen); `AMENDMENT_LATSEM_1/2` are the model for saying so. Run 2 recomputes from the same inputs: `cd theory/latsem2; export CUDA_VISIBLE_DEVICES=-1 WMACT_CODE_ROOT=<tip_1246c8dd>; python run_latsem2.py; python pictures_latsem2.py; python report_latsem2.py` (the script asserts the card and IMPL hashes; an amended control needs an IMPL/amendment sha bump in `run_latsem2.py`).

## Uncontrolled outcomes (NOT decisions)
No level SCENE-AWARE by the card (lead gap and agents within 30 m not separated positive at both seeds; only agents at L1 s0 +0.31). Lead present, 7-class label and turn ahead ENCODED at all levels, both seeds (+0.08 to +0.24). No SCENE-IMAGINING; a permuted-label probe is separated above 0 on 70 of 126 S2 cells (variance effect), the oracle is never separated on the gap. S3 accelerate minus brake > 0 (wrong sign) everywhere; label check: gap and ego speed correlate r +0.52. If the amendment holds this: card's next lever is MRATE W1, then a scene-target auxiliary for gap and agent count (own card). MRATE W0 / W1 not yet built.

## Not buildable
Nothing: all 7 targets built (n in IMPL_CHOICES section 3). Disclosed: join has 11 class strings, card says 10; closing speed drops |value| > 20 m/s (53 frames, lead-track switches); lead present = trainer's `lead_near` (<= 40 m).

## Manifest (all in `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/latsem2/` unless noted; sha256 first 12)
IMPL_CHOICES_LATSEM2.md `b17688dc0fea` (+ .sha256); RESULT_LATSEM2.md `95666577af69`; RESULT_LATSEM2.json `0c18d0251712`; run_latsem2.py `ccfa075f10c1`; labels_latsem2.py `92b673d1253d`; labels_latsem2.npz `ff49e0350a59`; pictures_latsem2.py `a628a9127378`; report_latsem2.py `2075d6da116b`; beside_gap_speed.py `8870aa186f5f`; align_check_latsem2.py `fc1efbeccb06`; BESIDE_GAP_SPEED.json, PICTURES_LATSEM2.json, labels_latsem2.report.json; logs run_latsem2.run1.log, labels_latsem2.log, pictures_latsem2.run2.log; fig/pca_levels_scene.png `dc0ef99d1098` (no frame, landable), fig/strip_CLOSE.png `9385359c599c`, fig/strip_FAR.png `bb61570c8339` (embed camera frames: keep OFF the public repo). Also edited: `theory/PREREG_V8F_LATSEM2.md` log lines only.
All unstaged on E:; land by the Master Mind.

## Deletions (own scratch, re-creatable)
latsem2/__pycache__ two .pyc (13,369 + 40,071 bytes) and dir; scratchpad/latsem2_scratch/align_check.py (1,280 bytes, copied to align_check_latsem2.py first) and dir. Nothing else touched; the pinned `tip_1246c8dd` export and `theory/wmact/__pycache__` belong to others and stay.
