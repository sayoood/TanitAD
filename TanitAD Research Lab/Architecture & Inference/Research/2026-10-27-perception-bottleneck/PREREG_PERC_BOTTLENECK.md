# PREREG — PERC-BOTTLENECK: where is refcv8's map / box bottleneck? (H-R9-PERCBN-1)

*TanitAD Research Lab (diagnosis), 2026-10-27 ~23:05 Berlin. Written and sealed (sha256 file beside it) BEFORE any
probe was fit or any number below was scored. Dev-box RTX 4060 only; no Thor contact.*

Checkpoint: refcv8 EMA-48,852, `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt` (md5 recorded at run time; battery
record `ckpt_48852.md5`). Config `E:/refcv8_eval_kit/ckpt/config.json`. Loader: the battery's `refcv7_loader`
through `r8_repoint` (launch tree `tree_fb47a58`), `R8_NO_POSE_SYNC=1` (the clock MAPSEL scored on).

## Why
refcv9 freezes trunk + BEV lift + BEV encoder + box head and trains only `_map_hires.{refine,near,near_refine}`.
MAPFT already showed the head's CE converged on frozen features without the map moving. Before asking for compute
to train perception we need the cheapest discriminating measurement: is the information that the head fails to
read PRESENT in the frozen features (head/loss is the lever) or ABSENT (features are the ceiling)?

## Q1 — features or head? (primary)

**Data.** FIT = the 139 TRAIN clips of `refcv6-b1-416x1024-train-a6` (all 139 have a SAM3 10 cm GT file in
`E:/refcv7_eval_kit/prior/gt_train300`; 0 overlap with eval139 by that set's rule). SCORE = eval139 clips with a GT
file (`sam3_gt_v3_eval`, 137/138 expected). Windows: 4 per clip at the quantiles 0.125/0.375/0.625/0.875 of the
clip's eval-index positions (deterministic, no look at content). GT = the window's `map_fine` (current frame),
read by the loader exactly as the trainer reads it.

**Cells.** 0.25 m lift-grid cell centres in the corridor x in [0, 40) m, |y| < 3 m, every 2nd row and column
(80 x 12 = 960 cells/window). Each cell is labelled by the 10 cm GT cell that contains its centre; the head is read
at the SAME 10 cm cell. Scored only where GT is seen (code != 255). Strata: lift-valid (any height projects into
the image) vs lift-invalid.
Bands: **A = x 0-20 m (primary)**, A0 = 0-5 m, **B = 5-40 m (secondary)**.

**Feature stages** (forward hooks, no code change to the model):
- `enc` — `HiresBEVEncoder` output, 64-d @0.25 m: the frozen input of `refine` in refcv9.
- `lift` — lift output, 64-d @0.25 m, before the BEV encoder.
- `img` — the trunk's stride-8 map (`fmap_s8`, 512-d) bilinearly sampled at the cell's projection at the 4 lift
  heights, valid-masked: 2,048-d = exactly the lift's (and the near lift's) input.
- `enc+img` — concatenation: everything frozen that the refcv9-trainable head can read at x < 20 m.

**Probes.** (i) linear softmax over the 8 classes per cell; (ii) MLP (one hidden layer 256, GELU) per cell;
(iii) 3x3-conv probe (two 3x3 convs, width 64) on the `enc` patch (x 0-40 m, |y| < 4 m for context).
All hyper-parameters (weight decay from {1e-4, 1e-3, 1e-2}, epochs by early stop) are chosen on a clip-disjoint
20 % VALIDATION split of FIT only; the SCORE split is never used for any choice. Standardisation statistics from
FIT only. Two probe seeds; both reported.

**Mandatory controls** (each must read its known value or the panel is void):
- C1 CONSTANT: FIT global class frequency as the posterior everywhere -> AP must equal the SCORE positive rate
  (within 1e-6) and AUROC must be 0.5.
- C2 POSITIONAL PRIOR: per-(row, col) FIT class frequency (no image).
- C3 n (cells, windows, clips) and d printed for every probe row.
- C4 HEAD ON ITS OWN DECISION: the head's per-cell precision/recall at its declared `prior_corrected` decision
  must reproduce the MAPSEL banded sidewalk precision (0-5 m ~0.25, 5-40 m ~0.72-0.74) within +-0.05, or the cell
  join is suspect and the verdict is withheld.

**Metrics** (per band x stratum): sidewalk (code 7) vs rest — **AP (primary)**, AUROC, precision at the head's own
recall; drivable set {1,2,3,4,6} vs rest — AP; 8-class pixel accuracy. Head score = its softmax posterior.
Added by the Master Mind before sealing (descriptive, not a decision input): the same sidewalk AP and precision at
the head's recall per range bin **0-5, 5-10, 10-20, 20-40 m**, for the head, the best probe and C2.
Estimator: paired episode-cluster bootstrap over SCORE clips (1,000 draws, seed 0, 95 % percentile). It answers
"another draw of EPISODES"; probe-seed spread is reported beside it; the head is ONE training seed.

**Decision rule (committed).** Δ = AP(best probe) − AP(head), sidewalk, band A lift-valid (primary) and band B.
- **O1 HEAD-LIMITED:** Δ >= +0.05 with CI lower bound > 0 -> the head/loss is the lever (refcv9's head-only
  training is aimed correctly; widen the head / change the loss).
- **O2 FEATURE-LIMITED:** Δ < +0.05 or CI includes 0 -> the frozen features are the ceiling; head-only training
  cannot move the map materially; the lever is perception training.
- **Localisation (read under O2):** Δ_lift = AP(best `img` probe) − AP(best `enc` probe). >= +0.05 with CI > 0 ->
  the lift/BEV encoder loses information (lever: unfreeze lift + BEV encoder, 0.54 M params); otherwise the image
  trunk is the ceiling (lever: unfreeze the backbone / a stronger pretrained encoder / temporal fusion).
- **Data-limited check:** the best probe refit on 50 % of FIT windows. If doubling the data adds >= 0.02 AP the
  probe is data-limited and Δ is a LOWER bound (stated).
- **Visibility (band A0):** share of A0 GT-seen cells that are lift-invalid, and share of the head's A0 sidewalk
  flags on them. If > 50 % of A0 flags sit on lift-invalid cells, the 0-5 m error is VISIBILITY (no single-frame
  feature can fix it; levers: temporal BEV fusion, or masking/supervising those cells), not features or head.
- Function class: a linear/MLP probe that fails shows only "not readable by that class"; the conv probe extends it.
  The verdict names the class.

## Q2 — is the GT right where the own map disagrees?
Population: own-sidewalk (decided code 7) cells on GT-drivable at x 0-20 m, |y| < 3 m, SCORE windows.
Instrument: `bev-lidar-gt-b1eval-20260913` (`tanitad.lidar_bev_gt/2`, 0.5 m cart grid, col0 = RIGHT).
Its schema carries occupancy (z 0.30-3.0 m), observed and point counts but **no ground-height channel**, so a curb
(0.10-0.15 m step) is below its 0.30 m band: the curb test as briefed is NOT computable from this artifact — stated
as a blocker, not narrowed silently. What IS computable: the share of those cells that LiDAR marks OCCUPIED (an
obstacle on a GT-"drivable" cell, i.e. a GT error or a parked object), against two controls on the same windows:
own-drivable-on-GT-drivable cells (agreement control) and the same cells read 20 frames away (mispaired-time
control). Committed reading: FP-cell occupancy >= 2x the agreement control AND above the mispaired control ->
a GT-or-obstacle share of the "false positives" exists and is reported as an upper bound on the precision
correction; otherwise the FPs are not explained by obstacles. Raw sweeps exist on the dev box for two clips
only (`E:/sam3map_lidar`); a curb-height read there is DESCRIPTIVE (n = 2 clips) if the clips are in eval139.

## Q3 — the box head (CPU, zero GPU)
Source: PERC8's banked eval-grade dump of the SAME checkpoint (`refcv8_pf/perc8/raw/A_nosync/perc_det.pkl`,
md5 b4364029... in its `perc.json`), S2 windows, 139 eval episodes. Read: per-class AP@2 m (re-derived beside
PERC8's), the class-confusion matrix of matched true positives (GT class x predicted argmax class), per-class
size error (|L|,|W|,|H|), and controls: C-MAJ (predict the majority class for every match: accuracy = majority
share; a collapsed head reads at or below it), C-SIZE (per-class train mean size as the prediction).
Added by the Master Mind before sealing (descriptive): AP@2 m by range bin (0-20 / 20-40 / 40-60 m of the GT
object) and by object size class (GT footprint L x W: small < 2 m², medium 2-10 m², large >= 10 m²).
Committed reading: class COLLAPSED iff TP class accuracy <= C-MAJ + 0.02 or >= 90 % of non-automobile TPs are
predicted automobile.

## Ranking the levers
The RESULT ranks {unfreeze backbone+lift with map/box losses, stronger pretrained encoder, temporal BEV fusion,
better/more map GT, head/loss change} ONLY by the Q1-Q3 measurements above; anything else is labelled
HYPOTHESIS or PUBLISHED-SECONDARY.
