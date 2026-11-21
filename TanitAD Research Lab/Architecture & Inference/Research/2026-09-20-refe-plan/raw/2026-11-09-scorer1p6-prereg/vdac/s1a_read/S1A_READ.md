# S1A_READ: the pre-registered VDAC S1 read of HA-s0 and HA-SHUF (held-out navtrain H48)

Evidence class: MEASURED (`s1_read.py` on the banked exports; raw output `s1a_read.json`, `s1a_read_tables.md`, `run.log`). Tier: T0-like (head perception on banked frames; never driving performance). No navtest value was read. Cards used and verified by sha256: `PREREG_VDAC_P6.md` 5594c2ae... with A1 8f121bdf..., A2 c3d56358..., A3 a7708573..., A4 045a10c8... Reader frozen before the run (`READER_FREEZE.md`: reader sha256 72b4488a... at first freeze; one crash fixed after, new sha recorded there; the crash was in a sensitivity block, before any number existed).

## Headline

**Mechanical verdict: INCOMPLETE, and the two bars it could read both FAIL.** S1-P1 = 0.7802 (bar >= 0.80) and S1-Q = 0.8332 (bar >= 0.90) both miss, each with its whole 95 % interval below the bar. Both controls are VALID. S1-F, and the in-sample IoU the card needs to split S1-FIT (HB) from S1-WEAK (HB + HC), are not in the exports, so the reader emits `MISSING-INPUT` and refuses to pick a row. It is **not S1-GO**, so S1b (HA-s1) is not triggered: no `S1B_GO` was created and the pod was not touched.

## The bars (HA-s0; paired log-cluster bootstrap, 10,000 draws, seed 20260927; 3,207 frames, 45 logs, 3,207 sets carrying P6 labels)

| id | value | 95 % CI | bar | holds |
|---|---|---|---|---|
| **S1-P1** drivable IoU x in [0, 40) m, city-reweighted (LV 4,064 / BOS 3,731 / PIT 2,437 / SG 1,914) | **0.7802** | [0.7587, 0.7974] | >= 0.80 | **no** |
| **S1-Q** pooled AUC of -d_k for NAVSIM DAC = 0 (37,137 DAC-0 and 168,111 DAC-1 proposals) | **0.8332** | [0.7951, 0.8691] | >= 0.90 | **no** |
| **S1-F** trained minus the same seed's initial head, on S1-P1 and S1-Q | MISSING-INPUT | - | both lower bounds > 0 | unread |
| **C-SHUF** HA-SHUF S1-P1 | 0.4468 (CI [0.4223, 0.4703]) | - | < 0.50 | valid |
| **C-FLIP** HA-s0 mask y-flipped, S1-Q | 0.7056 (CI [0.6648, 0.7438]) | - | < 0.80 | valid |

Per city, S1-P1 [0, 40) IoU, HA-s0: LV 0.8914, BOS 0.7394, PIT 0.7234, SG 0.6960 (frames per city are in `s1a_read.json`; logs with frames LV 15 / BOS 10 / PIT 11 / SG 9). S1-COMP's condition (S1-Q holds, S1-P1 fails) does not apply because S1-Q fails; the pattern (LV above 0.80, three other cities 0.70-0.74) is descriptive only.

## Reported, not deciding

| item | value |
|---|---|
| IoU by band, HA-s0, city-reweighted | 0-20 m 0.8090 [0.7886, 0.8300]; 20-40 m 0.7498 [0.7216, 0.7686]; 40-72 m 0.6608 [0.6288, 0.6822] |
| HA-SHUF, S1-Q | 0.6677 (HA-SHUF per-city IoU: LV 0.485, BOS 0.506, PIT 0.421, SG 0.284) |
| true-field AUC on the same proposals (the true 2 m mask run through the same d_k pipeline) | 0.9408 [0.9320, 0.9491]: the proposals admit 0.94 with a perfect 2 m raster, so the 0.90 bar is reachable by a good enough head; HA-s0 is 0.108 below it |
| near-boundary SDF channel error (\|SDF_gt\| < 3 m, 1,236,457 cells) | MAE 1.6297 m, RMS 2.3626 m |
| sigma_b (RMS of the head's mask-EDT field against the true 2 m mask's EDT field, \|SDF_gt\| < 3 m, 917,355 cells) | 6.5905 m. Caveat: an EDT field grows without bound away from a wrong boundary, so this is far above the SDF-channel RMS; it matters only for VDAC-Phi, which A2 declared unread |
| footprint points outside the raster extent / proposals with no point inside | 0.0 / 0 |
| P6's own DAC logit AUC | MISSING-INPUT (P6's logits are not in the export) |
| S1-QT (tracked d_k) | MISSING-INPUT (no navtrain metric cache with t0 ego states) |
| in-sample IoU | MISSING-INPUT (head predictions were exported only on H48 frames) |

## Next step for this outcome

The card's table decides by (S1-P1, S1-Q, in-sample IoU) once both bars fail: **in-sample IoU >= 0.85 gives S1-FIT, next HB (a LoRA-adapted copy of P6's trunk for the head only); below 0.85 gives S1-WEAK, next HB and HC as one registered pair under their own card.** The in-sample IoU of HA-s0 is the only missing input that moves the row. The cheapest way to read it is a forward of the trained head (`vdac/ha-s0/aux_drv.pt`, LFS sha256 281320071fea..., verified) on a sample of training-set frames through P6's trunk, which needs the pod or the dev box with P6's checkpoint: that is the MM's step, not mine. S1-F (the initial head on the 3,207 H48 frames) needs the same forward with the seed-0 initial head, and is moot for the outcome (it can only matter when both bars hold). No S1b / HA-s1 is justified by this read. Nothing was changed on the pod or in the repo.

## Deviations and disclosures (all decided before the run unless stated)

1. **City reweighting formula** is the reader's reading of "city-reweighted": IoU = sum_c w_c (sum inter_c / sum union_c), w_c = n_c / 12,146. The card does not spell it out. It was frozen before the real run; the unweighted pooled IoU is not reported as a headline.
2. **45 groups, not 48.** Three H48 logs have no frame in the banks (BOS `2021.08.09.17.55.59_veh-28_00960_01031`, PIT `2021.08.17.17.17.01_veh-45_02098_02251`, BOS `2021.08.31.14.40.58_veh-40_01268_01618`), so the bootstrap draws over the 45 logs that carry frames. The all-48 version (empty logs carry zero weight) gives S1-P1 CI [0.7590, 0.7971] and S1-Q CI [0.7946, 0.8690]: the same conclusion.
3. **Population is the 3,207 frames the head forwarded**, not the store's 4,036 labelled H48 frames: the bank holds 1,705 held-out + 1,502 extra-log frames, all 3,207 carry a label and a P6 sub-score. The shards decode to 3,860 distinct tokens against the manifest's 4,036 entries (duplicate-token entries; the harness's own loader also keeps one label per token).
4. **d_k source geometry:** the export carries P6's 20 poses at 5 Hz; the reader applies the seam's `to_navsim` (to 8 poses at 0.5 s, verbatim, tested against the scalar loop) then the card's 5.1 densification to 41 poses at 0.1 s, 164 footprint points. Pose-only (A2's tracked variant is S1-QT, unread).
5. **Estimator:** 95 % interval = 2.5th / 97.5th percentile of the resamples; the CI of a bar does not decide it, the point estimate does (as the card words the bars). All 10,000 resamples were finite for the headline reads (`n_nan_resamples` in the json). The estimator answers "another draw of logs" only, not another head training seed (S1-R is not read) and not another inference run.
6. **Disclosures that travel with every S1 number (A3):** all H48 logs are in P6's own training set (held-out for the head, in-sample for P6's trunk); the head drew from P6's 60,000-set bank with `--epochs 2` named; HA-SHUF's keys are deranged over all 103,139 store keys.
7. **Preflight blocker history (A4):** the G20 bar moved 0.95 to 0.99 after readings were seen; it is an instrument constant of the training chain, not an outcome bar of this read.

## Controls and tests

C-SHUF and C-FLIP are computed on the predictions in the exports (HA-SHUF's mask against the true H48 labels; HA-s0's mask flipped about y = 0 and re-run through the identical d_k and AUC code). The reader's tests (`test_s1_read.py`, 18, synthetic fixtures only, `CUDA_VISIBLE_DEVICES=-1`, all OK in `test_results.txt`) cover: a perfect head (IoU 1.0, AUC 1.0, S1-GO with an initial head), a constant head (IoU 0.125, AUC 0.5), a shuffled-input head that passes (INVALID), the y-flip on the flipped prediction and a symmetric-road fixture where the flip control is blind (INVALID), a known city-reweighted value 1 - 0.4 x 3731/12146, the no-reweighting mutation (pooled 3600/4080, goes RED) and the per-frame-resampling mutation (CI differs, goes RED), the band decomposition, boundary values of the bars, every row of the outcome table, `to_navsim` against the verbatim loop, the draw against `rng.choice`, and the pooled AUC against brute force.

## Cleanup (rule 4, literal paths)

Deleted after copying everything of value into this hold: `F:/tanitad_scratch/vdac_s1a_scratch` (199 MB: the HF downloads, re-creatable from `Sayood/tanitad-refe-scorer1p6` with the shas in `inputs_manifest.json`, and the exported cards/code, re-creatable from git), `C:/Users/Admin/AppData/Local/Temp/loadchk.py` (272 B), the hold's `__pycache__`. Because the inputs are deleted, `inputs_manifest.json` paths point to the (now removed) scratch; re-download into `F:/tanitad_scratch/vdac_s1a_scratch/dl/vdac/...` to re-run.
