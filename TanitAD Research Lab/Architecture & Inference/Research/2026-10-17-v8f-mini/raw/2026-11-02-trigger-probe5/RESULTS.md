# RESULTS -- trigger probe 5: do NATIVE-SCALE crops of the traffic-light band and the lead-vehicle band carry the go/stay trigger that the pooled frozen DINOv3 cache does not?

**STATUS: FINAL, 2026-11-02 (report written from `raw/VERDICT.json`, `raw/res_<pop>_p5.json`, `raw/results_strata_p5.json`, `raw/results_eval141_p5.json`; the box-2 encode ran 14:48-15:30 rc 0, `raw/CHAIN_P5.done` 15:31, the six readout stages ran 15:31-15:36, `raw/READOUT.done` 15:36).**
The design is frozen in `PREREG_FROZEN.md` (sha256 `791e8a62...aa62`, re-hashed for this report and equal, 2026-11-02T14:03:08+01:00, `FROZEN_SHA.txt`), before any crop feature, label statistic or probe output of this package. No post-freeze edit of the card.
**Evidence class: MEASURED. Tier: T0-class probe of representations on logged inputs, NOT driving performance.**
Estimator: episode-cluster bootstrap, B = 2,000, seed 0, on the mean of per-fold AUCs (pooled beside), paired on the same windows; it answers another draw of EPISODES. This run has no seed spread: the readout is an L2 logistic on PCA-64 (one PCA seed, 0), the encoder is one fixed frozen network, and no replicate was run, so another TRAINING run and another PCA draw are not answered (H-ESTIM-SEED-1).
Same 12,000 + 12,000 TRAIN windows (P-STOP 849 clips, P-MOVE 2,140 clips; 6,000 GO each), folds and resamples as probes 1-4; E and V+E are probe 1's banked out-of-fold scores (the run asserts the banked rows equal the sampled rows).

## Verdict (committed rule, PREREG 5): **CROPS DO NOT HELP**
Rule as written: *CROPS HELP iff dAUC >= 0.03 with CI lower > 0 in either population (P-STOP, P-MOVE); CROPS DO NOT HELP iff dAUC < 0.01 in both populations; otherwise INCONCLUSIVE; a cell whose controls FAIL is INVALID; a missing cell makes the verdict INCOMPLETE.* dAUC = fold-mean AUC(C+V+E) - AUC(V+E), paired, CI = 2.5 / 97.5 percentiles.

| decision cell | dAUC (fold-mean) [CI] | HELP test (>= 0.03 and CI lower > 0) | NOT-HELP test (< 0.01) | controls | cell status |
|---|---|---|---|---|---|
| P-STOP: C+V+E - V+E | **-0.0231 [-0.0422, -0.0050]** | no | yes | PASS | valid, < 0.01 |
| P-MOVE: C+V+E - V+E | **-0.0028 [-0.0081, +0.0024]** | no | yes | PASS | valid, < 0.01 |

Both cells present, both valid, both below 0.01, neither reaches +0.03: the rule returns CROPS DO NOT HELP (`raw/VERDICT.json`, `raw/logs/readout_decide.log`). The rule names no other decision cell; everything below is beside it and changes nothing.
* Pooled (non-decision) AUC gives the same sign: P-STOP -0.0211 [-0.0401, -0.0029], P-MOVE -0.0030 [-0.0081, +0.0022].
* On P-STOP the paired CI sits entirely below 0: the 200-feature arm reads below V+E. The card's own caveat applies (C+V+E has 128 more features than V+E, and the contrast includes that cost); this is not a statement that the crops carry anti-signal.

## Arm table (fold-mean AUC [CI]; pooled in the last line)
| arm | P-STOP | P-MOVE |
|---|---|---|
| E (ego only, banked) | 0.626 [0.613, 0.640] | 0.732 [0.719, 0.745] |
| V+E (probe 1, whole-vector PCA-64, banked) | 0.680 [0.650, 0.710] | 0.785 [0.771, 0.798] |
| TL+E (traffic-light crop PCA-64) | 0.661 [0.631, 0.690] | 0.756 [0.742, 0.771] |
| LEAD+E (lead-vehicle crop PCA-64) | 0.666 [0.637, 0.696] | 0.784 [0.771, 0.797] |
| C+E (TL64 + LEAD64 + E) | 0.656 [0.622, 0.688] | 0.777 [0.764, 0.791] |
| **C+V+E** (TL64 + LEAD64 + V64 + E, 200 features) | 0.657 [0.623, 0.690] | 0.782 [0.768, 0.795] |

Pooled AUC beside: P-STOP E 0.593, V+E 0.680, TL+E 0.659, LEAD+E 0.664, C+E 0.660, C+V+E 0.659; P-MOVE E 0.731, V+E 0.784, TL+E 0.756, LEAD+E 0.784, C+E 0.777, C+V+E 0.782 (E's pooled and fold-mean AUC differ on P-STOP, 0.593 against 0.626: the per-fold score offsets; the decision metric is the fold-mean).

Paired contrasts (fold-mean dAUC [CI], beside the verdict):

| contrast | P-STOP | P-MOVE |
|---|---|---|
| V+E - E | +0.054 [+0.025, +0.082] | +0.053 [+0.041, +0.065] |
| TL+E - E | +0.035 [+0.004, +0.065] | +0.025 [+0.014, +0.035] |
| LEAD+E - E | +0.040 [+0.011, +0.068] | +0.053 [+0.040, +0.065] |
| C+E - E | +0.031 [-0.004, +0.063] | +0.045 [+0.033, +0.059] |
| C+V+E - E | +0.031 [-0.004, +0.064] | +0.050 [+0.037, +0.064] |
| TL+E - V+E | -0.019 [-0.045, +0.007] | -0.029 [-0.037, -0.020] |
| LEAD+E - V+E | -0.014 [-0.036, +0.010] | -0.001 [-0.007, +0.006] |

Reading: each crop carries ego-independent signal (LEAD+E reaches V+E on P-MOVE, +0.053 over E), but it is signal the whole-frame V already carries: neither crop arm passes V+E, and stacking the crops on V+E adds nothing in either population. Inner-CV picked C = 0.001, the smallest value of the committed grid, in 5 of 5 outer folds for C+E and C+V+E in both populations (and for TL+E on P-MOVE); a smaller C was not searched (the grid is committed).

## Controls (PREREG 7, run for every source name; all PASS in both populations, `controls_ok` true, `control_failures` empty)
| control | P-STOP | P-MOVE |
|---|---|---|
| C1 shuffled labels, fold-mean AUC [CI], TL+E | 0.504 [0.494, 0.514] | 0.497 [0.486, 0.507] |
| C1 LEAD+E | 0.503 [0.493, 0.513] | 0.501 [0.490, 0.512] |
| C1 C+V+E | 0.501 [0.491, 0.511] | 0.505 [0.495, 0.515] |
| C1 V+E | 0.501 [0.491, 0.510] | 0.494 [0.484, 0.505] |
| C1 paired null C+V+E - V+E | +0.0008 [-0.0095, +0.0114] | **+0.0106 [-0.00004, +0.0210]** |
| C2 donor-episode labels (seed 2) TL+E | 0.518 [0.476, 0.558] | 0.501 [0.480, 0.523] |
| C2 LEAD+E | 0.469 [0.424, 0.512] | **0.468 [0.446, 0.489]** (misses 0.5, 5-draw band run) |
| C2 C+V+E | 0.488 [0.445, 0.528] | 0.484 [0.463, 0.506] |
| C2 V+E | 0.501 [0.456, 0.545] | 0.486 [0.465, 0.507] |
| C3 deliberate leak (GT label appended), TL+E / LEAD+E / C+V+E | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

C1 and C2 were also run for E and C+E and cover 0.5; C3 for V+E and C+E reads 1.000. Known answer (PREREG 3): PASS, see Disclosures.
* **P-MOVE C2 LEAD+E:** the seed-2 CI [0.446, 0.489] lies below 0.5, so seeds 3-6 were run: 0.504 / 0.501 / 0.528 / 0.480; the 5-draw mean fold-mean AUC is 0.4962, inside the committed band [0.47, 0.53] (`C2band:LEAD+E`), so the cell is not invalidated. Seed 5's own CI [0.506, 0.549] sits above 0.5; the committed criterion is the 5-draw mean and it passes.
* **P-MOVE C1 paired null:** its CI lower bound is -0.00004, so it covers 0 by 4e-5 and passes the committed test as written; its point estimate (+0.0106) is the size of the NOT-HELP threshold, so a null contrast on this rig can read about +0.01 and the C1 null is closer to the edge than the other controls. It does not touch the verdict (the real-label contrast is -0.0028 with CI upper +0.0024, below +0.01 and far below +0.03).

## Disclosures
1. **Align gate (known answer, PREREG 3): PASS.** 32 FULL frames (24 train, 8 eval windows) rectified at the cache's own 256x640 geometry, lossless PNG, through box 2's encoder entry, 4x4 fp32 average to `pool4x10`, against the cached rows: per-row relative L2 median **0.0264**, mean 0.0265, min 0.0261, max 0.0269 (gate: median <= 0.05, any row > 0.5 refuses). Weights sha256 `dcb2e451...8179` equal the asserted value (`raw/align_check.json`, `raw/align_info.json`, `raw/ALIGN_GATE_PASS`). It ran at 14:12, before the main encode.
2. **What the gate does and does not prove.** It proves the encoder entry, the weights, the frame index and the token order on the cache's frame. It does not run the native-scale path: the 768x1920 rectification and the two crop slices rest on pytest (`test_trigger_probe5.py`, 10 passed; re-run for this report in 3.1 s) for the slice identity (crop = slice of the full rectification, bit-exact) and the f_ref ratio (exactly 3.0). **PREREG 3 also lists a pixel test (average-pool-by-3 of the native-scale frame against the 256x640 frame is close); no test of that name or content exists in the test file, so that item was not run: UNVERIFIED.**
3. **Crops are cut from the 768x1920 native-scale cylindrical frame** (f_ref 916.7324 = 3 x 305.5775, hfov 120; measured median native focal 928.08 px/rad, 1.2 % above the frame's focal): TL rows [0,336) cols [384,1536) = 21x72 = 1,512 tokens; LEAD rows [272,608) cols [480,1440) = 21x60 = 1,260 tokens. 24,608 frames (24,000 train + 608 eval), 2,252 clips, 23 shards, every shard verified by sha256 on both ends (`raw/FEAT_SHA256.txt`, 23 lines). The cache saw 256x640; these crops see 3x the scale on a sub-frame, one frame only.
4. **JPEG transport changes the features by a median of 6.4 %, and it was NOT gated.** The crops travelled dev box -> box 2 as JPEG q95, 4:4:4 (PNG was about 50 GB; box 2 had 54 GB free); the card committed to measuring the effect and reporting it. MEASURED on the 32 alignment frames x 2 crops (64 rows, same encoder, lossless-PNG crop against JPEG crop, 10,240-d feature): relative L2 **median 0.0637, max 0.1228** (2.4x the align gate's own 0.0264). This is an **uncontrolled input difference**: the crop features come through a JPEG step that the lossless-PNG known answer did not include, and a PNG re-encode of the main run was not done. Direction of possible bias: every window of both classes and both splits went through the same transport, so it is a label-independent perturbation and cannot manufacture a GO/STAY signal; it can only add feature noise, which attenuates fine detail the crop arms read (a lamp state, a distant lead), i.e. it can only bias toward NOT-HELP. How much AUC it cost is not measured, so a lift smaller than that attenuation cannot be excluded by this run.
5. **Grid-edge regularisation.** C = 0.001 (the lowest committed value) was selected in every outer fold for C+E and C+V+E (both populations); TL+E and LEAD+E on P-STOP alternate between the two grid ends (0.001 and 10), i.e. the inner 3-fold is noisy there.
6. **Process note from `HANDOFF.md`:** `chain_p5.sh` was relaunched at 14:19:40 after a fix (the local `sha256sum` prints ` *name`; normalised before the diff). It changed no design element and no number. A smoke run on random features (`F:/TanitAD-scratch/trigger_probe5_smoke/`, disposable) preceded the real run; its numbers mean nothing and are not used.
7. **Decision metric differs from the pooled AUC for E on P-STOP** (0.626 fold-mean, 0.593 pooled); the card commits the fold-mean and both are reported.

## Beside (descriptive, never decisive)
**Strata (TRAIN, fold-mean, paired; `raw/results_strata_p5.json`; TL strata are clip-level VLM labels and lead_depart reads the future; TL_YELLOW is too small to score, 164 / 86 windows):**

| pop | stratum | n / clips | E | V+E | TL+E | LEAD+E | C+V+E | TL+E - E | LEAD+E - E | C+V+E - V+E |
|---|---|---|---|---|---|---|---|---|---|---|
| P-STOP | TL_RED | 4986 / 256 | 0.609 | 0.708 | 0.654 | 0.686 | 0.658 | +0.044 [-0.015, +0.098] | +0.077 [+0.023, +0.123] | -0.051 [-0.083, -0.021] |
| P-STOP | TL_GREEN | 1377 / 123 | 0.594 | 0.536 | 0.621 | 0.553 | 0.512 | +0.027 [-0.061, +0.148] | -0.040 [-0.111, +0.069] | -0.024 [-0.072, +0.034] |
| P-STOP | NO_TL | 5348 / 451 | 0.621 | 0.682 | 0.674 | 0.674 | 0.671 | +0.053 [+0.012, +0.087] | +0.053 [+0.014, +0.089] | -0.011 [-0.035, +0.014] |
| P-STOP | lead_depart | 1741 / 310 | 0.663 | 0.644 | 0.628 | 0.634 | 0.625 | -0.035 [-0.109, +0.040] | -0.029 [-0.096, +0.026] | -0.019 [-0.055, +0.015] |
| P-MOVE | TL_RED | 1233 / 302 | 0.876 | 0.879 | 0.860 | 0.884 | 0.871 | -0.016 [-0.036, +0.002] | +0.008 [-0.018, +0.031] | -0.008 [-0.019, +0.002] |
| P-MOVE | TL_GREEN | 1168 / 221 | 0.729 | 0.735 | 0.755 | 0.731 | 0.741 | +0.026 [-0.009, +0.061] | +0.003 [-0.053, +0.065] | +0.006 [-0.017, +0.027] |
| P-MOVE | NO_TL | 9430 / 1587 | 0.704 | 0.772 | 0.738 | 0.771 | 0.770 | +0.034 [+0.022, +0.048] | +0.066 [+0.053, +0.081] | -0.001 [-0.008, +0.005] |
| P-MOVE | lead_depart | 2010 / 641 | 0.739 | 0.749 | 0.749 | 0.749 | 0.747 | +0.010 [-0.015, +0.036] | +0.010 [-0.021, +0.043] | -0.001 [-0.016, +0.013] |

The other strata (TL_ANY, lead_now, no_lead) are in the JSON. C+V+E - V+E is negative in every scored P-STOP stratum (-0.011 to -0.051; CI excludes 0 only for TL_ANY, -0.030 [-0.057, -0.006], and TL_RED) and between -0.008 and +0.006 in every scored P-MOVE stratum, every CI covering 0.
* **TL_RED, TL_GREEN, lead_depart (the three the card names):** no stratum shows C+V+E separated above V+E (the one positive point estimate among them is P-MOVE TL_GREEN, +0.006 [-0.017, +0.027]); on P-STOP TL_RED it is separated below (-0.051). On P-STOP TL_GREEN the TL crop is the only image arm above E (+0.027 [-0.061, +0.148]; V+E, LEAD+E and C+V+E are below E), and on P-MOVE TL_GREEN TL+E is the highest arm (+0.026 [-0.009, +0.061]); neither CI excludes 0: a direction, not a finding. On lead_depart no image arm, crops included, beats E on P-STOP (E 0.663 against 0.625-0.644), and on P-MOVE all arms sit at 0.747-0.749 against E 0.739, every CI covering 0.
* **Eval-141, no refit** (fit once on all sampled TRAIN windows, applied unchanged, pooled AUC; small: P-STOP n 101 windows / 19 episodes, P-MOVE n 507 / 67):

| pop | E | TL+E | LEAD+E | V+E | C+V+E | C+V+E - E | C+V+E - V+E |
|---|---|---|---|---|---|---|---|
| P-STOP | 0.517 [0.413, 0.651] | 0.558 [0.341, 0.782] | 0.473 [0.274, 0.711] | 0.519 [0.293, 0.759] | 0.555 [0.310, 0.794] | +0.039 [-0.163, +0.231] | +0.036 [-0.047, +0.146] |
| P-MOVE | 0.708 [0.637, 0.780] | 0.700 [0.636, 0.770] | 0.716 [0.645, 0.791] | 0.748 [0.671, 0.823] | 0.742 [0.666, 0.815] | +0.034 [-0.043, +0.105] | -0.006 [-0.037, +0.024] |

Every eval-141 paired CI covers 0; the P-STOP window is too small to separate anything (E itself reads 0.517); the P-MOVE direction matches TRAIN (C+V+E at or just below V+E).

## What it implies (with probes 1-4)
* **The frozen DINOv3-L/16 caps the go/stay trigger at about V+E.** Probes 1-4 closed the pooled readout, the per-token linear and MLP readouts, the attention and max-pool spatial readouts and the 4x10 and 8x20 grids; probe 5 closes native-resolution crops (3x scale, 21x72 and 21x60 tokens) of the two bands where a lamp or a lead vehicle lives. All five readout families and both crop sources land at or below V+E (P-STOP 0.680, P-MOVE 0.785) and the 200-feature arm does not lift it (-0.023 and -0.003). Ego alone reads 0.626 / 0.732.
* **A trigger lever would not clear D2 even if one worked.** The register row **H-GAPDECOMP-1** (on the tip, `Project Steering/GOALS_AND_CLAIMS.md`, MEASURED 2026-11-02) measured the v8F Read-A gap as NOT TRIGGER-BOUND for DAG-OACC: no crawl row reaches TRIGGER-BOUND (ADE6 and yaw-rate NOT; curvature, L2 and M1 MIXED; the SOUP arm's L2 row is TRIGGER-BOUND at both seeds), and with the trigger windows' gap set to 0 the yaw-rate, curvature, L2 and M1 rows stay separated-worse (ESTIMATED), so a trigger-only fix would not clear D2. This probe's NOT-HELP therefore leaves D2 where that row put it, on the moving windows.
* **What remains is an architecture change and a PI decision:** a fine-tuned or different encoder (the encoder is the part of the pipeline probes 1-5 could not vary), or a dedicated traffic-light-state input. Each is a perception architecture change and GPU work, so each is a PI decision; none is started here.
* **Limits (PREREG 8):** ONE fixed readout (PCA-64 + L2 linear): a NOT-HELP says this readout on crop tokens of THIS frozen encoder at 3x scale does not carry the trigger, not that a learned readout, a fine-tuned encoder or a different encoder would not; the 2x4 grid is coarse; one frame (no motion); C+V+E carries 128 more features than V+E and the paired contrast includes that cost; the JPEG perturbation (Disclosure 4) is unmeasured on the AUC; strata and eval-141 are descriptive and small.

## Provenance
* Package `E:/Projects/TanitAD-artifacts/dagland/trigger_probe5/` (card `PREREG_FROZEN.md` + `FROZEN_SHA.txt`; code `crops_lib.py`, `stage_crops.py`, `encode_crops.py`, `align_check.py`, `run_p5.py`, `tp_lib.py` byte-identical to probe 1's, chains `chain_p5.sh` / `chain_readout.sh`; tests `test_trigger_probe5.py`, 10 passed; `HANDOFF.md`; `MANIFEST.txt` with sha256 and size of every file).
* Results: `raw/VERDICT.json`, `raw/res_pstop_p5.json`, `raw/res_pmove_p5.json`, `raw/results_strata_p5.json`, `raw/results_eval141_p5.json`, `raw/align_check.json`, `raw/align_info.json`, `raw/FEAT_SHA256.txt`, out-of-fold scores `raw/oof_<pop>_p5.npz`, per-fold PCA `raw/pca_<crop>_<pop>.npz`, logs `raw/logs/`.
* Big arrays (single copies on F:, second copy of the features on box 2): `F:/TanitAD-scratch/trigger_probe5_scratch/` (JPEG shards, `out/feat_shard_*.npy` 23 files, `FEAT_all.npy` 24,608 x 2 x 10,240 fp16, `pcacomp_*_all.npz`); box 2 `D:/tanitad/jobs/probe5/out`.
* Encoder: `facebook/dinov3-vitl16-pretrain-lvd1689m`, box-2 snapshot ea8dc286, bf16, weights sha256 `dcb2e451...8179`, RTX 4060, torch 2.11.0+cu128; 23 shards in 14:48-15:30 (GPU 58-62 C, no cool-down pause logged).
* The TRAIN cache behind V is UNVERIFIED byte-for-byte against box 2 beyond the 32-frame known answer (as in probes 1-4). The register row for this probe (H-CRAWL-06-PROBE5) is the Master Mind's to land; nothing was committed or pushed from here.
