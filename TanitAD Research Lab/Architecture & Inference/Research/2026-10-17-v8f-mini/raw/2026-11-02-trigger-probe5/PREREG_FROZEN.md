# PREREG_FROZEN -- trigger probe 5: do NATIVE-SCALE crops of the traffic-light band and the lead-vehicle band carry the go/stay trigger that the pooled frozen DINOv3 cache does not?

Zero training of the planner. Written and hashed BEFORE any crop feature, label statistic or probe output of this package (`FROZEN_SHA.txt`: sha256 + wall clock). Pre-freeze actions, structure only: read SCOPE.md, probe 1/3 code, `windows_train.npz` / `eval141_features.npz` array shapes, the
intrinsics CSV (native focal `fw_poly_1`, cy), mp4 presence for the 2,185 + 67 clips, box-2 state (no python running, GPU idle, D: 54 GB free).  Evidence class MEASURED; tier T0-class probe of representations on logged inputs, NOT driving performance.

## 1. Held fixed (REUSED, asserted)
Windows, labels, E, folds, bootstrap = probe 1: `dagland/trigger_probe/raw/windows_train.npz` (sha256 a9a57b32...5783ec asserted), `sample_pstop.npz`, `sample_pmove.npz` (12,000 + 12,000 windows, 24,000 unique (clip,t), 2,185 clips), `tp_lib.py` byte-identical (sha256 asserted).
Folds `outer = int(sha12,16)%5`, `inner = (int(sha12,16)//5)%3`; bootstrap = episode(clip)-cluster, B=2000, seed 0 (answers "another draw of EPISODES"; blind to training variance H-ESTIM-SEED-1; the encoder is one fixed frozen network).
Frame = the CURRENT frame of the window = cache grid row `t` = v2ep frame `2t` = raw mp4 frame `frame_idx[2t]` (`v2_compressed._resampled` recipe: `searchsorted` of 10 Hz query times on the timestamps parquet). Eval-141: the 608 windows of `eval141_features.npz` (67 clips; row `t0`), no refit.

## 2. Crops [PIN]
The cache saw the 120-degree f-theta front-wide frame rectified to a 256x640 CYLINDRICAL frame (hfov 120, f_ref 305.5775, per-clip principal point, unobserved pixels black).  NATIVE SCALE = the sensor's own scale: measured median native focal `fw_poly_1` = **928.08 px/rad** (min 913.7, max 940.9, 2,185 clips),
so the native-scale cylindrical frame is **768x1920, f_ref = 916.7324 = 3.0 x 305.5775 (hfov 120), same projection and per-clip principal point, no downscale** (1.2 % below the median native focal: nominal native).  Crops are cut from THIS frame (grid sliced before `grid_sample`: pixel-identical to slicing the full rectification, pytest):
* **TL band**: rows 0-45 %, cols 20-80 %  ->  rows [0,336), cols [384,1536): 336x1152 = 21x72 = 1,512 tokens.
* **LEAD band**: rows 35-80 %, cols 25-75 %  ->  rows [272,608), cols [480,1440): 336x960 = 21x60 = 1,260 tokens.   (dims cut to multiples of 16: 336 = 21x16.)
Transport dev box -> box 2: JPEG q95, 4:4:4 (PNG would be ~50 GB; box 2 has 54 GB).  The JPEG effect on the features is MEASURED on the 32 alignment frames (lossless PNG crop vs JPEG crop through the same encoder; reported, not gated).
Encoder: `facebook/dinov3-vitl16-pretrain-lvd1689m` (box-2 snapshot ea8dc286..., weights sha256 dcb2e451...8179 asserted), bf16, `(x/255 - mean)/std` of its processor, CLS = token 0, patches = tokens 5:, batch 2, resumable per shard.
Per crop keep **CLS (1024) + mean patch (1024) + 2x4 adaptive-average-pooled patch grid (8x1024)** = 10,240 d (fp32 pooling, stored fp16), row-major grid.

## 3. Known answer (gate; before the main encode)
32 FULL frames (24 train windows seed 11 + 8 eval windows), rectified at the cache's own geometry 256x640 (lossless PNG transport), through the SAME box-2 encoder entry, 4x4 fp32 average -> `pool4x10`, vs the cached row (`F:/TanitAD-scratch/speedread/cache/train/clips` / `sfresh_inputs/cache/eval/clips`, fp8->fp32): per-row relative L2.
**GATE: median <= 0.05; any row > 0.5 REFUSES the package** (wrong frame or token order reads ~1.0; probe 3 measured 3.0 % for the 8x20 path).  A refused gate stops the run: no crop feature is read.  Also pytest: crop = slice of the full rectification (bit-exact), and avg-pool-by-3 of the native-scale full frame vs the 256x640 frame is close in pixels (geometry consistency).

## 4. Readout [PIN] -- probe 1's whole-vector pipeline, applied per source
Per source (TL, LEAD) the flattened 10,240-d vector -> **PCA-64** (`tp_lib.StreamPCA`, seed 0, centred, not scaled), fitted per OUTER fold on the outer-train rows only (+1 on all rows for eval-141); V = probe 1's banked PCA-64 (`raw/pca_V_<pop>.npz`).
Standardise on train rows, **L2 logistic, C in {1e-3,1e-2,1e-1,1,10} by inner 3-fold (clip-grouped) mean AUC**, refit (`tp_lib.fit_probe`, unchanged).  Arms: **E** (banked OOF), **V+E** (banked OOF), **TL+E, LEAD+E, C+V+E** (TL64+LEAD64+V64+E = 200 features); beside: C+E (TL64+LEAD64+E).
**Metric: mean of per-fold AUCs** (`tp_lib.auc_ci_fold`), pooled AUC beside; paired dAUC on the same windows with the same resamples (`dauc_ci_fold`).

## 5. DECISION RULE (committed; TRAIN; dAUC = fold-mean AUC(C+V+E) - AUC(V+E), paired, CI = 2.5/97.5 percentiles)
* **CROPS HELP** iff dAUC >= 0.03 with CI lower > 0 in **either** population (P-STOP, P-MOVE).
* **CROPS DO NOT HELP** iff dAUC < 0.01 in **both** populations.
* **Otherwise INCONCLUSIVE** (both numbers reported).
* A cell whose controls FAIL (section 7) is INVALID: it cannot support HELP; NOT-HELP needs both cells valid.  A missing cell makes the verdict INCOMPLETE, never NOT-HELP.  Nothing beside (strata, TL+E, LEAD+E, eval-141) changes the verdict.

## 6. BESIDE, never decisive
Strata as probe 3's `strata_g.py` (>= 60 windows, >= 30 per class, >= 20 clips; TL_RED, TL_GREEN, TL_YELLOW, NO_TL, lead_now, no_lead, lead_depart): AUC of E, V+E, TL+E, LEAD+E, C+V+E and paired dAUC(arm - E), dAUC(C+V+E - V+E); attention on **TL_RED, TL_GREEN, lead_depart**.  Eval-141 no-refit: fit once on all sampled TRAIN windows (all-rows PCA, C by inner CV), apply unchanged; AUC of E, TL+E, LEAD+E, C+V+E and paired C+V+E - E, episode bootstrap by eval clip.

## 7. CONTROLS (run for EVERY source name; the probe-1 `if src == "V"` trap is closed and pytested)
C1 shuffled labels (seed 1): fold-mean CI of TL+E, LEAD+E, C+V+E, V+E covers 0.5, AND the paired null dAUC(C+V+E - V+E) CI covers 0.  C2 donor-episode labels (`donor_labels`, seed 2): fold-mean CI covers 0.5; if seed 2 misses for an arm, seeds 3-6 are run: not invalidated iff the 5-draw mean fold-mean AUC is in [0.47, 0.53], else INVALID (probe 3's band).
C3 deliberate leak (GT label appended): AUC >= 0.99 for TL+E, LEAD+E, C+V+E.  Known answer: section 3.

## 8. Resources and caveats
Dev box: CPU only, `CUDA_VISIBLE_DEVICES=-1`, <= 4 threads, `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, big arrays on `F:/TanitAD-scratch/trigger_probe5_scratch/`, nothing deleted, no git add/commit/push, no `Keys.txt`.  Box 2 (PI's machine): stay inside `D:\tanitad\`, ONE job, `launch.ps1`, SAFE MODE (`%RUNSAFE%`), batch 2, GPU-temperature cool-down pause (>= 78 C), resume-retry on native crash codes.
Caveats: ONE fixed readout (PCA-64 + linear): a NOT-HELP says this readout on crop tokens of THIS frozen encoder at 3x scale does not carry the trigger, not that a learned readout or a different encoder would not; the 2x4 grid is coarse; the crops see ONE frame (no motion); C+V+E has 128 more features than V+E, the paired contrast includes that cost; TL strata are clip-level VLM labels and lead_depart reads the future (strata only); eval-141 is small (P-STOP n=101).
