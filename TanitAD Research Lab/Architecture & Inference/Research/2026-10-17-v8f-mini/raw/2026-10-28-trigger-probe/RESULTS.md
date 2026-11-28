# RESULTS -- where is the go/stay trigger information lost?  (H-CRAWL-06 precursor, widened)

Zero training, dev-box CPU. Card frozen first: `PREREG_FROZEN.md` (sha256 0eb1c25c...ff97, 2026-10-28T22:10:08+0100, `FROZEN_SHA.txt`).
**Evidence class: MEASURED** (artifacts in `raw/`). **Tier:** T0-class probe of representations on logged inputs, not driving performance.
**Estimator:** episode(clip)-cluster bootstrap, B = 2000, seed 0: it answers "another draw of EPISODES", is blind to training variance (H-ESTIM-SEED-1), and the Z arm rests on ONE checkpoint (DAG-OACC s0, sha256 67265ec1...8bc45f).
**Samples:** TRAIN, 6000 GO + 6000 not-GO per population (P-STOP 849 clips, P-MOVE 2,140 clips) from the canonical 306,566-window list (digest e144adcd...8168 re-verified). Full-population GO rate: **P-STOP 57.4 %** (10,948 / 19,070 windows, 859 clips), **P-MOVE 53.8 %** (38,670 / 71,930).
GO = the registered launch mask (`v6 >= v0 + 1`, chord over the last 0.2 s of the 6 s horizon).

## 1. Verdict by the committed rule (TRAIN, linear probe; dAUC = AUC(V+E) - AUC(E), paired, same windows)

| population | dAUC(V+E - E) linear, CI | AUC V+E | AUC Z+E | rule |
|---|---|---|---|---|
| **P-STOP** | **+0.054 [0.025, 0.082]** (per-fold-mean metric); +0.087 [0.050, 0.125] (frozen pooled metric) | 0.680 | 0.800 | **D1** on the per-fold-mean reading. The frozen pooled reading is **FORMALLY VOID** (control C2 missed "CI covers 0.5" by 7e-5, section 3); it reads D1 as well. |
| **P-MOVE** | **+0.053 [0.041, 0.065]** (both metrics) | 0.785 | 0.881 | **D1** (all controls pass). MLP reading: dAUC +0.015 [0.006, 0.024]. |

The rule's letter is met in both populations (dAUC >= 0.05, CI excludes 0; AUC(Z+E) >> AUC(V+E) - 0.02). **Read it with the caveats of section 4: it is a thin D1, not a clean one.**

## 2. All arms (AUC [95 % CI]; pooled out-of-fold, as frozen; per-fold-mean in `raw/results_p*.json`)

| P-STOP (n = 12,000, 849 clips) | linear | MLP | | P-MOVE (n = 12,000, 2,140 clips) | linear | MLP |
|---|---|---|---|---|---|---|
| E (8 numbers O reads) | 0.593 [0.570, 0.616] | 0.595 [0.573, 0.617] | | E | 0.731 [0.718, 0.745] | **0.770** [0.758, 0.783] |
| E_H (+1.0 s speed / yaw-rate history) | 0.601 [0.578, 0.624] | | | E_H | 0.737 [0.724, 0.749] | |
| V (PCA-64 of pool4x10, current frame) | 0.631 [0.597, 0.667] | 0.626 [0.592, 0.661] | | V | 0.684 [0.667, 0.702] | 0.676 [0.659, 0.692] |
| V+E | 0.680 [0.651, 0.711] | 0.681 [0.652, 0.712] | | V+E | 0.784 [0.771, 0.798] | 0.786 [0.773, 0.799] |
| Z (h_O, PCA-64) | 0.801 [0.779, 0.825] | 0.792 [0.768, 0.816] | | Z | 0.877 [0.867, 0.886] | 0.874 [0.865, 0.884] |
| Z+E | 0.801 [0.779, 0.824] | 0.799 [0.776, 0.822] | | Z+E | 0.880 [0.871, 0.890] | 0.873 [0.863, 0.883] |
| dAUC(Z+E - V+E) | +0.121 [0.093, 0.146] | +0.118 [0.091, 0.144] | | dAUC(Z+E - V+E) | +0.096 [0.085, 0.107] | +0.087 [0.077, 0.097] |
| dAUC(V+E - E) | +0.087 [0.050, 0.125] | +0.086 [0.050, 0.125] | | dAUC(V+E - E) | +0.053 [0.041, 0.066] | +0.016 [0.007, 0.025] |
| dAUC(V+E - E_H) | +0.079 [0.042, 0.117] (fold-mean +0.039 [0.008, 0.068]) | | | dAUC(V+E - E_H) | +0.047 [0.036, 0.060] | |

Secondary label GO24 (the chord at the end of the executed 2.4 s horizon; caveat only), linear: P-STOP E 0.717, V+E 0.753 (dAUC +0.036 pooled / +0.018 fold-mean [0.002, 0.034]); P-MOVE E 0.882, V+E 0.897 (dAUC +0.014 [0.010, 0.019]). Z+E: 0.818 / 0.927.

## 3. Controls (`raw/res_*_V_linear.json`, `controls` in `results_p*.json`; pytest 9 passed, `raw/pytest_output.txt`)
* **C1 label shuffle:** V+E 0.4999 [0.490, 0.509], E 0.4949 [0.485, 0.505] (P-STOP); 0.4934 / 0.4964 (P-MOVE). All CIs cover 0.5. PASS.
* **C3 leak (GT label appended):** AUC 1.0000 for E and V+E in both populations. PASS: the pipeline can see a signal.
* **C2 episode-shuffled E (donor-episode labels):** P-MOVE 0.5023 [0.485, 0.518] PASS. **P-STOP 0.4629 [0.4267, 0.49993] -- FAILED the committed "CI covers 0.5" by 7e-5.** It is below chance, not above: the pipeline cannot manufacture signal. Diagnosis (measured by the unit test `test_foldmean_metric_unbiased_on_prevalence_shifted_null`):
  the pooled out-of-fold AUC is biased BELOW 0.5 on clip-clustered labels, because each fold's model carries an intercept fitted to the other folds' prevalence. It also deflates the 8-number E more than V+E (E pooled 0.593 vs per-fold mean 0.626), so **the frozen pooled metric inflates dAUC(V+E - E) on P-STOP (+0.087 vs +0.054)**.
  **POST-FREEZE correction, clearly marked:** the mean of per-fold AUCs (same bootstrap). C2 under it: 0.4831 [0.465, 0.502] PASS; all other controls unchanged. Under the frozen text the P-STOP pooled reading is void; the verdict above rests on the per-fold-mean reading, and both give D1.

## 4. What the D1 does and does not say (all caveats are measured; none was allowed to change the verdict)
1. **The image increment is small.** P-STOP: V alone 0.631, V+E 0.680; V-alone minus E is +0.007 [-0.030, 0.041] (fold-mean): the frozen current-frame pool4x10 features do not out-predict the 8 ego numbers. Against the stronger ego baseline E_H the P-STOP gain is +0.039 [0.008, 0.068] (< 0.05).
   P-MOVE: the linear gain (+0.053) is largely a **nonlinearity of E**, which the linear probe cannot use: with the MLP the gain is +0.016 [0.006, 0.025] (< 0.02).
2. **Eval-141 (unseen by the probes and by their choices, no refit) does not confirm the P-STOP increment, but it cannot refute it either.** P-STOP: n = 101 windows, 19 episodes, 61 GO: E 0.517 [0.41, 0.65], V+E 0.519 [0.29, 0.76], Z+E 0.655 [0.45, 0.85]; dAUC(V+E - E) = +0.002 [-0.17, 0.18]. P-MOVE: n = 507, 67 episodes, 266 GO: E 0.708, V+E 0.748, Z+E 0.804; dAUC(V+E - E) = +0.039 [-0.026, 0.103]; Z+E - V+E = +0.057 [-0.022, 0.145].
   The committed memorisation test (eval AUC(Z+E) < AUC(V+E) - 0.02) is NOT triggered in either population. On the GT-launch / GT-stay subsets (P-STOP, linear Z+E): launch-vs-stay AUC 0.718 (E 0.576, V+E 0.567); mean score percentile of launch windows 0.60 vs stay windows 0.37 (E: 0.43 / 0.30). 61 launch + 266 launch (=327) and 32 GT-stay windows, as in the brief.
3. **Z is not "V plus a better trunk".** The trunk also reads (a) the **nav token** (derived from the ego's own future path: v7.2 provenance "ORACLE (ego-future)") and the speed limit, and (b) **4 frames** (t-3..t), where V is the current frame only.
   POST-FREEZE diagnostic (`nav_arm.py`, `raw/results_nav_postfreeze.json`; caveat-only): nav + limit + E alone (N+E) reaches **0.719 pooled / 0.727 fold-mean on P-STOP (E: 0.593 / 0.626)**; V+N+E 0.741; so most of the go/stay predictability of Z comes from the oracle nav, not from the image.
   Z+E minus V+N+E = +0.060 [0.040, 0.081] on TRAIN (P-STOP) and +0.058 [0.050, 0.067] (P-MOVE) -- TRAIN windows the checkpoint was trained on (memorisation-suspect) -- but on **eval P-MOVE Z+E 0.804 vs V+N+E 0.800 (+0.004)**: on unseen windows the trunk adds nothing beyond its own non-image inputs plus the image features. Eval P-STOP is unresolved (Z+E 0.655, V+N+E 0.600, n = 101).
   The 3 extra frames (motion cues: a lead vehicle starting to move) are NOT separated here: UNVERIFIED whether they carry the remaining +0.06.
4. **Strata (TRAIN, linear OOF, fold-mean; `raw/results_strata.json`; the TL strata are CLIP-level and VLM-derived (`vlm-cot`, disputed, untimed), so they are coarse; none enters the rule).** GO-rate, full population:
   P-STOP: TL_RED clips 0.445 (7,655 windows), TL_GREEN 0.886 (2,425), TL_YELLOW 0.305 (243, AUC too small), NO_TL 0.607; lead present 0.590, no lead 0.575, **lead departs >= 2 m within 2 s 0.817** (3,008).
   P-STOP dAUC(V+E - E): **TL_RED +0.099** (E 0.609 -> V+E 0.708; Z+E 0.806), TL_GREEN **-0.058**, NO_TL +0.061, lead-now +0.044, no-lead +0.064, **lead-departs -0.019** (E 0.663, V+E 0.644). P-MOVE: dAUC about 0 inside TL clips (+0.002 to +0.006), +0.068 outside.
   So the frozen features carry something about the stopped-at-a-RED-clip windows and nothing measurable about a lead vehicle departing or a green light.

## 5. Verdict and what it implies for the next v8F lever
* **By the committed rule: D1 in both populations** (the trunk keeps what V+E carries: Z+E >= V+E - 0.02 by a wide margin, TRAIN and eval-141). The lever the rule names is a **loss-only stop/go (trigger) head on the trunk (H-CRAWL-06; architecture change, the PI's)**.
* **But the ceiling is low and the evidence thin:** the image-derived increment over ego is +0.05 AUC on TRAIN (P-STOP +0.054 borderline; P-MOVE +0.016 with a nonlinear ego baseline), +0.002 / +0.039 on eval-141 (wide CIs), and the largest single source of GO predictability in Z is the oracle nav. A trigger head fed by this trunk is expected to read about the nav+ego+image ceiling (TRAIN 0.80, eval P-MOVE 0.80, eval P-STOP 0.65 with CI +-0.2), not a sharp stop/go perception.
* **Cheapest next experiments (largest expected effect first):** (i) a V4 arm over the model's own 4 frames (or frame differences) to see whether motion cues, not the current frame, carry the trigger -- zero-training, same pipeline; (ii) the TL_RED / lead-departs strata point at where perception is missing (green-light and lead-departure windows show no V gain), so a perception-side lever (encoder resolution / temporal pooling) is the live alternative to D1's head; (iii) before any head is trained, decide with the PI whether GO-skill measured through nav counts, since nav is an oracle on this corpus.

## 6. Post-freeze choices (cannot move a verdict)
1. Per-fold-mean AUC metric (section 3). 2. E_H history for a window with t < 7: the missing steps are held at row 2 (a first version let a negative index wrap and left NaN omega; fixed and the sample verified identical, `scratch/*_v1*`). 3. `nav_arm.py` diagnostic. 4. `strata.py` thresholds (>= 30 windows per class, >= 20 episodes).
5. **RAM guard:** the brief asked free RAM >= 3 GB before heavy loads. Free RAM read 2,318-2,701 MB when the V extraction (peak working set 848 MB, mmap one clip at a time) was launched and 3.2-5.9 GB for the PCA / probe / Z stages. The < 2 GB RSS rule held throughout; the 3 GB reading was not met at that one start.
6. A duplicate `extract_v.py` launch hit the file lock and exited at once; the first process completed (2,185 clips, 317 s) and wrote every row (no all-zero row).

## 7. Provenance and what is UNVERIFIED
* Cache: `F:/TanitAD-scratch/speedread/cache/train/clips` (4,572 clips). Checked for the 2,185 clips read: `pose_grid` equals the native track's even frames (max abs diff 0.0) and T equals the manifest's n_frames. **UNVERIFIED byte-for-byte against box 2's `D:\tanitad\cache_full`.**
* The TRAIN join was REBUILT from md5-verified inputs (v9 release 83e30bd3...; lead_train4572 d4336a44...; speed sidecar bab41adf...), `scratch/join_train_rebuilt.npz` sha256 978f41bf...; **not compared with box 2's `join_train.npz` (UNVERIFIED)**.
* Z control: `head_o(h_O)` + prior reproduces the emitted `traj60` with max abs difference 0.0 (the hooked tensor IS what head_o reads); median TRAIN ADE over batches 1.33 m.
* Eval-141 window set from `crawl_diag.Env.window_set()` (2,399 windows, 141 episodes; v0 < 6: 608; launch 327; GT-stay 32), launch recount equals `mlaunch.launch_subset`.
* Inner-CV hyper-parameter choice is unstable on V+E (C hopping between 1e-3 and 10 across folds): the AUC surface is flat there; reported, not tuned.
