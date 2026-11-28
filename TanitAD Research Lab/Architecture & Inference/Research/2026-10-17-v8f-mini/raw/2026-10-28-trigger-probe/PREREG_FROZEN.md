# PREREG_FROZEN -- the trigger-information probe (H-CRAWL-06 precursor, widened)

Zero training of the planner, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`. Everything below is fixed BEFORE any label, feature or outcome statistic is computed.
The card (populations, labels, feature sets, CV, metric, decision rule, controls) is the Master Mind's brief, copied and made exact; the pins that the brief left open are
marked **[PIN]**. Anything not pinned here and decided later is written to RESULTS.md under "post-freeze choices" and cannot move a verdict.

PRE-FREEZE INSPECTIONS (structure only: no label, no outcome, no probe, no image feature was read or computed): the tip's `launch_mask` (v8f_mini.py:401-407),
`--launch-intent-w` (v8f_mini_train.py:128-131), `crawl_diag.py` (train_inputs, stay_mask), `v8f_mini_data.py` (ClipStore, V8FMiniWindows.item, build_label_join),
`v8f_mini.py` (V8FMini.run_o, `_forward`), the keys/shapes of one cached clip (pool4x10 = [T, 40, 1024] fp8; shape only), the key lists and `meta` of the two lead arrays,
the key structure of the v7.2 label record and the per-key clip counts of its TRAFFIC_LIGHT_REACT* goals (376 / 363 / 22 / 18, equal to the figures CLAUDE.md quotes), and the DAG-OACC s0
train directory (argv.json, SHA256SUMS.txt; ckpt_last.pt sha256 = 67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f, the same 8-hex prefix `crawl_diag.CKPT_PREFIX["CTRL"]` pins).

## 1. Windows [PIN]
The canonical TRAIN window list, rebuilt exactly as `dagland/crawl_diag/crawl_diag.py::train_inputs` does (306,566 windows, 4,512 clips, holdout `540:600` excluded;
window-list sha256 `e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168`, REFUSED if the rebuilt digest differs). A window is (clip, t), t = a 0.2 s cache step, first t = 3.
`v0` = `kin3[t, 0]` of `yaw_kin_from_track` (measured speed at raw frame 2t). The holdout `540:600` is never opened.

## 2. Populations and labels [PIN: the GO horizon]
* **GO horizon, pinned:** the SAME horizon as the registered launch mask, `v8f_mini.launch_mask` / `mlaunch.launch_subset` / `--launch-intent-w`'s mask:
  `v6 = |gt[29] - gt[28]| / 0.2` (the GT chord speed over the LAST 0.2 s of the 6 s horizon, knots 29 -> 30), **GO = (v6 >= v0 + 1.0)** (the registered `>=`), gt = `gt_future(pose_grid)[t]` in the ego frame at NOW.
  (The loss weight of `--launch-intent-w` sits on knots 1-12, but its MASK is this end-of-horizon chord; the mask is what defines the 327 windows, so it is the label.)
* **P-STOP:** v0 < 0.5 m/s; label = GO.   **P-MOVE:** 0.5 <= v0 < 6.0 m/s; label = GO (the launch definition with its own `v0 < 6`).
* **Secondary label GO24 (reported, never decisive) [PIN]:** `v2.4 = |gt[11] - gt[10]| / 0.2` (the chord over knots 11 -> 12, the end of the executed 2.4 s horizon), GO24 = (v2.4 >= v0 + 1.0). Computed for E, V, V+E (linear) only.
* GO-rate is reported on the FULL population (all windows of the population), not on the subsample.

## 3. Sampling [PIN]
Per population, per class (GO / not-GO): uniform without replacement of min(count, 6000) windows, `np.random.default_rng(0)`, drawn from the population's windows in canonical order.
AUC is invariant to class-balancing in expectation; GO-rates are always read off the full population. n per population / class / clip is reported.

## 4. Feature sets [PIN]
* **E (primary) = `p10f[t]`, the 8 numbers O's ego token reads:** `fk.kin_features(kin3[t])` with `kin3 = (v0, a0, kappa0)` of `yaw_kin_from_track` (a0 = the ONE-frame backward speed difference, kappa0 from the measured yaw rate; no steering),
  i.e. the exact tensor `V8FMini.run_o` feeds `e_ego(prior.standardise(.))` (v8f_mini.py run_o). The brief's "ego-history inputs the model receives" is, in this model, this one 8-vector: the model reads NO longer speed history at O.
* **E_H (sensitivity, reported, never decisive) = E ++ [v(t-k), omega(t-k)] for k = 1..5** (cache steps, i.e. the previous 1.0 s of measured speed and yaw rate, from the native track, causal: raw frames <= 2t only). It exists so a reader can see whether V adds to a stronger ego baseline.
* **V = the frozen DINOv3 `pool4x10` features of the CURRENT frame:** `torch.load(<clip>.pt)["pool4x10"][t]`, [40, 1024] fp8 -> fp32, flattened to 40,960, centred (no per-feature scaling), **PCA-64 fitted on the outer TRAIN folds only**
  (own streaming randomised SVD: oversample 32, 2 power iterations, seed 0, fp16 memmap read in chunks of <= 500 windows). Cache = the 4,572-clip train cache found at `F:/TanitAD-scratch/speedread/cache/train/clips`
  (UNVERIFIED against the box-2 training cache `D:\tanitad\cache_full` byte for byte: checked by clip count 4,572, key set, and T equal to the native-track length per clip, the same assert `crawl_diag` uses).
* **V+E** = PCA-64(V) ++ E. **E_H+V** is not run.
* **Z+E:** Z = **`h_O` = `V8FMini.run_o`'s `read_o(cond, scene, pad).flatten(1)`, v8f_mini.py:739 (tip 09ebfa3e), shape [B, n_q*d] = [B, 1024]** -- the last representation the plan head `head_o` reads, which already contains the
  ego token, nav, limit, the S checkpoints, T's code, the ages and the 4 scene frames. Taken from one `forward_infer`-path forward of the **DAG-OACC s0 checkpoint** (`E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt`,
  sha256 `67265ec1...8bc45f`, hooked on the `read_o` module output; model built as `v8f_t1_eval.cmd_run` builds it), CPU fp32, `a_t = a_s = 0` (all layers fresh at NOW), batch built by `V8FMiniWindows.item` + `collate` over the same join the run trained on.
  Z is PCA-64 reduced like V (train folds only), then ++ E. **Secondary Z tensors (reported, never decisive):** `h_T` = T's trunk output (v8f_mini.py:723, [B, 256]) and `code` (T's code O reads through `e_code`, [B, CODE_DIM]) and S's `ck` [B, 9].
  **Memorisation caveat committed in advance:** the checkpoint was TRAINED on these TRAIN windows (and their GT), so Z on TRAIN can carry memorised label information that V (frozen, never fitted to labels) cannot: TRAIN Z is biased toward looking BETTER than V. The eval-141 confirmation (unseen) is the counter-reading.
* Standardisation: every probe input is standardised with a scaler fitted on the training rows of the fold.

## 5. Probes, CV, metric [PIN]
* **Linear:** `LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)`, C chosen from {1e-3, 1e-2, 1e-1, 1, 10} by inner 3-fold CV (grouped by clip) AUC on the outer TRAIN folds.
* **MLP:** `MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, max_iter=300, random_state=0)`,
  `alpha` from {1e-2, 1, 10} by the same inner CV.
* **Outer CV:** 5 folds split by EPISODE (clip): `fold(clip) = int(sha12(clip), 16) % 5`. Inner folds: `(int(sha12(clip), 16) // 5) % 3`. PCA is fitted on the outer-train rows once per outer fold (the held-out fold never enters it; the inner selection reuses that PCA -- an unsupervised step, stated).
* **Metric:** pooled out-of-fold AUC (each window scored by the fold model that never saw its clip). **CI = episode-cluster bootstrap, B = 2000, seed 0**, resampling CLIPS with replacement, AUC recomputed with multiplicity weights (tie-aware, Mann-Whitney); the
  paired differences dAUC(V+E - E) and dAUC(Z+E - V+E) use the SAME resample on the SAME windows. It answers "would another draw of EPISODES say this?", not "would another training run say this?" (H-ESTIM-SEED-1); the Z comparison additionally stands on ONE checkpoint.
* **Confirmation, no refit:** the probes are fitted ONCE on all sampled TRAIN windows (C / alpha by the inner CV on all of them, PCA on all of them), then applied unchanged to eval-141's P-STOP / P-MOVE windows (the window set of `crawl_diag.Env.window_set()`,
  141 episodes); AUC with an episode-cluster bootstrap by eval episode. Eval-141 is not used for any choice. The 327 GT-launch windows (all GO by construction: reported as the probe's mean score and the share scored above the TRAIN median-of-negatives threshold) and the 32 GT-stay windows
  (`crawl_diag.stay_mask`: v0 < 0.5 and no 0.2 s chord above v0 + 1): reported as the AUC of launch-vs-stay within v0 < 0.5, and their mean scores.

## 6. Strata (TRAIN, within each population; GO-rate on the full population, AUC on the OOF predictions) [PIN]
* **(a) Traffic light.** From `_s2build-copy-20260919/release/v72/s2_labels_v7.2_train.jsonl.gz`, key `g_tac.goals`: TL_RED / TL_GREEN / TL_YELLOW = the clip carries `TRAFFIC_LIGHT_REACT_RED/_GREEN/_YELLOW` (colour-specific keys only); TL_ANY = any of the four `TRAFFIC_LIGHT_REACT*` keys;
  NO_TL = none. **CLIP-LEVEL and VLM-DERIVED (`provenance: vlm-cot`, `disputed: true`, `time_basis: untimed`)**: a window in a TL clip need not be near the light. It is a coarse stratum, not a ground-truth light state.
* **(b) Lead vehicle.** From `C:/Users/Admin/box2_ship/v9ship/lead_train4572.npz` (the file the DAG-OACC run trained with: `--lead-train ...lead_train4572.npz`), fields `sha12, k, lead, gap` only (the fields `build_label_join` itself reads; `s`, `tg`, `v` are NOT used: not guessed).
  Cache step t <-> row `k = 2t`, exactly `build_label_join`. LEAD_NOW = `lead == 1` with a finite `gap` at k = 2t; NO_LEAD = the clip has lead rows and `lead == 0` at k = 2t; clips with no lead rows are "lead-unknown" and fall in neither.
  **LEAD_DEPART** = LEAD_NOW and `gap[k = 2(t+10)] - gap[k = 2t] >= 2.0 m` (the lead opens >= 2 m within the next 2 s; label-side, it reads the future, so it is a stratum and never a feature).
* A stratum with < 30 windows in either class or < 20 episodes in the SAMPLE is reported as "too small", not computed. No stratum enters the decision rule.

## 7. Decision rule (committed as written; evaluated per population on TRAIN; "AUC(.)" is the pooled OOF AUC of the LINEAR probe unless stated)
* **D1 (information present but unused):** dAUC(V+E - E) >= 0.05 with the paired CI excluding 0, AND AUC(Z+E) >= AUC(V+E) - 0.02. Lever: a loss-only stop/go or trigger head on the trunk (H-CRAWL-06; an architecture change, the PI's).
* **D2 (the trunk discards it):** dAUC(V+E - E) >= 0.05 with the CI excluding 0, but AUC(Z+E) < AUC(V+E) - 0.02. Lever: a representation or auxiliary loss upstream of O.
* **D3 (the frozen features do not carry it):** dAUC(V+E - E) < 0.02 for BOTH the linear probe AND the MLP. Lever: perception (encoder resolution or pooling), the PI's.
* **Otherwise INCONCLUSIVE**, with the measured numbers. [PIN] D1/D2's "dAUC >= 0.05" is read on the LINEAR probe. If the MLP alone clears 0.05 with the CI excluding 0 while the linear probe does not, the verdict is INCONCLUSIVE-NONLINEAR (information present non-linearly) and is reported as such, not promoted.
  The MLP reading of the Z comparison is reported beside the linear one. E_H and GO24 can only add caveats; they cannot change a verdict.
* **[PIN] Direction of the TRAIN memorisation bias:** a D1 on TRAIN whose eval-141 AUC(Z+E) < AUC(V+E) - 0.02 is reported "D1 (TRAIN), eval-141 does not confirm: memorisation-suspect", not as a clean D1. A D2 on TRAIN is not softened by that caveat (the bias favours Z).

## 8. Controls (the pipeline must see a signal and refuse to invent one)
* **C1 label shuffle:** labels permuted across the population's sampled windows (seed 1), the full pipeline (PCA, C, CV) on V+E and on E: AUC must be 0.5 with a CI covering 0.5.
* **C2 episode-shuffled E:** E fitted with labels assigned from a DONOR episode (each clip's labels replaced by labels drawn from another clip, seed 2): AUC must not exceed the true E AUC and its CI must cover 0.5 (a pipeline that leaks clips across folds would fail it).
* **C3 deliberate leak:** the GT label (0/1) appended as a feature to E and to V+E: AUC must be >= 0.99.
* **C4 unit tests** (`test_trigger_probe.py`): the weighted AUC equals sklearn's `roc_auc_score` at unit weights; the streaming PCA matches a dense PCA's subspace on a small matrix; the launch label equals `mlaunch.launch_subset_loop` on random inputs; the fold function never splits a clip; a planted signal gives AUC > 0.9 and a null gives 0.5 +- CI.
A control that fails voids the arm it guards: the number is reported as FAILED-CONTROL and no verdict is drawn from it.

## 9. Resource rules
RAM free >= 3 GB before every heavy load (PowerShell `Win32_OperatingSystem.FreePhysicalMemory`); memmaps, chunks of <= 500 windows, RSS < 2 GB; `torch.set_num_threads(4)`; `CUDA_VISIBLE_DEVICES=-1`; bare numpy/sklearn from `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`.
Never touched: `dagland/bank_presteps/`, `Keys.txt`, the holdout `540:600`, any pod/box 2/GPU. Nothing is deleted; scratch lives under `trigger_probe/scratch/`. If RAM is short or a job would outlast 4 min of waiting, write `HANDOFF.md` and end.
