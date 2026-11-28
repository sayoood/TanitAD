# PREREG_FROZEN -- trigger probe 2: what the nav encodes, the trunk without the oracle nav, and motion cues (H-CRAWL-06 precursor, continued)

Zero training of the planner, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`, `torch.set_num_threads(4)`. Everything below is fixed BEFORE any label, feature, AUC or outcome statistic of this package is computed.
It continues the finished probe `E:/Projects/TanitAD-artifacts/dagland/trigger_probe/` (tip package `raw/2026-10-28-trigger-probe/`) and reuses its pipeline. Pins the brief left open are marked **[PIN]**.
Anything not pinned here and decided later is written to RESULTS.md under "post-freeze choices" and cannot move a verdict.

PRE-FREEZE INSPECTIONS (structure and code only; no label, outcome, AUC or image feature of this package was read or computed): the previous probe's PREREG_FROZEN.md / RESULTS.md / scripts (tp_lib, run_probe, extract_v, extract_z, eval_confirm, nav_arm, strata, build_windows);
the tip (`c1c30fb790a75469fc23f8ede23c251225040b51`) copies of `v8f_mini.py`, `v8f_mini_data.py`, `v9_labels.py`, `build_v9_labels.py` (extracted by `git show` into `scratch/tip_*.py`); the checkpoint's `cfg` keys (hist = 4, hmax = 16, n_q = 4, d = 256) and argv;
the key lists / shapes of the previous probe's npz files; free disk (E: 17 GB, F: 162 GB) and free RAM (5.4 GB). The exported tree used for EXECUTION is `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a` (as in the previous probe); its `v8f_mini.py` / `v8f_mini_data.py` differ from the tip only in training-side code
(`launch_intent_w`, `launch_mask`, the ego-perturb stream); the forward path and the join are identical. The control N1 (section 8) re-tests this empirically: the unmodified path must reproduce the banked Z bit for bit.

## 1. Windows, labels, folds, bootstrap [PIN: all identical to the previous probe]
* Same canonical TRAIN window list (306,566 windows, 4,512 clips, holdout `540:600` excluded; digest `e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168`), read from the previous package's `raw/windows_train.npz` (read-only, sha256 recorded in `raw/INPUTS_SHA256.json`).
* Same populations and labels: **P-STOP** v0 < 0.5, **P-MOVE** 0.5 <= v0 < 6; GO = the registered launch mask (`v6 >= v0 + 1.0`); same sample: `raw/sample_{pstop,pmove}.npz` of the previous package (6,000 GO + 6,000 not-GO per population, 12,000 windows each; P-STOP 849 clips, P-MOVE 2,140 clips). Every arm of this package is paired with the old arms on THESE rows in THIS order.
* Same folds: outer `fold(clip) = int(sha12(clip), 16) % 5`, inner `(int(sha12, 16) // 5) % 3`; same bootstrap: clip-cluster, B = 2,000, seed 0 (`tp_lib.boot_counts` on the same clip array, so the resamples are identical to the old ones).
* `tp_lib.py` is copied byte-identical from the previous package (checked with `cmp`); the old banked out-of-fold scores (`oof_*`) are the E / V+E / Z+E comparators (read-only).

## 2. Metric [PIN]
**Primary = the MEAN OF THE PER-FOLD AUCs over the 5 outer folds** (`tp_lib.auc_ci_fold`, `dauc_ci_fold`; CI = percentile 95 % of the clip-cluster bootstrap on that metric, paired differences on the same resample). The pooled out-of-fold AUC (`auc_ci`, `dauc_ci`) is reported beside, labelled "pooled".
The previous probe showed the pooled out-of-fold AUC is biased below 0.5 on clip-clustered labels (intercept fitted to the other folds' prevalence) and that it deflates the 8-number E more than image arms; the fold-mean is the unbiased reading. The banked old arms are re-scored on the fold-mean metric from their banked OOF scores (no refit).
Estimator named: the clip-cluster bootstrap answers "would another draw of EPISODES say this?"; it is blind to training variance (H-ESTIM-SEED-1) and the Z arms rest on ONE checkpoint (DAG-OACC s0, sha256 `67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f`).

## 3. Probes [PIN: identical to the previous probe]
Linear: `LogisticRegression(l2, lbfgs, max_iter 2000)`, C from {1e-3, 1e-2, 1e-1, 1, 10} by inner 3-fold grouped CV; MLP: `MLPClassifier((64,), relu, adam, lr 1e-3, early_stopping, val 0.15, n_iter_no_change 10, max_iter 300, random_state 0)`, alpha from {1e-2, 1, 10}.
**The LINEAR probe is the decisive reading; the MLP is reported beside it.** All inputs standardised on the training rows of the fold. PCA-64 = `tp_lib.StreamPCA` (centred, unscaled, randomised SVD, oversample 32, 2 power iterations, seed 0), fitted per outer fold on its training rows (+ one fit on all 12,000 rows for the eval-141 confirmation).

## 4. ARM 1 -- NAV: what the nav token encodes [PIN]
* **Code (to be quoted with tip file:line in RESULTS.md):** `v8f_mini.py` run_o / run_s / run_t (`e_nav = Linear(5, d)` on `_at(b["nav"], i)`, NOW only), `v8f_mini_data.py::build_label_join` (nav5 = FOLLOW/TURN_L/TURN_R one-hot ++ d_next/100 clipped [0,3] ++ args_valid; all-zero when the clip has no v9 row at that step) and `item` (nav of the hmax window frames),
  `build_v9_labels.py::turn_table / nav_fields` (turns from the WHOLE recording's sustained-yaw segments; `nav_token` = the next announced turn if inside `d_ann = max(30 m, 6 s * v_now)` of arc; the arc `s` is the ego's own future path), `policies.nav = "allow_oracle_nav (ego-future)"`.
* **Measurements (TRAIN):** nav5 at NOW of every sampled window = `J["nav"][off[clip] + t]` of the rebuilt join (`scratch` of the previous package, md5-verified inputs). Arms, each alone and +E, linear AND MLP, fold-mean primary:
  **N5** = nav5 (primary N: the token as the model reads it); **NL** = nav5 ++ limit4 ++ limit_ok (the previous probe's "N"; reproduces its N+E); **L** = limit4 ++ limit_ok alone. Decomposition (linear only, alone and +E, never decisive): **NV** = args_valid; **NT** = the 3 token one-hots; **ND** = (d_next/100, args_valid).
  Descriptive tables on the FULL populations (all windows of P-STOP / P-MOVE, not the sample): GO rate by args_valid, by token, by d_next bin {<=5, 5-15, 15-30, >30 m, none}; share of GO windows with a turn ahead; mean d_next in GO vs not-GO.
  Paired: dAUC(N5+E - E), dAUC(NL+E - E), dAUC(Z+E - N5+E), dAUC(Z+E - NL+E).
* No decision rule is attached to ARM 1: it is the description of what the nav carries. It sets the reading of ARM 2.

## 5. ARM 2 -- Z_nn: the trunk without the oracle nav [PIN]
* **Null-nav path in the model: none exists** (the pre-freeze grep over `v8f_mini.py`, `v8f_mini_data.py`, `v8f_mini_train.py`, `v8f_t1_eval.py` for nav_drop / null / mask / zero finds no nav dropout; `e_nav` is applied unconditionally, v8f_mini.py:650, 708, 719, 735). The brief's fallback applies:
  **PRIMARY neutralisation = DONOR NAV.** For each target window (clip c), draw a donor window uniformly from the canonical 306,566 TRAIN windows, with a different clip than c, `np.random.default_rng(0)`, drawn in the order of the extraction items (sorted by (clip, t, population)), re-drawing collisions with the same generator.
  The donor window's whole nav history `[hmax = 16, 5]` (`J["nav"]` rows of the donor's idc frames, exactly as `item` builds it) replaces `batch["nav"]`; the window's own feats / p10f / act / pose / limit / limit_ok / a_t = a_s = 0 are untouched. The donor nav is independent of the window, so it carries no information about the window's label by construction (a permutation of nav across windows that keeps its marginal).
  Windows whose donor nav equals their own nav vector are counted and reported (expected to be many: FOLLOW/no-turn is the modal nav); they are not removed.
* **Z_nn = h_O** (`read_o(...).flatten(1)`, [1024]) from the same hooked forward as extract_z.py, same checkpoint (sha256 above), CPU fp32, PCA-64 on the train folds, then ++ E. Arms: Z_nn, Z_nn+E (linear + MLP for Z_nn+E).
  Paired (fold-mean primary, pooled beside): **dAUC(Z_nn+E - E)**, **dAUC(Z+E - Z_nn+E)** (Z+E from the previous probe, same windows), and as secondary dAUC(Z_nn+E - V+E).
* **Optional sensitivities (pre-registered now; reported if run; never decisive):** S-ZERO nav := 0-vector (the join's own "no v9 row" encoding), S-FOLLOW nav := (1,0,0,0,0) (modal "no turn ahead"). Their absence is stated, not silent.
* **Caveats committed in advance:** (i) the checkpoint was TRAINED on these TRAIN windows and GT, so TRAIN Z_nn may carry memorised label information that no frozen feature can; the eval-141 reading is the counter-reading. (ii) The trunk was trained only with a nav consistent with scene and ego; a donor nav makes the input inconsistent, so Z_nn can UNDER-state what a nav-free trunk would hold (a lower bound on vision-through-trunk, an upper bound on nav-only claims). (iii) Z_nn still contains the 4 frames (motion).

## 6. ARM 3 -- V4: motion cues [PIN]
* Features from the model's own 4 frames (cache steps t-3..t, hist = 4) of the cached `pool4x10` DINOv3 features of the SAME cache the previous probe used (`F:/TanitAD-scratch/speedread/cache/train/clips`, 4,572 clips), fp8 -> fp16 (exact):
  **V4** = the 4 frames concatenated (4 x 40,960 = 163,840 dims), PCA-64 on the train folds; **Vdiff** = V_t - V_{t-3} (40,960 dims), PCA-64 on the train folds. Arms V4+E, Vdiff+E (linear + MLP). The current-frame V comes from the previous probe's stored `V_{pop}.npy` (verified equal to the new frame t read, control V1).
* Paired (fold-mean primary, pooled beside): dAUC(V4+E - V+E), dAUC(Vdiff+E - V+E), dAUC(V4+E - E), dAUC(Vdiff+E - E). Windows have t >= 3, so all 4 frames are real.

## 7. Decision rules (committed as written; per population on TRAIN; LINEAR fold-mean unless stated; "CI excludes 0" = lower bound > 0)
* **Z_nn:** dAUC(Z_nn+E - E) >= 0.05 with CI excluding 0 => **VISION-THROUGH-TRUNK** (the trunk holds go/stay information that is not the nav's and O does not use it; lever = a loss-only stop/go head or the output representation).
  dAUC(Z_nn+E - E) < 0.02 => **NAV-ONLY** (the trunk's go/stay information is the oracle nav's; the vision path does not carry the trigger; lever = perception). Anything between, or >= 0.05 with a CI reaching 0 => **INCONCLUSIVE**.
* **V4:** best = whichever of V4+E / Vdiff+E has the higher TRAIN linear fold-mean AUC in that population. dAUC(best+E - V+E) >= 0.03 with CI excluding 0 => **MOTION carries part of the trigger**. dAUC(best+E - V+E) < 0.01 => **it does not**. Between => **INCONCLUSIVE**.
  Choosing the best of two is a selection (winner's curse, not covered by the paired CI); both arms' deltas are reported so a reader can see it.
* The headline is per population (P-STOP, P-MOVE); if they disagree the headline is "SPLIT" with both stated. A population is not pooled with the other.
* **Flags (reported, cannot change a verdict):** F1 (nonlinear ego baseline): a linear VISION-THROUGH-TRUNK / MOTION verdict whose MLP reading (dAUC of the MLP arm vs the banked MLP E / MLP V+E) is < 0.02 (< 0.01 for V4) or has a CI reaching 0 is reported "not robust to a nonlinear ego baseline"
  (the previous probe found the P-MOVE linear gain over E was mostly E's nonlinearity). F2 (memorisation): a TRAIN VISION-THROUGH-TRUNK whose eval-141 dAUC(Z_nn+E - E) point estimate is < 0.02 is reported "memorisation-suspect, eval-141 does not confirm". A NAV-ONLY or NO-MOTION verdict is NOT softened by the memorisation caveat (the bias favours the Z arm).

## 8. Controls (the pipeline must see a signal and refuse to invent one; a failed control voids the arm it guards: FAILED-CONTROL, no verdict)
* **C1 label shuffle** (labels permuted across the sampled windows, seed 1), full pipeline, on E, N5+E, Z_nn+E, V4+E, Vdiff+E: fold-mean AUC CI must cover 0.5.
* **C2 episode-shuffled labels** (each clip's labels from a donor episode, seed 2) on E, Z_nn+E, V4+E: CI must cover 0.5 and the AUC must not exceed the true arm's AUC.
* **C3 deliberate leak** (the GT label appended) on N5+E, Z_nn+E, V4+E, Vdiff+E: fold-mean AUC >= 0.99.
* **C4 unit tests** `test_trigger_probe2.py` (fold-mean AUC unbiased on a prevalence-shifted null; donor draws never from the same clip and deterministic; the 4-frame gather picks t-3..t; the concatenation / difference readers equal dense computations; the swap changes `nav` and nothing else; tp_lib byte-identical to the old copy; the banked controls pass).
* **N1 identity:** with donor := self the modified extraction path reproduces the banked Z of 64 windows with max |diff| <= 1e-5 (proves the swap harness is a no-op when it should be, and that the exported tree's forward equals the banked one).
* **N2 reachability:** the donor nav must change the model: over the windows whose donor nav differs from their own, h_O changes (relative L2 > 0 for >= 99 % of them) and the emitted plan changes (median abs change of traj60 > 0). If the swap does not reach the output, ARM 2 is FAILED-CONTROL.
* **V1 frame check:** the frame t read by the new extraction equals the banked `V_{pop}` rows (max abs diff 0 on a 200-row subset); the frame order is checked (frame t-1 != frame t for >= 99 % of rows).

## 9. Eval-141 confirmation (no refit), every arm [PIN]
Probes fitted ONCE on all 12,000 sampled TRAIN windows of the population (C / alpha by inner CV on them, PCA on all of them), applied unchanged to eval-141's P-STOP / P-MOVE windows (the previous package's `raw/eval141_features.npz` window set: 608 windows with v0 < 6, 141 episodes; P-STOP n = 101, P-MOVE n = 507; windows have t >= 3).
Eval features: Z_nn with TRAIN donors (`default_rng(1)`, order of the 608 rows, donor from a different episode by construction); V4 / Vdiff from `batch["feats"][:, -4:]` of the eval ClipStore window the model itself reads; nav from the eval join (`join_eval141.npz`). Eval AUC = pooled (single fit), CI = episode-cluster bootstrap by eval episode (B = 2,000, seed 0), paired differences on the same resample.
The 327 GT-launch and 32 GT-stay windows are reported as in the previous probe (launch-vs-stay AUC within v0 < 0.5; mean score percentile against the TRAIN OOF distribution of the same arm). Eval-141 is used for NO choice.

## 10. Strata (TRAIN, as in the previous probe) [PIN]
Traffic light (CLIP-level, VLM-derived: TL_RED / TL_GREEN / TL_YELLOW / TL_ANY / NO_TL from `s2_labels_v7.2_train.jsonl.gz`) and lead vehicle (LEAD_NOW / NO_LEAD / LEAD_DEPART from `lead_train4572.npz`) flags are read from the previous package's `windows_train.npz` (identical definitions). Per stratum, per population: GO rate (full population) and the linear fold-mean AUC of E, V+E, Z+E (banked OOF) and of Z_nn+E, V4+E, Vdiff+E, N5+E (new OOF), with paired deltas against E and V+E.
"too small" = fewer than 30 windows in a class or fewer than 20 episodes in the sample (not computed). No stratum enters a decision rule.

## 11. Resource rules and order
RAM free >= 3 GB before every heavy load (PowerShell `Win32_OperatingSystem.FreePhysicalMemory`; if short: HANDOFF.md and end); memmaps, chunks <= 500 windows (<= 250 for the 163,840-dim reader), RSS < 2 GB; no job waited on for more than ~4 minutes (time-boxed, resumable stages; otherwise HANDOFF.md).
Order: ARM 1 (NAV) -> ARM 2 (Z_nn) -> ARM 3 (V4) -> eval-141 -> strata -> RESULTS.md + pytest; each arm banked to `raw/` when done. Nothing is deleted (unwanted files move to `aside/`). Never touched: other processes, `Keys.txt`, the holdout `540:600`, pods, box 2, GPU, the repo (nothing is staged, committed or pushed).
