# PREREG_P4_CUECLASS: does the LoRA arm lift the go/stay readout on single-frame cue classes and not on pure self-motion? (P4 of THEORY_CRAWL.md section 3, instrument and card; written before any L feature exists and before any per-class AUC exists)

Evidence classes: **[M]** MEASURED (artifact named), **[P]** PUBLISHED, **[E]** ESTIMATED, **[H]** HYPOTHESIS. Tier: every number this card produces is a **T0-class representation probe on logged windows**. It is a statement about what a feature set carries, never about driving.
Scope: zero training, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`. The arm L (LoRA r 32, alpha 32, q and v of all 24 blocks) trains on the PI's GPU pod and does not exist yet. This card, its classes and its decision rule are fixed first; the instrument that reads them is `motion_c/theory/p4/p4_cueclass.py`.

## 0. What was seen before this card froze (disclosure, so the reader can judge the freeze)
1. **Counts, not outcomes [M].** The window counts per class in section 3 come from `p4/p4_counts.py` (output `p4/P4_COUNTS_PREFREEZE.json`, sha256 `3b55f875390b0095ee0a50e9c0d52db6e05e0e3d947a02c23fc6319ebb802511`). The script opens no OOF-score file and no PCA file and computes no AUC.
2. **A coarser stratum read was published earlier and was seen.** `trigger_probe/RESULTS.md` section 4.4 reports V+E (and E) by the probe's own strata (`strata.py`: clip-level colour strata TL_RED / TL_GREEN, LEAD_NOW, NO_LEAD, LEAD_DEPART). That read used overlapping strata, not the partition below. No AUC under the classes of this card has been computed on any feature set. The V+E per-class baseline is run only after this card freezes (section 9), and it decides nothing.
3. **L does not exist.** No L feature, no L checkpoint and no L score has been read. The Delta this card tests has no prior reading.
4. The banked probe numbers used as reproduction targets are read from `trigger_probe/raw/res_p{stop,move}_V_linear.json` [M]: P-STOP V+E pooled 0.6805, fold-mean 0.6799; P-MOVE V+E pooled 0.7844, fold-mean 0.7845.

## 1. The prediction under test
**P4 [H]**, from `THEORY_CRAWL.md` T3: a LoRA on q and v with a per-frame encoder and the fixed 4x10 pool can re-weight what each frame's pooled tokens keep. Light state and lead-vehicle state are single-frame cues, so the go/stay AUC on L's features should rise over V+E on the LIGHT and LEAD classes. It cannot create motion inside one frame, so it should not rise on the self-motion class.
This card reads that statement at the **final checkpoint of each L seed**. "Rises before" is operationalised as lift on the cue classes without lift on SELF at the same checkpoint. An ordering across intermediate checkpoints is reported when such checkpoints are extracted and decides nothing, because their availability is not under this card's control.

## 2. Window sets
* **W-A (decides).** The trigger probe's own training sample: for each population, P-STOP (v0 < 0.5 m/s) and P-MOVE (0.5 <= v0 < 6.0 m/s), the 6,000 GO + 6,000 not-GO windows of `trigger_probe/raw/sample_{pstop,pmove}.npz` (indices into `raw/windows_train.npz`, sha256 `a9a57b32ffac558929b8f57e49a979826b88d17be17e5e9d7f9347e5875783ec`; window-list digest `e144adcd...8168`; P-STOP 849 clips, P-MOVE 2,140 clips). The read is the **out-of-fold** posterior of a probe refit by the probe's own recipe (section 5).
  *Why:* it is the only set on which all three classes clear the floor with a wide margin, it carries the banked V+E out-of-fold reference, and it is the set the banked 0.680 / 0.785 refer to.
  *Its known weakness [H]:* L trained on these windows, so L's features on them can carry label-correlated information that the frozen V never could (the probe's own memorisation caveat, `PREREG_FROZEN.md` section 4). Two guards, both in the rule: the contrast against SELF cancels any lift that is uniform across classes (condition C3, C4), and W-B reads unseen windows.
* **W-B (reported beside, decides nothing).** The eval-141 windows of `trigger_probe/raw/eval141_features.npz` (608 windows with v0 < 6: P-STOP 101, P-MOVE 507; sha256 in `P4_COUNTS_PREFREEZE.json`), with **no refit**: the probes are fitted once on all W-A windows of the population and applied unchanged, exactly as `trigger_probe/eval_confirm.py`. Class labels from the eval-side label files named in section 3. Read per class where the floor (section 6) holds; the counts below show that at least LIGHT in P-MOVE and LIGHT, LEAD, SELF in P-STOP fall below it, so W-B cannot carry the verdict.
* **Not in this card:** the RA-1 holdout clips. They need their own window builder and pre-registration; if the W-A read is HOLDS or F-c (section 7), that is the next set.

## 3. The three classes (labels may use ego, future and other agents; the features may not)
All class membership is label-side metadata. It is never a feature, never enters the probe, and the probe is class-blind: one probe per (feature set, population).
Built from the banked flags of `trigger_probe/raw/windows_train.npz` (`build_windows.py` wrote them from the files below); W-B flags are rebuilt by `p4_counts.py::eval_flags` from the eval files below with the same mapping (cache step t <-> row k = 2t).

| class | definition (a window is in the class iff) | source: file and field | label layer |
|---|---|---|---|
| **LIGHT** | its clip carries a traffic-light tactical goal AND the lead array says NO lead at the window's row | TL: `release/v72/s2_labels_v7.2_train.jsonl.gz` (sha256 `1f557d1cc4c061ef747cac6171a689447047ec368b52b90af26cb81debdd489e`, 4,572 records) record key `g_tac.goals`, any key starting `TRAFFIC_LIGHT_REACT` (the flag `TL_ANY`); eval: `s2_labels_v7.2_eval.jsonl.gz`. Lead: flag `no_lead` | **TL: AUGMENTED v7.2 label release** (provenance `vlm-cot`, `disputed: true`, `time_basis: untimed`), absent from the published PhysicalAI-AV features. **Clip-level and untimed**: a window in a TL clip need not see the light. Lead: see LEAD |
| **LEAD** | no traffic-light goal on its clip AND a lead agent is present with a finite gap at its row | `obstacle.offline` (PhysicalAI-AV 3D agent tracks) through the pod-side join: train `lead_train4572.npz` (sha256 `25494fd9b490c2e4195ea50de95b02c03fc3bb48ca355c2af053a306d718d77d`, fields `sha12, k, lead, gap`), flag `lead_now` = `lead == 1` with finite `gap` at k = 2t; eval `v9labels/inputs/lead_eval139.npz` (sha256 `d59e1c74654f6279e9c578a5a2940dc6e0420e3c57f2431d06273e844022546f`) | published-corpus tracks through our pod-side join (the sixth program-wide read feature), not the episode build |
| **SELF** | no traffic-light goal on its clip AND the lead array says NO lead at its row (the clip has lead rows and `lead == 0` at k = 2t) | flags `~TL_ANY & no_lead` | neither labelled cue is present |

Outside the three classes, reported and never decisive: **MIXED** (TL clip and a lead agent present) and **UNKNOWN** (the clip has no lead rows, so absence of a lead cannot be certified). The five cells partition each sample; the instrument asserts it.
**What SELF is and is not [H].** SELF is "no labelled cue". It can contain unlabelled single-frame cues (stop signs, crossings, cross traffic, the VLM missing a light). A lift on SELF is therefore evidence of a lift on something the labels do not name, not proof of a motion cue. This limits what F-c below can say.

**Counts [M, `P4_COUNTS_PREFREEZE.json`]** (windows n / GO / not-GO / clips):

| W-A | LIGHT | LEAD | SELF | MIXED | UNKNOWN |
|---|---|---|---|---|---|
| P-STOP (n 12,000) | 3,918 / 1,742 / 2,176 / 248 | 3,455 / 1,688 / 1,767 / 322 | 1,694 / 1,082 / 612 / 181 | 2,441 / 1,361 / 1,080 / 213 | 492 / 127 / 365 / 22 |
| P-MOVE (n 12,000) | 1,393 / 793 / 600 / 339 | 4,261 / 2,007 / 2,254 / 830 | 4,922 / 2,452 / 2,470 / 944 | 1,134 / 606 / 528 / 272 | 290 / 142 / 148 / 58 |

| W-B | LIGHT | LEAD | SELF | MIXED | UNKNOWN |
|---|---|---|---|---|---|
| P-STOP (n 101) | 8 / 0 / 8 / 2 | 31 / 27 / 4 / 6 | 26 / 11 / 15 / 7 | 32 / 19 / 13 / 6 | 4 / 4 / 0 / 1 |
| P-MOVE (n 507) | 31 / 15 / 16 / 8 | 126 / 67 / 59 / 17 | 266 / 137 / 129 / 39 | 78 / 41 / 37 / 11 | 6 / 6 / 0 / 1 |

## 4. Feature sets
* **V+E (the frozen baseline).** PCA-64 of the flattened 40,960-dim `pool4x10` of the CURRENT frame t from the frozen DINOv3-L cache, PCA fitted on the outer-train rows of each fold, concatenated with E (the 8 numbers `fk.kin_features(kin3[t])` that O's ego token reads: measured speed, acceleration, curvature from the measured yaw rate). The banked projections are `trigger_probe/raw/pca_V_{pstop,pmove}.npz` key `S` ([6, 12000, 64]; fold 5 = fit on all rows), E is `windows_train.npz` key `E`.
* **L+E (the arm, later).** The same window's `pool4x10` produced by the **merged L encoder**: `W + (alpha/r) B A` folded into the q and v slices of every block (`stack/tanitad/models/v8f_lora_encoder.py` at the tip: `DinoV3LoRAEncoder.merged_encoder`), LoRA tensors from the L checkpoint's `lora` key as saved by `stack/scripts/v8f_mini_train.py` (`enc.lora_state_dict()`), the builder-shaped batch of 8, the fp8 round trip, the same frame (cache step t = raw frame 2t), flattened to 40,960, a **fresh PCA-64** per outer fold by the same `StreamPCA` (oversample 32, 2 power iterations, seed 0), concatenated with the **identical E**. Neither feature set contains the nav token or the speed limit, so no lift can come through nav.
* Row r of an L feature file `V_L_{pop}.npy` is the window `sample_{pop}.npz["idx"][r]`; the file is fp16 [12000, 40960]. Extraction: `p4/extract_v_lora.py`. The L checkpoint's path, sha256, step and seed are recorded in the result and each (checkpoint, seed) is read on its own, never pooled.
* Anything else (E_H, Z, nav, MLP) is not part of this card. The probe family is linear.

## 5. Statistic and estimator
* **Probe refit, as the probe did.** `tp_lib.fit_probe(..., "linear")` (StandardScaler on the training rows; `LogisticRegression(l2, lbfgs, max_iter 2000)`; C in {1e-3, 1e-2, 1e-1, 1, 10} chosen by inner 3-fold CV grouped by clip), 5 outer folds `int(sha12, 16) % 5`, inner `(int(sha12, 16) // 5) % 3`, through `tp_lib.cv_oof`; the code is imported from `trigger_probe/tp_lib.py` and its sha256 (`44b8797740af5cb01dfbdb37d35f4caba80783393c932536127824ca98ee8932`) is asserted. This is the refit of `run_probe.py::stage_lin` and of `wsp_read/p1p_probe_refit.py` (which reproduced the banked out-of-fold scores to 5.3e-7 from raw features and to 2.7e-13 from the banked projections, `P1P_REFIT_R1.json`).
* **Per-class AUC.** AUC_c(F) = the mean over the outer folds of the within-fold AUC of the class-c windows' out-of-fold scores (`tp_lib.FoldWAuc`). This is the probe's reading of record after control C2 (`RESULTS.md` section 3: the pooled out-of-fold AUC is biased below 0.5 on clip-clustered labels because each fold's intercept follows the other folds' prevalence). The pooled out-of-fold AUC of each class is reported beside it. If the two disagree on a verdict condition the disagreement is stated and the verdict stays on the fold-mean.
* **Delta.** Delta(pi, c) = AUC_c(L+E) - AUC_c(V+E) in population pi. **Contrast** D(pi, c) = Delta(pi, c) - Delta(pi, SELF) for c in {LIGHT, LEAD}.
* **Estimator.** Episode-cluster bootstrap over the distinct clips of the population sample, **B 2,000, seed 0** (`tp_lib.boot_counts`); a window's weight is its clip's multiplicity; the SAME resample is used for both feature sets and for all classes, so Delta and D are paired. 95 % percentile interval. **"Separated > 0" means the lower bound is above 0.**
* **Which variance this answers.** One of three: it answers "would another draw of EPISODES say this?". It is blind to training variance (would another L training run say this? `H-ESTIM-SEED-1`) and to inference variance (not applicable: the probe and the encoder forward are deterministic). A separated Delta from one L seed is necessary and not sufficient for a claim that the LoRA lever moved the readout; section 7 names the replicate that is needed.

## 6. Evaluability
A class in a population is evaluable iff it has **at least 30 GO windows, at least 30 not-GO windows and at least 20 clips** in the set being read. **P4 is evaluable in a population iff LIGHT, LEAD and SELF are all evaluable there.** The result is **NOT EVALUABLE** when:
* neither population is evaluable; or
* L features are missing for any window of the sample in a population (the population is skipped and named); or
* a control of section 8 fails on the run (the result is labelled FAILED-CONTROL and no verdict is drawn from it).
Counts below the floor are named per class and never padded. On W-A every class clears the floor by at least 20x (smallest cell: P-MOVE LIGHT with 600 not-GO windows).

## 7. Decision rule, committed as written
For each evaluable population pi (P-STOP, P-MOVE):
* **C1** Delta(pi, LIGHT) is separated > 0.
* **C2** Delta(pi, LEAD) is separated > 0.
* **C3** Delta(pi, SELF) is NOT separated > 0.
* **C4** D(pi, LIGHT) > 0 and D(pi, LEAD) > 0 as point estimates.
* **HOLDS in pi** iff C1 and C2 and C3 and C4.

**P4 HOLDS** iff it is HOLDS in every evaluable population (at least one is evaluable). **P4 FAILS** iff at least one population is evaluable and P4 does not hold in some evaluable population. **P4 NOT EVALUABLE** as in section 6. These are the only three words.

Descriptive tags written next to the word, never extra verdicts:
* `contrast_separated`: true iff the lower bound of D(pi, LIGHT) and of D(pi, LEAD) is above 0 in every evaluable population.
* `self_uninformative`: true iff the upper bound of Delta(pi, SELF) is at least min(Delta(pi, LIGHT), Delta(pi, LEAD)) (C3 held only because the SELF interval is wide).
* FAILS sub-case, taken from the first line that applies in the first failing population: **F-e** some Delta separated < 0 (L reads worse than V+E); **F-c** Delta(pi, SELF) separated > 0 (the lift is not cue-specific); **F-a** neither LIGHT nor LEAD separated > 0; **F-b** exactly one of them; **F-d** both separated > 0, SELF not separated, but D(pi, LIGHT) or D(pi, LEAD) is not above 0.

**What each outcome sends next**
* **HOLDS.** T3 is supported at that checkpoint on W-A. Next, in this order: (i) the second L seed through the same instrument, because the interval cannot see training variance; (ii) the RA-1 holdout window set as W-C, under its own card, because L trained on W-A; (iii) the T2 stage of L proceeds as already planned. No driving claim follows from this card.
* **FAILS F-a or F-b.** The LoRA at r 32 on q and v with the 4x10 pool does not open the cue frontier. The next lever is chosen by largest expected effect among: the 8x20 pool (`GRIDS` in the encoder module), higher input resolution, or a temporal input for the motion cue that T3 says a per-frame encoder cannot give. The cue classes' AUC level against the banked ceilings (V+E 0.680 / 0.785, Z+E 0.801 / 0.880 on W-A) sets the size of the prize.
* **FAILS F-c.** The lift is uniform or lands on SELF: either a generic gain or memorisation of trained-on windows. Next: W-C (unseen windows) before any other claim, then the second seed.
* **FAILS F-d.** Cue classes lift but not more than SELF: treat as F-c.
* **FAILS F-e.** L degrades the readout the probe can use: read the LoRA drift against the frozen encoder (relative L2 of the pooled tokens per class) before any further L arm.
* **NOT EVALUABLE.** Name the blocker and what unblocks it (missing windows, a failed control, a population below the floor). Nothing is inferred.

## 8. Controls (the pipeline must see a signal, refuse to invent one, and read class-specific signal as class-specific)
Run by `p4_cueclass.py controls`; thresholds fixed here. Any failure labels the run FAILED-CONTROL.
* **(a) Reproduction of the banked V+E.** The refit on the banked projections must reproduce the banked V+E out-of-fold scores with maximum absolute difference <= 1e-6 and the banked pooled AUC to <= 1e-6, in both populations. Against the briefed round numbers 0.680 (P-STOP) and 0.785 (P-MOVE) the pooled and fold-mean AUC must agree within 0.001 (the round numbers are the banked 0.6799 / 0.6805 and 0.7844 / 0.7845 rounded to three places, so the half-width of rounding is 0.0005 and the pooled-versus-fold-mean gap is 0.0006 at most).
* **(b) Label shuffle.** The full pipeline on V+E with the GO labels permuted across the population's windows (seed 1): every class's fold-mean AUC within 0.05 of 0.5 (about three standard deviations of a class AUC at these counts [E]), and the overall population fold-mean AUC with an episode-cluster interval that covers 0.5.
* **(c) Planted signal.** One column x = (2y - 1) * 0.5 + N(0, 1) (seed 3) appended to V+E: every class's fold-mean AUC >= 0.74 (the column alone has the analytic AUC Phi(0.5 * sqrt 2) = 0.760 [E]); and the label itself appended: every class >= 0.99.
* **(d) Class-specific plant, through the whole verdict path.** The column of (c) multiplied by the class indicator, with N(0, 1) elsewhere, appended to V+E, as a synthetic "L+E" against V+E: planted in {LIGHT, LEAD} must read HOLDS in both populations; planted in {SELF} must read FAILS with Delta(SELF) separated > 0 in both populations; planted in all three must read FAILS with Delta(SELF) separated > 0 in both populations; a pure N(0, 1) column (seed 4) must read FAILS (never HOLDS), its sub-case reported.
* **(e) Unit tests** `p4/test_p4_cueclass.py`: the partition is exclusive and complete and equals the counts of section 3; the per-class AUC equals scikit-learn's at unit weights; the verdict rule returns each of HOLDS, FAILS F-a..F-e and NOT EVALUABLE on constructed intervals; **a deliberate-regression mutant (the rule with C3 dropped) must make the test go RED.**

## 9. Order of work and the L hand-off
1. Freeze this card (sha256 below). 2. Build the instrument and its tests. 3. Run the controls. 4. Run the **V+E baseline per class** (descriptive, decides nothing, banked as `P4_V+E.json`). 5. When an L checkpoint exists: `extract_v_lora.py` on the pod writes `V_L_{pstop,pmove}.npy` (and the W-B rows); `p4_cueclass.py fit --featureset L_<tag>` and `compare` read it; the result is filed beside this card with the checkpoint's sha256.
Nothing in the L path was run here. `extract_v_lora.py` is written and unit-tested on a zero-B synthetic LoRA; **it has not run on a real frame** (the v2ep frames live only on the pod).

## 10. Limits stated in advance
Clip-level untimed VLM traffic-light labels make LIGHT coarse; SELF is "no labelled cue"; W-A is trained-on for L; W-B cannot carry a verdict at current counts; one L seed per read; the linear probe only; the AUC is a go/stay discrimination score, not a driving metric; the banked probe was run on the TRAIN sample and its P-STOP V+E increment over E is small (RESULTS section 4.1), so the V+E reference per class may sit near the ego-only level in some classes and a Delta there has little room to be read.

## LOG (append-only after freeze)
The frozen body is every byte of this file before the line `## LOG (append-only after freeze)`; `python -I p4_cueclass.py verify-card` recomputes its sha256 and compares it with `PREREG_P4_CUECLASS.sha256`.
* **2026-11-07T02:40:08+0100 FROZEN.** sha256 of the frozen body `6717a0cfe5067f26b7d5175af56b88baf567b457c13810783edc19af207f6731`. No L feature and no per-class AUC existed at this moment. Counts file `P4_COUNTS_PREFREEZE.json` sha256 `3b55f875390b0095ee0a50e9c0d52db6e05e0e3d947a02c23fc6319ebb802511`.
* **2026-11-07T02:41:45+0100 baseline and controls started** (after the freeze). `p4_cueclass.py fit --featureset V+E --source banked` wrote `p4/P4_V+E.json`; `p4_cueclass.py controls` wrote `p4/P4_CONTROLS.json`: (a), (b), (c), (d) PASS in both populations; `pytest` 8 passed, including the C3-dropped mutant being caught. Implementation notes, rule unchanged: the V+E baseline decides nothing; the W-B read of section 2 (no refit, eval-141) has no subcommand yet and is not run; `extract_v_lora.py` has not run on a real frame.
