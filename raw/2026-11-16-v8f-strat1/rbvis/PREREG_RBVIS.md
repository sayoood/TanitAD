# PREREG_RBVIS (written 2026-11-18 BEFORE any number of this card exists)

Zero-training-of-the-model, frozen-feature probe. Follows RB-DIST-ZT (`rbdist/PREREG_RBDIST_ZT.md`, result TEXT-SILENT: ED - S = -0.0792, not separated). Evidence class MEASURED (ours). Tier **T0 vision-only decodability and T0 head read (NOT driving performance)**. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` on per-window correctness indicators, B 2,000, seed 0, the 16 episodes of the scored RB windows; it answers "another draw of EPISODES" only. The two seeds answer the training-run question only as two separate reads (the probe is a fresh fit per arm; no seed-variance interval is claimed).

## 1. Question
Do the frozen vision features of the N1R models carry the executed lateral decision (TAC-LAT, 5-way `lat_cls`) on roundabout (RB) windows beyond the state-only prior, and does the trained N1R head use what they carry?

## 2. Windows (identical to RB-DIST-ZT, same selection code)
* Eval: the 2,399 eval-141 grid windows (`read/t1/dump_STRAT1-C0_s0_m10_causal_heads`: eid, clip_index, ws); RB = `rbdist_zt.rb_state` at row `eval_off[clip]+ws`; scored = RB with `label_rule(has, lat_cls) != IGNORE`. The run refuses unless it reads **235 RB windows in 16 episodes** and **202 scored windows in 16 episodes**, and unless `win_ci`, `win_t`, `eid` of the N1R T0 dumps equal the dump's ws/clip_index/eid. Features of window (clip c, ws t): cache frames t-3..t of the eval cache clip (key `pool4x10`).
* Train fit set: rbdist's train RB windows (rows with flag 0, RB state, `has`, `lat_cls >= 0`; 24,722 windows / 426 episodes asserted before the t >= 3 filter). A window needs frames t-3..t, so windows with t < 3 are dropped and COUNTED. **Declared deterministic subsample: every 2nd remaining window in row order inside each clip (indices 0, 2, 4 ...)**, because the consecutive windows of one clip are near duplicates, and the float8 feature tensor (4 x 40 x 1024 bytes per window) must fit the shared dev-box commit budget with CPU-only fitting. The kept count is printed.
* Train / eval clip disjointness must read 0 (table train vs eval clips; train cache file names vs eval clips). Frames <= t only.

## 3. Arms (all vision-only at inference: no ego, no kinematics, no route text, no nav channel, no situation classifier)
* **V_s0, V_s1**: the frozen adapter (`adapter.0.*`, `adapter.1.*`, `pos`, `time`) of `navrb2/read/ckpts/N1R_s0|N1R_s1/ckpt_last.pt` (sha256 printed) + the attention-pool probe of `weatherprobe/wprobe_fit.py` (LayerNorm, one learned query, 4-head attention over the 160 tokens of frames t-3..t, LayerNorm, Linear to **5** classes), AdamW lr 1e-3, wd 0.01, 100 warm-up steps, cosine to 0, batch 256, plain cross-entropy (no class weights; the metric is accuracy). Fit once on TRAIN; epochs E chosen from {2, 4} by held-out 5-way accuracy on a train-only 20 % clip holdout (fit on the other 80 %), ties to the smaller E, then refit on all train windows at E. The eval windows enter no fit and no choice.
* **RAND**: the same probe (probe seed 0) on a random-init adapter (nn default LayerNorm/Linear init under seed 0, `pos`/`time` ~ N(0, 0.02^2)), same E-choice protocol. Control of the weather probe, where a random adapter also detected.
* **S**: rbdist's state-only arm; recomputed with `rbdist_zt.fit_predict` and asserted equal to `rbdist/out/RESULT_RBDIST_ZT.json` S accuracy (to 1e-12). On RB windows its prediction is the constant train-RB majority class.
* **H_s0, H_s1**: argmax of `lat_logits` of the trained N1R_s0 / N1R_s1 T0 "real" dumps (`navrb2/read/out/t0/eval_NAVRB2-N1R_s*_real.npz`) at the same 202 windows. T0 matched to the probe; the head was fed its real route/nav channels. T1 is not read here (named, not used).

## 4. Metric and cells
Primary: 5-way TAC-LAT accuracy on the 202 windows, level CI per arm and paired differences V - S, V - H, RAND - S, per seed. Descriptive (no bar): timing (lane-keep vs turn: class 0 vs {1,2} vs lane change {3,4}, 3-way timing accuracy) and side-given-turn (accuracy on the labelled-TURN windows, expected 43 windows) per arm; macro-F1; prediction counts.

## 5. Committed rules and both outcomes (margin M = 0.0383, the NAVRB2 rig floor at this stratum)
* **VIS-CARRIES** iff, at BOTH seeds, V - S >= M and the paired CI is separated (lo > 0). Otherwise **VIS-SILENT** (a point >= M with an unseparated CI is printed as SUGGESTIVE-UNSEPARATED inside VIS-SILENT).
* **HEAD-GAP** iff VIS-CARRIES and, at BOTH seeds, V - H >= M and separated.
* Readings: VIS-CARRIES + HEAD-GAP -> the information is in the frozen features and the head does not use it: next lever is the training signal (RB-window weight / per-stratum loss). VIS-CARRIES without HEAD-GAP -> the head already uses what the features carry: the lever is not the head. VIS-SILENT -> data / encoder lever (the DINOv3 pooled features at this grid do not carry the decision on these windows).
* **RAND flag**: if RAND - S >= M and separated, the carried information sits in the cached DINOv3 features, not the trained adapter; stated beside whichever reading holds.
* If any control below fails, the verdict is INVALID and no reading stands.

## 6. Controls (each must read as stated)
* **K-CONST**: S predicts the single train-RB majority class on all eval windows; its accuracy equals the eval share of that class exactly.
* **R-SHUF-TRAIN** (deliberate regression): V_s0 with the train labels permuted between train clips (seed 0, rows by position modulo), same E, same eval: V - S must be < M and not satisfy the VIS-CARRIES rule.
* **R-SHIFT** (deliberate regression, test): the window reader with a +1 frame shift must differ from the unshifted reader and the unit test must turn RED when the reader is mutated to read frame t+1.
* **K-DISJOINT**: train/eval clip overlap = 0 in every listed form.
* **K-IDENT**: the loaded V adapter equals the checkpoint tensors (sha256 of each tensor printed); the probe has no input other than the 4 x 40 x 1024 features.

## 7. Limits stated now
* 16 eval episodes, 43 labelled turn windows in about 6 episodes: wide intervals; timing and side cells are descriptive.
* The probe is a small learned reader on 12k windows: a silent result bounds what THIS probe reads, not what any reader could read (a negative from a probe is not a negative about learnability).
* The route supplied to H is along the driven path (optimistic); V and S use none.
* Eval cache provenance (encoder, pipeline) and train cache provenance are printed and compared; if they differ the difference is stated.
