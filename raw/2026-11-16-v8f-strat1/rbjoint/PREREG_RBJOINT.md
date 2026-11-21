# PREREG_RBJOINT (written 2026-11-18 BEFORE any number of this card exists)

Zero-training-of-the-model decodability test of the roundabout lateral decision (TAC-LAT, 5-way `lat_cls`). Follows RB-DIST-ZT (route text: TEXT-SILENT on the 202 eval windows) and RB-VIS (frozen N1R adapter + attention probe on the 4x10 grid: VIS-SILENT; its train holdout sat at the majority rate, it had no positive control, 16 eval episodes, never vision and exit together, never the 8x20 grid). Evidence class MEASURED (ours). Tier **T0 decodability (NOT driving performance)**. CPU only.
Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` on per-window correctness, B 2,000, seed 0, episodes = train clips (about 426). It answers "another draw of EPISODES" only; it is blind to training variance (a replicate arm would be needed for a lever claim) and the probe seed is fixed (0).

## 1. Windows
* Rbdist's train RB windows (flag 0, RB state, `label_rule != IGNORE`; 24,722 / 426 episodes asserted), with **t >= 3** (frames t-3..t needed; dropped count printed; expected 24,207). Clips whose cache frame count differs from the table are dropped whole and counted.
* **Declared stride 2** per clip (rows 0, 2, 4 ... of the t >= 3 windows in order) for every VISION arm and for the scored set of ALL arms (near-duplicate consecutive windows; fp8 memmap size). S and ED (HGB, cheap) are FIT on all t >= 3 windows of the fold-train clips and SCORED on the strided held-out windows, so all arms are paired on the same scored windows.
* **Primary instrument: 5-fold cross-validation grouped by episode (clip).** Fold = `sha256("rbjoint-fold|"+clip_id) mod 5`. Out-of-fold predictions are pooled; accuracy is over the pooled out-of-fold strided windows. Inside each fold the epoch is chosen on an inner train-only clip holdout (`sha256("rbjoint-inner|"+clip_id) mod 5 == 0`, about 20 % of the fold-train clips; the probe is fit on the other ~80 %, no refit).
* **Eval-202** (rbdist's 202 scored eval windows, 16 episodes): the arms fitted on ALL strided train windows (same inner-holdout epoch rule) are scored there, printed BESIDE the CV result with its own paired CI and **no label**.
* Frames <= t only. No ego / kinematic INPUT in any arm (yaw_rate is only the TARGET of the positive control). Labels may use future ego (they are built offline).

## 2. Arms
* **S**: state only, HGB as rbdist (`rbdist_zt.fit_predict`, cols S). **ED**: state + exit_cur + exit_last + dist + dist_valid, HGB as rbdist (cols ED).
* **RAW4**: raw DINOv3 `pool4x10`, frames t-3..t (4 x 40 tokens of 1024; fp8 cache values), probe below with a **learned linear input projection 1024 -> 256** (LayerNorm(1024) before it) and learned position (40) and time (4) embeddings, instead of the N1R adapter.
* **RAW8**: `pool8x20`, frames **{t-3, t}** (2 x 160 tokens), same probe (position 160, time 2).
* **VED8**: RAW8 with the ED features concatenated to the pooled vector before the head. ED features = [state, one-hot(exit_cur, 9 bins, clipped), one-hot(exit_last, 9 bins), dist/100 clipped [0,3], dist_valid] (21 dims), from the table arrays at rows <= t as rbdist.
* **Probe** (identical for every vision arm): LayerNorm(256), one learned query, 4-head attention over the tokens, LayerNorm(256), [concat ED for VED8], head MLP Linear(256 or 277 -> 128), GELU, Linear(128 -> n_out). AdamW lr 1e-3, wd 0.01, batch 64, 50 warm-up steps, cosine to 0 over **6 epochs**, plain cross-entropy (no class weights), probe seed 0. Checkpoints at epochs **{3, 6}** of the same run; the one with the higher inner-holdout accuracy is used (ties to epoch 3).
* Positive-control and regression arms use the same probe: **YAW4 / YAW8** (target below), **SHUF_RAW4**, **SHUF_VED8**, **SHUF_ED** (train labels permuted).

## 3. Metric
* Primary: 5-way TAC-LAT accuracy over the pooled out-of-fold windows; level CI per arm; paired differences with the paired episode-cluster bootstrap. "Separated" = paired CI lower bound > 0. Margin **M = 0.0383** (the NAVRB2 rig floor).
* Secondary, no bar: timing (3-way: lane keep / turn / lane change, as RB-VIS) gain over S, and side-given-turn accuracy (windows with a labelled turn class), per arm.

## 4. Validity (if any fails the whole read is INVALID and no label stands)
* **POS-YAW**: YAW4 and YAW8 decode the sign class of `yaw_rate` at frame t (cache index t; class L if > +0.05 rad/s, R if < -0.05, else straight; the sign convention does not matter for accuracy) at least **+0.10** above the majority-class rate (per fold, the fold-train majority class as a constant predictor), paired CI separated, on the same CV. Proves the probe can read these features at this budget.
* **R-SHUF-TRAIN**: for each of SHUF_ED, SHUF_RAW4, SHUF_VED8, accuracy minus S is **< M** (train labels permuted at window level inside each fold-train set, seed = fold; the shuffled labels also feed that arm's inner holdout).
* **K-CONST**: S predicts the fold-train majority class on every held-out window in every fold, and its pooled accuracy equals the pooled share of that class exactly.
* **Fold clip-disjointness** (a clip in two folds) reads 0 for every fold.
* **R-SHIFT** (test): the window reader with a +1 shift must read frames > t, and the unit test turns RED under that mutation.

## 5. Labels (every label that holds is reported; the next lever follows the first that holds, in this order)
* **JOINT** iff VED8 - S, VED8 - ED and VED8 - RAW8 are all >= M and separated => the exit/countdown training arm (RB-DIST) comes back.
* **GRID** iff RAW8 - S >= M separated AND RAW8 - RAW4 >= M separated => an input-resolution lever.
* **VIS** iff RAW4 - S or RAW8 - S >= M separated => vision alone carries it; the next test is the head gap on train windows.
* **TEXT** iff ED - S >= M separated.
* **SILENT** iff none of these (validity passing) => zero-training audit of the TAC-LAT label on roundabout windows.
* A point >= M with an unseparated CI is printed as suggestive-unseparated and does not satisfy a label.

## 6. Limits stated now
* One probe family, one seed, fixed budget: a SILENT result bounds what this probe reads (the positive control bounds how weak that is), not learnability by any reader.
* The route text of the train windows is along the driven path (optimistic); the eval-202 route is the same kind.
* Stride 2 halves the train windows seen by vision arms; S / ED see all.
* The eval-202 cells carry 16 episodes (RB-VIS intervals were 0.10-0.20 wide) and are descriptive.
* The bootstrap over clips answers "another draw of episodes"; it does not show that another training run would agree.
