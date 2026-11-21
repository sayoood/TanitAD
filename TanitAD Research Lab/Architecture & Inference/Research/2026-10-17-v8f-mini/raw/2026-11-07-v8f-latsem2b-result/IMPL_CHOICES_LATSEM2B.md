# IMPL_CHOICES_LATSEM2B: exact sources, scheme, rules and control wiring, fixed BEFORE any LATSEM2B statistic exists (agent, dev box, 2026-11-07)
Card: `../PREREG_V8F_LATSEM2B.md`, frozen prefix 4,611 bytes, sha256 `bf6b37178818f778cf9220c4abac1a44de11a954d0e3a2b57f39e8e8b1bc48a8`. Amendment 1: `../AMENDMENT_LATSEM2B_1.md`, frozen prefix 2,778 bytes, sha256 `120072ffb46081bec267fc4f515e5704848272792790eb2577f357c8739ece53`. Both prefixes were re-hashed before this file was written. Nothing here edits the card or the amendment or moves a bar. Tier T0 throughout: a world-model diagnostic, never a driving claim.
What exists when this file is frozen: the TRAIN label module and arrays (label counts, the label alignment check and the gap-speed correlation; labels only), and run 1's code and numbers (LATSEM2, `../latsem2/`, seen). No LATSEM2B probe, rollout, permutation draw, R2, SKILL, S3 or S3-PARTIAL value has been computed.
Run 1's rules apply unchanged wherever this file does not say otherwise: `../latsem2/IMPL_CHOICES_LATSEM2.md` (sha256 `b17688dc0fea5c639ab2d44c7088f88cded05ab99a1b75f89272cbb78492e71d`) sections 2 to 7 (bank, folds, probes, S1, S2, S3, targets, NaN rules, shrink, estimator).

## 1. Inputs (sha256)
| input | path | sha256 |
|---|---|---|
| TRAIN latents, windows (N 2,397; 141 clips) | `theory/wmact/latents_train.pt` | `55a9bdc95f95352916e5aa752cca6b8bdf82bffe110824d3014cf4878b64c55e` |
| eval-141 latents | `theory/wmact/latents.pt` | `7077ec717f633e0ef0814fec0a5fb43609e714bd8f521ccf6eb5e64ad02333a4` |
| L1 shrink factors c_k | `theory/wmact/RESULT_WMACT_SHRINK.json`, `rows["{tag}/L1/k{k}"]["c_train"]` | `bb6df3a324cb0f3ac432ab3ca4604d19f7ea7f840591d4c66f377b768fd9d4eb` |
| TRAIN labels and ego speed | `theory/latsem2b/labels_latsem2b_train.npz` (builder `labels_latsem2b.py` `eb2b6b10b0ce1e13b79185e872585d9e9e5909ac7ee250916df072eb86170dbe`) | `4ea428f15bc9d900238b32444acc6c5a82f73472a127665eeeed745ded85c6cb` |
| eval-141 labels (LATSEM2's frozen arrays; builder `labels_latsem2.py` `92b673d1...`) | `theory/latsem2/labels_latsem2.npz` | `ff49e0350a59bf37db801f885630f7c4e545f7e3accb1277060fff3c92f53eed` |
| ego speed field | `sfresh_inputs/inputs/native_tracks.pt`, `tracks[clip][2g, 3]` | `1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7` |
| models M2b s0 / s1 | `v8f-mini-hacc/M2b_s{0,1}_hacc.pt` | `71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0` / `32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a` |
| bootstrap helper | `theory/wmact/wmact_lib.py` | `951b01eb694eca99f0a1cd1a49194458bf2bb8026f22dcb078741b36e31d3b92` |
| code export (pinned tip; `tanitad.__file__` asserted under it) | `scratchpad/tip_1246c8dd` | commit 1246c8dd |
The TRAIN sources of the label arrays are in `labels_latsem2b_train.report.json` (`sources_sha256`): the lead file `lead_train.npz` (`f0440deb`), the v9 release `v9_labels_train.npz` (md5 `f63ece41`, sha `ddd8e141`), the 2-D B1 TRAIN agent join (`845bdbec`), `native_tracks.pt` (`1db431f1`), `latents_train.pt`, `clips141.json`. The eval-141 sources are those of run 1's IMPL_CHOICES section 1.

## 2. Datasets
* **TRAIN (deciding):** WMACT's TRAIN windows, 141 clips. Frame bank by LATSEM2's rule on `latents_train.pt` with every clip kept (no frame is read in this run): **9,857 frames, 141 episodes** (77 / 64 by fold). Windows with all 16 frames real: **1,833**. S3 windows (a near lead at frame 0): **461 in 56 episodes**.
* **eval-141 (beside, post hoc):** LATSEM2's bank and arrays: **9,460 frames, 139 episodes**, **1,948** windows, **505** S3 windows in 53 episodes.
* Folds, probes, targets and NaN rules are run 1's. Both datasets go through one code path (`run_latsem2b.py`); only the bank, the label arrays and the speed array differ.

### Label sources and n, TRAIN (frames with a label; episodes; episodes in fit fold 0 / 1)
| target | source | n frames | n episodes | episodes by fold |
|---|---|---|---|---|
| lead_present | `lead_train.npz` via `D.build_label_join` + `D.lead_label_arrays` (the trainer's own call) | 8,964 | 134 | 73 / 61 |
| lead_gap | same | 2,456 | 59 | 29 / 30 |
| lead_closing | same, 20 m/s cap | 2,330 | 59 | 29 / 30 |
| agents30, nearest | B1 TRAIN agent join (2-D), centre distance in the ego frame | 8,964; 8,877 | 134; 134 | 73 / 61 |
| lon7 | v9 release `lon_cls` | 8,758 | 131 | 71 / 60 |
| turn | `native_tracks.pt` heading, 4 s | 9,716 | 141 | 77 / 64 |
| beside: lead_closing_uncapped, agents30_view | as above | 2,350; 8,964 | 59; 134 | 29 / 30; 73 / 61 |
eval-141 n are run 1's (IMPL_CHOICES_LATSEM2 section 3).

### Disclosures (decide nothing)
* 7 of the 141 TRAIN clips (`006ffa2a 0566c11e 2388c864 40745350 9d700f6b d9bf0dc9 ebdc0b68`) have no v9 row, no lead label and no agent line: NO_LABEL for every target but `turn`. 3 more clips carry only the IGNORE `lon_cls`. They stay in the bank.
* The TRAIN agent join is the 2-D B1 TRAIN join (the 3-D `.part` file is incomplete); it carries `cx, cy, l, occ` unchanged from the 3-D eval join, which is all `agents30`, `nearest` and `agents30_view` read.
* Label alignment on TRAIN (corridor agent distance against the lead gap, raw-frame offset -1 / 0 / +1): r 0.682 / 0.652 / 0.667 (n about 1,980); peak at -1. Eval-141 peaked at 0. A one-raw-frame (0.1 s) question; `agents30` is insensitive to it. Nothing is shifted.
* The L1 shrink factor c_k was fitted on these TRAIN windows: an in-sample scalar per seed and k on the deciding set. On eval-141 it is out of sample.
* Gap and ego speed correlate at r +0.450 on TRAIN (n 2,450 of 2,456 lead_gap frames: 6 frames have no ego speed) and +0.522 on eval-141 (n 2,489). The 6 frames drop out of the S3-PARTIAL fit and the residualisation only.
* LATSEM2's "episode has a frame file" filter is not applied on TRAIN (no frame is used). It stays in force on eval-141 because eval-141 uses LATSEM2's own bank.
* The permutation labels move whole label sequences between episodes (below). For a target with few labelled episodes in a fit fold (lead_gap 29 / 30 on TRAIN) the reference is built from at most that many sources; each draw is a distinct random cyclic order of those episodes.

## 3. The permutation scheme (Amendment 1, as realised)
* A draw has an integer seed s. Reference: s = 1 to 199 (R = 199). Calibration: s = 1001 to 1020 (20 draws). The two sets are disjoint.
* **S1 and S2 probes.** Per (draw s, model seed tag): `rng = numpy.random.default_rng(s)`, created once and consumed in the order of run 1's section 8.3: level 1, 2, 3; targets in the order lead_present, lead_gap, lead_closing, agents30, nearest, lon7, turn, lead_closing_uncapped, agents30_view; fold 0, 1. Each call is run 1's `perm_labels(y_fit, episode_fit, rng)`, code unchanged: the fit-fold episodes with a finite label are put in a random order (`rng.permutation`), each takes the label sequence of its neighbour in that order (the previous one; the first takes the last's), cycled to its length; no episode keeps its own. The amendment states the direction as "next"; reading the random order backwards gives the same distribution of assignments, and the code is not edited. The probe (same kind and hyper-parameters as the real one) is refitted on the permuted fit-fold labels; fold k's probe is scored on fold 1-k.
* **Scoring a permuted probe.** Against the TRUE labels of the other fold; the baseline is the TRUE fit-fold mean (continuous) or majority class (classification), exactly the value the real probe stores. The same permuted probes are passed through the S2 queries (the decision queries: L1 shrunk, L2 and L3 raw) and scored against the true future labels; the start-state decode uses the same permuted probe.
* **S3 and S3-PARTIAL probes.** Per (draw s, tag): `rng = default_rng(s)`; for fold 0 then fold 1 one `rng.permutation(n)` over the fit-fold lead frames (finite `lead_gap`) moves the gaps across frames (run 1's control 4), shared across the three levels. The raw-S3 probe is a ridge (lambda 10) on those permuted gaps. The S3-PARTIAL probe is a ridge on the permuted gaps after the residualisation of section 5, on the fit-fold lead frames that also have an ego speed. S3 and S3-PARTIAL are then computed as in the real run.
* **Distinctness.** For tag s0 the SHA-256 (first 16 hex) of each permuted label vector is stored per (draw, level, target, fold). The result reports, per cell, how many of the 219 draws gave distinct label vectors. This is a diagnostic: it does not gate anything.

## 4. The two-part separation rule (card, Instrument 1)
* **Cells.** S1: (tag, level, target). S2: (tag, level, k, target) on the decision query. S3-PARTIAL: (tag, level). Each cell has its own reference, the R = 199 point estimates of the draws s = 1 to 199 (a point that is NaN is dropped and counted).
* **Thresholds.** `numpy.nanpercentile(ref, 97.5)` and `numpy.nanpercentile(ref, 2.5)`, default linear interpolation.
* **Claimed side.** Above 0 for S1 and S2. Below 0 for S3-PARTIAL.
* **A row is separated** iff its bootstrap interval (episode-cluster, B 2,000, seed 0, `wmact_lib.boot_weights`, run 1's `row()` and `bal_row()`) excludes 0 on the claimed side AND its point is strictly beyond the cell's threshold on that side (point > 97.5th percentile; point < 2.5th percentile for S3-PARTIAL). A row that is NaN is not separated.
* Beside each real row the result prints the reference rank `(1 + number of reference points at or beyond the point) / (R + 1)` on the claimed side.
* **Calibration draw as if real.** The calibration draw's probes are scored as a real row (same function, bootstrap with the probes fixed); it is declared separated by the same rule against the same cell's reference. Families pool tags, levels and all 9 targets: S1 classification (3 targets x 3 levels x 2 tags x 20 draws = 360 rows), S1 continuous (6 x 3 x 2 x 20 = 720), S2 (9 targets x 7 (level, k) x 2 x 20 = 2,520), S3-PARTIAL (3 x 2 x 20 = 120). Beside, the share declared by the bootstrap interval alone (the first part of the rule) is printed for each family.

## 5. S3-PARTIAL (card, Instrument 2)
* **Residualisation.** On the fit fold of fold k, the lead frames with a finite `lead_gap` and a finite ego speed: `a, b` = `numpy.polyfit(speed, gap, 1)` (OLS with intercept); residual target `r = gap - (a + b * speed)`. Ego speed is `native_tracks.pt` column 3 at raw row 2g (TRAIN: the `speed` array of `labels_latsem2b_train.npz`; eval-141: the same expression run 1's `beside_gap_speed.py` used, computed in the run).
* **Probe.** Ridge lambda 10 on the level's standardised latent (`fit_probe("cont", ...)`), fitted on that fit fold, one per (tag, level, fold). Cross-fitted: a window of fold k is decoded by the probe fitted on the other fold.
* **S3-PARTIAL** = mean over the S3 windows of [decoded residual (accelerate) - decoded residual (brake)] at the longest horizon (L1 k10, L2 k3, L3 k3), the L1 decision on the shrunk rollout (`z0 + c_k (zhat - z0)`, c_k of k10), per-episode sums, bootstrapped as run 1's S3. Raw actions, constant over the window: accelerate (+1.5, 0), brake (-3.0, 0), brake rolled a second time.
* **Raw S3** (run 1's: the gap probe fitted on the unresidualised gap) is computed beside, with the same frame-permuted reference; it decides nothing. A negative accelerate-minus-brake difference is the action-semantic sign.

## 6. Controls (each must read its value, or the read is INVALID; nothing is tuned to pass)
Each control is evaluated on both datasets. A failure on either dataset makes the read INVALID; the result names the dataset. Numbers of an INVALID read stay in the json under `uncontrolled` and decide nothing.
1. **Start-state query SKILL exactly 0.0:** every real S2 row of the `start` query, point and both interval ends exactly 0.0.
2. **Identical action sequences:** brake against its second rollout, S3 and S3-PARTIAL, per tag and level: point exactly 0.0, interval [0, 0].
3. **Calibration:** the 20 calibration draws of section 4. Pass iff the share declared separated is at most 5 % in each of the four families.
4. **Speed removed:** on every fit fold, `abs(corrcoef(residual target, ego speed)) <= 1e-6`. Beside: the same for the permuted residual targets of all 219 draws (max abs reported).
Also re-run and reported, not controls: run 1's copy checks (copies of one frame agree to 1e-5) and per-level checks (per-frame z2 / z3 equal `HJEPAStack.encode` to 1e-5); the self-test of the balanced-accuracy code; and **a reproduction check**: the real eval-141 S1 main points, S2 decision points and raw-S3 points against `../latsem2/RESULT_LATSEM2.json` (same probes, same labels, so they agree; the max abs difference is printed).

## 7. Outcomes (card, per level and target; deciding set first, eval-141 beside)
* **ENCODES(q, l)** on a dataset: the row is separated by the two-part rule at both seeds.
* **IMAGINES(q, l, k)**: the same on the S2 decision query.
* **SCENE-AWARE at l:** ENCODES `lead_gap` or ENCODES `agents30` on TRAIN, and on eval-141 the point of that same target and level is above 0 at both seeds ("the eval-141 estimate has the same sign"; its own separation is tabulated beside). The result names the target(s).
* **SCENE-IMAGINING at l:** IMAGINES `lead_gap` at l's longest horizon on TRAIN, and the eval-141 point of that row is above 0 at both seeds.
* **ACTION-SEMANTIC-LEAD at l:** S3-PARTIAL separated (point below 0 and below the 2.5th percentile, interval below 0) at both seeds on TRAIN, and the eval-141 S3-PARTIAL point below 0 at both seeds.
* Every other target's ENCODES, every other (k, target) IMAGINES, and eval-141's own separations are tabulated beside and decide nothing. Every target has TRAIN labels, so none is reported eval-only.
* **Next lever** (card, unchanged): no level SCENE-AWARE: MRATE W1, then a scene-target auxiliary under its own card. A level SCENE-AWARE but not SCENE-IMAGINING: the reliance arm on that level. `theory/mrate/` holds no W0 / W1 result, so no row beside is available.
* Run 1's numbers are shown beside in the md, labelled: eval-141, one-part rule, INVALID read.

## 8. Execution
* CPU only (`CUDA_VISIBLE_DEVICES=-1`). **Thread count 4:** the real-probe and rollout stages run in the main process with 4 threads; the permutation draws run in a pool of 4 worker processes, each with 1 BLAS thread (`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`), the main process idle meanwhile. Never more than 4 busy threads.
* `sys.dont_write_bytecode = True`: no .pyc is written into `latsem2/`, `wmact/` or here.
* **Timing gate.** Before the full run, 5 reference draws (s = 1 to 5) on TRAIN are timed in the pool; the projected total is printed (real stages + 2 datasets x 219 draws x mean draw time / 4 workers). If it exceeds 10,800 s the run stops and writes nothing but the numbers. Draws are deterministic in s, so the timed draws are reused as part of the 199.
* Order: TRAIN (real S1 / S2 / S3, controls 1, 2, 4; reference; calibration), then eval-141 the same. `RESULT_LATSEM2B.json` is written after each stage. Each finished draw is appended to `draws_{dataset}.jsonl`; a restart skips the draws present.
* Estimator: episode-cluster bootstrap, B 2,000, seed 0, over the dataset's bank episodes; ratios of per-episode sums; paired where quantities share frames or windows. The permutation reference answers "would this appear with no relation between latent and label?"; the two seeds answer "another training run". Neither speaks to driving: this is T0.
* Non-convergence of the logistic fits is counted and reported, never tuned.
* Outputs in `theory/latsem2b/`: `RESULT_LATSEM2B.json`, `RESULT_LATSEM2B.md` (generated from the json), `run_latsem2b.py`, `report_latsem2b.py`, `draws_train.jsonl`, `draws_eval141.jsonl`, logs, `fig/` (PCA maps per level, coloured by lead gap, no camera frame; sha256 listed in the md).
* If a committed value turns out not to be realisable as written, the run stops before computing and the evidence goes to the handoff.
