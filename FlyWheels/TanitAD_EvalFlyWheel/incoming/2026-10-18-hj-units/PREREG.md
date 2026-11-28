# PREREG -- HJ-UNITS D2: does "frozen DINOv3 adds nothing beyond kinematics" survive a CORRECT-UNIT `kd_x` base?

**Written 2026-10-18, BEFORE the first run of `code/probe_units_reread.py` in either mode.** Agent: HJ-UNITS (Master Mind brief
2026-10-18, BACKLOG row HJ-UNITS, second clause). CPU only; no pod, no Thor. Evidence class of everything below: HYPOTHESIS until run.

## 1. The claim under test
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refav1-trunk-probe/RESULT.md` s1: the BAR-T1 contrast
`P4 trunk - P2 kin` at 2 s = **+0.0042 [+0.0016, +0.0071]** (separated WORSE => NEGATIVE), on a base `P0 = kd_x` built from
`ha0_ext`, which `code/extract.py:71-78` integrated with `paths_from_controls(...)` at its DEFAULT `action_units="kappa"`.
`actions[:, 0]` is a road-wheel STEER angle (`physicalai.py:621`, `refav1_loader.py:25-29`), so that base was over-rotated ~2.9x
(RETRACTION_LOG R46: `kd_x` 6 s ADE 3.5584 -> 2.9396 m, `ha0_ext` 5.356 -> 2.838 m on this very set of 2,399 windows / 141 episodes).

## 2. What the unit can and cannot touch (derived from source, not assumed)
* NOT touched: the features (`trunk`, `raw1..3` come from `model.encode(feats)` / the loader's `feats`, no action enters), the GT (`g`, `g6`
  from `poses`), `kin` (stored RAW as `(v0, a0, steer0)`), and `ha0` (zero controls: 0 in either unit).
* TOUCHED: `ha0_ext`, `ha0_ext_6`, `ol`, `ha` (integrated), hence `damp50`, `kd_x` = `P0`, and the READOUT TARGET `y = g - kd_x`
  that every learned arm (P1..P5) is fitted to. `P2/P3/P4/P5` all see the 8 kinematic features; `P3/P4/P5` add an image PCA block.
* MECHANISM that could change the conclusion: on the over-rotated base the target is dominated by a kinematics-explainable term
  (lateral error ~ kappa v^2 t^2 / 2, and `v*v*k` is one of the 8 features), which can swamp the small image signal and push the inner
  CV to the smallest image capacity (it chose m = 8 in every fold). On the correct base that term is small, so an image signal, if any,
  has a better chance. The re-read therefore CAN come out either way; it is not a formality.

## 3. Design (ONE lever: the unit of the base)
Code `code/probe_units_reread.py` = the banked `probe.py` (`Readout`, `fit_select`, grids `LAMS`/`MS`, outer 5-fold CV by `episode_index mod 5`,
inner 4-fold grouped CV, `default_rng(0)`, arms P0..P5, 2 s and 6 s, `pb` = `taniteval.ci.paired_episode_cluster_bootstrap`, `n_boot` 2000,
seed 0) VERBATIM, with the base built by `taniteval/floors_kin.py` (`ha0_paths`, `ha0_ext_paths`, `kd_x`):
* `--units kappa`  : base from the STORED `ha0`, `ha0_ext`, `ha0_ext_6` (the banked arrays, wrong unit) -- the CONTROL run.
* `--units steer`  : base REBUILT from the stored `kin` with `action_units="steer"` -- the TREATMENT run.
Data: `D:/Archive/devbox-C/refav1_probe/feat_s4/ep*.npz` (141 files, 2,399 windows; the banked extraction, md5 recorded in the run JSON).

## 4. Controls that must read their known value BEFORE any reading is made (all must hold, else INVALID)
* **C1 (harness = banked).** `--units kappa` reproduces the banked `out/probe_result.json`: every arm's mean ADE@2s, mean ADE@6s and mean FDE@6s
  within **5e-5 m**, and `P4-P2` @2s delta/lo/hi within 5e-4. (5e-5 m is the float32 noise of a path of ~60 m, ulp 3.8e-6, summed over 30 steps.)
* **C2 (rebuild = stored in the old unit).** `fk.ha0_ext_paths(kin, 0.2, n, action_units="kappa")` equals the stored `ha0_ext` (n=10) and
  `ha0_ext_6` (n=30) to **2e-4 m** (R46 measured 9.57e-5); and the `"steer"` rebuild differs from stored by **> 1 m** somewhere (R46: 87 m).
* **C3 (cross-package).** 6 s mean ADE of `ha0_ext`: `[kappa]` = **5.356479 +- 1e-3**, `[steer]` = **2.838042 +- 1e-3**; `kd_x[steer]` = **2.939635 +- 1e-3**
  (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/raw/refav1_units_check.json`).
* **C4 (retime port).** In `--units kappa` the base equals the banked `out/dump_P0_kdx/ep*.npz['cl']` to **1e-5 m**.
* **V (validity).** `P5 shuf - P2` @2 s must NOT be separated BETTER (hi < 0 would mean a shuffled trunk helps = a leak/instrument fault) and
  `P1 const - P2` must not be separated better. Either failure => INVALID, no reading.

## 5. PRIMARY contrast and both outcomes, committed in advance
Primary = `P4 trunk - P2 kin`, ADE @ 2 s, `--units steer`, paired episode-cluster bootstrap over the 141 episodes (n = 2,399 windows).
* **O-STANDS (claim survives at its s1 / 2 s leg):** CI upper bound `hi >= 0`, i.e. the trunk readout is NOT separated better than kinematics.
  Sub-label O-STANDS-WORSE if `lo > 0` (as banked) or O-STANDS-NULL if the CI straddles 0. Consequence: the s1 2 s statement is unit-robust;
  the claim as a whole STILL rests on s2 (spatial) and s3 (A2/A3, train-fitted) which are not re-read here (see s7).
* **O-REVERSED (claim breaks at its s1 leg):** `hi < 0`, i.e. on the correct base the trunk readout is separated BETTER than kinematics.
  Consequence: the banked NEGATIVE was (at least partly) a unit artefact; "adds nothing beyond kinematics" is WITHDRAWN at s1 scope, the
  `P10`/A3 conclusion is re-opened, and the next lever is the same probe on the correct base with the 600-clip TRAIN fit (blocked, s7).
* **INVALID:** any control of s4 fails.
No other outcome label exists. The thresholds are not moved after the data. Secondary rows (6 s `P4-P2` ADE and FDE, `P3-P2`, `P4-P3`, `P2-P0`)
are REPORTED with no decision weight; 6 s is read as exploratory, because the banked 6 s fits hit the lambda-grid ceiling in 3/5 folds.

## 6. Which question the interval answers (CLAUDE.md "three variances")
The episode-cluster bootstrap answers *"would another draw of EPISODES say this?"* only. The ridge fit is closed-form (no training-seed
variance) except the `P5` shuffle (seeded). **Fold-assignment variance (episode mod 5) is NOT measured**, and the checkpoint is a single one
(`refav1-b1-v72-ep3-speed` step 21,109). No lever-effect claim is made; this is a claim about information content at one checkpoint.

## 7. Not answered by this run (named blockers)
* **A2/A3 (the train-fitted MLP, `P9` vs `P10`, 5,398 windows / 600 TRAIN clips).** The train feature dump `C:/Users/Admin/refav1_trainprobe/`
  is NOT on this box (searched C:, D:/Archive, D:/Projects, E:/); regenerating it means re-encoding 600 train clips from Thor (2,834 s of GPU
  encode + a Thor read). Unblock: a PI-authorised Thor/pod slot, or a copy of `refav1_trainprobe/` from wherever it was last held.
* **s2 (4x10 spatial ridge, `P6..P8`).** `feat_s4sp` (705 MB) IS present; not run here (Gram-trick PCA in 40,960 dims, heavier than the budget).
  It is cheap to add next with the same `--units` switch.

## 8. AMENDMENT A1 (2026-10-18, written after a FIRST LAUNCH that produced NO result and BEFORE any result exists)
The first launch (07:01, two processes, `np.linalg.svd`) was killed by me at ~07:08 by explicit PID: 0 folds had completed. MEASURED cost of the banked
`Readout.__init__` SVD on this box: **10.4 s** for a 1,440 x 3,072 raw block and **2.3 s** for 1,440 x 1,024 -> ~150 SVDs per mode = **~12 min solo, ~25 min for two
concurrent modes** -- past the ~8 min detach bar and the token-budget rule. Gram + `eigh` on the same blocks: **1.5 s**.
Changes (speed only; no estimator, grid, fold, seed or arm changes):
* `Readout.__init__`: the top-128 right singular vectors come from the **Gram trick** (eigendecomposition of the smaller of `Z Z^T` / `Z^T Z`), the equivalence-checked
  method the banked package itself used for the spatial arms (`code/chunked_gram.py`: max |d| 2.6e-7). Signs are arbitrary and do not change a ridge prediction.
* The basis depends on the image rows only (not on the base or the target), so `--units both` runs the two modes in ONE process and shares a content-hashed basis cache;
  the rng draws are identical in the two modes, so the shuffled-trunk permutations (and hence the cached bases) are identical too.
* **New control C0 (equivalence of the speed change):** on the first outer fold's P3 and P4 training blocks, Gram vs `np.linalg.svd`: min |cos| over the top 64 components
  >= 1 - 1e-8, and the rank-128 projector difference `max |V^T V - V'^T V'|` <= 1e-6. C1 (kappa mode reproduces the banked means to 5e-5 m) remains the end-to-end guard: if either
  fails, NO steer-mode reading is made and the SVD version must be run instead.
