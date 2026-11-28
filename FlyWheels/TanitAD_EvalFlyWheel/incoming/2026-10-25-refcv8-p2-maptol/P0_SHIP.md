# SHIP -- P0: refcv8's TRAIN-fitted box post-processing (NMS 2.5 m + re-fitted gate), bound to EMA-48,852

*EvalFlyWheel, 2026-10-25 (Berlin local). PI-approved package. NOTHING here is staged or committed; the Master Mind lands it. No git command was run.*

## What is in this pack (and where the Master Mind lands each file)

| pack path | md5 | sha256 | land at (repo path) |
|---|---|---|---|
| `stack/tanitad/configs/refcv8_det_nms_train.json` | `375a0014bf145a49cb87a01bb274738b` | `926b87c7c86566e3dffc6af727fdc9da4380d34486291ecfbd191021e5866976` | `stack/tanitad/configs/refcv8_det_nms_train.json` (NEW file) |
| `stack/tanitad/configs/refcv8_det_presence_gates_train.json` | `c7df00ba6439a944ab0d80a651c1c5ba` | `f4cb9a94f1dc3208599438e6748d19dfd12a0261dd7e476a8934b7ae53b94611` | `stack/tanitad/configs/refcv8_det_presence_gates_train.json` (NEW file) |
| `stack/tests/test_refcv8_det_shipped_pin.py` | see `MD5SUMS.txt` | | `stack/tests/test_refcv8_det_shipped_pin.py` (NEW file, recommended: literal-value pin + deliberate-regression arm) |

* The two JSONs are **byte-identical** copies (`cmp`) of `C:/Users/Admin/refcv8_pf/perc8/raw/T_nosync/refcv8_det_nms_train.json` / `refcv8_det_presence_gates_train.json` -- the outputs of `stack/scripts/refit_perception_thresholds.py` (tool sha256 `5a6db392be352a07daa819c51eca0753e3cbbf890f2db81ccc190a34cdc30d61`, re-checked equal to the launch tree's copy) run on the TRAIN-DIAG pack `train_diag.packs.pkl` (md5 `985537b0895ec5ea882836a3835f99a6`).
* **Naming:** the repo's convention is `<run>_det_nms_train.json` / `<run>_det_presence_gates_train.json` in `stack/tanitad/configs/` (refcv7's pair is `refcv7_det_*`). The consumers take an explicit path, so the name is a convention, not a lookup -- but ⛔ **do not overwrite or rename the refcv7 pair**: `test_refcv8_det_nms.py` pins `refcv7_det_nms_train.json` (checkpoint md5 `d5f104ee...`, gate 0.2145) and `test_refcv7_det_presence_gates.py` pins the gates file. The refcv7 files are byte-identical in this pack's view (md5 `fa76f8ac9503bcafbbec3b2aba466aaf` / `30a5cfe9108c0adcf84f7dbc4d1605e4`).

## Binding

| | |
|---|---|
| **Checkpoint** | `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`, **md5 `b4364029b68f29a33eeb4aebbbbd002b`**, step 48,852, EMA (decay 0.9998). Stamped in both files' `provenance` (`run` = `refcv8-r101-s0-ema48852`, `checkpoint_step` 48852, `checkpoint_md5`). |
| **Fit set (TRAIN, EVAL never used for fitting)** | TRAIN-DIAG: 139 TRAIN episodes x 8 windows of the refcv8 train4572 cache (sha12-sorted, first 139 with a 10 cm GT file); 1,074 box windows, 3,164 positives; EMA-48,852 weights; unshifted pose clock. |
| **Values** | box3d: P = R gate (no NMS) **0.2660**; NMS radius **2.5 m** (TRAIN-AP@2 m argmax; grid 0.5 m 0.3532 ... 2 m 0.4658, **2.5 m 0.4722**, 3 m 0.4561, 4 m 0.4202; no-NMS 0.3462); gate re-fitted after the NMS **0.2165**; `p_floor` 0.05. agent: P = R gate 0.2394; NMS 3.0 m; gate after NMS 0.1844. |
| **Valid for** | exactly that checkpoint. For any other weights (a refcv9, the WiSE blends, a different seed) the pair MUST be re-fitted with `refit_perception_thresholds.py` -- the perc8 control "decoder-only blend DEC: 281 box windows bit-identical to EMA-48,852" is INHERITED and covers only that one blend. |

## Measured effect (MEASURED, ours; T0 perception read; ONE training seed per arm)

Source: `C:/Users/Admin/refcv8_pf/perc8/raw/box_read_final.json` (md5 `5851ac0663596540150a08ee94468314`, tool `perc8_box_read2.py --light`), `perc8/RESULT.md` section 4. S2 eval windows, **4,532 box windows / 139 episodes / 14,586 VIS-1 positives**; estimator = episode-cluster bootstrap, `taniteval.ci` draw stream, seed 0, 1,000 draws, 95 % percentile; within-arm effect = the SAME draws for both variants. The CI answers *another draw of episodes only* (`H-ESTIM-SEED-1`; the forward is deterministic).

| read (refcv8 EMA-48,852) | declared gate 0.5, no NMS | **shipped NMS 2.5 m + gate 0.2165** | within-arm effect [95 % CI] |
|---|---|---|---|
| box AP@2 m (all classes, pooled) | 0.328 [0.301, 0.354] | **0.447 [0.411, 0.484]** | **+0.119 [0.096, 0.139]** SEP |
| class-mean AP@2 m | 0.102 | 0.133 | +0.031 [0.019, 0.043] SEP |
| F1 | 0.026 | **0.457** [0.432, 0.482] | +0.431 [0.405, 0.457] SEP |
| recall / precision | 0.013 / 1.000 | 0.460 / 0.454 | |
| boxes per detected object | 1.000 (only 195 confident slots) | **1.088** [1.075, 1.104] | (without NMS at the P=R gate: 2.142) |
| conf_ratio (n_conf / n_pos; A10 band 0.5-1.5) | 0.013 (**alarm**) | 1.013 [0.945, 1.085] (in band) | |

Context: refcv7@50,400 under ITS shipped pair: 0.249 -> 0.345 (within-arm +0.096 [0.080, 0.113]); refcv8 - refcv7 under the NMS: **+0.102 [0.089, 0.115]** paired. The declared 0.5 gate is unusable on refcv8 (recall 0.013), so any recall / precision statement needs the gate file.
Not scored: the **agent** head's NMS / gate were fitted (same tool) but perc8 scored only box3d -- the agent row of the files is UNVERIFIED on EVAL (EVIDENCE: fit only).

## Who consumes it (read from the launch tree `fb47a583763f` and the code, not from a README)

1. `stack/scripts/refc_v3_train.py` -- opt-in flags **`--det-presence-gates <json>`** (loader `tanitad.eval.detection_metrics.load_head_gates`) and **`--det-nms <json>`** (loader `tanitad.eval.detection_nms.load_head_nms`), both need `--slot-vis1`. The in-run eval then ALSO reports the census at the file's gate under NEW keys `eval_<head>_gated_*` / `eval_<head>_nms_*`; the file's sha256 is stamped into `config.json[vis1][calib]`; `declared_vs_built` registers both as `data`. **refcv8's own launch argv carries neither flag** (checked in `E:/refcv8_eval_kit/ckpt/config.json`, 210 argv entries), so the files change nothing for the finished run; they matter for the NEXT run's argv and for post-hoc reads.
2. Post-hoc reads: `stack/scripts/refit_perception_thresholds.py` (writes this schema), the perc8 box-read scripts (`perc8_box_read.py` / `perc8_box_read2.py --a-gates/--a-nms`), any T0 perception scorer that wants the P = R operating point.
3. ⛔ **NOT consumed by the planner or any loss, and no deploy-path consumer exists in the tree.** `detection_nms` is imported by no module under `tanitad/models` or `tanitad/refs` (pinned by `test_refcv8_det_nms.py`), and the gates file's own caveat says *the planner consumes the raw presence probability as a soft scale and is NOT re-gated by these values*. So "shipping" P0 lands the refcv8-bound operating point for the readout; **wiring an NMS into an inference path that emits boxes is a separate integration this pack does not do** -- the +0.119 AP@2 m is an effect on the box-head readout, not on driving.

## Verification performed (no repo file touched)

* consumer loaders accept both files: `load_head_nms` -> box3d {2.5 m, 0.21648472547531128, 0.05}, agent {3.0 m, 0.1843564212322235, 0.05}; `load_head_gates` -> box3d 0.26599645614624023, agent 0.2393922209739685 (stamps = the sha256s above).
* **42 passed** in a SCRATCH copy of the launch tree (`C:/Users/Admin/refcv8_pf/perc8tol/tree`, extracted from `E:/r8launch/tree_fb47a583763f.tar`, md5 `8996c9e2f04d20d17da1ecbf084f72e9` = `ship_fb47a583763f.md5`) with the two files and the pin test added to its `configs/` / `tests/`: `test_refcv8_det_shipped_pin.py` (6), `test_refcv8_det_nms.py`, `test_refcv7_det_presence_gates.py` -- i.e. landing the new files does not turn the existing refcv7 pins red. Log: `C:/Users/Admin/refcv8_pf/perc8tol/logs/pytest_ship_pin.log`. (The full suite was NOT run; the acceptance_full tests need banked packs.)
* ⚠️ The E: repo checkout is OLD (HEAD `37645fcc6`, 2026-09-19) and holds only the refcv7 gates file; the check that no refcv8 pair already exists was made on the launch tree `fb47a58`, the eval kit's `tree_de4e15b`, the thor bundle tree and the NT-A1 launch trees under `F:/ntgate/gsw/` (none has a `refcv8_det_*` config). The G: HEAD was NOT consulted (hard limit) -- the Master Mind should `git ls-files | grep refcv8_det` before landing.

## Caveats

* Fitted on ONE checkpoint of ONE run (the files' own caveat); the NMS radius is the TRAIN-AP@2 m argmax and is flat near the optimum (2 m 0.4658, 2.5 m 0.4722, 3 m 0.4561).
* Unshifted pose clock for both fit and read; refcv8 TRAINED on the pose-synced clock (mean shift 16.5 ms, max 35.4 ms), so the tight-threshold numbers are, if anything, conservative. The corrected-clock read was not run.
* The NT A1 launch is pinned to commit `54c6028397c8`; these files are not in it and landing them does not alter that pinned tree -- but landing creates a new HEAD, which is the Master Mind's call to time.
