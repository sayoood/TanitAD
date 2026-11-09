# HANDOFF: FIT arm logits (2026-11-09). Built and tested on the dev box (CPU, tiny stand-in model, plus a real C0 load). Never run on the pod.

## What it does
`fit_arm_logits.py` runs the P6 base model ONCE over the 8,489 FIT sets of `fit_pack.npz` (GPU), taps what `REFe.forward` hands its scorer (the proposals fed to `score_q_mlp`, the context `score_dec[0]` reads), and applies every arm's three scorer modules (`score_q_mlp`, `score_dec`, `score_head`) to those same tensors. Output: `OUT/<NAME>/fit_logits.npz` (`log token step rank logits[N,64,6]`, pack order) for each arm and `P6_control`, plus `OUT/report.json`. It imports `eval/arm_logits_pass.py` (helpers only: `load_ckpt verify_arm ScorerArm Tap`; that file imports only numpy and torch, the seam is loaded lazily and not needed here). Copy BOTH files into one pod directory.

## Pod command (after the three arms finish; the GPU must be free; do not touch the GPU lock logic)
```
cd /workspace/armfit        # fit_arm_logits.py + arm_logits_pass.py
export PYTHONPATH=/workspace/dz/DriveZero/DriveRL/nuplan-devkit   # CORRECTED by the Master Mind: the path written here first does not exist on the pod; this is the running fine-tune's value
# ⚠️ ALSO PASS --targets /workspace/data/refe_navtrain/train_grow_cmd_3c_p7: that is what labels_manifest.json records for the label job; the default below is not
/workspace/venv-teacher/bin/python fit_arm_logits.py --refe-dir /workspace/refe-plan-scorer1p6/refe \
  --base-ckpt /workspace/data/refe_runs/p6_paper_faithful/model_final.pt \
  --arm C0=/workspace/data/refe_scorer1p6/c0/model_sft_B.pt --arm C0r=/workspace/data/refe_scorer1p6/c0r/model_sft_B.pt --arm S1E=/workspace/data/refe_scorer1p6/s1e/model_sft_B.pt \
  --pack /workspace/data/p6base_labels/fit_pack/fit_pack.npz --out /workspace/data/p6base_labels/arm_fit > /workspace/data/p6base_labels/arm_fit.log 2>&1
```
Launch detached (setsid nohup ... < /dev/null &), watch the log for `ZZFITARM_OK` (exit 0) or `ZZFITARM_FAIL <reason>` (exit 1, hand the GPU on). Smoke first: add `--limit 64` (about 1 min, writes PARTIAL files; use another `--out`). Resume = same command; a changed argument in the same `--out` is refused.
Defaults match the label chain: `--targets /workspace/data/refe_navtrain/train_grow_cmd_3c --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --backbone vitl16 --batch-size 4 --workers 4`.

## Where it differs from the label chain (checked against `labels/p6_labels_chain.sh`)
- Chain defaults were `PKG=/workspace/refe-plan`, `MODEL=/workspace/data/refe_p6/model_final.pt` ("guess, set it"), `PY=python`. The paths above (from the brief) differ; the base sha256 is checked against `968a469d...` so a wrong file refuses.
- The chain dumped FIT in two runs (heldout: 1,705 sets in the local pack; fit_extra: the rest), 4 per batch in sorted (log, token) order. This tool re-forms those batches from `is_heldout` and the pack order. It reads the pack, not `fit_sets/`, so `fit_sets` and `fit_pdm` are not needed here.

## Controls (each fails fast per batch, and again over the assembled result)
internal: `P6_control` via the hooks == the planner's own score, max abs diff exactly 0.0 · reference: control logits vs pack logits <= `--ref-tol` (default 1e-3; pack stores 6 decimals) · proposals vs pack `traj_xy`/`yaw`: both are stored to 5 decimals (MEASURED on the local pack, 2026-11-09: xy and yaw sit on the 1e-5 grid; ⚠️ this line first said yaw had 6 decimals with a 1e-6 tolerance, and the first pod run refused at 5.007e-6, which is rounding), so defaults are xy 2e-5, yaw 2e-5 (`--prop-xy-tol`, `--prop-yaw-tol`) · guard: all non-scorer state keys bit-equal to the base, before and after, and the checkpoint files unchanged on disk · the control modules still equal the planner's. The value of every tolerance is printed at the start. `report.json` holds the observed maxima; a healthy run should show control logit diff near 0 (rounding only, about 5e-7).

## Then, on the dev box
`make_arm_pack.py --pack F:/tanitad_scratch/p6fit/fit_pack8.npz --arm-logits <OUT>/C0/fit_logits.npz --out <pack_C0>.npz [--expect-pack-sha256 <sha of the pod's fit_pack.npz>]` -> prints `ZZARMPACK_OK <sha256>`. Only `logits` changes; it refuses missing or extra keys and any other array differing. Partial (`--limit`) files are refused (missing keys). Feed the result to `fit_selectors.py --pack`.

## Time
ESTIMATED about 20 min (card: 0.142 s/set MEASURED by the label job x 8,489 sets; the arms' scorers add a few ms per batch, ESTIMATED). MEASURED here: only CPU tests (20 tests, 7 s). `report.json` records the real per-set time.

## Tests
`CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_arm_fit.py --basetemp F:/tanitad_scratch/p6armfit/pt` : 20 pass (the `*_RED_*` cases are deliberate regressions: moved trunk key, control over tolerance, proposals off, nonzero internal diff, missing dataset row, wrong base sha, other run in the same out, make_arm_pack missing key / extra key / changed non-logit array).

## ⚠️ It needs the GPU to itself, and P7's wait does not protect it (MEASURED 2026-11-09)
P7's `wait_clear` passes after 3 idle polls of `--query-compute-apps`. This tool loads for over a minute before it appears on the GPU, so P7 passed its wait at 19:06:44 and launched at 19:07:47 while this tool was already running. The overlap was stopped by PID after about 20 s of P7's training. To run it while P7 is alive: stop P7 cleanly first, or run it when P7 has ended. Never launch it on the assumption that P7 will wait. `arm_fit2/` on the pod holds the interrupted run's partial state; the same arguments resume it.

## UNVERIFIED (only the pod can say)
1. All pod paths and the venv above (taken from the brief; the chain's defaults differ); `--refe-dir` holding `model.py ckpt_io.py train.py`.
2. That P6's batch-4 bf16 numerics are reproducible on a re-run with the same batch composition. If the reference control refuses on the first batch with logits diff above 1e-3, the cause is composition or TF32/kernel choice; do not loosen blindly: read the printed set key and diff. Resumed label-dump batches (a restart shifts batches) could make a few sets differ.
3. That `TargetBank` items keep `it[0] it[1] it[2] it[8]` as `p6_fan_dump.py` assumes, and that `ds.rows[i]` carries `step`/`rank` (missing keys default to 0, as the dump did).
4. That score output is `[B,64,6]` and the hooks fire under the real model (tested with a stand-in; `arm_logits_pass.Tap` was written for the real one).
5. Peak GPU memory with 4 arm copies: not measured (the three scorers are small).
