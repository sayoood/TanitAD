# AMENDMENT 2 to the P6 read (2026-11-08 19:01Z, Master Mind): which file is the P6 checkpoint

**Written before any P6 value was read.** Nothing was pulled or opened; only sizes and sha256 values were read.

**Ruling.** The P6 checkpoint is `model_final.pt`, sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`, 1,288,514,360 bytes, on HF private repo `Sayood/tanitad-refe-p6-paper-faithful` at `run/model_final.pt`.

**Why.** It is the file `summary.json` binds with `done: true`, and the full state_dict the planner loads strictly. `ckpt_last.pt` (228,324,493 bytes) is the partial resume state with the optimizer, which loads for inference only by rebuilding the frozen trunk. The card and Amendment 1 name neither file; the runbook's `ckpt_last.pt` is corrected by this text. No pinned file changes, because `--ckpt` and `--expect-ckpt-sha` are arguments. The same file is used for the navtest read and the dump, so gates G6-1 and G6-3 compare like with like.

**Navtest read.** `eval/eval_checkpoint.py` and `eval/refe_navtest_seam.py` hardcode paths on a D: drive this machine no longer has. They are not pinned by this read. A scratch copy with only the root paths changed runs the navtest read; the diff is recorded with the output. Neither script nor the planner contains autocast, bf16 or half(), by search. G6-1 and G6-3 compare that seam with the FP32 dump, so a precision or path defect fails a gate instead of passing silently.
