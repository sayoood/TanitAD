# LAUNCH_HANDOFF -- P6 read, 2026-11-08. STATE: STOPPED AT STEP 1 (checkpoint identity needs a Master Mind ruling). Nothing pulled, nothing launched, no GPU/lock touched, no P6 content opened.

## What exists (MEASURED today; pod sha256sum run once on both files, read-only; HF listing via model_info)
Pod `/workspace/data/refe_runs/p6_paper_faithful/` (the run dir is flat: there is NO `run/` subfolder on the pod):
| file | bytes | sha256 | on HF `Sayood/tanitad-refe-p6-paper-faithful` |
|---|---|---|---|
| `ckpt_last.pt` | 228,324,493 | a9ed60506b2f80b158c237d696fd848704787f760ee2f4b6ad50344700885f39 | `run/ckpt_last.pt`, same size, LFS sha256 identical |
| `model_final.pt` | 1,288,514,360 | 968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1 | `run/model_final.pt`, same size, LFS sha256 identical |
| `config.json` | 4,577 | (not LFS) | `run/config.json`, same size |
Also on HF: `run/summary.json` (346 B), `run/metrics.jsonl`, `gate/ckpt_last_probe.pt` (a G-EGR probe, not the run), `logs/*`. Upload is COMPLETE for both files; sizes and shas match the pod.

## Why I stopped
The runbook (sec. 1 row 1, sec. 2 `CK=.../ckpt_last.pt`, `selc6_dump.py` docstring) names `ckpt_last.pt`; `summary.json` and your brief name `model_final.pt`. They are different files in different formats (code: `refe/ckpt_io.py` header; `refe/train.py` ~1562 and ~1901):
* `ckpt_last.pt` = PARTIAL resume state (trainable tensors + buffers + optimizer + frozen-trunk fingerprint). Loads for inference only by rebuilding the frozen DINOv3 trunk from published weights.
* `model_final.pt` = FULL state_dict, what `planner.py` always loaded strictly. Written by `train.py` immediately after a final `_save_ckpt()`, so by construction (code reading, NOT checked on the files) both carry the same final trainable weights.
Neither the card, Amendment 1 nor `p6_read/AMENDMENT_1.md` names the file (grep for both names: no hit). `--expect-ckpt-sha` and the chain's G6 gates take whichever file is passed, so the choice changes the pinned sha.

## Decision needed (Master Mind)
Rule which file is "the P6 checkpoint". My recommendation: `model_final.pt` (summary-named, strict full load, no trunk rebuild, 1.29 GB pull). If you choose it, record that in an amendment (the runbook text says `ckpt_last.pt`); I edited no pinned file.

## eval_checkpoint.py findings (read-only, for step 5)
Path is `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/eval_checkpoint.py` (there is no `eval/` at the repo root). It passes no autocast/bf16 flag; it calls `refe_navtest_seam.py`, which also has no autocast/bf16 hit; REFePlanner dtype was NOT inspected, so FP32 for the navtest read is UNVERIFIED (G6-FP32 only gates the dump). Outputs: `D:/Projects/TanitAD/data/refe_navtest/seams/refe_<name>.npz`, `score/refe_<name>/refe_<name>.csv`; `--tokens` needs the full-navtest token list (12,146 tokens; path to confirm). `--name p6final` gives label `refe_p6final`.

## Next steps once ruled
1. Pull `run/<chosen>.pt` + `run/config.json` to `F:/TanitAD_p6read_work/ckpt/`, sha256-verify against the table above (HF_HUB_DISABLE_XET=1, token from Keys.txt in the environment only).
2. `train_logs.txt`: one `ls $DATA/refe_navtrain/metric_cache | sort` on the pod via pd.sh.
3. RUNBOOK steps (a), (b), then the detached WMI chain: eval_checkpoint.py then p6_read.py, markers under `F:/TanitAD_p6read_work/`.

Deleted: nothing. Pre-existing `F:/TanitAD_p6read_scratch` (114 MB, previous agent's) still present.
