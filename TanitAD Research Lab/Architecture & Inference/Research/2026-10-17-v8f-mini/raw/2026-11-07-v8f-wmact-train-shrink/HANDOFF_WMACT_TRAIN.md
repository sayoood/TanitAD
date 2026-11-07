# HANDOFF_WMACT_TRAIN (finished; nothing pending)

Card `../PREREG_V8F_WMACT.md` (frozen prefix sha256 `d21b14cc...` re-checked, card not edited). Tier T0, CPU only. All four controls read their known values on TRAIN, so the read is valid.

## Result (details: `RESULT_WMACT_TRAIN.md`, raw `RESULT_WMACT_TRAIN.json`)
* **L1 DOES NOT FIT.** TRAIN L1 k1 SKILL is -0.0010 [-0.0141, +0.0122] (s0) and +0.0063 [-0.0078, +0.0196] (s1): not above 0 and separated at either seed.
* n: TRAIN 1,833 windows / 141 clips; eval 1,976 windows / 141 clips. Wall time: stream and latents 271 s, statistics 34 s.

## State
* Sample: `train_stream/clips141.json` (clip list, md5, bytes; clips sha256 `e9e0e2c4...`), `train_stream/clips141.txt` (sha256 `0e61bd24...`). Excluded 540:600 of the sha12 order; digests equal M2b's `summary.json`.
* `latents_train.pt` (same layout as `latents.pt`, plus `clips`, `ctrl4`, `starts`). `latents_train_partial.pt` is a mid-run checkpoint (140 of 141 clips); safe to remove by the PI.
* Scratch: `train_stream/cache/train/clips/_current.pt` holds the LAST streamed clip only (one file, overwritten each time).

## Re-run (Git bash, this directory)
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
export WMACT_CODE_ROOT="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tip_1246c8dd"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY select_train_clips.py                 # needs train_stream/box2_meta/manifest.jsonl (sftp from box 2, read only)
$PY -u collect_latents_train.py           # about 5 min; resumable
$PY -u run_wmact_train.py                 # about 35 s
$PY report_wmact_train.py                 # checks the controls first, regenerates the generated part of RESULT_WMACT_TRAIN.md
$PY -m pytest -q test_train_item_shim.py  # shim equals the real item on eval windows
```
`report_wmact_train.py` overwrites `RESULT_WMACT_TRAIN.md`; the "Reading" section at its end was added by hand.

## For the Master Mind
* Suggested card log line (not written by this agent): "TRAIN read done, `theory/wmact/RESULT_WMACT_TRAIN.md`: controls valid; L1 k1 SKILL -0.0010 / +0.0063, not separated: L1 DOES NOT FIT; L2 and L3 SKILL and ACTION-USE are larger on TRAIN than on eval."
* Open lever (card does not commit one for "does not fit"): the cheapest zero-training read is the L1 training-loss term against its own copy baseline in M2b's `train_log`.

## Manifest (E: artifacts, `repo`-external; nothing staged, committed or pushed; each exists in one place)
| file | where |
|---|---|
| select_train_clips.py, collect_latents_train.py, run_wmact_train.py, report_wmact_train.py, test_train_item_shim.py | `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wmact/` |
| latents_train.pt, latents_train_partial.pt, collect_latents_train.log, run_wmact_train.run1.log | same |
| RESULT_WMACT_TRAIN.json, RESULT_WMACT_TRAIN.md, HANDOFF_WMACT_TRAIN.md | same |
| train_stream/ (clips141.json / .txt, box2_meta/, cache/train/manifest.jsonl + meta.json + the one scratch clip) | same |
