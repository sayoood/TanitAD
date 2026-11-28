# HANDOFF: P6 PDM label job (2026-11-09). Written, tested on the dev box (CPU, fakes), never run on the pod.

## What it makes
`p6_labels_chain.sh` dumps P6's fan (command-mode dataset) and PDM-labels it. Three scene lists, all rank-0 scenes of `train_grow_cmd_3c`:
- `heldout` = the 24 logs, expected 3,137 sets (the script refuses another count).
- `fit` = further WHOLE train logs, seeded 20261109, until heldout+fit reach about 8,000 sets. FIT = heldout + fit: never in the arms' training sets; for selection rules, calibrators, stackers.
- `train` = N_TRAIN scenes drawn from the remaining logs (default 60,000).
For the 8,000 FIT sets the dump also saves P6's logits; `fit_pack/fit_pack.npz` holds geometry, logits, PDM labels (one file, no images needed).

## Files to copy to the pod (one directory, e.g. /workspace/p6labels/)
`p6_labels_chain.sh p6_labels_tool.py p6_fan_dump.py` (LF endings; `bash -n` before use). `rl_fan_dump_nav_mode.patch` is for the repo (MM lands), not needed on the pod: the wrapper does the same without editing the pod's refe/.

## Pod paths it reads (defaults; override by env; ALL UNVERIFIED on the pod)
`PKG=/workspace/refe-plan` (refe/ code/ splits/sft_heldout_logs.txt), `MODEL=/workspace/data/refe_p6/model_final.pt` (guess, set it), `TARGETS=/workspace/data/refe_navtrain/train_grow_cmd_3c`, `IMAGES=/workspace/data/navtrain_pixels`, `CALIB=/workspace/data/refe_navtrain/train_grow/calib_table.json`, `OUT=/workspace/data/p6base_labels`, `NAVSIM_PY` = the interpreter of the NAVSIM stack (default `python`), `STACK=/workspace/TanitAD/stack` (becomes PYTHONPATH).

## Commands (P7 paused first; `nvidia-smi` must show no compute process)
```
cd /workspace/p6labels && bash -n p6_labels_chain.sh
export MODEL=... NAVSIM_PY=... N_TRAIN=60000 WORKERS=6        # WORKERS 1..14, N_TRAIN integer or all
setsid nohup bash p6_labels_chain.sh > $OUT/chain.stdout 2>&1 < /dev/null &  echo $! > $OUT/chain.pid
```
Watch with a script, not turns: success line `P6LABELS_ALL_OK` in `$OUT/chain.log` plus file `$OUT/labels.DONE`; failure line `P6LABELS_ABORT stage=.. reason=..` plus `$OUT/labels.FAILED`. Stage markers `$OUT/stages/*.done`; tool logs `$OUT/logs/`. Resume = run the same command again (finished stages and finished dump sets are skipped). `FORCE=1` redoes everything. Stop by explicit PID: `kill $(cat $OUT/chain.pid) $(cat $OUT/label_loop.pid)`.
Output for the launch gate: `$OUT/labels_manifest.json` (`data_expect` counts, sha256 of every set and PDM file, FIT/train log lists and sha256s). The profile's `data_expect` keys are mapped from SFT4R's names; the mapping is UNVERIFIED until the profile exists.

## Time (ESTIMATED: dump 0.066 s/set, WIRING_NOTE section 1; labels 4.49 sets/s at 6 workers, SPEC_P6:187, scaled linearly with workers: an assumption)
| N_TRAIN | sets dumped+labelled | dump (L40) | labels 6 workers | labels 14 workers |
|---|---|---|---|---|
| 30,000 (floor) | 38.1k | 0.7 h | 2.4 h | 1.0 h |
| 60,000 (default) | 68.1k | 1.2 h | 4.2 h | 1.8 h |
| all (about 94.9k) | 103.0k | 1.9 h | 6.4 h | 2.7 h |
Labelling is the long pole. The GPU is needed only until `dump_train.done`; the CPU loop runs on after it. 189,888 is the pre-3C tuple count: this bank has 103,037 rank-0 scenes, and rank-1 twins are not dumped (their PDM label uses the scenario route, SPEC_P6 D7).

## Why 60,000 is enough
An arm draws 400 x 8 x 8 = 25,600 sets of its own seeded permutation of the labelled train sets (`scorer_finetune.py:1556-1557`), so any pool of at least 25,600 labelled sets gives no repeat; 60,000 leaves 2.3x. It also keeps C0 and C0r on partly different sets (expected overlap about 43 %), so C0r's floor includes data draw. The script refuses N_TRAIN below 30,000.

## Defect found: `rl_fan_dump.py:561-563`
Yes, it bypasses `config_for_checkpoint`: the dataset gets nav_mode=goal while the model is built command-mode. `TargetBank` then feeds 4 goal-point floats into the one-hot slot; nothing raises because 2 x n_goal_points = n_commands = 4. The unpatched dump would be wrong for P6. `p6_fan_dump.py` fixes it and refuses a non-command checkpoint; `rl_fan_dump_nav_mode.patch` is the same fix upstream.

## Unverified on the pod
P6 metric cache and bank state (the labeller builds scenarios from the NAVSIM stack: `PT._setup()` unrun); the model path; the `--served-only` call on lean sets; CPU count (6 labelling workers plus 4 image threads); whether `--backbone vitl16` matches P6 (read it from the checkpoint meta); that logits are `[64, 6]`; the real dump on the L40; the 3,137 count on this bank; that the 24 held-out logs are in P6's training data (in-sample for P6, per PREREG section 1). Plan stage loads the bank once for coverage (minutes); `SKIP_PLAN=1` skips it.
