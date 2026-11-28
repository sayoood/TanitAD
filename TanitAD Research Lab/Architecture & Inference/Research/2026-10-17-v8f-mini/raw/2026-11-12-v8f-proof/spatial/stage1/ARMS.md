# ARMS: PREREG_V8F_PROOF_ADD12 stage 1, exact argv (2026-11-13)

Tree: commit `3e264bbf` of the mirror plus `SPATIAL.patch` (same folder). Base argv = the HIERREP HIER argv of `pod/POD_TRAIN_RECIPE.md` section 2 (arm M2b, from scratch, 19,164 steps from `--steps-mult 4`).
Pod roots are placeholders the lead sets: `$CACHE` (train token cache, holds `native_tracks.pt`), `$V9` (labels dir), `$RUN` (join and p10), `$OUT`. Nothing here is a box path.

## Common prefix (every arm, seed S in {0, 1})

```
--arm M2b --seed S --grid pool4x10
--cache-dir $CACHE
--v9-train $V9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759
--lead-train $V9/lead_train4572.npz
--speed-train $V9/refcv6_speed_max_v8_train.jsonl
--join-train $RUN/join_train.npz
--p10 $RUN/p10/p10_prior.pt
--kin-mode yaw --tracks $CACHE/native_tracks.pt
--exclude 540:600 --batch 64 --steps-mult 4 --device cuda
```

## The four arms (lever flags appended, `--out` differs)

| arm | seed | out | lever flags |
|---|---|---|---|
| G | 0 | `$OUT/M2b_G_s0` | `--wm-state grid --wm-cell-dim 16` |
| G | 1 | `$OUT/M2b_G_s1` | `--wm-state grid --wm-cell-dim 16` |
| W | 0 | `$OUT/M2b_W_s0` | `--wm-state mean --wm-dim 640` |
| W | 1 | `$OUT/M2b_W_s1` | `--wm-state mean --wm-dim 640` |

Full line, arm G seed 0:
```
python stack/scripts/v8f_mini_train.py --arm M2b --seed 0 --grid pool4x10 --out $OUT/M2b_G_s0 --cache-dir $CACHE --v9-train $V9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train $V9/lead_train4572.npz --speed-train $V9/refcv6_speed_max_v8_train.jsonl --join-train $RUN/join_train.npz --p10 $RUN/p10/p10_prior.pt --kin-mode yaw --tracks $CACHE/native_tracks.pt --exclude 540:600 --batch 64 --steps-mult 4 --device cuda --wm-state grid --wm-cell-dim 16
```
Arm W seed 1: same with `--seed 1 --out $OUT/M2b_W_s1` and `--wm-state mean --wm-dim 640` in place of the G flags. The baseline (HIER) argv is the prefix with no lever flag, which is the default `--wm-state mean --wm-dim 128`, today's model bit for bit (check c1).
Fallback named in the card (G8, only if G seed 0 diverges or c4 flags it): the G line with `--wm-cell-dim 8` (state 320, predictor 256), out `M2b_G8_s0`.

What the flags build (MEASURED on CPU, `checks/selfchecks/`): G level-1 state 640 (40 cells x 16), level-1 predictor 256; W level-1 state 640 (mean projected to 640), predictor 256; levels 2 and 3 are 128 wide in both; baseline 128 / 128.
Trainable parameters at d 128: base 7,214,495 (equals the recipe's HIER count), G 9,091,503, W 9,251,871.

## Gate procedure (per arm, per seed; no training without it)

1. Environment: `PYTHONPATH=<tree>/stack:<tree>/taniteval` with the pod's separator, `tanitad.__file__` asserted inside the tree, tree = commit `3e264bbf` plus `SPATIAL.patch` (sha256 of the patched `v8f_mini.py`, `v8f_mini_train.py`, `v8f_beta.py` recorded in the gate token's key files).
2. Selfcheck on the EXACT argv: the line above with `--selfcheck-only` added (G-DVB, G-LIVE, probes), the chain's existing gate procedure of the HIERREP queue (`gate_token.json` bound to commit, argv sha256, data sha256). The real train line must equal the gated line (the chain asserts the argv hash).
3. Required in `selfcheck.json`: `declared_vs_built` PASS; `live.pass` true; `levers.wm_state` equals the declared value; `levers.wm_l1_dim` 640 for G and W; `levers.wm_l1_pred_width` 256; `levers.wm_upper_dims` [128, 128]; `live.groups_grad.wm` above 0.
4. A grid or width flag that the gate reports as declared and not built is a launch blocker, not a footnote.
5. Resume and fine-tune: `--init-ckpt` refuses a checkpoint whose wm_state or wm_dim differs from the argv (checked, `checks/c_roundtrip.json`).
6. Reading the trained checkpoints: `v8f_t1_eval.py` builds a grid checkpoint unchanged (strict load, `checks/c_roundtrip.json`). The rung-beta planner REFUSES a grid model by design.

Selfcheck of the arm shape on the dev box was run with `--synthetic --device cpu --exclude 1:3` (the synthetic world has 6 clips; the real exclusion `540:600` drops 0 of them and the gate correctly refuses that), all other flags as above.
