# G-MAP-OVERFIT /2-banded binding harness argv for closure_run.py --binding (one token per line; `#` lines are comments).
# refcv9 M-GATE (E1-MAP-OWN-RECORD): the BANDED map head's OWN overfit record = the SAME pre-registered protocol as refcv8_bind_map.args.tpl
# (the A18 spec: 16 fixed TRAIN frames, 3,000 steps, AdamW lr 1e-3 -> cosine from 2,700, batch 4, seed 0, gated x 0-20 m, 1 ms time guard)
# run on tanitad/models/map_banded.py via `map_hires_overfit.py --banded`. Inputs/data roots are refcv8_bind_map.args.tpl's, unchanged.
# Differences from the /1 template, all of them forced by the banded head: --banded; the A18 spec (the arms' must-fail rows live in the
# harness); --near-lift-m 10 (the banded near band); no --class-weights (the banded loss has none); --arms healthy,lane_w0,s8_zeros.
# {CODE} {COMMIT} {ARGV} {ARGV_SHA} {OUT} {INPUTS} are filled by `r9e1_bind.sh`.
stack/scripts/map_hires_overfit.py
--banded
--spec
{INPUTS}/gmo_spec_A18.json
--frameset
{INPUTS}/gmo_frameset.json
--decision-rule
prior_corrected
--near-lift-m
10
--near-refine-blocks
1
--v2-cache
/home/nvidia/data/refcv6-b1-416x1024-train
--gt-root
/home/nvidia/data/sam3_gt_v3
--extrinsics
/home/nvidia/data/refcv6_train_eval139_extrinsics.json
--cam-ts-dir
/home/nvidia/data/_b1stage416/r0/camera_front_wide
--launch-commit
{COMMIT}
--launch-argv-sha256
{ARGV_SHA}
--arms
healthy,lane_w0,s8_zeros
--out
{OUT}
--device
cuda
# data roots whose OPENED files the closure records (identical to the /1 template)
DATA-ROOT /home/nvidia/data
DATA-ROOT {INPUTS}
