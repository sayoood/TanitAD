#!/usr/bin/env bash
# THE REPLICATE ARM — without it the A/B's intervals answer the wrong question.
#
# ⛔ H-ESTIM-SEED-1 (CLAUDE.md): the paired episode-cluster bootstrap resamples
# EPISODES with the models held fixed, so it estimates "would another draw of
# episodes say this?" and never "would another RUN say this?". MEASURED on the
# v7-tiny rig: an arm with identical flags and identical seed produced
# "separated" differences on 3 of 18 metrics — a ~17 % false-positive rate for
# `separated`. A separated CI is therefore NECESSARY, NOT SUFFICIENT.
#
# ⭐ AND HERE THE RISK IS HIGHER THAN USUAL, because refav1's planner is
# STOCHASTIC: `icem_plan` draws coloured noise from `PlanConfig.seed`. So two
# runs of the SAME arm differ by construction. `--plan-seed 1` against the A/B's
# `--plan-seed 0` measures exactly that run-to-run floor, on the same windows.
#
# Read the A/B like this:
#     |prior050 - argmax|  vs  |argmax@seed1 - argmax@seed0|
# If the lever's delta is not clearly larger than the replicate's, the lever has
# not been shown to move anything, however tight its own interval is.
#
# ⚠️ RUN THIS ONLY AFTER run_ab.sh HAS FINISHED — one GPU, and a second job
# would contend for it. Do NOT edit run_ab.sh while it is live (bash reads a
# script lazily by byte offset; an in-place edit can make a running shell
# execute garbage from mid-line).
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_drive/ab"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6

LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
NEP=${NEP:-30}

echo "=== ARM argmax_seed1 (the replicate)  $(date -u +%FT%TZ) ==="
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache C:/Users/Admin/refav1_eval_full/fp8 \
  --episodes C:/Users/Admin/refav1_eval_full/eps \
  --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n "$NEP" --window-stride 40 --no-navshuf \
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,0.0,64.29715042415070 \
  --plan-seed 1 \
  --dump-dir "$OUT/dump_argmax_seed1" --out "$OUT/rec_argmax_seed1.json" \
  --arm refav1-21109-ab-argmax-seed1 \
  >> "$OUT/argmax_seed1.log" 2>&1
echo "    exit=$? $(date -u +%FT%TZ)"
echo "REPLICATE_DONE $(date -u +%FT%TZ)"
