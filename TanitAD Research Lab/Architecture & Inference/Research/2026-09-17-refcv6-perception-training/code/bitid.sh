#!/usr/bin/env bash
# Bit-identity control: the TIP trainer vs the PATCHED trainer, same seed,
# both perception weights at their 0.0 default.
set -u
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUTROOT="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/<session-id>/scratchpad/bitid"
rm -rf "$OUTROOT"; mkdir -p "$OUTROOT"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6

run () {   # $1 = worktree, $2 = tag
  local WT="$1" TAG="$2"
  export PYTHONPATH="$WT/stack;$WT;$WT/taniteval"
  ( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
      --arm hier --out "$OUTROOT/$TAG" --smoke --steps 3 \
      --device cpu --synth-episodes 4 --seed 0 --log-every 1 \
  ) > "$OUTROOT/$TAG.log" 2>&1
  echo "$TAG exit=$?"
}

run "C:/Users/Admin/tanitad-wt-perctrain-tip" tip
run "C:/Users/Admin/tanitad-wt-perctrain"     patched
# ⭐ THE DISCRIMINATING CONTROL: the SAME trainer, run TWICE. Any field that
# differs HERE is wall-clock, not code -- so it cannot be evidence about the
# patch. Without it, "metrics.jsonl differs" is unreadable.
run "C:/Users/Admin/tanitad-wt-perctrain-tip" tip2
