#!/usr/bin/env bash
# BIT-IDENTITY CONTROL: the TIP trainer vs the PATCHED trainer, same seed,
# with the refcv6 §4/§5 flags ABSENT (their 0.0 / off defaults).
#
# ⛔ THE CLAIM BEING TESTED is *"a recipe that does not pass --tac-decoder-v6 /
# --w-tac-v6 / --max-speed-input-v6 is bit-identical to the pre-wiring
# trainer"*. That is what lets this land beside a live recipe without a
# rebalancing decision (which is the PI's, not this package's).
#
# ⛔ NO SESSION PATH IS BAKED IN — the caller supplies the scratch root, because
# `tests/test_refcv6_no_session_paths.py` forbids one in a repo artifact:
#
#   TAC_OUT=<scratch dir> TAC_WT_TIP=<tip worktree> TAC_WT=<patched worktree> \
#     bash bitid.sh
set -u
OUTROOT="${TAC_OUT:?set TAC_OUT to a scratch output dir}"
WT_TIP="${TAC_WT_TIP:?set TAC_WT_TIP to the TIP worktree}"
WT_PATCH="${TAC_WT:?set TAC_WT to the PATCHED worktree}"
PY="${TAC_PY:?set TAC_PY to the venv python}"

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
  # ⛔ `$?` here is the subshell's, NOT a pipeline's -- there is no pipe on
  # purpose. `cmd | tail` would report tail's status and hide a failure.
  echo "$TAG exit=$?"
}

run "$WT_TIP"   tip
run "$WT_PATCH" patched
# ⭐ THE DISCRIMINATING CONTROL: the SAME trainer, run TWICE. Any field that
# differs HERE is wall-clock, not code -- so it cannot be evidence about the
# patch. Without it, "metrics.jsonl differs" is unreadable.
run "$WT_TIP"   tip2
