#!/usr/bin/env bash
# BIT-IDENTITY AGAINST THE **NEW** BASELINE (PI RULING 2026-09-17).
#
# ⛔ WHY THIS EXISTS RATHER THAN A CITATION. Every bit-identity proof banked
# before tonight was taken against a forward that did NOT carry the BEV encoder.
# Those proofs are not void -- they are about a DIFFERENT forward. This run
# re-establishes the property against the tip the ruling landed on (ee1635a):
# with no perception weight and no behaviour decoder, the patched trainer must
# produce a bitwise-identical checkpoint and a byte-identical metrics.jsonl.
#
# ⭐ THE CHECKER IS NOT COPIED. It is the one already banked at
#   .../2026-09-17-refcv6-perception-training/code/bitid_check.py
# invoked by path, so this package cannot drift from it. It asserts the SHAPE of
# every comparison first (a missing file, an empty digest or a zero tensor count
# is INCONCLUSIVE, never MATCH) and carries its own MUTATION CONTROL: one bit of
# byte 0 of EVERY tensor is flipped and the predicate must see all of them. A
# comparison that cannot fail proves nothing.
#
# ⚠️ `tip2` is the DISCRIMINATING CONTROL: the SAME trainer run twice. Any field
# that differs there is wall-clock, not code, so it cannot be evidence about the
# patch.
set -u
# ⛔ NO ABSOLUTE HOME PATH AND NO SESSION-ID PATH IN A DELIVERABLE
# (tests/test_refcv6_no_session_paths.py, which caught this file on its first
# run). Every location is an env var with a loud failure, never a literal.
PY="${TAC_PY:?set TAC_PY to the venv python}"
OUTROOT="${BITID_OUT:?set BITID_OUT to a scratch directory}"
CHECK="${BITID_CHECK:?set BITID_CHECK to bitid_check.py of 2026-09-17-refcv6-perception-training}"
TIP_WT="${BITID_TIP_WT:?set BITID_TIP_WT}"
PATCHED_WT="${BITID_PATCHED_WT:?set BITID_PATCHED_WT}"

rm -rf "$OUTROOT"; mkdir -p "$OUTROOT"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=""          # the GPU is held by an RL arm
export OMP_NUM_THREADS=6                # torch spawns ~113 threads per process

run () {   # $1 = worktree, $2 = tag
  local WT="$1" TAG="$2"
  export PYTHONPATH="$WT/stack;$WT;$WT/taniteval"
  ( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
      --arm hier --out "$OUTROOT/$TAG" --smoke --steps 3 \
      --device cpu --synth-episodes 4 --seed 0 --log-every 1 \
  ) > "$OUTROOT/$TAG.log" 2>&1
  # ⛔ NEVER `$?` THROUGH A PIPE -- it reports the LAST element's status.
  echo "$TAG exit=$?"
}

run "$TIP_WT"     tip
run "$PATCHED_WT" patched
run "$TIP_WT"     tip2

"$PY" "$CHECK" "$OUTROOT"
