#!/usr/bin/env bash
# W7 2026-09-20 — the navhard refcv4b run, launched DETACHED through W1's suite CLI.
#
# ⛔ NEVER CHAIN THE LAUNCHER BEHIND A PIPE (auto-memory "never chain the lander behind a pipe",
# and CLAUDE.md "$? after a pipeline is the LAST element's status"): the exit status is written to
# a FILE and read back, and the admissible evidence that the run started is BENCH_RUN_DIR in the
# log, never an exit code.
#
# Usage: launch_run.sh <arms> <cpu-threads> <device>
set -u
ARMS="${1:-A1}"
THREADS="${2:-16}"
DEVICE="${3:-auto}"
REPO="D:/Projects/TanitAD"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BANK="C:/Users/Admin/tanitad-caches/navsim-frames-navhard-20260920"
CKPT="D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt"
MD5="99b573e8277d94a5e3bfbf630cb4d751"
PKG="$REPO/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b"
LOG="$PKG/raw/bench_launch.log"

# the bank must be FINALISED (BUILD.json + content assertion) or the bridge refuses the frame tag
if [ ! -f "$BANK/BUILD.json" ]; then
  echo "REFUSED: $BANK/BUILD.json is missing — run code/finalize_bank.py first" | tee -a "$LOG"
  exit 2
fi

# ⛔ THE EDITABLE `tanitad` INSTALL IN THIS VENV STILL POINTS AT THE DEAD G: MOUNT.
# MEASURED 2026-09-20: the first launch died in 4.3 s with StackShadowError — sys.path carried
# 'G:\Meine Ablage\...\stack' (intent installed:find_spec) ahead of the pinned root. ⭐ The guard
# FAILING LOUD is the correct outcome and the reason a plausible wrong number was not published:
# CLAUDE.md records exactly this trap ("the venv's EDITABLE `tanitad` install points at the G:
# mount"), and a run that silently imported a pre-move stack would have been unfalsifiable.
# ⛔ Never downgrade it with TANITEVAL_STACK_GUARD=warn — name the tree instead.
export TANITEVAL_STACK_OVERRIDE="$REPO/stack"

cd "$REPO/taniteval" || exit 2
nohup env PYTHONIOENCODING=utf-8 OMP_NUM_THREADS="$THREADS" \
  TANITEVAL_STACK_OVERRIDE="$REPO/stack" "$PY" -m taniteval.bench navsim_v2 \
  --ckpt "$CKPT" \
  --split navhard_two_stage \
  --arms "$ARMS" \
  --device "$DEVICE" \
  --registry-key refcv4b-b1-v72-40k \
  --ckpt-md5 "$MD5" \
  --frame-bank "$BANK" \
  --cpu-threads "$THREADS" \
  >> "$LOG" 2>&1 &
echo "LAUNCH_PID=$!"
