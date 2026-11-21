#!/usr/bin/env bash
# pod_tokprobe_run.sh -- card PREREG_V8F_PROOF_ADD27 (TOKPROBE), the ONE pod job. Runs ON THE POD, started by the lead. DRY=1 prints the commands only.
#   1. refuses under 20 GB free in the container (memory.max minus the anon line of memory.stat) and over a finished marker
#   2. tok_probe.py train   : four probe arms (TK16_s0 TK16_s1 TK64_s0 TK64_s1), one process, the GPU, 19,164 steps (--expect-steps)
#   3. tok_probe.py extract : the four arms -> $STATES/<ARM>_{eval,train}.npz(+.json) and <ARM>_{eval,train}_lead.npz(+.json,+.head.npz)   (the folder the pull chain reads)
#   4. control p3 : tok_probe.py train --shuffle-closing-labels --steps 2000 (arm TK16SH_s0), then its extract into the same folder
#   5. writes $OUT/TOKPROBE_DONE.txt (or $OUT/TOKPROBE_STOP.txt with the reason)
# Files needed on the pod first (push by the usual route, check the sha256 on both sides): $PB/tok_probe.py. The tree $TREE is the existing /workspace/v8proof/tree_leadsup.
# Tier T0 (a diagnostic on logged frames), never driving performance. Nothing of the model is trained or touched.
set -u
PY=${PY:-/workspace/venv/bin/python}
TREE=${TREE:-/workspace/v8proof/tree_leadsup}
PB=${PB:-/workspace/v8proof/probe}
OUT=${OUT:-/workspace/v8proof/runs/tokprobe}
STATES=${STATES:-$PB/states}
CACHE=${CACHE:-/workspace/v8proof/data/cache_pod}
STEPS_EXPECT=${STEPS_EXPECT:-19164}
P3_STEPS=${P3_STEPS:-2000}
MIN_FREE_GB=${MIN_FREE_GB:-20}
DRY=${DRY:-0}
DEVICE=${DEVICE:-cuda}                  # dev-box test only: DEVICE=cpu
EVAL_RANGE=${EVAL_RANGE:-0:141}        # the card: eval clips 0:141, train probe clips 600:1800 (dev-box test only: smaller)
TRAIN_RANGE=${TRAIN_RANGE:-600:1800}
TRAINER_ARGV=${TRAINER_ARGV:-}         # dev-box test only: a synthetic trainer argv (the default is the argv record of LSD_s0 with pod paths)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ARMS="TK16_s0 TK16_s1 TK64_s0 TK64_s1"
TA=(); [ -n "$TRAINER_ARGV" ] && TA=(--trainer-argv "$TRAINER_ARGV")
EXP=(--expect-steps "$STEPS_EXPECT"); [ "$STEPS_EXPECT" = 0 ] && EXP=()
TP=$PB/tok_probe.py

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] tokprobe: $*"; }
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi; say "run: $*"; "$@"; }
stop() {  # reason
  say "STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$OUT"; printf 'TOKPROBE_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$OUT/TOKPROBE_STOP.txt"; fi
  exit 1
}

if [ "$DRY" = 1 ]; then
  say "DRY=1: commands only (no check, no run)"
else
  [ -f "$OUT/TOKPROBE_DONE.txt" ] && { say "REFUSED: $OUT/TOKPROBE_DONE.txt exists (a finished marker); nothing is run"; exit 3; }
  [ -f "$TP" ] || stop "missing $TP"
  [ -d "$TREE/stack" ] || stop "missing tree $TREE"
  [ -d "$CACHE/train/clips" ] && [ -d "$CACHE/eval/clips" ] || stop "missing cache $CACHE"
  mkdir -p "$OUT" "$STATES"
  [ -f "$OUT/TOKPROBE_STOP.txt" ] && mv "$OUT/TOKPROBE_STOP.txt" "$OUT/TOKPROBE_STOP.prev.$(date -u +%H%M%S).txt"
  # free memory of the CONTAINER: memory.max minus the anon line of memory.stat (free shows the host, not the cap)
  maxb=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
  anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  if [ "$maxb" = max ] || [ -z "$maxb" ]; then
    freeb=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
    say "memory.max = max (no cap); MemAvailable used"
  else
    [ -n "$anon" ] || stop "cannot read anon from /sys/fs/cgroup/memory.stat"
    freeb=$((maxb - anon))
  fi
  freeb=${freeb:-0}
  freegb=$((freeb / 1073741824))
  say "memory.max=$maxb anon=${anon:-na} free=${freegb} GB (floor ${MIN_FREE_GB} GB)"
  [ "$freegb" -ge "$MIN_FREE_GB" ] || stop "container free memory ${freegb} GB is under ${MIN_FREE_GB} GB"
  if [ "$DEVICE" = cuda ]; then nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || stop "no GPU"; fi
fi

# ---- 2. train (one process, four arms, the same batches)
run "$PY" "$TP" train --tree "$TREE" --out "$OUT/train" "${EXP[@]}" "${TA[@]}" --device "$DEVICE"
rc=$?
if [ "$DRY" != 1 ]; then
  [ $rc -eq 0 ] || [ $rc -eq 2 ] || stop "train exited $rc"
  [ -f "$OUT/train/TRAIN_DONE.json" ] || stop "train wrote no TRAIN_DONE.json (rc $rc)"
  P5_TRAIN=$([ $rc -eq 0 ] && echo pass || echo FAIL)
else
  P5_TRAIN=dry
fi

# ---- 3. extract (the pull chain reads $STATES)
for A in $ARMS; do
  run "$PY" "$TP" extract --tree "$TREE" --run-dir "$OUT/train" --arm "$A" --cache "$CACHE" --states-dir "$STATES" --eval-range "$EVAL_RANGE" --train-range "$TRAIN_RANGE" --device "$DEVICE" || stop "extract $A failed"
done

# ---- 4. control p3: shuffled closing-speed labels, 2,000 steps, TK16 seed 0
run "$PY" "$TP" train --tree "$TREE" --out "$OUT/p3" --steps "$P3_STEPS" --shuffle-closing-labels "${TA[@]}" --device "$DEVICE"
rc3=$?
if [ "$DRY" != 1 ]; then
  [ $rc3 -eq 0 ] || [ $rc3 -eq 2 ] || stop "p3 train exited $rc3"
  P5_P3=$([ $rc3 -eq 0 ] && echo pass || echo FAIL)
else
  P5_P3=dry
fi
run "$PY" "$TP" extract --tree "$TREE" --run-dir "$OUT/p3" --arm TK16SH_s0 --cache "$CACHE" --states-dir "$STATES" --eval-range "$EVAL_RANGE" --train-range "$TRAIN_RANGE" --device "$DEVICE" || stop "extract TK16SH_s0 failed"

# ---- 4b. sensitivity pass of note ADD27_N1 (SENS=1): the same four arms with NO token normalisation, into $OUT/raw and $STATES_RAW.
#          It names nothing, runs AFTER the done marker of the main pass, and its failure does not touch that marker.
SENS=${SENS:-0}
STATES_RAW=${STATES_RAW:-$PB/states_raw}
sens_pass() {
  [ "$DRY" = 1 ] || mkdir -p "$STATES_RAW"
  run "$PY" "$TP" train --tree "$TREE" --out "$OUT/raw" "${EXP[@]}" "${TA[@]}" --device "$DEVICE" --input-norm none
  local rcs=$? note
  [ "$DRY" = 1 ] && return 0
  if { [ $rcs -eq 0 ] || [ $rcs -eq 2 ]; } && [ -f "$OUT/raw/TRAIN_DONE.json" ]; then
    note="train_rc=$rcs"
    for A in $ARMS; do
      run "$PY" "$TP" extract --tree "$TREE" --run-dir "$OUT/raw" --arm "$A" --cache "$CACHE" --states-dir "$STATES_RAW" --eval-range "$EVAL_RANGE" --train-range "$TRAIN_RANGE" --device "$DEVICE" || note="$note extract_${A}_FAILED"
    done
    case "$note" in *FAILED*) printf 'TOKPROBE_SENS_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$note" > "$OUT/TOKPROBE_SENS_STOP.txt";;
      *) { echo "TOKPROBE_SENS_DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ') $note input_norm=none"; for A in $ARMS; do echo "ckpt $A $(sha256sum "$OUT/raw/$A/ckpt_last.pt" | cut -c1-64)"; done; } > "$OUT/TOKPROBE_SENS_DONE.txt";; esac
  else
    printf 'TOKPROBE_SENS_STOP %s\ntrain rc %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$rcs" > "$OUT/TOKPROBE_SENS_STOP.txt"
  fi
  say "sensitivity pass: ${note:-train rc $rcs}"
}

# ---- 5. the done marker, last
if [ "$DRY" = 1 ]; then [ "$SENS" = 1 ] && sens_pass; say "DRY=1: done (nothing written)"; exit 0; fi
{
  echo "TOKPROBE_DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "p5_main_run=$P5_TRAIN (control p5 of the card, per arm in $OUT/train/TRAIN_DONE.json)  p5_p3_run=$P5_P3"
  echo "tok_probe.py sha256: $(sha256sum "$TP" | cut -c1-64)"
  echo "states folder: $STATES (per arm: <ARM>_{eval,train}.npz(+.json), <ARM>_{eval,train}_lead.npz(+.json,+.head.npz))"
  for A in TK16_s0 TK16_s1 TK64_s0 TK64_s1 TK16SH_s0; do
    d=$OUT/train; [ "$A" = TK16SH_s0 ] && d=$OUT/p3
    echo "ckpt $A $(sha256sum "$d/$A/ckpt_last.pt" | cut -c1-64) $d/$A/ckpt_last.pt"
  done
  echo "train logs: $OUT/train/<ARM>/train_log.jsonl  $OUT/p3/TK16SH_s0/train_log.jsonl; meta: $OUT/train/train_meta.json $OUT/p3/train_meta.json"
} > "$OUT/TOKPROBE_DONE.txt"
say "DONE: $OUT/TOKPROBE_DONE.txt"
[ "$SENS" = 1 ] && sens_pass
exit 0
