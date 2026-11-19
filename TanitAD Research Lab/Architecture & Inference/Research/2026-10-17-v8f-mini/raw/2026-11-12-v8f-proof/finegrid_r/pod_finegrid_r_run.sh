#!/usr/bin/env bash
# pod_finegrid_r_run.sh -- card PREREG_V8F_PROOF_ADD43 (FINEGRID-R). Runs ON THE POD under nohup, started by the lead. DRY=1 prints every command only.
#   0. refuses over a finished marker; a second start is refused by a marker created with noclobber ($OUT/FINEGRID_R_STARTED.txt). Nothing is removed or moved.
#   1. g2 on the CPU (as in the pilot) -> $OUT/G2_GRIDS.json
#   2. train process A: --grid pool4x10, 16 arms TK16_sK TK64_sK (K = 2..9), --sampler-chunk-like pool8x20 (the same windows in the same order as process B, control r1)   -> $OUT/train_A
#      extract of the 16 arms (EVAL split only)
#   3. train process B: --grid pool8x20, 8 arms F16_sK                                                                                            -> $OUT/train_B
#      extract of the 8 arms (EVAL split only)
#   4. control r1: the batch logs of A and B compared (CPU) -> $OUT/R1_SAMPLER.json
#   5. control arms (p3, 2,000 steps, permuted closing labels): TK16SH_s2 (A side, chunk like pool8x20) and F16SH_s2, each with its extract
#   6. $OUT/FINEGRID_R_DONE.txt, last line of the log: ZZ FINEGRID-R DONE (or ZZ FINEGRID-R STOP: <reason> with $OUT/FINEGRID_R_STOP.txt)
# ONE GPU process at a time. Before EVERY GPU step the slot wait of the pilot script: fewer than 3 GPU compute processes (nvidia-smi --query-compute-apps=pid) AND more than 28 GB free
# (memory.max minus the anon line of memory.stat), two probes 45 s apart, poll 60 s, at most 10 h, then STOP. No process-name test, no rm, no mv. Tier T0.
# Needed on the pod first (compare sha256 on both sides): $PB/tok_probe_gr.py. Pilot files in $PB are not touched (new names, new folders).
set -u
PY=${PY:-/workspace/venv/bin/python}
TREE=${TREE:-/workspace/v8proof/tree_leadsup}
PB=${PB:-/workspace/v8proof/finegrid}
OUT=${OUT:-/workspace/v8proof/runs/finegrid_r}
STATES=${STATES:-/workspace/v8proof/finegrid_r/states}
CACHE=${CACHE:-/workspace/v8proof/data/cache_pod}
SEEDS=${SEEDS:-"2 3 4 5 6 7 8 9"}
STEPS_EXPECT=${STEPS_EXPECT:-19164}
P3_STEPS=${P3_STEPS:-2000}
MIN_FREE_GB=${MIN_FREE_GB:-28}
MAX_GPU_JOBS=${MAX_GPU_JOBS:-3}
PROBE_GAP_S=${PROBE_GAP_S:-45}
POLL_S=${POLL_S:-60}
MAX_WAIT_S=${MAX_WAIT_S:-36000}
CG_MAX=${CG_MAX:-/sys/fs/cgroup/memory.max}
CG_STAT=${CG_STAT:-/sys/fs/cgroup/memory.stat}
DRY=${DRY:-0}
DEVICE=${DEVICE:-cuda}
EVAL_RANGE=${EVAL_RANGE:-0:141}
G2_ARGS=${G2_ARGS:-}
TRAINER_ARGV=${TRAINER_ARGV:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
TA=(); [ -n "$TRAINER_ARGV" ] && TA=(--trainer-argv "$TRAINER_ARGV")
EXP=(--expect-steps "$STEPS_EXPECT"); [ "$STEPS_EXPECT" = 0 ] && EXP=()
TP=$PB/tok_probe_gr.py
LOG=$OUT/finegrid_r.log
S0=${SEEDS%% *}
ARMS_A=""; ARMS_B=""; LIST_A=()
for k in $SEEDS; do ARMS_A="${ARMS_A:+$ARMS_A,}TK16_s$k,TK64_s$k"; ARMS_B="${ARMS_B:+$ARMS_B,}F16_s$k"; LIST_A+=("TK16_s$k" "TK64_s$k"); done
LIST_B=(); for k in $SEEDS; do LIST_B+=("F16_s$k"); done
SH_A=TK16SH_s$S0; SH_B=F16SH_s$S0

say() {
  local line="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid_r: $*"
  echo "$line"
  if [ "$DRY" != 1 ] && [ -d "$OUT" ]; then echo "$line" >> "$LOG"; fi
}
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi; say "run: $*"; "$@"; }
stop() {  # reason
  say "STOP: $1"
  say "ZZ FINEGRID-R STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$OUT"; printf 'FINEGRID_R_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$OUT/FINEGRID_R_STOP.txt"; fi
  exit 1
}
gpu_jobs() {
  local o rc
  o=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo 99; return; fi
  printf '%s\n' "$o" | grep -c .
}
free_gb() {
  local maxb anon freeb
  maxb=$(cat "$CG_MAX" 2>/dev/null || echo max)
  anon=$(awk '$1=="anon"{print $2}' "$CG_STAT" 2>/dev/null)
  if [ "$maxb" = max ] || [ -z "$maxb" ]; then
    freeb=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo 2>/dev/null)
  elif [ -z "$anon" ]; then
    echo 0; return
  else
    freeb=$((maxb - anon))
  fi
  freeb=${freeb:-0}
  echo $((freeb / 1073741824))
}
slot_probe() {
  local n g
  if [ "$DEVICE" = cuda ]; then n=$(gpu_jobs); else n=0; fi
  g=$(free_gb)
  say "slot probe: gpu_jobs=$n (need < $MAX_GPU_JOBS), container free=${g} GB (need > $MIN_FREE_GB)"
  [ "$n" -lt "$MAX_GPU_JOBS" ] && [ "$g" -gt "$MIN_FREE_GB" ]
}
# the slot wait, before EVERY GPU step (argument: the step name)
wait_slot() {
  [ "$DRY" = 1 ] && return 0
  local t0=$SECONDS
  while true; do
    if slot_probe; then
      sleep "$PROBE_GAP_S"
      if slot_probe; then say "slot free on two probes ${PROBE_GAP_S} s apart: starting $1"; return 0; fi
    fi
    [ $((SECONDS - t0)) -ge "$MAX_WAIT_S" ] && stop "no free slot for $1 after $((SECONDS - t0)) s (limit ${MAX_WAIT_S} s): never a fourth GPU job"
    say "waiting for a slot for $1 ($((SECONDS - t0)) s of ${MAX_WAIT_S} s)"
    sleep "$POLL_S"
  done
}
gpu() { local step=$1; shift; wait_slot "$step"; run "$@"; }
extract() {  # grid run_dir arm
  gpu "extract $3" "$PY" "$TP" extract --grid "$1" --tree "$TREE" --run-dir "$2" --arm "$3" --cache "$CACHE" --states-dir "$STATES" --splits eval --eval-range "$EVAL_RANGE" --device "$DEVICE" || stop "extract $3 failed"
}
train_rc() {  # name rc: 0 and 2 (p5 not met) go on, the reader applies p5
  [ "$2" -eq 0 ] || [ "$2" -eq 2 ] || stop "$1 exited $2"
}

if [ "$DRY" = 1 ]; then
  say "DRY=1: commands only (no check, no wait, no run)"
else
  mkdir -p "$OUT" "$STATES"
  [ -f "$OUT/FINEGRID_R_DONE.txt" ] && { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid_r: REFUSED: $OUT/FINEGRID_R_DONE.txt exists (a finished marker); nothing is run"; exit 3; }
  ( set -o noclobber; printf 'FINEGRID_R_STARTED %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$OUT/FINEGRID_R_STARTED.txt" ) 2>/dev/null || { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid_r: REFUSED: $OUT/FINEGRID_R_STARTED.txt exists (a second start); nothing is run. Use a new OUT for a new run."; exit 3; }
  say "start: PB=$PB TREE=$TREE CACHE=$CACHE OUT=$OUT STATES=$STATES device=$DEVICE seeds=$SEEDS"
  [ -f "$TP" ] || stop "missing $TP"
  [ -d "$TREE/stack" ] || stop "missing tree $TREE"
  [ -d "$CACHE/train/clips" ] && [ -d "$CACHE/eval/clips" ] || stop "missing cache $CACHE"
fi

# ---- 1. control g2 (CPU)
if [ "$DRY" = 1 ]; then
  echo "DRY: CUDA_VISIBLE_DEVICES=-1 $PY $TP g2 --tree $TREE --cache $CACHE --out $OUT/G2_GRIDS.json $G2_ARGS"
else
  say "run: g2 on the CPU"
  CUDA_VISIBLE_DEVICES=-1 "$PY" "$TP" g2 --tree "$TREE" --cache "$CACHE" --out "$OUT/G2_GRIDS.json" $G2_ARGS
  say "g2 exit $? (the bar is applied by the reader; the run continues)"
fi

# ---- 2. process A: grid 4 by 10, 16 arms
gpu "train A" "$PY" "$TP" train --grid pool4x10 --input-norm ln --tree "$TREE" --out "$OUT/train_A" --arms "$ARMS_A" --sampler-chunk-like pool8x20 "${EXP[@]}" "${TA[@]}" --device "$DEVICE"
rc=$?
if [ "$DRY" != 1 ]; then train_rc "train A" $rc; [ -f "$OUT/train_A/TRAIN_DONE.json" ] || stop "train A wrote no TRAIN_DONE.json (rc $rc)"; say "train A done rc=$rc (0 = p5 met on every arm, 2 = not)"; fi
for A in "${LIST_A[@]}"; do extract pool4x10 "$OUT/train_A" "$A"; done

# ---- 3. process B: grid 8 by 20, 8 arms
gpu "train B" "$PY" "$TP" train --grid pool8x20 --input-norm ln --tree "$TREE" --out "$OUT/train_B" --arms "$ARMS_B" "${EXP[@]}" "${TA[@]}" --device "$DEVICE"
rc=$?
if [ "$DRY" != 1 ]; then train_rc "train B" $rc; [ -f "$OUT/train_B/TRAIN_DONE.json" ] || stop "train B wrote no TRAIN_DONE.json (rc $rc)"; say "train B done rc=$rc"; fi
for A in "${LIST_B[@]}"; do extract pool8x20 "$OUT/train_B" "$A"; done

# ---- 4. control r1 (CPU): the two processes saw the same windows in the same order
if [ "$DRY" = 1 ]; then
  echo "DRY: CUDA_VISIBLE_DEVICES=-1 $PY $TP r1 --a $OUT/train_A --b $OUT/train_B --out $OUT/R1_SAMPLER.json"
else
  say "run: r1 comparison"
  CUDA_VISIBLE_DEVICES=-1 "$PY" "$TP" r1 --a "$OUT/train_A" --b "$OUT/train_B" --out "$OUT/R1_SAMPLER.json"
  say "r1 exit $? (the reader applies it; the run continues)"
fi

# ---- 5. control arms (permuted closing labels, 2,000 steps)
gpu "p3 A" "$PY" "$TP" train --grid pool4x10 --input-norm ln --tree "$TREE" --out "$OUT/p3_A" --arms "$SH_A" --steps "$P3_STEPS" --shuffle-closing-labels --sampler-chunk-like pool8x20 "${TA[@]}" --device "$DEVICE"
rc=$?
[ "$DRY" = 1 ] || { train_rc "p3 A" $rc; say "p3 A done rc=$rc"; }
extract pool4x10 "$OUT/p3_A" "$SH_A"
gpu "p3 B" "$PY" "$TP" train --grid pool8x20 --input-norm ln --tree "$TREE" --out "$OUT/p3_B" --arms "$SH_B" --steps "$P3_STEPS" --shuffle-closing-labels "${TA[@]}" --device "$DEVICE"
rc=$?
[ "$DRY" = 1 ] || { train_rc "p3 B" $rc; say "p3 B done rc=$rc"; }
extract pool8x20 "$OUT/p3_B" "$SH_B"

# ---- 6. the done marker, last
if [ "$DRY" = 1 ]; then say "DRY=1: done (nothing written)"; exit 0; fi
{
  echo "FINEGRID_R_DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "tok_probe_gr.py sha256: $(sha256sum "$TP" | cut -c1-64)"
  echo "g2: $OUT/G2_GRIDS.json   r1: $OUT/R1_SAMPLER.json"
  echo "states folder (eval only): $STATES"
  for A in "${LIST_A[@]}"; do echo "ckpt $A $(sha256sum "$OUT/train_A/$A/ckpt_last.pt" | cut -c1-64)"; done
  for A in "${LIST_B[@]}"; do echo "ckpt $A $(sha256sum "$OUT/train_B/$A/ckpt_last.pt" | cut -c1-64)"; done
  echo "ckpt $SH_A $(sha256sum "$OUT/p3_A/$SH_A/ckpt_last.pt" | cut -c1-64)"
  echo "ckpt $SH_B $(sha256sum "$OUT/p3_B/$SH_B/ckpt_last.pt" | cut -c1-64)"
  echo "train logs: $OUT/{train_A,train_B}/<ARM>/train_log.jsonl  $OUT/{p3_A,p3_B}/<ARM>/train_log.jsonl; batch logs: $OUT/{train_A,train_B,p3_A,p3_B}/r1_batches.json"
} > "$OUT/FINEGRID_R_DONE.txt"
say "DONE: $OUT/FINEGRID_R_DONE.txt"
say "ZZ FINEGRID-R DONE"
exit 0
