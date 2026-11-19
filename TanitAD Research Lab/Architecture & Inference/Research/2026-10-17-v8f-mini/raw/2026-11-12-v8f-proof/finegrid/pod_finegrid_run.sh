#!/usr/bin/env bash
# pod_finegrid_run.sh -- card PREREG_V8F_PROOF_ADD41 (FINEGRID), the ONE pod job. Runs ON THE POD, started by the lead. DRY=1 prints the commands only.
#   0. refuses over a finished marker ($OUT/FINEGRID_DONE.txt)
#   1. SLOT WAIT: proceeds only when fewer than 3 GPU compute processes run (two probes, 45 s apart) AND the free memory of the container (memory.max minus the anon line of
#      memory.stat) is above 28 GB (same two probes); polls every 60 s for at most 10 h, then writes $OUT/FINEGRID_STOP.txt and exits. Never starts a fourth GPU job.
#   2. control g2 on the CPU: tok_probe_g.py g2 -> $OUT/G2_GRIDS.json (20 train clips of the manifest order, 10 evenly spaced frames each). It does not stop the run.
#   3. tok_probe_g.py train --grid pool8x20 --input-norm ln --expect-steps 19164 : the four arms F16_s0 F16_s1 F64_s0 F64_s1, one process
#   4. tok_probe_g.py extract : the four arms -> $STATES/<ARM>_{eval,train}.npz(+.json) and <ARM>_{eval,train}_lead.npz(+.json,+.head.npz)   (eval 0:141, train 600:1800)
#   5. control p3: arm F16SH_s0 (--shuffle-closing-labels --steps 2000), then its extract into the same folder
#   6. $OUT/FINEGRID_DONE.txt
# Every step writes a UTC time-stamped line to $OUT/finegrid.log. No process-name test anywhere: the guards read marker files and the GPU compute-apps list.
# Needed on the pod first (push by the usual route, compare sha256 on both sides): $PB/tok_probe_g.py. The tree $TREE and the cache $CACHE exist.
# Tier T0 (a diagnostic on logged frames), never driving performance. Nothing of the model is trained or touched.
set -u
PY=${PY:-/workspace/venv/bin/python}
TREE=${TREE:-/workspace/v8proof/tree_leadsup}
PB=${PB:-/workspace/v8proof/finegrid}
OUT=${OUT:-/workspace/v8proof/runs/finegrid}
STATES=${STATES:-$PB/states}
CACHE=${CACHE:-/workspace/v8proof/data/cache_pod}
STEPS_EXPECT=${STEPS_EXPECT:-19164}
P3_STEPS=${P3_STEPS:-2000}
MIN_FREE_GB=${MIN_FREE_GB:-28}
MAX_GPU_JOBS=${MAX_GPU_JOBS:-3}            # proceed only when the count of GPU compute processes is BELOW this
PROBE_GAP_S=${PROBE_GAP_S:-45}
POLL_S=${POLL_S:-60}
MAX_WAIT_S=${MAX_WAIT_S:-36000}            # 10 h
CG_MAX=${CG_MAX:-/sys/fs/cgroup/memory.max}
CG_STAT=${CG_STAT:-/sys/fs/cgroup/memory.stat}
DRY=${DRY:-0}
DEVICE=${DEVICE:-cuda}                     # dev-box test only: DEVICE=cpu
EVAL_RANGE=${EVAL_RANGE:-0:141}            # the card: eval clips 0:141, train probe clips 600:1800 (dev-box test only: smaller)
TRAIN_RANGE=${TRAIN_RANGE:-600:1800}
G2_ARGS=${G2_ARGS:-}                       # dev-box test only (smaller than 20 clips)
TRAINER_ARGV=${TRAINER_ARGV:-}             # dev-box test only: a synthetic trainer argv (the default is the argv record of LSD_s0 with pod paths)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ARMS="F16_s0 F16_s1 F64_s0 F64_s1"
GRID=pool8x20
TA=(); [ -n "$TRAINER_ARGV" ] && TA=(--trainer-argv "$TRAINER_ARGV")
EXP=(--expect-steps "$STEPS_EXPECT"); [ "$STEPS_EXPECT" = 0 ] && EXP=()
TP=$PB/tok_probe_g.py
LOG=$OUT/finegrid.log

say() {
  local line="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid: $*"
  echo "$line"
  if [ "$DRY" != 1 ] && [ -d "$OUT" ]; then echo "$line" >> "$LOG"; fi
}
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi; say "run: $*"; "$@"; }
stop() {  # reason
  say "STOP: $1"
  if [ "$DRY" != 1 ]; then mkdir -p "$OUT"; printf 'FINEGRID_STOP %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" > "$OUT/FINEGRID_STOP.txt"; fi
  exit 1
}

# number of GPU compute processes (the list nvidia-smi prints); an unreadable list counts as busy
gpu_jobs() {
  local o rc
  o=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo 99; return; fi
  printf '%s\n' "$o" | grep -c .
}
# free memory of the CONTAINER in whole GB: memory.max minus the anon line of memory.stat (free shows the host, not the cap)
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
# one probe: 0 = a slot is free (fewer than MAX_GPU_JOBS GPU jobs and more than MIN_FREE_GB GB free)
slot_probe() {
  local n g
  if [ "$DEVICE" = cuda ]; then n=$(gpu_jobs); else n=0; fi
  g=$(free_gb)
  say "slot probe: gpu_jobs=$n (need < $MAX_GPU_JOBS), container free=${g} GB (need > $MIN_FREE_GB)"
  [ "$n" -lt "$MAX_GPU_JOBS" ] && [ "$g" -gt "$MIN_FREE_GB" ]
}

if [ "$DRY" = 1 ]; then
  say "DRY=1: commands only (no check, no wait, no run)"
else
  [ -f "$OUT/FINEGRID_DONE.txt" ] && { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid: REFUSED: $OUT/FINEGRID_DONE.txt exists (a finished marker); nothing is run"; exit 3; }
  mkdir -p "$OUT" "$STATES"
  [ -f "$OUT/FINEGRID_STOP.txt" ] && mv "$OUT/FINEGRID_STOP.txt" "$OUT/FINEGRID_STOP.prev.$(date -u +%H%M%S).txt"
  say "start: PB=$PB TREE=$TREE CACHE=$CACHE OUT=$OUT STATES=$STATES device=$DEVICE"
  [ -f "$TP" ] || stop "missing $TP"
  [ -d "$TREE/stack" ] || stop "missing tree $TREE"
  [ -d "$CACHE/train/clips" ] && [ -d "$CACHE/eval/clips" ] || stop "missing cache $CACHE"
  # ---- 1. slot wait: two probes 45 s apart must both find a slot; poll every 60 s for at most 10 h
  SECONDS=0
  while true; do
    if slot_probe; then
      sleep "$PROBE_GAP_S"
      if slot_probe; then say "slot free on two probes ${PROBE_GAP_S} s apart: starting"; break; fi
    fi
    [ "$SECONDS" -ge "$MAX_WAIT_S" ] && stop "no free slot after ${SECONDS} s (limit ${MAX_WAIT_S} s): never a fourth GPU job"
    say "waiting for a slot (${SECONDS} s of ${MAX_WAIT_S} s)"
    sleep "$POLL_S"
  done
  if [ "$DEVICE" = cuda ]; then nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || stop "no GPU"; fi
fi

# ---- 2. control g2 (CPU): the 2 by 2 mean of the stored 8 by 20 tokens against the stored 4 by 10 tokens; the reader applies the bar, the run goes on
if [ "$DRY" = 1 ]; then
  echo "DRY: CUDA_VISIBLE_DEVICES=-1 $PY $TP g2 --tree $TREE --cache $CACHE --out $OUT/G2_GRIDS.json $G2_ARGS"
else
  say "run: g2 on the CPU"
  CUDA_VISIBLE_DEVICES=-1 "$PY" "$TP" g2 --tree "$TREE" --cache "$CACHE" --out "$OUT/G2_GRIDS.json" $G2_ARGS
  say "g2 exit $? (the bar is applied by the reader; the run continues)"
fi

# ---- 3. train (one process, four arms, the same batches)
run "$PY" "$TP" train --grid "$GRID" --input-norm ln --tree "$TREE" --out "$OUT/train" "${EXP[@]}" "${TA[@]}" --device "$DEVICE"
rc=$?
if [ "$DRY" != 1 ]; then
  [ $rc -eq 0 ] || [ $rc -eq 2 ] || stop "train exited $rc"
  [ -f "$OUT/train/TRAIN_DONE.json" ] || stop "train wrote no TRAIN_DONE.json (rc $rc)"
  P5_TRAIN=$([ $rc -eq 0 ] && echo pass || echo FAIL)
  say "train done rc=$rc p5=$P5_TRAIN"
else
  P5_TRAIN=dry
fi

# ---- 4. extract (the pull chain reads $STATES)
for A in $ARMS; do
  run "$PY" "$TP" extract --grid "$GRID" --tree "$TREE" --run-dir "$OUT/train" --arm "$A" --cache "$CACHE" --states-dir "$STATES" --eval-range "$EVAL_RANGE" --train-range "$TRAIN_RANGE" --device "$DEVICE" || stop "extract $A failed"
done

# ---- 5. control p3: shuffled closing-speed labels, 2,000 steps, arm F16SH_s0
run "$PY" "$TP" train --grid "$GRID" --input-norm ln --tree "$TREE" --out "$OUT/p3" --steps "$P3_STEPS" --shuffle-closing-labels "${TA[@]}" --device "$DEVICE"
rc3=$?
if [ "$DRY" != 1 ]; then
  [ $rc3 -eq 0 ] || [ $rc3 -eq 2 ] || stop "p3 train exited $rc3"
  P5_P3=$([ $rc3 -eq 0 ] && echo pass || echo FAIL)
  say "p3 train done rc=$rc3 p5=$P5_P3"
else
  P5_P3=dry
fi
run "$PY" "$TP" extract --grid "$GRID" --tree "$TREE" --run-dir "$OUT/p3" --arm F16SH_s0 --cache "$CACHE" --states-dir "$STATES" --eval-range "$EVAL_RANGE" --train-range "$TRAIN_RANGE" --device "$DEVICE" || stop "extract F16SH_s0 failed"

# ---- 6. the done marker, last
if [ "$DRY" = 1 ]; then say "DRY=1: done (nothing written)"; exit 0; fi
{
  echo "FINEGRID_DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "p5_main_run=$P5_TRAIN (control p5, per arm in $OUT/train/TRAIN_DONE.json)  p5_p3_run=$P5_P3"
  echo "tok_probe_g.py sha256: $(sha256sum "$TP" | cut -c1-64)"
  echo "g2: $OUT/G2_GRIDS.json"
  echo "states folder: $STATES (per arm: <ARM>_{eval,train}.npz(+.json), <ARM>_{eval,train}_lead.npz(+.json,+.head.npz))"
  for A in F16_s0 F16_s1 F64_s0 F64_s1 F16SH_s0; do
    d=$OUT/train; [ "$A" = F16SH_s0 ] && d=$OUT/p3
    echo "ckpt $A $(sha256sum "$d/$A/ckpt_last.pt" | cut -c1-64) $d/$A/ckpt_last.pt"
  done
  echo "train logs: $OUT/train/<ARM>/train_log.jsonl  $OUT/p3/F16SH_s0/train_log.jsonl; meta: $OUT/train/train_meta.json $OUT/p3/train_meta.json"
} > "$OUT/FINEGRID_DONE.txt"
say "DONE: $OUT/FINEGRID_DONE.txt"
exit 0
