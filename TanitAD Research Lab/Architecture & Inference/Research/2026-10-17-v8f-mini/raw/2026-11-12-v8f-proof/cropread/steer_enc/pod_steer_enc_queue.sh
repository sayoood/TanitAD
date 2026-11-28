#!/bin/bash
# pod_steer_enc_queue.sh -- ADD100 arm S data-side queue (pod, DETACHED, RESUMABLE). Never deletes or moves anything.
#   control s3 (gate) -> windows + tokens of arm S for all clips -> control s8 -> DONE.
# Needs: the data kit's outputs under $BASE/enc_out (lists, box, tok/B, staged videos under $BASE/stage) and the steering files
#        $S_ROOT/steer/<clip>.npy for every clip of enc_out/lists/all_clips.txt (written by the localiser step, not by this queue).
# Markers in $S_ROOT: START.txt, DONE.txt, STOP.txt (one line why). Log: $S_ROOT/queue.log. Re-running continues (every step is idempotent).
# Environment (all optional):
#   SMOKE=1      control s3 on 6 evaluation clips, then the full route on 2 evaluation clips with MADE-UP steering (fake_steer.py), then s8 on those 2;
#                new folder $BASE/s_smokeN (N = first free); keeps no camera pixels
#   WORKERS=3    window threads (CPU)
#   S_ROOT=<dir> output folder (default /workspace/v8proof/cropread/s_out ; smoke: $BASE/s_smokeN)
#   S3_GATE=1    stop after s3 when it fails (set 0 to continue anyway)
#   GPU_WAIT_S=3600  seconds to wait while three GPU processes are listed
# Start (full run, by the lead):  nohup setsid env WORKERS=3 bash /workspace/v8proof/cropread/kit/steer_enc/pod_steer_enc_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
BASE=/workspace/v8proof/cropread
KIT=$BASE/kit/steer_enc
ENCKIT=$BASE/kit/enc
ENC_OUT=$BASE/enc_out
STAGE=$BASE/stage
PY=/workspace/venv/bin/python
SMOKE=${SMOKE:-0}
WORKERS=${WORKERS:-3}
S3_GATE=${S3_GATE:-1}
GPU_WAIT_S=${GPU_WAIT_S:-3600}
if [ "$SMOKE" = "1" ]; then
  if [ -z "${S_ROOT:-}" ]; then
    n=1; while [ -e "$BASE/s_smoke$n" ]; do n=$((n+1)); done
    S_ROOT=$BASE/s_smoke$n
  fi
else
  S_ROOT=${S_ROOT:-$BASE/s_out}
fi
mkdir -p "$S_ROOT" "$S_ROOT/ctl" "$S_ROOT/steer" "$S_ROOT/tok"
LOG=$S_ROOT/queue.log
exec 9>"$S_ROOT/queue.lock"; flock -n 9 || { echo "queue already running in $S_ROOT" >> "$LOG"; exit 1; }
say()  { echo "[queue $(date -u +%FT%TZ)] $*" >> "$LOG"; }
stop() { say "STOP $*"; echo "$*" > "$S_ROOT/STOP.txt"; exit 1; }
echo "$(date -u +%FT%TZ) SMOKE=$SMOKE WORKERS=$WORKERS" > "$S_ROOT/START.txt"
[ -e "$S_ROOT/DONE.txt" ] && say "DONE marker already present; re-running steps are idempotent"

export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/workspace/v8proof/hf PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TANITAD_STACK=/workspace/v8proof/tree/stack
export PYTHONPATH=/workspace/v8proof/tree/stack:/workspace/v8proof/tree/stack/scripts:/workspace/v8proof/tree/taniteval:$ENCKIT:$KIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_TORCH_THREADS=2 PAI_DECODE_THREADS=1
export DINO_SAFETENSORS=$(cat /workspace/v8proof/hierrep/dino_safetensors.path 2>/dev/null)
CPU="env CUDA_VISIBLE_DEVICES=-1"
GPU="env -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

memcheck() {   # $1 = GB that must stay free
  local mx an free
  mx=$(cat /sys/fs/cgroup/memory.max); an=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2)
  free=$(( (mx - an) / 1000000000 ))
  [ "$free" -ge "$1" ] || stop "container memory free ${free} GB < $1 GB"
}
gpuwait() {    # other jobs of the fork may use the GPU: wait while three processes are listed
  local t=0 n
  while :; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "$n" -lt 3 ] && return 0
    [ "$t" -ge "$GPU_WAIT_S" ] && stop "GPU still lists $n processes after ${GPU_WAIT_S} s: not started"
    sleep 30; t=$((t+30))
  done
}

# ------------------------------------------------------------------ inputs
EV=$ENC_OUT/lists/eval141.txt
[ -s "$EV" ] || stop "missing $EV"
[ -d "$ENC_OUT/box" ] && [ -d "$ENC_OUT/tok/B" ] || stop "missing enc_out/box or enc_out/tok/B"

# ------------------------------------------------------------------ s3 (gate)
say "step 1: control s3"
if [ ! -s "$S_ROOT/ctl/S3.json" ]; then
  memcheck 28; gpuwait
  $GPU $PY $KIT/ctl_s3.py --eval-list $EV --root $STAGE --box-dir $ENC_OUT/box --b-tok-dir $ENC_OUT/tok/B --s-root $S_ROOT --workers 2 >> "$LOG" 2>&1
  rc=$?
  say "s3 rc=$rc"
  if [ $rc -ne 0 ] && [ "$S3_GATE" = "1" ]; then stop "control s3 failed (see ctl/S3.json); arm S encode not started"; fi
fi

# ------------------------------------------------------------------ list + steering
if [ "$SMOKE" = "1" ]; then
  $CPU $PY - <<PYEOF >> "$LOG" 2>&1 || stop "smoke list failed"
import hashlib
ev=[l.split()[0] for l in open("$EV") if l.strip()]
ev=sorted(ev,key=lambda c:hashlib.sha256(c.encode()).hexdigest())[:2]
open("$S_ROOT/clips_smoke.txt","w").write("\n".join(ev)+"\n")
PYEOF
  LST=$S_ROOT/clips_smoke.txt
  $CPU $PY $KIT/fake_steer.py --list $LST --box-dir $ENC_OUT/box --s-root $S_ROOT >> "$LOG" 2>&1 || stop "fake steering failed"
  KW=""                                   # no camera pixels are kept (steer_encode.py --keep-windows exists but the queue never passes it)
else
  LST=$ENC_OUT/lists/all_clips.txt
  [ -s "$LST" ] || stop "missing $LST"
  KW=""
fi

# ------------------------------------------------------------------ windows + tokens
say "step 2: arm S windows and tokens"
memcheck 28; gpuwait
$GPU $PY $KIT/steer_encode.py --list $LST --root $STAGE --s-root $S_ROOT --workers $WORKERS $KW >> "$LOG" 2>&1 || stop "steer_encode failed or refused (see queue.log, tok/STEER_REFUSED.json)"

# ------------------------------------------------------------------ s8
say "step 3: control s8"
if [ "$SMOKE" = "1" ]; then
  $CPU $PY $KIT/ctl_s8.py --eval-list $EV --train-final $ENC_OUT/lists/train_final_ids.txt --clips $LST --s-root $S_ROOT >> "$LOG" 2>&1
else
  $CPU $PY $KIT/ctl_s8.py --eval-list $EV --train-final $ENC_OUT/lists/train_final_ids.txt --s-root $S_ROOT >> "$LOG" 2>&1
fi
say "s8 rc=$?"
echo "ZZ STEER ENC DONE ZZ $(date -u +%FT%TZ) run=$S_ROOT" > "$S_ROOT/DONE.txt"
say "DONE"
exit 0
