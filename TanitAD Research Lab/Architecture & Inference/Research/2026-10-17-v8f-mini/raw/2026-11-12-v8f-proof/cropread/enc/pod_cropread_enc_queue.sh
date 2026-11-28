#!/bin/bash
# pod_cropread_enc_queue.sh -- CROPREAD data-side queue (pod, DETACHED, RESUMABLE). Never deletes or moves anything.
#   lists -> stage eval + train -> train_final (replacement rule) -> control c1 (gate) -> lead boxes -> windows + tokens (both arms)
#   -> control p1 -> control c7 -> DONE.
# Markers in $RUN: START.txt, DONE.txt, STOP.txt (one line why). Log: $RUN/queue.log. Re-running continues (every step is idempotent).
#
# Environment (all optional):
#   SMOKE=1        2 evaluation + 2 train clips, new folder smokeN (N = first free), keeps the private windows + one contact sheet
#   WORKERS=3      window threads (CPU) ; STAGE_WORKERS=4 download threads
#   RUN=<dir>      output folder (default $BASE/enc_out ; smoke: $BASE/smokeN)
#   C1_GATE=1      stop after c1 when c1 fails (set 0 to continue anyway)
# Start (full run, by the lead):  nohup setsid env WORKERS=3 bash /workspace/v8proof/cropread/kit/enc/pod_cropread_enc_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
BASE=/workspace/v8proof/cropread
KIT=$BASE/kit/enc
PY=/workspace/venv/bin/python
SMOKE=${SMOKE:-0}
WORKERS=${WORKERS:-3}
STAGE_WORKERS=${STAGE_WORKERS:-4}
C1_GATE=${C1_GATE:-1}
if [ "$SMOKE" = "1" ]; then
  if [ -z "${RUN:-}" ]; then
    n=1; while [ -e "$BASE/smoke$n" ]; do n=$((n+1)); done
    RUN=$BASE/smoke$n
  fi
  STAGE=$RUN/stage
else
  RUN=${RUN:-$BASE/enc_out}
  STAGE=${STAGE:-$BASE/stage}
fi
HFDIR=$BASE/hfdl
mkdir -p "$RUN" "$RUN/lists" "$RUN/ctl" "$RUN/box" "$RUN/tok" "$HFDIR" "$STAGE"
LOG=$RUN/queue.log
exec 9>"$RUN/queue.lock"; flock -n 9 || { echo "queue already running in $RUN" >> "$LOG"; exit 1; }
say()  { echo "[queue $(date -u +%FT%TZ)] $*" >> "$LOG"; }
stop() { say "STOP $*"; echo "$*" > "$RUN/STOP.txt"; exit 1; }
: > "$RUN/START.txt"; echo "$(date -u +%FT%TZ) SMOKE=$SMOKE WORKERS=$WORKERS" > "$RUN/START.txt"
[ -e "$RUN/DONE.txt" ] && say "DONE marker already present; re-running steps are idempotent"

export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/workspace/v8proof/hf PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TANITAD_STACK=/workspace/v8proof/tree/stack
export PYTHONPATH=/workspace/v8proof/tree/stack:/workspace/v8proof/tree/stack/scripts:/workspace/v8proof/tree/taniteval:$KIT
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
gpucheck() {
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" = "0" ] || stop "a GPU process is already running ($n): not started"
}

# ------------------------------------------------------------------ 1 lists
say "step 1: lists"
if [ ! -s "$RUN/lists/LISTS_RECORD.json" ]; then
  $CPU $PY $KIT/make_lists.py --out $RUN/lists >> "$LOG" 2>&1 || stop "make_lists failed"
fi
grep -q '"eligible_matches_expectation": true' $RUN/lists/LISTS_RECORD.json || stop "eligible train count is not 1,679 (see LISTS_RECORD.json)"
EV=$RUN/lists/eval141.txt
ORD=$RUN/lists/train_order.txt
if [ "$SMOKE" = "1" ]; then
  if [ ! -s $RUN/lists/eval_smoke.txt ]; then
    $CPU $PY - <<PYEOF >> "$LOG" 2>&1 || stop "smoke list failed"
import hashlib
ev=[l.split()[0] for l in open("$RUN/lists/eval141.txt") if l.strip()]
ev=sorted(ev,key=lambda c:hashlib.sha256(c.encode()).hexdigest())[:2]
open("$RUN/lists/eval_smoke.txt","w").write("\n".join(ev)+"\n")
tr=[l for l in open("$RUN/lists/train_order.txt") if l.strip()][:2]
open("$RUN/lists/train_order_smoke.txt","w").write("".join(tr))
PYEOF
  fi
  EV=$RUN/lists/eval_smoke.txt
  ORD=$RUN/lists/train_order_smoke.txt
fi

# ------------------------------------------------------------------ 2 stage
say "step 2: stage evaluation clips"
$CPU $PY $KIT/stage_clips.py --list $EV --root $STAGE --hf-dir $HFDIR --kind eval --workers $STAGE_WORKERS >> "$LOG" 2>&1 || stop "stage eval failed"
FIRST=500; [ "$SMOKE" = "1" ] && FIRST=2
say "step 2b: stage train (first $FIRST of the order), replacement rule"
for N in $FIRST 510 520; do
  [ "$SMOKE" = "1" ] && [ "$N" != "$FIRST" ] && break
  $CPU $PY $KIT/stage_clips.py --list $ORD --root $STAGE --hf-dir $HFDIR --kind train --first $N --workers $STAGE_WORKERS >> "$LOG" 2>&1 || stop "stage train failed"
  $CPU $PY $KIT/make_final.py --order $ORD --ledger $STAGE/ledger.jsonl --out $RUN/lists >> "$LOG" 2>&1
  nf=$(wc -l < $RUN/lists/train_final.txt 2>/dev/null || echo 0)
  say "train_final has $nf clips after staging $N positions"
  [ "$nf" -ge "$FIRST" ] && break
done
[ -s $RUN/lists/train_final.txt ] || stop "train_final.txt is empty"
if [ "$SMOKE" = "1" ]; then
  # make_final writes 450/50 roles; for 2 clips the roles are meaningless, keep the file as is
  :
fi
cut -d' ' -f1 $RUN/lists/train_final.txt > $RUN/lists/train_final_ids.txt
cat $EV $RUN/lists/train_final_ids.txt > $RUN/lists/all_clips.txt

# ------------------------------------------------------------------ 3 control c1 (gate)
say "step 3: control c1"
memcheck 28; gpucheck
if [ ! -s $RUN/ctl/C1.json ]; then
  CL=""; [ "$SMOKE" = "1" ] && CL="--clips $(paste -sd, $EV)"
  $GPU $PY $KIT/ctl_c1.py --eval-list $RUN/lists/eval141.txt --root $STAGE --out $RUN/ctl/C1.json $CL >> "$LOG" 2>&1
  rc=$?
  say "c1 rc=$rc"
  if [ $rc -ne 0 ] && [ "$C1_GATE" = "1" ]; then stop "control c1 failed (see ctl/C1.json); encode not started"; fi
fi

# ------------------------------------------------------------------ 4 lead boxes
say "step 4: lead boxes"
$CPU $PY $KIT/lead_box.py boxes --list $RUN/lists/all_clips.txt --root $STAGE --out $RUN/box >> "$LOG" 2>&1 || stop "lead_box boxes failed"

# ------------------------------------------------------------------ 5 windows + tokens
say "step 5: windows and tokens"
memcheck 28; gpucheck
KW=""; [ "$SMOKE" = "1" ] && KW="--keep-windows"
$GPU $PY $KIT/window_encode.py --list $RUN/lists/all_clips.txt --root $STAGE --box-dir $RUN/box --out $RUN --workers $WORKERS $KW >> "$LOG" 2>&1 || stop "window_encode failed (see queue.log)"

# ------------------------------------------------------------------ 6 controls p1, c7
say "step 6: controls p1 and c7"
$CPU $PY $KIT/lead_box.py p1 --eval-list $EV --train-final $RUN/lists/train_final_ids.txt --box-dir $RUN/box --out $RUN/ctl/P1.json >> "$LOG" 2>&1
say "p1 rc=$?"
if [ "$SMOKE" = "1" ]; then
  $CPU $PY $KIT/ctl_c7.py --eval-list $EV --train-final $RUN/lists/train_final_ids.txt --tok-dir $RUN/tok --n-eval 2 --n-train-min 2 --out $RUN/ctl/C7.json >> "$LOG" 2>&1
  say "c7 (smoke thresholds) rc=$?"
  $CPU $PY $KIT/contact_sheet.py --run $RUN --out $RUN/sheet_private.png >> "$LOG" 2>&1
else
  $CPU $PY $KIT/ctl_c7.py --eval-list $EV --train-final $RUN/lists/train_final_ids.txt --tok-dir $RUN/tok --verify-sha --out $RUN/ctl/C7.json >> "$LOG" 2>&1
  say "c7 rc=$?"
fi
echo "ZZ CROPREAD ENC DONE ZZ $(date -u +%FT%TZ) run=$RUN" > "$RUN/DONE.txt"
say "DONE"
exit 0
