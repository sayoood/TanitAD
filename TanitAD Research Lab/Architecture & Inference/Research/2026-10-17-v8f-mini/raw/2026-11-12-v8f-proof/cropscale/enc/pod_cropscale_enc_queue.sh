#!/bin/bash
# pod_cropscale_enc_queue.sh -- ENCODE queue of card ADD106 (CROPSCALE10K). Pod, DETACHED, RESUMABLE. Never deletes, moves or stops anything of others.
#   memory check -> lists -> identity encode of 3 pilot clips + compare (FIRST; a failure STOPS before any new clip is staged)
#   -> stage the new clips (unchanged stage tool of card ADD99) -> encode arm D only -> final list -> controls g0, g1 -> DONE.
# Markers in $ROOT: START_ENC.txt, DONE_ENC.txt, STOP_ENC.txt (one line why). One log: $ROOT/enc_queue.log. Time stamps: $ROOT/ENC_TIMES.txt (at start and after every stage chunk).
# Stop rules of the card (checked before every step, after every stage chunk, and every 30 s during the encode):
#   files of this card ($ROOT, du) above 60 GB, or free disk under 100 GB -> STOP_ENC.txt (the encode child of this queue is ended with TERM; nothing else is touched).
# A queue that stopped on a stop rule does not start again by itself: RESUME_AFTER_STOP=1 (a lead decision) continues.
#
# Environment (all optional):
#   DRY=1           print every command, make nothing (no folder, no file, no lock, no marker)
#   WORKERS=5       window threads of the encode ; STAGE_WORKERS=4 download threads ; CHUNK=150 clips per stage call
#   LIMIT_GB=60     files limit ; MIN_FREE_GB=100 free-disk floor
# Start (lead):  nohup setsid env WORKERS=5 bash /workspace/v8proof/cropscale/kit_enc/pod_cropscale_enc_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/workspace/v8proof/cropscale
KIT=$ROOT/kit_enc
PBASE=/workspace/v8proof/cropread
PKIT=$PBASE/kit/enc
PRUN=$PBASE/run1
PSTAGE=$PBASE/stage
PY=/workspace/venv/bin/python
DRY=${DRY:-0}
WORKERS=${WORKERS:-5}
STAGE_WORKERS=${STAGE_WORKERS:-4}
CHUNK=${CHUNK:-150}
LIMIT_GB=${LIMIT_GB:-60}
MIN_FREE_GB=${MIN_FREE_GB:-100}
STAGE=$ROOT/stage
HFDIR=$ROOT/hfdl
OUT=$ROOT/enc_out
LISTS=$ROOT/lists
CHECK=$ROOT/check_tok
LOG=$ROOT/enc_queue.log
TIMES=$ROOT/ENC_TIMES.txt
ELIG=$PBASE/enc_out/lists/train_eligible.txt
PFINAL=$PRUN/lists/train_final.txt
PIDS=$PRUN/lists/train_final_ids.txt
PEVAL=$PRUN/lists/eval141.txt
PTOK=$PRUN/tok/D
WENC=$KIT/window_encode_d.py

# ---------------------------------------------------------------- environment: exactly the pilot queue's (lines 40 to 46 of pod_cropread_enc_queue.sh)
export HF_HUB_DISABLE_XET=1 HF_HUB_CACHE=/workspace/v8proof/hf PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TANITAD_STACK=/workspace/v8proof/tree/stack
export PYTHONPATH=/workspace/v8proof/tree/stack:/workspace/v8proof/tree/stack/scripts:/workspace/v8proof/tree/taniteval:$PKIT:$KIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 V2_TORCH_THREADS=2 PAI_DECODE_THREADS=1
export DINO_SAFETENSORS=$(cat /workspace/v8proof/hierrep/dino_safetensors.path 2>/dev/null)
CPU="env CUDA_VISIBLE_DEVICES=-1"
GPU="env -u CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

say()   { if [ "$DRY" = "1" ]; then echo "DRY say: $*"; else echo "[queue $(date -u +%FT%TZ)] $*" >> "$LOG"; fi; }
stamp() { if [ "$DRY" = "1" ]; then echo "DRY stamp: $*"; else echo "$(date -u +%FT%TZ) $*" >> "$TIMES"; fi; }
stop()  { say "STOP $*"; stamp "STOP $*"; if [ "$DRY" = "1" ]; then exit 0; fi; echo "$*" > "$ROOT/STOP_ENC.txt"; exit 1; }
# runlog <command...>: run it with its output appended to the log; in DRY mode only print it
runlog() { if [ "$DRY" = "1" ]; then echo "DRY run: $*"; return 0; fi; "$@" >> "$LOG" 2>&1; }

memcheck() {   # $1 = GB that must stay free (memory.max minus the anon line of memory.stat)
  if [ "$DRY" = "1" ]; then echo "DRY memcheck: keep $1 GB free (memory.max - anon)"; return 0; fi
  local mx an free
  mx=$(cat /sys/fs/cgroup/memory.max); an=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2)
  free=$(( (mx - an) / 1000000000 ))
  [ "$free" -ge "$1" ] || stop "container memory free ${free} GB < $1 GB"
}
gpucheck() {
  if [ "$DRY" = "1" ]; then echo "DRY gpucheck: no other GPU process"; return 0; fi
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" = "0" ] || stop "a GPU process is already running ($n): not started"
}
diskcheck() {  # the card's stop rules: files of this card above LIMIT_GB, or free disk under MIN_FREE_GB
  if [ "$DRY" = "1" ]; then echo "DRY diskcheck: du -sb $ROOT <= ${LIMIT_GB} GB and free >= ${MIN_FREE_GB} GB"; return 0; fi
  local used free
  used=$(du -sb "$ROOT" 2>/dev/null | cut -f1); free=$(df -B1 --output=avail "$ROOT" 2>/dev/null | tail -n 1 | tr -d ' ')
  [ "${used:-0}" -le $((LIMIT_GB * 1000000000)) ] || stop "files of this card are $((used / 1000000000)) GB, above the limit of ${LIMIT_GB} GB"
  [ "${free:-0}" -ge $((MIN_FREE_GB * 1000000000)) ] || stop "free disk is $((free / 1000000000)) GB, under the floor of ${MIN_FREE_GB} GB"
}

# ---------------------------------------------------------------- start: lock, markers, preflight
if [ "$DRY" = "1" ]; then
  echo "DRY=1: nothing is created, no folder, no lock, no marker; commands are printed"
else
  [ -d "$ROOT" ] || { echo "no $ROOT"; exit 2; }
  exec 9>"$ROOT/enc_queue.lock"; flock -n 9 || { echo "[queue $(date -u +%FT%TZ)] queue already running: second start refused" >> "$LOG"; echo "queue already running"; exit 1; }
  if [ -e "$ROOT/STOP_ENC.txt" ] && [ "${RESUME_AFTER_STOP:-0}" != "1" ]; then
    echo "STOP_ENC.txt is present ($(cat "$ROOT/STOP_ENC.txt")): a stop rule fired; set RESUME_AFTER_STOP=1 to continue (lead decision)"; exit 3
  fi
  mkdir -p "$LISTS" "$STAGE" "$HFDIR"
  echo "$(date -u +%FT%TZ) WORKERS=$WORKERS STAGE_WORKERS=$STAGE_WORKERS CHUNK=$CHUNK" > "$ROOT/START_ENC.txt"
  for f in "$ELIG" "$PFINAL" "$PIDS" "$PEVAL" "$PTOK/MANIFEST.json" "$PKIT/stage_clips.py" "$PKIT/window_encode.py" "$PKIT/enc_common.py" "$KIT/make_large_list.py" "$KIT/ctl_g0g1.py" "$KIT/gen_window_encode_d.py"; do
    [ -s "$f" ] || stop "missing input $f"
  done
  [ -s /root/.hf_token ] || stop "the stage tool's token file is missing"
fi
stamp "START_ENC"
say "start: DRY=$DRY WORKERS=$WORKERS STAGE_WORKERS=$STAGE_WORKERS CHUNK=$CHUNK LIMIT_GB=$LIMIT_GB MIN_FREE_GB=$MIN_FREE_GB"

# ---------------------------------------------------------------- 1 memory, lists
say "step 1: memory check, lists"
memcheck 28; diskcheck
runlog $CPU $PY $KIT/make_large_list.py new-order --out $LISTS || stop "make_large_list new-order failed (see enc_queue.log)"
if [ "$DRY" = "1" ]; then
  echo "DRY run: head -n 3 $PIDS > $LISTS/identity3.txt   (only if absent)"
else
  [ -s "$LISTS/identity3.txt" ] || head -n 3 "$PIDS" > "$LISTS/identity3.txt"
  [ "$(grep -c . "$LISTS/identity3.txt")" = "3" ] || stop "identity3.txt does not hold 3 ids"
fi

# ---------------------------------------------------------------- 2 the variant of the encode tool, then the identity encode and compare (FIRST)
say "step 2: window_encode_d.py, identity encode of 3 pilot clips and compare"
runlog $CPU $PY $KIT/gen_window_encode_d.py --src $PKIT/window_encode.py --out $WENC || stop "gen_window_encode_d failed (source hash or an edit count)"
IDROOT=$PSTAGE
if [ "$DRY" = "1" ]; then
  echo "DRY: identity inputs are read from the pilot's stage root $PSTAGE when its three clips are there, else the three are staged into $STAGE first"
else
  for c in $(cat "$LISTS/identity3.txt"); do
    { [ -s "$PSTAGE/r0/camera/$c.mp4" ] && [ -s "$PSTAGE/ts/$c.timestamps.parquet" ] && [ -s "$PSTAGE/ego/$c.parquet" ]; } || IDROOT=$STAGE
  done
  [ -s "$PSTAGE/calibration/physicalai_front_wide_intrinsics.csv" ] || IDROOT=$STAGE
  if [ "$IDROOT" = "$STAGE" ]; then
    say "pilot stage root lacks the three identity clips: staging them (pilot clips, not new clips) into $STAGE"
    runlog $CPU $PY $PKIT/stage_clips.py --list $LISTS/identity3.txt --root $STAGE --hf-dir $HFDIR --kind train --workers $STAGE_WORKERS || stop "stage of the identity clips failed"
  fi
fi
memcheck 28; gpucheck; diskcheck
runlog $GPU $PY $WENC --list $LISTS/identity3.txt --root $IDROOT --out $CHECK --workers 2 --min-free-gb $MIN_FREE_GB || stop "identity encode failed (see enc_queue.log)"
runlog $CPU $PY $KIT/ctl_g0g1.py identity --root $ROOT --pilot-run $PRUN --ids $LISTS/identity3.txt --out $ROOT/ENC_IDENTITY.json || stop "identity compare failed: three pilot clips encoded again are not equal to the pilot's tokens within 1e-3; no new clip staged"
stamp "identity compare passed"

# ---------------------------------------------------------------- 3 stage the new clips (unchanged stage tool of card ADD99, from its pod folder)
say "step 3: stage the new clips in chunks of $CHUNK (every eligible clip is tried; one that does not stage is left out and counted)"
if [ "$DRY" = "1" ]; then N_NEW=1179; else N_NEW=$(grep -c . "$LISTS/train_new_order.txt"); fi
[ "$N_NEW" -gt 0 ] || stop "train_new_order.txt is empty"
n=0
while [ "$n" -lt "$N_NEW" ]; do
  n=$((n + CHUNK)); [ "$n" -gt "$N_NEW" ] && n=$N_NEW
  memcheck 28; diskcheck
  runlog $CPU $PY $PKIT/stage_clips.py --list $LISTS/train_new_order.txt --root $STAGE --hf-dir $HFDIR --kind train --first $n --workers $STAGE_WORKERS --min-free-gb $MIN_FREE_GB || stop "stage failed at first=$n (see enc_queue.log)"
  stamp "stage chunk first=$n of $N_NEW done"
  diskcheck
done
if [ "$DRY" = "1" ]; then
  echo "DRY run: make_large_list.py staged --write  -> LIST=<train_new_staged_<hash>.txt>"
  LIST=$LISTS/train_new_staged_DRY.txt
else
  OUTS=$($CPU $PY $KIT/make_large_list.py staged --out $LISTS --stage-ledger $STAGE/ledger.jsonl --write 2>> "$LOG") || stop "make_large_list staged failed"
  echo "$OUTS" >> "$LOG"
  LIST=$(echo "$OUTS" | sed -n 's/^LIST=//p' | tail -n 1)
  [ -s "$LIST" ] || stop "no clip staged ok (staged list is empty)"
fi

# ---------------------------------------------------------------- 4 encode arm D only (the pilot's call, generated variant); the card's files limit is watched every 30 s
say "step 4: encode arm D of the staged new clips"
memcheck 28; gpucheck; diskcheck
if [ "$DRY" = "1" ]; then
  echo "DRY run: $GPU $PY $WENC --list $LIST --root $STAGE --out $OUT --workers $WORKERS --min-free-gb $MIN_FREE_GB   (in the background; du/df watched every 30 s; TERM to this child only on a stop rule)"
else
  $GPU $PY $WENC --list "$LIST" --root $STAGE --out $OUT --workers $WORKERS --min-free-gb $MIN_FREE_GB >> "$LOG" 2>&1 &
  EPID=$!
  while kill -0 $EPID 2>/dev/null; do
    sleep 30
    used=$(du -sb "$ROOT" 2>/dev/null | cut -f1); free=$(df -B1 --output=avail "$ROOT" 2>/dev/null | tail -n 1 | tr -d ' ')
    if [ "${used:-0}" -gt $((LIMIT_GB * 1000000000)) ] || [ "${free:-0}" -lt $((MIN_FREE_GB * 1000000000)) ]; then
      kill -TERM $EPID 2>/dev/null; wait $EPID 2>/dev/null
      stop "stop rule during the encode: files $((used / 1000000000)) GB (limit ${LIMIT_GB}), free $((free / 1000000000)) GB (floor ${MIN_FREE_GB}); the encode child was ended"
    fi
  done
  wait $EPID; rc=$?
  say "encode rc=$rc"
  [ -e "$OUT/tok/ENCODE_FAILS.json" ] && cp -n "$OUT/tok/ENCODE_FAILS.json" "$OUT/tok/ENCODE_FAILS_$(date -u +%Y%m%dT%H%M%SZ).json"
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then stop "encode ended with rc=$rc (7 = disk floor); see enc_queue.log"; fi
fi
stamp "encode done"

# ---------------------------------------------------------------- 5 final list, controls g0 and g1
say "step 5: final list, controls g0 and g1"
runlog $CPU $PY $KIT/make_large_list.py final --out $LISTS --stage-ledger $STAGE/ledger.jsonl --tok-dir $OUT/tok/D || stop "make_large_list final failed (see enc_queue.log)"
runlog $CPU $PY $KIT/ctl_g0g1.py run --root $ROOT --pilot-run $PRUN --workers 4 --out $ROOT/ENC_CONTROLS.json
crc=$?
say "controls g0 g1 rc=$crc (0 = both pass; the record is ENC_CONTROLS.json)"
stamp "controls rc=$crc"
if [ "$DRY" = "1" ]; then echo "DRY: would write DONE_ENC.txt"; exit 0; fi
echo "ZZ CROPSCALE ENC DONE ZZ $(date -u +%FT%TZ) controls_rc=$crc root=$ROOT" > "$ROOT/DONE_ENC.txt"
say "DONE"
exit 0
