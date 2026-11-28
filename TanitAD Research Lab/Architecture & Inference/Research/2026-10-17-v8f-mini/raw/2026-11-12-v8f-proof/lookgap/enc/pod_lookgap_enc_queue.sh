#!/bin/bash
# pod_lookgap_enc_queue.sh -- ENCODE queue of card ADD108 (LOOKGAP-T0), stage B. Pod, DETACHED, RESUMABLE. Deletes and moves nothing; ends only its own encode child on a stop rule.
#   REFUSES to start unless GO_STAGE_B.txt (written by the lead after the PI's go) and DONE_DIRS.txt (stage A is finished) exist in $ROOT.
#   gate -> lists + inputs -> variant of the pilot's tool -> identity run on 3 clips (pilot box folder, arm name B_ident) + ctl_m1 clause i (a failure STOPS)
#   -> encode of the 641 clips (500 train + 141 evaluation of the pilot's lists) from $ROOT/dirs into $ROOT/tok/P -> ctl_m1 (all clauses) -> DONE.
# Markers in $ROOT: START_ENC.txt, DONE_ENC.txt, STOP_ENC.txt (one line why). One log: $ROOT/enc_queue.log. Time stamps: $ROOT/ENC_TIMES.txt.
# Stop rules of the card (section 7), measured by this queue (checked before every step and every 30 s during the encode):
#   encode jobs above 2 GPU-hours (wall time of its identity run and its encode, summed in $ROOT/ENC_GPU_SECONDS.txt), or $ROOT/tok above 20 GB (du -sb, not df)
#   -> STOP_ENC.txt (the encode child of this queue is ended with TERM; nothing else is touched). A stopped queue continues only with RESUME=1 (a lead decision).
# Slot: an encode job starts when fewer than 3 GPU compute processes run and more than 28 GB of the container memory are free (memory.max minus anon).
#
# Environment (all optional):
#   DRY=1           print every command, make nothing (no folder, no file, no lock, no marker)
#   RESUME=1        continue after STOP_ENC.txt (the ledger of the encode tool skips the clips that are done)
#   WORKERS=5       window threads of the encode ; GPU_LIMIT_S=7200 ; TOK_LIMIT_GB=20 ; MIN_FREE_GB=25 ; SLOT_WAIT_MAX_S=21600
# Start (lead):  nohup setsid env WORKERS=5 bash /workspace/v8proof/lookgap/kit_enc/pod_lookgap_enc_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/workspace/v8proof/lookgap
KIT=$ROOT/kit_enc
PBASE=/workspace/v8proof/cropread
PKIT=$PBASE/kit/enc
PRUN=$PBASE/run1
PSTAGE=$PBASE/stage
PBOX=$PBASE/enc_out/box
PY=/workspace/venv/bin/python
DRY=${DRY:-0}
RESUME=${RESUME:-0}
WORKERS=${WORKERS:-5}
GPU_LIMIT_S=${GPU_LIMIT_S:-7200}
TOK_LIMIT_GB=${TOK_LIMIT_GB:-20}
MIN_FREE_GB=${MIN_FREE_GB:-25}
SLOT_WAIT_MAX_S=${SLOT_WAIT_MAX_S:-21600}
DIRS=$ROOT/dirs
TOK=$ROOT/tok
CHECK=$ROOT/check_tok
LISTS=$ROOT/lists
LOG=$ROOT/enc_queue.log
TIMES=$ROOT/ENC_TIMES.txt
GPUSEC=$ROOT/ENC_GPU_SECONDS.txt
PTRAIN=$PRUN/lists/train_final_ids.txt
PEVAL=$PRUN/lists/eval141.txt
ID3=$LISTS/identity3.txt
ENC641=$LISTS/enc641.txt
WENC=$KIT/window_encode_p.py

# ---------------------------------------------------------------- environment: the pilot queue's, as in the kit of card ADD106
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

gpu_seconds() { if [ -s "$GPUSEC" ]; then awk '{s+=$2} END {print s+0}' "$GPUSEC"; else echo 0; fi; }
mem_free_gb() { local mx an; mx=$(cat /sys/fs/cgroup/memory.max); an=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); echo $(( (mx - an) / 1000000000 )); }
tok_bytes()   { du -sb "$TOK" 2>/dev/null | cut -f1; }
free_bytes()  { df -B1 --output=avail "$ROOT" 2>/dev/null | tail -n 1 | tr -d ' '; }

# wait_slot: fewer than 3 GPU compute processes AND more than 28 GB of container memory free; polls every 60 s; stops after SLOT_WAIT_MAX_S
wait_slot() {
  if [ "$DRY" = "1" ]; then echo "DRY wait_slot: until fewer than 3 GPU compute processes and more than 28 GB free of the cgroup memory (poll 60 s, at most ${SLOT_WAIT_MAX_S} s)"; return 0; fi
  local waited=0 n fr
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    fr=$(mem_free_gb)
    if [ "${n:-9}" -lt 3 ] && [ "${fr:-0}" -gt 28 ]; then [ "$waited" -gt 0 ] && say "slot free after ${waited} s (gpu procs $n, memory free ${fr} GB)"; return 0; fi
    [ "$waited" -ge "$SLOT_WAIT_MAX_S" ] && stop "no slot after ${waited} s (gpu procs $n, memory free ${fr} GB)"
    [ "$waited" = "0" ] && say "waiting for a slot (gpu procs $n, memory free ${fr} GB)"
    sleep 60; waited=$((waited + 60))
  done
}
# limits: the card's two stop rules
limits() {
  if [ "$DRY" = "1" ]; then echo "DRY limits: GPU seconds of the encode jobs <= ${GPU_LIMIT_S}; du -sb $TOK <= ${TOK_LIMIT_GB} GB; free disk >= ${MIN_FREE_GB} GB"; return 0; fi
  local g t f; g=$(gpu_seconds); t=$(tok_bytes); f=$(free_bytes)
  [ "${g:-0}" -le "$GPU_LIMIT_S" ] || stop "encode jobs used ${g} s, above the limit of ${GPU_LIMIT_S} s"
  [ "${t:-0}" -le $((TOK_LIMIT_GB * 1000000000)) ] || stop "$TOK is $((t / 1000000000)) GB (du), above the limit of ${TOK_LIMIT_GB} GB"
  [ "${f:-0}" -ge $((MIN_FREE_GB * 1000000000)) ] || stop "free disk is $((f / 1000000000)) GB, under the floor of ${MIN_FREE_GB} GB"
}
# gpujob <label> <command...>: run an encode job in the background, watch the two stop rules every 30 s, record its wall seconds; rc of the job is returned
gpujob() {
  local label=$1; shift
  if [ "$DRY" = "1" ]; then echo "DRY run (background, watched every 30 s; TERM to this child only on a stop rule): $*"; return 0; fi
  local t0 pid g0 el g t rc
  t0=$(date +%s); g0=$(gpu_seconds)
  "$@" >> "$LOG" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    el=$(( $(date +%s) - t0 )); g=$(( g0 + el )); t=$(tok_bytes)
    if [ "$g" -gt "$GPU_LIMIT_S" ] || [ "${t:-0}" -gt $((TOK_LIMIT_GB * 1000000000)) ]; then
      kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null
      echo "$label $(( $(date +%s) - t0 ))" >> "$GPUSEC"
      stop "stop rule during $label: encode jobs ${g} s (limit ${GPU_LIMIT_S}), $TOK $((t / 1000000000)) GB (limit ${TOK_LIMIT_GB}); the encode child was ended"
    fi
  done
  wait $pid; rc=$?
  echo "$label $(( $(date +%s) - t0 ))" >> "$GPUSEC"
  return $rc
}

# ---------------------------------------------------------------- gate, lock, markers, preflight
if [ "$DRY" = "1" ]; then
  echo "DRY=1: nothing is created, no folder, no lock, no marker; commands are printed"
  echo "DRY gate: REFUSED unless $ROOT/GO_STAGE_B.txt and $ROOT/DONE_DIRS.txt exist"
else
  [ -d "$ROOT" ] || { echo "no $ROOT"; exit 2; }
  [ -e "$ROOT/GO_STAGE_B.txt" ] || { echo "REFUSED: $ROOT/GO_STAGE_B.txt is missing (the lead writes it after the PI's go for stage B)"; exit 4; }
  [ -e "$ROOT/DONE_DIRS.txt" ] || { echo "REFUSED: $ROOT/DONE_DIRS.txt is missing (stage A, the direction files, is not finished)"; exit 4; }
  exec 9>"$ROOT/enc_queue.lock"; flock -n 9 || { echo "[queue $(date -u +%FT%TZ)] queue already running: second start refused" >> "$LOG"; echo "queue already running"; exit 1; }
  if [ -e "$ROOT/STOP_ENC.txt" ] && [ "$RESUME" != "1" ]; then
    echo "STOP_ENC.txt is present ($(cat "$ROOT/STOP_ENC.txt")): a stop rule fired; set RESUME=1 to continue (lead decision)"; exit 3
  fi
  if [ -e "$ROOT/DONE_ENC.txt" ] && [ "$RESUME" != "1" ]; then echo "DONE_ENC.txt is present: the encode is finished"; exit 3; fi
  mkdir -p "$LISTS" "$TOK" "$CHECK"
  echo "$(date -u +%FT%TZ) WORKERS=$WORKERS RESUME=$RESUME GPU_LIMIT_S=$GPU_LIMIT_S TOK_LIMIT_GB=$TOK_LIMIT_GB" >> "$ROOT/START_ENC.txt"
  for f in "$PTRAIN" "$PEVAL" "$PRUN/tok/B/MANIFEST.json" "$PRUN/tok/D/MANIFEST.json" "$PKIT/window_encode.py" "$PKIT/enc_common.py" \
           "$KIT/gen_window_encode_p.py" "$KIT/ctl_m1.py"; do
    [ -s "$f" ] || stop "missing input $f"
  done
  [ -d "$PBOX" ] || stop "missing the pilot's box folder $PBOX"
  [ -d "$DIRS" ] || stop "missing the direction folder $DIRS"
fi
stamp "START_ENC"
say "start: DRY=$DRY RESUME=$RESUME WORKERS=$WORKERS GPU_LIMIT_S=$GPU_LIMIT_S TOK_LIMIT_GB=$TOK_LIMIT_GB MIN_FREE_GB=$MIN_FREE_GB"

# ---------------------------------------------------------------- 1 lists and inputs
say "step 1: lists (500 train + 141 evaluation clips of the pilot, 3 identity clips), direction files, staged clips"
limits
if [ "$DRY" = "1" ]; then
  echo "DRY run: first columns of $PTRAIN and $PEVAL > $ENC641 (641 distinct required);  first 3 of the first column of $PEVAL > $ID3"
  echo "DRY run: every clip of $ENC641 has $DIRS/<clip>.npz; $CPU $PY -c <every clip of $ENC641 has its mp4 and timestamps under $PSTAGE>"
else
  [ -s "$ENC641" ] || awk 'NF {print $1}' "$PTRAIN" "$PEVAL" > "$ENC641"      # only when absent; an existing list is never rewritten
  [ -s "$ID3" ] || awk 'NF {print $1}' "$PEVAL" | head -n 3 > "$ID3"
  [ "$(grep -c . "$ENC641")" = "641" ] && [ "$(sort -u "$ENC641" | grep -c .)" = "641" ] || stop "enc641.txt does not hold 641 distinct ids"
  [ "$(grep -c . "$ID3")" = "3" ] || stop "identity3.txt does not hold 3 ids"
  miss=0; for c in $(cat "$ENC641"); do [ -s "$DIRS/$c.npz" ] || miss=$((miss + 1)); done
  [ "$miss" = "0" ] || stop "$miss of the 641 clips have no direction file in $DIRS"
  $CPU $PY -c "
import os, sys
from enc_common import Layout
L = Layout(sys.argv[1])
ids = [l.split()[0] for l in open(sys.argv[2]) if l.strip()]
miss = [c for c in ids if not (os.path.exists(L.mp4(c)) and os.path.exists(L.ts(c)))]
print('staged clips missing:', len(miss), miss[:3])
sys.exit(1 if miss else 0)
" "$PSTAGE" "$ENC641" >> "$LOG" 2>&1 || stop "some of the 641 clips are not staged in the pilot's stage root $PSTAGE (see enc_queue.log)"
fi

# ---------------------------------------------------------------- 2 the variant, the identity run and clause i (FIRST; a failure stops before the 641 clips)
say "step 2: window_encode_p.py from the pilot's tool; identity run of 3 clips with the pilot's box folder (arm name B_ident); ctl_m1 identity"
runlog $CPU $PY $KIT/gen_window_encode_p.py --src $PKIT/window_encode.py --out $WENC || stop "gen_window_encode_p failed (source hash or an edit count)"
wait_slot; limits
gpujob identity $GPU $PY $WENC --list $ID3 --root $PSTAGE --box-dir $PBOX --arm B_ident --out $CHECK --workers 2 --min-free-gb $MIN_FREE_GB || stop "identity encode failed (see enc_queue.log)"
runlog $CPU $PY $KIT/ctl_m1.py identity --root $ROOT --pilot-run $PRUN --ids $ID3 --ident-arm B_ident --out $ROOT/ENC_M1_IDENTITY.json || stop "ctl_m1 clause i failed: three pilot clips encoded again are not byte-equal to the pilot's arm B; the 641 clips are not encoded"
stamp "identity (clause i) passed"

# ---------------------------------------------------------------- 3 encode of the 641 clips, arm P
say "step 3: encode of the 641 clips, arm P, from $DIRS into $TOK/P"
wait_slot; limits
gpujob encode $GPU $PY $WENC --list $ENC641 --root $PSTAGE --box-dir $DIRS --arm P --out $ROOT --workers $WORKERS --min-free-gb $MIN_FREE_GB
rc=$?
say "encode rc=$rc"
if [ "$DRY" != "1" ]; then
  [ -e "$TOK/ENCODE_FAILS.json" ] && cp -n "$TOK/ENCODE_FAILS.json" "$TOK/ENCODE_FAILS_$(date -u +%Y%m%dT%H%M%SZ).json"
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then stop "encode ended with rc=$rc (7 = disk floor); see enc_queue.log"; fi
fi
stamp "encode done"

# ---------------------------------------------------------------- 4 control m1 (all clauses; ii and iii read the files now, clause i the identity run)
say "step 4: control m1"
runlog $CPU $PY $KIT/ctl_m1.py all --root $ROOT --pilot-run $PRUN --ids $ID3 --ident-arm B_ident --workers 8 --out $ROOT/ENC_M1.json
crc=$?
say "control m1 rc=$crc (0 = all clauses pass; the record is ENC_M1.json)"
stamp "control m1 rc=$crc"
if [ "$DRY" = "1" ]; then echo "DRY: would write DONE_ENC.txt"; exit 0; fi
echo "ZZ LOOKGAP ENC DONE ZZ $(date -u +%FT%TZ) m1_rc=$crc gpu_seconds=$(gpu_seconds) root=$ROOT" > "$ROOT/DONE_ENC.txt"
say "DONE"
exit 0
