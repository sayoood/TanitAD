#!/usr/bin/env bash
# pod_preshead_fit_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD118 (PRESHEAD-T0), section 2: three fits of the presence head (pres_fit.py, the generated variant of crop_fit.py)
# on the existing tokens of arm D, and their three dumps on the evaluation clips (pres_dump.py). Tier T0. No encode, no rollout.
# Order: fit seed 0 and fit seed 1 side by side; then the permuted fit of seed 0 beside the dumps of seeds 0 and 1; then the dump of the permuted fit.
# Admission: the lead's test output of the fit kit ($GP/lead_tests/lead_pytest_fit.out with a line of the form "<n> passed" and no "failed" or "error"), the generator record reads pass_p0 true
#   and names the hash of the fit tool in the kit, the inputs exist, memory free above 28 GB (memory.max minus anon).
# Stop rule: $STOP_S seconds from the queue start (card: 4 pod-h); every step runs under the time that is left. Markers in $GP: PRESHEAD_START.txt, PRESHEAD_DONE.txt, PRESHEAD_STOP.txt;
#   per step $GP/jobs/<step>.START/.DONE/.STOP and $GP/logs/<step>.log. A finished fit (FIT_RECORD.json) or dump (pres.npz with pres_meta.json) is not run again. Never deletes, moves or kills.
# Usage: bash pod_preshead_fit_queue.sh          DRY=1 bash pod_preshead_fit_queue.sh   (prints every command, makes nothing)
set -u
V=/workspace/v8proof
GP=${PH_ROOT:-$V/preshead}
K=${PH_KIT:-$GP/kit_fit}
FR=${PH_TOKROOT:-$V/cropscale/fitroot}
JOIN=${PH_JOIN_TRAIN:-$V/data/inputs/join_train.npz}
EJ=${PH_EVAL_JOIN:-$V/podread/data/join_eval141.npz}
PY=${PH_PY:-/workspace/venv/bin/python}
DEV=${PH_DEVICE:-cuda}
EXTRA=${PH_FIT_EXTRA:-}
MINGB=${PH_MIN_FREE_GB:-28}
STOP_S=${PH_STOP_S:-14400}
LT=${PH_TESTS:-$GP/lead_tests/lead_pytest_fit.out}
DRY=${DRY:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fitcmd() { echo "$PY $K/pres_fit.py --arm D --head H5 --seed $1${2:+ $2} --root $FR --out-root $GP --join-train $JOIN --device $DEV${EXTRA:+ $EXTRA}"; }
dumpcmd() { echo "$PY $K/pres_dump.py --seed $1${2:+ $2} --root $FR --out-root $GP --eval-join $EJ --device $DEV"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: admission: $LT has a passed line, $K/PRES_FIT_GEN_RECORD.json reads pass_p0 true and names the hash of $K/pres_fit.py, memory free > $MINGB GB, else STOP"
  echo "CMD: mkdir -p $GP/jobs $GP/logs $GP/fits $GP/dump"
  echo "JOB fit_s0:"; echo "  CMD: $(fitcmd 0)"
  echo "JOB fit_s1 (beside fit_s0):"; echo "  CMD: $(fitcmd 1)"
  echo "JOB fit_s0_perm (after the two fits):"; echo "  CMD: $(fitcmd 0 --permute-labels)"
  echo "JOB dump_s0 (beside fit_s0_perm):"; echo "  CMD: $(dumpcmd 0)"
  echo "JOB dump_s1 (beside fit_s0_perm):"; echo "  CMD: $(dumpcmd 1)"
  echo "JOB dump_s0_perm (last):"; echo "  CMD: $(dumpcmd 0 --permute-labels)"
  echo "DRY run complete: stop rule $STOP_S s"; exit 0
fi
stop() { echo "$(ts) $*" >> "$GP/PRESHEAD_STOP.txt"; echo "STOP $*"; exit 3; }
[ -d "$K" ] || { echo "REFUSED: no $K"; exit 2; }
[ -f "$GP/PRESHEAD_DONE.txt" ] && { echo "ALREADY-DONE"; exit 0; }
[ -f "$GP/PRESHEAD_STOP.txt" ] && { echo "REFUSED: $GP/PRESHEAD_STOP.txt exists"; exit 2; }
[ -f "$LT" ] && grep -qE '^[0-9]+ passed' "$LT" && ! grep -qiE 'failed|error' "$LT" || { echo "REFUSED: no passing lead test output $LT"; exit 4; }
for f in "$K/pres_fit.py" "$K/pres_dump.py" "$K/PRES_FIT_GEN_RECORD.json" "$JOIN" "$EJ" "$FR/lists/train_final.txt" "$FR/lists/eval141.txt"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
[ -d "$FR/tok/D" ] || stop "MISSING-INPUT $FR/tok/D"
adm=$("$PY" - "$K/PRES_FIT_GEN_RECORD.json" "$K/pres_fit.py" <<'PYEOF'
import hashlib, json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
h = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if d.get("pass_p0") is not True:
    print("NOT-PASS p0"); sys.exit(1)
if d.get("variant_sha256") != h:
    print("FIT-TOOL-HASH-IS-NOT-THAT-OF-THE-RECORD"); sys.exit(1)
print("ADMIT-OK generator record, fit tool %s" % h[:16])
PYEOF
) || stop "GENERATOR-RECORD $adm"
echo "$adm"
mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max 2>/dev/null); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null | head -n 1)
if [[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then [ $(( (mmax - anon) / 1073741824 )) -gt "$MINGB" ] || stop "MEMORY-FREE-TOO-LOW"; fi
mkdir -p "$GP/jobs" "$GP/logs" "$GP/fits" "$GP/dump" || stop "CANNOT-MAKE-FOLDERS"
( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/PRESHEAD_START.txt" ) 2>/dev/null || { echo "REFUSED: $GP/PRESHEAD_START.txt exists"; exit 2; }
T0=$(date +%s)
left() { echo $(( STOP_S - ( $(date +%s) - T0 ) )); }
step() { # NAME DONEFILE CMD...   runs CMD under the time that is left unless DONEFILE exists
  local j=$1 donef=$2; shift 2
  echo "$(ts) START $j" > "$GP/jobs/$j.START"
  if [ ! -f "$donef" ]; then
    local l; l=$(left)
    [ "$l" -gt 0 ] || { echo "$(ts) stop rule reached before $j" > "$GP/jobs/$j.STOP"; return 1; }
    timeout "$l" "$@" > "$GP/logs/$j.log" 2>&1 || { echo "$(ts) $j rc $?" > "$GP/jobs/$j.STOP"; return 1; }
  fi
  echo "$(ts) DONE $j" > "$GP/jobs/$j.DONE"
}
fail() { echo "$(ts) step $1 failed" >> "$GP/PRESHEAD_STOP.txt"; }
( step fit_s0 "$GP/fits/D_H5_s0/FIT_RECORD.json" $(fitcmd 0) || fail fit_s0 ) &
( step fit_s1 "$GP/fits/D_H5_s1/FIT_RECORD.json" $(fitcmd 1) || fail fit_s1 ) &
wait
[ -f "$GP/PRESHEAD_STOP.txt" ] && { echo "STOPPED: see $GP/PRESHEAD_STOP.txt"; exit 3; }
( step fit_s0_perm "$GP/fits/D_H5_s0_perm/FIT_RECORD.json" $(fitcmd 0 --permute-labels) || fail fit_s0_perm ) &
( step dump_s0 "$GP/dump/D_H5_s0/pres_meta.json" $(dumpcmd 0) || fail dump_s0 ) &
( step dump_s1 "$GP/dump/D_H5_s1/pres_meta.json" $(dumpcmd 1) || fail dump_s1 ) &
wait
[ -f "$GP/PRESHEAD_STOP.txt" ] && { echo "STOPPED: see $GP/PRESHEAD_STOP.txt"; exit 3; }
step dump_s0_perm "$GP/dump/D_H5_s0_perm/pres_meta.json" $(dumpcmd 0 --permute-labels) || { fail dump_s0_perm; echo "STOPPED: see $GP/PRESHEAD_STOP.txt"; exit 3; }
echo "ZZ PRESHEAD DONE $(ts) all steps done" > "$GP/PRESHEAD_DONE.txt"
echo "ZZ PRESHEAD DONE"
