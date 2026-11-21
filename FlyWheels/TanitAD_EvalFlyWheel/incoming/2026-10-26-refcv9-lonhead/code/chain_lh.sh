#!/usr/bin/env bash
# chain_lh.sh -- PREREG_LONHEAD_C s2 rolls: identities -> for each of LHS s0, LHS s1, DER s0 (one at a time): free-commit gate -> shared GPU lock
# (gpu_lock.py: exclusive lock file + no other python compute app on nvidia-smi) -> one os-only oracle-label roll (4,754 windows) -> release.
# Adapted from refcv9_l2/selfsharp/chain_ss.sh / chain_ss1.sh (only the label file, seed, job name and paths differ). Markers ZZLH<TAG>ZZ; END always.
set -u
PK=C:/Users/Admin/refcv8_pf/refcv9_lonhead; CODE=C:/Users/Admin/refcv8_pf/refcv9_l2/code; CH=$PK/logs/chain_lh.log; BK=${LH_BK:-F:/refcv9_lonhead/rolls}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
MINCOMMIT=${MINCOMMIT:-6}
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1 TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$PK/logs"
mk() { echo "ZZLH$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0; STAGE=start; JOB=refcv9-lonhead
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$PK/logs/lock_lh.log" 2>&1; mk GPUFREE "$1"; }
cleanup() {
  rc=$?
  if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi
  release "EXIT rc=$rc"
  mk END "rc=$rc stage=$STAGE"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() { "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"; }
mk START "pid $$"

# ---- 1. identities (no GPU) ---- #
STAGE=identity
cd "$PK"
chk() { "$PY" -I -c "import hashlib,sys; w=sys.argv[2]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
regline() { awk -v f="$1" '$2=="*"f {print $1}' "$2"; }
seal=$(chk PREREG_LONHEAD_C.md 6c81d4b0510c463743ed814491710eea6e0f0384088ca296d20e39e444321048)
am=$(chk PREREG_LONHEAD_C_AMENDMENT1.md 715b9df1d78fed24fc8acc6a5f6363dc15265dd3656579ae531cff1cddebb2c8)
lh=$(chk LH_FROZEN.pt "$(regline LH_FROZEN.pt REGISTERED_LH.sha256)"); der=$(chk DER_FROZEN.pt "$(regline DER_FROZEN.pt REGISTERED_LH.sha256)")
llh=$(chk labels/labels_LHS.npz "$(regline labels/labels_LHS.npz labels/labels.sha256)"); lder=$(chk labels/labels_DER.npz "$(regline labels/labels_DER.npz labels/labels.sha256)")
go=$("$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(1 if (d.get('go') is True and d['controls']['ALL_PASS'] is True) else 0)" result/K_HEAD.json 2>/dev/null)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32); w1=$(md5sum "$BANKT/windows_s1.json" | cut -c1-32)
mk IDENT "seal=$seal amend=$am lh=$lh der=$der labLHS=$llh labDER=$lder khead_go=$go ck=$m8 w0=$w0 w1=$w1"
if [ "$seal" != 1 ] || [ "$am" != 1 ] || [ "$lh" != 1 ] || [ "$der" != 1 ] || [ "$llh" != 1 ] || [ "$lder" != 1 ] || [ "$go" != 1 ] \
   || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] || [ "$w1" != a3a78dba889e70aa69a05271559794ad ]; then
  mk ABORT "identity check failed"; exit 4; fi
if [ "${LH_DRY:-0}" = 1 ]; then mk DRYSTOP "LH_DRY=1: stopped before the lock"; exit 0; fi

roll_one() {   # $1 tag  $2 label file  $3 seed
  local tag=$1 lab=$2 sd=$3 name="${1}_s${3}"
  STAGE="lockwait_$name"
  if [ "$(final_n "$BK/trace_${name}.npz")" = 4754 ]; then mk HAVE "$name"; return 0; fi
  local t0=$(date +%s) i=0
  until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
    i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "$name free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
    [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "$name: free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
    sleep 30
  done
  mk NVSMI "$name $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d '\r\n')"
  rm -f "$PK/logs/lock_${name}.json"
  "$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$PK/logs/lock_${name}.json" >> "$PK/logs/lock_lh.log" 2>&1
  "$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$PK/logs/lock_${name}.json" || { mk NOLOCK "$name see logs/lock_lh.log"; exit 3; }
  HELD=1; mk LOCKED "$name free_commit_gb=$(free_virt_gb)"
  STAGE="roll_$name"
  local TS=$(date +%s)
  rm -rf "$BK/dump_${name}"
  runbg "$PY" -X utf8 "$CODE/l2_roll.py" --arm oracle --labels-npz "$PK/labels/$lab" --trace-out "$BK/trace_${name}.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed "$sd" --dump-dir "$BK/dump_${name}" --windows-json "$BANKT/windows_s${sd}.json" --device cuda --out-json "$BK/roll_${name}.json" --os-only \
      > "$PK/logs/roll_${name}.log" 2>&1
  local n=$(final_n "$BK/trace_${name}.npz")
  mk ROLL "$name wall=$(( $(date +%s) - TS ))s windows=$n want=4754"
  [ "$n" = 4754 ] || { mk ABORT "$name artifact"; exit 7; }
  rm -rf "$BK/dump_${name}"
  release "$name roll done"
}

roll_one LHS labels_LHS.npz 0
roll_one LHS labels_LHS.npz 1
roll_one DER labels_DER.npz 0
STAGE=done
mk ALLROLLS "LHS_s0 LHS_s1 DER_s0 banked in $BK"
exit 0
