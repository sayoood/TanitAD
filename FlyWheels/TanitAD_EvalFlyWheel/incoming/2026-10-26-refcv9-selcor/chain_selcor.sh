#!/usr/bin/env bash
# chain_selcor.sh -- PREREG_SELCOR.md: identities -> shared GPU lock -> fresh-clip LOCUS roll (seed 0) -> release.  Scoring is a separate CPU step.
# Adapted from refcv9_l2/selfsharp/chain_ss1.sh.  Markers ZZSC<TAG>ZZ in logs/chain.log; END always written.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l2c; CODE=$W/code; CH=$W/logs/chain.log; BK=${SC_BK:-F:/refcv9_selcor}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
KIT=F:/l1c/kit_fresh; REMAP=C:/Users/Admin/refcv8_pf/refcv9_l1c/code/remap_l1c_fresh.json; WJ=$KIT/windows_s0.json
JOB=refcv9-selcor-fresh
MINCOMMIT=${MINCOMMIT:-6}
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$W/logs"
mk() { echo "ZZSC$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0; STAGE=start
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; mk GPUFREE "$1"; }
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
want_n() { "$PY" -I -c "import json,sys; print(sum(len(v) for v in json.load(open(sys.argv[1])).values()))" "$1"; }
mk START "pid $$"

# ---- 1. identities (no GPU) ---- #
STAGE=identity
cd "$W"
chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
seal=$(chk PREREG_SELCOR.md PREREG_SELCOR.sha256)
m8=$(md5sum "$CK" | cut -c1-32); v9=$(tr -d '\r\n ' < "$KIT/data/v9/v9_labels_eval.md5"); want=$(want_n "$WJ")
mk IDENT "seal=$seal ck=$m8 v9=$v9 want=$want"
if [ "$seal" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$v9" != cc06d03a453f87f778e7555d5607a5aa ] || [ "$want" != 10456 ]; then
  mk ABORT "identity check failed"; exit 4; fi
if [ "${SC_DRY:-0}" = 1 ]; then mk DRYSTOP "SC_DRY=1: stopped before the lock"; exit 0; fi

# ---- 2. free-commit gate (<= 3 h), then the shared GPU lock (<= 12 h) ---- #
STAGE=lockwait
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$W/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1; mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 3. the fresh locus roll ---- #
STAGE=roll
TS=$(date +%s)
if [ "$(final_n "$BK/locus_fresh_s0.npz")" = "$want" ]; then mk HAVE locus_fresh; else
  rm -rf "$BK/dump_fresh_s0"
  runbg "$PY" -X utf8 "$CODE/l2c_locus_roll_fresh.py" --kit "$KIT" --remap "$REMAP" --v9-md5 "$v9" --trace-out "$BK/locus_fresh_s0.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$BK/dump_fresh_s0" --windows-json "$WJ" --device cuda --out-json "$BK/roll_fresh_s0.json" --os-only \
      > "$W/logs/roll_fresh_s0.log" 2>&1
fi
n=$(final_n "$BK/locus_fresh_s0.npz")
mk ROLL "wall=$(( $(date +%s) - TS ))s windows=$n want=$want"
[ "$n" = "$want" ] || { mk ABORT "locus_fresh artifact"; exit 7; }
release "roll done"
STAGE=done
exit 0
