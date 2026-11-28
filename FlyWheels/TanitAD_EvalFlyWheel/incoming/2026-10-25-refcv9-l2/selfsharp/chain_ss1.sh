#!/usr/bin/env bash
# chain_ss1.sh -- SSGUARD seed-1 replicate (ssgate/PREREG_SSGUARD_S1.md): identities -> shared GPU lock -> SELFSHARP os-only roll at seed 1 -> release -> ssguard_score --seed 1.
# Adapted from probegate/chain_pg.sh.  Markers ZZSS<TAG>ZZ in chain.log; END always written.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l2; SS=$W/selfsharp; CODE=$W/code; CH=$SS/logs/chain_s1.log; BK=${SS_BK:-F:/refcv9_selfsharp}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-selfsharp-s1
MINCOMMIT=${MINCOMMIT:-6}
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$SS/logs" "$SS/result" "$SS/bank"
rm -f "$SS/ssgate/result_guard/VERDICT_s1.md"
mk() { echo "ZZSS$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0; STAGE=start
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$SS/logs/lock_s1.log" 2>&1; mk GPUFREE "$1"; }
cleanup() {
  rc=$?
  if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi
  release "EXIT rc=$rc"
  if [ ! -s "$SS/ssgate/result_guard/VERDICT_s1.md" ]; then printf 'NO VERDICT: the chain ended at stage %s (rc=%s) before the mechanical score produced one. See logs/chain.log.\n' "$STAGE" "$rc" > "$SS/ssgate/result_guard/VERDICT_s1.md"; fi
  mk END "rc=$rc stage=$STAGE verdict=$(head -1 "$SS/ssgate/result_guard/VERDICT_s1.md" | tr -d '\r' | cut -c1-120)"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() { "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"; }
mk START "pid $$"

# ---- 1. identities (no GPU) ---- #
STAGE=identity
cd "$SS"
chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
seal=$(chk ssgate/PREREG_SSGUARD_S1.md ssgate/PREREG_SSGUARD_S1.sha256)
lab=$(chk labels_selfsharp_s0.npz labels.sha256)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s1.json" | cut -c1-32)
n_orc=$(final_n "$W/bank/trace_oracle_s0.npz")
mk IDENT "seal=$seal labels=$lab ck=$m8 w0=$w0 oracle_trace_n=$n_orc"
if [ "$seal" != 1 ] || [ "$lab" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] || [ "$n_orc" != 4754 ]; then
  mk ABORT "identity check failed"; exit 4; fi
if [ "${SS_DRY:-0}" = 1 ]; then mk DRYSTOP "SS_DRY=1: stopped before the lock"; exit 0; fi

# ---- 2. free-commit gate (<= 3 h), then the shared GPU lock (<= 12 h) ---- #
STAGE=lockwait
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$SS/logs/lock_s1.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$SS/logs/lock_s1.json" >> "$SS/logs/lock_s1.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$SS/logs/lock_s1.json" || { mk NOLOCK "see logs/lock_s1.log"; exit 3; }
HELD=1; mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 3. the roll ---- #
STAGE=roll
TS=$(date +%s)
if [ "$(final_n "$BK/trace_selfsharp_s1.npz")" = 4754 ]; then mk HAVE selfsharp; else
  rm -rf "$BK/dump_selfsharp_s1"
  runbg "$PY" -X utf8 "$CODE/l2_roll.py" --arm oracle --labels-npz "$SS/labels_selfsharp_s0.npz" --trace-out "$BK/trace_selfsharp_s1.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed 1 --dump-dir "$BK/dump_selfsharp_s1" --windows-json "$BANKT/windows_s1.json" --device cuda --out-json "$BK/roll_selfsharp_s1.json" --os-only \
      > "$SS/logs/roll_selfsharp_s1.log" 2>&1
fi
n=$(final_n "$BK/trace_selfsharp_s1.npz")
mk ROLL "wall=$(( $(date +%s) - TS ))s windows=$n want=4754"
[ "$n" = 4754 ] || { mk ABORT "selfsharp artifact"; exit 7; }
rm -rf "$BK/dump_selfsharp_s1"
release "roll done"

# ---- 4. the mechanical score (CPU) ---- #
STAGE=score
for f in trace_selfsharp_s1.npz roll_selfsharp_s1.json; do [ -s "$BK/$f" ] && cp "$BK/$f" "$SS/bank/$f"; done
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$SS/ssgate/ssguard_score.py" --seed 1 --out "$SS/ssgate/result_guard" --bank-ss "$BK" > "$SS/logs/score_s1.log" 2>&1 )
if [ -s "$SS/ssgate/result_guard/SSGUARD_RESULT_s1.json" ] && grep -q ZZSSGUARDENDZZ "$SS/ssgate/result_guard/VERDICT_s1.md"; then mk SCORED "ssgate/result_guard/VERDICT_s1.md"; else mk SCOREFAIL "see logs/score.log"; fi
STAGE=done
exit 0
