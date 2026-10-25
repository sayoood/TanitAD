#!/bin/bash
# chain_l2.sh -- refcv9 L2 (PREREG_L2.md + Amendment 1): ONE GPU chain on the shared dev-box RTX 4060.
#   identity checks -> CPU prep (oracle labels, smoke windows) -> free-commit gate -> shared gpu_lock (O_EXCL; a foreign holder is never broken; asserted on its
#   --out JSON, never the exit code; released in a trap).  The lock wait already covers L1's three rolls, so L1's EMA-48,852 TRAIN bank exists when the lock is
#   taken: the CPU fit (TRAIN only) then runs CONCURRENTLY with the smokes + the CONST roll, which need no calibrator.
#   NATIVE-300 / IDENT-300 smokes -> CONST s0 -> [gate] -> CAL s0 -> [gate] -> ORACLE s0 -> CAL s1 (unless the gate says skip) -> release -> mechanical score.
# * every roll is asserted on its ARTIFACT (final trace npz with the expected window count); markers are opaque ZZL2<NAME>ZZ lines in logs/chain.log
# * kills only the children it started (trap); nothing on E:/F:/G:; no git; OMP_NUM_THREADS=4; each roll's dump dir is deleted after its trace is final.
# * env overrides for a DRY RUN: L2_DRY=1 (stop before the lock), L2_L1_DIR, L2_BK, L2_CH.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l2; CODE=$W/code; CH=${L2_CH:-$W/logs/chain.log}; BK=${L2_BK:-$W/bank}
L1=${L2_L1_DIR:-C:/Users/Admin/refcv8_pf/refcv9_l1}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
R8=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-l2-lonhead
MINCOMMIT=${MINCOMMIT:-6}
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$W/logs" "$W/result" "$(dirname "$CH")"
mk() { echo "ZZL2$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; PREP=0; HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; rm -f "$W/GPU_BUSY_L2"; echo "$1 $(date +%FT%T%z)" > "$W/GPU_DONE_L2"; mk GPUFREE "$1"; }
cleanup() { rc=$?; for p in $CHILD $PREP; do if [ "$p" != 0 ]; then kill "$p" 2>/dev/null; fi; done; sleep 3; for p in $CHILD $PREP; do if [ "$p" != 0 ]; then kill -9 "$p" 2>/dev/null; fi; done; release "EXIT rc=$rc"; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() {  # $1 trace npz -> the window count iff the record is final, else BAD / NOFILE
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"
}
mk START "pid $$"

# ---- 1. identities (no GPU) ------------------------------------------------------------------------------------------------------------- #
cd "$W"
chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
seal1=$(chk PREREG_L2.md PREREG_L2.sha256); seal2=$(chk PREREG_L2_AMENDMENT1.md PREREG_L2_AMENDMENT1.sha256)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32); w1=$(md5sum "$BANKT/windows_s1.json" | cut -c1-32)
( cd "$CODE" && sha256sum l2_common.py l2_fit.py l2_zero_gpu.py l2_roll.py l2_score.py chain_l2.sh refcv8_eval_remap_devbox_l2.json ) > "$W/CODE_SHA256_AT_START.txt" 2>/dev/null
mk IDENT "seal_prereg=$seal1 seal_amend1=$seal2 ck=$m8 w0=$w0 w1=$w1"
if [ "$seal1" != 1 ] || [ "$seal2" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] || [ "$w1" != a3a78dba889e70aa69a05271559794ad ]; then
  mk ABORT "identity check failed"; exit 4; fi

# ---- 2. CPU prep that needs nothing from L1: oracle labels + the 9-clip smoke windows ------------------------------------------------------- #
"$PY" -I -c "
import json
w = json.load(open('$BANKT/windows_s0.json')); ks = sorted(w)[:9]
json.dump({k: w[k] for k in ks}, open('$BK/windows_smoke.json', 'w'))
print(sum(len(w[k]) for k in ks))
" > "$BK/windows_smoke.n" 2>/dev/null
NSMOKE=$(cat "$BK/windows_smoke.n" | tr -d '\r\n ')
"$PY" -I -c "
import numpy as np
z = np.load('$BANKT/dump_s0/refcv6_extras.npz')
np.savez('$BK/oracle_labels.npz', clip_index=z['clip_index'], ws=z['ws'], lon_v7=z['lon_v7'])
" 2>/dev/null
mk PREP "n_smoke_windows=$NSMOKE labels=$([ -s $BK/oracle_labels.npz ] && echo ok || echo MISSING)"
[ -n "$NSMOKE" ] && [ -s "$BK/oracle_labels.npz" ] || { mk ABORT "prep failed"; exit 5; }

# ---- 3. the TRAIN-bank prep (copy -> fit -> 0-GPU readings), a function so it can run concurrently with the first rolls -------------------------- #
CAL_OK=0
train_prep() {
  local t0=$(date +%s)
  while :; do   # at lock time L1's train roll is done; a short grace covers marker lag
    if grep -q "ZZL1ROLL_train_s0ZZ" "$L1/chain.log" 2>/dev/null && [ "$(final_n "$L1/bank/ztac_train_s0.npz")" = "4855" ]; then break; fi
    grep -q -E "ZZL1(ABORT|STOP|END)ZZ" "$L1/chain.log" 2>/dev/null && { mk TRAINBANK "have=0 (L1 chain ended without a final train bank)"; return 1; }
    [ $(( $(date +%s) - t0 )) -gt ${L2_TRAIN_GRACE_S:-900} ] && { mk TRAINBANK "have=0 (not final after ${L2_TRAIN_GRACE_S:-900}s)"; return 1; }
    sleep 30
  done
  mk TRAINBANK "have=1"
  rm -rf "$BK/train48852" && mkdir -p "$BK/train48852"
  cp "$L1/bank/train_dump_s0/refcv6_extras.npz" "$BK/train48852/" && cp -r "$L1/bank/train_dump_s0/decisions" "$BK/train48852/decisions" || return 2
  ( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2_fit.py" fit --dump-dir "$BK/train48852" --out "$BK/calibrator_48852.json" --tag "EMA48852-train139-pf-s0" > "$W/logs/fit.log" 2>&1 )
  [ -s "$BK/calibrator_48852.json" ] || return 3
  ( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2_zero_gpu.py" --out "$W/result/zero_gpu_real" --calibrator "$BK/calibrator_48852.json" --tag "REAL EMA48852 calibrator" > "$W/logs/zero_gpu_real.log" 2>&1 )
  [ -s "$W/result/zero_gpu_real/ZERO_GPU.json" ] && mk ZEROGPU "result/zero_gpu_real" || mk ZEROGPUFAIL "see logs/zero_gpu_real.log"
  return 0
}
if [ "${L2_DRY:-0}" = 1 ]; then train_prep; mk DRYSTOP "train_prep rc=$? (L2_DRY=1: stopped before the lock)"; exit 0; fi

# ---- 4. free-commit gate (wait <= 3 h), then the shared GPU lock (wait <= 12 h) ------------------------------------------------------------ #
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$W/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1; echo "l2 chain pid $$ $(date +%FT%T%z)" > "$W/GPU_BUSY_L2"
mk LOCKED "free_commit_gb=$(free_virt_gb)"
train_prep & PREP=$!                                    # CPU only: concurrent with the smokes + CONST

# ---- 5. the rolls --------------------------------------------------------------------------------------------------------------------------- #
roll() {  # $1 arm  $2 seed  $3 windows json  $4 expected windows  $5 tag
  local arm="$1" seed="$2" wj="$3" want="$4" tag="$5" TS n; TS=$(date +%s)
  if [ "$(final_n "$BK/trace_${tag}.npz")" = "$want" ]; then mk HAVE "$tag"; return 0; fi
  rm -rf "$BK/dump_${tag}"
  local extra=()
  [ "$arm" = cal ] && extra=(--calibrator "$BK/calibrator_48852.json")
  [ "$arm" = oracle ] && extra=(--labels-npz "$BK/oracle_labels.npz")
  runbg "$PY" -X utf8 "$CODE/l2_roll.py" --arm "$arm" ${extra[@]+"${extra[@]}"} --trace-out "$BK/trace_${tag}.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed "$seed" --dump-dir "$BK/dump_${tag}" --windows-json "$wj" --device cuda --out-json "$BK/roll_${tag}.json" --os-only \
      > "$W/logs/roll_${tag}.log" 2>&1
  n=$(final_n "$BK/trace_${tag}.npz")
  mk "ROLL_${tag}" "wall=$(( $(date +%s) - TS ))s windows=$n want=$want commit_free_gb=$(free_virt_gb)"
  [ "$n" = "$want" ] && rm -rf "$BK/dump_${tag}"
  [ "$n" = "$want" ]
}
score() {  # $1 gate name -> prints the gate token
  ( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2_score.py" --bank "$BK" --out "$W/result/gate_$1" --gate "$1" > "$W/logs/gate_$1.log" 2>&1 )
  grep -o -E "(SKIP_CAL_S1|RUN_CAL_S1|CONST_BLOCKS|CONST_OPEN)" "$W/logs/gate_$1.log" | tail -1
}
roll native 0 "$BK/windows_smoke.json" "$NSMOKE" native300_s0 || { mk ABORT "native smoke artifact"; exit 7; }
roll ident 0 "$BK/windows_smoke.json" "$NSMOKE" ident300_s0 || { mk ABORT "ident smoke artifact"; exit 7; }
roll const_cruise 0 "$BANKT/windows_s0.json" 4754 const_cruise_s0 || { mk ABORT "const s0 artifact"; exit 8; }
wait $PREP; PREPRC=$?; PREP=0
[ -s "$BK/calibrator_48852.json" ] && CAL_OK=1
mk FIT "calibrator_ok=$CAL_OK prep_rc=$PREPRC"
g=$(score after_const); mk GATE "after_const=$g"
if [ $CAL_OK -eq 1 ]; then
  roll cal 0 "$BANKT/windows_s0.json" 4754 cal_s0 || { mk ABORT "cal s0 artifact"; exit 9; }
  g=$(score after_cal0); mk GATE "after_cal0=$g"
else g=SKIP_CAL_S1; mk GATE "cal skipped: NOTRAINBANK (PENDING)"; fi
roll oracle 0 "$BANKT/windows_s0.json" 4754 oracle_s0 || { mk ABORT "oracle s0 artifact"; exit 10; }
if [ $CAL_OK -eq 1 ] && [ "$g" = "RUN_CAL_S1" ]; then
  roll cal 1 "$BANKT/windows_s1.json" 4754 cal_s1 || { mk ABORT "cal s1 artifact"; exit 11; }
fi
release "rolls done"

# ---- 6. the mechanical score ----------------------------------------------------------------------------------------------------------------- #
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2_score.py" --bank "$BK" --out "$W/result/final" > "$W/logs/score_final.log" 2>&1 )
if [ -s "$W/result/final/L2_RESULT.json" ]; then mk SCORED "result/final"; else mk SCOREFAIL "see logs/score_final.log"; fi
mk END ""
