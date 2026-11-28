#!/bin/bash
# chain_pg.sh -- refcv9 PROBEGATE (PREREG_PROBEGATE.md): ONE detached GPU chain on the shared dev-box RTX 4060 (gpu_lock).
#   identity + seal + label-verification checks (0 GPU) -> free-commit gate -> gpu_lock acquire (non-FIFO; a foreign holder is NEVER broken; asserted on its --out JSON,
#   never the exit code; released in a trap) -> PROBEGATE s0 roll (4,754 windows) -> RANDGATE s0 roll iff the 1.6 h cap still holds -> release -> CPU mechanical score
#   -> result/VERDICT.md.   The roll is l2_roll.py --arm oracle (UNCHANGED, sha-pinned) with the label npz as the force table.
# * every roll is asserted on its ARTIFACT (final trace npz with 4,754 windows); markers are opaque ZZPG<NAME>ZZ lines in chain.log; ZZPGENDZZ is written by the EXIT trap
#   on EVERY path (and a VERDICT.md is written if the scorer never produced one).
# * heavy data on F: ($BK); C: only small text; kills only the children it started; no git; OMP_NUM_THREADS=4; each roll's dump dir is deleted after its trace is final.
# * env overrides for a DRY RUN: PG_DRY=1 (stop before the lock), PG_CH, PG_BK.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l2; PG=$W/probegate; CODE=$W/code; CH=${PG_CH:-$PG/chain.log}; BK=${PG_BK:-F:/refcv9_probegate}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-probegate
MINCOMMIT=${MINCOMMIT:-6}
CAP_S=${PG_CAP_S:-5760}          # 1.6 h of lock-held GPU time
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$PG/logs" "$PG/result" "$PG/bank" "$(dirname "$CH")"
rm -f "$PG/result/VERDICT.md"
mk() { echo "ZZPG$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0; T_LOCK=0; STAGE=start
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$PG/logs/lock.log" 2>&1; rm -f "$PG/GPU_BUSY_PG"; echo "$1 $(date +%FT%T%z)" > "$PG/GPU_DONE_PG"; mk GPUFREE "$1"; }
cleanup() {
  rc=$?
  if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi
  release "EXIT rc=$rc"
  if [ ! -s "$PG/result/VERDICT.md" ]; then printf 'NO VERDICT: the chain ended at stage %s (rc=%s) before the mechanical score produced one. See chain.log.\n' "$STAGE" "$rc" > "$PG/result/VERDICT.md"; fi
  mk END "rc=$rc stage=$STAGE verdict=$(head -1 "$PG/result/VERDICT.md" | tr -d '\r' | cut -c1-120)"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() {  # $1 trace npz -> the window count iff the record is final, else BAD / NOFILE
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"
}
mk START "pid $$"

# ---- 1. identities (no GPU) ---------------------------------------------------------------------------------------------------------------- #
STAGE=identity
cd "$PG"
chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
seal=$(chk PREREG_PROBEGATE.md PREREG_PROBEGATE.sha256)
seal_l2=$(cd "$W" && chk PREREG_L2.md PREREG_L2.sha256); seal_l2a=$(cd "$W" && chk PREREG_L2_AMENDMENT1.md PREREG_L2_AMENDMENT1.sha256)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32)
# every code file the prereg pins (CODE_PINS.txt = "<sha256>  <path relative to the L2 dir>"): re-hash now
pins_bad=$("$PY" -I -c "
import hashlib,sys
bad=[]
for ln in open('$PG/CODE_PINS.txt'):
    h,_,p=ln.strip().partition('  ')
    if h and hashlib.sha256(open('$W/'+p,'rb').read()).hexdigest()!=h: bad.append(p)
print(','.join(bad) if bad else 'NONE')" 2>/dev/null)
pins_in_prereg=$(grep -c "$(sha256sum "$PG/CODE_PINS.txt" | cut -c1-64)" "$PG/PREREG_PROBEGATE.md")
lab_ok=$("$PY" -I -c "
import json,hashlib
c=json.load(open('$PG/LABELS_CHECK.json')); v=json.load(open('$PG/LABELS_VERIFY.json'))
sh=all(hashlib.sha256(open('$PG/'+k,'rb').read()).hexdigest()==h for k,h in c['sha256'].items())
print(1 if (c.get('ALL_PASS') is True and v.get('ALL_PASS') is True and sh) else 0)" 2>/dev/null)
n_orc=$(final_n "$W/bank/trace_oracle_s0.npz")
mk IDENT "seal=$seal seal_l2=$seal_l2 seal_l2a=$seal_l2a ck=$m8 w0=$w0 pins_bad=$pins_bad pins_in_prereg=$pins_in_prereg labels_ok=$lab_ok oracle_trace_n=$n_orc"
if [ "$seal" != 1 ] || [ "$seal_l2" != 1 ] || [ "$seal_l2a" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] \
   || [ "$pins_bad" != NONE ] || [ "$pins_in_prereg" != 1 ] || [ "$lab_ok" != 1 ] || [ "$n_orc" != 4754 ]; then mk ABORT "identity check failed"; exit 4; fi
if [ "${PG_DRY:-0}" = 1 ]; then mk DRYSTOP "PG_DRY=1: stopped before the lock"; exit 0; fi

# ---- 2. free-commit gate (wait <= 3 h), then the shared GPU lock (wait <= 12 h) ------------------------------------------------------------- #
STAGE=lockwait
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$PG/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$PG/logs/lock.json" >> "$PG/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$PG/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1; T_LOCK=$(date +%s); echo "probegate chain pid $$ $(date +%FT%T%z)" > "$PG/GPU_BUSY_PG"
mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 3. the rolls ---------------------------------------------------------------------------------------------------------------------------- #
roll() {  # $1 tag (probegate|randgate)
  local tag="$1" TS n; TS=$(date +%s)
  STAGE="roll_$tag"
  if [ "$(final_n "$BK/trace_${tag}_s0.npz")" = 4754 ]; then mk HAVE "$tag"; return 0; fi
  rm -rf "$BK/dump_${tag}_s0"
  runbg "$PY" -X utf8 "$CODE/l2_roll.py" --arm oracle --labels-npz "$PG/labels_${tag}_s0.npz" --trace-out "$BK/trace_${tag}_s0.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$BK/dump_${tag}_s0" --windows-json "$BANKT/windows_s0.json" --device cuda --out-json "$BK/roll_${tag}_s0.json" --os-only \
      > "$PG/logs/roll_${tag}.log" 2>&1
  n=$(final_n "$BK/trace_${tag}_s0.npz")
  mk "ROLL_${tag}" "wall=$(( $(date +%s) - TS ))s windows=$n want=4754 commit_free_gb=$(free_virt_gb)"
  [ "$n" = 4754 ] && rm -rf "$BK/dump_${tag}_s0"
  [ "$n" = 4754 ]
}
roll probegate || { mk ABORT "probegate artifact"; exit 7; }
W1=$(( $(date +%s) - T_LOCK ))
if [ $(( W1 + W1 * 105 / 100 )) -le "$CAP_S" ]; then          # the second roll costs about what the first did (+5 %); stay inside the 1.6 h cap
  roll randgate || { mk ABORT "randgate artifact"; exit 8; }
else
  mk CAP "RANDGATE not rolled: lock-held ${W1}s + ~1.05x the first roll would exceed the ${CAP_S}s cap; P3 will read PENDING"
fi
release "rolls done"

# ---- 4. the mechanical score (CPU) ------------------------------------------------------------------------------------------------------------ #
STAGE=score
for f in trace_probegate_s0.npz trace_randgate_s0.npz roll_probegate_s0.json roll_randgate_s0.json; do [ -s "$BK/$f" ] && cp "$BK/$f" "$PG/bank/$f"; done
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$PG/pg_score.py" --out "$PG/result" --bank-pg "$BK" > "$PG/logs/score.log" 2>&1 )
if [ -s "$PG/result/PG_RESULT.json" ] && [ -s "$PG/result/VERDICT.md" ]; then mk SCORED "result/VERDICT.md"; else mk SCOREFAIL "see logs/score.log"; fi
STAGE=done
exit 0
