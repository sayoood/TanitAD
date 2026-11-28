#!/bin/bash
# chain_l1.sh -- refcv9 L1 z_tac bank (PREREG_L1.md, sealed): identity checks -> GPU quiet + free-commit gate -> the shared dev-box
# GPU lock (gpu_lock.py: O_EXCL lock file + no other python compute app; a foreign holder is never broken) -> train smoke (1 clip,
# 3 windows) -> eval seed 0 -> eval seed 1 -> train seed 0 -> release -> CPU fit + score (l1_fit_score.py).
# * every step asserts on its ARTIFACT (final bank npz with the expected window count), never on an exit code read through a pipe.
# * markers: opaque ZZL1<NAME>ZZ lines in $W/chain.log.  Kills only the child it started (trap).  Nothing on E:/F:/G:.  No git.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l1; CODE=$W/code; CH=$W/chain.log; BK=$W/bank
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
R8=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-l1-ztac
MINCOMMIT=${MINCOMMIT:-6}
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$W/logs"
mk() { echo "ZZL1$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; rm -f "$W/GPU_BUSY_L1"; echo "$1 $(date +%FT%T%z)" > "$W/GPU_DONE_L1"; mk GPUFREE "$1"; }
cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() {  # $1 bank npz -> prints the window count iff the record is final, else BAD
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"
}
mk START "pid $$"

# ---- 1. identities (no GPU) ------------------------------------------------------------------------------------------------------------ #
cd "$W"
seal_ok=$("$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" PREREG_L1.md PREREG_L1.sha256 2>/dev/null)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32); w1=$(md5sum "$BANKT/windows_s1.json" | cut -c1-32)
sa=$(sha256sum "$R8/roll_seed_r8_train.py" | cut -c1-64); sb=$(sha256sum "$CODE/train_shim/roll_seed_r8_train.py" | cut -c1-64)
mk IDENT "seal_ok=$seal_ok ck=$m8 w0=$w0 w1=$w1 shim_eq=$([ ${#sa} -eq 64 ] && [ "$sa" = "$sb" ] && echo 1 || echo 0)"
if [ "$seal_ok" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] \
   || [ "$w1" != a3a78dba889e70aa69a05271559794ad ] || [ ${#sa} -ne 64 ] || [ "$sa" != "$sb" ]; then mk ABORT "identity check failed"; exit 4; fi

# ---- 2. free-commit gate (wait <= 3 h), then the shared GPU lock (no other python compute app; wait <= 6 h) --------------------------- #
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$W/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 21600 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1; echo "l1 chain pid $$ $(date +%FT%T%z)" > "$W/GPU_BUSY_L1"
mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 3. train smoke: 1 clip, 3 windows (the hook, the shim, the replicated pf main) -------------------------------------------------- #
runbg "$PY" -X utf8 "$CODE/l1_ztac_roll.py" --mode train --ztac-out "$W/smoke/ztac_train_smoke.npz" --require-seal -- \
    --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$W/smoke/train_dump" --out-json "$W/smoke/train_roll.json" --device cuda \
    --max-clips 1 --max-windows-per-clip 3 > "$W/logs/smoke_train.log" 2>&1
n=$(final_n "$W/smoke/ztac_train_smoke.npz"); mk SMOKE "train windows=$n"
[ "$n" = "3" ] || { mk ABORT "train smoke artifact n=$n"; exit 7; }

# ---- 4. the three rolls ------------------------------------------------------------------------------------------------------------- #
roll() {  # $1 mode  $2 seed  $3 expected windows
  local tag="$1_s$2" TS; TS=$(date +%s)
  if [ "$(final_n "$BK/ztac_$tag.npz")" = "$3" ]; then mk HAVE "$tag"; return 0; fi
  rm -rf "$BK/${1}_dump_s$2"
  if [ "$1" = eval ]; then
    runbg "$PY" -X utf8 "$CODE/l1_ztac_roll.py" --mode eval --ztac-out "$BK/ztac_$tag.npz" --require-seal -- \
        --ckpt "$CK" --config "$CFG" --seed "$2" --dump-dir "$BK/eval_dump_s$2" --windows-json "$BANKT/windows_s$2.json" \
        --device cuda --out-json "$BK/eval_roll_s$2.json" --os-only > "$W/logs/roll_$tag.log" 2>&1
  else
    runbg "$PY" -X utf8 "$CODE/l1_ztac_roll.py" --mode train --ztac-out "$BK/ztac_$tag.npz" --require-seal -- \
        --ckpt "$CK" --config "$CFG" --seed "$2" --dump-dir "$BK/train_dump_s$2" --out-json "$BK/train_roll_s$2.json" \
        --device cuda > "$W/logs/roll_$tag.log" 2>&1
  fi
  local n; n=$(final_n "$BK/ztac_$tag.npz")
  mk "ROLL_$tag" "wall=$(( $(date +%s) - TS ))s windows=$n want=$3 commit_free_gb=$(free_virt_gb)"
  [ "$n" = "$3" ]
}
roll eval 0 4754 || { mk ABORT "eval s0 artifact"; exit 8; }
roll eval 1 4754 || { mk ABORT "eval s1 artifact"; exit 9; }
roll train 0 4855 || { mk ABORT "train s0 artifact"; exit 10; }
release "rolls done"

# ---- 5. CPU fit + score (mechanical from the sealed prereg) ------------------------------------------------------------------------- #
( export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$T/stack;$T/taniteval;$R8"
  "$PY" -X utf8 "$CODE/l1_fit_score.py" --bank "$BK" --out "$W/result" > "$W/logs/fit_score.log" 2>&1 )
if [ -s "$W/result/L1_RESULT.json" ]; then mk FITTED "result/L1_RESULT.json"; else mk FITFAIL "see logs/fit_score.log"; fi
mk END ""
