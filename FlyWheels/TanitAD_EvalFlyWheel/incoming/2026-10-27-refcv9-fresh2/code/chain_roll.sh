#!/bin/bash
# chain_roll.sh -- FRESH2 step 3a: the z_tac roll of refcv8 EMA-48,852 over every usable FRESH2 window, EXACTLY chain_l1c.sh's roll() with the input paths swapped
# (kit_fresh2 / remap_fresh2.json / its v9 md5 / new bank + tree names).  Identities -> the shared dev-box GPU lock (gpu_lock.py; a foreign holder is NEVER broken)
# -> roll -> release.  Judged on the ARTIFACT (final bank with the expected window count).  Markers ZZF2R<NAME>ZZ in logs/chain_roll.log.  No git.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_fresh2; CODE=$W/code; CH=$W/logs/chain_roll.log; FB=F:/l1c/bank; FT=F:/l1c/tree
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-fresh2-ztac
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1 TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE R8_EVAL_KIT
mkdir -p "$FB" "$FT" "$W/logs"
mk() { echo "ZZF2R$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
CHILD=0; HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; mk GPUFREE "$1"; }
cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() { "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"; }
want_n() { "$PY" -I -c "import json,sys; print(sum(len(v) for v in json.load(open(sys.argv[1])).values()))" "$1"; }
mk START "pid $$"
cd "$W"
seal=$("$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w and g.startswith('956f87cd')) else 0)" PREREG_FRESH2.md PREREG_FRESH2.sha256 2>/dev/null || echo 0)
m8=$(md5sum "$CK" | cut -c1-32)
mk IDENT "seal_fresh2=$seal ck=$m8"
if [ "$seal" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ]; then mk ABORT "identity check failed"; exit 4; fi
KIT=F:/l1c/kit_fresh2; WJ=$KIT/windows_s0.json; TAG=fresh2; TREE=step48852ema_fresh2
md5=$(tr -d '\r\n ' < "$KIT/data/v9/v9_labels_eval.md5"); want=$(want_n "$WJ")
if [ "$(final_n "$FB/ztac_$TAG.npz")" = "$want" ]; then mk HAVE "$TAG $want"; exit 0; fi
rm -f "$W/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1
mk LOCKED "want $want windows"
rm -rf "$FT/$TREE/dump_s0"; TS=$(date +%s)
runbg "$PY" -X utf8 "$CODE/fresh2_ztac_roll.py" --mode eval --kit "$KIT" --remap "$CODE/remap_fresh2.json" --v9-md5 "$md5" --ztac-out "$FB/ztac_$TAG.npz" --require-seal -- \
    --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$FT/$TREE/dump_s0" --windows-json "$WJ" --device cuda \
    --out-json "$FB/roll_$TAG.json" --os-only > F:/l1c/fresh2/roll_$TAG.log 2>&1
n=$(final_n "$FB/ztac_$TAG.npz")
mk "ROLL" "wall=$(( $(date +%s) - TS ))s windows=$n want=$want"
release "roll done"
[ "$n" = "$want" ] || { mk ABORT "roll artifact: windows=$n want=$want (log F:/l1c/fresh2/roll_$TAG.log)"; exit 9; }
mk END "ok"
