#!/bin/bash
# chain_l1c.sh -- refcv9 L1C z_tac rolls (PREREG_L1C.md + AMENDMENT 1 [+ AMENDMENT 2]): identities -> free-commit gate -> the shared dev-box GPU lock
# (gpu_lock.py; a foreign holder is NEVER broken; wait <= 12 h) -> fresh roll (inference seed 0) -> platform-control roll (24 eval139 clips rebuilt on this box, the BANKED windows)
# -> release -> CPU: panels, fit + score, braking AUC, verdict, next-lever arms.  Every step asserts on its ARTIFACT (final bank npz with the expected window count), never on an exit code read
# through a pipe.  Markers: opaque ZZL1C<NAME>ZZ lines in $W/chain.log.  Kills only the child it started (trap).  Nothing on E:/G:/Thor.  No git.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l1c; CODE=$W/code; CH=$W/chain.log; FB=F:/l1c/bank; FT=F:/l1c/tree
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-l1c-ztac
MINCOMMIT=${MINCOMMIT:-6}
ONLY=${ONLY:-plat,fresh}
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1 TMP=F:/l1c/tmp TEMP=F:/l1c/tmp
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE R8_EVAL_KIT
mkdir -p "$FB" "$FT" "$W/logs" F:/l1c/tmp
mk() { echo "ZZL1C$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; rm -f "$W/GPU_BUSY_L1C"; echo "$1 $(date +%FT%T%z)" > "$W/GPU_DONE_L1C"; mk GPUFREE "$1"; }
cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() {  # $1 bank npz -> prints the window count iff the record is final, else BAD
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"
}
want_n() { "$PY" -I -c "import json,sys; print(sum(len(v) for v in json.load(open(sys.argv[1])).values()))" "$1"; }
mk START "pid $$ only=$ONLY"

# ---- 1. identities (no GPU) ------------------------------------------------------------------------------------------------------------ #
cd "$W"
sealA=$("$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" PREREG_L1C.md PREREG_L1C.sha256 2>/dev/null || echo 0)
sealB=$("$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" PREREG_L1C_AMENDMENT1.md PREREG_L1C_AMENDMENT1.sha256 2>/dev/null || echo 0)
sealC=$("$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" PREREG_L1C_AMENDMENT2.md PREREG_L1C_AMENDMENT2.sha256 2>/dev/null || echo 0)
m8=$(md5sum "$CK" | cut -c1-32)
mk IDENT "seal_prereg=$sealA seal_a1=$sealB seal_a2=$sealC ck=$m8"
if [ "$sealA" != 1 ] || [ "$sealB" != 1 ] || [ "$sealC" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ]; then mk ABORT "identity check failed"; exit 4; fi

# ---- 2. free-commit gate (wait <= 3 h), then the shared GPU lock (no other python compute app; wait <= 12 h) --------------------------- #
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$W/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1; echo "l1c chain pid $$ $(date +%FT%T%z)" > "$W/GPU_BUSY_L1C"
mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 3. the rolls ------------------------------------------------------------------------------------------------------------------- #
roll() {  # $1 tag  $2 kit  $3 remap  $4 windows-json  $5 tree-name
  local tag="$1" kit="$2" remap="$3" wj="$4" tree="$5" TS md5 want n
  md5=$(tr -d '\r\n ' < "$kit/data/v9/v9_labels_eval.md5"); want=$(want_n "$wj")
  if [ "$(final_n "$FB/ztac_$tag.npz")" = "$want" ]; then mk HAVE "$tag"; return 0; fi
  rm -rf "$FT/$tree/dump_s0"; TS=$(date +%s)
  runbg "$PY" -X utf8 "$CODE/l1c_ztac_roll.py" --mode eval --kit "$kit" --remap "$remap" --v9-md5 "$md5" --ztac-out "$FB/ztac_$tag.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$FT/$tree/dump_s0" --windows-json "$wj" --device cuda \
      --out-json "$FB/roll_$tag.json" --os-only > "$W/logs/roll_$tag.log" 2>&1
  n=$(final_n "$FB/ztac_$tag.npz")
  mk "ROLL_$tag" "wall=$(( $(date +%s) - TS ))s windows=$n want=$want commit_free_gb=$(free_virt_gb)"
  [ "$n" = "$want" ]
}
case ",$ONLY," in *,fresh,*)
  roll fresh F:/l1c/kit_fresh "$CODE/remap_l1c_fresh.json" F:/l1c/kit_fresh/windows_s0.json step48852ema_fresh || { mk ABORT "fresh roll artifact"; exit 9; } ;;
esac
case ",$ONLY," in *,plat,*)
  roll plat F:/l1c/kit_plat "$CODE/remap_l1c_plat.json" F:/l1c/kit_plat/windows_banked24.json step48852ema_plat || mk PLATFAIL "platform control roll artifact (diagnostic only; the chain continues)" ;;
esac
release "rolls done"

# ---- 4. CPU: panels -> fit + score -> braking AUC -> verdict -> next-lever arms (mechanical from the sealed texts) -------------------------------- #
R8=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
cpu() { ( export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$T/stack;$T/taniteval;$R8"; "$@" ); }
mkdir -p "$W/result"
if [ "$(final_n "$FB/ztac_fresh.npz")" = "$(want_n F:/l1c/kit_fresh/windows_s0.json)" ]; then
  cpu "$PY" -X utf8 "$CODE/l1c_panels.py" --root "$FT" --tree step48852ema_fresh > "$W/logs/panels_fresh.log" 2>&1
  if [ -s "$FT/step48852ema_fresh/panel_s0/manifest.json" ] && [ -s "$FT/step48852ema_fresh/s6_s0/manifest.json" ]; then mk PANELS "fresh panel_s0 + s6_s0"; else mk PANELFAIL "see logs/panels_fresh.log"; fi
  cpu "$PY" -X utf8 "$CODE/l1c_fit_score.py" --out "$W/result" > "$W/logs/fit_score.log" 2>&1
  if [ -s "$W/result/L1C_RESULT.json" ]; then
    mk FITTED "result/L1C_RESULT.json"
    cpu "$PY" -X utf8 "$CODE/l1c_roc.py" "$W/result" > "$W/logs/roc.log" 2>&1
    if [ -s "$W/result/L1C_ROC.json" ]; then mk ROC "result/L1C_ROC.json"; else mk ROCFAIL "see logs/roc.log"; fi
    cpu "$PY" -X utf8 "$CODE/l1c_verdict.py" "$W/result" > "$W/logs/verdict.log" 2>&1
    if [ -s "$W/result/VERDICT.json" ]; then mk VERDICT "result/VERDICT.json"; else mk VERDICTFAIL "see logs/verdict.log"; fi
    cpu "$PY" -X utf8 "$CODE/l1c_next.py" --out "$W/result" > "$W/logs/next.log" 2>&1
    if [ -s "$W/result/L1C_NEXT.json" ]; then mk NEXT "result/L1C_NEXT.json"; else mk NEXTFAIL "see logs/next.log"; fi
  else mk FITFAIL "see logs/fit_score.log"; fi
fi
mk END ""
