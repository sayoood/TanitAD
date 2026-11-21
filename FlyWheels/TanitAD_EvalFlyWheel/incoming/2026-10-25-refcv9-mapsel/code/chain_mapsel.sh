#!/bin/bash
# chain_mapsel.sh -- refcv9 MAP-SELECT (PREREG_MAPSEL.md): ONE GPU chain on the shared dev-box RTX 4060 (gpu_lock).
#   identity checks (both seals) -> K_HUMAN + K_SENS (CPU, 0 GPU; STOP if the human is > 2 % off-drivable OR the +-3.5 m shifted human is not separated above it) -> free-commit gate -> gpu_lock acquire (non-FIFO; a foreign holder is NEVER broken; released in a trap)
#   -> 9-clip SMOKE roll + CPU controls gate -> FULL seed-0 roll (4,754 windows; budget probe at 150 windows, STOP if the projection exceeds 5,400 s) -> release -> CPU score -> VERDICT.
# * markers are opaque ZZMS<NAME>ZZ lines in logs/chain.log; every step asserts on its ARTIFACT (final trace npz with the expected window count), never on an exit code read through a pipe
# * heavy data on F: (BK); C: only logs/results; kills only the children it started; no git; OMP_NUM_THREADS=4; the battery dump dir of each roll is deleted after its trace is final.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_mapsel; CODE=$W/code; CH=${MS_CH:-$W/logs/chain.log}; BK=${MS_BK:-F:/refcv9_mapsel}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
GTROOT=E:/refcv6_eval_kit/data/sam3_gt_v3_eval
JOB=refcv9-mapsel
MINCOMMIT=${MINCOMMIT:-6}
BUDGET_S=5400
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$W/logs" "$W/result" "$(dirname "$CH")"
mk() { echo "ZZMS$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock.log" 2>&1; rm -f "$W/GPU_BUSY_MAPSEL"; echo "$1 $(date +%FT%T%z)" > "$W/GPU_DONE_MAPSEL"; mk GPUFREE "$1"; }
cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() {  # $1 trace npz -> the window count iff the record is final, else BAD / NOFILE
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"
}
mk START "pid $$"

# ---- 1. identities (no GPU) ---------------------------------------------------------------------------------------------------------------- #
cd "$W"
chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
seal=$(chk PREREG_MAPSEL.md PREREG_MAPSEL.sha256)
seal2=$(chk PREREG_MAPSEL_AMENDMENT1.md PREREG_MAPSEL_AMENDMENT1.sha256)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32)
( cd "$CODE" && sha256sum mapsel_lib.py mapsel_data.py mapsel_roll.py mapsel_score.py mapsel_human_control.py test_mapsel_lib.py chain_mapsel.sh refcv8_eval_remap_devbox_mapsel.json ) > "$W/CODE_SHA256_AT_START.txt" 2>/dev/null
mk IDENT "seal_prereg=$seal seal_amendment1=$seal2 ck=$m8 w0=$w0"
if [ "$seal" != 1 ] || [ "$seal2" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ]; then mk ABORT "identity check failed"; exit 4; fi
unit=$( ( cd "$CODE" && CUDA_VISIBLE_DEVICES=-1 "$PY" -X utf8 test_mapsel_lib.py 2>&1 ) | grep -c UNITPASS )
mk UNIT "unitpass=$unit"
[ "$unit" = 1 ] || { mk ABORT "unit tests of mapsel_lib failed"; exit 4; }

# ---- 2. K_HUMAN (sanity bound) + K_SENS (blind lateral-shift sensitivity), CPU: failing either => the metric / geometry is wrong => STOP, NO GPU ---------- #
hum_ok() { "$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/result/K_HUMAN.json" 2>/dev/null; }
if ! hum_ok; then
  ( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/mapsel_human_control.py" --out "$W/result/K_HUMAN.json" > "$W/logs/k_human.log" 2>&1 )
fi
if hum_ok; then
  mk KHUMAN "ok: $("$PY" -I -c "import json; r=json.load(open('$W/result/K_HUMAN.json')); s=r['sensitivity']; print('human_off_S6c=%.4f n=%d parity_m=%.2e | SENS plus=%.4f [%.4f,%.4f] minus=%.4f [%.4f,%.4f]' % (r['S6_compliant']['off_drivable'], r['S6_compliant']['n'], r['parity_vs_banked_S6_g_max_abs_m'], s['plus_left']['rate'], s['plus_left']['lo'], s['plus_left']['hi'], s['minus_right']['rate'], s['minus_right']['lo'], s['minus_right']['hi']))")"
else
  mk STOP "K_HUMAN (> 2 % off-drivable) or K_SENS (shifted human not separated above) failed: the metric / geometry is wrong (result/K_HUMAN.json); NO GPU was taken"; exit 3
fi

# ---- 3. free-commit gate (<= 3 h), then the shared GPU lock (<= 12 h) ------------------------------------------------------------------------- #
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$W/logs/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/logs/lock.json" >> "$W/logs/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock.json" || { mk NOLOCK "see logs/lock.log"; exit 3; }
HELD=1; echo "mapsel chain pid $$ $(date +%FT%T%z)" > "$W/GPU_BUSY_MAPSEL"
TLOCK=$(date +%s)
mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 4. the rolls ----------------------------------------------------------------------------------------------------------------------------- #
"$PY" -I -c "
import json
w = json.load(open('$BANKT/windows_s0.json')); ks = sorted(w)[:9]
json.dump({k: w[k] for k in ks}, open('$BK/windows_smoke.json', 'w'))
print(sum(len(w[k]) for k in ks))
" > "$BK/windows_smoke.n" 2>/dev/null
NSMOKE=$(cat "$BK/windows_smoke.n" | tr -d '\r\n ')
[ -n "$NSMOKE" ] || { mk ABORT "smoke windows"; exit 5; }
roll() {  # $1 tag  $2 windows json  $3 expected windows  $4 extra roll opts
  local tag="$1" wj="$2" want="$3" extra="${4:-}" TS n; TS=$(date +%s)
  rm -rf "$BK/dump_${tag}" "$BK/rast_${tag}"
  runbg "$PY" -X utf8 "$CODE/mapsel_roll.py" --trace-out "$BK/trace_${tag}.npz" --rast-dir "$BK/rast_${tag}" --map-gt-root "$GTROOT" --require-seal $extra -- \
      --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$BK/dump_${tag}" --windows-json "$wj" --device cuda --out-json "$BK/roll_${tag}.json" --os-only \
      > "$W/logs/roll_${tag}.log" 2>&1
  n=$(final_n "$BK/trace_${tag}.npz")
  LAST_WALL=$(( $(date +%s) - TS ))
  mk "ROLL_${tag}" "wall=${LAST_WALL}s windows=$n want=$want commit_free_gb=$(free_virt_gb)"
  [ "$n" = "$want" ] && rm -rf "$BK/dump_${tag}"
  [ "$n" = "$want" ]
}
roll smoke "$BK/windows_smoke.json" "$NSMOKE" || { mk ABORT "smoke artifact"; exit 7; }
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/mapsel_score.py" --seed 0 --trace "$BK/trace_smoke.npz" --rast "$BK/rast_smoke" --out "$W/result/smoke" --controls-only > "$W/logs/score_smoke.log" 2>&1 )
SMOKE_OK=$("$PY" -I -c "
import json
r = json.load(open('$W/result/smoke/MAPSEL_CONTROLS.json'))
print(1 if r.get('hard_controls_pass') is True else 0)
" 2>/dev/null)
mk SMOKEGATE "hard_controls_ok=$SMOKE_OK (K_G fields+raster shapes+decide parity, K_B, K_C, K_A == L1 BASE-os, K_RAST, K_LIVE own>deranged, K_DERANGED, K_BASE)"
[ "$SMOKE_OK" = 1 ] || { mk ABORT "smoke controls failed: see result/smoke/MAPSEL_CONTROLS.json"; exit 8; }
roll s0 "$BANKT/windows_s0.json" 4754 "--budget-probe-windows 150 --budget-max-s $BUDGET_S" || { mk ABORT "seed-0 artifact (see logs/roll_s0.log; exit 7 = over budget)"; exit 9; }
release "roll done"

# ---- 5. the mechanical score (CPU) ------------------------------------------------------------------------------------------------------------ #
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/mapsel_score.py" --seed 0 --trace "$BK/trace_s0.npz" --rast "$BK/rast_s0" --out "$W/result/s0" --human-json "$W/result/K_HUMAN.json" > "$W/logs/score_s0.log" 2>&1 )
if [ -s "$W/result/s0/VERDICT.md" ]; then
  cp "$W/result/s0/VERDICT.md" "$W/result/VERDICT.md"
  mk VERDICT "$(grep -o 'ZZMSVERDICTZZ.*' "$W/logs/score_s0.log" | tail -1 | tr -d '\r') -> result/VERDICT.md"
else
  mk VERDICTFAIL "see logs/score_s0.log"
fi
mk END ""
echo ZZMSENDZZ >> "$CH"
