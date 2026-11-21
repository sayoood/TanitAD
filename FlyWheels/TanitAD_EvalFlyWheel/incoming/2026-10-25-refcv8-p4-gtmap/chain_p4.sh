#!/bin/bash
# chain_p4.sh -- P4 GT-MAP ORACLE (PREREG_P4.md, sealed): wait for the dev-box GPU, roll seed 1 (compute + cache delta-p), roll seed 0
# (read the cache), then the CPU analysis -> run/P4.json + run/RESULT.md.
# * waits until C:/Users/Admin/refcv8_pf/perc8tol/GPU_DONE exists AND no process command line names the NT gate's final-pinned script
#   (the searched token is ASSEMBLED FROM PIECES inside PowerShell, so this script never matches its own text); 2 consecutive polls,
#   every 120 s, gives up after 10 h with an opaque STOP marker.
# * then: identity checks (prereg seal, ckpt md5, windows md5, the unit test), GPU quiet (3 samples <= 20 %, no compute app), commit >= 8 GB.
# * every step asserts on its ARTIFACT (final npz with 4,754 windows), never on an exit code read through a pipe.
# * markers: opaque ZZP4<NAME>ZZ lines in chain.log.  Kills only the child it started (trap).
set -u
P=C:/Users/Admin/refcv8_pf/gtmap_oracle; CODE=$P/code; T=$P/tree_fb47a58; CH=$P/chain.log; RUN=$P/run
CACHE=F:/p4_gtmap/delta_cache
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OTHER_DONE=C:/Users/Admin/refcv8_pf/perc8tol/GPU_DONE
BANK=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
BUSY=$P/GPU_BUSY_P4; MYDONE=$P/GPU_DONE_P4
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES P4_ZERO_DELTA REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO \
      R8_MUTATION R7_MUTATION
mkdir -p "$RUN"
mk() { echo "ZZP4$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }   # de-DE would print 5,7
gate_live() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.ProcessId -ne \$PID -and \$_.CommandLine -match ('final_pinned_' + 'f') }).Count" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0
LOCKED=0
unlock() { [ $LOCKED -eq 1 ] || return; LOCKED=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$RUN/lock.log" 2>&1; }
release() { [ $HELD -eq 1 ] || return; HELD=0; rm -f "$BUSY"; echo "$1 $(date +%FT%T%z)" > "$MYDONE"; mk GPUFREE "$1"; }
cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; unlock; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
mk START "pid $$"

# ---- 0. the SHARED dev-box GPU lock (gpu_lock.py, the same one the L1 chain holds) -- MM 2026-10-25 10:3x CET -------------------- #
# The old handover (P2's GPU_DONE + the NT final-pinned process gone) did not see the L1 agent, which took the GPU at 10:15. Queue on the
# lock instead: it is exclusive (O_CREAT|O_EXCL) and also requires no other python compute app. Assert on the JSON, never the exit code.
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py; JOB=refcv8-p4-gtmap
rm -f "$RUN/lock.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$RUN/lock.json" >> "$RUN/lock.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$RUN/lock.json"   || { mk STOP "no GPU lock in 12 h -- see run/lock.log"; exit 3; }
LOCKED=1
mk HANDOVER "gpu_lock acquired job=$JOB commit_free_gb=$(free_virt_gb)"
fw=$(mkdir -p "$CACHE" && echo p4 > "$CACHE/.wtest" && cat "$CACHE/.wtest" && rm -f "$CACHE/.wtest")
[ "$fw" = p4 ] || { mk ABORT "cache dir $CACHE not writable (F: dropped this morning)"; exit 7; }

# ---- 1. identities (no GPU) ---------------------------------------------------------------------------------------------------------- #
cd "$P"
seal_ok=$("$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" PREREG_P4.md PREREG_P4.sha256 2>/dev/null)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANK/windows_s0.json" | cut -c1-32); w1=$(md5sum "$BANK/windows_s1.json" | cut -c1-32)
( cd "$CODE" && CUDA_VISIBLE_DEVICES=-1 "$PY" -X utf8 test_p4_pool_shortcut.py > "$RUN/unit_test.log" 2>&1 )
unit=$(grep -c "UNITPASS" "$RUN/unit_test.log" 2>/dev/null); unit=${unit:-0}
mk IDENT "seal_ok=$seal_ok ck=$m8 w0=$w0 w1=$w1 unit=$unit"
if [ "$seal_ok" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] \
   || [ "$w1" != a3a78dba889e70aa69a05271559794ad ] || [ "$unit" != 1 ]; then mk ABORT "identity check failed"; exit 4; fi

# ---- 2. GPU quiet (3 consecutive samples <= 20 % and no compute app) + commit >= 8 GB ------------------------------------------------ #
q=0; t1=$(date +%s)
while [ $q -lt 3 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci "python")   # WDDM lists desktop apps: count python only
  # MM 2026-10-25 06:5x CET: util is NOT a valid quiet signal on this box at night -- a desktop SCREENSAVER (Ribbons.scr)
  # + Blender hold it at 41-45 % with NO compute app, which aborted run 1 after 2 h. Quiet = no python compute app only.
  if [ -n "$u" ] && [ "$apps" -eq 0 ]; then q=$((q+1)); else q=0; fi
  [ $(( $(date +%s) - t1 )) -gt 7200 ] && { mk ABORT "GPU never quiet util=$u apps=$apps"; exit 5; }
  sleep 20
done
t0=$(date +%s); until awk "BEGIN{exit !($(free_virt_gb)+0 >= 8)}"; do [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk ABORT "commit stayed < 8 GB for 3 h"; exit 6; }; sleep 30; done
echo "p4 chain pid $$ $(date +%FT%T%z)" > "$BUSY"; HELD=1
mk GPUQUIET "util=$u apps=$apps commit_free_gb=$(free_virt_gb)"

final_ok() {  # $1 npz -> prints OK iff the final record covers 4,754 windows
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1],allow_pickle=True); m=json.loads(str(z['meta_json'])); print('OK' if (m.get('final') is True and m.get('n_windows')==4754) else 'BAD n=%s final=%s' % (m.get('n_windows'), m.get('final')))" "$1" 2>/dev/null || echo "BAD no-artifact"
}
INV="--inv-steps 80 --inv-lr-rel 0.05 --inv-lambda 0.01 --inv-amp bf16 --inv-chunk 3 --save-every 250 --require-seal"
COMMON="--map-gt-root E:/refcv6_eval_kit/data/sam3_gt_v3_eval --remap-json $CODE/refcv8_eval_remap_devbox_p4.json"

# ---- 3. seed 1: compute (the inversion) + cache delta-p; a budget probe after 150 windows (stop above 4.5 h for both seeds) ----------- #
mkdir -p "$CACHE"
TS=$(date +%s)
runbg "$PY" -X utf8 "$CODE/p4_roll.py" --p4-out "$RUN/p4_s1.npz" --delta-mode compute --delta-cache "$CACHE" $COMMON $INV \
    --budget-probe-windows 150 --budget-max-s 30000 -- \
    --ckpt "$CK" --config "$CFG" --seed 1 --dump-dir "$RUN/dump_s1" --windows-json "$BANK/windows_s1.json" --device cuda \
    --out-json "$RUN/roll_s1.json" --os-only > "$RUN/roll_s1.log" 2>&1
RC=$?
OK1=$(final_ok "$RUN/p4_s1.npz")
ob=$(grep -c "OVERBUDGET" "$RUN/roll_s1.log" 2>/dev/null); ob=${ob:-0}
mk ROLL_S1 "rc=$RC wall=$(( $(date +%s) - TS ))s artifact=$OK1 over=$ob $(grep -h 'ZZP4BUDGETZZ' "$RUN/roll_s1.log" | tail -1 | cut -c1-160)"
if [ "$ob" -ge 1 ]; then mk HALTBUDGET "projected wall above 8.3 GPU-h (Amendment 1): design + cost reported instead"; exit 7; fi
[ "$OK1" = "OK" ] || { mk ABORT "seed 1 artifact"; exit 8; }

# ---- 4. seed 0: read the cache (both seeds receive the identical injected tensor) ------------------------------------------------------ #
TS=$(date +%s)
runbg "$PY" -X utf8 "$CODE/p4_roll.py" --p4-out "$RUN/p4_s0.npz" --delta-mode read --delta-cache "$CACHE" $COMMON $INV -- \
    --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$RUN/dump_s0" --windows-json "$BANK/windows_s0.json" --device cuda \
    --out-json "$RUN/roll_s0.json" --os-only > "$RUN/roll_s0.log" 2>&1
RC=$?
OK0=$(final_ok "$RUN/p4_s0.npz")
mk ROLL_S0 "rc=$RC wall=$(( $(date +%s) - TS ))s artifact=$OK0"
release "rolls done s1=$OK1 s0=$OK0"
[ "$OK0" = "OK" ] || { mk ABORT "seed 0 artifact"; exit 9; }

# ---- 5. the CPU analysis (mechanical from the sealed prereg) ----------------------------------------------------------------------------- #
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/p4_analyze.py" --root "$RUN" --seeds 0,1 > "$RUN/analyze.log" 2>&1 )
V=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['verdict'][:160])" "$RUN/P4.json" 2>/dev/null || echo "no-P4.json")
if [ -s "$RUN/RESULT.md" ] && [ -s "$RUN/P4.json" ]; then mk ANALYZED "$V"; else mk ANALYZEFAIL "see run/analyze.log"; fi
mk END ""
