#!/bin/bash
# chain_l2b.sh -- refcv9 L2-B (PREREG_L2B.md): the brake-LOCUS roll, ONE GPU chain on the shared dev-box RTX 4060 (gpu_lock).
#   identity checks -> free-commit gate -> gpu_lock acquire (non-FIFO; a foreign holder is NEVER broken; released in a trap) -> 9-clip SMOKE (field + control gate)
#   -> FULL seed-0 roll (4,754 windows, asserted on the FINAL trace's window count) -> seed 1 iff the 1.5 h budget rule holds -> release -> CPU score (seeds rolled) -> verdict.
# * markers are opaque ZZL2B<NAME>ZZ lines in logs/chain_l2b.log; assert on the ARTIFACT (final trace npz), never on an exit code
# * heavy data on F: (BK); C: only logs/results; kills only the children it started; no git; OMP_NUM_THREADS=4; each roll's battery dump dir is deleted after its trace is final.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l2; CODE=$W/code; CH=${L2B_CH:-$W/logs/chain_l2b.log}; BK=${L2B_BK:-F:/refcv9_l2_locus}
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
LOCKPY=$T/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BANKT=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
JOB=refcv9-l2b-locus
MINCOMMIT=${MINCOMMIT:-6}
BUDGET_S=5400
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION PF_R8_CODE
mkdir -p "$BK" "$W/logs" "$W/result" "$(dirname "$CH")"
mk() { echo "ZZL2B$1ZZ $(date +%FT%T%z) $2" >> "$CH"; }
free_virt_gb() { powershell -NoProfile -Command "([math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)).ToString([cultureinfo]::InvariantCulture)" 2>/dev/null | tr -d '\r\n '; }
CHILD=0; HELD=0
release() { [ $HELD -eq 1 ] || return; HELD=0; "$PY" "$LOCKPY" release --job "$JOB" --pid $$ >> "$W/logs/lock_l2b.log" 2>&1; rm -f "$W/GPU_BUSY_L2B"; echo "$1 $(date +%FT%T%z)" > "$W/GPU_DONE_L2B"; mk GPUFREE "$1"; }
cleanup() { rc=$?; if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 3; kill -9 "$CHILD" 2>/dev/null; fi; release "EXIT rc=$rc"; }
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
final_n() {  # $1 trace npz -> the window count iff the record is final, else BAD / NOFILE
  "$PY" -I -c "import json,sys,numpy as np; z=np.load(sys.argv[1]); m=json.loads(str(z['meta_json'])); print(m['n_windows'] if m.get('final') is True else 'BAD')" "$1" 2>/dev/null || echo "NOFILE"
}
mk START "pid $$"

# ---- 1. identities (no GPU) -------------------------------------------------------------------------------------------------------------- #
cd "$W"
chk() { "$PY" -I -c "import hashlib,sys; w=open(sys.argv[2]).read().split()[0]; g=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(1 if (len(w)==64 and g==w) else 0)" "$1" "$2" 2>/dev/null; }
seal=$(chk PREREG_L2B.md PREREG_L2B.sha256)
m8=$(md5sum "$CK" | cut -c1-32); w0=$(md5sum "$BANKT/windows_s0.json" | cut -c1-32); w1=$(md5sum "$BANKT/windows_s1.json" | cut -c1-32)
( cd "$CODE" && sha256sum l2_common.py l2_score.py l2b_locus_roll.py l2b_score.py l2b_verdict.py chain_l2b.sh refcv8_eval_remap_devbox_l2.json ) > "$W/CODE_SHA256_L2B_AT_START.txt" 2>/dev/null
mk IDENT "seal_prereg_l2b=$seal ck=$m8 w0=$w0 w1=$w1"
if [ "$seal" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$w0" != a3a78dba889e70aa69a05271559794ad ] || [ "$w1" != a3a78dba889e70aa69a05271559794ad ]; then
  mk ABORT "identity check failed"; exit 4; fi
"$PY" -I -c "
import json
w = json.load(open('$BANKT/windows_s0.json')); ks = sorted(w)[:9]
json.dump({k: w[k] for k in ks}, open('$BK/windows_smoke.json', 'w'))
print(sum(len(w[k]) for k in ks))
" > "$BK/windows_smoke.n" 2>/dev/null
NSMOKE=$(cat "$BK/windows_smoke.n" | tr -d '\r\n ')
[ -n "$NSMOKE" ] || { mk ABORT "smoke windows"; exit 5; }

# ---- 2. free-commit gate (<= 3 h), then the shared GPU lock (<= 12 h) --------------------------------------------------------------------- #
t0=$(date +%s); i=0
until awk "BEGIN{exit !($(free_virt_gb)+0 >= $MINCOMMIT)}"; do
  i=$((i+1)); [ $((i % 20)) -eq 1 ] && mk COMMITWAIT "free_commit_gb=$(free_virt_gb) need=$MINCOMMIT"
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { mk STOP "free commit stayed below ${MINCOMMIT} GB for 3 h (blocker)"; exit 6; }
  sleep 30
done
rm -f "$W/logs/lock_l2b.json"
"$PY" "$LOCKPY" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/logs/lock_l2b.json" >> "$W/logs/lock_l2b.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$W/logs/lock_l2b.json" || { mk NOLOCK "see logs/lock_l2b.log"; exit 3; }
HELD=1; echo "l2b chain pid $$ $(date +%FT%T%z)" > "$W/GPU_BUSY_L2B"
TLOCK=$(date +%s)
mk LOCKED "free_commit_gb=$(free_virt_gb)"

# ---- 3. the rolls ------------------------------------------------------------------------------------------------------------------------- #
roll() {  # $1 seed  $2 windows json  $3 expected windows  $4 tag
  local seed="$1" wj="$2" want="$3" tag="$4" TS n; TS=$(date +%s)
  rm -rf "$BK/dump_${tag}"
  runbg "$PY" -X utf8 "$CODE/l2b_locus_roll.py" --trace-out "$BK/locus_${tag}.npz" --require-seal -- \
      --ckpt "$CK" --config "$CFG" --seed "$seed" --dump-dir "$BK/dump_${tag}" --windows-json "$wj" --device cuda --out-json "$BK/roll_${tag}.json" --os-only \
      > "$W/logs/roll_l2b_${tag}.log" 2>&1
  n=$(final_n "$BK/locus_${tag}.npz")
  LAST_WALL=$(( $(date +%s) - TS ))
  mk "ROLL_${tag}" "wall=${LAST_WALL}s windows=$n want=$want commit_free_gb=$(free_virt_gb)"
  [ "$n" = "$want" ] && rm -rf "$BK/dump_${tag}"
  [ "$n" = "$want" ]
}
# 3a. the 9-clip smoke is the gate for the field capture + the hard controls (K_G, K_B, K_C) BEFORE the 40-minute roll
roll 0 "$BK/windows_smoke.json" "$NSMOKE" smoke_s0 || { mk ABORT "smoke artifact"; exit 7; }
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2b_score.py" --seed 0 --trace "$BK/locus_smoke_s0.npz" --out "$W/result/l2b_smoke" --controls-only > "$W/logs/score_l2b_smoke.log" 2>&1 )
SMOKE_OK=$("$PY" -I -c "
import json
c = json.load(open('$W/result/l2b_smoke/L2B_CONTROLS.json'))['controls']
ok = all(c[k].get('pass') is True for k in ('K_G_fields', 'K_B_emitted_equals_fan_at_sel_idx', 'K_C_sel_idx_is_argmax_over_e9_eligible', 'K_A_reroll_equals_L1_BASEos'))
print(1 if ok else 0)
" 2>/dev/null)
mk SMOKEGATE "controls_ok=$SMOKE_OK (K_G fields, K_B emitted==fan[sel], K_C argmax over e9-eligible, K_A == L1 BASE-os)"
[ "$SMOKE_OK" = 1 ] || { mk ABORT "smoke controls failed: see result/l2b_smoke/L2B_CONTROLS.json"; exit 8; }
# 3b. seed 0, full
roll 0 "$BANKT/windows_s0.json" 4754 s0 || { mk ABORT "seed-0 artifact"; exit 9; }
WALL0=$LAST_WALL
# 3c. seed 1 iff elapsed-since-lock + 1.05 x wall0 fits the budget
ELAPSED=$(( $(date +%s) - TLOCK ))
if awk "BEGIN{exit !($ELAPSED + 1.05*$WALL0 <= $BUDGET_S)}"; then
  mk SEED1 "run: elapsed=${ELAPSED}s wall0=${WALL0}s budget=${BUDGET_S}s"
  roll 1 "$BANKT/windows_s1.json" 4754 s1 || mk SEED1FAIL "seed-1 artifact not final (seed 1 PENDING)"
else
  mk SEED1 "skipped: elapsed=${ELAPSED}s + 1.05 x wall0=${WALL0}s > budget=${BUDGET_S}s (seed 1 PENDING, never inferred)"
fi
release "rolls done"

# ---- 4. the mechanical scores (CPU) -------------------------------------------------------------------------------------------------------- #
for s in 0 1; do
  if [ "$(final_n "$BK/locus_s${s}.npz")" = 4754 ]; then
    ( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2b_score.py" --seed $s --trace "$BK/locus_s${s}.npz" --out "$W/result/l2b_s${s}" > "$W/logs/score_l2b_s${s}.log" 2>&1 )
    mk "SCORED_s${s}" "$(grep -o 'ZZL2BLABELZZ.*' "$W/logs/score_l2b_s${s}.log" | tail -1)"
  fi
done
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/l2b_verdict.py" --root "$W/result" --out "$W/result/l2b_final" > "$W/logs/verdict_l2b.log" 2>&1 )
[ -s "$W/result/l2b_final/VERDICT.md" ] && mk VERDICT "result/l2b_final/VERDICT.md" || mk VERDICTFAIL "see logs/verdict_l2b.log"
mk END ""
