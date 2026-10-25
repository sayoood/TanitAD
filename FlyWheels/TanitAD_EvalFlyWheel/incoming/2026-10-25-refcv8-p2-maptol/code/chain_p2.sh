#!/bin/bash
# chain_p2.sh -- P2 (boundary-tolerant map read): wait for the NT gate to finish, then the GPU passes (preflight, refcv8, refcv7), then the analysis.
# Markers: opaque ZZP2<NAME>ZZ tokens -> $P2/chain.log (counts are computed here; the searched-for gate token is assembled from pieces and read
# from the GATE's log only, so nothing here can match its own echoed text).  GPU_DONE is written when the GPU is released, whatever the outcome
# (content = the outcome).  Never reads $? through a pipe; evidence = artifacts.  Kills only PIDs it started.
source /c/Users/Admin/refcv8_pf/perc8tol/code/p2_env.sh
CH=$P2/chain.log
GATELOG=/c/Users/Admin/refcv8_pf/nt_build/launch/final_pinned_54c6028397c8.log
GPUDONE=$P2/GPU_DONE
mk() { echo "ZZP2$1ZZ $(date +%FT%T%z) $2" >> $CH; }
RELEASED=0
release_gpu() { [ $RELEASED -eq 1 ] && return; RELEASED=1; echo "$1 $(date +%FT%T%z)" > $GPUDONE; mk GPUFREE "$1"; }
trap 'release_gpu "EXIT_without_gpu_pass_or_aborted"' EXIT
mk START "pid $$"

# ---- 0. wait for the NT gate: its end-marker in ITS log AND no process whose command line names the gate script; 2 consecutive polls ---------- #
TOK="ZZFPEND""ZZ"
good=0
for i in $(seq 1 180); do
  endn=$(grep -c "$TOK" $GATELOG 2>/dev/null); endn=${endn:-0}
  live=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match ('final_pinned_' + 'f') }).Count" 2>/dev/null | tr -d '\r\n ')
  case "$live" in ''|*[!0-9]*) live=-1;; esac
  if [ "$endn" -ge 1 ] && [ "$live" -eq 0 ]; then good=$((good+1)); else good=0; fi
  [ $((i % 10)) -eq 1 ] && mk WAIT "poll=$i ends=$endn live=$live good=$good commit_free_gb=$(free_virt_gb)"
  [ $good -ge 2 ] && break
  sleep 120
done
if [ $good -lt 2 ]; then mk STOP "gave up after $i polls: ends=$endn live=$live"; release_gpu "NOT_RUN_gate_wait_timeout"; exit 3; fi
mk GATEOVER "polls=$i ends=$endn live=$live"

# ---- 1. identities (no GPU) ---------------------------------------------------------------------------------------------------------------- #
cd $P2
pre_ok=$(sha256sum -c PREREG_P2.md.sha256 2>/dev/null | grep -c ": OK")
m8=$(md5sum /e/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt | cut -c1-32); m7=$(md5sum /e/refcv7_eval_kit/ckpt/ckpt_50400.pt | cut -c1-32)
mw=$(md5sum /e/refcv8_eval_kit/windows/s2_windows.json | cut -c1-32)
evn=$(grep -c done raw/md5_p2_evalcache.done 2>/dev/null)
evok=$($PY -I -X utf8 $P2/code/p2_evalcache_check.py 2>/dev/null)
mk IDENT "prereg_ok=$pre_ok ck8=$m8 ck7=$m7 win=$mw evalcache_done=$evn evalcache_total_equal=$evok"
if [ "$pre_ok" != 1 ] || [ "$m8" != b4364029b68f29a33eeb4aebbbbd002b ] || [ "$m7" != b418d0fc4a92a6848c246a6a7c50207b ] || [ "$mw" != a3a78dba889e70aa69a05271559794ad ] || [ "$evok" != "139 139" ]; then
  mk ABORT "identity check failed"; release_gpu "ABORT_identity_check"; exit 4; fi

# ---- 2. GPU quiet (3 consecutive samples <= 20 % like the gate), enough commit ------------------------------------------------------------- #
q=0; t1=$(date +%s)
while [ $q -lt 3 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
  if [ -n "$u" ] && [ "$u" -le 20 ]; then q=$((q+1)); else q=0; fi
  [ $(( $(date +%s) - t1 )) -gt 28800 ] && { mk ABORT "GPU never quiet"; release_gpu "ABORT_gpu_busy"; exit 5; }
  sleep 20
done
mk GPUQUIET "util=$u commit_free_gb=$(free_virt_gb)"

# ---- 3. one pass (with a commit wait + one retry); assert on the ARTIFACT ------------------------------------------------------------------ #
wait_commit() { t0=$(date +%s); until awk "BEGIN{exit !($(free_virt_gb)+0 >= 8)}"; do [ $(( $(date +%s) - t0 )) -gt 1200 ] && return 1; sleep 20; done; return 0; }
pass_ok() {  # $1 = out dir, $2 = arm, $3 = min map windows  -> prints 1 iff the artifact exists, is complete and every in-pass control passed
  $PY -I -X utf8 $P2/code/p2_pass_ok.py "$1/p2_$2.json" "$3"
}
run_pass() {  # $1 label, $2 ckpt, $3 config, $4 out dir, $5 arm, $6.. extra args ; sets PASS_OK
  lab=$1; ck=$2; cf=$3; out=$4; arm=$5; shift 5; PASS_OK=0
  for attempt in 1 2; do
    wait_commit || mk COMMITLOW "$lab attempt=$attempt free_gb=$(free_virt_gb)"
    mkdir -p $out
    mk PASSBEGIN "$lab attempt=$attempt"
    ( cd $P2/code; export R8_REMAP_FILE=${RF:-refcv8_eval_remap_p2.json}; exec $PY -X utf8 p2_tol_pass.py --ckpt $ck --config $cf --windows-json $WIN --arm $arm --out-dir $out "$@" ) > $P2/logs/pass_${lab}_a${attempt}.log 2>&1 < /dev/null &
    PID=$!
    t0=$(date +%s)
    while kill -0 $PID 2>/dev/null; do
      [ $(( $(date +%s) - t0 )) -gt 18000 ] && { kill $PID 2>/dev/null; mk PASSTIMEOUT "$lab"; break; }
      sleep 15
    done
    wait $PID 2>/dev/null; rc=$?
    art=$([ -s $out/p2_$arm.json ] && echo 1 || echo 0)
    mk PASSEND "$lab attempt=$attempt rc=$rc artifact=$art wall=$(( $(date +%s) - t0 ))s"
    PASS_OK=$(pass_ok $out $arm $MINW)
    [ "$PASS_OK" = 1 ] && break
  done
}
OUT=$P2/out
# preflight: 4 windows of each arm (GPU), same code path as the full pass
MINW=2
RF=refcv8_eval_remap_p2.json
if [ "$(pass_ok $OUT/pre_refcv8 refcv8_ema48852 2)" = 1 ]; then P8=1; mk SKIP "pre8 already complete"; else
  run_pass pre8 $CK8 $CF8 $OUT/pre_refcv8 refcv8_ema48852 --max-windows 4 --law-control 1 --controls-n 4; P8=$PASS_OK; fi
RF=refcv7_eval_remap_none.json      # refcv7 keeps the kit-default remap (refcv8's overrides would re-point its argv at refcv8 files: tau mismatch -> loader refuses)
if [ "$(pass_ok $OUT/pre_refcv7 refcv7_50400 2)" = 1 ]; then P7=1; mk SKIP "pre7 already complete"; else
  run_pass pre7 $CK7 $CF7 $OUT/pre_refcv7 refcv7_50400 --max-windows 4 --law-control 1 --controls-n 4; P7=$PASS_OK; fi
mk PREFLIGHT "refcv8_ok=$P8 refcv7_ok=$P7"
if [ "$P8" != 1 ] || [ "$P7" != 1 ]; then mk ABORT "preflight failed (see logs/pass_pre*)"; release_gpu "ABORT_preflight_failed"; exit 6; fi

# full passes: refcv8 first (primary), then refcv7
MINW=4600
RF=refcv8_eval_remap_p2.json
if [ "$(pass_ok $OUT/refcv8 refcv8_ema48852 4600)" != 1 ]; then run_pass full8 $CK8 $CF8 $OUT/refcv8 refcv8_ema48852 --law-control 2; else PASS_OK=1; mk SKIP "full8 already complete"; fi
F8=$PASS_OK
RF=refcv7_eval_remap_none.json
if [ "$(pass_ok $OUT/refcv7 refcv7_50400 4600)" != 1 ]; then run_pass full7 $CK7 $CF7 $OUT/refcv7 refcv7_50400 --law-control 2; else PASS_OK=1; mk SKIP "full7 already complete"; fi
F7=$PASS_OK
mk FULLPASSES "refcv8_ok=$F8 refcv7_ok=$F7"
if [ "$F8" = 1 ] && [ "$F7" = 1 ]; then release_gpu "OK_both_passes_complete"; else release_gpu "FAIL_pass_incomplete_refcv8=${F8}_refcv7=${F7}"; fi

# ---- 4. analysis (CPU) --------------------------------------------------------------------------------------------------------------------- #
if [ "$F8" = 1 ] && [ "$F7" = 1 ]; then
  mkdir -p $P2/result
  ( cd $P2/code; $PY -X utf8 p2_tol_read.py --a-dir $OUT/refcv8 --a-arm refcv8_ema48852 --b-dir $OUT/refcv7 --b-arm refcv7_50400 --out $P2/result --n-boot 1000 ) > $P2/logs/read.log 2>&1 < /dev/null
  rrc=$?
  art=$([ -s $P2/result/RESULT.md ] && [ -s $P2/result/P2_TOL.json ] && echo 1 || echo 0)
  mk ANALYSIS "rc=$rrc artifact=$art"
fi
mk END "done"
