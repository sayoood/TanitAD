#!/bin/bash
# STRAT1 C0 same-seed replicates C0_s10r / C0_s11r (MM 2026-11-17; PREREG_ADDENDUM_C0REP.md, written before either arm trains; PI go "2: queue s10 ans s11").
# Same tree (new4 = a4357e79), inputs, checker, mint, runner and post check as strat1_n1s10.sh; only the run root differs (runs_c0rep), so the argv differs from the banked C0_s10 / C0_s11 in --out only.
# Queued behind the CKPT120 fix3 chain: started as `flock /root/gpu_queue.lock bash /root/strat1/strat1_c0rep.sh` (one GPU job at a time on the pod).
# usage: nohup setsid flock /root/gpu_queue.lock bash /root/strat1/strat1_c0rep.sh < /dev/null > /root/strat1/logs/c0rep.log 2>&1 &
# Ends with `ZZ STRAT1 C0REP DONE ZZ` or `ZZ STRAT1 C0REP FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
set -u
R=/root/strat1; TREE=$R/new4; INP=$R/inputs; RUNS=$R/runs_c0rep; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=a4357e7986f3aa30dab2ed83fde2351fda8d5662
SUITE=$R/gsuite_result6.json
ARMS="C0:s10 C0:s11"
fail() { echo "ZZ STRAT1 C0REP FAIL ZZ $1"; exit 1; }
echo "$(date -u +%FT%TZ) C0REP start (holding the GPU queue lock)"
sha256sum -c $R/fix6.sha256 > $R/logs/fix6_verify_c0rep.log 2>&1 || fail "new4 files are not the a4357e79 blobs"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_c0rep.log 2>&1 || fail "inputs sha256 mismatch"
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT runs=$RUNS"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && { echo "skip $tag (ARM_DONE)"; continue; }
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used) before $tag"; sleep 10; done
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  # K-ARGV: the replicate's argv equals the banked arm's, token for token, once the run root is mapped back
  $PY - "$RUNS/$tag/argv.json" "$R/runs/$tag/argv.json" <<'PYEOF' || fail "K-ARGV $tag: the replicate argv differs from the banked arm beyond the run root"
import json, sys
a = json.dumps(json.load(open(sys.argv[1]))).replace("/root/strat1/runs_c0rep/", "/root/strat1/runs/")
b = json.dumps(json.load(open(sys.argv[2])))
assert a == b, "argv differs"
print("K-ARGV ok")
PYEOF
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1|s11) init=$INP/M2b_s1/ckpt_last.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
done
echo "ZZ STRAT1 C0REP DONE ZZ"
