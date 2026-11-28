#!/bin/bash
# mint_n1dw.sh -- waits (CPU, sleep 30, cap 4 h) for the N1DW G-SUITE sentinel, runs the CPU binding preflight, then (GPU, under flock /root/gpu_queue.lock) the R2 same-code repeat and the mint of the four arm tokens.
# Mint = strat1_check.py mint (trainer --selfcheck-only on the exact argv + a 4-step crash/resume per arm); tokens are written by launch-gate code only (strat1_check.write_token), never by hand.
# R2: two selfcheck-only runs of the N1D argv (w = 1) on the n1dw tree are the same-code repeat; r2_compare.py reads each arm selfcheck terms against it.
# Log: /root/strat1/logs/n1dw_mint.log; last line `ZZ N1DW MINT DONE ZZ` or `ZZ N1DW MINT FAIL ZZ <why>`.
# LAUNCH: nohup setsid bash /root/strat1/n1dw/mint_n1dw.sh < /dev/null > /root/strat1/logs/n1dw_mint.log 2>&1 &
R=/root/strat1; N=$R/n1dw; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod; PY=/root/venv/bin/python; CHK=$TREE/stack/scripts/strat1_check.py
COMMIT=134f5da1ee50a3c2fa242bb340a26dd6e4072ade; SUITE=$N/gsuite_result_n1dw.json; L=$R/logs/n1dw_gsuite.log
fail() { echo "ZZ N1DW MINT FAIL ZZ $1"; exit 1; }
for i in $(seq 1 480); do tail -n1 $L 2>/dev/null | grep -q "ZZ N1DW GSUITE" && break; sleep 30; done
tail -n1 $L | grep -q "ZZ N1DW GSUITE DONE ZZ" || fail "suite: $(tail -n1 $L)"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $N/b2
taskset -c 48-95 nice -n 10 $PY $N/b2/binding_preflight_n1dw.py > $N/b2/binding_preflight.out 2>&1 || fail "binding preflight (see b2/binding_preflight.out)"
echo "$(date -u +%T) preflight PASS; waiting for the GPU lock"
exec 8>/root/gpu_queue.lock
flock 8 || fail "flock"
echo "$(date -u +%T) holding the GPU lock"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle"; sleep 10; done
# R2 repeat: the N1D_s0 argv (w = 1) twice, selfcheck-only, on this tree
mkdir -p $N/b2/r2
for r in a b; do
  rm -rf $N/b2/r2/N1D_$r
  $PY $N/b2/r2_run.py $r > $N/b2/r2/run_$r.out 2>&1 || fail "R2 repeat $r (see b2/r2/N1D_$r.log)"
done
for arm in N1DW4:s0 N1DW4:s1 N1DW2:s0 N1DW2:s1; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  mkdir -p $RUNS/$tag
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed --root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0 --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  $PY $CHK pair $kind --argv-json $RUNS/$tag/argv.json --ref-argv-json $R/navrbd/runs/N1D_$seed/argv.json > $RUNS/pair_$tag.log 2>&1 || fail "pairing $tag"
done
$PY $N/b2/r2_compare.py $N/b2/r2_result.json $N/b2/r2/N1D_a/selfcheck.json $N/b2/r2/N1D_b/selfcheck.json \
  N1DW4_s0=$RUNS/N1DW4_s0/gate/selfcheck.json N1DW4_s1=$RUNS/N1DW4_s1/gate/selfcheck.json N1DW2_s0=$RUNS/N1DW2_s0/gate/selfcheck.json N1DW2_s1=$RUNS/N1DW2_s1/gate/selfcheck.json > $N/b2/r2_compare.out 2>&1 \
  || echo "R2 note: a term moved more than the repeat (see b2/r2_compare.out); tokens are minted, the MM reads R2 before launch"
echo "ZZ N1DW MINT DONE ZZ"
