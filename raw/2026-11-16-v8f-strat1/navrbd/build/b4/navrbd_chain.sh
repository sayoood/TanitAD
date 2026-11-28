#!/bin/bash
# STRAT1 NAVRBD chain (PREREG_V8F_NAVRBD sections 4-6, frozen card sha256 881d7f3d...). Pattern: navrb2_chain_fix2.sh.
# 3 arms in priority order: N1D:s0 N1D:s1 N1Dx:s0 (N1D = N1R + --nav-rb-feat exitdist + the 15-column table; N1Dx = N1D with --nav-rb-variant shuffle, paired with N1R_s0).
# Tree = /root/strat1/navrbd/tree (0c3b374f, verified against navrbd_tree.sha256); G-SUITE result gsuite_result_navrbd.json (checker `suite`, for this commit).
# Data inputs are the N1R arms' own files (INP=/root/strat1/inputs, same paths, sha-checked); only the 15-column table and its report are new (inputs/strat1/navtext_nav_rbd*.npz|json).
# Per arm: argv pair check (N1R argv + exactly the new flag) -> strat1_check.py mint (token) -> run (refuses without the token) -> post -> zerograd_gate on e_nav_rb.weight -> `ZZ ARM <tag> DONE ZZ`.
# STAGED, NOT LAUNCHED. By hand under the pod GPU lock:
#   nohup setsid flock /root/gpu_queue.lock bash /root/strat1/navrbd/navrbd_chain.sh < /dev/null > /root/strat1/logs/navrbd.log 2>&1 &
# Ends with `ZZ STRAT1 NAVRBD DONE ZZ` or `ZZ STRAT1 NAVRBD FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
cd /root/strat1/navrbd || exit 1
set -u
R=/root/strat1; N=$R/navrbd; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"; ZG="$TREE/stack/scripts/zerograd_gate.py"
COMMIT=0c3b374fb1036b0ca4ca7620f98137dd6c59cca4
SUITE=$R/gsuite_result_navrbd.json
ARMS="N1D:s0 N1D:s1 N1Dx:s0"
fail() { echo "ZZ STRAT1 NAVRBD FAIL ZZ $1"; exit 1; }
case "$COMMIT" in __MM*) fail "COMMIT not set";; esac
tail -c 600 $R/logs/navrbd_gsuite.log 2>/dev/null | grep -q "ZZ NAVRBD GSUITE DONE ZZ" || fail "the NAVRBD G-SUITE has not passed (logs/navrbd_gsuite.log)"
[ -s "$SUITE" ] || fail "no G-SUITE result $SUITE"
(cd $TREE && sha256sum -c $N/navrbd_tree.sha256) > $R/logs/navrbd_tree_verify.log 2>&1 || fail "the 16 NAVRBD files are not the 0c3b374f blobs (logs/navrbd_tree_verify.log)"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_navrbd.log 2>&1 || fail "inputs sha256 mismatch"
(cd $INP/strat1 && sha256sum -c $R/navrb2/navrb2_inputs.sha256) >> $R/logs/inputs_verify_navrbd.log 2>&1 || fail "navtext_nav_rb.npz / report sha256 mismatch"
(cd $INP/strat1 && sha256sum -c $N/navrbd_inputs.sha256) >> $R/logs/inputs_verify_navrbd.log 2>&1 || fail "navtext_nav_rbd.npz / report sha256 mismatch"
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ] && { echo "skip $tag (ARM_DONE, ckpt present)"; continue; }
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used) before $tag"; sleep 10; done
  # pairing BEFORE any GPU work: render the arm argv (CPU) and check it against N1R's banked argv
  mkdir -p $RUNS/$tag
  $PY $N/render_argv.py $kind $seed $RUNS/pre_argv_$tag.json $RUNS/$tag/train > $RUNS/pre_argv_$tag.log 2>&1 || fail "argv render $tag: $(tail -c 300 $RUNS/pre_argv_$tag.log)"
  $PY $CHK pair $kind --argv-json $RUNS/pre_argv_$tag.json --ref-argv-json $R/navrb2/runs/N1R_$seed/argv.json > $RUNS/pre_pair_$tag.log 2>&1 || fail "pre-mint pairing $tag: $(tail -c 300 $RUNS/pre_pair_$tag.log)"
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  # the pairing rule on the argv the mint wrote: N1R's banked argv of the same seed + exactly the new flag / table / sha / out (N1Dx also the shuffle variant)
  $PY $CHK pair $kind --argv-json $RUNS/$tag/argv.json --ref-argv-json $R/navrb2/runs/N1R_$seed/argv.json > $RUNS/pair_$tag.log 2>&1 || fail "pairing $tag: $(tail -c 300 $RUNS/pair_$tag.log)"
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1) init=$INP/M2b_s1/ckpt_last.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  # the channel weight must have TRAINED (Adam state, no zero moments): --allow '.*' exempts every other parameter (this gate reads only e_nav_rb.weight), --control requires it among the trained ones
  PYTHONPATH=$TREE/stack:$TREE $PY $ZG --ckpt $RUNS/$tag/train/ckpt_last.pt --allow '.*' --control 'e_nav_rb\.weight' --json-out $RUNS/$tag/zerograd_navrbd.json > $RUNS/zerograd_$tag.log 2>&1 || fail "zerograd gate $tag: $(tail -c 300 $RUNS/zerograd_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
done
echo "ZZ STRAT1 NAVRBD DONE ZZ"
