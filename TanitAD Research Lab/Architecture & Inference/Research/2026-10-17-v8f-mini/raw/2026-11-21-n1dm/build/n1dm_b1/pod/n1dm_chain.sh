#!/bin/bash
# N1DM chain (PREREG_V8F_N1DM). STAGED, NOT LAUNCHED (running needs the PI's second go with the cost). Adapted from n1dh_chain.sh (which ran to ZZ N1DH DONE ZZ; Amendments 3 and 4 of N1DH folded in).
# One process, one log. It (1) takes /root/gpu_queue.lock on fd 9 and HOLDS it to the end (G-SUITE, the arms and the CPU dumps all run under it, one job at a time), (2) tree + inputs checks (the tree file list INCLUDES the one Research
# Lab file, the eval-141 grid, which a --dagger-bank arm's K-B3 check reads: N1DH Amendment 3), (3) G-SUITE on the clean tree vs the 1f217bf0 junit (junit/n1dh), (4) argv render + pairing with negative controls,
# (5) arm N1DM_s0 then N1DM_s1 (selfcheck -> mint -> pair -> train -> post -> zero-grad gate with route_head / route_br controls -> K-CLIP), (6) the D1-HO CPU dumps (10: s0/s1 x follow/zero/zuse/unif at hb0.2, + follow at old).
# There is NO ROLLHIST check here (that chain has ended; the N1DH chain log shows ZZ ROLLHIST DONE ZZ). The GPU lock is the only order.
# Every step leaves done/<name>.done; a relaunch skips finished steps (a half-finished arm is moved aside, its gate token kept).
# Not here (dev box / MM reads): eval-141 dumps, real K-LIM (n1dh_klim.py on a real probe batch; it reads route_logits and works unchanged on a modes checkpoint), Z-POST, S1-slot view, the paired reads.
# Log: /root/strat1/n1dm/n1dm_chain.log ; its LAST line is `ZZ N1DM DONE ZZ` or `ZZ N1DM FAIL ZZ <why>`.
# LAUNCH: after n1dm_stage.sh and the dry mint:  nohup setsid bash /root/strat1/n1dm/n1dm_chain.sh < /dev/null > /dev/null 2>&1 &   (n1dm_launch.sh starts the marker watcher first, then this chain)
R=/root/strat1; N=$R/n1dm; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; PY=/root/venv/bin/python
LOG=$N/n1dm_chain.log; D=$N/done
mkdir -p $N $D $RUNS $N/out $R/logs
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ N1DM FAIL ZZ $1"; exit 1; }
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
COMMIT=@@N1DM_COMMIT@@
case "$COMMIT" in @@*) fail "COMMIT placeholder not substituted (run n1dm_stage.sh)";; esac
say "chain start commit=$COMMIT (n1dm)"
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 1. tree + kit + inputs
chk_tree() {
  (cd $TREE && sha256sum -c $N/n1dm_tree.sha256) > $R/logs/n1dm_tree_verify.log 2>&1 || return 1
  $PY $N/blobcheck.py $TREE $N/all_blobs.txt > $R/logs/n1dm_blobcheck.log 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $N/all_blobs.txt)" ] || return 1      # no extra file; all_blobs.txt lists the eval-141 grid
  [ "$(find "$TREE/TanitAD Research Lab" -type f | wc -l)" = 1 ] || return 1      # exactly one Research Lab file: the eval-141 grid (K-B3), listed in all_blobs.txt
  (cd $N/hb && echo "74fd8445e377e763d17b1fa4c60753f23b8240eb94d0bdf7f4633a9e3ef80608 *v8f_t1_eval_hb.py" | sha256sum -c) >> $R/logs/n1dm_tree_verify.log 2>&1 || return 1
  (cd $N && echo "8e78e69e202e6fa1422ae680d6bbda3c24af0786ae5234b9f744fefb4004f53c *kclip.py" | sha256sum -c) >> $R/logs/n1dm_tree_verify.log 2>&1 || return 1
}
step tree chk_tree
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_n1dm.log 2>&1 || return 1
  (cd $INP/strat1 && sha256sum -c $R/navrb2/navrb2_inputs.sha256 && sha256sum -c $R/n1dw/navrbd_inputs.sha256) >> $R/logs/inputs_verify_n1dm.log 2>&1 || return 1
}
step inputs chk_inputs
# 2. G-SUITE (CPU) on the clean tree, against the 1f217bf0 junit of N1DH's own run (junit/n1dh); it holds no GPU, but the lock is ours
gs() { bash $N/n1dm_gsuite.sh > $R/logs/n1dm_gsuite.driver.log 2>&1; tail -n1 $R/logs/n1dm_gsuite.log | grep -q "ZZ N1DM GSUITE DONE ZZ"; }
step gsuite gs
[ -s $N/gsuite_result_n1dm.json ] || fail "no G-SUITE result"
# 3. argv + pairing + negative controls
pr() { bash $N/pairing_n1dm.sh > $N/b2/pairing.out 2>&1; }
mkdir -p $N/b2
step pairing pr
# 4. the arms, one at a time
for seed in s0 s1; do
  tag=N1DM_$seed
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  if [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ]; then say "skip $tag (ARM_DONE, ckpt present)"; continue; fi
  if [ -d $RUNS/$tag/train ] || [ -d $RUNS/$tag/logs ]; then
    A=$RUNS/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train logs; do [ -e $RUNS/$tag/$d ] && mv $RUNS/$tag/$d $A/; done
    rm -f $RUNS/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $N/n1dm_arm.sh $seed || fail "arm $tag (see $RUNS/*_$tag.log)"
done
# 5. D1-HO CPU dumps, both arms (real, D-channel zero twin, branches-zeroed copy, uniform-p copy), lock still held
dm() { bash $N/n1dm_dumps.sh > $N/out/dumps.log 2>&1; [ -e $N/out/DUMPS_DONE ]; }
step dumps dm
echo "ZZ N1DM DONE ZZ"
