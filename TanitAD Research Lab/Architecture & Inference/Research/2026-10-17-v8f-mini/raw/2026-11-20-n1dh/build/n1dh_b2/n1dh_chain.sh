#!/bin/bash
# N1DH chain (PREREG_V8F_N1DH, card sha256 f1df05bc). STAGED, NOT LAUNCHED. Adapted from n1dc_chain.sh (which ran to ZZ N1DC DONE ZZ); code of record cc8c98a1 (parent bcf3ba3f).
# One process, one log. It (0) refuses unless the ROLLHIST chain's last log line is `ZZ ROLLHIST DONE ZZ` (env N1DH_SKIP_RH_CHECK=1 overrides, an MM decision), (1) takes /root/gpu_queue.lock on fd 9 and HOLDS it to the end
# (G-SUITE, the arms and the CPU dumps all run under it, one job at a time), (2) tree + inputs checks, (3) G-SUITE on the clean tree vs the bcf3ba3f junit (junit/n1dc), (4) argv render + pairing with negative controls,
# (5) arm N1DH_s0 then N1DH_s1 (selfcheck -> mint -> pair -> train -> post -> zero-grad gate with route_emb/route_head controls -> K-CLIP), (6) the D1-HO CPU dumps (8: s0/s1 x follow/zero/zuse at hb0.2, + follow at old).
# Every step leaves done/<name>.done; a relaunch skips finished steps (a half-finished arm is moved aside, its gate token kept).
# Not here (dev box / MM reads): eval-141 dumps, real K-LIM (n1dh_klim.py on a real probe batch), Z-POST, S1-slot view, the paired reads.
# Log: /root/strat1/n1dh/n1dh_chain.log ; its LAST line is `ZZ N1DH DONE ZZ` or `ZZ N1DH FAIL ZZ <why>`.
# LAUNCH: use /root/strat1/n1dh/n1dh_launch_after_rollhist.sh (waits for ZZ ROLLHIST DONE ZZ, then the marker watcher and this chain), or directly:  nohup setsid bash /root/strat1/n1dh/n1dh_chain.sh < /dev/null > /dev/null 2>&1 &
R=/root/strat1; N=$R/n1dh; TREE=$N/tree; INP=$R/inputs; RUNS=$N/runs; PY=/root/venv/bin/python
LOG=$N/n1dh_chain.log; D=$N/done
mkdir -p $N $D $RUNS $N/out $R/logs
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ N1DH FAIL ZZ $1"; exit 1; }
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
say "chain start commit=cc8c98a1 (n1dh)"
# 0. the ROLLHIST chain must have ended in DONE (the launch gate waits for it; this is the second check)
if [ "${N1DH_SKIP_RH_CHECK:-0}" != 1 ]; then
  rl=$(tail -n1 /root/rollhist/rh_chain.log 2>/dev/null)
  case "$rl" in "ZZ ROLLHIST DONE ZZ"*) say "ROLLHIST ended: $rl";; *) fail "the ROLLHIST chain's last line is not ZZ ROLLHIST DONE ZZ (got: $rl)";; esac
fi
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 1. tree + kit + inputs
chk_tree() {
  (cd $TREE && sha256sum -c $N/n1dh_tree.sha256) > $R/logs/n1dh_tree_verify.log 2>&1 || return 1
  $PY $N/blobcheck.py $TREE $N/all_blobs.txt > $R/logs/n1dh_blobcheck.log 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $N/all_blobs.txt)" ] || return 1      # no extra file (a stray eval_grid_windows.json is gone)
  [ ! -e "$TREE/TanitAD Research Lab" ] || return 1
  (cd $N/hb && echo "74fd8445e377e763d17b1fa4c60753f23b8240eb94d0bdf7f4633a9e3ef80608 *v8f_t1_eval_hb.py" | sha256sum -c) >> $R/logs/n1dh_tree_verify.log 2>&1 || return 1
  (cd $N && echo "8e78e69e202e6fa1422ae680d6bbda3c24af0786ae5234b9f744fefb4004f53c *kclip.py" | sha256sum -c) >> $R/logs/n1dh_tree_verify.log 2>&1 || return 1
}
step tree chk_tree
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_n1dh.log 2>&1 || return 1
  (cd $INP/strat1 && sha256sum -c $R/navrb2/navrb2_inputs.sha256 && sha256sum -c $R/n1dw/navrbd_inputs.sha256) >> $R/logs/inputs_verify_n1dh.log 2>&1 || return 1
}
step inputs chk_inputs
# 2. G-SUITE (CPU) on the clean tree, against the bcf3ba3f junit of N1DC's own run (junit/n1dc); it holds no GPU, but the lock is ours
gs() { bash $N/n1dh_gsuite.sh > $R/logs/n1dh_gsuite.driver.log 2>&1; tail -n1 $R/logs/n1dh_gsuite.log | grep -q "ZZ N1DH GSUITE DONE ZZ"; }
step gsuite gs
[ -s $N/gsuite_result_n1dh.json ] || fail "no G-SUITE result"
# 3. argv + pairing + negative controls
pr() { bash $N/pairing_n1dh.sh > $N/b2/pairing.out 2>&1; }
mkdir -p $N/b2
step pairing pr
# 4. the arms, one at a time
for seed in s0 s1; do
  tag=N1DH_$seed
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  if [ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ]; then say "skip $tag (ARM_DONE, ckpt present)"; continue; fi
  if [ -d $RUNS/$tag/train ] || [ -d $RUNS/$tag/logs ]; then
    A=$RUNS/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train logs; do [ -e $RUNS/$tag/$d ] && mv $RUNS/$tag/$d $A/; done
    rm -f $RUNS/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $N/n1dh_arm.sh $seed || fail "arm $tag (see $RUNS/*_$tag.log)"
done
# 5. D1-HO CPU dumps, both arms (real, D-channel zero twin, Z-USE copy), lock still held
dm() { bash $N/n1dh_dumps.sh > $N/out/dumps.log 2>&1; [ -e $N/out/DUMPS_DONE ]; }
step dumps dm
echo "ZZ N1DH DONE ZZ"
