#!/bin/bash
# ROLLHIST next-arm chain "RA": the RH argv + `--dagger-mode add` on the SAME bank, SAME dose p = 0.1 (the add-mode lever: own states APPENDED, logged rows kept; PREREG_V8F_ROLLADD). Code of record the RA commit (ra_env.sh: own tree tree_ra = 97fe0a56 + the --dagger-mode flag), readout causal_g1x. STAGED, NOT LAUNCHED. One process, one log.
# Modelled on /root/rollhist/rh_chain.sh (sha256 in MANIFEST). Differences, and only these: NO bank step, NO render of RH/B0 from the template, a FRESH G-SUITE at the RA commit (ra_gsuite.sh, step `gsuite`, CPU under the lock, about 36 min as RH's) and NO pilot (PILOT_RH.json is the
# step-time record, not bound to the commit; ra_gate_record.py re-hashes it and gsuite_result_ra.json at mint), own log / done dir, and the arms are RA_s0, RA_s1 (stage B only; K-B2 holds: each arm inits from its own UE_sK.A).
# The comparator is the EXISTING B0_s0 / B0_s1 (their argv.json and checkpoints are only read). RH_s0 / RH_s1 are read-only inputs (their argv is the template of RA's argv).
# Order: (1) flock fd 9 on /root/gpu_queue.lock, HELD TO THE END  (2) tree + inputs + disk + the finished RH / B0 arms and the bank sha  (3) render the two RA argv files (ra_argv.py: RH's argv with exactly --dagger-mode add and --out)
#        (4) RA_s0 (ra_arm.sh: pair check, selfcheck, gate record, study-side token mint, train 3,000 steps through the token, post gates)  (5) RA_s1.
# Every step leaves $R/done_ra/<name>.done; a relaunch skips finished steps (a half-finished arm is MOVED ASIDE, never deleted). STOP file: $R/chains/STOP, read between steps.
# Log: /root/rollhist/rh_ra_chain.log ; its LAST line is `ZZ ROLLHIST-RA DONE ZZ` or `ZZ ROLLHIST-RA FAIL ZZ <why>`.
# MARKER PROTOCOL (as rh_marker_watch.sh): the watcher rh_ra_marker_watch.sh sets /root/N1DW_READY and removes it when this chain holds the lock.   LAUNCH: see launch_ra.sh (MM only).
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/ra_env.sh
D=$R/done_ra; mkdir -p $D $R/logs $R/runs $R/chains
LOG=$R/rh_ra_chain.log
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ ROLLHIST-RA FAIL ZZ $1"; exit 1; }
rh_readout_flags
[ "$RH_READOUT" = causal_g1x ] || fail "RH_READOUT is $RH_READOUT: the RH bank and arms are causal_g1x"
export_py
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
say "chain start commit=${COMMIT:0:8} readout=$RH_READOUT (decl token $DECL_TOKEN) arms RA_s0 RA_s1 (--dagger-p 0.1 --dagger-mode add, same banks)"
# 1. the GPU queue lock, held to the end
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 2. tree, inputs, disk, the finished arms
chk_tree() {
  $PY $R/blobcheck.py $TREE $BLOBS > $R/logs/blobcheck_ra.log 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || return 1
}
step tree chk_tree
gs_passed() { [ -s $GSUITE_RA ] && [ -s $R/logs/ra_gsuite.log ] && tail -n 1 $R/logs/ra_gsuite.log | grep -q "ZZ RA GSUITE DONE ZZ"; }
gs() {   # fresh G-SUITE at the RA commit (0 regressions, 0 new-test failures, 0 missing); a pass from the standalone prep run (ra_gsuite.sh before the launch) is kept, not re-run under the lock
  gs_passed && { say "G-SUITE at ${COMMIT:0:8} passed in the prep run: kept (ra_gate_record.py re-checks its commit and counts at mint)"; return 0; }
  bash $HERE/ra_gsuite.sh; gs_passed; }
step gsuite gs
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_ra.log 2>&1 || return 1          # the two stage-A checkpoints (written by rh_chain.sh's inputs step)
  [ -s $R/study/dscale10k_check_ra.py ] && [ -s $R/study/dscale10k_merge.py ] || { echo "no dscale10k_check_ra.py / dscale10k_merge.py in $R/study" >> $R/logs/inputs_verify_ra.log; return 1; }
  for f in $HERE/ra_arm.sh $HERE/chk_arm_ra.sh $HERE/ra_argv.py $HERE/ra_env.sh $HERE/ra_gsuite.sh $HERE/ra_gate_record.py $GSUITE_RA $R/pilot/PILOT_RH.json $DS/argv/U_s0.B.tmpl.json $DS/argv/U_s1.B.tmpl.json $DS/chainlib.py; do
    [ -s "$f" ] || { echo "missing input $f" >> $R/logs/inputs_verify_ra.log; return 1; }
  done
  for s in 0 1; do                                                                         # the SAME banks and the existing comparators: finished, with checkpoints
    for t in RH_s$s B0_s$s; do
      [ -e $R/runs/$t/ARM_DONE ] && [ -s $R/runs/$t/train/ckpt_last.pt ] || { echo "$t is not finished (ARM_DONE + ckpt): RA compares against it" >> $R/logs/inputs_verify_ra.log; return 1; }
    done
    [ -s $R/bank/s$s/BANK_CHECK.json ] && [ -s $R/bank/s$s/bank.json ] || { echo "bank s$s missing" >> $R/logs/inputs_verify_ra.log; return 1; }
  done
}
step inputs chk_inputs
free_gb() { df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; }
disk_ok() { f=$(free_gb); say "free disk ${f} GB"; [ "${f:-0}" -ge 20 ]; }
step disk0 disk_ok
# 3. the two argv files
rn() { $PY $HERE/ra_argv.py render > $R/logs/ra_render.log 2>&1; tail -n 1 $R/logs/ra_render.log | grep -q "ZZ RA-RENDER PASS"; }
step render rn
# 4/5. the arms (same function as rh_chain.sh: a partial arm is moved aside, never deleted; a finished one is skipped)
arm() {
  local tag=$1
  [ -e $R/runs/$tag/ARM_DONE ] && [ -s $R/runs/$tag/train/ckpt_last.pt ] && { say "skip $tag (ARM_DONE, ckpt present)"; return 0; }
  if [ -d $R/runs/$tag/train ] || [ -d $R/runs/$tag/selfcheck ]; then
    local A=$R/runs/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train selfcheck; do [ -e $R/runs/$tag/$d ] && mv $R/runs/$tag/$d $A/; done
    rm -f $R/runs/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $HERE/ra_arm.sh $tag
}
for s in 0 1; do
  disk_ok || fail "free disk below 20 GB before RA_s$s"
  step RA_s$s arm RA_s$s
done
say "free disk $(free_gb) GB at the end"
echo "ZZ ROLLHIST-RA DONE ZZ"
