#!/bin/bash
# KINLAB chain "RK" -- REVISION 2 (commit 1a8b11a1, Amendment 1 frozen sha256 948670d9: the label cache). RL's stage-B argv with `--dagger-label speed_v1` -> `kin_v1` and the execution-only token pair
# `--dagger-label-cache $R/kin_cache/s<K>_kin_v1.npz` on the SAME RH bank, SAME dose p = 0.1, swap mode (PREREG_V8F_KINLAB + Amendment 1). Code of record the commit 1a8b11a1 (rk_env.sh: own tree tree_rk = df250a00's tree_ra + 8 blobs).
# Readout causal_g1x. STAGED, NOT LAUNCHED. One process, one log. Modelled on rh_ra_chain.sh; revision 2 runs from the kit folder /root/rollhist/kit_rk. Differences from revision 1, and only these:
#   * the tree prep, the two label-cache precomputes and the G-SUITE are run BEFORE this chain (CPU, nice 19, outside the lock; PREP_RK2.md). The chain only VALIDATES them, before the lock and in seconds, every time (the done-markers
#     are written but never trusted): tree_rk blob by blob against all_blobs_rk.txt (3,420 rows); gsuite_result_rk.json at commit 1a8b11a1 with 0 regressions / 0 new-test failures / 0 missing (a result at another commit or with a
#     failure REFUSES before the lock, it is never recomputed or overwritten; a MISSING result is computed here, under the marker, as in revision 1); both caches present and read back by their CPU precompute's selfcheck
#     (rk_gate_record.py precheck: sha256 of the file now, path, label kin_v1, census >= 0.5); the RK argv built in memory (rk_argv.py show: exactly three differences from RL). Any refusal is BEFORE the lock;
#   * the chain still WAITS (cap 12 h) for the ADD123 chain's last line `ZZ ADD123 ` in /root/dscale_lead/mm_add123_chain.log (step wait_add123) and the marker take-over (rh_rk_marker_watch.sh) is unchanged;
#   * the lock (fd 9, flock on /root/gpu_queue.lock) is taken after those checks and HELD TO THE END; every exit, a refusal included, hands it back at once (EXIT trap: flock -u);
#   * render = rk_argv.py (RL's argv with the label value, --out and the cache pair); the arms are RK_s0, RK_s1 (stage B only; K-B2 holds: each arm inits from its own UE_sK.A); RL / RH / B0 are finished arms, only read;
#   * the study-side checker is $HERE/dscale10k_check_rk.py (admits the cache pair); the shared pod files (rh_env.sh, rh_selfcheck.py, rh_evidence.py, blobcheck.py, study/dscale10k_merge.py) are read from $R.
#   * DRY=1: a static pre-lock walk that WRITES NOTHING, takes no lock, touches no marker and does not wait: it runs the same validations and prints the steps the chain would run. Last line ZZ ROLLHIST-RK DRY PASS ZZ / DRY FAIL ZZ <why>.
# Order: (1) wait for ADD123  (2) tree  (3) G-SUITE result  (4) caches  (5) argv  (6) flock fd 9, HELD TO THE END  (7) tree check, inputs, disk  (8) render the two argv files (rk_argv.py)
#        (9) RK_s0 (rk_arm.sh: argv verify, pair check, selfcheck (a cache HIT), gate record incl. the cache binding and the census bar, study-side token, train 3,000 steps through the token, lever-acts, summary, post gates, zerograd, K-CLIP)
#        (10) RK_s1  (11) the RA chain's end steps.
# Every step leaves $R/done_rk/<name>.done; a relaunch skips finished arm steps (a half-finished arm or tree.part is MOVED ASIDE, never deleted). STOP file: $R/chains/STOP, read between steps.
# Log: /root/rollhist/rh_rk_chain.log ; its LAST line is `ZZ ROLLHIST-RK DONE ZZ` or `ZZ ROLLHIST-RK FAIL ZZ <why>`.
# MARKER PROTOCOL: rh_rk_marker_watch.sh (see there). LAUNCH: see launch_rk.sh (MM only).
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rk_env.sh
DRYRUN=0; [ "${DRY:-0}" = 1 ] && DRYRUN=1
D=$R/done_rk
LOG=$R/rh_rk_chain.log
ADDLOG=/root/dscale_lead/mm_add123_chain.log
BCLOG=$R/logs/blobcheck_rk.log
if [ $DRYRUN = 1 ]; then
  BCLOG=/dev/null
  say() { echo "$(date -u +%FT%TZ) [DRY] $*"; }
  fail() { echo "ZZ ROLLHIST-RK DRY FAIL ZZ $1"; exit 1; }
  [ "$(cat $R/RH_READOUT.chosen 2>/dev/null)" = causal_g1x ] || fail "$R/RH_READOUT.chosen is not causal_g1x"
  RH_READOUT=causal_g1x; DECL_TOKEN=causal_g1x
else
  mkdir -p $D $R/logs $R/runs $R/chains
  exec >> $LOG 2>&1
  say() { echo "$(date -u +%FT%TZ) $*"; }
  fail() { echo "ZZ ROLLHIST-RK FAIL ZZ $1"; exit 1; }
  trap 'flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null' EXIT          # a refusal (or any exit) hands the GPU queue lock back at once, even if a child still holds the descriptor
  rh_readout_flags
  [ "$RH_READOUT" = causal_g1x ] || fail "RH_READOUT is $RH_READOUT: the RH bank and arms are causal_g1x"
fi
export_py
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
# a validation step: runs EVERY time (a stale done-marker is never trusted), writes its marker after a pass
vstep() { local n=$1; shift; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "validate $n"; "$@" || fail "step $n (refused before the lock)"; [ $DRYRUN = 1 ] || date -u +%FT%TZ > $D/$n.done; }

# ---------------- the validations (shared by the real chain and DRY=1)
chk_tree() {
  $PY $R/blobcheck.py $TREE $BLOBS > $BCLOG 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || return 1
}
tp() {
  if [ $DRYRUN = 1 ]; then [ -d $TREE ] || { echo "no $TREE: the tree prep is a step before the chain"; return 1; }; chk_tree && echo "tree_rk: $(wc -l < $BLOBS) blobs verified, would skip the build"; return; fi
  [ -e $TREE.part ] && { local A=$TREE.part.aside_$(date -u +%H%M%S); mv $TREE.part $A; say "half-built tree moved aside to $A"; }
  if [ -d $TREE ]; then chk_tree || { echo "tree_rk exists but fails the blob check against ${BLOBS##*/} (commit ${COMMIT:0:8}): refusing, not rebuilding"; return 1; }; return 0; fi
  bash $HERE/rk_tree_prep.sh > $R/logs/rk_tree_prep.log 2>&1; tail -n 1 $R/logs/rk_tree_prep.log | grep -q "ZZ RK TREE OK ZZ"
}
gs_valid() {                      # the result file's own commit and counts
  [ -s $GSUITE_RK ] || return 2
  $PY - $GSUITE_RK $COMMIT <<'PYE'
import json, sys
r = json.load(open(sys.argv[1]))
ok = (r.get("new_commit") == sys.argv[2] or r.get("new") == sys.argv[2]) and not r.get("n_regressions") and not r.get("new_test_failures") and not r.get("missing_in_new")
print("gsuite_result_rk.json: new", r.get("new_commit", r.get("new")), "n_regressions", r.get("n_regressions"), "new_test_failures", len(r.get("new_test_failures") or []), "missing", len(r.get("missing_in_new") or []), "->", "VALID" if ok else "NOT A PASS AT THIS COMMIT")
sys.exit(0 if ok else 1)
PYE
}
gs_log_done() { [ -s $R/logs/rk_gsuite.log ] && tail -n 1 $R/logs/rk_gsuite.log | grep -q "ZZ RK GSUITE DONE ZZ"; }
gs() {
  gs_valid; local rc=$?
  if [ $rc = 0 ]; then gs_log_done || { echo "gsuite_result_rk.json is valid but $R/logs/rk_gsuite.log does not end in ZZ RK GSUITE DONE ZZ"; return 1; }; say "G-SUITE at ${COMMIT:0:8} passed earlier: kept (rk_gate_record.py re-checks it at mint)"; return 0; fi
  [ $rc = 2 ] || { echo "gsuite_result_rk.json exists and is NOT a pass at ${COMMIT:0:8}: refusing (never recomputed over it; move it aside by hand)"; return 1; }
  [ $DRYRUN = 1 ] && { echo "no gsuite_result_rk.json: the chain would RUN rk_gsuite.sh here (about 37 min) under the marker, before the lock"; return 1; }
  bash $HERE/rk_gsuite.sh; gs_valid && gs_log_done
}
chk_caches() { local s; for s in 0 1; do $PY $HERE/rk_gate_record.py precheck --tag RK_s$s || return 1; done; }
chk_argv() { $PY $HERE/rk_argv.py show --seeds 0 1; }

# ---------------- DRY=1: the static pre-lock walk, nothing written
if [ $DRYRUN = 1 ]; then
  say "DRY chain walk, commit=${COMMIT:0:8} readout=$RH_READOUT arms RK_s0 RK_s1; writes nothing, no lock, no marker, no wait"
  say "step wait_add123: would poll $ADDLOG for 'ZZ ADD123 ' (now: $(cat $ADDLOG 2>/dev/null | grep -c 'ZZ ADD123 ') such lines)"
  vstep tree_prep tp
  vstep gsuite gs
  vstep caches chk_caches
  vstep argv chk_argv
  say "would now take flock fd 9 on /root/gpu_queue.lock (held to the end), then: tree check, inputs, disk0, render, RK_s0 (rk_arm.sh), RK_s1, end steps"
  echo "ZZ ROLLHIST-RK DRY PASS ZZ"
  exit 0
fi

say "chain start commit=${COMMIT:0:8} readout=$RH_READOUT (decl token $DECL_TOKEN) arms RK_s0 RK_s1 (--dagger-p 0.1 --dagger-label kin_v1 --dagger-label-cache $KC/s<K>_kin_v1.npz, same banks)"
# 1. wait for the ADD123 chain's last line (DONE, STOP or FAIL): it holds the priority marker until then
w_add() {
  local i
  for i in $(seq 1 8640); do
    grep -q "ZZ ADD123 " $ADDLOG 2>/dev/null && { say "ADD123 ended: $(grep 'ZZ ADD123 ' $ADDLOG | tail -n 1)"; return 0; }
    [ -e $R/chains/STOP ] && return 1
    sleep 5
  done
  echo "ADD123 chain did not end within 12 h"; return 1
}
step wait_add123 w_add
# 2-5. validations, all BEFORE the lock (seconds); a stale marker is never trusted
vstep tree_prep tp
vstep gsuite gs
vstep caches chk_caches
vstep argv chk_argv
# 6. the GPU queue lock, held to the end
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 7. tree, inputs, disk
step tree chk_tree
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_rk.log 2>&1 || return 1          # the two stage-A checkpoints (written by rh_chain.sh's inputs step)
  [ -s $HERE/dscale10k_check_rk.py ] && [ -s $R/study/dscale10k_merge.py ] || { echo "no $HERE/dscale10k_check_rk.py / $R/study/dscale10k_merge.py" >> $R/logs/inputs_verify_rk.log; return 1; }
  for f in $HERE/rk_arm.sh $HERE/chk_arm_rk.sh $HERE/rk_argv.py $HERE/rk_env.sh $HERE/rk_gsuite.sh $HERE/rk_gate_record.py $R/rh_selfcheck.py $R/rh_evidence.py $GSUITE_RK $R/pilot/PILOT_RH.json $DS/argv/U_s0.B.tmpl.json $DS/argv/U_s1.B.tmpl.json $DS/chainlib.py \
           $KC/s0_kin_v1.npz $KC/s1_kin_v1.npz $KC/s0_selfcheck_cpu.json $KC/s1_selfcheck_cpu.json; do
    [ -s "$f" ] || { echo "missing input $f" >> $R/logs/inputs_verify_rk.log; return 1; }
  done
  for s in 0 1; do                                                                         # the SAME banks and the existing comparators: finished, with checkpoints (RL: the argv template and the MECH3 contrast)
    for t in RH_s$s RL_s$s B0_s$s; do
      [ -e $R/runs/$t/ARM_DONE ] && [ -s $R/runs/$t/train/ckpt_last.pt ] || { echo "$t is not finished (ARM_DONE + ckpt): RK compares against it" >> $R/logs/inputs_verify_rk.log; return 1; }
    done
    [ -s $R/bank/s$s/BANK_CHECK.json ] && [ -s $R/bank/s$s/bank.json ] || { echo "bank s$s missing" >> $R/logs/inputs_verify_rk.log; return 1; }
  done
}
step inputs chk_inputs
free_gb() { df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; }
disk_ok() { f=$(free_gb); say "free disk ${f} GB"; [ "${f:-0}" -ge 20 ]; }
step disk0 disk_ok
# 8. the two argv files
rn() { $PY $HERE/rk_argv.py render > $R/logs/rk_render.log 2>&1; tail -n 1 $R/logs/rk_render.log | grep -q "ZZ RK-RENDER PASS"; }
step render rn
# 9/10. the arms (a partial arm is moved aside, never deleted; a finished one is skipped)
arm() {
  local tag=$1
  [ -e $R/runs/$tag/ARM_DONE ] && [ -s $R/runs/$tag/train/ckpt_last.pt ] && { say "skip $tag (ARM_DONE, ckpt present)"; return 0; }
  if [ -d $R/runs/$tag/train ] || [ -d $R/runs/$tag/selfcheck ]; then
    local A=$R/runs/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train selfcheck; do [ -e $R/runs/$tag/$d ] && mv $R/runs/$tag/$d $A/; done
    rm -f $R/runs/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $HERE/rk_arm.sh $tag
}
for s in 0 1; do
  disk_ok || fail "free disk below 20 GB before RK_s$s"
  step RK_s$s arm RK_s$s
done
# 11. the end (as the RA chain)
say "free disk $(free_gb) GB at the end"
echo "ZZ ROLLHIST-RK DONE ZZ"
