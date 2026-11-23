#!/bin/bash
# K-INPUT chain "KI" (commit 4097a89e, card PREREG_V8F_KINPUT frozen sha256 05e786a8): RH's stage-B argv + `--dagger-base logged` (and its own --out) on the SAME RH bank, SAME dose p = 0.1, swap mode, human label, NO label cache.
# Code of record the commit 4097a89e (ki_env.sh: own tree tree_ki = tree_rk (1a8b11a1) + 9 blobs). Readout causal_g1x. STAGED, NOT LAUNCHED. One process, one log. Modelled on rh_rk_chain.sh. Differences from the RK chain, and only these:
#   * it WAITS FOR NO OTHER CHAIN: no ADD123 wait; it starts at launch. The tree prep, the G-SUITE and the dry mint are run BEFORE this chain (CPU, nice 19, outside the lock; PREP_KI.md); the chain only VALIDATES them, before the lock and in seconds,
#     every time (the done-markers are written but never trusted): the kit's own files against SHA256_KI.txt; tree_ki blob by blob against all_blobs_ki.txt (3,425 rows); gsuite_result_ki.json at commit 4097a89e with 0 regressions / 0 new-test
#     failures / 0 missing (a missing or failing result REFUSES before the lock: it is never computed here, never overwritten); the KI argv built in memory (ki_argv.py show: RH's argv + exactly two changes) and the gate-record INPUTS
#     (ki_gate_record.py precheck on that argv: G-SUITE commit, pilot, bank check, bank sha, no other lever flag). Any refusal is BEFORE the lock;
#   * no label cache, no precompute: the human label needs none;
#   * the lock (fd 9, flock on /root/gpu_queue.lock) is taken after those checks and HELD TO THE END; every exit, a refusal included, hands it back at once (EXIT trap: flock -u);
#   * render = ki_argv.py (RH's argv + `--dagger-base logged` and --out); the arms are KI_s0, KI_s1 (stage B only; each inits from its own UE_sK.A); RH / B0 are finished arms, only read;
#   * the study-side checker is $HERE/dscale10k_check_ki.py (the KI kind); the shared pod files (rh_env.sh, rh_selfcheck.py, rh_evidence.py, blobcheck.py, study/dscale10k_merge.py) are read from $R.
#   * DRY=1: a static pre-lock walk that WRITES NOTHING, takes no lock, touches no marker and does not wait: it runs the same validations. Last line ZZ ROLLHIST-KI DRY PASS ZZ / DRY FAIL ZZ <why>.
# Order: (1) kit  (2) tree  (3) G-SUITE result  (4) argv  (5) gate-record inputs  (6) flock fd 9, HELD TO THE END  (7) tree check, inputs, disk  (8) render the two argv files (ki_argv.py)
#        (9) KI_s0 (ki_arm.sh: argv verify, pair check, selfcheck, gate record incl. the base-lever checks, study-side token, train 3,000 steps through the token, lever-acts, summary, post gates, zerograd, K-CLIP)  (10) KI_s1  (11) the end steps.
# Every step leaves $R/done_ki/<name>.done; a relaunch skips finished arm steps (a half-finished arm is MOVED ASIDE, never deleted). STOP file: $R/chains/STOP, read between steps.
# Log: /root/rollhist/rh_ki_chain.log ; its LAST line is `ZZ ROLLHIST-KI DONE ZZ` or `ZZ ROLLHIST-KI FAIL ZZ <why>`.
# MARKER PROTOCOL: rh_ki_marker_watch.sh (see there). LAUNCH: see launch_ki.sh (MM only).
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/ki_env.sh
DRYRUN=0; [ "${DRY:-0}" = 1 ] && DRYRUN=1
D=$R/done_ki
LOG=$R/rh_ki_chain.log
BCLOG=$R/logs/blobcheck_ki.log
if [ $DRYRUN = 1 ]; then
  BCLOG=/dev/null
  say() { echo "$(date -u +%FT%TZ) [DRY] $*"; }
  fail() { echo "ZZ ROLLHIST-KI DRY FAIL ZZ $1"; exit 1; }
  [ "$(cat $R/RH_READOUT.chosen 2>/dev/null)" = causal_g1x ] || fail "$R/RH_READOUT.chosen is not causal_g1x"
  RH_READOUT=causal_g1x; DECL_TOKEN=causal_g1x
else
  mkdir -p $D $R/logs $R/runs $R/chains
  exec >> $LOG 2>&1
  say() { echo "$(date -u +%FT%TZ) $*"; }
  fail() { echo "ZZ ROLLHIST-KI FAIL ZZ $1"; exit 1; }
  trap 'flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null' EXIT          # a refusal (or any exit) hands the GPU queue lock back at once, even if a child still holds the descriptor
  rh_readout_flags
  [ "$RH_READOUT" = causal_g1x ] || fail "RH_READOUT is $RH_READOUT: the RH bank and arms are causal_g1x"
fi
export_py
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
# a validation step: runs EVERY time (a stale done-marker is never trusted), writes its marker after a pass
vstep() { local n=$1; shift; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "validate $n"; "$@" || fail "step $n (refused before the lock)"; [ $DRYRUN = 1 ] || date -u +%FT%TZ > $D/$n.done; }

# ---------------- the validations (shared by the real chain and DRY=1); none of them writes under $R except the real chain's blobcheck log
chk_kit() { (cd $HERE && sha256sum -c SHA256_KI.txt > /dev/null 2>&1) || { echo "the kit's files do not match $HERE/SHA256_KI.txt"; return 1; }; echo "kit: $(wc -l < $HERE/SHA256_KI.txt) files match SHA256_KI.txt"; }
chk_tree() {
  $PY $R/blobcheck.py $TREE $BLOBS > $BCLOG 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || return 1
}
tp() {
  [ -d $TREE ] || { echo "no $TREE: the tree prep (ki_tree_prep.sh) is a step before the chain"; return 1; }
  chk_tree || { echo "tree_ki fails the blob check against ${BLOBS##*/} (commit ${COMMIT:0:8}): refusing, not rebuilding"; return 1; }
  echo "tree_ki: $(wc -l < $BLOBS) blobs verified"
}
gs_valid() {                      # the result file's own commit and counts
  [ -s $GSUITE_KI ] || return 2
  $PY - $GSUITE_KI $COMMIT <<'PYE'
import json, sys
r = json.load(open(sys.argv[1]))
ok = (r.get("new_commit") == sys.argv[2] or r.get("new") == sys.argv[2]) and not r.get("n_regressions") and not r.get("new_test_failures") and not r.get("missing_in_new")
print("gsuite_result_ki.json: new", r.get("new_commit", r.get("new")), "n_regressions", r.get("n_regressions"), "new_test_failures", len(r.get("new_test_failures") or []), "missing", len(r.get("missing_in_new") or []), "->", "VALID" if ok else "NOT A PASS AT THIS COMMIT")
sys.exit(0 if ok else 1)
PYE
}
gs_log_done() { [ -s $HERE/prep_logs/ki_gsuite.log ] && tail -n 1 $HERE/prep_logs/ki_gsuite.log | grep -q "ZZ KI GSUITE DONE ZZ"; }
gs() {
  gs_valid; local rc=$?
  [ $rc = 2 ] && { echo "no $GSUITE_KI: the G-SUITE (ki_gsuite.sh, about 37 min) is a step before the chain; it is never run or recomputed here"; return 1; }
  [ $rc = 0 ] || { echo "$GSUITE_KI exists and is NOT a pass at ${COMMIT:0:8}: refusing"; return 1; }
  gs_log_done || { echo "$GSUITE_KI is valid but $HERE/prep_logs/ki_gsuite.log does not end in ZZ KI GSUITE DONE ZZ"; return 1; }
  say "G-SUITE at ${COMMIT:0:8} passed earlier: kept (ki_gate_record.py re-checks it at mint)"
}
chk_argv() { $PY $HERE/ki_argv.py show --seeds 0 1; }
chk_gate_inputs() {               # the gate-record inputs on the in-memory KI argv (no argv file exists before the render under the lock)
  local s J
  for s in 0 1; do
    J=$($PY $HERE/ki_argv.py json --seeds $s) || return 1
    printf '%s' "$J" | $PY $HERE/ki_gate_record.py precheck --tag KI_s$s --readout $RH_READOUT --argv - || return 1
  done
}

# ---------------- DRY=1: the static pre-lock walk, nothing written
if [ $DRYRUN = 1 ]; then
  say "DRY chain walk, commit=${COMMIT:0:8} readout=$RH_READOUT arms KI_s0 KI_s1; writes nothing, no lock, no marker, no wait, waits for no other chain"
  vstep kit chk_kit
  vstep tree_prep tp
  vstep gsuite gs
  vstep argv chk_argv
  vstep gate_inputs chk_gate_inputs
  say "would now take flock fd 9 on /root/gpu_queue.lock (held to the end), then: tree check, inputs, disk0, render, KI_s0 (ki_arm.sh), KI_s1, end steps"
  echo "ZZ ROLLHIST-KI DRY PASS ZZ"
  exit 0
fi

say "chain start commit=${COMMIT:0:8} readout=$RH_READOUT (decl token $DECL_TOKEN) arms KI_s0 KI_s1 (--dagger-p 0.1 --dagger-base logged, human label, same banks, no label cache)"
# 1-5. validations, all BEFORE the lock (seconds); a stale marker is never trusted
vstep kit chk_kit
vstep tree_prep tp
vstep gsuite gs
vstep argv chk_argv
vstep gate_inputs chk_gate_inputs
# 6. the GPU queue lock, held to the end
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 7. tree, inputs, disk
step tree chk_tree
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_ki.log 2>&1 || return 1          # the two stage-A checkpoints (written by rh_chain.sh's inputs step)
  [ -s $HERE/dscale10k_check_ki.py ] && [ -s $R/study/dscale10k_merge.py ] || { echo "no $HERE/dscale10k_check_ki.py / $R/study/dscale10k_merge.py" >> $R/logs/inputs_verify_ki.log; return 1; }
  for f in $HERE/ki_arm.sh $HERE/chk_arm_ki.sh $HERE/ki_argv.py $HERE/ki_env.sh $HERE/ki_gsuite.sh $HERE/ki_gate_record.py $R/rh_selfcheck.py $R/rh_evidence.py $GSUITE_KI $R/pilot/PILOT_RH.json $DS/argv/U_s0.B.tmpl.json $DS/argv/U_s1.B.tmpl.json $DS/chainlib.py; do
    [ -s "$f" ] || { echo "missing input $f" >> $R/logs/inputs_verify_ki.log; return 1; }
  done
  for s in 0 1; do                                                                         # the SAME banks and the existing comparators: finished, with checkpoints (RH: the argv template; B0: the pair)
    for t in RH_s$s B0_s$s; do
      [ -e $R/runs/$t/ARM_DONE ] && [ -s $R/runs/$t/train/ckpt_last.pt ] || { echo "$t is not finished (ARM_DONE + ckpt): KI compares against it" >> $R/logs/inputs_verify_ki.log; return 1; }
    done
    [ -s $R/bank/s$s/BANK_CHECK.json ] && [ -s $R/bank/s$s/bank.json ] || { echo "bank s$s missing" >> $R/logs/inputs_verify_ki.log; return 1; }
  done
}
step inputs chk_inputs
free_gb() { df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; }
disk_ok() { f=$(free_gb); say "free disk ${f} GB"; [ "${f:-0}" -ge 20 ]; }
step disk0 disk_ok
# 8. the two argv files
rn() { $PY $HERE/ki_argv.py render > $R/logs/ki_render.log 2>&1; tail -n 1 $R/logs/ki_render.log | grep -q "ZZ KI-RENDER PASS"; }
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
  bash $HERE/ki_arm.sh $tag
}
for s in 0 1; do
  disk_ok || fail "free disk below 20 GB before KI_s$s"
  step KI_s$s arm KI_s$s
done
# 11. the end
say "free disk $(free_gb) GB at the end"
echo "ZZ ROLLHIST-KI DONE ZZ"
