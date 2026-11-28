#!/bin/bash
# SEED3 chain "S3" (card PREREG_V8F_SEED3 frozen sha256 75f3b191): the third seed pair of rolled history on UE_s2.A: B0_s2, then the s2 bank (by UE_s2.A, RH's generator), then RH_s2, then KI_s2. STAGED, NOT LAUNCHED (the MM launches). One process, one log.
# Modelled on rh_ki_chain.sh (pre-lock validations, lock held to the end) and rh_k_chain.sh (the bank step, then the arm). Code of record: B0_s2 / RH_s2 at tree_cdb 97fe0a56 (the commit B0_s0 / RH_s0 trained at), KI_s2 at tree_ki 4097a89e.
# Order: (1-5) VALIDATE, seconds, BEFORE the lock, every time (done-markers are written, never trusted): the kit's own files (SHA256_S3.txt) and the shared pod files it reads (SHARED_S3.sha256); both trees blob by blob;
#        the two G-SUITE results; UE_s2.A's sha256 and size; the B0_s2 argv built in memory (and RH_s2 / KI_s2 with a placeholder sha), the seed-2 template, the gate-record inputs (G-SUITE, pilot).  Any refusal is BEFORE the lock.
#        (6) flock fd 9 on /root/gpu_queue.lock, HELD TO THE END; every exit hands it back (EXIT trap). The log carries `LOCK-START epoch=<s>`: the MM's waiter applies the 4.0 pod-h (14,400 s) return line from it.
#        (7) trees, inputs, disk  (8) B0_s2 argv + pair skeleton rendered  (9) B0_s2 (s3_arm.sh)  (10) the s2 bank (4 shards, merge, stock check, s3_bank_check.py: generator record names UE_s2.A, coverage >= 0.5)  (11) RH_s2 / KI_s2 argv rendered
#        (12) RH_s2  (13) KI_s2  (14) the end. Every step leaves $R/done_s3/<name>.done; a relaunch skips finished steps (a half-finished arm is MOVED ASIDE, never deleted; an incomplete bank dir is deleted by literal path: own temp, logged).
# DRY=1: a static pre-lock walk that WRITES NOTHING, takes no lock, touches no marker, waits for nothing. Last line ZZ ROLLHIST-S3 DRY PASS ZZ / DRY FAIL ZZ <why>.
# Log: /root/rollhist/rh_s3_chain.log ; its LAST line is `ZZ ROLLHIST-S3 DONE ZZ` or `ZZ ROLLHIST-S3 FAIL ZZ <why>` (a coverage failure reads `... FAIL ZZ BLOCKED ...`). STOP file: $R/chains/STOP, read between steps.
# MARKER PROTOCOL: rh_s3_marker_watch.sh (sets /root/N1DW_READY if absent, clears its own token at the "GPU queue lock held to the end of the chain" line). LAUNCH: launch_s3.sh (MM only), RH_READOUT=causal_g1x.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/s3_env.sh
DRYRUN=0; [ "${DRY:-0}" = 1 ] && DRYRUN=1
D=$R/done_s3
LOG=$R/rh_s3_chain.log
BCLOG=$R/logs/blobcheck_s3
if [ $DRYRUN = 1 ]; then
  BCLOG=/dev/null.unused
  say() { echo "$(date -u +%FT%TZ) [DRY] $*"; }
  fail() { echo "ZZ ROLLHIST-S3 DRY FAIL ZZ $1"; exit 1; }
  [ "$(cat $R/RH_READOUT.chosen 2>/dev/null)" = causal_g1x ] || fail "$R/RH_READOUT.chosen is not causal_g1x"
  RH_READOUT=causal_g1x; DECL_TOKEN=causal_g1x
else
  mkdir -p $D $R/logs $R/runs $R/chains
  exec >> $LOG 2>&1
  say() { echo "$(date -u +%FT%TZ) $*"; }
  fail() { echo "ZZ ROLLHIST-S3 FAIL ZZ $1"; exit 1; }
  trap 'flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null' EXIT          # a refusal (or any exit) hands the GPU queue lock back at once, even if a child still holds the descriptor
  rh_readout_flags
  [ "$RH_READOUT" = causal_g1x ] || fail "RH_READOUT is $RH_READOUT: the s2 bank and arms are causal_g1x"
fi
s3_arm_env B0_s2; export_py
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "step $n${T_LOCK:+ (lock held $(( $(date +%s) - T_LOCK )) s)}"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
vstep() { local n=$1; shift; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "validate $n"; "$@" || fail "step $n (refused before the lock)"; [ $DRYRUN = 1 ] || date -u +%FT%TZ > $D/$n.done; }

# ---------------- the validations (shared by the real chain and DRY=1); none of them writes under $R except the real chain's blobcheck logs
chk_kit() { (cd $HERE && sha256sum -c SHA256_S3.txt > /dev/null 2>&1) || { echo "the kit's files do not match $HERE/SHA256_S3.txt"; return 1; }; echo "kit: $(wc -l < $HERE/SHA256_S3.txt) files match SHA256_S3.txt"; }
chk_shared() {   # the pod files the kit reads (not its own): pinned at the build; S3_TEST=1 (dev-box test only) reads the pin files from $R
  local fr=$HERE/SHARED_S3.sha256 fd=$HERE/SHARED_DS_S3.sha256
  [ "${S3_TEST:-0}" = 1 ] && { fr=$R/SHARED_S3.sha256; fd=$R/SHARED_DS_S3.sha256; }
  (cd $R && sha256sum -c $fr > /dev/null 2>&1) || { echo "a shared pod file under $R changed since the kit was built ($fr)"; (cd $R && sha256sum -c $fr 2>&1 | grep -v ": OK" | head -5); return 1; }
  (cd $DS && sha256sum -c $fd > /dev/null 2>&1) || { echo "a shared file under $DS changed since the kit was built ($fd)"; (cd $DS && sha256sum -c $fd 2>&1 | grep -v ": OK" | head -5); return 1; }
  echo "shared: $(cat $fr $fd | wc -l) pod files match the pins"
}
chk_one_tree() {   # TREE BLOBS NAME
  local lg=$BCLOG.$3.log; [ $DRYRUN = 1 ] && lg=/dev/null
  $PY $R/blobcheck.py $1 $2 > $lg 2>&1 || { echo "$1 fails the blob check against ${2##*/}"; return 1; }
  [ "$(find $1 -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $2)" ] || { echo "$1 file count differs from $(wc -l < $2)"; return 1; }
  echo "$1: $(wc -l < $2) blobs verified"
}
chk_trees() { chk_one_tree $RH_TREE_DIR $RH_BLOBS rh && chk_one_tree $KI_TREE_DIR $KI_BLOBS ki; }
gs_valid() {   # JSON COMMIT
  [ -s $1 ] || { echo "no $1: the G-SUITE is never computed here"; return 1; }
  $PY - $1 $2 <<'PYE'
import json, sys
r = json.load(open(sys.argv[1]))
ok = (r.get("new_commit") == sys.argv[2] or r.get("new") == sys.argv[2]) and not r.get("n_regressions") and not r.get("new_test_failures") and not r.get("missing_in_new")
print(sys.argv[1].rsplit("/", 1)[-1] + ": new", str(r.get("new_commit", r.get("new")))[:8], "n_regressions", r.get("n_regressions"), "new_test_failures", len(r.get("new_test_failures") or []), "missing", len(r.get("missing_in_new") or []), "->", "VALID" if ok else "NOT A PASS AT THIS COMMIT")
sys.exit(0 if ok else 1)
PYE
}
chk_gsuite() { gs_valid $RH_GSUITE $RH_COMMIT && gs_valid $KI_GSUITE $KI_COMMIT; }
chk_ue2() {
  local f=$UE2 sha=$UE2_SHA sz=$UE2_SIZE
  [ "${S3_TEST:-0}" = 1 ] && { f=${S3_TEST_UE2:-$UE2}; sha=${S3_TEST_UE2_SHA:-$UE2_SHA}; sz=${S3_TEST_UE2_SIZE:-$UE2_SIZE}; }     # dev-box test only: a planted stand-in file
  [ "$(stat -c %s $f)" = $sz ] || { echo "$f is not $sz bytes"; return 1; }
  [ "$(sha256sum $f | cut -d' ' -f1)" = $sha ] || { echo "$f sha256 is not $sha"; return 1; }
  echo "UE_s2.A: sha256 $sha, $sz B"
}
chk_argv() {
  $PY $HERE/s3_argv.py show --arm B0_s2 || return 1
  $PY $HERE/s3_argv.py show --arm RH_s2 --bank-sha @BANKSHA@ || return 1
  $PY $HERE/s3_argv.py show --arm KI_s2 --bank-sha @BANKSHA@ || return 1
  $PY $HERE/s3_argv.py template --check $HERE/U_s2.B.tmpl.json || return 1
}
chk_gate_inputs() {
  $PY $HERE/rh_gate_record_s3.py precheck-lite --tag B0_s2 --readout $RH_READOUT || return 1
  $PY $HERE/rh_gate_record_s3.py precheck-lite --tag RH_s2 --readout $RH_READOUT || return 1
  $PY $HERE/ki_gate_record_s3.py precheck-lite --tag KI_s2 --readout $RH_READOUT || return 1
}
chk_inputs() {
  local f
  for f in $HERE/s3_arm.sh $HERE/chk_arm_s3.sh $HERE/s3_argv.py $HERE/s3_env.sh $HERE/s3_bank_check.py $HERE/dscale10k_check_s3.py $HERE/rh_gate_record_s3.py $HERE/ki_gate_record_s3.py $HERE/U_s2.B.tmpl.json \
           $R/rh_selfcheck.py $R/rh_evidence.py $R/rh_bank.sh $R/rh_pilot_check.py $R/study/dscale10k_merge.py $R/pilot/PILOT_RH.json $DS/argv/B_s0.B.tmpl.json $DS/argv/UE_s2.A.json $DS/dry/UE_s2.A.dry.json \
           $DS/gate/evidence_UE_s2.json $DS/chainlib.py $S1/eval_grid_windows.json $S1/n1dc/kclip.py $R/blobcheck.py; do
    [ -s "$f" ] || { echo "missing input $f"; return 1; }
  done
  for t in B0_s0 RH_s0 KI_s0; do
    [ -e $R/runs/$t/ARM_DONE ] && [ -s $R/runs/$t/argv.json ] || { echo "$t is not finished (ARM_DONE + argv): the s2 argv are rendered from it"; return 1; }
  done
  [ ! -e $R/runs/B0_s2/ARM_DONE ] || [ -s $R/runs/B0_s2/train/ckpt_last.pt ] || { echo "B0_s2 has ARM_DONE but no checkpoint"; return 1; }
}

# ---------------- DRY=1: the static pre-lock walk, nothing written
if [ $DRYRUN = 1 ]; then
  say "DRY chain walk, readout=$RH_READOUT arms B0_s2 (tree_cdb ${RH_COMMIT:0:8}) RH_s2 (tree_cdb) KI_s2 (tree_ki ${KI_COMMIT:0:8}); writes nothing, no lock, no marker, no wait"
  vstep kit chk_kit
  vstep shared chk_shared
  vstep trees chk_trees
  vstep gsuite chk_gsuite
  vstep ue2 chk_ue2
  vstep inputs chk_inputs
  vstep argv chk_argv
  vstep gate_inputs chk_gate_inputs
  say "would now take flock fd 9 on /root/gpu_queue.lock (held to the end), log LOCK-START, then: trees, disk, render B0_s2, B0_s2, the s2 bank + check, render RH_s2 / KI_s2, RH_s2, KI_s2"
  echo "ZZ ROLLHIST-S3 DRY PASS ZZ"
  exit 0
fi

say "chain start readout=$RH_READOUT (decl token $DECL_TOKEN; bank flags: $BANK_RO_FLAGS) arms B0_s2 RH_s2 (tree_cdb ${RH_COMMIT:0:8}) KI_s2 (tree_ki ${KI_COMMIT:0:8}); the s2 bank by UE_s2.A"
vstep kit chk_kit
vstep shared chk_shared
vstep trees chk_trees
vstep gsuite chk_gsuite
vstep ue2 chk_ue2
vstep inputs chk_inputs
vstep argv chk_argv
vstep gate_inputs chk_gate_inputs
# 6. the GPU queue lock, held to the end
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
T_LOCK=$(date +%s)
say "LOCK-START epoch=$T_LOCK utc=$(date -u +%FT%TZ) (return line 4.0 pod-h = 14400 s of lock: epoch $((T_LOCK + 14400)))"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
step tree chk_trees
free_gb() { df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; }
disk_ok() { f=$(free_gb); say "free disk ${f} GB (need ${1:-20})"; [ "${f:-0}" -ge ${1:-20} ]; }
step disk0 disk_ok 20
date -u +%FT%TZ > $D/tree_rh.done; date -u +%FT%TZ > $D/tree_ki.done          # the gate records' tree markers: written ONLY after the blob check of the step above passed under the lock
# 8. B0_s2's argv + the pair skeleton
rn_b0() { $PY $HERE/s3_argv.py render --arms B0_s2 > $R/logs/s3_render_b0.log 2>&1; tail -n 1 $R/logs/s3_render_b0.log | grep -q "ZZ S3-ARGV RENDER PASS"; }
step render_b0 rn_b0
# arm: a partial arm is moved aside, never deleted; a finished one is skipped
arm() {
  local tag=$1
  [ -e $R/runs/$tag/ARM_DONE ] && [ -s $R/runs/$tag/train/ckpt_last.pt ] && { say "skip $tag (ARM_DONE, ckpt present)"; return 0; }
  if [ -d $R/runs/$tag/train ] || [ -d $R/runs/$tag/selfcheck ]; then
    local A=$R/runs/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train selfcheck; do [ -e $R/runs/$tag/$d ] && mv $R/runs/$tag/$d $A/; done
    rm -f $R/runs/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $HERE/s3_arm.sh $tag
}
# 9. B0_s2 (needs no bank)
disk_ok 20 || fail "free disk below 20 GB before B0_s2"
step B0_s2 arm B0_s2
# 10. the s2 bank by UE_s2.A with RH's generator invocation (rh_chain.sh:mk_bank): 4 shards in parallel, merge, stock full-bank check (b, c, d, e), then s3_bank_check.py
mk_bank_s2() {
  local B=$R/bank/s2 ba ec ew A
  disk_ok 30 || { say "free disk below 30 GB before the s2 bank (the s0 bank is 2.0 GB)"; return 1; }
  if [ -d "$B" ] && [ ! -s "$B/S3_BANK_CHECK.json" ]; then       # an incomplete s2 bank of an earlier launch: own temp, re-creatable, deleted by literal path, logged
    [ "$B" = "$R/bank/s2" ] || { say "refusing to delete $B"; return 1; }
    say "incomplete bank dir $B deleted ($(du -sk "$B" | cut -f1) KiB)"; echo "$(date -u +%FT%TZ) $B $(du -sk "$B" | cut -f1)KiB (incomplete s2 bank of a failed launch)" >> $R/deletions.log
    rm -rf "${B:?}"
  fi
  $PY $DS/chainlib.py m2bargv $DS/argv/UE_s2.A.json $R/bank/m2b_argv_s2.json || return 1
  ba=$($PY $DS/chainlib.py bankargs $DS/dry/UE_s2.A.dry.json) || return 1; ec=${ba% *}; ew=${ba#* }
  A=$($PY - $DS/argv/UE_s2.A.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
g = lambda f: t[t.index(f) + 1]
print(g("--cache-dir"), g("--tracks"), g("--join-train"))
PYE
) || return 1
  set -- $A
  say "bank s2: n_shards 4 parallel, $ec clips, $ew windows, ckpt UE_s2.A, flags [$BANK_RO_FLAGS]"
  s3_arm_env B0_s2; export_py
  bash $R/rh_bank.sh $TREE $UE2 $R/bank/m2b_argv_s2.json $1 $2 $3 $B $ec $ew 0 "$BANK_RO_FLAGS" > $R/bank/bank_s2.log 2>&1 || { tail -n 3 $R/bank/bank_s2.log; return 1; }
  RH_TREE=$TREE $PY $R/rh_pilot_check.py bank --bank $B --readout $RH_READOUT --expect-windows $ew --out $B/BANK_CHECK.json > $R/bank/bank_check_s2.log 2>&1 || {
    if grep -q "coverage" $R/bank/bank_check_s2.log; then echo "ZZ ROLLHIST-S3 FAIL ZZ BLOCKED coverage (< 0.5 or blocked) in the s2 bank: $(tail -c 500 $R/bank/bank_check_s2.log | tr '\n' ' ')"; exit 1; fi
    tail -c 600 $R/bank/bank_check_s2.log; return 1; }
  $PY $HERE/s3_bank_check.py --bank $B --expect-windows $ew --ue-sha $UE2_SHA --out $B/S3_BANK_CHECK.json > $R/bank/s3_bank_check.log 2>&1 || {
    if grep -q "coverage" $R/bank/s3_bank_check.log; then echo "ZZ ROLLHIST-S3 FAIL ZZ BLOCKED coverage in the s2 bank: $(tail -c 500 $R/bank/s3_bank_check.log | tr '\n' ' ')"; exit 1; fi
    tail -c 600 $R/bank/s3_bank_check.log; return 1; }
  tail -n 2 $R/bank/s3_bank_check.log
}
step bank_s2 mk_bank_s2
# 11. RH_s2 / KI_s2 argv, rendered after the bank (its sha256); then the pair-skeleton consistency (RH_s2 = skeleton with the sha filled in)
rn_rhki() { $PY $HERE/s3_argv.py render --arms RH_s2 KI_s2 > $R/logs/s3_render_rhki.log 2>&1 && tail -n 1 $R/logs/s3_render_rhki.log | grep -q "ZZ S3-ARGV RENDER PASS" && $PY $HERE/s3_argv.py verify --arms B0_s2 RH_s2 KI_s2 > $R/logs/s3_verify_all.log 2>&1; }
step render_rhki rn_rhki
for t in RH_s2 KI_s2; do
  disk_ok 20 || fail "free disk below 20 GB before $t"
  step $t arm $t
done
say "free disk $(free_gb) GB at the end; lock held $(( $(date +%s) - T_LOCK )) s; s2 bank kept: $(du -sh $R/bank/s2 2>/dev/null | tr '\n' ' ')"
echo "ZZ ROLLHIST-S3 DONE ZZ"
