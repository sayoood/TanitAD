#!/bin/bash
# ROLLHIST arm chain "RHK": a DAgger bank generated under readout K (--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2, the eval's R0KHb20, token causal_g0x), then RH-like arms at p = 0.1 trained on it.
# Amendment 8 DRAFT. Code of record 97fe0a56. STAGED, NOT LAUNCHED. One process, one log.
# Modelled on /root/rollhist/rh_p03_chain.sh plus the BANK STEP of /root/rollhist/rh_chain.sh (mk_bank). Differences from rh_chain.sh, and only these: the bank goes to $R/bank_k/ (NEVER $R/bank/), the readout flags are K's
# (rhk_env.sh), the bank check is rhk_bank_check.py (gain 0 + vacuity control), the argv is rendered AFTER each bank (rhk_argv.py: RH's argv with only the bank flags and --out changed), no G-SUITE / pilot step (both are the RH
# chain's, at the SAME commit, bound by gsuite_result_rh.json / PILOT_RH.json which rhk_gate_record.py re-hashes at the mint), arms RHK_s0, RHK_s1 (stage B only; K-B2 holds: each arm inits from its own UE_sK.A).
# The comparator is the EXISTING B0_s0 / B0_s1 (read only). RH_s0 / RH_s1 and the RH bank are read-only inputs.
# Order: (1) flock fd 9 on /root/gpu_queue.lock, HELD TO THE END  (2) tree + inputs + disk  (3) bank_k s0 (4 shards, merge) + full-bank check (b,c,d,e; coverage >= 0.5 else BLOCKED) + render RHK_s0 argv  (4) RHK_s0
#        (5) bank_k s1 + check + render  (6) RHK_s1.
# RHK_BANK_ONLY=1 (env, MM's optional dry-mint route): builds and checks the K banks and renders nothing further than the argv, runs NO arm; a later launch without it skips the banks (done markers) and runs the arms.
# Every step leaves $R/done_k/<name>.done; a relaunch skips finished steps (a half-finished arm is MOVED ASIDE, never deleted; an incomplete bank_k dir is deleted by literal path: own temp, logged). STOP file: $R/chains/STOP.
# Log: /root/rollhist/rh_k_chain.log ; its LAST line is `ZZ ROLLHIST-K DONE ZZ` or `ZZ ROLLHIST-K FAIL ZZ <why>` (a coverage failure reads `... FAIL ZZ BLOCKED ...`).
# MARKER PROTOCOL (as rh_p03_marker_watch.sh): rh_k_marker_watch.sh sets /root/N1DW_READY and removes it when this chain holds the lock.   LAUNCH: launch_k.sh (MM only), RH_READOUT=causal_g0x.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rhk_env.sh
D=$R/done_k; mkdir -p $D $R/logs $R/runs $R/chains $R/bank_k
LOG=$R/rh_k_chain.log
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ ROLLHIST-K FAIL ZZ $1"; exit 1; }
rh_readout_flags
[ "$RH_READOUT" = causal_g0x ] || fail "RH_READOUT is $RH_READOUT: the RHK bank is causal_g0x"
export_py
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
say "chain start commit=${COMMIT:0:8} readout=$RH_READOUT (decl token $DECL_TOKEN; bank flags: $BANK_RO_FLAGS) arms RHK_s0 RHK_s1 (--dagger-p 0.1, bank under $R/bank_k)"
# 1. the GPU queue lock, held to the end
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 2. tree, inputs, disk
chk_tree() {
  $PY $R/blobcheck.py $TREE $R/all_blobs_97fe0a56.txt > $R/logs/blobcheck_k.log 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $R/all_blobs_97fe0a56.txt)" ] || return 1
}
step tree chk_tree
chk_inputs() {
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_k.log 2>&1 || return 1
  for f in $R/study/dscale10k_check_rhk.py $R/study/dscale10k_merge.py $HERE/rhk_arm.sh $HERE/chk_arm_rhk.sh $HERE/rhk_argv.py $HERE/rhk_pair.py $HERE/rhk_bank_check.py $HERE/rhk_gate_record.py $HERE/rhk_env.sh \
           $HERE/rh_bank.sh $HERE/rh_pilot_check.py $R/gsuite_result_rh.json $R/pilot/PILOT_RH.json $DS/argv/U_s0.B.tmpl.json $DS/argv/U_s1.B.tmpl.json $DS/chainlib.py \
           $DS/argv/UE_s0.A.json $DS/argv/UE_s1.A.json $DS/dry/UE_s0.A.dry.json $DS/dry/UE_s1.A.dry.json $S1/eval_grid_windows.json; do
    [ -s "$f" ] || { echo "missing input $f" >> $R/logs/inputs_verify_k.log; return 1; }
  done
  for s in 0 1; do                                                                         # the finished RH and B0 arms (RHK's argv is RH's; the comparator is B0) and the RH bank (read only)
    for t in RH_s$s B0_s$s; do
      [ -e $R/runs/$t/ARM_DONE ] && [ -s $R/runs/$t/train/ckpt_last.pt ] && [ -s $R/runs/$t/argv.json ] || { echo "$t is not finished (ARM_DONE + ckpt + argv)" >> $R/logs/inputs_verify_k.log; return 1; }
    done
    [ -s $R/bank/s$s/bank.json ] || { echo "RH bank s$s missing" >> $R/logs/inputs_verify_k.log; return 1; }
  done
}
step inputs chk_inputs
free_gb() { df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; }
disk_ok() { f=$(free_gb); say "free disk ${f} GB"; [ "${f:-0}" -ge ${1:-20} ]; }
# 3/5. a K bank: all shards, merge, full re-check, then the RHK argv rendered from RH's with only the bank flags changed
mk_bank_k() {   # mk_bank_k SEED
  local s=$1 B=$R/bank_k/s$1 ba ec ew A
  disk_ok 60 || { say "free disk below 60 GB before the s$s K bank"; return 1; }
  if [ -d "$B" ] && [ ! -s "$B/BANK_CHECK.json" ]; then       # an incomplete K bank of an earlier launch: own temp, re-creatable, deleted by literal path (under $R/bank_k/ only), logged
    case "$B" in "$R/bank_k/s0"|"$R/bank_k/s1") ;; *) say "refusing to delete $B"; return 1;; esac
    say "incomplete bank dir $B deleted ($(du -sk "$B" | cut -f1) KiB)"; echo "$(date -u +%FT%TZ) $B $(du -sk "$B" | cut -f1)KiB (incomplete K bank of a failed launch)" >> $R/deletions.log
    rm -rf "${B:?}"
  fi
  $PY $DS/chainlib.py m2bargv $DS/argv/UE_s$s.A.json $R/bank_k/m2b_argv_s$s.json || return 1
  ba=$($PY $DS/chainlib.py bankargs $DS/dry/UE_s$s.A.dry.json) || return 1; ec=${ba% *}; ew=${ba#* }
  A=$($PY - $DS/argv/UE_s$s.A.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
g = lambda f: t[t.index(f) + 1]
print(g("--cache-dir"), g("--tracks"), g("--join-train"))
PYE
) || return 1
  set -- $A
  say "bank_k s$s: n_shards 4 parallel, $ec clips, $ew windows, flags [$BANK_RO_FLAGS]"
  bash $HERE/rh_bank.sh $TREE $DS/runs/UE_s$s/A/ckpt_last.pt $R/bank_k/m2b_argv_s$s.json $1 $2 $3 $B $ec $ew 0 "$BANK_RO_FLAGS" > $R/bank_k/bank_s$s.log 2>&1 || { tail -n 3 $R/bank_k/bank_s$s.log; return 1; }
  RH_TREE=$TREE $PY $HERE/rhk_bank_check.py --bank $B --expect-windows $ew --out $B/BANK_CHECK.json > $R/bank_k/bank_check_s$s.log 2>&1 || {
    if grep -q "coverage" $R/bank_k/bank_check_s$s.log; then echo "ZZ ROLLHIST-K FAIL ZZ BLOCKED coverage (< 0.5 or blocked) in the K bank s$s: $(tail -c 500 $R/bank_k/bank_check_s$s.log | tr '\n' ' ')"; exit 1; fi
    tail -c 600 $R/bank_k/bank_check_s$s.log; return 1; }
  $PY $HERE/rhk_argv.py render --seeds $s > $R/logs/rhk_render_s$s.log 2>&1 || { tail -n 5 $R/logs/rhk_render_s$s.log; return 1; }
  tail -n 1 $R/logs/rhk_render_s$s.log | grep -q "ZZ RHK-RENDER PASS"
}
# 4/6. the arms (same function as rh_p03_chain.sh: a partial arm is moved aside, never deleted; a finished one is skipped)
arm() {
  local tag=$1
  [ -e $R/runs/$tag/ARM_DONE ] && [ -s $R/runs/$tag/train/ckpt_last.pt ] && { say "skip $tag (ARM_DONE, ckpt present)"; return 0; }
  if [ -d $R/runs/$tag/train ] || [ -d $R/runs/$tag/selfcheck ]; then
    local A=$R/runs/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train selfcheck; do [ -e $R/runs/$tag/$d ] && mv $R/runs/$tag/$d $A/; done
    rm -f $R/runs/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $HERE/rhk_arm.sh $tag
}
for s in 0 1; do
  step bank_k_s$s mk_bank_k $s
  [ -n "${RHK_BANK_ONLY:-}" ] && { say "RHK_BANK_ONLY set: arm RHK_s$s skipped (DRY_MINT_PLAN phase B on the real K bank)"; continue; }
  disk_ok 20 || fail "free disk below 20 GB before RHK_s$s"
  step RHK_s$s arm RHK_s$s
done
say "free disk $(free_gb) GB at the end; K banks kept: $(du -sh $R/bank_k/s0 $R/bank_k/s1 2>/dev/null | tr '\n' ' ')"
[ -n "${RHK_BANK_ONLY:-}" ] && { echo "ZZ ROLLHIST-K DONE ZZ (banks only: no arm was run)"; exit 0; }
echo "ZZ ROLLHIST-K DONE ZZ"
