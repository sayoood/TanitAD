#!/bin/bash
# blockreps1.sh -- DSCALE10K REPS1 (pod, LoRA pod): the seed-1 replicate pair UE_s1 then UM_s1 (Amendment UM-A2 DRAFT; PI go 2026-11-19 "Both, N1D8 first"), STAGE A ONLY, 4 x S_U = 61,568 steps each, union set, seed 1.
# NOT launched by the builder: the MM launches it (queue_reps1.sh). FORK of block3/block3.sh (UE arm) and blockum/blockum.sh (UM arm). Changes against the forks (ALL of them):
#   1 `cd $D` is the first command.  2 arms UE_s1 (tree = $D/tree, cdbff971) then UM_s1 (tree = $D/tree_um, 648d4ca3): each on the tree its seed-0 twin trained on.  3 prep = blockreps1_prep.sh; runner = chk_arm_reps1.sh
#   (checker = the patched copy blockreps1/study/dscale10k_check.py: arms UE_s1 / UM_s1 added, nothing else).  4 UM_s1 only: um_checks_reps1.py pre (fresh reach, token / list / selfcheck bindings) BEFORE its stage A, and
#   post (K-TACSRC: tac_src_w 0.0, list sha256, weight share exactly 1.0, rows > 0, tac_src_empty_batches <= 5 AS WRITTEN, join_union.npz unchanged) AFTER it; a miss FAILS the chain, as in blockum.sh.
#   5 ORDER on UM_s1: the checkpoint is announced to the banker BEFORE the post check, so a post-check FAIL cannot strand it (blockum.sh banked after the check; UM_s0 was banked by hand after its FAIL).  ARM_A_DONE is still
#   written only after the post check passes.  6 REFUSAL: stage A output (summary.json or ckpt_last.pt) present without ARM_A_DONE = a finished or failed earlier run -> the chain FAILS instead of retraining over it.
#   7 no BL arm, no stage B, no bank build, no mint of stage B.  8 sweep covers UE_s1 / UM_s1 only; sentinels `ZZ DSCALE10K REPS1 ...`.
# Kept: gate_wait (STOP file, >= 60 GB free, idle GPU), the token-refusing runner, the stage-A summary and post-training zero-grad gates (chainlib summary / post), the banker after the stage (same private HF repo and
# <arm>/<stage>/ layout as UE_s0 / UM_s0: dscale_banker.py is arm-agnostic).
# Ends with `ZZ DSCALE10K REPS1 DONE ZZ` or `ZZ DSCALE10K REPS1 FAIL ZZ <why>`. A file /root/dscale10k/chains/STOP ends it between steps.
#   launch (the MM), as its OWN ssh command:  nohup setsid bash /root/dscale10k/blockreps1/queue_reps1.sh < /dev/null > /root/dscale10k/blockreps1.log 2>&1 &
# Dry run: DS_FAKE=1 DS_ROOT=<fake root> (dryrun_reps1.sh).
D=${DS_ROOT:-/root/dscale10k}
cd $D || { echo "ZZ DSCALE10K REPS1 FAIL ZZ cd $D"; exit 1; }
set -u
PY=${DS_PY:-/root/venv/bin/python}; LIB="$PY $D/chainlib.py"; export DS_ROOT=$D
MINFREE=${DS_MIN_FREE_GB:-60}
STEPS_A=61568; RP=${DS_RP:-$D/blockreps1}
mkdir -p $D/chains $D/runs $D/gate/tokens $D/bank_ready $D/bank_ok
say() { echo "$(date -u +%T) $*"; }
fail() { echo "ZZ DSCALE10K REPS1 FAIL ZZ $1"; exit 1; }
[ -n "${DS_FAKE:-}" ] && LEAF=$D/fake_leaf.sh

gpu_idle() { if [ -n "${DS_FAKE:-}" ]; then return 0; fi; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
free_gb() { if [ -n "${DS_FAKE:-}" ]; then echo "${DS_FREE_GB:-500}"; else df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; fi; }
sweep_all() { for a in UE_s1 UM_s1; do [ -e $D/runs/$a/ARM_DONE ] && $LIB sweep $a; done; return 0; }
gate_wait() {   # gate_wait WHAT : STOP file, free disk (a sweep of banked data first), idle GPU
  [ -e $D/chains/STOP ] && fail "STOP file before $1"
  f=$(free_gb)
  if [ "${f:-0}" -lt "$MINFREE" ]; then sweep_all; f=$(free_gb); fi
  [ "${f:-0}" -ge "$MINFREE" ] || fail "free disk ${f} GB < ${MINFREE} GB before $1 (nothing banked to delete)"
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt ${DS_GPU_TRIES:-360} ] && fail "GPU not idle (>= 3500 MiB used) before $1"; sleep ${DS_POLL:-10}; done
}
leaf() {        # leaf KIND ARM STAGE : the operations a dry run replaces
  case $1 in
    run)  if [ -n "${DS_FAKE:-}" ]; then bash $LEAF run $2 $3; else DS_ROOT=$D bash $RP/chk_arm_reps1.sh run $2 $3; fi;;
    prep) if [ -n "${DS_FAKE:-}" ]; then bash $LEAF prep $2 $3; else DS_ROOT=$D bash $RP/blockreps1_prep.sh; fi;;
  esac
}
bank_now() {    # bank_now [TAG] : run the banker (the only writer of bank_ok); a failure defers the deletions and never stops the chain
  if [ -n "${DS_FAKE:-}" ]; then $PY $D/dscale_banker.py --root $D --local-hub $D/fakehub ${1:+--only $1} >> $D/banker.log 2>&1 || say "BANK-DEFERRED ${1:-all}: see $D/banker.log"; return 0; fi
  $PY $D/dscale_banker.py --root $D ${1:+--only $1} >> $D/banker.log 2>&1 || say "BANK-DEFERRED ${1:-all}: see $D/banker.log (bank_ok not written, nothing is deleted for it)"
  return 0
}
ready_to_bank() { # ready_to_bank TAG PATH : announce the checkpoint, then the banker runs
  echo "$2 $(sha256sum "$2" | cut -d' ' -f1) $(date -u +%FT%TZ)" > $D/bank_ready/$1.txt
  bank_now "$1"
  return 0
}
um_check() {    # um_check pre|post (UM_s1 only): the chain's own checks (um_checks_reps1.py); a miss REFUSES / FAILS the chain
  mkdir -p $D/runs/UM_s1
  if [ -n "${DS_FAKE:-}" ]; then bash $LEAF um $1 > $D/runs/UM_s1/um_$1.out 2>&1 || fail "UM_s1 $1 check refused (fake): $(tail -c 300 $D/runs/UM_s1/um_$1.out | tr '\n' ' ')"; return 0; fi
  DS_ROOT=$D $PY $RP/um_checks_reps1.py $1 > $D/runs/UM_s1/um_$1.out 2>&1 || fail "UM_s1 $1 check refused: $(tail -c 500 $D/runs/UM_s1/um_$1.out | tr '\n' ' ')"
  say "um_checks $1: $(tail -n 1 $D/runs/UM_s1/um_$1.out | cut -c1-200)"
}
stage_a() {     # stage_a ARM : one stage-A run through the runner, then the post-training gates
  arm=$1; out=$D/runs/$arm/A; want=$STEPS_A
  mkdir -p $D/runs/$arm
  if [ -e $out/summary.json ] || [ -e $out/ckpt_last.pt ]; then fail "$arm stage A output exists without ARM_A_DONE ($out): an earlier run finished or failed a gate; the MM decides, the chain does not retrain over it"; fi
  [ $arm = UM_s1 ] && um_check pre
  say "run $arm stage A"
  leaf run $arm A > $D/runs/$arm/A.run.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "$arm stage A rc=$rc: $(tail -c 300 $D/runs/$arm/A.run.log | tr '\n' ' ')"
  $LIB summary $out $want > $out/summary_check.json 2> $out/summary_check.err || fail "$arm stage A: $(cat $out/summary_check.err)"
  $LIB post $out/ckpt_last.pt > $out/post.json 2> $out/post.err || fail "$arm stage A post gate: $(cat $out/post.err)"
  if [ $arm = UM_s1 ]; then
    ready_to_bank $arm.A $out/ckpt_last.pt       # banked BEFORE the K-TACSRC post check (change 5)
    um_check post
    echo "$(sha256sum $out/ckpt_last.pt | cut -d' ' -f1) $(date -u +%FT%TZ)" > $out/ARM_A_DONE
  else
    echo "$(sha256sum $out/ckpt_last.pt | cut -d' ' -f1) $(date -u +%FT%TZ)" > $out/ARM_A_DONE
    ready_to_bank $arm.A $out/ckpt_last.pt
  fi
  say "$arm stage A done: $(cat $out/summary_check.json)"
}
do_arm() {      # do_arm ARM
  arm=$1
  [ -e $D/runs/$arm/ARM_DONE ] && { say "skip $arm (ARM_DONE)"; return 0; }
  if [ ! -e $D/runs/$arm/A/ARM_A_DONE ]; then gate_wait "$arm stage A"; stage_a $arm; fi
  echo "$(date -u +%FT%TZ)" > $D/runs/$arm/ARM_DONE
  echo "ZZ ARM $arm DONE ZZ"
  sweep_all
}
# ---- pre-flight
say "REPS1 start (free $(free_gb) GB, root $D)"
for f in chainlib.py dscale_banker.py blockreps1/chk_arm_reps1.sh blockreps1/blockreps1_prep.sh blockreps1/um_checks_reps1.py blockreps1/study/dscale10k_check.py blockreps1/PAIRING_PROOF.json gate/gate_items.py; do [ -s $D/$f ] || [ -n "${DS_FAKE:-}" ] || fail "missing $D/$f"; done
[ -n "${DS_FAKE:-}" ] || $PY - $D <<'E' || fail "S_U check: 4 x S_U of dry/U_s0.A.dry.json is not $STEPS_A"
import json, sys
assert 4 * json.load(open(sys.argv[1] + "/dry/U_s0.A.dry.json"))["S_U"] == 61568
E
[ -n "${DS_FAKE:-}" ] || $PY - $D <<'E' || fail "the pairing proof does not read all-pass"
import json, sys
p = json.load(open(sys.argv[1] + "/blockreps1/PAIRING_PROOF.json"))
assert p["all_pairs_pass"] and p["all_regressions_refused"], p
E
bank_now
sweep_all
for a in UE_s1 UM_s1; do [ -e $D/runs/$a/ARM_DONE ] || [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || NEEDPREP=1; done
if [ -n "${NEEDPREP:-}" ]; then
  gate_wait "REPS1 prep"
  leaf prep - - > $D/gate/blockreps1_prep.out 2>&1 || fail "REPS1 prep: $(tail -n 1 $D/gate/blockreps1_prep.out | cut -c1-300)"
  tail -n 1 $D/gate/blockreps1_prep.out | grep -q "ZZ REPS1-PREP DONE ZZ" || fail "REPS1 prep did not end DONE: $(tail -n 1 $D/gate/blockreps1_prep.out | cut -c1-300)"
fi
for a in UE_s1 UM_s1; do [ -e $D/runs/$a/ARM_DONE ] || [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || fail "no stage-A gate token for $a ($D/gate/tokens/$a.A/gate_token.json) after the prep"; done
# ---- the arms (stage A only), UE_s1 first
do_arm UE_s1
do_arm UM_s1
sweep_all
bank_now
echo "banking pending (no bank_ok marker): $(for f in $D/bank_ready/*.txt; do t=$(basename $f .txt); [ -e $D/bank_ok/$t.sha256 ] || echo -n "$t "; done)"
echo "ZZ DSCALE10K REPS1 DONE ZZ"
