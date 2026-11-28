#!/bin/bash
# block2.sh -- PREREG_V8F_DSCALE10K block 2 (pod, LoRA pod): the second seed, B_s1 then U_s1, each stage A -> DAgger bank (4 shards, one chain step, then merge) -> stage B (card 2.3 rows 5 and 6, section 5).
# Built from block1.sh: the same gate_wait (STOP file, >= 60 GB free on /root, idle GPU < 3,500 MiB), the same runner (`chk_arm.sh run` -> `dscale10k_check.py run`, which REFUSES rc 2 without a token that binds the exact
# commit, argv, data files, tree, trainer, checker and merge tool), the same per-stage summary and post-training gates, the same in-chain stage-B mint. DIFFERENCES from block1.sh:
#   * pre-flight runs block2_prep.sh (CPU dry records, init record, the two reach probes under this chain's lock, evidence, the stage-A tokens of B_s1 and U_s1) when a token is missing: no GPU job of the
#     card's gate is needed beyond the two reach probes, and no token is hand-written.
#   * after every stage the banker (dscale_banker.py) runs at once (a failure is logged and the chain goes on: its deletions stay deferred); at start it banks anything still pending from block 1.
#   * the sweep covers every arm of blocks 1 and 2 that has ARM_DONE and a bank_ok marker equal to its stage-B checkpoint (chainlib.py sweep), and the R release once R_s0.A is banked.
# Ends with `ZZ DSCALE10K BLOCK2 DONE ZZ` or `ZZ DSCALE10K BLOCK2 FAIL ZZ <why>`. A file /root/dscale10k/chains/STOP ends it between steps.
#   launch (the MM): nohup setsid flock /root/gpu_queue.lock bash /root/dscale10k/block2.sh < /dev/null > /root/dscale10k/block2.log 2>&1 &
# Dry run: DS_FAKE=1 DS_ROOT=<fake root> (dryrun_block2.sh): leaf operations are replaced, the control flow is this file.
set -u
D=${DS_ROOT:-/root/dscale10k}; PY=${DS_PY:-/root/venv/bin/python}; LIB="$PY $D/chainlib.py"; export DS_ROOT=$D
MINFREE=${DS_MIN_FREE_GB:-60}
TREE=$D/tree
STEPS_A=19164; STEPS_B=3000
mkdir -p $D/chains $D/runs $D/gate/tokens $D/bank_ready $D/bank_ok
say() { echo "$(date -u +%T) $*"; }
fail() { echo "ZZ DSCALE10K BLOCK2 FAIL ZZ $1"; exit 1; }
[ -n "${DS_FAKE:-}" ] && LEAF=$D/fake_leaf.sh

gpu_idle() { if [ -n "${DS_FAKE:-}" ]; then return 0; fi; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
free_gb() { if [ -n "${DS_FAKE:-}" ]; then echo "${DS_FREE_GB:-500}"; else df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; fi; }
sweep_all() { for a in B_s0 U_s0 B_rep B_s1 U_s1; do [ -e $D/runs/$a/ARM_DONE ] && $LIB sweep $a; done; [ -e $D/runs/R_s0/ARM_DONE ] && $LIB sweep_r; return 0; }
gate_wait() {   # gate_wait WHAT : STOP file, free disk (a sweep of banked data first), idle GPU
  [ -e $D/chains/STOP ] && fail "STOP file before $1"
  f=$(free_gb)
  if [ "${f:-0}" -lt "$MINFREE" ]; then sweep_all; f=$(free_gb); fi
  [ "${f:-0}" -ge "$MINFREE" ] || fail "free disk ${f} GB < ${MINFREE} GB before $1 (nothing banked to delete)"
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt ${DS_GPU_TRIES:-360} ] && fail "GPU not idle (>= 3500 MiB used) before $1"; sleep ${DS_POLL:-10}; done
}
leaf() {        # leaf KIND ARM STAGE : the operations a dry run replaces
  case $1 in
    run)       if [ -n "${DS_FAKE:-}" ]; then bash $LEAF run $2 $3; else DS_ROOT=$D bash $D/chk_arm.sh run $2 $3; fi;;
    mint)      if [ -n "${DS_FAKE:-}" ]; then bash $LEAF mint $2 $3; else DS_ROOT=$D bash $D/chk_arm.sh mint $2 $3; fi;;
    bank)      if [ -n "${DS_FAKE:-}" ]; then bash $LEAF bank $2 $3; else bash $D/bank4.sh "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}"; fi;;
    prep)      if [ -n "${DS_FAKE:-}" ]; then bash $LEAF prep $2 $3; else DS_ROOT=$D bash $D/block2_prep.sh; fi;;
    selfcheck) if [ -n "${DS_FAKE:-}" ]; then bash $LEAF selfcheck $2 $3; else $PY $D/gate/sb_selfcheck.py $D/runs/$2/argv_B.json $D/runs/$2/B_selfcheck $D/runs/$2/B_selfcheck.result.json; fi;;
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
stage() {       # stage ARM A|B : run one stage through the runner, then the post-training gate
  arm=$1; st=$2; out=$D/runs/$arm/$st; want=$STEPS_A; [ $st = B ] && want=$STEPS_B
  mkdir -p $D/runs/$arm
  say "run $arm stage $st"
  leaf run $arm $st > $D/runs/$arm/$st.run.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "$arm stage $st rc=$rc: $(tail -c 300 $D/runs/$arm/$st.run.log | tr '\n' ' ')"
  $LIB summary $out $want > $out/summary_check.json 2> $out/summary_check.err || fail "$arm stage $st: $(cat $out/summary_check.err)"
  if [ $st = B ]; then $LIB post $out/ckpt_last.pt $D/runs/$arm/A/ckpt_last.pt > $out/post.json 2> $out/post.err || fail "$arm stage B post gate: $(cat $out/post.err)"
  else $LIB post $out/ckpt_last.pt > $out/post.json 2> $out/post.err || fail "$arm stage A post gate: $(cat $out/post.err)"; fi
  echo "$(sha256sum $out/ckpt_last.pt | cut -d' ' -f1) $(date -u +%FT%TZ)" > $out/ARM_${st}_DONE
  ready_to_bank $arm.$st $out/ckpt_last.pt
  say "$arm stage $st done: $(cat $out/summary_check.json)"
}
bank_and_mint_b() {   # bank_and_mint_b ARM : DAgger bank over the arm's own TRAIN windows from its own stage-A checkpoint, stage-B argv, selfcheck, mint
  arm=$1; r=$D/runs/$arm
  if [ ! -e $r/bank/bank.json ]; then
    gate_wait "$arm bank"
    $LIB m2bargv $D/argv/$arm.A.json $r/bank_m2b_argv.json || fail "$arm m2b argv"
    ba=$($LIB bankargs $D/dry/$arm.A.dry.json) || fail "$arm bank args"
    ec=${ba% *}; ew=${ba#* }
    A=$($PY - "$D/argv/$arm.A.json" <<'E'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
g = lambda f: t[t.index(f) + 1]
print(g("--cache-dir"), g("--tracks"), g("--join-train"))
E
) || fail "$arm bank paths"
    cache=${A%% *}; rest=${A#* }; tracks=${rest%% *}; join=${rest#* }
    say "bank $arm: n_shards 4 parallel, $ec clips, $ew windows"
    leaf bank $arm B $r/A/ckpt_last.pt $r/bank_m2b_argv.json $cache $tracks $join $r/bank $ec $ew > $r/bank.log 2>&1 || fail "$arm bank: $(tail -c 300 $r/bank.log | tr '\n' ' ')"
    tail -n 2 $r/bank.log
  fi
  $LIB subst $D/argv/$arm.B.tmpl.json $r/argv_B.json $r/A/ckpt_last.pt $r/bank || fail "$arm stage-B argv"
  if [ ! -e $r/B_selfcheck/selfcheck.json ]; then
    gate_wait "$arm stage-B selfcheck"
    leaf selfcheck $arm B > $r/B_selfcheck.log 2>&1 || fail "$arm stage-B selfcheck: $(tail -c 300 $r/B_selfcheck.log | tr '\n' ' ')"
  fi
  [ -s $r/B_selfcheck/selfcheck.json ] || fail "$arm stage-B selfcheck wrote no selfcheck.json"
  if [ ! -e $D/gate/tokens/$arm.B/gate_token.json ]; then
    say "mint $arm stage B"
    leaf mint $arm B > $r/mint_B.log 2>&1 || fail "$arm stage-B mint refused: $(tail -c 400 $r/mint_B.log | tr '\n' ' ')"
  fi
}
do_arm() {      # do_arm ARM WITH_STAGE_B
  arm=$1; withb=$2
  [ -e $D/runs/$arm/ARM_DONE ] && { say "skip $arm (ARM_DONE)"; return 0; }
  if [ ! -e $D/runs/$arm/A/ARM_A_DONE ]; then gate_wait "$arm stage A"; stage $arm A; fi
  if [ $withb = yes ]; then
    if [ ! -e $D/runs/$arm/B/ARM_B_DONE ]; then
      bank_and_mint_b $arm
      gate_wait "$arm stage B"
      stage $arm B
    fi
  fi
  echo "$(date -u +%FT%TZ)" > $D/runs/$arm/ARM_DONE
  echo "ZZ ARM $arm DONE ZZ"
  sweep_all
}
# ---- pre-flight
say "block 2 start (free $(free_gb) GB, root $D)"
for f in chainlib.py bank4.sh chk_arm.sh dscale_banker.py block2_prep.sh; do [ -s $D/$f ] || [ -n "${DS_FAKE:-}" ] || fail "missing $D/$f"; done
bank_now
sweep_all                                   # block 1's banked bank shards / selfcheck scratch go now (chainlib sweep: only what bank_ok covers; logged)
for a in B_s1 U_s1; do
  [ -e $D/runs/$a/ARM_DONE ] && continue
  [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || NEEDPREP=1
done
if [ -n "${NEEDPREP:-}" ]; then
  gate_wait "block 2 prep"
  leaf prep - - > $D/gate/block2_prep.out 2>&1 || fail "block 2 prep: $(tail -n 1 $D/gate/block2_prep.out | cut -c1-300)"
  tail -n 1 $D/gate/block2_prep.out | grep -q "ZZ BLOCK2-PREP DONE ZZ" || fail "block 2 prep did not end DONE: $(tail -n 1 $D/gate/block2_prep.out | cut -c1-300)"
fi
for a in B_s1 U_s1; do
  [ -e $D/runs/$a/ARM_DONE ] && continue
  [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || fail "no stage-A gate token for $a ($D/gate/tokens/$a.A/gate_token.json) after the prep"
done
# ---- the arms
do_arm B_s1 yes
do_arm U_s1 yes
sweep_all
bank_now
echo "banking pending (no bank_ok marker): $(for f in $D/bank_ready/*.txt; do t=$(basename $f .txt); [ -e $D/bank_ok/$t.sha256 ] || echo -n "$t "; done)"
echo "ZZ DSCALE10K BLOCK2 DONE ZZ"
