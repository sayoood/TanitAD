#!/bin/bash
# block1.sh -- PREREG_V8F_DSCALE10K block 1 (pod, LoRA pod). WRITTEN, NOT LAUNCHED: the Master Mind launches it.
#   usage: nohup setsid bash /root/dscale10k/block1.sh < /dev/null > /root/dscale10k/block1.log 2>&1 &
# Arms, one at a time, in the card's order (section 2.3): B_s0, U_s0, B_rep  = stage A -> DAgger bank (4 shards, the 4 shard processes at the same time, one chain step, then merge) -> stage B;
#                                                         R_s0              = stage A only, last.
# Every GPU step waits for an idle GPU (< 3,500 MiB), needs >= 60 GB free on /root (the card's guard), and runs through `dscale10k_check.py run`, which REFUSES (rc 2) without a gate token that
# binds the exact commit, argv sha256, data files, tree, trainer, checker and merge tool. Stage-A tokens are minted before launch (chk_arm.sh mint). A stage-B token binds the arm's OWN stage-A
# checkpoint and bank, which do not exist before the run, so the chain mints it right after the bank is built (chk_arm.sh mint ARM B -> C-BANK + G-DAGGER on the REAL bank and the trainer's own
# --selfcheck-only of the real stage-B argv), and the checker mints it only when every required check reads PASS.
# Per arm after stage A and after stage B: summary.json says done at the right step, ckpt_last.pt is at that step, and the post-training gate (every registered parameter has Adam state, the count
# equals B_s0's 252) passes. Deletions (chainlib.py sweep / sweep_r) happen ONLY for what is banked (bank_ok/<ARM>.<A|B>.sha256 equal to the checkpoint's sha256, written by the banker, never by this
# chain) and are logged (path, size) in /root/dscale10k/deletions_block1.log.
# Ends with `ZZ DSCALE10K BLOCK1 DONE ZZ` (all four arms ran) or `ZZ DSCALE10K BLOCK1 FAIL ZZ <why>`. A file /root/dscale10k/chains/STOP ends it between steps.
# Dry run: DS_FAKE=1 DS_ROOT=<fake root> (see /root/dscale10k/gate/dryrun_block1.sh): leaf operations (trainer, bank, mint, selfcheck, GPU / disk probes) are replaced, the control flow is this file.
set -u
D=${DS_ROOT:-/root/dscale10k}; PY=${DS_PY:-/root/venv/bin/python}; LIB="$PY $D/chainlib.py"; export DS_ROOT=$D
MINFREE=${DS_MIN_FREE_GB:-60}
TREE=$D/tree
STEPS_A=19164; STEPS_B=3000
mkdir -p $D/chains $D/runs $D/gate/tokens $D/bank_ready $D/bank_ok
say() { echo "$(date -u +%T) $*"; }
fail() { echo "ZZ DSCALE10K BLOCK1 FAIL ZZ $1"; exit 1; }
[ -n "${DS_FAKE:-}" ] && LEAF=$D/fake_leaf.sh

gpu_idle() { if [ -n "${DS_FAKE:-}" ]; then return 0; fi; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
free_gb() { if [ -n "${DS_FAKE:-}" ]; then echo "${DS_FREE_GB:-500}"; else df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; fi; }
sweep_all() { for a in B_s0 U_s0 B_rep; do [ -e $D/runs/$a/ARM_DONE ] && $LIB sweep $a; done; [ -e $D/runs/R_s0/ARM_DONE ] && $LIB sweep_r; return 0; }
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
    selfcheck) if [ -n "${DS_FAKE:-}" ]; then bash $LEAF selfcheck $2 $3; else $PY $D/gate/sb_selfcheck.py $D/runs/$2/argv_B.json $D/runs/$2/B_selfcheck $D/runs/$2/B_selfcheck.result.json; fi;;
  esac
}
ready_to_bank() { # ready_to_bank TAG PATH : tell the banker (never banks itself)
  echo "$2 $(sha256sum "$2" | cut -d' ' -f1) $(date -u +%FT%TZ)" > $D/bank_ready/$1.txt
  [ -x $D/bank_hook.sh ] && $D/bank_hook.sh "$1" "$2"
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
say "block 1 start (free $(free_gb) GB, root $D)"
for a in B_s0 U_s0 B_rep; do
  [ -e $D/runs/$a/ARM_DONE ] && continue
  [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || fail "no stage-A gate token for $a ($D/gate/tokens/$a.A/gate_token.json): mint it first (chk_arm.sh mint $a A)"
done
for f in chainlib.py bank4.sh chk_arm.sh; do [ -s $D/$f ] || [ -n "${DS_FAKE:-}" ] || fail "missing $D/$f"; done
# ---- the arms
do_arm B_s0 yes
do_arm U_s0 yes
do_arm B_rep yes
if [ -e $D/runs/R_s0/ARM_DONE ] || [ -s $D/gate/tokens/R_s0.A/gate_token.json ] || [ -e $D/runs/R_s0/A/ARM_A_DONE ]; then
  do_arm R_s0 no
else
  pend=$(ls $D/bank_ready 2>/dev/null | tr '\n' ' ')
  fail "R_s0 not run: no stage-A gate token (the MM rules on R_COLUMNS.md first, then mint with chk_arm.sh mint R_s0 A and relaunch: finished arms are skipped). B_s0, U_s0, B_rep ran. Banking pending for: $pend"
fi
sweep_all
echo "banking pending (no bank_ok marker): $(for f in $D/bank_ready/*.txt; do t=$(basename $f .txt); [ -e $D/bank_ok/$t.sha256 ] || echo -n "$t "; done)"
echo "ZZ DSCALE10K BLOCK1 DONE ZZ"
