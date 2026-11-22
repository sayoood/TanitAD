#!/bin/bash
# blockreps2.sh -- DSCALE10K REPS2 (pod, LoRA pod): the seed-2 and seed-3 replicates UE_s2 then UE_s3 of UE_s0 (PI 2026-11-22 "Yes, two more seeds"), STAGE A ONLY, 4 x S_U = 61,568 steps each, union set, code of record cdbff971.
# NOT launched by the builder: the MM launches it (launch command in HANDOFF_REPS2.md). FORK of blockreps1.sh (UE arm only). Changes against the fork (ALL of them):
#   1 THE GPU PROTOCOL of /root/rollhist/rh_rl_chain.sh: the chain takes fd 9 on /root/gpu_queue.lock and holds it to the end (children inherit it, so the lock outlives only a living trainer); it logs the line
#     "GPU queue lock held to the end of the chain" (reps2_marker_watch.sh keys on it, as rh_rl_marker_watch.sh does) and never creates, removes or edits /root/N1DW_READY itself, EXCEPT on its own refusal/exit when
#     the watcher marked it (chains/reps2_marker_mine): then it clears the marker at once so the other session is not held by a dead chain.  The lock wait is capped (DS_LOCK_WAIT_S, default 30 h).
#   2 all cheap refusals (missing files, pairing proof, STOP file, argv or token drift) happen BEFORE the lock is taken, and again before each arm, so a refusal never holds the GPU.
#   3 RESUMABLE: an arm with ARM_DONE is skipped; an arm with ARM_A_DONE only finishes banking; a stage-A directory with summary.json + ckpt_last.pt and no ARM_A_DONE (a finished run that stopped at a gate) goes
#     straight to the post gates with NO training; a stage-A directory with ckpt_last.pt and NO summary.json (a killed run) is RESUMED by the trainer from ckpt_last.pt (v8f_mini_train.py:3499 resume if the file exists and not --no-resume,
#     :3501 refuses a checkpoint of a different argv, :3661 saves every --ckpt-every 500 steps through ckpt.tmp + os.replace :3537-3538), through the SAME token-checking runner, output APPENDED to A.run.log. (reps1 refused such a directory.)
#   4 no UM arm, no um checks, no sweep (nothing is deleted by this chain), no queue wrapper: the lock is taken here.  sentinels `ZZ DSCALE10K REPS2 ...`.
# Per arm: pair check (gen_argv_reps2.py re-derives the argv from UE_s0.A and refuses a third change) -> token verify (verify_token_reps2.py) -> runner (token-refusing; the trainer's own selfcheck runs at its start) ->
#   selfcheck.json must read declared_vs_built PASS -> summary check -> post gate (every registered parameter has Adam state: the zero-grad gate, chainlib post) -> ARM_A_DONE -> bank (HF private repo, sha256 read back) -> ARM_DONE.
# Kept: the banker (dscale_banker.py, arm-agnostic), the disk guard (>= 60 GB free by default: DS_MIN_FREE_GB), the idle-GPU wait, the STOP file chains/STOP read between steps.
# Ends with `ZZ DSCALE10K REPS2 DONE ZZ` or `ZZ DSCALE10K REPS2 FAIL ZZ <why>` (last line of the log).
# Dry run: DS_FAKE=1 DS_ROOT=<fake root> DS_LOCK=<fake lock> DS_MARKER=<fake marker> (dryrun_reps2.sh).
D=${DS_ROOT:-/root/dscale10k}
cd $D || { echo "ZZ DSCALE10K REPS2 FAIL ZZ cd $D"; exit 1; }
set -u
PY=${DS_PY:-/root/venv/bin/python}; LIB="$PY $D/chainlib.py"; export DS_ROOT=$D
LOG=${DS_LOG:-$D/blockreps2.log}; LOCKF=${DS_LOCK:-/root/gpu_queue.lock}; MARKER=${DS_MARKER:-/root/N1DW_READY}
MINFREE=${DS_MIN_FREE_GB:-60}; STEPS_A=61568; RP=${DS_RP:-$D/blockreps2}; MINE=$D/chains/reps2_marker_mine
mkdir -p $D/chains $D/runs $D/gate/tokens $D/bank_ready $D/bank_ok
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
clear_marker() { if [ -e $MINE ]; then rm -f $MARKER $MINE; say "marker $MARKER cleared (it was set for this chain)"; fi; }
fail() { clear_marker; echo "ZZ DSCALE10K REPS2 FAIL ZZ $1"; exit 1; }
[ -n "${DS_FAKE:-}" ] && LEAF=$D/fake_leaf.sh

gpu_idle() { if [ -n "${DS_FAKE:-}" ]; then return 0; fi; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
free_gb() { if [ -n "${DS_FAKE:-}" ]; then echo "${DS_FREE_GB:-500}"; else df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; fi; }
stop_check() { [ -e $D/chains/STOP ] && fail "STOP file before $1"; return 0; }
disk_check() { f=$(free_gb); [ "${f:-0}" -ge "$MINFREE" ] || fail "free disk ${f} GB < ${MINFREE} GB before $1"; return 0; }
gpu_wait() { n=0; until gpu_idle; do n=$((n+1)); [ $n -gt ${DS_GPU_TRIES:-360} ] && fail "GPU not idle (>= 3500 MiB used) before $1 although the lock is held"; sleep ${DS_POLL:-10}; done; }
leaf() {        # leaf KIND ARM : the operations a dry run replaces (run, verify, pair, prep)
  case $1 in
    run)    if [ -n "${DS_FAKE:-}" ]; then bash $LEAF run $2 A; else DS_ROOT=$D bash $RP/chk_arm_reps2.sh run $2 A; fi;;
    verify) if [ -n "${DS_FAKE:-}" ]; then bash $LEAF verify $2 A; else DS_ROOT=$D PYTHONPATH= $PY $RP/verify_token_reps2.py $2 A > $D/gate/verify_$2.A.out 2>&1; fi;;
    pair)   if [ -n "${DS_FAKE:-}" ]; then bash $LEAF pair $2 A; else DS_ROOT=$D $PY $RP/gen_argv_reps2.py > $D/gate/gen_argv_reps2.recheck.out 2>&1; fi;;
    prep)   if [ -n "${DS_FAKE:-}" ]; then bash $LEAF prep - -; else DS_ROOT=$D bash $RP/blockreps2_prep.sh; fi;;
  esac
}
bank_now() {    # bank_now [TAG] : the banker (the only writer of bank_ok), without the lock fd; a failure defers and never stops the chain
  if [ -n "${DS_FAKE:-}" ]; then $PY $D/dscale_banker.py --root $D --local-hub $D/fakehub ${1:+--only $1} >> $D/banker.log 2>&1 9>&- || say "BANK-DEFERRED ${1:-all}: see $D/banker.log"; return 0; fi
  $PY $D/dscale_banker.py --root $D ${1:+--only $1} >> $D/banker.log 2>&1 9>&- || say "BANK-DEFERRED ${1:-all}: see $D/banker.log (bank_ok not written)"
  return 0
}
ready_to_bank() { echo "$2 $(sha256sum "$2" | cut -d' ' -f1) $(date -u +%FT%TZ)" > $D/bank_ready/$1.txt; bank_now "$1"; return 0; }
ckpt_step() { $PY -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('step'))" "$1" 2>/dev/null; }
arm_preflight() {   # arm_preflight ARM : the refusals that need no GPU (pair, token)
  leaf pair $1 || fail "$1 pair check refused (the argv is not UE_s0's with --seed and --out only): $(tail -c 400 $D/gate/gen_argv_reps2.recheck.out 2>/dev/null | tr '\n' ' ')"
  [ -s $D/gate/tokens/$1.A/gate_token.json ] || fail "no stage-A gate token for $1 ($D/gate/tokens/$1.A/gate_token.json)"
  leaf verify $1 || fail "$1 token verify refused: $(tail -c 400 $D/gate/verify_$1.A.out 2>/dev/null | tr '\n' ' ')"
}
stage_a() {     # stage_a ARM : one stage-A run (fresh, resumed, or gates only) through the runner, then the post gates
  arm=$1; out=$D/runs/$arm/A; want=$STEPS_A
  mkdir -p $D/runs/$arm $out
  if [ -s $out/summary.json ] && [ -s $out/ckpt_last.pt ]; then
    say "$arm: finished run found ($out/summary.json + ckpt_last.pt, no ARM_A_DONE): post gates only, NO training"
  else
    if [ -s $out/ckpt_last.pt ]; then say "$arm: RESUME from $out/ckpt_last.pt (step $(ckpt_step $out/ckpt_last.pt)); the trainer resumes by itself, same argv, same token"; fi
    stop_check "$arm stage A"; disk_check "$arm stage A"; gpu_wait "$arm stage A"
    say "run $arm stage A"
    leaf run $arm >> $D/runs/$arm/A.run.log 2>&1; rc=$?
    [ $rc -eq 0 ] || fail "$arm stage A rc=$rc: $(tail -c 300 $D/runs/$arm/A.run.log | tr '\n' ' ')"
  fi
  $PY - $out/selfcheck.json <<'E' || fail "$arm: selfcheck.json does not read declared_vs_built PASS ($out/selfcheck.json)"
import json, sys
sc = json.load(open(sys.argv[1]))
assert sc.get("declared_vs_built") == "PASS", sc.get("declared_vs_built")
E
  $LIB summary $out $want > $out/summary_check.json 2> $out/summary_check.err || fail "$arm stage A: $(cat $out/summary_check.err)"
  $LIB post $out/ckpt_last.pt > $out/post.json 2> $out/post.err || fail "$arm stage A post (zero-grad) gate: $(cat $out/post.err)"
  echo "$(sha256sum $out/ckpt_last.pt | cut -d' ' -f1) $(date -u +%FT%TZ)" > $out/ARM_A_DONE
  say "$arm stage A done: $(cat $out/summary_check.json)"
}
do_arm() {      # do_arm ARM
  arm=$1; out=$D/runs/$arm/A
  [ -e $D/runs/$arm/ARM_DONE ] && { say "skip $arm (ARM_DONE)"; return 0; }
  stop_check "$arm"
  if [ ! -e $out/ARM_A_DONE ]; then arm_preflight $arm; stage_a $arm; fi
  ready_to_bank $arm.A $out/ckpt_last.pt
  echo "$(date -u +%FT%TZ)" > $D/runs/$arm/ARM_DONE
  echo "ZZ ARM $arm DONE ZZ"
}
# ---- pre-flight, BEFORE the lock: nothing here needs the GPU, so a refusal here never held it
say "REPS2 start (free $(free_gb) GB, root $D, lock $LOCKF)"
stop_check "the lock"
for f in chainlib.py dscale_banker.py blockreps2/chk_arm_reps2.sh blockreps2/blockreps2_prep.sh blockreps2/gen_argv_reps2.py blockreps2/verify_token_reps2.py blockreps2/study/dscale10k_check.py blockreps2/PAIRING_PROOF.json argv/UE_s0.A.json argv/UE_s2.A.json argv/UE_s3.A.json; do
  [ -s $D/$f ] || [ -n "${DS_FAKE:-}" ] || fail "missing $D/$f (run install_reps2.sh first)"
done
[ -n "${DS_FAKE:-}" ] || $PY - $D <<'E' || fail "S_U check: 4 x S_U of dry/U_s0.A.dry.json is not $STEPS_A"
import json, sys
assert 4 * json.load(open(sys.argv[1] + "/dry/U_s0.A.dry.json"))["S_U"] == 61568
E
[ -n "${DS_FAKE:-}" ] || $PY - $D <<'E' || fail "the pairing proof does not read all-pass"
import json, sys
p = json.load(open(sys.argv[1] + "/blockreps2/PAIRING_PROOF.json"))
assert p["all_pairs_pass"] and p["all_regressions_refused"], p
E
disk_check "the lock"
# ---- the GPU queue lock, held to the end (the protocol of rh_rl_chain.sh); the marker watcher clears the other session's priority marker when it reads the line below
exec 9>$LOCKF
say "waiting for the GPU queue lock"
flock -w ${DS_LOCK_WAIT_S:-108000} 9 || fail "flock: no lock within ${DS_LOCK_WAIT_S:-108000} s"
say "GPU queue lock held to the end of the chain"
stop_check "prep"
# ---- the tokens (prep under the lock: the reach probes use the GPU for about 15 s each; the dry builds and mints are CPU)
need=""
for a in UE_s2 UE_s3; do [ -e $D/runs/$a/ARM_DONE ] || [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || need=1; done
if [ -n "$need" ]; then
  disk_check "REPS2 prep"; gpu_wait "REPS2 prep"
  leaf prep > $D/gate/blockreps2_prep.out 2>&1 || fail "REPS2 prep: $(tail -n 1 $D/gate/blockreps2_prep.out | cut -c1-300)"
  tail -n 1 $D/gate/blockreps2_prep.out | grep -q "ZZ REPS2-PREP DONE ZZ" || fail "REPS2 prep did not end DONE: $(tail -n 1 $D/gate/blockreps2_prep.out | cut -c1-300)"
fi
for a in UE_s2 UE_s3; do [ -e $D/runs/$a/ARM_DONE ] || [ -e $D/runs/$a/A/ARM_A_DONE ] || [ -s $D/gate/tokens/$a.A/gate_token.json ] || fail "no stage-A gate token for $a after the prep"; done
bank_now
# ---- the arms (stage A only), UE_s2 first
do_arm UE_s2
do_arm UE_s3
bank_now
echo "banking pending (no bank_ok marker): $(for f in $D/bank_ready/UE_s2.A.txt $D/bank_ready/UE_s3.A.txt; do t=$(basename $f .txt); [ -e $D/bank_ok/$t.sha256 ] || echo -n "$t "; done)"
clear_marker
echo "ZZ DSCALE10K REPS2 DONE ZZ"
