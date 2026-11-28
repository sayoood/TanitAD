#!/usr/bin/env bash
# chain_k8.sh : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N6 (frozen before any of these checkpoints existed): tool runs on the dose arms at bases
# 0 and 1 (chain actcons_f: IPo IPi, for control p5b) and the run list of source cand-imagkin on RLFT and RLAm at the new bases 2 and 3 (chains actcons_e2, actcons_e3),
# decel family, suffix x, then p3 in the subset form (suffix m), all under the memory guard of pod_imagplan_seq6.sh. A run starts only with fewer than three GPU jobs
# (the run script refuses and the sequence waits). The pilot of card ADD41 goes first: this script waits until the pilot has begun its training (its train_meta.json or
# a TRAIN_STARTED.txt exists), or is done or stopped, or 8 h have passed. It never reads a checkpoint in training (it waits for ARM DONE in the chain log).
# No unsafe rate is read by this script. Deletes nothing. Last line: ZZ CHAIN-K8 DONE or ZZ CHAIN-K8 STOP ...
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof/imagplan
cd "$H" || exit 1
export IP_NEED_GB=20
T=/workspace/v8proof/tree_actcons
ts() { date -u +%H:%M:%SZ; }
pod() { timeout 60 bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
run() { # name queue arm runs sfx rsfx
  bash deploy_k6.sh "$1" "$2" "$3" $T "$4" "$5" "$6" > "deploy_k_$1.out" 2>&1
  echo "[$(ts)] $1: $(tail -n 1 "deploy_k_$1.out" | cut -c1-160)"
  grep -aq "ZZ SEQ STOP .*memory guard" "deploy_k_$1.out" && { echo "ZZ CHAIN-K8 STOP after a memory-guard stop in $1"; exit 4; }
  return 0
}
waitarm() { # queue arm : 0 when ARM DONE <arm> is in the chain log of <queue>; 1 if the chain stopped, was never started, or the bound passed
  local n=0 s q=$1 arm=$2 st dir ns dn
  while :; do
    s=$(pod "cd /workspace/v8proof; ls chains/$q/${q}_STOP.txt 2>/dev/null | wc -l; ls -d chains/$q 2>/dev/null | wc -l; ls runs/actcons_z/ACTCONS_G_NOT_STARTED.txt 2>/dev/null | wc -l; grep -ac 'ARM DONE $arm' chains/$q/chain.log 2>/dev/null || true" | tr '\n' ' ')
    read -r st dir ns dn _ <<< "$s"
    [ "${dn:-0}" -ge 1 ] 2>/dev/null && return 0
    [ "${st:-0}" -ge 1 ] 2>/dev/null && { echo "[$(ts)] chain $q stopped before $arm was done"; return 1; }
    [ "${ns:-0}" -ge 1 ] 2>/dev/null && [ "${dir:-1}" = 0 ] && { echo "[$(ts)] chain $q was not started (the launcher refused)"; return 1; }
    n=$((n+1)); [ $n -gt 720 ] && { echo "[$(ts)] $arm not done after 12 h"; return 1; }
    sleep 60
  done
}
# the pilot first
n=0
while :; do
  d=$(pod "cd /workspace/v8proof/runs/finegrid 2>/dev/null && ls FINEGRID_DONE.txt FINEGRID_STOP.txt TRAIN_STARTED.txt train/train_meta.json 2>/dev/null | wc -l" | tr -d ' \n')
  [ "${d:-0}" -ge 1 ] 2>/dev/null && { echo "[$(ts)] the pilot has begun or ended ($d marker files)"; break; }
  n=$((n+1)); [ $n -gt 480 ] && { echo "[$(ts)] no pilot marker after 8 h, going on"; break; }
  sleep 60
done
for arm in RLA3_s0 RLA2_s0 RLA3_s1 RLA2_s1; do
  tg=$(echo "$arm" | tr 'A-Z' 'a-z' | tr -d '_')
  if waitarm actcons_f "$arm"; then run "${tg}k2" actcons_f "$arm" "IPo IPi" x x; else echo "[$(ts)] $arm left out"; fi
done
for K in 2 3; do
  for a in RLFT RLAm; do
    arm=${a}_s$K; tg=$(echo "$a" | tr 'A-Z' 'a-z')
    if waitarm actcons_e$K "$arm"; then
      run "${tg}${K}k2" actcons_e$K "$arm" "IPo IPi IPk IPkid1 IPkid3 IPks IPc IPl" x x
      IP_MAX_WINDOWS=600 IP_NEED_NOISE_GB=20 run "${tg}${K}m1" actcons_e$K "$arm" "IPk IPkn" m m
      timeout 240 bash "$P" "bash -s -- $arm" < pod_cmp_subset.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "cmp_subset_$arm.out"; cut -c1-200 "cmp_subset_$arm.out"
    else echo "[$(ts)] $arm left out"; fi
  done
done
echo "ZZ CHAIN-K8 DONE $(ts)"
