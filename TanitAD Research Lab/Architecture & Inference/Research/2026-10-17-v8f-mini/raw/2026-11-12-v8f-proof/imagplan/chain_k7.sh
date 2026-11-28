#!/usr/bin/env bash
# chain_k7.sh <base 0|1> : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N5 section 3 (frozen before these runs): the run list of source cand-imagkin
# on the control arm RLFT and on the dose arm RLAm of one base, decel family, suffix x, under the memory guard (pod_imagplan_seq6.sh):
#   <arm>:  IPo IPi IPk IPkid1 IPkid3 IPks IPc IPl      (runs that have their T1_DONE are kept)
#   p3 in the form of note N5 section 2: IPk and IPkn on the first 600 windows (suffix m)
# For base 1 the script first waits until the checkpoint of the arm exists AND its arm is marked done in chain actcons_c1 (it never reads a checkpoint in training).
# No unsafe rate is read by this script. Deletes nothing. Last line: ZZ CHAIN-K7 <base> DONE or STOP.
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof/imagplan
K=${1:?base}
cd "$H" || exit 1
export IP_NEED_GB=20
T=/workspace/v8proof/tree_actcons
run() { # name arm runs sfx rsfx
  bash deploy_k6.sh "$1" actcons_c$K "$2" $T "$3" "$4" "$5" > "deploy_k_$1.out" 2>&1
  tail -n 1 "deploy_k_$1.out" | cut -c1-200
  grep -aq "ZZ SEQ STOP .*memory guard" "deploy_k_$1.out" && { echo "ZZ CHAIN-K7 $K STOP after a memory-guard stop in $1"; exit 4; }
  return 0
}
waitarm() { # arm : ARM DONE <arm> in the chain log of actcons_c<K>
  local n=0 s arm=$1 c1 c2
  while :; do
    s=$(timeout 60 bash "$P" "grep -ac 'ARM DONE $arm' /workspace/v8proof/chains/actcons_c$K/chain.log 2>/dev/null; ls /workspace/v8proof/chains/actcons_c$K/actcons_c${K}_STOP.txt 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r' | tr '\n' ' ')
    read -r c1 c2 _ <<< "$s"
    [ "${c1:-0}" -ge 1 ] 2>/dev/null && return 0
    [ "${c2:-0}" -ge 1 ] 2>/dev/null && { echo "chain actcons_c$K stopped"; return 1; }
    n=$((n+1)); [ $n -gt 480 ] && { echo "arm not done after 8 h"; return 1; }
    sleep 60
  done
}
for a in RLFT RLAm; do
  arm=${a}_s$K; tg=$(echo "$a" | tr 'A-Z' 'a-z')
  waitarm "$arm" || { echo "ZZ CHAIN-K7 $K STOP $arm is not done"; exit 3; }
  run "${tg}${K}k2" "$arm" "IPo IPi IPk IPkid1 IPkid3 IPks IPc IPl" x x
  IP_MAX_WINDOWS=600 IP_NEED_NOISE_GB=20 run "${tg}${K}m1" "$arm" "IPk IPkn" m m
done
echo "ZZ CHAIN-K7 $K DONE $(date -u +%H:%M:%SZ)"
