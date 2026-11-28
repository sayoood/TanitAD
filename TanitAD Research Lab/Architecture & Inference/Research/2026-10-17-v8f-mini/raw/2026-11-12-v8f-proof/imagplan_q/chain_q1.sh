#!/usr/bin/env bash
# chain_q1.sh : dev-box side, script polling. Note PREREG_V8F_PROOF_ADD40_N7 (source cand-cvq: cand-cv with the gap min(g_head, q_head)). Written by the build agent from chain_k9.sh; nothing of it was run.
# First pair LS_s0, LS_s1 (queue leadsup_a), then the second pair LSD_s0, LSD_s1 (queue leadsup_b; run whatever the first pair reads). Per checkpoint the runs "IPo IPc IPl IPcq IPcqid IPcq0", decel family,
# suffix q (run names IPoq_<arm>, IPcq_<arm>, IPlq_<arm>, IPcqq_<arm> = cand-cvq, IPcqidq_<arm> = cand-cvq with the constant 1e9, IPcq0q_<arm> = with the constant 0), tree tree_leadsup.
# The head of each checkpoint is chosen by the sequence script (LS_*: qhead_dose_ls/heads, LSD_*: qhead_dose/heads). One sequence at a time, under the memory guard and the three-job limit of
# pod_imagplan_seq_q.sh / pod_imagplan_run_q.sh. No unsafe rate is read by this script. Deletes nothing. IP_MAX_WINDOWS must NOT be set.
# Last line: ZZ CHAIN-Q1 DONE <time> or ZZ CHAIN-Q1 STOP ...
H=/f/TanitAD-scratch/v8proof/imagplan_q
cd "$H" || exit 1
[ -z "${IP_MAX_WINDOWS:-}" ] || { echo "ZZ CHAIN-Q1 STOP IP_MAX_WINDOWS is set"; exit 5; }
export IP_NEED_GB=20
T=/workspace/v8proof/tree_leadsup
ts() { date -u +%H:%M:%SZ; }
run() { # name queue arm
  bash deploy_q1.sh "$1" "$2" "$3" $T "IPo IPc IPl IPcq IPcqid IPcq0" q q > "deploy_q_$1.out" 2>&1
  echo "[$(ts)] $1: $(tail -n 1 "deploy_q_$1.out" | cut -c1-160)"
  grep -aq "ZZ SEQ STOP" "deploy_q_$1.out" && { echo "ZZ CHAIN-Q1 STOP after a stop of sequence $1"; exit 4; }
  grep -aq "ZZ DEPLOY-Q $1 STOP" "deploy_q_$1.out" && { echo "ZZ CHAIN-Q1 STOP in deploy $1"; exit 4; }
  return 0
}
run q1ls0 leadsup_a LS_s0
run q1ls1 leadsup_a LS_s1
run q1lsd0 leadsup_b LSD_s0
run q1lsd1 leadsup_b LSD_s1
echo "ZZ CHAIN-Q1 DONE $(ts)"
