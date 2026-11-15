#!/usr/bin/env bash
# chain_k.sh : dev-box side, script polling. After the plumbing sequence lft0k1 has ended, tool checks (they name nothing) on the anchored arms of base 0 that exist:
# the dose arm RLAm_s0 (the early single-base look reads a response with the level kept) and the control RLFT_s0. Runs per arm, in this order: IPi (cand-imag with the
# p4/p5 log: p5b, p5r and the second diagnostic), IPk (cand-imagkin with the p4/p5 log), IPo (off). Card PREREG_V8F_PROOF_ADD40, note N4 (frozen before these runs).
# No unsafe rate is read by this script. Deletes nothing. Last line: ZZ CHAIN-K DONE or STOP.
H=/f/TanitAD-scratch/v8proof/imagplan
cd "$H" || exit 1
n=0
until grep -q "ZZ DEPLOY-K lft0k1" deploy_k_lft0k1.out 2>/dev/null; do n=$((n+1)); [ $n -gt 240 ] && { echo "ZZ CHAIN-K STOP lft0k1 not ended after 4 h"; exit 2; }; sleep 60; done
grep -q "ZZ DEPLOY-K lft0k1 DONE" deploy_k_lft0k1.out || { echo "ZZ CHAIN-K STOP lft0k1 did not end with DONE"; exit 3; }
for pr in "rlam0k1:RLAm_s0" "rlft0k1:RLFT_s0"; do
  nm=${pr%%:*}; arm=${pr##*:}
  bash deploy_k.sh "$nm" actcons_c0 "$arm" /workspace/v8proof/tree_actcons "IPi IPk IPo" x x > "deploy_k_$nm.out" 2>&1
  tail -n 1 "deploy_k_$nm.out" | cut -c1-200
  grep -q "ZZ DEPLOY-K $nm DONE" "deploy_k_$nm.out" || { echo "ZZ CHAIN-K STOP $nm"; exit 4; }
done
echo "ZZ CHAIN-K DONE $(date -u +%H:%M:%SZ)"
