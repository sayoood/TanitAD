#!/usr/bin/env bash
# chain_k6.sh : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N4. The two sequences that chain_k5.sh did not start: its stop test matched the words
# "memory guard" in the START line of sequence lft0k2 (which ended DONE), a fault of the test, not a stop. Here the test is the stop line of the pod sequence itself.
#   rlam0k1 RLAm_s0 "IPi IPk IPo"   (tool checks on the dose arm of base 0: p5b, p5r, the second diagnostic; they name nothing)
#   rlft0k1 RLFT_s0 "IPi IPk IPo"   (the same on its control)
# A run of these lists peaks at 2 GB resident (MEASURED, sequence lft0k2), so a run starts at 20 GB free (the bound of the run script itself); the guard stops it below 8 GB.
# No unsafe rate is read by this script. Deletes nothing. Last line: ZZ CHAIN-K6 DONE or STOP.
H=/f/TanitAD-scratch/v8proof/imagplan
cd "$H" || exit 1
export IP_NEED_GB=20
run() { # name queue arm tree runs sfx rsfx
  bash deploy_k5.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" > "deploy_k_$1.out" 2>&1
  tail -n 1 "deploy_k_$1.out" | cut -c1-200
  grep -aq "ZZ SEQ STOP .*memory guard" "deploy_k_$1.out" && { echo "ZZ CHAIN-K6 STOP after a memory-guard stop in $1"; exit 4; }
  return 0
}
run rlam0k1 actcons_c0 RLAm_s0 /workspace/v8proof/tree_actcons "IPi IPk IPo" x x
run rlft0k1 actcons_c0 RLFT_s0 /workspace/v8proof/tree_actcons "IPi IPk IPo" x x
echo "ZZ CHAIN-K6 DONE $(date -u +%H:%M:%SZ)"
