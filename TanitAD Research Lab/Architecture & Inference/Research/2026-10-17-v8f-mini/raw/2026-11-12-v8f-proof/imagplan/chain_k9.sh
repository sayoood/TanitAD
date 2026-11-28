#!/usr/bin/env bash
# chain_k9.sh : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD40, note N6 section 4 (committed before the data): control p5b is met at both bases by the
# consistency weight 0.001 (RLA3_s0 0.7719, RLA3_s1 0.5755; row H-V8F-PROOF-91), so source cand-imag is evaluable on that arm under the card as written, own row.
# Run list of the card for source cand-imag on RLA3 at bases 0 and 1 (chain actcons_f), decel family, suffix x: IPo and IPi exist (kept by their T1_DONE marker);
# new: IPid1, IPid3 (control p1), IPs (control p6), IPc (extrapolation), IPl (label-fed, control p2), IPk (imagined level, beside). Then control p3 in the subset form
# of note N5 (first 600 windows, suffix m): IPi and IPn. All under the memory guard of pod_imagplan_seq6.sh; a run starts only with fewer than three GPU jobs (the run
# script refuses and the sequence waits). One sequence at a time. No unsafe rate is read by this script. Deletes nothing.
# Last line: ZZ CHAIN-K9 DONE or ZZ CHAIN-K9 STOP ...
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof/imagplan
cd "$H" || exit 1
export IP_NEED_GB=20
T=/workspace/v8proof/tree_actcons
ts() { date -u +%H:%M:%SZ; }
run() { # name queue arm runs sfx rsfx
  bash deploy_k6.sh "$1" "$2" "$3" $T "$4" "$5" "$6" > "deploy_k_$1.out" 2>&1
  echo "[$(ts)] $1: $(tail -n 1 "deploy_k_$1.out" | cut -c1-160)"
  grep -aq "ZZ SEQ STOP .*memory guard" "deploy_k_$1.out" && { echo "ZZ CHAIN-K9 STOP after a memory-guard stop in $1"; exit 4; }
  return 0
}
for K in 0 1; do
  arm=RLA3_s$K
  run "rla3s${K}i3" actcons_f "$arm" "IPo IPi IPid1 IPid3 IPs IPc IPl IPk" x x
  IP_MAX_WINDOWS=600 IP_NEED_NOISE_GB=20 run "rla3s${K}n1" actcons_f "$arm" "IPi IPn" m m
done
echo "ZZ CHAIN-K9 DONE $(ts)"
