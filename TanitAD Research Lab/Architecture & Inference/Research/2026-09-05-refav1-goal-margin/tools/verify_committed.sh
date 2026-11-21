#!/usr/bin/env bash
# End-of-turn content verification, with RETRIES and a same-breath CONTROL.
#
# ⛔ CLAUDE.md: `git rev-parse HEAD:<p>` and `git hash-object <p>` can BOTH
# return the empty string during a mount outage, and `[ "$a" = "$b" ]` is then
# TRUE — the check reports success while nothing was read. So: require BOTH
# sides to be 40 chars, treat anything else as INCONCLUSIVE, and retry.
#
# ⛔ And a run of INCONCLUSIVEs is not evidence about the commits — it is
# evidence about the channel. The CONTROL below (`HEAD:CLAUDE.md`, a path that
# has existed for months) must read 40 chars in the same breath; if it does not,
# the mount is down and the whole sweep is void rather than failing.
set -u
cd "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD" || exit 1
B="TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-goal-margin"
PATHS=(
  "$B/SPEC.md" "$B/ROOT_CAUSE.md" "$B/STEP2_RESULT.md" "$B/ESCALATION.md" "$B/RESULT.md"
  "$B/tools/margin_anatomy.py" "$B/tools/vocab_fit.py" "$B/tools/threshold_sweep.py"
  "$B/tools/vocab_design.py" "$B/tools/realised_kappa.py" "$B/tools/extract_intent.py"
  "$B/tools/kappa_head.py" "$B/tools/soft_kappa.py" "$B/tools/p4_turn_gate.py"
  "$B/tools/run_p4.sh" "$B/tools/run_p4c.sh"
  "$B/raw/margin_anatomy_s40.json" "$B/raw/margin_anatomy_s5.json"
  "$B/raw/vocab_fit_s40.json" "$B/raw/vocab_fit_dense.json"
  "$B/raw/threshold_sweep_s40.json" "$B/raw/threshold_sweep_dense.json"
  "$B/raw/vocab_design.json" "$B/raw/realised_kappa.json"
  "$B/raw/kappa_head_fit.json" "$B/raw/soft_kappa_s40.json"
  "$B/raw/intent_stride2_prov.json" "$B/raw/p4_episodes.json"
  "stack/tanitad/refs/refa_v1.py" "stack/tests/test_refa_v1_goal_kappa.py"
  "stack/tests/test_argparse_help_percent.py" "taniteval/tools/refav1_arm.py"
  "Project Steering/GOALS_AND_CLAIMS.md" "Project Steering/RETRACTION_LOG.md"
)
ok=0; bad=0; inc=0
for p in "${PATHS[@]}"; do
  res="INCONCLUSIVE"
  for try in 1 2 3 4 5 6 7 8; do
    ctl=$(git rev-parse "HEAD:CLAUDE.md" 2>/dev/null)     # same-breath control
    a=$(git rev-parse "HEAD:$p" 2>/dev/null)
    b=$(git hash-object "$p" 2>/dev/null)
    if [ ${#ctl} -ne 40 ]; then sleep 5; continue; fi      # channel down
    if [ ${#a} -eq 40 ] && [ ${#b} -eq 40 ]; then
      if [ "$a" = "$b" ]; then res="VERIFIED"; else res="MISMATCH"; fi
      break
    fi
    sleep 5
  done
  case "$res" in
    VERIFIED) ok=$((ok+1)) ;;
    MISMATCH) echo "MISMATCH      $p"; bad=$((bad+1)) ;;
    *) echo "INCONCLUSIVE  $p"; inc=$((inc+1)) ;;
  esac
done
echo "---- VERIFIED $ok  MISMATCH $bad  INCONCLUSIVE $inc  of ${#PATHS[@]} ----"
