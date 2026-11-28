#!/usr/bin/env bash
# pod_ruleattr_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD29 (RULEATTR). Evaluation only, it trains nothing.
# The distance rule of card ADD26 (note N1, cap-only) with a mixed lead source, one rollout at a time and never as a fourth GPU job:
#   controls : LSXM1_s0 (all parts from the labels; m1: arrays equal to the label-fed dump LSRL_s0), LSXM2_s0 (all parts from the head; m2: equal to LSRH_s0)
#   arms     : XP (presence from the head), XG (gap from the head), XC (closing speed from the head), XZ (closing speed 0) on LS_s0, LS_s1, LSD_s0, LSD_s1
# Every rollout goes through pod_ruleattr_run.sh, a copy of pod_distrule_run.sh with the mixed wrapper (memory check, pinned tool check, no overwrite).
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/ruleattr_queue.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/RULEATTR_DONE.txt" ] && { echo "REFUSED: $D/RULEATTR_DONE.txt exists"; exit 2; }
[ -f "$D/v8f_t1_eval_rule_mix.py" ] || { echo "REFUSED: no v8f_t1_eval_rule_mix.py"; exit 2; }
RUN=$D/pod_ruleattr_run.sh
if [ ! -f "$RUN" ]; then
  sed -e 's/label|head|never|push|plumb|off) ;;/label|head|never|push|plumb|off|mix) ;;/' \
      -e 's|^ARGV=(--tree "$TREE" --dist-rule "$MODE")$|if [ "$MODE" = mix ]; then set -- $DR_MIX; ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3"); else ARGV=(--tree "$TREE" --dist-rule "$MODE"); fi|' \
      -e 's|v8f_t1_eval_rule\.py|${DR_RULEPY:-v8f_t1_eval_rule.py}|g' "$D/pod_distrule_run.sh" > "$RUN"
fi
[ "$(grep -c -- '--mix-presence' "$RUN")" = 1 ] && bash -n "$RUN" || { echo "REFUSED: the derived run script is not as expected"; exit 2; }
log "queue start; run script sha256 $(sha256sum "$RUN" | cut -c1-16) wrapper $(sha256sum "$D/v8f_t1_eval_rule_mix.py" | cut -c1-16)"
stop() { log "ZZ RULEATTR STOP: $1"; echo "$1" > "$D/RULEATTR_STOP.txt"; exit 4; }
run1() { # NEW Q BASE "P G C"
  local NEW=$1 Q=$2 BASE=$3 MIXS=$4 rc n i
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; return 1; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $Q/$BASE"; return 1; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only DR_MIX="$MIXS" DR_RULEPY=v8f_t1_eval_rule_mix.py bash "$RUN" "$NEW" "$Q" "$BASE" mix > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (mix $MIXS on $Q/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  return $rc
}
cmp() { /workspace/venv/bin/python "$D/cmp_dumps.py" "$R/$1/dump_$1_m10_causal_heads" "$R/$2/dump_$2_m10_causal_heads" 2>&1 | tail -n 1; }
# controls m1 and m2
run1 LSXM1_s0 leadsup_a LS_s0 "label label label" || stop "M1-RUN"
c1=$(cmp LSXM1_s0 LSRL_s0); log "m1 all-label mix against the label-fed dump: $c1"
echo "$c1" | grep -q 'differing=0 ' || stop "M1"
run1 LSXM2_s0 leadsup_a LS_s0 "head head head" || stop "M2-RUN"
c2=$(cmp LSXM2_s0 LSRH_s0); log "m2 all-head mix against the head-fed dump: $c2"
echo "$c2" | grep -q 'differing=0 ' || stop "M2"
log "controls m1 and m2 passed; 16 rollouts, one at a time"
bad=0
for s in 0 1; do
  for spec in "LS leadsup_a" "LSD leadsup_b"; do
    set -- $spec; b=$1; q=$2
    run1 "${b}XP_s$s" "$q" "${b}_s$s" "head label label" || bad=$((bad+1))
    run1 "${b}XG_s$s" "$q" "${b}_s$s" "label head label" || bad=$((bad+1))
    run1 "${b}XC_s$s" "$q" "${b}_s$s" "label label head" || bad=$((bad+1))
    run1 "${b}XZ_s$s" "$q" "${b}_s$s" "label label zero" || bad=$((bad+1))
  done
done
nd=$(ls -d $R/*X[PGCZ]_s[01]/T1_DONE 2>/dev/null | wc -l)
log "ZZ RULEATTR QUEUE DONE: $nd of 16 rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/RULEATTR_DONE.txt"
