#!/usr/bin/env bash
# r9gate3_chain.sh -- THOR-SIDE master chain for the refcv9 E1 arms (profile refcv9-e1rig).  GATE3, 2026-11-03; per-arm E-G gate5.
#
#   A  wait for the exploratory gate's smoke evidence (G-LIVE, G-CKPT) -> B start the commit-bound Thor full suite (CPU, nice 19, setsid)
#   C  wait for the exploratory gate to end; ALL SIX non-suite checks must be PASS, else kill the suite and STOP
#   D  wait for the suite -> gate_verdict -> suite-bind -> precheck with the gate's OWN judge (a FAIL STOPs: an env pin edit is the MM's call)
#   E-G PER ARM (gate5): wait for that arm's dev-box G-SUITE-PINNED evidence -> its real gate -> train it under the lock, release, next arm
#      the real gate (--stage thor) writes $R/gate3/<ARM>/PASS_<c12>.json (by the gate, never by hand); training runs sup_refcv7.sh, which
#      re-verifies the arm's own token (G1-G4) and REFUSES (exit 5) without it.  The next arm starts after this one ends, success or not.
#      Arms come from $R/gate3/ARMS (r9gate3_restart.sh); an arm whose run dir already holds a done summary is skipped (re-entry).
# Every verdict is read from an ARTIFACT (JSON / marker), never from $?.  No model turns poll this: it is plain bash.
# Status: $R/gate3/CHAIN_STATUS.txt (append-only, one event per line); opaque markers ZZR9E1CHAIN-<EVENT>ZZ also go to stdout.
set -u
R=/home/nvidia/r8ns/r9e1; T=$R/launch_tree; PY=/home/nvidia/venvs/tanitad-train/bin/python; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
SHA="$(cat $R/launch_tree.commit)"; C12="${SHA:0:12}"; G=$R/suite_$C12; STAT=$R/gate3/CHAIN_STATUS.txt
EXP=$R/gate3_explore/${C12}_C0; KIT=$R/kit
# the arms, in training order (written by r9gate3_restart.sh; default = the floor pair). C0 stays the explore arm.
ARMS="$(cat $R/gate3/ARMS 2>/dev/null)"; ARMS="${ARMS:-C0 C1}"
mkdir -p $R/gate3 $R/runs.d
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$STAT"; echo "ZZR9E1CHAIN-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
stop() { ev "STOP-$1" "${@:2}"; exit 3; }
status_of() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("status"))' "$1" 2>/dev/null; }
wait_for() { # $1 = file, $2 = max seconds -> 0 when present+non-empty
  local t0=$(date +%s); while [ ! -s "$1" ]; do [ $(( $(date +%s) - t0 )) -gt "$2" ] && return 1; sleep 20 9>&- 200>&-; done; return 0; }
[ "${#SHA}" = 40 ] || stop BADCOMMIT "launch_tree.commit"
ev START "commit $C12 tree $T arms [$ARMS]"

# ---- A: wait for the exploratory smoke evidence (a crashed/refused explore ends the wait) -------------------------------------
t0=$(date +%s)
while :; do
  [ -s "$EXP/evidence/G-LIVE.json" ] && [ -s "$EXP/evidence/G-CKPT.json" ] && break
  grep -q 'ZZR9E1G3-C0-explore-\(DONE\|REFUSED\)ZZ' $R/logs/r9gate3_explore_${C12}_C0.out 2>/dev/null && break
  [ $(( $(date +%s) - t0 )) -gt 5400 ] && stop EXPLORE-SMOKE-TIMEOUT "no smoke evidence in 90 min"
  sleep 20 9>&- 200>&-
done
grep -q 'ZZR9E1G3-C0-explore-REFUSEDZZ' $R/logs/r9gate3_explore_${C12}_C0.out 2>/dev/null && stop EXPLORE-REFUSED "see $R/logs/r9gate3_explore_${C12}_C0.out"

# ---- B: start the commit-bound Thor suite (CPU only, lowest priority) ----------------------------------------------------------
if [ ! -e "$G/results/ALLDONE" ] && [ ! -s "$G/runner.pid" ]; then
  [ -s "$G/plan_thor.json" ] && [ -s "$G/tree_tip.tar" ] && [ -s "$G/overlay.tar" ] || stop NO-SUITE-PLAN "$G"
  cp -p /home/nvidia/gate_fb47a583763f/thor_gate_runner2.py /home/nvidia/gate_fb47a583763f/gate_verdict.py $G/ || stop NO-RUNNER "runner files"
  ( cd $G; nohup setsid nice -n 19 env CUDA_VISIBLE_DEVICES= $PY $G/thor_gate_runner2.py $G /home/nvidia/refcv8_audit/pyt > $G/runner.out 2> $G/runner.err < /dev/null 9>&- 200>&- & echo $! > $G/runner.pid )
  sleep 5; ev SUITE-STARTED "runner pid $(cat $G/runner.pid 2>/dev/null)"
fi

# ---- C: the exploratory gate must end with six PASS ----------------------------------------------------------------------------
t0=$(date +%s)
while ! grep -q 'ZZR9E1G3-C0-explore-\(DONE\|REFUSED\)ZZ' $R/logs/r9gate3_explore_${C12}_C0.out 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { kill -- -"$(cat $G/runner.pid)" 2>/dev/null; stop EXPLORE-TIMEOUT "explore gate did not end in 2 h"; }
  sleep 30 9>&- 200>&-
done
BAD=""; for c in G-HYG G-DVB G-LIVE G-CKPT G-EVAL G-CLOCK; do s="$(status_of $EXP/evidence/$c.json)"; [ "$s" = PASS ] || BAD="$BAD $c=${s:-MISSING}"; done
if [ -n "$BAD" ]; then kill -- -"$(cat $G/runner.pid)" 2>/dev/null; stop EXPLORE-NOT-PASS "$BAD (read $EXP/evidence and $R/gate3_explore/${C12}_C0.timings.txt); suite runner killed"; fi
ev EXPLORE-PASS "six non-suite checks PASS on C0 (non-token run)"

# ---- D: the suite -> verdict -> suite-bind -> precheck --------------------------------------------------------------------------
if [ -s "$G/suite_bound.json" ] && tail -1 $G/precheck.out 2>/dev/null | grep -q '^G-SUITE PRECHECK: PASS'; then ev SUITE-BOUND "already bound + precheck PASS (re-entry): $G/suite_bound.json"; else
wait_for "$G/results/ALLDONE" 14400 || stop SUITE-TIMEOUT "no ALLDONE in 4 h ($G/runner.err)"
for f in TIP CAND; do grep -q '"done"[[:space:]]*:[[:space:]]*true' $G/results/$f.json || stop SUITE-INCOMPLETE "$f.json is not done"; done
mkdir -p $G/v/res && ln -sfn $G/results $G/v/res/results && ( cd $G && $PY gate_verdict.py $G/v > $G/verdict.out 2>&1 9>&- 200>&- )
V="$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print("VERDICT", d.get("verdict","NONE"), "regressions", len(d.get("regressions",[])), "new_fail", len(d.get("new_fail",[])))' $G/v/verdict.json 2>/dev/null)"
ev SUITE-VERDICT "$V"
# the AUTHORITY is the gate's own judge_suite_verdict (every failing id, incl. regressions / new_fail, must be inside the pinned env list):
# a not-clean line is logged and judged below by suite-bind + precheck, never by a second, private bar here.  Only an unreadable verdict stops.
case "$V" in VERDICT\ *) ;; *) stop SUITE-VERDICT-UNREADABLE "no verdict.json (read $G/verdict.out)";; esac
( cd $G/CAND && PYTHONPATH=$G/CAND/stack $PY stack/scripts/launch_gate.py suite-bind --verdict $G/v/verdict.json --cand-tree $G/CAND --commit $SHA --out $G/suite_bound.json > $G/suite_bind.out 2>&1 9>&- 200>&- )
grep -q "ZZSUITE-BOUND-${C12}ZZ" $G/suite_bind.out && [ -s $G/suite_bound.json ] || stop SUITE-BIND "no bound file (read $G/suite_bind.out)"
( cd /tmp && $PY $KIT/r9w_gsuite_precheck.py $T $G/suite_bound.json $SHA refcv9-e1rig > $G/precheck.out 2>&1 9>&- 200>&- )
tail -1 $G/precheck.out | grep -q '^G-SUITE PRECHECK: PASS' || stop SUITE-PRECHECK-FAIL "the gate's own judge refuses the suite verdict: read $G/precheck.out (outside-pinned failures need a reviewed pin-file edit: MM)"
ev SUITE-BOUND "suite_bound.json precheck PASS: $G/suite_bound.json"
fi

# ---- E/F/G, PER ARM (gate5, MM 2026-11-03): wait for THIS arm's dev-box pinned evidence -> its real gate -> train it under the lock -> RELEASE the
#      lock -> next arm.  (gate4 waited for every arm's evidence, then gated every arm, then trained them all under one lock: with 5 arms and a
#      loaded dev box (~95 min per pinned run) the GPU idled for hours.)  The lock is released between arms because the next arm's gate
#      (r9gate3_thor_gate.sh final) takes the SAME lock itself: holding it here would deadlock that gate.
[ -s "$R/gate3/zerograd_controls.env" ] || stop NO-ZEROGRAD-CONTROLS "$R/gate3/zerograd_controls.env is absent (the post-run zero-gradient gate must have controls)"
PINNED_WAIT_S="${R9_PINNED_WAIT_S:-14400}"
PASSED=""; TRAINED=""
for A in $ARMS; do
  D=$R/devbox_evidence_${C12}_${A}; ARGV=$T/stack/ops/runs.d/r9e1_${A}.argv.json; ARM=r9e1_$A
  OUTD="$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(a[a.index("--out")+1])' "$ARGV" 2>/dev/null)"
  if [ -n "$OUTD" ] && [ -s "$OUTD/summary.json" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$OUTD/summary.json"; then
    ev "$A-ALREADY-DONE" "$OUTD/summary.json is done: not relaunched (re-entry)"; continue; fi
  # E: this arm's dev-box pinned evidence
  t0=$(date +%s); while [ ! -s "$D/evidence/G-SUITE-PINNED.json" ] && [ ! -s "$D/G-SUITE-PINNED.json" ]; do
    [ $(( $(date +%s) - t0 )) -gt "$PINNED_WAIT_S" ] && break; sleep 30 9>&- 200>&-; done
  if [ -s "$D/evidence/G-SUITE-PINNED.json" ] || [ -s "$D/G-SUITE-PINNED.json" ]; then ev PINNED-PRESENT "$A at $D"
  else ev PINNED-ABSENT "$A: no G-SUITE-PINNED evidence under $D"; ev GATE-SKIPPED "$A: no dev-box pinned evidence"; continue; fi
  # F: the real gate (an existing token is reused on re-entry only if the gate's own verify MATCHes it)
  GOT=0
  if [ -s "$R/gate3/$A/PASS_$C12.json" ]; then
    rm -f $R/gate3/$A/verify_reentry.json; $PY $T/stack/scripts/launch_gate.py verify --token $R/gate3/$A/PASS_$C12.json --argv-file $ARGV --tree $T --commit $SHA --json $R/gate3/$A/verify_reentry.json > /dev/null 2>&1 9>&- 200>&-
    if [ "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict","NONE"))' $R/gate3/$A/verify_reentry.json 2>/dev/null)" = MATCH ]; then
      ev GATE-PASS "$A token reused on re-entry (verify MATCH): $R/gate3/$A/PASS_$C12.json"; GOT=1; fi
  fi
  if [ "$GOT" = 0 ]; then
    ( bash $KIT/r9gate3_thor_gate.sh final $A $G/suite_bound.json $D > $R/logs/r9gate3_final_$A.out 2>&1 9>&- 200>&- )
    if grep -q "ZZR9E1G3-${A}-final-PASSZZ" $R/logs/r9gate3_final_$A.out && [ -s "$R/gate3/$A/PASS_$C12.json" ]; then
      ev GATE-PASS "$A token $R/gate3/$A/PASS_$C12.json"; GOT=1
    else ev GATE-NOT-PASS "$A: $(grep -h 'verdict is\|REFUSED' $R/logs/r9gate3_final_$A.out | tail -1 | cut -c1-200) (read $R/gate3/$A.timings.txt)"; continue; fi
  fi
  PASSED="$PASSED $A"
  # G: train THIS arm under the Thor GPU lock, then release it
  exec 9>"$LOCKF"; ev LOCK-WAIT "$A: taking the Thor GPU lock (<= 6 h)"
  flock -w 21600 9 || stop LOCK-TIMEOUT "$A: the Thor GPU lock stayed held for 6 h; tokens are intact"
  NA="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"; [ "$NA" = 0 ] || stop GPU-NOT-EMPTY "$NA compute app(s) under the lock; never share the GPU"
  TOK=$R/gate3/$A/PASS_$C12.json; VJ=$R/gate3/$A/verify_prelaunch.json
  rm -f "$VJ"; $PY $T/stack/scripts/launch_gate.py verify --token "$TOK" --argv-file "$ARGV" --tree "$T" --commit "$SHA" --json "$VJ" > $R/gate3/$A/verify_prelaunch.log 2>&1 200>&-
  if [ "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict","NONE"))' "$VJ" 2>/dev/null)" != MATCH ]; then
    ev "$A-REFUSED" "verify did not MATCH ($VJ)"; flock -u 9; exec 9>&-; continue; fi
  cat > $R/runs.d/$ARM.env <<EOF
# $ARM -- launch manifest for stack/ops/sup_refcv7.sh (generated by r9gate3_chain.sh; sourced ONCE by the supervisor)
export CODE=$T
export COMMIT=$SHA
PYTHON=$PY
GATE_TOKEN=$TOK
GATE_ARGV_FILE=$ARGV
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
MAX_RELAUNCH=4
POLL_S=300
EOF
  cat $R/gate3/zerograd_controls.env >> $R/runs.d/$ARM.env
  ev "$A-LAUNCH" "supervisor $T/stack/ops/sup_refcv7.sh $ARM; run dir $OUTD"
  bash $T/stack/ops/sup_refcv7.sh $ARM $R/runs.d > $R/sup_$ARM.out 2>&1 200>&-
  RC=$?
  flock -u 9; exec 9>&-     # release the GPU lock: the next arm's gate takes it
  if [ -s "$OUTD/summary.json" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$OUTD/summary.json" && grep -q '"zerograd_gate"' "$OUTD/summary.json" && [ ! -e "$OUTD/ZEROGRAD_FAIL.json" ]; then ev "$A-DONE" "rc $RC; $OUTD/summary.json done + zerograd record"; TRAINED="$TRAINED $A"
  elif [ -e "$OUTD/ZEROGRAD_FAIL.json" ]; then ev "$A-ZEROGRAD-FAIL" "rc $RC; $OUTD/ZEROGRAD_FAIL.json"
  elif [ "$RC" = 5 ] || grep -q "ZZGATEREFUSED-$ARM" "$OUTD/sup_$ARM.log" 2>/dev/null; then ev "$A-REFUSED" "supervisor refused (rc $RC): $OUTD/sup_$ARM.log"
  else ev "$A-FAILED" "rc $RC: no done-marker at $OUTD (read $OUTD/sup_$ARM.log, $R/sup_$ARM.out)"; fi
done
[ -n "$PASSED" ] || stop NO-TOKEN "no arm passed the gate: nothing launched"
ev ALLDONE "chain finished (gated:$PASSED; trained:$TRAINED)"
