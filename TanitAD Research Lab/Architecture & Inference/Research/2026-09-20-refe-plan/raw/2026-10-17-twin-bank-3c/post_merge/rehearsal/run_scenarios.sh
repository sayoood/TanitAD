#!/usr/bin/env bash
# REHEARSAL ONLY.  Runs post_merge_to_p6gate.sh end to end against the MOCK pod (a directory tree inside WSL, see build_mock_pod.sh) through every decision it can take, and writes
# rehearsal/RESULTS.tsv.  The REAL pod is never contacted.   usage (Git bash on the dev box):  bash run_scenarios.sh [scenario-id-prefix ...]
# Each scenario: reset the mock -> set its knobs (ctl/*, tampered files) -> run the orchestrator -> assert the exit code, the ZZPM lines that must / must not appear, and the pod-side effects.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PM="$(cd "$HERE/.." && pwd)"; PKG="$(cd "$PM/.." && pwd)"
export MSYS_NO_PATHCONV=1
WPKG="$(wsl.exe -e wslpath -a "$(cygpath -w "$PKG")" | tr -d '\0\r')"
P6REFE_WSL="${P6REFE_WSL:-$(wsl.exe -e wslpath -a "$(cygpath -w "${P6REFE_SRC:?set P6REFE_SRC to a dir holding the repo-tip online_labeller.py and p6_prep.py}")" | tr -d '\0\r')}"
OUTD="$HERE/transcripts"; mkdir -p "$OUTD"; RES="$HERE/RESULTS.tsv"; [ $# -gt 0 ] && RES="$HERE/RESULTS.partial.tsv"; : > "$RES"   # a filtered run never overwrites the full results
W=/tmp/p6mock/workspace
FILTER=("$@")
wsl_sh() { wsl.exe -e bash -c "$1" 2>&1 | tr -d '\0\r'; }
reset_mock() { wsl.exe -e bash "$WPKG/post_merge/rehearsal/build_mock_pod.sh" "$WPKG" "$P6REFE_WSL" > /dev/null 2>&1 || echo "MOCK BUILD FAILED"; }
ctl() { wsl_sh "echo '$2' > /tmp/p6mock/ctl/$1" > /dev/null; }
kept_part() {              # kept_part src|dst  -> path of the first KEPT twin's part, computed with P6's own OL.parts_path (kept_part.py)
  local base="$W/data/refe_navtrain/metric_cache_twins_3c"; [ "$1" = dst ] && base="$W/data/refe_navtrain/metric_cache"
  wsl_sh "PYTHONPATH=/tmp/p6mock/stubs python3 $WPKG/post_merge/rehearsal/kept_part.py $W/data/refe_navtrain/train_grow_cmd_3c $base $W/refe-plan-p6/refe" | tail -n 1
}
snap() { wsl_sh "find $W -type f -printf '%p %s %T@\n' | sort"; }          # the full listing (path size mtime) so a failure can print WHAT changed
snapdiff() { diff <(printf '%s\n' "$1") <(printf '%s\n' "$2") | head -n 12; }
COMMON_ENV=(SSH="bash $HERE/mock_ssh.sh" P6N_ALLOW_ANY_SSH=1 P6N_ROOT=$W P6N_POLL_S=2 P6N_GATE_POLL_S=2 P6N_RETRY_S=1 P6N_IDLE_POLL_S=3 P6N_IDLE_RECHECK_S=2 P6N_LAUNCH_POLL_S=2
            P6N_FIRST_STEP_S=12 P6N_LAUNCH_MAX_S=60 P6N_SHIP_COMMIT=1111111111111111111111111111111111111111 P6N_HF_TOKEN_FILE=/tmp/p6mock/hf_token_mock P6N_GPU_WAIT_MAX_H=0.01
            P6N_CODE_HASH_FILES="reroll_bank_3c.py cprime_check.py route_lane_rank_patch.py build_teacher_rollouts.py watch_3c.py" P6N_SSH_TIMEOUT=120)
PASSN=0; FAILN=0; CUR=""; LOG=""; RC=0
pm() {                     # pm [extra VAR=val ...] -- args    runs the orchestrator, output to $LOG, exit code in $RC
  local extra=() a
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
  LOG="$OUTD/$CUR.log"
  env "${COMMON_ENV[@]}" "${extra[@]}" P6N_LOGDIR="$OUTD/${CUR}_run" bash "$PM/post_merge_to_p6gate.sh" "$@" > "$LOG" 2>&1; RC=$?
}
want_rc() { [ "$RC" = "$1" ] || { echo "  rc=$RC, wanted $1"; return 1; }; }
has()   { grep -qF -- "$1" "$LOG" || { echo "  missing in output: $1"; return 1; }; }
hasnt() { ! grep -qF -- "$1" "$LOG" || { echo "  must NOT appear: $1"; return 1; }; }
scn() {                    # scn <id> <description>   (the body is the function s_<id> defined below; it returns 0 on success)
  local id="$1" desc="$2" run=1 f
  if [ ${#FILTER[@]} -gt 0 ]; then run=0; for f in "${FILTER[@]}"; do [[ "$id" == "$f"* ]] && run=1; done; fi
  [ "$run" = 1 ] || return 0
  CUR="$id"; reset_mock
  local msg; msg="$("s_$id" 2>&1)"; local rc=$?
  if [ $rc = 0 ]; then PASSN=$((PASSN + 1)); printf '%s\tPASS\t%s\t%s\n' "$id" "$desc" "rc=$RC" >> "$RES"; echo "SCN $id PASS  $desc"
  else FAILN=$((FAILN + 1)); printf '%s\tFAIL\t%s\t%s\n' "$id" "$desc" "$(echo "$msg" | tr '\n' ';')" >> "$RES"; echo "SCN $id FAIL  $desc"; echo "$msg"; fi
}

# =============================================================================== the happy path, and what it leaves behind
s_H0() { pm -- ; want_rc 0 && has "ZZPM a OK" && has "ZZPM b OK" && has "ZZPM c OK" && has "ZZPM c ZZTWIN_HOLD_OK" && has "ZZPM d OK" && has "ZZPM e OK" && has "ZZPM f OK" && has "ZZPM P6_TRAINING" \
  && hasnt "P6_NOT_LAUNCHED" && has "DONE result=OK through=f" || return 1
  # pod-side evidence: the dry run used the 3-C bank; the real launch ran without a stop point and with the fallback OFF; the cache holds the kept twins' parts
  wsl_sh "grep -c 'launch gate PASS' $W/data/refe_p6/chain.log; grep -c 'train.py preflight OK' $W/data/refe_p6/chain.log; ls $W/data/refe_navtrain/metric_cache/*/*__lane_rank_* | wc -l; test -f $W/data/refe_runs/p6_paper_faithful/metrics.jsonl && echo metrics" | tr '\n' ' ' | grep -q '^1 1 23 metrics' || { echo "  pod-side state wrong: $(wsl_sh "grep -c 'launch gate PASS' $W/data/refe_p6/chain.log; grep -c 'train.py preflight OK' $W/data/refe_p6/chain.log" | tr '\n' ' ')"; return 1; }; }
s_H1() { pm -- --no-launch; want_rc 0 && has "ZZPM e OK" && has "P6_LAUNCH_SKIPPED" && hasnt "P6_TRAINING" && hasnt "ZZPM f" || return 1
  wsl_sh "test -e $W/data/refe_runs/p6_paper_faithful && echo LAUNCHED" | grep -q LAUNCHED && { echo "  --no-launch launched P6"; return 1; }; return 0; }
s_H2() { local b a; b="$(snap)"; pm -- --dry; a="$(snap)"; want_rc 0 && has "ZZPM_DRY" && has "ZZPM e OK OK_DRY" && has "ZZPM f OK OK_DRY" && ! grep -q '^ZZPM P6_TRAINING' "$LOG" || return 1
  [ "$b" = "$a" ] || { echo "  --dry changed the mock pod:"; snapdiff "$b" "$a"; return 1; }; }
s_H3() { local b a; b="$(snap)"; pm -- --probe; a="$(snap)"; want_rc 0 && has "PROBE_DONE" && has "probe e pod-free check" || return 1; [ "$b" = "$a" ] || { echo "  --probe changed the mock pod:"; snapdiff "$b" "$a"; return 1; }; }
s_H4() { pm -- ; want_rc 0 || return 1; local n1; n1="$(wsl_sh "grep -c 'train.py preflight OK' $W/data/refe_p6/chain.log")"
  pm -- --from f; want_rc 13 && has "P6_NOT_LAUNCHED" && hasnt "LAUNCHING P6" && hasnt "P6_TRAINING" || return 1       # the second run re-judges the dry-run slice, sees the real launch's trainer lines, and refuses
  [ "$(wsl_sh "grep -c 'train.py preflight OK' $W/data/refe_p6/chain.log")" = "$n1" ] || { echo "  a second launch happened"; return 1; }; }

# =============================================================================== (a) the merge
s_A1() { wsl_sh "printf 'ZZ3C_MERGE_REFUSED shards were run on another input bank (sha256 differs)\n' > $W/ops/bank3c/full/merge.out" > /dev/null; pm -- ; want_rc 3 && has "MERGE REFUSED (rc 3)" && has "P6_NOT_LAUNCHED" && hasnt "ZZPM b" ; }
s_A2() { wsl_sh "printf 'ZZ3C_YIELD_GUARD_TRIPPED {k1 share 0.11} -- NO bank is handed to P6\n' > $W/ops/bank3c/full/merge.out" > /dev/null; pm -- ; want_rc 4 && has "YIELD GUARD TRIPPED (rc 4)" && has "hand NOTHING to P6" && hasnt "ZZPM b" ; }
s_A3() { wsl_sh "printf 'Traceback (most recent call last):\n ValueError: boom\n' > $W/ops/bank3c/full/merge.out" > /dev/null; pm -- ; want_rc 5 && has "no merge process is running (crash)" ; }
s_A4() { wsl_sh "rm $W/ops/bank3c/full/RUN_DONE; echo 'shards done 700 of 720' > $W/ops/bank3c/full/RUN_INCOMPLETE" > /dev/null; pm -- ; want_rc 5 && has "RUN_INCOMPLETE" ; }
s_A5() { wsl_sh "echo '{}' >> $W/data/refe_navtrain/train_grow_cmd_3c/targets_rank1.jsonl" > /dev/null; pm -- ; want_rc 5 && has "do not hold up" && hasnt "ZZPM b" ; }
s_A6() { wsl_sh "sed -i 's/e8365f34/e8365f35/' $W/data/refe_navtrain/train_grow_cmd_3c/merge_report.json" > /dev/null; pm -- ; want_rc 5 && has "frozen_PREREG_3C" ; }
s_A7() { wsl_sh "rm $W/ops/bank3c/full/RUN_DONE $W/ops/bank3c/full/merge.out" > /dev/null; pm P6N_WAIT_MAX_H=0.0006 P6N_POLL_S=1 -- ; want_rc 5 && has "timeout: no finished merge" ; }
s_A8() { wsl_sh "sed -i 's/\"status\": \"HOLDS\"/\"status\": \"HOLDS\", \"note\": \"the two denominators DISAGREE\"/' $W/data/refe_navtrain/train_grow_cmd_3c/merge_report.json" > /dev/null; pm -- ; want_rc 5 && has "disagree" ; }
s_A9() { wsl_sh "rm $W/ops/bank3c/full/merge.out" > /dev/null; pm P6N_MERGE_GRACE_S=3 -- ; want_rc 5 && has "no merge.out after" ; }
s_A10() { wsl_sh "sed -i 's/watch_3c/watch_3d/' $W/ops/bank3c/code/cprime_check.py; echo '# tampered' >> $W/ops/bank3c/code/cprime_check.py" > /dev/null; pm -- ; want_rc 2 && has "differs from MANIFEST.json" ; }

# =============================================================================== (b) (c'2)
s_B1() { ctl c2mode slow; pm -- ; want_rc 6 && has "3C-SLOW" && has "P6_NOT_LAUNCHED" && hasnt "ZZPM c " ; }
s_B2() { ctl c2mode drop; pm -- ; want_rc 7 && has "INCOMPLETE" ; }
s_B3() { pm -- --to b; want_rc 0 && has "(c'2) PASS"; }

# =============================================================================== (c) (c'3)
s_C1() { ctl c3mode bfail; pm -- ; want_rc 8 && has "(c'3) FAILS" && has "K2(a=True,b=False" && hasnt "ZZPM d" ; }
s_C2() { ctl c3mode nosummary; pm -- ; want_rc 8 && has "state=missing" ; }
s_C3() { pm -- --to c; want_rc 0 && has "c_reported=False" && has "ZZTWIN_HOLD_OK" ; }
s_C4() { # the REAL reroll_twins.py check on a bank where one lane row lost aug.route_hold: must print ZZTWIN_HOLD_BAD, rc 1
  wsl_sh "cd $W/ops/bank3c/code && mkdir -p /tmp/p6mock/badhold && python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('$W/data/refe_navtrain/train_grow_cmd_3c/targets_rank1.jsonl')]
done=False
with open('/tmp/p6mock/badhold/targets_rank1.jsonl','w') as fo:
    for r in rows:
        if not done and isinstance(r.get('aug'),dict) and r['aug'].get('kind')=='lane_rank':
            r['aug'].pop('route_hold'); done=True
        fo.write(json.dumps(r)+'\n')
PY
python3 reroll_twins.py check --bank /tmp/p6mock/badhold; echo rc=\$?" > "$OUTD/C4.log"
  grep -q "ZZTWIN_HOLD_BAD" "$OUTD/C4.log" && grep -q "rc=1" "$OUTD/C4.log" || { echo "  hold check did not go RED: $(cat "$OUTD/C4.log")"; return 1; }; }

# =============================================================================== (d) parts
s_D1() { local f; f="$(kept_part src)"; [ -n "$f" ] || { echo "  no kept part path"; return 1; }; wsl_sh "rm $f" > /dev/null; pm -- ; want_rc 9 && has "missing_src=1" && has "P6 must NOT rebuild twins" && hasnt "ZZPM e" ; }
s_D2() { local d; d="$(kept_part dst)"; [ -n "$d" ] || { echo "  no kept part path"; return 1; }; wsl_sh "echo different-content > $d" > /dev/null; pm -- ; want_rc 9 && has "conflict=1" ; }
s_D3() { pm -- --to d; local l1; l1="$(wsl_sh "ls $W/data/refe_navtrain/metric_cache/*/*__lane_rank_* | wc -l")"; pm -- --from d --to d; want_rc 0 && has "nothing to place" && [ "$l1" = 23 ] || { echo "  placed=$l1"; return 1; }
  wsl_sh "f=\$(ls $W/data/refe_navtrain/metric_cache/*/*__lane_rank_* | head -n 1); stat -c '%h' \$f" | grep -q '^2$' || { echo "  not hard-linked"; return 1; }; }
s_D4() { pm P6N_PARTS_HOW=copy -- --to d; want_rc 0 && has "mode=copy" || return 1
  wsl_sh "f=\$(ls $W/data/refe_navtrain/metric_cache/*/*__lane_rank_* | head -n 1); stat -c '%h' \$f" | grep -q '^1$' || { echo "  copy mode left a hard link"; return 1; }; }
s_D5() { pm -- --to d; has "P6's D7 acceptance K=1" && has "ZZD7_DISTINCT_OK" && has "P6's D7 acceptance K=3" ; }

# =============================================================================== (e) the pod must be FREE, and the dry run must hold up
s_G1() { pm -- --to d; want_rc 0 || return 1; ctl gpu busy; ( sleep 22; wsl_sh "echo idle > /tmp/p6mock/ctl/gpu" > /dev/null ) & pm P6N_GPU_WAIT_MAX_H=0.05 -- --from e --to f; wait; want_rc 0 && has "WAITING for a free pod" && has "gpu_busy" && has "pod is FREE" && has "ZZPM e OK" && has "ZZPM P6_TRAINING" && hasnt "GATE_DEFERRED" ; }
s_G2() { ctl gpu busy; pm P6N_GPU_WAIT_MAX_H=0.002 -- --from d --to f; want_rc 10 && has "GATE_DEFERRED gpu_busy_wait_cap_0.002h" && has "P6_NOT_LAUNCHED" && hasnt "ZZPM e OK" ; }
s_G3() { ctl gpu busy; pm -- --from d --to f --no-wait; want_rc 10 && has "ZZPM GATE_DEFERRED gpu_busy" && has "P6_NOT_LAUNCHED" ; }
s_G4() { ctl gpu unknown; pm -- --from d --to f --no-wait; want_rc 10 && has "GATE_DEFERRED gpu_busy" && has "gpu_unknown" ; }
s_G5() { wsl_sh "setsid -f nohup bash $W/refe-plan-p6/refe/rls1_pod.sh 300 > /dev/null 2>&1 < /dev/null; sleep 1" > /dev/null
  pm -- --from d --to f --no-wait; local rc=$RC; wsl_sh "kill \$(cat /tmp/p6mock/rls1.pid)" > /dev/null; RC=$rc
  want_rc 10 && has "rls1_pod.sh_running" && has "GATE_DEFERRED gpu_busy" && hasnt "ZZPM e OK" ; }
s_G6() { wsl_sh "setsid -f nohup flock -x $W/data/rls1/chain.lock sh -c 'echo \$\$ > /tmp/p6mock/lock.pid; exec sleep 300' > /dev/null 2>&1 < /dev/null; sleep 1" > /dev/null
  pm -- --from d --to f --no-wait; local rc=$RC; wsl_sh "kill \$(cat /tmp/p6mock/lock.pid)" > /dev/null; RC=$rc
  want_rc 10 && has "chain_lock:rls1" && has "GATE_DEFERRED" ; }
s_G7() { wsl_sh "setsid -f nohup flock -x $W/data/refe_p6/chain.lock sh -c 'echo \$\$ > /tmp/p6mock/lock.pid; exec sleep 300' > /dev/null 2>&1 < /dev/null; sleep 1" > /dev/null
  pm -- --from d --to f; local rc=$RC; wsl_sh "kill \$(cat /tmp/p6mock/lock.pid)" > /dev/null; RC=$rc
  want_rc 12 && has "already holds its chain.lock" && has "P6_NOT_LAUNCHED" && hasnt "GATE_DEFERRED" ; }
s_E1() { wsl_sh "sed -i 's/P6_CBANK/P6_XBANK/g' $W/refe-plan-p6/refe/p6_pod.sh" > /dev/null; pm -- --from d --to f; want_rc 12 && has "no P6_CBANK knob" && has "P6_NOT_LAUNCHED" && hasnt "ZZPM e OK" ; }
s_E2() { pm P6N_SHIP_COMMIT=2222222222222222222222222222222222222222 -- --from d --to f; want_rc 12 && has "re-ship" ; }
s_E3() { pm P6N_SHIP_COMMIT= -- --from d --to f; want_rc 12 && has "P6N_SHIP_COMMIT is not set" ; }
s_E4() { ctl p6mode refuse; pm -- --from d --to f; want_rc 12 && has "the P6 chain REFUSED during the dry run" && has "P6_NOT_LAUNCHED" && hasnt "ZZPM P6_TRAINING" || return 1
  wsl_sh "test -e $W/data/refe_runs/p6_paper_faithful && echo LAUNCHED" | grep -q LAUNCHED && { echo "  P6 was launched after a refused gate"; return 1; }; return 0; }
s_E5() { ctl p6mode fallback; pm -- --from d --to f; want_rc 12 && has "D7_FALLBACK" && has "P6_NOT_LAUNCHED" ; }
s_E6() { ctl p6mode wrongbank; pm -- --from d --to f; want_rc 12 && has "not_the_3-C_bank" && has "P6_NOT_LAUNCHED" ; }
s_E7() { ctl p6mode rebuild; pm -- --from d --to f; want_rc 12 && has "parts_in_MC_were_written_during_the_dry_run" ; }
s_E8() { ctl p6mode nopass; pm -- --from d --to f; want_rc 12 && has "exactly_one_PASS_token" && has "P6_NOT_LAUNCHED" ; }
s_E9() { ctl p6mode train; pm -- --from d --to f; want_rc 12 && has "trainer_side_of_the_chain_ran" && has "P6_NOT_LAUNCHED" ; }
s_E10() { wsl_sh "rm $W/data/refe_p6/stage/cache.ok" > /dev/null; pm -- --from d --to f; want_rc 12 && has "MISSING:cache" ; }

# =============================================================================== (f) the launch
s_F1() { ctl p6launch refuse; pm -- --from d --to f; want_rc 13 && has "ZZPM P6_NOT_LAUNCHED f: the real launch did not start training" && hasnt "P6_TRAINING" && hasnt "P6_LAUNCH_UNCONFIRMED" ; }
s_F2() { ctl p6launch nostep; pm -- --from d --to f; want_rc 13 && has "no first step/loss line within" && has "P6_LAUNCH_UNCONFIRMED" && hasnt "P6_TRAINING" ; }
s_F3() { ctl p6launch exit; pm -- --from d --to f; want_rc 13 && has "the trainer exited before its first step" && has "P6_LAUNCH_UNCONFIRMED" ; }
s_F4() { ctl p6launch stepnopre; pm -- --from d --to f; want_rc 13 && has "AMBIGUOUS" && hasnt "P6_TRAINING" ; }
s_F5() { ctl p6launch gone; pm -- --from d --to f; want_rc 13 && has "P6_NOT_LAUNCHED" && has "the chain refused or never ran" ; }
s_F6() { pm -- --from d --to e; want_rc 0 || return 1; pm -- --from f; want_rc 0 && has "dry-run evidence re-judged from the pod" && has "ZZPM P6_TRAINING" ; }
s_F7() { pm -- --from f; want_rc 13 && has "no recorded gate dry run" && has "P6_NOT_LAUNCHED" ; }
s_F8() { ctl gpu busy; pm P6N_GPU_WAIT_MAX_H=0.002 -- --from f; want_rc 13 ; }
s_F9() { pm -- --from d --to e; ctl gpu busy; pm P6N_GPU_WAIT_MAX_H=0.002 -- --from f; want_rc 10 && has "GATE_DEFERRED gpu_busy_wait_cap" && has "P6_NOT_LAUNCHED" && hasnt "LAUNCHING P6" ; }

# =============================================================================== plumbing
s_X1() { pm SSH="bash $HERE/ssh_dead.sh" -- ; want_rc 2 && has "pod unreachable (ssh failed 3x)" && has "P6_NOT_LAUNCHED" ; }
s_X1b() { pm SSH=false -- ; want_rc 2 && has "pod answered but not as expected" && has "P6_NOT_LAUNCHED" ; }
s_X6() { pm P6N_MERGE_GRACE_S=abc -- ; want_rc 2 && has "P6N_MERGE_GRACE_S=abc is not a whole number" && hasnt "ZZPM a" ; }
s_X7() { # SIGTERM while waiting for the merge: the EXIT trap must turn the unexpected exit into a FAIL line + P6_NOT_LAUNCHED + exit 14, never silence
  wsl_sh "rm $W/ops/bank3c/full/RUN_DONE $W/ops/bank3c/full/merge.out" > /dev/null; LOG="$OUTD/$CUR.log"
  env "${COMMON_ENV[@]}" P6N_LOGDIR="$OUTD/${CUR}_run" timeout -s TERM 14 bash "$PM/post_merge_to_p6gate.sh" > "$LOG" 2>&1; RC=$?
  echo "  (timeout rc=$RC)"; has "internal error: the script exited unexpectedly" && has "P6_NOT_LAUNCHED" && has "DONE result=FAIL step=internal exit=14" && hasnt "result=OK" ; }
s_X8() { # MUTANT: put back the original unvalidated arithmetic (fractional hours -> a fatal expansion error that used to skip the step loop and print DONE result=OK).
         # The end-of-pipeline guard must still turn it into a FAIL.
  local m="$PM/_mutant_nowait.sh"
  sed -e 's/^WAIT_S=.*/WAIT_S=0/' -e 's/-gt "\$WAIT_S"/-gt $(( WAIT_H * 3600 ))/' -e 's/^\[\[ "\$WAIT_H" =~.*$//' "$PM/post_merge_to_p6gate.sh" > "$m"
  wsl_sh "rm $W/ops/bank3c/full/RUN_DONE $W/ops/bank3c/full/merge.out" > /dev/null; LOG="$OUTD/$CUR.log"
  env "${COMMON_ENV[@]}" P6N_WAIT_MAX_H=0.0006 P6N_POLL_S=1 P6N_LOGDIR="$OUTD/${CUR}_run" bash "$m" > "$LOG" 2>&1; RC=$?; rm -f "$m"
  grep -q 'arithmetic syntax error' "$LOG" || { echo "  the mutant did not hit the arithmetic error (mutation not effective)"; return 1; }
  [ "$RC" != 0 ] && has "DONE result=FAIL" && hasnt "result=OK" ; }
s_X2() { pm SSH= -- ; want_rc 2 && has "SSH is not set" ; }
s_X3() { pm P6N_ALLOW_ANY_SSH=0 SSH="ssh -p 1 nobody@invalid.example" -- ; want_rc 2 && has "BatchMode=yes" ; }
s_X4() { pm -- --from b --to b; ctl c2mode pass; wsl_sh "rm $W/ops/bank3c/full/merge.out" > /dev/null; pm -- --from b; want_rc 5 && has "no finished, valid merge" ; }
s_X5() { # nothing landable may carry a pod address, port, key path, user@host or a token.  GENERIC patterns only: this file must not itself spell out the real values.
  # (the real values, if the operator exports them as P6N_SCAN_FIXED="host port keyname", are searched too; they are never stored.)
  local pat='root[@]|([0-9]{1,3}[.]){3}[0-9]{1,3}|[ ]-p[ ]+[0-9]{4,5}|[ ]-i[ ]+[^ ]*[.]ssh|[.]ssh/|hf_[A-Za-z0-9]{12,}|Authorization:|Bearer[ ]'
  local hits; hits="$(grep -rnIE "$pat" "$PM" --include='*.sh' --include='*.py' --include='*.md' --include='*.diff' --include='*.tsv' --include='*.json' 2>/dev/null | grep -v '/transcripts/' | grep -v 'run_scenarios.sh' || true)"
  [ -z "$hits" ] || { echo "  secret-looking strings in the package: $(echo "$hits" | head -n 5)"; return 1; }
  if [ -n "${P6N_SCAN_FIXED:-}" ]; then for w in $P6N_SCAN_FIXED; do
    grep -rnIF -e "$w" "$PM" --include='*' 2>/dev/null | grep -v '/transcripts/' | grep -q . && { echo "  a real pod identifier is present in the package"; return 1; }; done; fi
  return 0; }

# ===============================================================================  WIRING MUTANTS ("built, tested, and unreachable from its caller": delete the line, the scenario must go RED)
# Each mutant is the orchestrator with ONE guard line neutralised (sed), run on a scenario whose unmutated outcome is a refusal.  The mutant must then DO the unsafe thing, which proves the
# refusal in the unmutated run really came from that line.  Where the design has a second layer (the gate judge runs at (e) AND again at (f)), one layer removed must STILL refuse.
mutant_run() {             # mutant_run <sed expr> [<sed expr> ...] -- [pm args after --]     -> LOG/RC of the mutant run
  local m="$PM/_mutant_$CUR.sh" exprs=() a
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do exprs+=(-e "$1"); shift; done; [ "${1:-}" = "--" ] && shift
  sed "${exprs[@]}" "$PM/post_merge_to_p6gate.sh" > "$m"; cmp -s "$m" "$PM/post_merge_to_p6gate.sh" && { echo "  mutation not effective (sed changed nothing)"; rm -f "$m"; return 1; }
  LOG="$OUTD/$CUR.log"; env "${COMMON_ENV[@]}" P6N_LOGDIR="$OUTD/${CUR}_run" bash "$m" "$@" > "$LOG" 2>&1; RC=$?; rm -f "$m"; return 0
}
E_JUDGE='s/finish_fail e 12 "the dry run ended/: "the dry run ended/'
F_JUDGE='s/finish_fail f 13 "the dry-run evidence does not hold up/: "the dry-run evidence does not hold up/'
s_X9() { ctl p6mode wrongbank; mutant_run "$E_JUDGE" -- --from d --to f || return 1                       # the gate judge at (e) removed: the token that vouches for the PRE-3C bank is launched over (E6's refusal came from that line)
  has "P6_TRAINING" && want_rc 0 ; }
s_X9b() { ctl p6mode wrongbank; mutant_run "$E_JUDGE" -- --from d --to e || return 1; want_rc 0 || return 1       # (e) mutated: the wrong-bank dry run is accepted ...
  pm -- --from f; want_rc 13 && has "P6_NOT_LAUNCHED" && hasnt "LAUNCHING P6" && has "the dry-run evidence does not hold up" ; }   # ... and the UNMUTATED (f), run on its own, re-judges the pod's evidence and refuses
s_X9c() { ctl p6mode wrongbank; mutant_run "$E_JUDGE" -- --from d --to e || return 1; want_rc 0 || return 1
  mutant_run "$F_JUDGE" -- --from f || return 1; has "P6_TRAINING" && want_rc 0 ; }                         # (f)'s re-judge removed as well: launched (the re-judge line was the refusal)
s_X10() { ctl gpu busy; mutant_run 's/if \[ "\$state" = idle \]; then/if true; then/' -- --from d --to f --no-wait || return 1     # the busy check not honoured: the gate would start on a busy A40
  has "pod is FREE" && hasnt "GATE_DEFERRED" ; }
E_REF='s/finish_fail e 12 "the P6 chain REFUSED during the dry run/: "the P6 chain REFUSED during the dry run/'
E_RC='s/\[ "\$rc" = 0 \] || finish_fail e 12 "p6_pod.sh exited rc/true || finish_fail e 12 "p6_pod.sh exited rc/'
F_LANESOK='s/finish_fail f 13 "\$P6\/stage\/lanes.ok is missing/: "\$P6\/stage\/lanes.ok is missing/'
F_GATEOK='s/finish_fail f 13 "\$P6\/stage\/launch_gate.ok is missing/: "\$P6\/stage\/launch_gate.ok is missing/'
s_X11() { ctl p6mode refuse; mutant_run "$E_REF" -- --from d --to f || return 1                              # the REFUSED-count guard removed: the later layers (rc, gate judge, re-judge, launch_gate.ok) still stop it
  has "P6_NOT_LAUNCHED" && hasnt "P6_TRAINING" && want_rc 12 ; }
s_X11b() { ctl p6mode refuse; mutant_run "$E_REF" "$E_RC" "$E_JUDGE" "$F_GATEOK" "$F_LANESOK" -- --from d --to f || return 1      # every layer (REFUSED count, rc, gate judge, launch_gate.ok, lanes.ok) removed: the REFUSED gate is launched over
  has "P6_TRAINING" && want_rc 0 ; }

for id in H0 H1 H2 H3 H4 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 B1 B2 B3 C1 C2 C3 C4 D1 D2 D3 D4 D5 G1 G2 G3 G4 G5 G6 G7 E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 F1 F2 F3 F4 F5 F6 F7 F8 F9 X1 X1b X2 X3 X4 X5 X6 X7 X8 X9 X9b X9c X10 X11 X11b; do
  desc="$(grep -m1 -o "^s_$id() {.*" "$HERE/run_scenarios.sh" | head -c 0)"
  case "$id" in
    H0) d="happy path a-f: auto-launch, P6_TRAINING";; H1) d="--no-launch stops at the gate PASS, P6 not launched";; H2) d="--dry executes no pod write";; H3) d="--probe executes no pod write";; H4) d="second run while P6 runs: fail, no double launch";;
    A1) d="merge rc 3 (refused) => STOP exit 3";; A2) d="merge rc 4 (yield guard) => STOP exit 4";; A3) d="merge crashed (no marker)";; A4) d="RUN_INCOMPLETE";; A5) d="merge.out OK but the bank was tampered";; A6) d="merge report not on the frozen PREREG";;
    A7) d="wait timeout";; A8) d="yield-guard denominators disagree => Master Mind decides";; A9) d="RUN_DONE but the watcher never merged";; A10) d="frozen 3-C code changed on the pod";;
    B1) d="(c'2) fails => 3C-SLOW exit 6";; B2) d="(c'2) incomplete (re-rolls missing) => exit 7";; B3) d="(c'2) passes";;
    C1) d="(c'3) (b) fails at K=2 => exit 8";; C2) d="(c'3) verify crashed (no summary) => exit 8";; C3) d="(c'3) passes, (c) reported not judged, ZZTWIN_HOLD_OK";; C4) d="REAL reroll_twins.py check goes RED on a lane row without route_hold";;
    D1) d="a kept twin's part missing in the source => exit 9, no P6 rebuild";; D2) d="conflicting file at the destination => exit 9";; D3) d="hard-link placement, idempotent re-run";; D4) d="copy placement (no hard link)";; D5) d="P6's own D7 acceptance runs on the placed parts";;
    G1) d="GPU busy, frees up: WAIT then proceed and launch";; G2) d="GPU busy past the cap => GATE_DEFERRED, exit 10";; G3) d="--no-wait: GATE_DEFERRED gpu_busy";; G4) d="nvidia-smi cannot answer => busy";; G5) d="rls1_pod.sh (R0d) alive, GPU idle => busy";;
    G6) d="another chain's lock => busy";; G7) d="P6's own lock held => FAIL not wait";; E1) d="no P6_CBANK knob in the shipped tree => FAIL";; E2) d="stale ship commit => FAIL";; E3) d="P6N_SHIP_COMMIT unset => FAIL";;
    E4) d="gate REFUSED => P6_NOT_LAUNCHED, nothing launched";; E5) d="chain took the D7 fallback => FAIL";; E6) d="token vouches for the PRE-3C bank => FAIL";; E7) d="a part was written during the dry run => FAIL";; E8) d="no PASS token => FAIL";; E9) d="trainer side ran in the dry run => FAIL";; E10) d="cache.ok missing => FAIL";;
    F1) d="real launch REFUSED => NOT_LAUNCHED exit 13";; F2) d="preflight OK but no first step => UNCONFIRMED exit 13";; F3) d="trainer exited before step 0";; F4) d="step line without preflight => AMBIGUOUS";; F5) d="chain gone without a marker";;
    F6) d="--from f re-judges the dry run from the pod and launches";; F7) d="--from f with no recorded dry run => NOT_LAUNCHED";; F8) d="--from f, pod busy past cap";; F9) d="busy again after the dry run PASSed: no launch";;
    X1) d="ssh connection dead (255) => exit 2, NOT_LAUNCHED";; X1b) d="ssh answers nothing sensible (rc 1) => exit 2";; X6) d="malformed numeric knob => exit 2 at once";; X7) d="SIGTERM mid-wait => EXIT trap: FAIL + NOT_LAUNCHED, exit 14";; X8) d="MUTANT: arithmetic crash in step a still ends FAIL, never result=OK";; X9) d="MUTANT: gate judge removed at (e): the wrong-bank gate is launched over";; X9b) d="MUTANT: judge removed at (e) only: a stand-alone (f) re-judges and refuses";; X9c) d="MUTANT: judge removed at (e) AND (f): launched";; X10) d="MUTANT: busy check not honoured: gate would start on a busy pod";; X11) d="MUTANT: REFUSED-count guard removed: later layers still refuse";; X11b) d="MUTANT: all REFUSED/lanes/launch_gate layers removed: the refused gate is launched over";; X2) d="SSH unset => exit 2";; X3) d="SSH without BatchMode => exit 2";; X4) d="--from b with no finished merge";; X5) d="no pod address/port/key/token in the package";;
  esac
  scn "$id" "$d"
done
echo "SCENARIOS: $PASSN passed, $FAILN failed  (details: $RES)"
[ "$FAILN" = 0 ]
