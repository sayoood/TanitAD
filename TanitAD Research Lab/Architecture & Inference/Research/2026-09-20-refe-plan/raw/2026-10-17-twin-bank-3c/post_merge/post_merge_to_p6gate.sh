#!/usr/bin/env bash
# post_merge_to_p6gate.sh -- from the 3-C bank merge to P6's launch-gate DRY RUN.   Dev-box (Git bash) script that drives the pod over $SSH.
# Written 2026-10-17 by the ops-build agent.  Scope: HANDOFF.md steps 4-8 of the 3-C package + P6_POD_RUNBOOK.md section 3 ("Dry run: P6_STOP_AFTER=launch_gate").
# Documentation, cost estimates, what each FAIL means, rehearsal evidence: POST_MERGE.md next to this file.
#
#   (a) wait for the merge to finish, judge it from its ARTIFACTS: rc 0 only; rc 4 (yield guard) and rc 3 (refused) STOP        exit 4 / 3 ; 5 = any other merge failure
#   (b) (c'2) no-slowdown check, as committed (HANDOFF step 6); a failing rank = 3C-SLOW, STOP                                  exit 6 (3C-SLOW) / 7 (invalid or incomplete)
#   (c) (c'3) D7 verify (a),(b) judged, (c) reported + `reroll_twins.py check` must print ZZTWIN_HOLD_OK (HANDOFF step 7)        exit 8
#   (d) step 8: hard-link (else atomic copy) the KEPT twins' parts into metric_cache; 100 % coverage or STOP                       exit 9
#   (e) P6's launch-gate DRY RUN (P6_STOP_AFTER=launch_gate): mints and VERIFIES a token, stops BEFORE train.py.  PI decision 2026-10-17: P6 gets the A40 BEFORE the
#       safety arm's Stage 2, so if the pod is busy (a GPU compute process, another chain's lock, or a running refe/rls1_pod.sh = R0d) it WAITS (poll 5 min, cap 12 h)
#       and never starts the gate while R0d trains; at the cap it prints `ZZPM GATE_DEFERRED gpu_busy_wait_cap_12h` (or, with --no-wait, `GATE_DEFERRED gpu_busy`)  exit 10 / 12 (FAIL)
#   (f) AUTO-LAUNCH (PI decision 2026-10-17): on a gate PASS, re-check the pod is free, launch P6's REAL run (p6_pod.sh without P6_STOP_AFTER, detached, as the runbook says),
#       and confirm training started (chain.log 'train.py preflight OK' + a first step/loss line in metrics.jsonl within ~10 min) -> `ZZPM P6_TRAINING`.      exit 13 (FAIL)
#       FAIL-CLOSED: any REFUSED/FAIL line, a missing PASS line, an ambiguous state, a deferral -> `ZZPM P6_NOT_LAUNCHED <why>` and a non-zero exit.
#   Every step prints `ZZPM <step> OK|FAIL ...`; the script exits non-zero on the first FAIL.  `ZZPM DONE ...` is always the last line.   exit 2 = usage / pod unreachable / pre-flight
#   This script NEVER kills anything on the pod and never uses pkill/pgrep -f.
#
# usage:  SSH='<full ssh command to the pod>' [env] bash post_merge_to_p6gate.sh [--probe] [--dry] [--from a|b|c|d|e|f] [--to a|b|c|d|e|f] [--no-launch] [--no-wait] [--wait-max-h N]
#   --probe   READ-ONLY inventory of every pre-condition against the real pod; writes nothing, starts nothing; exits.
#   --dry     every pod WRITE / launch is printed (`ZZPM_DRY ...`) instead of executed; reads still run; a step that needs a missing merge output is skipped with DRY_SKIP.
#   --no-launch  stop after the gate dry run (same as --to e): the token is minted and verified, P6 is NOT launched (prints P6_LAUNCH_SKIPPED).
#   --no-wait    do not wait for a busy pod: print `ZZPM GATE_DEFERRED gpu_busy` and exit 10 at once (the pre-2026-10-17 behaviour).
# env (nothing below is stored in this file; the repo is PUBLIC):
#   SSH                    REQUIRED. The ssh command incl. host, port and key, e.g. "ssh -o BatchMode=yes -o ConnectTimeout=25 -i <key> -p <port> <user>@<host>"; or P6N_SSH_ENV=<file that sets SSH>
#   P6N_SHIP_COMMIT        REQUIRED for (e): the 40-hex commit of the P6 tree you shipped to the pod (must equal its SHIP_MANIFEST.json commit)
#   P6N_HF_TOKEN_FILE      REQUIRED for (e): path ON THE POD of the file holding the write-scoped HF token (read by the pod, never printed)
#   P6N_DD_GB              optional, default = P6_DD_GB in the pod's p6_start.env, else 16: the chain's disk write test
#   P6N_P6POD_MD5          optional: expected md5 (first 12) of the shipped refe/p6_pod.sh
#   P6N_WAIT_MAX_H (16)  P6N_C2_MAX_S (7200)  P6N_C3_MAX_S (14400)  P6N_PARTS_MAX_S (3600)  P6N_GATE_MAX_S (14400)  P6N_POLL_S (300)  P6N_PARTS_HOW (auto|link|copy)
#   P6N_GPU_WAIT_MAX_H (12)  P6N_IDLE_POLL_S (300)  P6N_IDLE_RECHECK_S (30)  P6N_LAUNCH_MAX_S (1800)  P6N_FIRST_STEP_S (600)  P6N_LAUNCH_POLL_S (30)
#   P6N_ROOT (/workspace)  P6N_LOGDIR (./runs/<utc>)  P6N_CODE_HASH_FILES  P6N_SKIP_CODE_HASH=1
#   P6N_ALLOW_ANY_SSH=1    for the offline rehearsal harness only (skips the BatchMode check on $SSH)
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKGDIR="$(cd "$HERE/.." && pwd)"                                   # the 3-C package: PREREG_3C.sha256, MANIFEST.json
HELPERS="$HERE/pod/pod_helpers.py"
# a LOCAL python that works (on a Windows dev box python3 can be the Store stub), and a path form it can open (Git bash /c/... -> C:/...)
LPY=""; for _p in python3 python py; do if command -v "$_p" > /dev/null 2>&1 && "$_p" -c 'import json,sys' > /dev/null 2>&1; then LPY="$_p"; break; fi; done
lpath() { if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

# ---- $SSH: from the environment, NEVER from this file ---------------------------------------------------------------------------------------------------------
if [ -z "${SSH:-}" ] && [ -n "${P6N_SSH_ENV:-}" ] && [ -f "$P6N_SSH_ENV" ]; then . "$P6N_SSH_ENV"; fi

DRY=0; PROBE=0; FROM=a; TO=f; NOWAIT=0; WAIT_H="${P6N_WAIT_MAX_H:-16}"
usage() { sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=1 ;; --probe) PROBE=1 ;; --no-launch) TO=e ;; --no-wait) NOWAIT=1 ;;
    --from) FROM="${2:-}"; shift ;; --to) TO="${2:-}"; shift ;; --wait-max-h) WAIT_H="${2:-16}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ZZPM 0 FAIL unknown argument: $1"; exit 2 ;;
  esac; shift
done

# ---- output ---------------------------------------------------------------------------------------------------------------------------------------------------
LOGD="${P6N_LOGDIR:-$HERE/runs/$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$LOGD"
LAST_OK="-"
# builtins only (no `| tee`): Ctrl-C or `timeout` signals the whole process group, and a tee child killed that way would swallow the very FAIL line the EXIT trap is about to print
say()  { printf 'ZZPM %s\n' "$*"; printf 'ZZPM %s\n' "$*" >> "$LOGD/transcript.log"; }
info() { printf 'ZZPM_INFO %s\n' "$*"; printf 'ZZPM_INFO %s\n' "$*" >> "$LOGD/transcript.log"; }
dry()  { printf 'ZZPM_DRY %s\n' "$*"; printf 'ZZPM_DRY %s\n' "$*" >> "$LOGD/transcript.log"; }
step_ok() { local st="$1"; shift; LAST_OK="$st"; say "$st OK $*"; }
LAUNCHED=0                 # 1 only once step (f) has started the real P6 run
not_launched() {           # armed only when the auto-launch step is part of this run.  After the launch statement ran, "not launched" would be a lie: say UNCONFIRMED.
  [ "$TO" = f ] || return 0
  if [ "$LAUNCHED" = 0 ]; then say "P6_NOT_LAUNCHED $*"
  else say "P6_LAUNCH_UNCONFIRMED $* -- the launch statement RAN; the P6 chain may still be alive on the pod (this script never kills anything): inspect chain.log before acting"; fi
  return 0
}
FINISHED=0                 # set to 1 by every deliberate exit; the EXIT trap below turns any OTHER exit (an unbound variable, an arithmetic error, a signal) into a FAIL
finish_fail() {            # finish_fail <step> <exit code> <message...>
  local st="$1" code="$2"; shift 2
  FINISHED=1; say "$st FAIL $*"; not_launched "$st: $*"; say "DONE result=FAIL step=$st exit=$code last_ok=$LAST_OK dry=$DRY logs=$LOGD"; exit "$code"
}
finish_deferred() { FINISHED=1; say "GATE_DEFERRED $1"; not_launched "gate_deferred: $1"; say "DONE result=DEFERRED through=$LAST_OK exit=10 reason=$1 dry=$DRY logs=$LOGD"; exit 10; }
on_exit() {                # fail-closed: reaching here with FINISHED=0 means the script died on something nobody handled -- never read that as success
  local rc=$?
  trap '' TERM INT HUP QUIT                          # `timeout` and Ctrl-C signal the whole group, sometimes twice: the final lines below must not be cut by a second signal
  if [ "$FINISHED" = 0 ] && [ "${PROBE:-0}" = 0 ]; then
    FINISHED=1
    say "${LAST_OK:--} FAIL internal error: the script exited unexpectedly (rc=$rc) after step ${LAST_OK:--} OK -- nothing was judged; treat as FAIL"
    not_launched "internal_error_after_step_${LAST_OK:--}"
    say "DONE result=FAIL step=internal exit=14 last_ok=$LAST_OK dry=$DRY logs=$LOGD"
    exit 14
  fi
}
trap on_exit EXIT
trap 'exit 143' TERM; trap 'exit 130' INT; trap 'exit 129' HUP
kv() { local line="$1" key="$2"; if [[ " $line " =~ \ $key=([^\ ]*) ]]; then printf '%s' "${BASH_REMATCH[1]}"; fi; }

# ---- every numeric knob must be a number (a malformed one would turn a timeout test into an error that evaluates false = wait forever) ---------------------------------
for _kn in P6N_C2_MAX_S P6N_C3_MAX_S P6N_PARTS_MAX_S P6N_GATE_MAX_S P6N_POLL_S P6N_IDLE_POLL_S P6N_IDLE_RECHECK_S P6N_LAUNCH_MAX_S P6N_FIRST_STEP_S P6N_LAUNCH_POLL_S P6N_MERGE_GRACE_S P6N_GATE_POLL_S P6N_RETRY_S P6N_SSH_TIMEOUT; do
  _kv="${!_kn:-}"; [ -z "$_kv" ] || [[ "$_kv" =~ ^[0-9]+$ ]] || finish_fail 0 2 "$_kn=$_kv is not a whole number of seconds"
done
for _kn in P6N_GPU_WAIT_MAX_H P6N_WAIT_MAX_H; do
  _kv="${!_kn:-}"; [ -z "$_kv" ] || [[ "$_kv" =~ ^[0-9]+([.][0-9]+)?$ ]] || finish_fail 0 2 "$_kn=$_kv is not a number of hours"
done
[[ "$WAIT_H" =~ ^[0-9]+([.][0-9]+)?$ ]] || finish_fail 0 2 "--wait-max-h $WAIT_H is not a number of hours"
WAIT_S="$(awk -v h="$WAIT_H" 'BEGIN{printf "%d", h*3600}')"

# ---- pod access (fail-closed: an ssh that cannot answer is never read as "not yet" or "fine") --------------------------------------------------------------------
if [ -z "${SSH:-}" ]; then finish_fail 0 2 "\$SSH is not set (give the ssh command in the environment, or P6N_SSH_ENV=<file>); it is never stored in this file"; fi
if [ "${P6N_ALLOW_ANY_SSH:-0}" != 1 ]; then
  case "$SSH" in *BatchMode=yes*) ;; *) finish_fail 0 2 "\$SSH must contain -o BatchMode=yes (a password prompt would hang this script)" ;; esac
fi
[ -f "$HELPERS" ] || finish_fail 0 2 "missing $HELPERS"
PREREG_SHA="$(tr -d '[:space:]' < "$PKGDIR/PREREG_3C.sha256" 2>/dev/null | cut -c1-64)"
[ ${#PREREG_SHA} = 64 ] || finish_fail 0 2 "cannot read the frozen PREREG_3C sha256 from $PKGDIR/PREREG_3C.sha256"

ROOT="${P6N_ROOT:-/workspace}"; OPS="$ROOT/ops"; DATA="$ROOT/data"
F="$OPS/bank3c/full"; ENV3C="$OPS/bank3c/code/env_3c.sh"
IN="$DATA/refe_navtrain/train_grow_cmd"; OUT="$DATA/refe_navtrain/train_grow_cmd_3c"
MC_TWIN="$DATA/refe_navtrain/metric_cache_twins_3c"; MC="$DATA/refe_navtrain/metric_cache"
P6="$DATA/refe_p6"; P6PKG="$ROOT/refe-plan-p6"; P6N="$OPS/p6next"; RUN="$DATA/refe_runs/p6_paper_faithful"
POLL="${P6N_POLL_S:-300}"; GATE_JUDGED=0

pod() {                    # pod "<remote command>"  ->  POD_OUT (stdout), POD_RC.  Retries only a dead connection (255) or a timeout (124).
  local tries=0
  while :; do
    POD_OUT="$(timeout "${P6N_SSH_TIMEOUT:-900}" $SSH "$1" < /dev/null 2>>"$LOGD/ssh_stderr.log")"; POD_RC=$?
    if [ "$POD_RC" -ne 255 ] && [ "$POD_RC" -ne 124 ]; then return 0; fi
    tries=$((tries + 1)); [ "$tries" -ge 3 ] && return 1
    sleep "${P6N_RETRY_S:-20}"
  done
}
podh() {                   # podh <subcommand args...>: run a pod_helpers.py subcommand with the pod's python3 over stdin (no file written on the pod); ZV = its ZZV lines
  local tries=0
  while :; do
    POD_OUT="$(timeout "${P6N_SSH_TIMEOUT:-1800}" $SSH "PYTHONDONTWRITEBYTECODE=1 python3 - $*" < "$HELPERS" 2>>"$LOGD/ssh_stderr.log")"; POD_RC=$?
    if [ "$POD_RC" -ne 255 ] && [ "$POD_RC" -ne 124 ]; then break; fi
    tries=$((tries + 1)); [ "$tries" -ge 3 ] && return 1
    sleep "${P6N_RETRY_S:-20}"
  done
  ZV="$(printf '%s\n' "$POD_OUT" | grep '^ZZV ' || true)"
  printf '%s\n' "$POD_OUT" >> "$LOGD/pod_helpers.log"
  return 0
}
podhpy() {                 # like podh, but with the 3-C env sourced and the teacher venv python ($PY: numpy, the P6 tree's online_labeller)
  local tries=0
  while :; do
    POD_OUT="$(timeout "${P6N_SSH_TIMEOUT:-1800}" $SSH "source $ENV3C; PYTHONDONTWRITEBYTECODE=1 \$PY - $*" < "$HELPERS" 2>>"$LOGD/ssh_stderr.log")"; POD_RC=$?
    if [ "$POD_RC" -ne 255 ] && [ "$POD_RC" -ne 124 ]; then break; fi
    tries=$((tries + 1)); [ "$tries" -ge 3 ] && return 1
    sleep "${P6N_RETRY_S:-20}"
  done
  ZV="$(printf '%s\n' "$POD_OUT" | grep '^ZZV ' || true)"
  printf '%s\n' "$POD_OUT" >> "$LOGD/pod_helpers.log"
  return 0
}
zv() { printf '%s\n' "$ZV" | grep "^ZZV $1 " | tail -n 1; }            # the last verdict line of a kind
pod_has() { pod "test -e '$1'; echo ZZD\$?ZZ" || return 2; case "$POD_OUT" in *ZZD0ZZ*) return 0 ;; *ZZD1ZZ*) return 1 ;; *) return 2 ;; esac; }
podw() {                   # a pod WRITE / launch: printed, not executed, under --dry
  if [ "$DRY" = 1 ]; then local c="$1"; [ -n "${P6N_HF_TOKEN_FILE:-}" ] && c="${c//${P6N_HF_TOKEN_FILE}/<token-file>}"; dry "pod: $c"; POD_OUT=""; POD_RC=0; return 0; fi
  pod "$1"
}
ship() {                   # ship <local file> <pod name under $P6N>  (a WRITE inside the scratch dir only)
  if [ "$DRY" = 1 ]; then dry "ship $1 -> $P6N/$2"; return 0; fi
  timeout 300 $SSH "mkdir -p $P6N && cat > $P6N/$2" < "$1" 2>>"$LOGD/ssh_stderr.log"
}
pod_epoch() { pod "date +%s" && printf '%s' "$POD_OUT"; }
gpu_note() { podh gpu || return 0; local l; l="$(zv gpu)"; info "$1 gpu: ${l#ZZV gpu }"; }
detached_wait() {          # detached_wait <label> <step> <exit code if dead/timeout> <finished-marker> <pid file> <max s>
  local label="$1" st="$2" code="$3" mark="$4" pidf="$5" max="$6" t0 rc nfail=0
  t0="$(date +%s)"
  while :; do
    pod_has "$mark"; rc=$?
    if [ "$rc" = 0 ]; then return 0; fi
    if [ "$rc" = 2 ]; then nfail=$((nfail + 1)); [ "$nfail" -ge 5 ] && finish_fail "$st" 2 "$label: pod unreachable 5x while waiting"; else nfail=0; fi
    if [ "$rc" = 1 ] && [ $(( $(date +%s) - t0 )) -gt 60 ]; then               # grace for the job to write its pid
      POD_OUT=""
      if pod "if [ -f '$pidf' ]; then kill -0 \$(cat '$pidf') 2>/dev/null; echo ZZL\$?ZZ; else echo ZZL9ZZ; fi"; then
        case "$POD_OUT" in
          *ZZL1ZZ*|*ZZL9ZZ*) pod_has "$mark" && return 0
                             finish_fail "$st" "$code" "$label: the detached pod job is gone and left no finished-marker ($pidf) -- OOM kill or crash; see $LOGD and the job's logs on the pod" ;;
        esac
      fi
    fi
    if [ $(( $(date +%s) - t0 )) -gt "$max" ]; then finish_fail "$st" "$code" "$label: not finished after ${max}s (the pod job may still be running -- NOT killed by this script)"; fi
    sleep "$POLL"
  done
}

# ---- state shared between steps -----------------------------------------------------------------------------------------------------------------------------
KEPT="?"; K1="?"; K2="?"; K3="?"; OUT_SHA=""; KS="1 2 3"
load_facts() {             # load_facts <full|quick> : the merge verdict -> KEPT/K*/OUT_SHA. Returns 0 iff state=ok
  podh merge --f "$F" --out "$OUT" --inp "$IN" --prereg-sha "$PREREG_SHA" $([ "$1" = quick ] && echo --quick) || return 2
  local l; l="$(zv merge)"; MERGE_LINE="$l"
  [ "$(kv "$l" state)" = ok ] || return 1
  KEPT="$(kv "$l" kept)"; K1="$(kv "$l" kept_1)"; K2="$(kv "$l" kept_2)"; K3="$(kv "$l" kept_3)"; OUT_SHA="$(kv "$l" out_sha)"
  KS=""; for k in 1 2 3; do v="K$k"; [ "${!v}" -gt 0 ] 2>/dev/null && KS="$KS $k"; done; KS="${KS# }"
  return 0
}

# =================================================================================================================================================================
# step 0 -- pre-flight: pod reachable, the frozen 3-C code unchanged, the paths the later steps need exist.  READ-ONLY.
# =================================================================================================================================================================
preflight() {
  pod "echo ZZR\$((6*7))ZZ" || finish_fail 0 2 "pod unreachable (ssh failed 3x)"
  case "$POD_OUT" in *ZZR42ZZ*) ;; *) finish_fail 0 2 "pod answered but not as expected (got: ${POD_OUT:0:80})" ;; esac
  if [ "${P6N_SKIP_CODE_HASH:-0}" != 1 ]; then
    [ -n "$LPY" ] || finish_fail 0 2 "no working local python (python3/python/py) to read MANIFEST.json"
    local files="${P6N_CODE_HASH_FILES:-reroll_bank_3c.py reroll_twins.py cprime_check.py route_lane_rank_patch.py build_teacher_rollouts.py env_3c.sh watch_3c.py}" bad="" f exp got
    pod "cd $OPS/bank3c/code && sha256sum $files" || finish_fail 0 2 "cannot hash the pod's 3-C code"
    for f in $files; do
      exp="$($LPY - "$(lpath "$PKGDIR/MANIFEST.json")" "code/$f" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))["files"].get(sys.argv[2], {})
print(m.get("sha256", "NONE"))
PY
)"
      got="$(printf '%s\n' "$POD_OUT" | awk -v f="$f" '$2 == f {print $1}')"
      [ -n "$exp" ] && [ "$exp" = "$got" ] || bad="$bad $f"
    done
    [ -z "$bad" ] || finish_fail 0 2 "the pod's frozen 3-C code differs from MANIFEST.json (or is missing):$bad  (shard_meta pins it; do not change it before the merge is consumed)"
  fi
  pod "source $ENV3C && for p in \"\$PKG/refe/diag_d7_twin_route.py\" \"\$CODE3C/cprime_check.py\" \"\$CODE3C/reroll_twins.py\" \"$P6PKG/refe/p6_prep.py\" \"$P6PKG/refe/online_labeller.py\" \"$MC_TWIN\" \"$MC\" \"$IN/targets_rank1.jsonl\"; do [ -e \"\$p\" ] || echo MISSING:\$p; done; echo ZZE\$?ZZ" \
    || finish_fail 0 2 "cannot check the pod's paths"
  case "$POD_OUT" in *MISSING:*) finish_fail 0 2 "pod paths missing: $(printf '%s\n' "$POD_OUT" | grep MISSING: | tr '\n' ' ' | cut -c1-300)" ;; esac
  step_ok 0 "pod reachable; frozen 3-C code == MANIFEST.json; pod paths present"
}

# =================================================================================================================================================================
# (a) the merge -- wait, then judge from artifacts (the watcher's rc is not saved anywhere on the pod: merge.out holds the python stdout only)
# =================================================================================================================================================================
step_a() {
  local t0 st="" first_ns="" first_mg="" nfail=0 n=0 prog="" state l
  t0="$(date +%s)"
  while :; do
    load_facts full; local rc=$?
    if [ "$rc" = 2 ]; then nfail=$((nfail + 1)); [ "$nfail" -ge 5 ] && finish_fail a 2 "pod unreachable 5x while waiting for the merge"; sleep 60; continue; fi
    nfail=0
    l="${MERGE_LINE:-}"; state="$(kv "$l" state)"
    case "$state" in
      ok)
        if [ "$(kv "$l" k1_share_all)" != None ] && awk -v x="$(kv "$l" k1_share_all)" 'BEGIN{exit !(x+0 < 0.20)}'; then
          finish_fail a 5 "merge rc 0 but the K=1 keep share over ALL input twins is $(kv "$l" k1_share_all) < 0.20 while the evaluated basis holds: the two denominators disagree -- Master Mind decides (addendum 1 item 4)"
        fi
        step_ok a "merge rc 0 (judged from artifacts, input+output re-hashed): kept=$KEPT (K1=$K1 K2=$K2 K3=$K3) dropped=$(kv "$l" dropped) K1_share_eval=$(kv "$l" k1_share_eval) all=$(kv "$l" k1_share_all) shares[$(kv "$l" shares_eval)] out_sha=${OUT_SHA:0:16}"
        info "a expected keep shares K1/K2/K3 = 0.37/0.29/0.22 (mini bank, PREREG_3C); yield guard: K1 < 0.20 STOPs"
        return 0 ;;
      yield)   finish_fail a 4 "YIELD GUARD TRIPPED (rc 4): hand NOTHING to P6; the bank is renamed .REJECTED_YIELD_GUARD; report to the Master Mind. $(kv "$l" why)" ;;
      refused) finish_fail a 3 "MERGE REFUSED (rc 3): $(kv "$l" why)" ;;
      run_incomplete) finish_fail a 5 "RUN_INCOMPLETE: shards unfinished -- re-run \`run_full.sh launch\` on the pod to resume, then the merge. $(kv "$l" why)" ;;
      invalid) finish_fail a 5 "merge.out says OK but its artifacts do not hold up: $(kv "$l" why)" ;;
      crashed) finish_fail a 5 "merge.out has no terminal marker and no merge process is running (crash): $(kv "$l" why)" ;;
      not_started)
        [ -n "$first_ns" ] || first_ns="$(date +%s)"
        [ $(( $(date +%s) - first_ns )) -gt "${P6N_MERGE_GRACE_S:-1500}" ] && finish_fail a 5 "RUN_DONE exists but no merge.out after ${P6N_MERGE_GRACE_S:-1500}s: the watcher did not run the merge -- run reroll_bank_3c.py merge (HANDOFF step 4) by hand" ;;
      merging)
        [ -n "$first_mg" ] || first_mg="$(date +%s)"
        [ $(( $(date +%s) - first_mg )) -gt 3600 ] && finish_fail a 5 "the merge process has been running for over an hour" ;;
      run_not_done) : ;;
      *) finish_fail a 5 "unreadable merge verdict: ${l:-<none>}" ;;
    esac
    if [ "$DRY" = 1 ]; then say "a DRY_SKIP merge state=$state (dry run: not waiting); later steps print their plans with KEPT unknown"; OUT_SHA="DRYRUN"; LAST_OK=a; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt "$WAIT_S" ]; then finish_fail a 5 "timeout: no finished merge after ${WAIT_H} h (state=$state)"; fi
    n=$((n + 1))
    if [ $((n % 6)) = 1 ] || [ "$state" != "$st" ]; then
      pod "tail -n 1 $F/progress.log 2>/dev/null | cut -c1-330" && prog="$POD_OUT"
      info "a waiting state=$state progress: ${prog:-<none>}"
    fi
    st="$state"; sleep "$POLL"
  done
}

# =================================================================================================================================================================
# (b) (c'2) -- HANDOFF step 6 verbatim, judged as committed: per rank n >= 5, median path ratio >= 0.95 AND median end-speed change >= -0.3 m/s
# =================================================================================================================================================================
step_b() {
  gpu_note b
  local l oc
  if [ "$DRY" = 1 ] && [ "$OUT_SHA" = DRYRUN ]; then
    dry "ship pod/c2_run.sh; launch detached: ENV3C=.. F=.. OUT=.. IN=.. EXPECT_SHA=<merged sha> setsid -f nohup bash $P6N/c2_run.sh > $P6N/c2_run.out 2>&1 < /dev/null"
    step_ok b "OK_DRY (planned: sample <=300/rank seed 20261017, 12 shards, analyze, judge)"; return 0
  fi
  pod "cat $F/c2/merged_sha.txt 2>/dev/null; echo; test -f $F/c2/C2_FINISHED; echo ZZF\$?ZZ" || finish_fail b 2 "pod unreachable"
  local prev; prev="$(printf '%s\n' "$POD_OUT" | head -n 1 | tr -d '[:space:]')"
  if [ -n "$prev" ] && [ "$prev" != "$OUT_SHA" ]; then finish_fail b 7 "$F/c2 holds a (c'2) run for ANOTHER merged bank ($prev != $OUT_SHA): move it aside (this script never deletes)"; fi
  if [ -n "$prev" ] && [[ "$POD_OUT" == *ZZF0ZZ* ]]; then
    info "b reuse: a finished (c'2) run for this exact merged bank exists; judging it"
  else
    ship "$HERE/pod/c2_run.sh" c2_run.sh || finish_fail b 2 "could not ship c2_run.sh"
    podw "ENV3C=$ENV3C F=$F OUT=$OUT IN=$IN EXPECT_SHA=$OUT_SHA setsid -f nohup bash $P6N/c2_run.sh > $P6N/c2_run.out 2>&1 < /dev/null" || finish_fail b 2 "could not launch c2_run.sh"
    info "b launched (c'2) detached on the pod: 12 shards, CPU, <=300 twins/rank; polling every ${POLL}s"
    if [ "$DRY" = 1 ]; then step_ok b "OK_DRY (the launch above was printed, not executed; nothing to wait for or judge)"; return 0; fi
    detached_wait "(c'2)" b 7 "$F/c2/C2_FINISHED" "$F/c2/PID" "${P6N_C2_MAX_S:-7200}"
  fi
  podh c2 --f "$F" --out "$OUT" || finish_fail b 2 "pod unreachable"
  l="$(zv c2)"; oc="$(kv "$l" outcome)"
  info "b (c'2) K1[$(kv "$l" r1)] K2[$(kv "$l" r2)] K3[$(kv "$l" r3)]"
  case "$oc" in
    pass) step_ok b "(c'2) PASS: per judged rank median path ratio >= 0.95 and median end-speed change >= -0.3 m/s" ;;
    slow) finish_fail b 6 "3C-SLOW: (c'2) FAILS at scale -- $(kv "$l" why). Hand nothing to P6; next lever per PREREG_3C: h8 + commit filter. STOP and report." ;;
    incomplete) finish_fail b 7 "(c'2) INCOMPLETE: $(kv "$l" why) -- some re-rolls are missing; re-run this step (it resumes the shards) before reading any outcome" ;;
    stale) finish_fail b 7 "(c'2) STALE: $(kv "$l" why)" ;;
    *) finish_fail b 7 "(c'2) not judged: outcome=$oc $(kv "$l" why)" ;;
  esac
}

# =================================================================================================================================================================
# (c) (c'3) -- HANDOFF step 7 verbatim: D7 verify (a),(b) judged, (c) reported; reroll_twins.py check must print ZZTWIN_HOLD_OK
# =================================================================================================================================================================
step_c() {
  gpu_note c
  local c3ks="$KS" fin
  if [ "$DRY" = 1 ] && [ "$OUT_SHA" = DRYRUN ]; then
    dry "ship pod/c3_run.sh; launch detached: KS='1 2 3' (ranks with kept > 0 once the merge is known) ... diag_d7_twin_route.py verify --limit 1000 per rank, then reroll_twins.py check"
    step_ok c "OK_DRY (planned: verify per rank in parallel, then the hold check)"; return 0
  fi
  [ -n "$c3ks" ] || finish_fail c 8 "no rank has any kept twin"
  pod "cat $F/c3_merged_sha.txt 2>/dev/null; echo; test -f $F/C3_FINISHED; echo ZZF\$?ZZ" || finish_fail c 2 "pod unreachable"
  local prev; prev="$(printf '%s\n' "$POD_OUT" | head -n 1 | tr -d '[:space:]')"
  if [ -n "$prev" ] && [ "$prev" != "$OUT_SHA" ]; then finish_fail c 8 "$F/verify_k*.log belong to ANOTHER merged bank ($prev != $OUT_SHA): move them aside (this script never deletes)"; fi
  if [ -n "$prev" ] && [[ "$POD_OUT" == *ZZF0ZZ* ]]; then
    info "c reuse: a finished (c'3) run for this exact merged bank exists; judging it"
  else
    ship "$HERE/pod/c3_run.sh" c3_run.sh || finish_fail c 2 "could not ship c3_run.sh"
    podw "ENV3C=$ENV3C F=$F OUT=$OUT MC=$MC MC_TWIN=$MC_TWIN KS='$c3ks' EXPECT_SHA=$OUT_SHA setsid -f nohup bash $P6N/c3_run.sh > $P6N/c3_run.out 2>&1 < /dev/null" || finish_fail c 2 "could not launch c3_run.sh"
    info "c launched (c'3) detached on the pod: verify for ranks [$c3ks] in parallel (--limit 1000), then the hold check; polling every ${POLL}s"
    if [ "$DRY" = 1 ]; then step_ok c "OK_DRY (the launch above was printed, not executed; nothing to wait for or judge)"; return 0; fi
    detached_wait "(c'3)" c 8 "$F/C3_FINISHED" "$F/c3.PID" "${P6N_C3_MAX_S:-14400}"
  fi
  podh c3 --f "$F" --ks "${c3ks// /,}" || finish_fail c 2 "pod unreachable"
  local l bad=""
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    info "c ${l#ZZV }"
  done < <(printf '%s\n' "$ZV" | grep -E '^ZZV (c3|c3hold) ')
  for k in $c3ks; do
    l="$(printf '%s\n' "$ZV" | grep "^ZZV c3 k=$k " | tail -n 1)"
    if [ "$(kv "$l" state)" != judged ] || [ "$(kv "$l" a)" != True ] || [ "$(kv "$l" b)" != True ]; then bad="$bad K$k(a=$(kv "$l" a),b=$(kv "$l" b),state=$(kv "$l" state))"; fi
  done
  [ -z "$bad" ] || finish_fail c 8 "(c'3) FAILS: D7 (a) DISTINCT and/or (b) SPEC LANE not satisfied:$bad. (c) is reported, not judged. Hand nothing to P6."
  l="$(zv c3hold)"; [ "$(kv "$l" state)" = ok ] || finish_fail c 8 "reroll_twins.py check did not print ZZTWIN_HOLD_OK: $(kv "$l" detail)"
  step_ok c "(c'3) PASS: D7 (a),(b) hold for ranks [$c3ks]; (c) reported only; ZZTWIN_HOLD_OK $(kv "$l" detail)"
  say "c ZZTWIN_HOLD_OK $(kv "$l" detail)"
}

# =================================================================================================================================================================
# (d) step 8 -- the kept twins' parts into metric_cache.  DECISION (see POST_MERGE.md): LINK-OR-COPY from metric_cache_twins_3c, never a P6-side rebuild/skip.
# =================================================================================================================================================================
step_d() {
  local l how="${P6N_PARTS_HOW:-auto}" st
  if [ "$DRY" = 1 ] && [ "$OUT_SHA" = DRYRUN ]; then
    dry "podhpy parts --bank $OUT --src $MC_TWIN --dst $MC --mode plan  (read-only)"
    dry "ship pod/parts_run.sh + pod_helpers.py; launch detached: parts --mode apply --how $how --report $P6N/parts_copy_report.json"
    dry "P6's own D7 acceptance, as its lanes stage will run it: p6_prep.py lane-distinct --bank $OUT --cache $MC --k <K> --n 40 --min-frac 0.9"
    step_ok d "OK_DRY (planned)"; return 0
  fi
  podhpy parts --bank "$OUT" --src "$MC_TWIN" --dst "$MC" --p6refe "$P6PKG/refe" --mode plan || finish_fail d 2 "pod unreachable"
  l="$(zv parts)"; [ -n "$l" ] || finish_fail d 9 "parts plan produced no verdict: $(printf '%s\n' "$POD_OUT" | tail -n 3 | tr '\n' '|' | cut -c1-300)"
  info "d plan: ${l#ZZV parts }"
  [ "$(kv "$l" state)" = plan_ok ] || finish_fail d 9 "parts plan is not clean: missing_src=$(kv "$l" missing_src) missing_logged=$(kv "$l" missing_logged) conflict=$(kv "$l" conflict) first=$(kv "$l" first_bad). P6 must NOT rebuild twins: its shipped build_metric_cache.py lacks the PDM-side route-rank fix, so a rebuilt twin would sit on the LOGGED route."
  if [ "$(kv "$l" to_place)" != 0 ]; then
    ship "$HERE/pod/parts_run.sh" parts_run.sh && ship "$HELPERS" pod_helpers.py || finish_fail d 2 "could not ship the parts runner"
    podw "ENV3C=$ENV3C OUT=$OUT MC=$MC MC_TWIN=$MC_TWIN P6PKG=$P6PKG P6N=$P6N HOW=$how HELPERS=$P6N/pod_helpers.py setsid -f nohup bash $P6N/parts_run.sh > $P6N/parts_run.out 2>&1 < /dev/null" || finish_fail d 2 "could not launch parts_run.sh"
    info "d launched the parts placement (mode=$how, $(kv "$l" to_place) files, $(kv "$l" bytes_to_place) bytes if copied; ~0 bytes if hard-linked)"
    if [ "$DRY" = 1 ]; then step_ok d "OK_DRY (read-only plan above; the placement was printed, not executed)"; return 0; fi
    detached_wait "parts" d 9 "$P6N/PARTS_FINISHED" "$P6N/parts.PID" "${P6N_PARTS_MAX_S:-3600}"
    pod "cat $P6N/parts_apply.out | grep '^ZZV ' | tail -n 1; tail -n 3 $P6N/parts_apply.out | grep -v '^ZZV ' | head -n 3" || true
    st="$(printf '%s\n' "$POD_OUT" | grep '^ZZV parts ' | tail -n 1)"
    info "d apply: ${st#ZZV parts }"
    [ "$(kv "$st" state)" = ok ] || finish_fail d 9 "parts placement FAILED: $(kv "$st" err_first) errors=$(kv "$st" errors) dst_missing_after=$(kv "$st" dst_missing_after) (atomic per file, so re-running resumes)"
  else
    info "d nothing to place: every kept twin's part is already in metric_cache"
  fi
  # re-plan from the filesystem: the proof is the destination, not the loop's counters
  podhpy parts --bank "$OUT" --src "$MC_TWIN" --dst "$MC" --p6refe "$P6PKG/refe" --mode plan || finish_fail d 2 "pod unreachable"
  l="$(zv parts)"
  { [ "$(kv "$l" state)" = plan_ok ] && [ "$(kv "$l" to_place)" = 0 ] && [ "$(kv "$l" missing_src)" = 0 ] && [ "$(kv "$l" conflict)" = 0 ]; } || finish_fail d 9 "after placement the plan is not clean: ${l#ZZV parts }"
  local distinct; distinct="$(kv "$l" distinct)"
  [ "$KEPT" = "?" ] || [ "$distinct" = "$KEPT" ] || finish_fail d 9 "distinct kept twins in the bank ($distinct) != merge_report kept ($KEPT)"
  # P6's own D7 acceptance, exactly as its lanes stage runs it ($PY p6_prep.py lane-distinct ... from $P6PKG/refe), BEFORE the long gate run
  local k
  for k in $KS; do
    pod "source $ENV3C; cd $P6PKG/refe && env -u HF_TOKEN PYTHONDONTWRITEBYTECODE=1 \$PY p6_prep.py lane-distinct --bank $OUT --cache $MC --k $k --n 40 --min-frac 0.9 2>&1 | tail -n 1" || finish_fail d 2 "pod unreachable"
    case "$POD_OUT" in
      *ZZD7_DISTINCT_OK*) info "d P6's D7 acceptance K=$k: ${POD_OUT:0:140}" ;;
      *) finish_fail d 9 "P6's own D7 acceptance for K=$k FAILS on the placed parts (${POD_OUT:0:200}): the gate would refuse at its lanes stage" ;;
    esac
  done
  step_ok d "kept twins' parts placed ($(kv "$l" distinct) distinct, mode=$how, present=$(( $(kv "$l" present_same_inode) + $(kv "$l" present_identical) )), 0 missing, 0 conflicts) and P6's D7 acceptance passes for ranks [$KS]"
}

# =================================================================================================================================================================
# wait_idle <step> -- block until the pod is FREE for P6, or give up.   PI decision 2026-10-17: P6 gets the A40 BEFORE the safety arm's Stage 2.
# FREE = (1) nvidia-smi answers and lists NO compute process, AND (2) no other chain holds its chain.lock, AND (3) no `rls1_pod.sh` (R0d) process is alive.
# It must read FREE on two consecutive polls P6N_IDLE_RECHECK_S apart.  Poll every P6N_IDLE_POLL_S (300), cap P6N_GPU_WAIT_MAX_H (12) -> finish_deferred.
# The P6 chain's OWN lock being held is not something to wait for (someone started P6): FAIL.   --no-wait: defer at once (the old behaviour).   --dry: one read, no wait.
# =================================================================================================================================================================
wait_idle() {
  local st="$1" t0 cap idle=0 n=0 nfail=0 l state why poll recheck
  t0="$(date +%s)"; cap="$(awk -v h="${P6N_GPU_WAIT_MAX_H:-12}" 'BEGIN{printf "%d", h*3600}')"
  poll="${P6N_IDLE_POLL_S:-300}"; recheck="${P6N_IDLE_RECHECK_S:-30}"
  while :; do
    if ! podh busy --data "$DATA"; then nfail=$((nfail + 1)); [ "$nfail" -ge 5 ] && finish_fail "$st" 2 "pod unreachable 5x while waiting for the pod to be free"; sleep 30; continue; fi
    nfail=0
    l="$(zv busy)"; state="$(kv "$l" state)"; why="$(kv "$l" why)"
    if [ -z "$l" ]; then finish_fail "$st" 2 "the pod's busy-check produced no verdict: $(printf '%s\n' "$POD_OUT" | tail -n 2 | tr '\n' '|' | cut -c1-200)"; fi
    if [ "$(kv "$l" p6_held)" = 1 ]; then finish_fail "$st" 12 "the P6 chain at $P6 already holds its chain.lock: someone started P6; this script does not touch it"; fi
    if [ "$state" = idle ]; then
      idle=$((idle + 1))
      if [ "$idle" -ge 2 ] || [ "$DRY" = 1 ]; then info "$st pod is FREE (no GPU compute process, no other chain lock, no rls1_pod.sh)"; return 0; fi
      sleep "$recheck"; continue
    fi
    idle=0
    if [ "$DRY" = 1 ]; then info "$st (dry) pod is BUSY: $why -- a real run would wait here (poll ${poll}s, cap ${P6N_GPU_WAIT_MAX_H:-12} h)"; return 0; fi
    if [ "$NOWAIT" = 1 ]; then info "$st pod is BUSY: $why"; finish_deferred "gpu_busy"; fi
    if [ $(( $(date +%s) - t0 )) -ge "$cap" ]; then info "$st pod still BUSY after the cap: $why"; finish_deferred "gpu_busy_wait_cap_${P6N_GPU_WAIT_MAX_H:-12}h"; fi
    n=$((n + 1)); info "$st WAITING for a free pod ($(( ($(date +%s) - t0) / 60 )) min so far, cap ${P6N_GPU_WAIT_MAX_H:-12} h): $why"
    sleep "$poll"
  done
}

# =================================================================================================================================================================
# (e) P6's launch-gate DRY RUN
# =================================================================================================================================================================
step_e() {
  local l offset dd rc t0 nfail=0 refused
  # ---- e.0 pre-conditions that need no deferral to read: P6N_* settings, the shipped P6 tree, the stage markers.  A missing one is a FAIL (exit 12), not a deferral:
  #          the pipeline could not go on even with an idle GPU, and the operator learns it at once.
  [ -n "${P6N_SHIP_COMMIT:-}" ] || finish_fail e 12 "P6N_SHIP_COMMIT is not set (the 40-hex commit of the P6 tree shipped to the pod); refusing to run the gate on an unidentified code tree"
  [ -n "${P6N_HF_TOKEN_FILE:-}" ] || finish_fail e 12 "P6N_HF_TOKEN_FILE is not set (a path ON THE POD; the chain needs HF_TOKEN for its watcher and G-EGR)"
  podh p6tree --pkg "$P6PKG" || finish_fail e 2 "pod unreachable"
  l="$(zv p6tree)"; info "e p6tree: ${l#ZZV p6tree }"
  [ "$(kv "$l" state)" = ok ] || finish_fail e 12 "no shipped P6 tree at $P6PKG"
  [ "$(kv "$l" commit)" = "$P6N_SHIP_COMMIT" ] || finish_fail e 12 "the pod's P6 tree is commit $(kv "$l" commit), not $P6N_SHIP_COMMIT: re-ship (ship-manifest + scp, P6_POD_RUNBOOK section 2) -- the pre-R42/R43/R44 tree reproduces the 2026-10-16 18:50Z refusal"
  [ "$(kv "$l" cbank_knob)" = 1 ] || finish_fail e 12 "the shipped p6_pod.sh has no P6_CBANK knob: its CBANK is hard-wired to train_grow_cmd, so the gate would vouch for the PRE-3C bank. Apply p6_cbank_knob.diff (or the Master Mind's alternative), re-ship, retry"
  [ "$(kv "$l" cbank_commands_guard)" = 1 ] || finish_fail e 12 "the shipped p6_pod.sh lacks the commands-stage guard that stops P6_CBANK from being rm -rf'd"
  [ "$(kv "$l" r43_expandable_segments)" = 1 ] || finish_fail e 12 "the shipped p6_pod.sh lacks the R43 PYTORCH_CUDA_ALLOC_CONF line"
  [ "$(kv "$l" stop_after_launch_gate)" = 1 ] || finish_fail e 12 "the shipped p6_pod.sh has no P6_STOP_AFTER=launch_gate stop point"
  if [ -n "${P6N_P6POD_MD5:-}" ] && [ "$(kv "$l" p6_pod_md5)" != "$P6N_P6POD_MD5" ]; then finish_fail e 12 "p6_pod.sh md5 $(kv "$l" p6_pod_md5) != expected $P6N_P6POD_MD5"; fi
  pod "for s in commands cache; do test -f $P6/stage/\$s.ok || echo MISSING:\$s; done; test -s '$P6N_HF_TOKEN_FILE' || echo MISSING:tokenfile; echo ZZE0ZZ" || finish_fail e 2 "pod unreachable"
  case "$POD_OUT" in *MISSING:*) finish_fail e 12 "pre-condition missing on the pod: $(printf '%s\n' "$POD_OUT" | grep MISSING: | tr '\n' ' ')  (commands.ok / cache.ok: P6's tree cannot rebuild the bank or a twin part)" ;; esac
  # ---- e.1 WAIT for a free pod (PI 2026-10-17: P6 gets the A40 before the safety arm's Stage 2; R0d may be training now).  Never add load to a pod that is training.
  wait_idle e
  if [ "$DRY" = 1 ] && [ "$OUT_SHA" = DRYRUN ]; then
    dry "pod: rm -f $P6/stage/lanes.ok   (so the chain re-runs its lanes stage on the 3-C bank; this also invalidates lab/mem/egr/launch_gate)"
    dry "ship pod/gate_dryrun.sh; launch detached: P6PKG=.. P6=.. OUT=$OUT DD_GB=<p6_start.env P6_DD_GB, else 16> HF_TOKEN_FILE=<token-file> P6N=$P6N setsid -f nohup bash $P6N/gate_dryrun.sh"
    dry "wait for gate_dryrun.rc; judge: one PASS token newer than the start, binding --targets == $OUT, commit == P6N_SHIP_COMMIT, no part written during the run, chain.log has 'launch gate PASS (token verified)' and no 'D7 FALLBACK'"
    step_ok e "OK_DRY (planned; no pod write executed)"; return 0
  fi
  # ---- e.2 the bank is still the one steps a-d vetted, and its parts are in place
  load_facts full || finish_fail e 12 "the merged bank no longer judges OK at the gate's start: ${MERGE_LINE:-<none>}"
  podhpy parts --bank "$OUT" --src "$MC_TWIN" --dst "$MC" --p6refe "$P6PKG/refe" --mode plan || finish_fail e 2 "pod unreachable"
  l="$(zv parts)"
  if [ "$DRY" = 1 ] && [ "$(kv "$l" state)" = plan_ok ] && [ "$(kv "$l" to_place)" != 0 ]; then
    dry "step (d) was only PRINTED in this --dry run, so $(kv "$l" to_place) kept twin parts are still to be placed; a real run places them in (d) and (e) re-checks it: ${l#ZZV parts }"
  else
    { [ "$(kv "$l" state)" = plan_ok ] && [ "$(kv "$l" to_place)" = 0 ] && [ "$(kv "$l" distinct)" = "$KEPT" ]; } || finish_fail e 12 "step (d) is not in place (every kept twin's part must already be in metric_cache): ${l#ZZV parts }"
  fi
  # ---- e.3 launch
  dd="${P6N_DD_GB:-}"
  if [ -z "$dd" ]; then pod "grep -E '^P6_DD_GB=' $OPS/p6_start.env 2>/dev/null | tail -n 1 | cut -d= -f2" && dd="$(printf '%s' "$POD_OUT" | tr -dc '0-9')"; fi
  [ -n "$dd" ] || dd=16
  pod "if [ -f $P6/chain.log ]; then wc -l < $P6/chain.log; else echo 0; fi" || finish_fail e 2 "pod unreachable"; offset="$(printf '%s' "$POD_OUT" | tr -dc '0-9')"
  [ -n "$offset" ] || finish_fail e 2 "could not read the length of $P6/chain.log"
  info "e launching the gate dry run: P6_STOP_AFTER=launch_gate P6_D7_FALLBACK=0 P6_CBANK=$OUT P6_OVERLAP=0 P6_DD_GB=$dd (chain.log offset $offset); polling every ${P6N_GATE_POLL_S:-60}s"
  podw "mkdir -p $P6N && : > $P6N/gate_start.ref && echo $offset > $P6N/gate_offset.txt && rm -f $P6/stage/lanes.ok" || finish_fail e 2 "could not invalidate the lanes stage"
  ship "$HERE/pod/gate_dryrun.sh" gate_dryrun.sh || finish_fail e 2 "could not ship gate_dryrun.sh"
  podw "P6PKG=$P6PKG P6=$P6 OUT=$OUT DD_GB=$dd HF_TOKEN_FILE='$P6N_HF_TOKEN_FILE' P6N=$P6N setsid -f nohup bash $P6N/gate_dryrun.sh > $P6N/gate_dryrun.out 2>&1 < /dev/null" || finish_fail e 2 "could not launch gate_dryrun.sh"
  if [ "$DRY" = 1 ]; then step_ok e "OK_DRY (the launch above was printed, not executed; nothing to wait for or judge)"; return 0; fi
  # ---- e.4 wait (the chain holds its own lock; this script only reads)
  t0="$(date +%s)"
  while :; do
    sleep "${P6N_GATE_POLL_S:-60}"
    pod_has "$P6N/gate_dryrun.rc"; rc=$?
    if [ "$rc" = 0 ]; then break; fi
    if [ "$rc" = 2 ]; then nfail=$((nfail + 1)); [ "$nfail" -ge 8 ] && finish_fail e 2 "pod unreachable 8x while the gate dry run runs (it keeps running on the pod)"; else nfail=0; fi
    if pod "tail -n +$((offset + 1)) $P6/chain.log | tail -n 1 | cut -c1-220"; then info "e chain: $POD_OUT"; fi
    if [ $(( $(date +%s) - t0 )) -gt "${P6N_GATE_MAX_S:-14400}" ]; then finish_fail e 12 "the gate dry run is not finished after ${P6N_GATE_MAX_S:-14400}s (it may still run on the pod; NOT killed by this script)"; fi
  done
  pod "cat $P6N/gate_dryrun.rc; tail -n +$((offset + 1)) $P6/chain.log | grep -c 'REFUSED:' | sed 's/^/ZZR/;s/\$/ZZ/'; tail -n +$((offset + 1)) $P6/chain.log | grep 'REFUSED:' | tail -n 1 | cut -c1-300" || finish_fail e 2 "pod unreachable"
  rc="$(printf '%s\n' "$POD_OUT" | head -n 1 | tr -dc '0-9')"
  if [[ "$POD_OUT" != *ZZR0ZZ* ]]; then
    refused="$(printf '%s\n' "$POD_OUT" | grep -v '^[0-9]*$' | grep -v ZZR | tail -n 1)"
    finish_fail e 12 "the P6 chain REFUSED during the dry run (rc=$rc): ${refused:-see $P6/chain.log}"
  fi
  [ "$rc" = 0 ] || finish_fail e 12 "p6_pod.sh exited rc=$rc without a REFUSED line (see $P6/p6_stdout.log on the pod)"
  # ---- e.5 judge the token and the chain log (times are NFS-server times: the reference is a file created just before the launch, not the pod's clock)
  podh gate --lg "$P6/lg" --out "$OUT" --commit "$P6N_SHIP_COMMIT" --mc "$MC" --kept "$KEPT" --since-file "$P6N/gate_start.ref" --p6 "$P6" --ranks "${KS// /,}" --chain-offset "$offset" || finish_fail e 2 "pod unreachable"
  l="$(zv gate)"
  [ "$(kv "$l" state)" = ok ] || finish_fail e 12 "the dry run ended but its evidence does not hold up: $(kv "$l" why)"
  GATE_JUDGED=1
  step_ok e "P6 launch-gate DRY RUN PASS: token $(kv "$l" token) commit=$(kv "$l" commit) argv_sha=$(kv "$l" argv_sha) tree_sha=$(kv "$l" tree_sha) --targets=$(kv "$l" targets) (${KEPT} kept twins, $(kv "$l" mc_twin_parts) twin parts in the cache, $(kv "$l" parts_newer_than_start) written during the run); chain stopped BEFORE train.py"
}

# =================================================================================================================================================================
# (f) AUTO-LAUNCH of P6's real run (PI decision 2026-10-17).  Reached only after (e) PASSed in THIS run, or after the dry-run evidence is re-judged from the pod (--from f).
#     FAIL-CLOSED: nothing here launches unless EVERY pre-condition reads clean; any doubt prints `ZZPM P6_NOT_LAUNCHED <why>` and exits 13.
# =================================================================================================================================================================
step_f() {
  local l offset offset2 dd mb t0 tpre="" nfail=0 st state
  if [ "$DRY" = 1 ]; then
    wait_idle f
    dry "re-judge the dry-run evidence from the pod ($P6N/gate_offset.txt, gate_start.ref); require $P6/stage/launch_gate.ok and a free P6 chain lock"
    dry "ship pod/p6_launch.sh; launch detached (P6_POD_RUNBOOK section 3, without P6_STOP_AFTER): P6PKG=.. P6=.. OUT=$OUT DD_GB=<as the dry run> HF_TOKEN_FILE=<token-file> P6N=$P6N setsid -f nohup bash $P6N/p6_launch.sh > $P6N/p6_launch.out 2>&1 < /dev/null"
    dry "watch chain.log (from the pre-launch offset) + $RUN/metrics.jsonl (new lines): 'train.py preflight OK' then the first step line with a finite loss within ${P6N_FIRST_STEP_S:-600}s -> ZZPM P6_TRAINING; REFUSED / exit / ambiguity -> ZZPM P6_NOT_LAUNCHED"
    step_ok f "OK_DRY (planned; nothing launched)"; return 0
  fi
  [ -n "${P6N_SHIP_COMMIT:-}" ] && [ -n "${P6N_HF_TOKEN_FILE:-}" ] || finish_fail f 13 "P6N_SHIP_COMMIT and P6N_HF_TOKEN_FILE must be set"
  # ---- f.0 the dry run's evidence, from the POD (not from this shell's memory), unless (e) judged it a moment ago in this very run
  if [ "$GATE_JUDGED" != 1 ]; then
    load_facts full || finish_fail f 13 "the merged bank no longer judges OK: ${MERGE_LINE:-<none>}"
    pod "cat $P6N/gate_offset.txt 2>/dev/null; test -f $P6N/gate_start.ref; echo ZZF\$?ZZ" || finish_fail f 2 "pod unreachable"
    offset="$(printf '%s\n' "$POD_OUT" | head -n 1 | tr -dc '0-9')"
    { [ -n "$offset" ] && [[ "$POD_OUT" == *ZZF0ZZ* ]]; } || finish_fail f 13 "no recorded gate dry run on the pod (gate_offset.txt / gate_start.ref missing): run step (e) first"
    podh gate --lg "$P6/lg" --out "$OUT" --commit "$P6N_SHIP_COMMIT" --mc "$MC" --kept "$KEPT" --since-file "$P6N/gate_start.ref" --p6 "$P6" --ranks "${KS// /,}" --chain-offset "$offset" || finish_fail f 2 "pod unreachable"
    l="$(zv gate)"
    [ "$(kv "$l" state)" = ok ] || finish_fail f 13 "the dry-run evidence does not hold up (a PASS line is missing or something refused): $(kv "$l" why)"
    info "f dry-run evidence re-judged from the pod: token $(kv "$l" token)"
  fi
  # ---- f.1 the dry run took ~1 h: R0d must not be training when P6's trainer starts
  wait_idle f
  # ---- f.2 invariants of the real launch
  pod "test -f $P6/stage/launch_gate.ok; echo ZZG\$?ZZ; test -f $P6/stage/lanes.ok; echo ZZH\$?ZZ; test -s '$P6N_HF_TOKEN_FILE'; echo ZZI\$?ZZ; test -e $RUN/summary.json; echo ZZJ\$?ZZ" || finish_fail f 2 "pod unreachable"
  [[ "$POD_OUT" == *ZZG0ZZ* ]] || finish_fail f 13 "$P6/stage/launch_gate.ok is missing: the gate stage did not complete -- the real run would re-run the gate, not train"
  [[ "$POD_OUT" == *ZZH0ZZ* ]] || finish_fail f 13 "$P6/stage/lanes.ok is missing: the lanes stage did not complete"
  [[ "$POD_OUT" == *ZZI0ZZ* ]] || finish_fail f 13 "the HF token file is missing on the pod"
  [[ "$POD_OUT" == *ZZJ1ZZ* ]] || finish_fail f 13 "$RUN/summary.json already exists: a P6 run has already finished or been resumed -- ambiguous, not launching"
  dd="${P6N_DD_GB:-}"
  if [ -z "$dd" ]; then pod "grep -E '^P6_DD_GB=' $OPS/p6_start.env 2>/dev/null | tail -n 1 | cut -d= -f2" && dd="$(printf '%s' "$POD_OUT" | tr -dc '0-9')"; fi
  [ -n "$dd" ] || dd=16
  pod "if [ -f $RUN/metrics.jsonl ]; then wc -l < $RUN/metrics.jsonl; else echo 0; fi; wc -l < $P6/chain.log" || finish_fail f 2 "pod unreachable"
  mb="$(printf '%s\n' "$POD_OUT" | sed -n 1p | tr -dc '0-9')"; offset2="$(printf '%s\n' "$POD_OUT" | sed -n 2p | tr -dc '0-9')"
  { [ -n "$mb" ] && [ -n "$offset2" ]; } || finish_fail f 13 "could not read metrics.jsonl / chain.log lengths"
  # ---- f.3 the launch: P6_POD_RUNBOOK section 3, detached, `setsid -f nohup ... < /dev/null` as its OWN statement
  ship "$HERE/pod/p6_launch.sh" p6_launch.sh || finish_fail f 13 "could not ship p6_launch.sh"
  info "f LAUNCHING P6's real run (no P6_STOP_AFTER; P6_CBANK=$OUT P6_D7_FALLBACK=0 P6_OVERLAP=0 P6_DD_GB=$dd; chain.log offset $offset2, metrics lines before: $mb)"
  LAUNCHED=1
  podw "P6PKG=$P6PKG P6=$P6 OUT=$OUT DD_GB=$dd HF_TOKEN_FILE='$P6N_HF_TOKEN_FILE' P6N=$P6N setsid -f nohup bash $P6N/p6_launch.sh > $P6N/p6_launch.out 2>&1 < /dev/null" || finish_fail f 13 "the launch statement failed"
  # ---- f.4 confirm training started: 'train.py preflight OK' in chain.log AND a first step/loss line in metrics.jsonl, within ~10 min of the preflight line
  t0="$(date +%s)"
  while :; do
    sleep "${P6N_LAUNCH_POLL_S:-30}"
    if podh launchwatch --p6 "$P6" --run "$RUN" --chain-offset "$offset2" --metrics-before "$mb"; then
      nfail=0; l="$(zv launch)"; state="$(kv "$l" state)"
      case "$state" in
        training)
          step_ok f "P6 LAUNCHED and training started: first step line step=$(kv "$l" step) $(kv "$l" key)=$(kv "$l" value); chain alive=$(kv "$l" chain_alive)"
          say "P6_TRAINING step=$(kv "$l" step) $(kv "$l" key)=$(kv "$l" value) chain_alive=$(kv "$l" chain_alive) -- watch: $P6/chain.log, $RUN/metrics.jsonl, the private HF repo (P6_POD_RUNBOOK section 4)"
          return 0 ;;
        refused|gone)        LAUNCHED=0; finish_fail f 13 "the real launch did not start training (the chain refused or never ran): state=$state $(kv "$l" why)" ;;
        failed)              finish_fail f 13 "the chain reports ZZP6_FAILED: $(kv "$l" why)" ;;
        trainer_exited)      finish_fail f 13 "the trainer exited before its first step: $(kv "$l" why) -- the chain may retry (it sleeps 60 s between attempts); inspect $P6/train.log" ;;
        ambiguous)           finish_fail f 13 "AMBIGUOUS: $(kv "$l" why) (step=$(kv "$l" step)) -- do not assume either way; inspect the pod" ;;
        preflight_ok)        [ -n "$tpre" ] || { tpre="$(date +%s)"; info "f 'train.py preflight OK' seen; waiting up to ${P6N_FIRST_STEP_S:-600}s for the first step line"; }
                             if [ $(( $(date +%s) - tpre )) -gt "${P6N_FIRST_STEP_S:-600}" ]; then finish_fail f 13 "no first step/loss line within ${P6N_FIRST_STEP_S:-600}s of 'train.py preflight OK'"; fi ;;
        starting) : ;;
        *) finish_fail f 13 "unreadable launch verdict: ${l:-<none>}" ;;
      esac
    else
      nfail=$((nfail + 1)); [ "$nfail" -ge 8 ] && finish_fail f 2 "pod unreachable 8x while confirming the launch"
    fi
    if [ $(( $(date +%s) - t0 )) -gt "${P6N_LAUNCH_MAX_S:-1800}" ]; then finish_fail f 13 "training not confirmed ${P6N_LAUNCH_MAX_S:-1800}s after the launch (last state: ${state:-none})"; fi
  done
}

# =================================================================================================================================================================
# --probe: READ-ONLY inventory against the real pod (nothing written, nothing started, no flock probe)
# =================================================================================================================================================================
probe() {
  say "PROBE begin (read-only; $(date -u +%FT%TZ))"
  preflight
  podh merge --f "$F" --out "$OUT" --inp "$IN" --prereg-sha "$PREREG_SHA" || finish_fail probe 2 "pod unreachable"
  info "probe a merge verdict: $(zv merge | sed 's/^ZZV merge //')"
  pod "tail -n 1 $F/progress.log | cut -c1-330; ls $F/PARTS_DONE $F/RUN_DONE $F/RUN_INCOMPLETE $F/merge.out 2>&1 | sed 's/^/ls: /' | cut -c1-120; ls -d $OUT $P6N 2>&1 | sed 's/^/ls: /' | cut -c1-120" || true
  info "probe a progress/markers: $(printf '%s' "$POD_OUT" | tr '\n' '|' | cut -c1-700)"
  podh busy --data "$DATA" --no-lock-probe || true; info "probe e pod-free check (GPU + rls1_pod.sh; the chain-lock flock probe is skipped under --probe): $(zv busy | sed 's/^ZZV busy //')  (busy => (e) WAITS: poll ${P6N_IDLE_POLL_S:-300}s, cap ${P6N_GPU_WAIT_MAX_H:-12} h)"
  podh p6tree --pkg "$P6PKG" || true; info "probe e p6tree: $(zv p6tree | sed 's/^ZZV p6tree //')  (expected for a PASS: commit == P6N_SHIP_COMMIT, cbank_knob=1, cbank_commands_guard=1, r43=1)"
  pod "for s in env weights recover inputs commands earlylab cache lanes lab mem egr launch_gate; do printf '%s=%s ' \$s \$(test -f $P6/stage/\$s.ok && echo ok || echo -); done; echo; grep -E '^P6_' $OPS/p6_start.env 2>/dev/null | grep -v '^#' | tr '\n' ' '; echo; tail -n 2 $P6/chain.log | cut -c1-200" || true
  info "probe e P6 stage markers / p6_start.env knobs / last chain.log lines: $(printf '%s' "$POD_OUT" | tr '\n' '|' | cut -c1-900)"
  info "probe e P6N_SHIP_COMMIT=${P6N_SHIP_COMMIT:+set}${P6N_SHIP_COMMIT:-UNSET}  P6N_HF_TOKEN_FILE=${P6N_HF_TOKEN_FILE:+set}${P6N_HF_TOKEN_FILE:-UNSET}"
  local pb="${P6N_PROBE_PARTS_BANK:-$IN}"
  if [ ! -e "$OUT/targets_rank1.jsonl" ]; then
    podhpy parts --bank "$pb" --src "$MC_TWIN" --dst "$MC" --p6refe "$P6PKG/refe" --mode plan || true
    info "probe d parts plan on the INPUT bank as a stand-in for the merged bank ($pb): $(zv parts | sed 's/^ZZV parts //')"
  else
    podhpy parts --bank "$OUT" --src "$MC_TWIN" --dst "$MC" --p6refe "$P6PKG/refe" --mode plan || true
    info "probe d parts plan on the merged bank: $(zv parts | sed 's/^ZZV parts //')"
  fi
  say "PROBE_DONE"
  exit 0
}

# =================================================================================================================================================================
# main
# =================================================================================================================================================================
say "START $(date -u +%FT%TZ) from=$FROM to=$TO dry=$DRY probe=$PROBE logs=$LOGD"
if [ "$PROBE" = 1 ]; then probe; fi
case "$FROM" in a|b|c|d|e|f) ;; *) finish_fail 0 2 "--from must be one of a b c d e f" ;; esac
case "$TO"   in a|b|c|d|e|f) ;; *) finish_fail 0 2 "--to must be one of a b c d e f" ;; esac
preflight
ord() { case "$1" in a) echo 1;; b) echo 2;; c) echo 3;; d) echo 4;; e) echo 5;; f) echo 6;; esac; }
for s in a b c d e f; do
  [ "$(ord "$s")" -lt "$(ord "$FROM")" ] && continue
  [ "$(ord "$s")" -gt "$(ord "$TO")" ] && break
  if [ "$s" != a ] && [ -z "$OUT_SHA" ]; then          # started after (a): load the merge facts (the report; each later step re-checks the bank's hash itself)
    if load_facts quick; then info "facts: kept=$KEPT K1=$K1 K2=$K2 K3=$K3 out_sha=${OUT_SHA:0:16}"
    elif [ "$DRY" = 1 ]; then OUT_SHA=DRYRUN; info "dry: no finished merge on the pod yet ($(kv "${MERGE_LINE:-}" state)); later steps print their plans"
    else finish_fail "$s" 5 "no finished, valid merge to start from: ${MERGE_LINE:-<no verdict>}"; fi
  fi
  "step_$s" || finish_fail "$s" 14 "step_$s returned non-zero without reporting (internal error)"
  [ "$LAST_OK" = "$s" ] || finish_fail "$s" 14 "step_$s returned without printing its OK line (internal error)"
done
[ "$LAST_OK" = "$TO" ] || finish_fail "$TO" 14 "the pipeline ended without reaching step $TO OK (last OK: $LAST_OK): internal error"
FINISHED=1
say "DONE result=OK through=$TO exit=0 dry=$DRY logs=$LOGD"
if [ "$TO" = e ] && [ "$DRY" = 0 ]; then
  say "P6_LAUNCH_SKIPPED (--no-launch / --to e): the gate token is minted and verified; P6 was NOT launched. To launch: re-run with --from f."
fi
exit 0
