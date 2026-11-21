#!/usr/bin/env bash
# PREREG_V8F_KINDX, the SEED-1 CONFIRMATION of CANDIDATE (speed) on box 2: train X = M2b + --code-mix kind at SEED 1 (MIXO's argv, `--seed 1` and nothing else), run KIND's per-kind oracle evals on it,
# and re-read gain_k at s1. "k is a CARRIER only if it clears 2F = 0.2415 m at s1 too (KIND's both-seed rule)"; at s0 speed was +0.716 [+0.623, +0.818]. ARMED ONLY ON THE PI's go for box 2 (it is the
# PI's personal machine; one job at a time under SAFE MODE). Derived from chain_v8f_mixo.sh (the training chain: launch gate, busy marker, refresh from the remote tip, checkpoint sha checks, v8f_guard.sh,
# waitjob polling, DONE / STOP markers) and chain_v8f_kscore.sh (the predecessor predicate). Markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_kindx_s1.out), which v8f_guard.sh's anchored end test knows.
#   chain_v8f_kindx_s1.sh --dry-run                     PRINTS the exact command every stage runs against box 2 (the queue's own `gatemixo` / `trainmixo` / `evalmixo` lines from the tip's queue or $V8F_QUEUE,
#                                                       the driver's own job line and the ten `v8f_mini_eval.py run` argvs of the KINDX evals), the checkpoint sha256 values it asserts, the argv-diff
#                                                       assertion run on the two printed `trainmixo` lines, and the reader's arguments, and exits; nothing is shipped, launched or waited for.
#   chain_v8f_kindx_s1.sh --classify-predecessor <out file> [<guard out file>]
#                                                       prints what the wait loop would do with a predecessor log (PROCEED / HOLD / WAIT, with the reason) and exits 0 / 1 / 2. The wait loop calls the SAME function.
#   0. wait for the predecessor (default: PREREG_V8F_KSCORE's chain, chain_v8f_kscore.out; box-2 order MIXO -> FLAT -> KSCORE -> KINDX-s1):
#        ZZV8FMINI-DONE                                  -> PROCEED
#        a ZZV8FMINI-STOP that is a RUN-level stop (a KSCORE hyps / gate / fit / eval / read job returned rc != 0: "<job> rc=<n> (...)"), the VERDICT-level stop (KSCORE's reader refused: its `read` job's rc 2)
#        or a GATE-level stop (KSCORE's scorer gate assertions refused: "gate s<seed>[_perm] assertions") -> PROCEED, PROVIDED the logs carry no box-level evidence (ZZGUARD-STOP, ZZCRASH, ZZUNREACHABLE, ZZTIMEOUT)
#        the PRE-FLIGHT stops (KSCORE stopped on the dev box before box 2 ran anything) -> PROCEED; every other stop (guard, thermal, reboot, unreachable, timeout, busy marker, a failed ship / launch / pull / hash, the
#        refresh job, the guard checks, the predecessor's own held predecessor) and ANY stop this predicate does not recognise (FAIL CLOSED) -> HOLD: this chain stops with ZZV8FMINI-STOP, because the box is not
#        known to be healthy. A predecessor that ended on its own verdict must not leave box 2 idle for hours (an unconditional "any STOP stops me" idled a GPU for 8.3 h); one that left the box unhealthy must not
#        hand it a 4-hour training. AFTER_OUT / AFTER_GUARD / AFTER_DONE / AFTER_STOP override the logs and the markers; the busy marker is then taken ATOMICALLY (noclobber)
#   1. refresh box 2's code from the REMOTE tip; sha-check 9 key files (MIXO's 8 + the KINDX driver), md5-check both v9 releases; extract the readers HERE from the same archive
#   2. B = M2b is fixed: box 2's two M2b checkpoints ARE the banked runs' (sha256); a different file stops the chain while box 2 is idle
#   3. PRE-FLIGHT, before any training: the argv-diff assertion on the tip queue's own `trainmixo 0` / `trainmixo 1` lines (the seed, and the output directory that carries it, are the ONLY difference), B's seed-1
#      KIND evals staged, and `kindx_read.py --seed 1 --check-inputs` (the banked s0 read is the CANDIDATE, B's evals are on the committed 2,399 / 141 windows, the strata file exists)
#   4. the in-process launch gate for X (`gatemixo`), pulled and asserted HERE (`v8f_mini_eval_kindx.py assert-gate`: MIXO's assertions, a port of its heredoc, plus the tip's o_interface = full record and probe, and
#      X's BUILT model equal to MIXO's banked s0 gate apart from o_interface = full). The queue's printed line carries `--code-mix kind` exactly once and none of the other levers' flags. A token record
#      (gate_token.json) is written beside it. No PASS => no training (the trainer's own in-process gate refuses as well, ZZDVB-REFUSEDZZ)
#   5. TRAIN X s1 (`trainmixo 1`: MIXO's line with the seed the only difference, ESTIMATED at B's own wall-clock, MIXO s0 took 4.0 h): the checkpoint sha256 is measured after training; the run's OWN records
#      (selfcheck, summary, argv.json) are pulled and asserted at once (`assert-run`: the in-process gate passes, arm M2b / seed 1 / 19,164 steps, argv = MIXO's s0 argv in the seed only)
#   6. EVAL: `evalmixo 1` (MIXO's seven evals: they are the s1 analogue of the banked `eval_M2b-mixo_s0*` files the PREREG's controls 1 and 2 compare with) and the ten KINDX evals (own, none = the identity
#      oracle, all, union, six single kinds) in ONE process (`v8f_mini_eval_kindx.py evals`, whose name carries `v8f_mini_eval` so the guard that is already running can stop it); the sha256 is measured AGAIN
#   7. pull; `kindx_controls.py --seed 1 1` and `2` (the PREREG's controls: own / union / full-GT reproduce the standard evals to <= 1e-4 m, the identity oracle gains EXACTLY 0.0, the 34 no-v9-row windows gain exactly
#      0.0, the checkpoint sha256 is the measured one and is not X_s0's); then `kindx_read.py --seed 1` -> READ_s1.json (CARRIER / NOT A CARRIER, the CANDIDATE row applied verbatim)
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the training wait is 1,320 polls of 30 s = 11 h, one hour above the guard's 10 h cap (V8F_MAX_RUN_H); the guard stops any run it PROJECTS above the cap. The seven evals are ~4 min in all (MIXO s0:
# 13:30 -> 13:34), the ten KINDX evals ~6 min (waits: 600 and 240 polls).
# THE STOP TEXTS of this chain (a successor's predicate must classify them): "<job> rc=<n>" and "<job> wait=<n> (...)" for the jobs v8fkindx<MMDDHHMM>_(gatemixo|t_M2b_s1_mixo|e_M2b_s1_mixo|k_M2b_s1_mixo|refresh);
# verdict-level: "gate X assertions", "run assertions", "argv assertion", "kindx check-inputs", "kindx controls 1", "kindx controls 2", "kindx read s1"; everything else as in chain_v8f_mixo.sh / chain_v8f_kscore.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
# the banked checkpoints (raw/full/MANIFEST.md); the reader pins the same two values (BANKED_CKPT_SHA) and a test pins that they agree
CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
X0_SHA=c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d     # X_s0 (MIXO's seed-0 checkpoint): X_s1 is another run, so its sha256 must differ
PYX=${KINDX_PYX:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
DRV=${KINDX_DRIVER:-$HERE/../../../../../../stack/scripts/v8f_mini_eval_kindx.py}
ksm() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
# the seven evals of evalmixo: the standard eval (empty suffix) and six tagged ones (the same list chain_v8f_mixo.sh pulls)
X_SUFFIXES="-tacabs -tacorc -tacorc-all-latcons -tacpred -tacorc-constraints -tacorc-constraints-a50"
# the ten KINDX evals (v8f_mini_eval_kindx.py EVAL_NAMES): own, the identity oracle, the full GT code, the union, the six single kinds (tagged `only-<kind>`)
KINDX_NAMES="own none all union only-speed only-angle only-tte only-timegap only-geom only-leadgap"
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8fkindx_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
reader_args() {   # reader_args <seed>: the one argument string of the read, printed by --dry-run and run by the chain
  echo "--seed $1"
}
# ---- the predecessor predicate (the PREREG's order: box 2 runs MIXO -> FLAT -> KSCORE -> this chain). ONE function: the wait loop below calls it, and so does `--classify-predecessor`, which a test runs on
# synthetic logs. It sits BEFORE every statement that touches box state (the mkdir of $E, the ssh / scp definitions), so the classify mode has no side effect.
AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
# What the predecessor's STOP may say for this chain to proceed. The predecessor is KSCORE's chain (chain_v8f_kscore.sh); EVERY text it can print after ZZV8FMINI-STOP is its own `stop "<msg>"` (25 calls, 24
# distinct texts) or one of three direct `echo ZZV8FMINI-STOP ...` lines. Classified here by what they say about the BOX (not about the arm); a test (test_v8f_kindx_s1.py) reads KSCORE's chain SOURCE and fails
# if it grows a text this table has not classified:
#   VERDICT PROCEED  "<job> rc=<n> (...)" of the `read` job   KSCORE's reader refused (its rc 2: a failed stop-control or a state the table does not map): the read, not the box, failed
#   RUN     PROCEED  "<job> rc=<n> (...)"   a job that ran KSCORE's code on box 2 came back nonzero: hyps_eval_<s> | hyps_train_<s> | gate_s<s>[_perm] | fit_s<s>[_perm] | eval. The job name is KSCORE's own,
#                                           v8fkscore<MMDDHHMM>_<stage> (kjob() in the chain). rc must be a positive integer: "rc=" (nothing parsed), "rc=0" and a signed value (a native crash) HOLD. THE REFRESH JOB'S rc DOES
#                                           NOT PROCEED: it runs no model code (a tar into box 2's repo), so its failure is the box's. The text ends with KSCORE's fixed parenthetical (waitrc), required here
#   GATE    PROCEED  "gate s<seed>[_perm] assertions"   KSCORE's scorer launch gate did not pass (assert-gate): nothing was trained
#   PREFLIGHT PROCEED  KSCORE's chain stopped on the DEV BOX (the mirror, `git archive`, the tip's source) before box 2 was asked to run anything: "remote tip <sha> is not in the mirror", "the tip has no <file>",
#                    "the tip's v8f_kscore.py has no FLAG_CHECKS ...", "the tip's v8f_guard.sh does not know v8f_kscore ...", "archive", "no such stage <st>" (a chain defect, before a job is shipped). They say nothing about
#                    box 2's health, and this chain runs its own pre-flight on its own tip, so a cause it shares with KSCORE (a stale mirror, a full dev-box disk) stops it there. Holding on them would leave box 2 idle for a
#                    night on a dev-box staging problem (memory: 'chain refusal must not stall the pod'). This class is an EXTENSION of the brief ("run / verdict / gate"), taken from KSCORE's own predicate for FLAT
#   BOX     HOLD     everything else, in the groups below; ANY text not listed is treated the same (FAIL CLOSED):
#     transport / box   "<job> wait=<n> (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", "box 2 unreachable", "ship <job>", "launch <job>", "pull <path>", "ship the code archive", "ship refresh job", "launch refresh",
#                       "<refresh job> rc=<n> (...)", "cannot hash box 2's M2b_s<n> checkpoint", "no fresh telemetry (<t1> / <t2>): the temperature guard would be blind"
#     box state         "guard stop" (thermal / time), "box 2 <file> != tip <sha>", "box 2 has no <R>/join_train.npz ...", "<phase>: box 2's M2b_s<n> checkpoint sha256 <a> != the banked run's <b>"
#     marker / chain    "busy marker present: ...", "busy marker taken while arming: ...", "the predecessor's end does not clear the box: HOLD ..." (KSCORE's own predecessor held), "v8f_guard.sh does not know this chain's markers
#                       (anchored): it would never exit", "v8f_guard.sh (box2_ship) does not kill v8f_kscore: ..." (the guard is the box's safety: a guard that cannot stop a job holds)
# Independent of the message, a log that carries ZZGUARD-STOP (the temperature / time guard), ZZCRASH (a reboot), ZZUNREACHABLE or ZZTIMEOUT (waitjob.sh) holds: a run-level `rc=1` that came with a thermal stop or
# a reboot is the box's failure. All four are printed at column 0 (v8f_guard.sh, waitjob.sh). A DONE anywhere in the log wins over everything. Re-pointing AFTER_OUT at a different predecessor needs THAT chain's
# texts classified: these regexes are KSCORE's, and a text they do not know HOLDS.
VERDICT_LEVEL_RE='^v8fkscore[0-9]{8}_read rc=[1-9][0-9]* \(the read.s rc 2 = a failed stop-control or an UNMAPPED state: D:/tanitad/runs/v8f_kscore/READ\.json on box 2\)$'
RUN_LEVEL_RE='^v8fkscore[0-9]{8}_(hyps_eval_[01]|hyps_train_[01]|gate_s[01](_perm)?|fit_s[01](_perm)?|eval|read) rc=[1-9][0-9]* \(the read.s rc 2 = a failed stop-control or an UNMAPPED state: D:/tanitad/runs/v8f_kscore/READ\.json on box 2\)$'
GATE_LEVEL_RE='^gate s[01](_perm)? assertions$'
PREFLIGHT_LEVEL_RE="^(archive|remote tip [0-9a-f]{7,40} is not in the mirror|the tip has no .+|the tip's v8f_kscore\.py has no FLAG_CHECKS \(the gate would refuse every flag\)|the tip's v8f_guard\.sh does not know v8f_kscore \(WIRING\.patch not applied\): the temperature guard could not stop our python|no such stage .+)\$"
after_decision() {   # after_decision <predecessor chain log> [<its guard log>]: prints the decision; rc 0 = PROCEED, 1 = HOLD, 2 = WAIT (no end marker yet)
  local out=$1 guard=${2:-} stop msg ev
  if grep -a -q "^$AFTER_DONE" "$out" 2>/dev/null; then echo "PROCEED the predecessor ended: $(grep -a "^$AFTER_DONE" "$out" | tail -1 | tr -d '\r' | cut -c1-160)"; return 0; fi
  stop=$(grep -a "^$AFTER_STOP" "$out" 2>/dev/null | tail -1 | tr -d '\r')
  [ -n "$stop" ] || { echo "WAIT no $AFTER_DONE / $AFTER_STOP in $out yet"; return 2; }
  msg=${stop#"$AFTER_STOP"}; msg=${msg# }
  ev=$(grep -a -h -E '^(ZZGUARD-STOP|ZZCRASH|ZZUNREACHABLE|ZZTIMEOUT)' "$out" ${guard:+"$guard"} 2>/dev/null | head -1 | tr -d '\r')
  [ -z "$ev" ] || { echo "HOLD box-level evidence in the predecessor's logs: $ev (its stop: $msg)"; return 1; }
  if [[ $msg =~ $VERDICT_LEVEL_RE ]]; then echo "PROCEED verdict-level stop (KSCORE's reader refused; the box is fine): $msg"; return 0; fi
  if [[ $msg =~ $RUN_LEVEL_RE ]]; then echo "PROCEED run-level stop: $msg"; return 0; fi
  if [[ $msg =~ $GATE_LEVEL_RE ]]; then echo "PROCEED gate-level stop (the scorer launch gate refused, nothing was trained): $msg"; return 0; fi
  if [[ $msg =~ $PREFLIGHT_LEVEL_RE ]]; then echo "PROCEED pre-flight stop (on the dev box, before box 2 ran anything; this chain re-checks its own tip): $msg"; return 0; fi
  if [[ $msg =~ ^v8fkscore[0-9]{8}_refresh\ rc= ]]; then echo "HOLD box-level: KSCORE's code-refresh job (a tar into box 2's repo, no model code) failed: $msg"; return 1; fi
  echo "HOLD a stop that is not recognised as verdict-, gate- or run-level (fail closed): $msg"; return 1
}
if [ "${1:-}" = "--classify-predecessor" ]; then after_decision "${2:?usage: --classify-predecessor <out file> [<guard out file>]}" "${3:-}"; exit $?; fi
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8fkindx_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<logs>/chain_v8f_kscore.out}: ^${AFTER_DONE:-ZZV8FMINI-DONE} (stop on ^${AFTER_STOP:-ZZV8FMINI-STOP})"
  echo "== checkpoint sha256 (asserted on box 2 before the gate; X_s1's measured after training and after the evals, and it must differ from X_s0's $X0_SHA)"
  for s in 0 1; do echo "M2b_s$s $R/M2b_s$s/ckpt_last.pt ${CKPT_SHA[$s]}"; done
  D0=${TMPDIR:-/tmp}/v8fkindx_dry_train0_$$.txt; D1=${TMPDIR:-/tmp}/v8fkindx_dry_train1_$$.txt
  for st in "gatemixo" "trainmixo 0" "trainmixo 1" "evalmixo 1"; do
    echo "== $st"; o=$(dry_stage "$QF" $st) || exit 1; echo "$o"
    case "$st" in "trainmixo 0") echo "$o" | grep -a 'v8f_mini_train.py' > $D0;; "trainmixo 1") echo "$o" | grep -a 'v8f_mini_train.py' > $D1;; esac
  done
  echo "== argv diff (trainmixo 0 vs trainmixo 1)"; "$PYX" "$(ksm "$DRV")" assert-argv --base "$(ksm $D0)" --other "$(ksm $D1)" || { echo "ZZV8FMINI-STOP argv assertion"; exit 1; }
  echo "== kindx evals (the driver's job line; the sha256 is X_s1's, measured after training)"
  "$PYX" "$(ksm "$DRV")" cmdline evals --seed 1 --expect-sha X_s1_sha256 || exit 1
  echo "== kindx evals (the ten v8f_mini_eval.py argvs the job's process runs, in order)"
  "$PYX" "$(ksm "$DRV")" evals-argv --seed 1 || exit 1
  echo "== read"
  echo "kindx_read.py --check-inputs $(reader_args 1)   [KINDX_OUT=<E> KINDX_CODE=<tip archive> KINDX_EV=<E>/eval KINDX_BE=<E>/b_kind/eval KINDX_SEED=1]"
  echo "kindx_controls.py $(reader_args 1) 1"
  echo "kindx_controls.py $(reader_args 1) 2"
  echo "kindx_read.py $(reader_args 1)   -> <E>/READ_s1.json"
  echo "== stages"
  echo "wait(KSCORE) refresh(tip) ckpt-sha(B) pre-flight(argv diff, B s1 evals, check-inputs) gatemixo(+assert-gate,+token) trainmixo(1) ckpt-sha(X) assert-run evalmixo(1) evals(kindx) ckpt-sha(X, again) pull controls read DONE"
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_kindx_s1; mkdir -p $E/runs $E/eval $E/b_kind/eval
CK=$E/ckpt_sha256.json
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=$PYX
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
M2B_WINDOWS=306566   # B's n_windows_all_clips (banked summary.json); X trains on B's window LIST (the lever is the code mask only)
TRAIN_POLLS=1320; EVAL_POLLS=600; KINDX_POLLS=240   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fkindx_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kscore.out}; AFTER_GUARD=${AFTER_GUARD:-$L/v8fkscore_guard.out}   # AFTER_DONE / AFTER_STOP are set above, with the predicate
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8fkindx_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8fkindx${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
kjob() {     # kjob <name> <polls> <driver args...>: the DRIVER's job .cmd (the KSCORE recipe: `mkjob` prints the CRLF job text)
  local j="v8fkindx${TAG}_$1" m=$2; shift 2
  "$PYX" "$(ksm "$DRVL")" mkjob $j "$@" > $B/$j.cmd && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
pull_opt() { # a file the instrument writes only sometimes (the --project side file): its absence is not a stop
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no $R/$1 (optional)"
}
ckset() {    # ckset <key> <sha256>: one measured value into the json the controls read
  $PY -c "import json,os,sys;p=sys.argv[1];d=json.load(open(p,encoding='utf-8')) if os.path.exists(p) else {};d[sys.argv[2]]=sys.argv[3];json.dump(d,open(p,'w',encoding='utf-8'),indent=1)" "$(cygpath -m $CK)" "$1" "$2"
}
ckptsha_b() {   # ckptsha_b <phase>: box 2's two M2b checkpoints ARE the banked runs' (raw/full/MANIFEST.md)
  local s g
  for s in 0 1; do
    g=$(rsha "$R/M2b_s$s/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s$s checkpoint"
    [ "$g" = "${CKPT_SHA[$s]}" ] || stop "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}"
    ckset "M2b_s$s" "$g"
  done
  echo "$(date '+%m-%d %H:%M') $1: both M2b checkpoints are the banked runs' (sha256 asserted)"
}
ckptsha_x() {   # ckptsha_x <trained|after_evals>: X_s1's checkpoint, measured on box 2; after the evals it must be the value measured after training (the scored checkpoint IS the trained one)
  local g; g=$(rsha "$R/M2b_s1_mixo/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "cannot hash box 2's M2b_s1_mixo checkpoint ($1)"
  [ "$g" != "$X0_SHA" ] || stop "box 2's M2b_s1_mixo checkpoint is X_s0's (sha256 $g): no second seed was trained"
  if [ "$1" = after_evals ]; then
    local g0; g0=$($PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $CK)" "M2b-mixo_s1_trained")
    [ "$g" = "$g0" ] || stop "M2b_s1_mixo's checkpoint changed during the evals: $g != $g0"
  fi
  ckset "M2b-mixo_s1_$1" "$g"
  XSHA=$g
  echo "$(date '+%m-%d %H:%M') M2b_s1_mixo checkpoint sha256 ($1) ${g:0:16}..."
}
finish() {   # finish <summary>: the committed end of the chain
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fkindx.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
jget() { $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $E/$1)" $2; }
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
while :; do
  d=$(after_decision "$AFTER_OUT" "$AFTER_GUARD"); r=$?
  if [ $r = 0 ]; then echo "$(date '+%m-%d %H:%M') predecessor: $d"; break; fi
  if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor's end does not clear the box: $d"; exit 1; fi
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_KINDX seed-1 confirmation (X = M2b + --code-mix kind trained at seed 1, then ten oracle evals) since $(date '+%m-%d %H:%M'), ~4.5 h. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8fkindx_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8fkindx_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run line
for st in gatemixo trainmixo evalmixo; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in "$PK/raw/2026-10-25-kindx/kindx_read.py" "$PK/raw/2026-10-25-kindx/kindx_controls.py" "$PK/raw/2026-10-25-kindx/READ.json" "$PK/raw/2026-10-25-mixo/run/runs/M2b_s0_mixo/argv.json" \
         "$PK/raw/2026-10-25-mixo/run/gate_M2b-mixo/selfcheck.json" stack/scripts/v8f_mini_eval_kindx.py; do
  git -C $MIRROR cat-file -e "$T:$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"code_mix": cfg.code_mix' || stop "the tip's built_levers does not record code_mix"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '^CODE_MIX_UNITS = {' || stop "the tip's v8f_mini.py has no CODE_MIX_UNITS partition"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_end": cfg.p_gt_end' || stop "the tip's built_levers does not record p_gt_end"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"o_interface": cfg.o_interface' || stop "the tip's built_levers does not record o_interface"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--code-mix"' || stop "the tip's trainer has no --code-mix flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"code_mix": "lever:code_mix"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --code-mix"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'code_mix_probe(model, probe, cfg)' || stop "the tip's trainer does not run the code_mix probe in its in-process gate"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--o-interface"' || stop "the tip's trainer has no --o-interface flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"o_interface": "lever:o_interface"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --o-interface"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'o_interface_probe(model, probe, cfg)' || stop "the tip's trainer does not run the o_interface probe in its in-process gate"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fkindx_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fkindx_tip.tar tanitad-box2:D:/tanitad/repo_v8fkindx_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fkindx_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FKINDX.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fkindx${TAG}_refresh.cmd
$SCP $B/v8fkindx${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fkindx${TAG}_refresh || stop "launch refresh"
waitrc v8fkindx${TAG}_refresh 40
KEYSHA=""
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/scripts/v8f_mini_eval_kindx.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
  KEYSHA="$KEYSHA\"$f\": \"$l\", "
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (9 key files sha-checked; both v9 releases md5-checked)"
# the readers and the driver run HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8fkindx_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8fkindx_tip.tar -C $X || stop "extract the reader"
DRVL="$X/stack/scripts/v8f_mini_eval_kindx.py"
KX="$X/$PK/raw/2026-10-25-kindx"
MXR="$X/$PK/raw/2026-10-25-mixo/run"
# ---- 2. the base is fixed (B = M2b): its checkpoints ARE the banked runs' (box 2 is idle: a different file costs nothing)
rm -f $CK
ckptsha_b "before the gate"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/runs/v8f_mini_full/M2b_s1_mixo/summary.json) { 'EXISTS' }\"" 2>/dev/null | tr -d '\r' | grep -q EXISTS \
  && echo "$(date '+%m-%d %H:%M') NOTE: box 2 already holds a finished M2b_s1_mixo: trainmixo skips it, and assert-run / the sha256 / the controls still read it"
# ---- 3. PRE-FLIGHT, before any training (a missing file stops the chain while box 2 is still idle)
# the PREREG's "MIXO's argv, --seed 1": the tip queue's own two printed lines differ in the seed (and the output directory that carries it) and nothing else
dry_stage "$QC" trainmixo 0 | grep -a 'v8f_mini_train.py' > $E/queue_trainmixo_s0.txt; dry_stage "$QC" trainmixo 1 | grep -a 'v8f_mini_train.py' > $E/queue_trainmixo_s1.txt
[ -s $E/queue_trainmixo_s0.txt ] && [ -s $E/queue_trainmixo_s1.txt ] || stop "the tip's queue prints no trainmixo line"
"$PYX" "$(ksm "$DRVL")" assert-argv --base "$(cygpath -m $E/queue_trainmixo_s0.txt)" --other "$(cygpath -m $E/queue_trainmixo_s1.txt)" || stop "argv assertion"
# ... and MIXO's banked s0 run argv.json is that line's seed-0 form (so the line, not just the diff, is MIXO's)
mkdir -p $E/runs/M2b_s1
cp "$L/v8f_mini/runs/M2b_s1/summary.json" $E/runs/M2b_s1/ || stop "stage the banked M2b_s1/summary.json"
for n in kind-own kind-all kind-union kind-only-speed kind-only-angle kind-only-tte kind-only-timegap kind-only-geom kind-only-leadgap; do
  for x in npz json; do cp "$L/v8f_kind/eval/eval_M2b_s1-$n.$x" $E/b_kind/eval/ || stop "stage B s1 KIND eval eval_M2b_s1-$n.$x"; done
done
export PYTHONIOENCODING=utf-8 KINDX_OUT="$(cygpath -m $E)" KINDX_CODE="$(cygpath -m $X)" KINDX_EV="$(cygpath -m $E/eval)" KINDX_BE="$(cygpath -m $E/b_kind/eval)" KINDX_SEED=1
"$PY" -I "$(cygpath -m "$KX")/kindx_read.py" $(reader_args 1) --check-inputs || stop "kindx check-inputs"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
grep -qF 'v8f_mini_train|v8f_mini_eval' $B/v8f_guard.sh || stop "v8f_guard.sh (box2_ship) does not kill v8f_mini_train / v8f_mini_eval: copy the tip's v8f_guard.sh over it first"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_kindx_s1.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fkindx_guard.out 2>&1 &
# ---- 4. the launch gate for X, asserted here as well as in-process. GLINE is the line the queue PRINTS for the stage: a static test pins it to the line the stage RUNS.
GLINE=$(dry_stage "$QC" gatemixo | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gatemixo line"
queue gatemixo 120 gatemixo
pull "gate_M2b-mixo/selfcheck.json" "$E/gate_M2b-mixo"
"$PYX" "$(ksm "$DRVL")" assert-gate --selfcheck "$(cygpath -m $E/gate_M2b-mixo/selfcheck.json)" --b-summary "$(cygpath -m $E/runs/M2b_s1/summary.json)" --line "$GLINE" \
  --holdout $HOLDOUT_SHA --steps $M2B_STEPS --windows $M2B_WINDOWS --base-selfcheck "$(cygpath -m "$MXR")/gate_M2b-mixo/selfcheck.json" || stop "gate X assertions"
# the token record: the exact commit, the sha256 of the queue and of the gate's selfcheck, the key files box 2 was checked against, and the data manifests the gate read (written by the chain, never by hand)
"$PYX" "$(ksm "$DRVL")" write-token --selfcheck "$(cygpath -m $E/gate_M2b-mixo/selfcheck.json)" --out "$(cygpath -m $E/gate_token.json)" --commit "$T" --queue-sha "$(sha256sum $QT | cut -c1-64)" \
  --key-files "{${KEYSHA%, }}" || stop "write the gate token"
# ---- 5. TRAIN X s1, then its OWN records
g="M2b_s1_mixo"
echo "$g" > $CUR
queue "t_$g" $TRAIN_POLLS trainmixo 1
echo "" > $CUR
ckptsha_x trained
for f in selfcheck.json summary.json argv.json train_log.jsonl train.log; do pull "$g/$f" "$E/runs/$g"; done
"$PYX" "$(ksm "$DRVL")" assert-run --run-dir "$(cygpath -m $E/runs/$g)" --b-summary "$(cygpath -m $E/runs/M2b_s1/summary.json)" --base-argv "$(cygpath -m "$MXR")/runs/M2b_s0_mixo/argv.json" \
  --base-selfcheck "$(cygpath -m "$MXR")/gate_M2b-mixo/selfcheck.json" --seed 1 || stop "run assertions"
grep ZZGUARD-STOP $L/v8fkindx_guard.out 2>/dev/null && stop "guard stop"
# ---- 6. EVAL: MIXO's seven evals (the controls' "banked" files at s1), then the ten KINDX evals in one process
queue "e_$g" $EVAL_POLLS evalmixo 1
kjob "k_$g" $KINDX_POLLS evals --seed 1 --expect-sha "$XSHA"
ckptsha_x after_evals
# ---- 7. pull, controls, read
pull "$g/ckpt_last.pt" "$E/runs/$g"
for f in "" $X_SUFFIXES; do for x in json npz log; do pull "eval/eval_M2b-mixo_s1$f.$x" "$E/eval"; done; done
for n in $KINDX_NAMES; do for x in json npz log; do pull "eval/eval_M2b-mixo_s1-kindx-$n.$x" "$E/eval"; done; done
pull_opt "eval/eval_M2b-mixo_s1-kindx-own_proj.npz" "$E/eval"
grep ZZGUARD-STOP $L/v8fkindx_guard.out 2>/dev/null && stop "guard stop"
"$PY" -I "$(cygpath -m "$KX")/kindx_controls.py" $(reader_args 1) 1 || stop "kindx controls 1"
"$PY" -I "$(cygpath -m "$KX")/kindx_controls.py" $(reader_args 1) 2 || stop "kindx controls 2"
"$PY" -I "$(cygpath -m "$KX")/kindx_read.py" $(reader_args 1) || stop "kindx read s1"
finish "$(jget READ_s1.json reading) | next: $(jget READ_s1.json committed_next)"
