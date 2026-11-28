#!/usr/bin/env bash
# PREREG_V8F_OACC on box 2: does training O to match the human's ACCELERATION (X = M2b + --w-acc <rule>, ONE lever) make the deployed v8F output (HX = X with HBLEND's per-step weights re-fitted on X's own training windows) dominate
# its kinematic prior P10-yaw-4512? B = M2b, fixed in the PREREG. QUEUED AFTER FLAT's chain on box 2 and ARMED ONLY ON THE PI's go for box 2. One job at a time under SAFE MODE (mk_v8f_job.py). Derived from chain_v8f_flat.sh (the training chain:
# predecessor wait with an atomic busy marker, refresh, checkpoint sha256 assertions, check-inputs on the real B side, guard, gate, seeds, read). Markers ZZV8FMINI-DONE / -STOP in ITS OWN log (chain_v8f_oacc.out), which v8f_guard.sh's
# anchored end test already knows.
#   chain_v8f_oacc.sh --dry-run   PRINTS the exact command every stage runs against box 2 (the queue's own `waccoacc` / `gateoacc` / `trainoacc` / `evaloacc` / `fitoacc` lines, from the tip's queue or $V8F_QUEUE), the checkpoint sha256 values it
#                                 asserts and the reader's arguments, and exits; nothing is shipped, launched or waited for. The line printed IS the line that runs. (Before the rule has run, the gate / train lines carry the placeholder
#                                 WACC_FROM_RULE; export WACC=<value> to see the real argv.)
#   chain_v8f_oacc.sh --classify-predecessor <out file> [<guard out file>]
#                                 prints what the wait loop would do with a predecessor log (PROCEED / HOLD / WAIT, with the reason) and exits 0 / 1 / 2. The wait loop calls the SAME function.
#   0. wait for the predecessor (default: PREREG_V8F_FLAT's chain, chain_v8f_flat.out) -- PREREG "Compute and order" (after FLAT s0 / s1, before DSEFF):
#        ZZV8FMINI-DONE ANYWHERE in the log                -> PROCEED
#        a ZZV8FMINI-STOP that is a VERDICT-level stop ("flat read"), a RUN-level stop (a box-2 job of FLAT returned rc != 0), a GATE-level stop ("gate FLAT assertions") or a PRE-FLIGHT stop (on the dev box, before box 2 ran
#        anything: the same list chain_v8f_kscore.sh uses) -> PROCEED, PROVIDED the log carries no box-level evidence (ZZGUARD-STOP, ZZCRASH, ZZUNREACHABLE, ZZTIMEOUT)
#        a guard, thermal, reboot or unreachable STOP, a timeout, a busy marker, a failed ship / launch / pull / hash, and ANY stop this predicate does not recognise (FAIL CLOSED) -> HOLD: this chain stops with ZZV8FMINI-STOP
#      AFTER_OUT / AFTER_DONE / AFTER_STOP / AFTER_GUARD override the log and the markers; the busy marker is then taken ATOMICALLY (noclobber)
#   1. refresh box 2's code from the REMOTE tip; sha-check the key files (the trainer, the eval, the model, the data, hjepa, the queue, the OACC eval driver, HBLEND's fit set and banked arrays, the P10 prior and predictions), md5-check both
#      v9 releases; extract the reader HERE from the same archive
#   2. B = M2b is fixed in the PREREG: box 2's two M2b checkpoints ARE the banked runs' (sha256, raw/full/MANIFEST.md); a different file stops the chain while box 2 is idle
#   3. the B side is staged from the banked M2b files and `v8f_oacc_read.py --check-inputs` runs on it BEFORE any training: runs, evals, F reproducing 0.1207, P10's ADE6 reading 2.4870, the committed window counts (2,399 / 141)
#   4. THE RULE (`waccoacc`): w_acc = 0.5 * L_traj(B_s0) / L_acc(B_s0) on HBLEND's fit windows with B's s0 checkpoint, CPU on box 2, cross-checked against HBLEND's banked arrays (rel 1e-3). The value is pulled HERE, asserted (schema, cross-check
#      pass, B_s0's sha256, 9,582 windows, w_acc x L_acc / L_traj = 0.5) and is the ONLY value that reaches X's argv: the queue reads it from the file the stage wrote
#   5. the in-process launch gate for X (`gateoacc`), pulled and asserted HERE: the PASS token (declared_vs_built PASS, G-LIVE pass with a frozen-prior control of exactly 0.0, no dead weight, the acceleration term in the loss dict, acc_probe pass for
#      THIS w_acc: the total moves by w_acc x the term, O's head receives the term's gradient, every control exactly 0.0), 4,512 clips, the RA-1 holdout withheld, B's clip list and WINDOW SET (306,566 windows) and 19,164 steps, the BUILT w_acc,
#      B's levers / weights / p_gt schedule (1.0 -> 0.2) / hist 4 / hmax 16 / window start 3 / code_mix whole / o_interface full, and X's parameter count = B's EXACTLY. The queue's own printed line carries `--w-acc <the rule's value>` exactly once
#      and none of the other levers' flags. A token record (gate_token.json) is written beside it. No PASS => no training (the trainer's own in-process gate refuses as well, ZZDVB-REFUSEDZZ)
#   6. seed 0 (`trainoacc 0`) -> its standard eval (`evaloacc 0`) -> HX's weights re-fitted on X_s0's own training windows (`fitoacc 0`); the checkpoint sha256 is measured after training AND after the evals (one value)
#   7. `v8f_oacc_read.py --seeds 0` -> oacc_read_s0.json: PRIMARY-ACC, PRIMARY-KEEP, DOMINANCE. SEED 1 RUNS ONLY IF s0 passes PRIMARY-ACC and PRIMARY-KEEP (the reader's `s1_required`); otherwise the s0 read IS the outcome (NO EFFECT / COSTS POSITION)
#   8. seed 1 (conditional): `trainoacc 1`, `evaloacc 1`, `fitoacc 1`, then `v8f_oacc_read.py --seeds 0,1` -> oacc_read.json: the PREREG's outcome table verbatim (DOMINATES ITS PRIOR | SMOOTHER, NOT DOMINANT | COSTS POSITION | NO EFFECT)
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: the training wait is 1,320 polls of 30 s = 11 h, one hour above the guard's 10 h cap (V8F_MAX_RUN_H); X trains M2b's graph plus one cheap loss term, so a seed is ESTIMATED at B's own wall-clock (banked summary.json `seconds`,
# about 4.4 h), and v8f_guard.sh stops any run it PROJECTS above the cap. The eval is ~2-4 min (wait 600 polls = 5 h), the rule and the two HX fits are CPU forward passes over 9,582 windows (~1-5 min each, wait 120 polls = 1 h).
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
# the banked checkpoints (raw/full/MANIFEST.md); the reader pins the same two values (BANKED_CKPT_SHA) and a test pins that they agree
CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
# ---- the predecessor predicate (PREREG_V8F_OACC "Compute and order"). ONE function: the wait loop below calls it, and so does `--classify-predecessor`, which a test runs on synthetic logs.
AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
# What a predecessor's STOP may say for this chain to proceed. The messages are FLAT's chain's own `stop "<msg>"` texts (chain_v8f_flat.sh):
#   run-level      "v8fflat<MMDDhhmm>_(gateflat|t_M2b_s<n>_flat|e_M2b_s<n>_flat) rc=<n>"   a box-2 job of FLAT came back with a nonzero rc (a trainer / eval failure, with no box-level evidence in the logs)
#   verdict-level  "flat read"                                                              the reader refused its inputs (rc 2): the read, not the box, failed
#   gate-level     "gate FLAT assertions"                                                    the launch gate did not pass: nothing was trained
#   pre-flight     the list below                                                           a stop on the dev box BEFORE box 2 ran anything (the same list chain_v8f_kscore.sh uses)
# ANYTHING ELSE holds: "<job> wait=<n> (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", "box 2 unreachable", "no fresh telemetry", "busy marker ...", a failed ship / launch / pull / hash, "guard stop", FLAT's own refresh job failing,
# the predecessor's own predecessor, and any text this list has never seen (FAIL CLOSED: the box is not known to be healthy). Independent of the message, a log that carries ZZGUARD-STOP (the temperature / time guard), ZZCRASH (a reboot),
# ZZUNREACHABLE or ZZTIMEOUT (waitjob.sh) holds.
RUN_LEVEL_RE='^v8fflat[0-9]{8}_(gateflat|t_M2b_s[01]_flat|e_M2b_s[01]_flat) rc=[1-9][0-9]*$'
VERDICT_LEVEL_RE='^flat read$'
GATE_LEVEL_RE='^gate FLAT assertions$'
PREFLIGHT_LEVEL_RE="^(archive|extract the reader|remote tip [0-9a-f]{7,40} is not in the mirror|the reader cannot use the M2b side \(check-inputs\)|stage the banked M2b_s[01]/(summary|argv|selfcheck)\.json|stage eval_M2b_s[01]|stage eval_M2b_s[01]-(tacabs\.npz|tacabs\.json|tacorc\.npz|tacorc-all-goal\.npz|tacorc-all-latcons\.npz)|stage floors_perwindow\.npz \(the P10 reproduction guard\)|the tip's queue has no (gateflat|trainflat|evalflat) stage|the tip's queue prints no gateflat line|the tip has no (v8f_flat_read|v8f_kind_read|v8f_grid_read|v8f_pgt_read|v8f_dk_read|v8f_wc_read|v8f_tac_groups_read|v8f_mini_full_report)\.py|the tip has no per-window strata file|the tip has no B1 lead block|the tip's built_levers does not record (o_interface|p_gt_start|p_gt_end)|the tip's v8f_mini\.py has no (O_INTERFACE_MODES|cut line)|the tip's trainer has no --o-interface flag|the tip's trainer has no G-DVB entry \(FLAG_CHECKS\) for --o-interface|the tip's trainer does not run the o_interface probe in its in-process gate)\$"
after_decision() {   # after_decision <predecessor chain log> [<its guard log>]: prints the decision; rc 0 = PROCEED, 1 = HOLD, 2 = WAIT (no end marker yet)
  local out=$1 guard=${2:-} stop msg ev
  if grep -a -q "^$AFTER_DONE" "$out" 2>/dev/null; then echo "PROCEED the predecessor ended: $(grep -a "^$AFTER_DONE" "$out" | tail -1 | tr -d '\r' | cut -c1-160)"; return 0; fi
  stop=$(grep -a "^$AFTER_STOP" "$out" 2>/dev/null | tail -1 | tr -d '\r')
  [ -n "$stop" ] || { echo "WAIT no $AFTER_DONE / $AFTER_STOP in $out yet"; return 2; }
  msg=${stop#"$AFTER_STOP"}; msg=${msg# }
  ev=$(grep -a -h -E '^(ZZGUARD-STOP|ZZCRASH|ZZUNREACHABLE|ZZTIMEOUT)' "$out" ${guard:+"$guard"} 2>/dev/null | head -1 | tr -d '\r')
  [ -z "$ev" ] || { echo "HOLD box-level evidence in the predecessor's logs: $ev (its stop: $msg)"; return 1; }
  if [[ $msg =~ $RUN_LEVEL_RE ]]; then echo "PROCEED run-level stop: $msg"; return 0; fi
  if [[ $msg =~ $VERDICT_LEVEL_RE ]]; then echo "PROCEED verdict-level stop: $msg"; return 0; fi
  if [[ $msg =~ $GATE_LEVEL_RE ]]; then echo "PROCEED gate-level stop (the launch gate refused, nothing was trained): $msg"; return 0; fi
  if [[ $msg =~ $PREFLIGHT_LEVEL_RE ]]; then echo "PROCEED pre-flight stop (on the dev box, before box 2 ran anything; this chain re-checks its own tip): $msg"; return 0; fi
  if [[ $msg =~ ^v8fflat[0-9]{8}_refresh\ rc= ]]; then echo "HOLD box-level: FLAT's code-refresh job (a tar into box 2's repo, no model code) failed: $msg"; return 1; fi
  echo "HOLD a stop that is not recognised as verdict-, gate- or run-level (fail closed): $msg"; return 1
}
if [ "${1:-}" = "--classify-predecessor" ]; then after_decision "${2:?usage: --classify-predecessor <out file> [<guard out file>]}" "${3:-}"; exit $?; fi
dry_stage() {   # dry_stage <queue file> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN. cmd.exe cannot be handed `"<path with an ampersand>" args`
                # through MSYS quoting (it strips the quotes it needs): a one-line wrapper batch file can
  local qf=$1; shift
  local w=${TMPDIR:-/tmp}/v8foacc_dry_$$.cmd
  printf '@echo off\r\nset V8F_DRYRUN=1\r\ncall "%s" %s\r\n' "$(cygpath -w "$qf")" "$*" > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
}
reader_args() {   # reader_args <eval dir> <runs dir> <weights dir> <w_acc json> <ckpt sha json> <seeds> <out json>: the one argument string of a read, printed by --dry-run and run by the chain
  echo "--eval-dir $1 --runs-dir $2 --weights-dir $3 --wacc-json $4 --ckpt-sha $5 --seeds $6 --out $7"
}
if [ "${1:-}" = "--dry-run" ]; then
  QF=${V8F_QUEUE:-}
  if [ -z "$QF" ]; then
    MIRROR=${MIRROR:-/c/Users/Admin/tanitad-push}; BR=${BR:-agent/arch-inf-20260803}
    T=$(git -C $MIRROR ls-remote origin $BR | cut -f1); QF=${TMPDIR:-/tmp}/v8foacc_queue_dry_$T.cmd
    git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > "$QF" || { echo "ZZV8FMINI-STOP no queue at the tip"; exit 1; }
  fi
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<logs>/chain_v8f_flat.out}: ^${AFTER_DONE:-ZZV8FMINI-DONE} (stop on ^${AFTER_STOP:-ZZV8FMINI-STOP})"
  echo "== checkpoint sha256 (asserted on box 2 before the rule and the gate; X's measured after training and after the evals)"
  for s in 0 1; do echo "M2b_s$s $R/M2b_s$s/ckpt_last.pt ${CKPT_SHA[$s]}"; done
  for st in "waccoacc" "gateoacc" "trainoacc 0" "evaloacc 0" "fitoacc 0" "trainoacc 1" "evaloacc 1" "fitoacc 1"; do
    echo "== $st"; WACC=${WACC:-} dry_stage "$QF" $st || exit 1
  done
  echo "== read"
  echo "v8f_oacc_read.py --check-inputs --eval-dir <eval> --runs-dir <runs> --out <eval>/../check_inputs.json"
  echo "v8f_oacc_read.py $(reader_args '<eval>' '<runs>' '<w>' '<eval>/../w_acc.json' '<eval>/../ckpt_sha256.json' 0 '<eval>/../oacc_read_s0.json')"
  echo "-- seed 1 runs iff oacc_read_s0.json says s1_required true (seed 0 passes PRIMARY-ACC and PRIMARY-KEEP); then:"
  echo "v8f_oacc_read.py $(reader_args '<eval>' '<runs>' '<w>' '<eval>/../w_acc.json' '<eval>/../ckpt_sha256.json' 0,1 '<eval>/../oacc_read.json')"
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_oacc; mkdir -p $E/runs $E/eval $E/w
CK=$E/ckpt_sha256.json
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
HJ="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
HB="$PK/raw/2026-10-25-hblend"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9
M2B_STEPS=19164
M2B_WINDOWS=306566   # B's n_windows_all_clips (banked summary.json); X trains on B's window LIST (the lever is a loss term only)
FIT_WINDOWS=9582     # HBLEND's fit set: 141 clips, k = 32
TRAIN_POLLS=1320; EVAL_POLLS=600; FIT_POLLS=120   # 11 h: one hour above the guard cap, so the GUARD (a committed rule), never the wait, ends a slow run
TAG=$(date '+%m%d%H%M')
CUR=$L/v8foacc_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}; AFTER_GUARD=${AFTER_GUARD:-$L/v8fflat_guard.out}
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; grep ZZGUARD-STOP $L/v8foacc_guard.out 2>/dev/null; stop "$1 rc=$rc"; }
}
queue() {    # queue <job> <polls> <queue args...>: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local j="v8foacc${TAG}_$1" m=$2; shift 2
  $PY "$(cygpath -m $B)/mk_v8f_job.py" $j "$@" >/dev/null && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $m
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path under R> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$R/$1" "$2/" || stop "pull $R/$1"
}
ckset() {    # ckset <key> <sha256>: one measured value into the json the reader takes as --ckpt-sha
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
ckptsha_x() {   # ckptsha_x <seed> <trained|after_evals>: X's checkpoint, measured on box 2; after the evals it must be the value measured after training (the scored checkpoint IS the trained one)
  local g; g=$(rsha "$R/M2b_s${1}_oacc/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "cannot hash box 2's M2b_s${1}_oacc checkpoint ($2)"
  if [ "$2" = after_evals ]; then
    local g0; g0=$($PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$(cygpath -m $CK)" "M2b-oacc_s${1}_trained")
    [ "$g" = "$g0" ] || stop "M2b_s${1}_oacc's checkpoint changed during the evals: $g != $g0"
  fi
  ckset "M2b-oacc_s${1}_$2" "$g"
  echo "$(date '+%m-%d %H:%M') M2b_s${1}_oacc checkpoint sha256 ($2) ${g:0:16}..."
}
finish() {   # finish <summary>: the committed end of the chain (also after a stage-1 STOP)
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8foacc.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
runone() {   # runone <seed>: train X, eval (the standard eval), re-fit HX's weights on X's own training windows, pull, record the checkpoint sha256
  local s=$1 g="M2b_s${1}_oacc" tg="M2b-oacc_s${1}" f x
  echo "$g" > $CUR
  queue "t_$g" $TRAIN_POLLS trainoacc $s
  echo "" > $CUR
  ckptsha_x $s trained
  queue "e_$g" $EVAL_POLLS evaloacc $s
  ckptsha_x $s after_evals
  queue "f_$g" $FIT_POLLS fitoacc $s
  for f in selfcheck.json summary.json argv.json train_log.jsonl train.log ckpt_last.pt; do pull "$g/$f" "$E/runs/$g"; done
  for x in json npz log; do pull "eval/eval_$tg.$x" "$E/eval"; done
  pull "oacc/weights_$tg.json" "$E/w"
  grep ZZGUARD-STOP $L/v8foacc_guard.out 2>/dev/null && stop "guard stop"
}
oaccread() {   # oaccread <seeds> <out>: the reader (stdout to the log; rc 2 = a refusal -> STOP)
  PYTHONIOENCODING=utf-8 $PY "$RD/v8f_oacc_read.py" $(reader_args "$(cygpath -m $E/eval)" "$(cygpath -m $E/runs)" "$(cygpath -m $E/w)" "$(cygpath -m $E/w_acc.json)" "$(cygpath -m $CK)" "$1" "$(cygpath -m $E/$2)") || stop "oacc read $1"
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
( set -C; echo "v8F-mini PREREG_V8F_OACC (X = M2b + --w-acc <rule>, B = M2b; seed 0, then seed 1 iff seed 0 passes PRIMARY-ACC and PRIMARY-KEEP, then the read) since $(date '+%m-%d %H:%M'), ~B's wall-clock per seed plus eval and fit. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
QT=$B/v8foacc_queue_$TAG.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" | tr -d '\r' > $QT
QC=$B/v8foacc_queue_$TAG.crlf.cmd; git -C $MIRROR show "$T:$PK/code/v8f_mini_box2_queue.cmd" > $QC      # the committed bytes (CRLF): what cmd.exe can run for the dry-run line
for st in waccoacc gateoacc trainoacc evaloacc fitoacc; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done
for f in v8f_oacc_read.py v8f_flat_read.py v8f_kind_read.py v8f_grid_read.py v8f_pgt_read.py v8f_dk_read.py v8f_wc_read.py v8f_tac_groups_read.py v8f_mini_full_report.py; do
  git -C $MIRROR cat-file -e "$T:$PK/code/$f" || stop "the tip has no $f"
done
for f in "stack/scripts/v8f_mini_eval_oacc.py" "$HB/v8f_hblend.py" "$HB/fit_clips.json" "$HB/fit_windows_M2b_s0.npz"; do
  git -C $MIRROR cat-file -e "$T:$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"w_acc": cfg.w_acc' || stop "the tip's built_levers does not record w_acc"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '^def accel_term(' || stop "the tip's v8f_mini.py has no accel_term"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q 'total = total + cfg.w_acc \* T\["acc"\] if cfg.w_acc > 0.0 else total' || stop "the tip's v8f_mini.py has no acceleration line in the loss total"
git -C $MIRROR show "$T:stack/tanitad/models/v8f_mini.py" | grep -q '"p_gt_start": cfg.p_gt_start' || stop "the tip's built_levers does not record p_gt_start"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--w-acc"' || stop "the tip's trainer has no --w-acc flag"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q '"w_acc": "cfg:w_acc"' || stop "the tip's trainer has no G-DVB entry (FLAG_CHECKS) for --w-acc"
git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q 'acc_probe(model, probe, cfg)' || stop "the tip's trainer does not run the acc probe in its in-process gate"
git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8foacc_tip.tar $T -- stack taniteval "$HJ" "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8foacc_tip.tar tanitad-box2:D:/tanitad/repo_v8foacc_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8foacc_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FOACC.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8foacc${TAG}_refresh.cmd
$SCP $B/v8foacc${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8foacc${TAG}_refresh || stop "launch refresh"
waitrc v8foacc${TAG}_refresh 40
KEYSHA=""
for f in stack/scripts/v8f_mini_train.py stack/scripts/v8f_mini_eval.py stack/scripts/v8f_mini_eval_oacc.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py \
         stack/tanitad/models/hjepa.py "$PK/code/v8f_mini_box2_queue.cmd" "$HB/v8f_hblend.py" "$HB/fit_clips.json" "$HB/fit_windows_M2b_s0.npz" "$HJ/raw/p10_yaw4512/p10_prior.pt" "$HJ/raw/p10_yaw4512/p10_eval_pred6.npy"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
  KEYSHA="$KEYSHA\"$f\": \"$l\", "
done
for kv in v9_labels_train.npz=83e30bd3791f0767707eade5f4671759 v9_labels_eval139.npz=6b5c7f207cffc3b7eb3cd527fd433599; do
  [ "${kv#*=}" = "$(rsha "D:/tanitad/data/v9/${kv%%=*}" MD5)" ] || stop "box 2 v9 ${kv%%=*} md5 != ${kv#*=}"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (12 key files sha-checked; both v9 releases md5-checked)"
# the reader runs HERE at the same tip, from the archive box 2 got (a fresh directory, never rm -rf)
X=$B/v8foacc_tip_$(date +%Y%m%d%H%M%S); mkdir -p $X && tar -xf $B/repo_v8foacc_tip.tar -C $X || stop "extract the reader"
RD="$(cygpath -m "$X/$PK/code")"
# ---- 2. the base is fixed (B = M2b): its checkpoints ARE the banked runs' (box 2 is idle: a different file costs nothing)
rm -f $CK
ckptsha_b "before the rule"
# ---- 3. the B side, staged into this chain's dirs BEFORE any training (a missing file stops the chain while box 2 is still idle)
for s in 0 1; do
  mkdir -p $E/runs/M2b_s$s
  for f in summary.json argv.json selfcheck.json; do cp "$L/v8f_mini/runs/M2b_s$s/$f" $E/runs/M2b_s$s/ || stop "stage the banked M2b_s$s/$f"; done
  cp "$L/v8f_mini/runs/M2b_s$s/train_log.jsonl" $E/runs/M2b_s$s/ 2>/dev/null || echo "$(date '+%m-%d %H:%M') NOTE: no banked M2b_s$s/train_log.jsonl: its p_gt schedule is NOT-CHECKED (by construction 1.0 -> 0.2)"
  cp $L/v8f_mini/eval/eval_M2b_s$s.npz $L/v8f_mini/eval/eval_M2b_s$s.json $E/eval/ || stop "stage eval_M2b_s$s"
done
cp "$L/v8f_mini/eval/floors_perwindow.npz" $E/eval/ || stop "stage floors_perwindow.npz (the P10 reproduction guard)"
# the reader must load the REAL B side BEFORE any training: runs, evals, F, P10's ADE6, the committed window counts
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_oacc_read.py" --check-inputs --eval-dir "$(cygpath -m $E/eval)" --runs-dir "$(cygpath -m $E/runs)" --out "$(cygpath -m $E/check_inputs.json)" \
  || stop "the reader cannot use the M2b side (check-inputs)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_oacc.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8foacc_guard.out 2>&1 &
# ---- 4. THE RULE: w_acc = 0.5 * L_traj(B_s0) / L_acc(B_s0), computed ONCE on box 2 (CPU), pulled and asserted here; the value that reaches X's argv is the one in this file and no other
queue waccoacc $FIT_POLLS waccoacc
$SCP "tanitad-box2:$R/oacc/w_acc.json" "$E/w_acc.json" && $SCP "tanitad-box2:$R/oacc/w_acc.txt" "$E/w_acc.txt" || stop "pull the w_acc rule files"
$PY - "$(cygpath -m $E/w_acc.json)" "$(cygpath -m $E/w_acc.txt)" ${CKPT_SHA[0]} $FIT_WINDOWS <<'EOF' || stop "w_acc rule assertions"
import json, sys
r, txt, b0, nfit = json.load(open(sys.argv[1])), open(sys.argv[2]).read().strip(), sys.argv[3], int(sys.argv[4])
bad = []
if r.get("schema") != "v8f-oacc-wacc-v1": bad.append("schema")
if not (isinstance(r.get("w_acc"), float) and r["w_acc"] > 0): bad.append(f"w_acc {r.get('w_acc')!r}")
if float(txt) != r.get("w_acc") or repr(r.get("w_acc")) != txt: bad.append(f"w_acc.txt {txt!r} != the json's {r.get('w_acc')!r}")
cc = r.get("banked_crosscheck") or {}
if cc.get("pass") is not True: bad.append(f"the cross-check against HBLEND's banked arrays: {cc.get('pass')}")
if (r.get("source") or {}).get("ckpt_sha256") != b0: bad.append("the rule was not evaluated with the banked M2b_s0 checkpoint")
if r.get("n_windows") != nfit: bad.append(f"n_windows {r.get('n_windows')} != HBLEND's fit set {nfit}")
if abs(r.get("check_w_acc_x_L_acc_over_L_traj", 0) - 0.5) > 1e-4: bad.append("w_acc x L_acc / L_traj != 0.5")
if r.get("frac") != 0.5: bad.append("frac")
print("w_acc rule:", "PASS" if not bad else bad, "| w_acc", r.get("w_acc"), "| L_traj", r.get("L_traj"), "| L_acc", r.get("L_acc"), "| n", r.get("n_windows"))
sys.exit(1 if bad else 0)
EOF
WACC=$(tr -d '\r\n' < $E/w_acc.txt); export WACC
echo "$(date '+%m-%d %H:%M') w_acc = $WACC (the rule's value; it is the only value X's argv carries)"
# ---- 5. the launch gate for X, asserted here as well as in-process. GLINE is the line the queue PRINTS for the stage: a static test pins it to the line the stage RUNS.
GLINE=$(dry_stage "$QC" gateoacc | grep -a 'v8f_mini_train.py'); [ -n "$GLINE" ] || stop "the tip's queue prints no gateoacc line"
queue gateoacc 120 gateoacc
pull "gate_M2b-oacc/selfcheck.json" "$E/gate_M2b-oacc"
$PY - "$(cygpath -m $E/gate_M2b-oacc/selfcheck.json)" $HOLDOUT_SHA $M2B_STEPS "$(cygpath -m $E/runs/M2b_s0/summary.json)" "$GLINE" $M2B_WINDOWS $WACC <<'EOF' || stop "gate OACC assertions"
import json, sys
sc, hold, steps, b, line, windows, w = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4])), sys.argv[5], int(sys.argv[6]), float(sys.argv[7])
i, lv, bl = sc["info"], sc["levers"], b["levers"]
bad = []
# THE PASS TOKEN: the trainer's in-process gate wrote these; without all of them there is no token
if sc.get("declared_vs_built") != "PASS": bad.append("declared_vs_built")
live = sc["live"]
if not live.get("pass") or live.get("control_prior_grad_sum") != 0.0: bad.append(f"live {live.get('pass')} / control {live.get('control_prior_grad_sum')}")
if live.get("dead_params") != [] or live.get("declared_dead") != []: bad.append(f"G-LIVE dead set {live.get('dead_params')} (declared {live.get('declared_dead')}) != []: the lever kills no weight")
if "acc" not in live.get("terms", {}): bad.append("G-LIVE's loss terms carry no 'acc'")
pr = sc.get("acc_probe")
if not isinstance(pr, dict) or pr.get("pass") is not True: bad.append(f"acc_probe {pr}")
else:
    if pr.get("w_acc_declared") != w or pr.get("w_probe") != w: bad.append(f"the acc probe ran for w_acc {pr.get('w_acc_declared')} / probe weight {pr.get('w_probe')}, the rule's value is {w}")
    if not pr.get("head_o_grad_diff", 0.0) > 0.0: bad.append("O's head receives no gradient from the term")
    if abs(pr.get("contribution", 0.0) - pr.get("w_x_acc", 1.0)) > 1e-3 * pr.get("w_x_acc", 1.0) + 1e-5: bad.append(f"the total moved by {pr.get('contribution')}, w_acc x the term is {pr.get('w_x_acc')}")
    for k in ("control_prior_grad_diff", "control_repeat_grad_diff", "control_repeat_total_diff"):
        if pr.get(k) != 0.0: bad.append(f"acc probe control {k} = {pr.get(k)} (must be exactly 0.0)")
    if pr.get("control_wm_grad_diff") not in (None, 0.0): bad.append("the world-model auxiliary's gradient moved with the weight")
if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (4512, 60, 60, 0): bad.append("clips / holdout counts")
if i["excluded_sha256"] != hold: bad.append("the excluded set is not the RA-1 holdout")
if i["train_sha256"] != b["train_sha256"]: bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
if i["kin_mode"] != "yaw": bad.append("kin_mode")
if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"): bad.append(f"levers {lv}")
# THE LEVER, from the BUILT model: w_acc = the rule's value; X keeps B's full interface and whole-code teacher forcing (ONE lever)
if lv.get("w_acc") != w: bad.append(f"built w_acc {lv.get('w_acc')!r} != the rule's {w!r}")
if lv.get("o_interface", "full") != "full": bad.append(f"built o_interface {lv.get('o_interface')!r} != 'full' (B's)")
if lv.get("code_mix", "whole") != "whole": bad.append(f"built code_mix {lv.get('code_mix')!r} != 'whole' (B's)")
if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (1.0, 0.2): bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 1.0 -> 0.2")
wd = lv.get("w") or {}
for k in ("w_pen", "w_pen_self", "w_wm"):
    if wd.get(k) != bl["w"][k]: bad.append(f"{k} {wd.get(k)} != B's {bl['w'][k]}")
if (wd.get("w_tac"), wd.get("w_cons")) != (1.0, 0.5): bad.append(f"w_tac / w_cons {wd.get('w_tac')} / {wd.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
# THE PREREG'S CLAIMS: B's steps and window set are X's
if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}")
if i.get("n_windows_all_clips") != windows: bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {windows}")
if sc["steps_total"] != steps: bad.append(f"steps {sc['steps_total']} != B's {steps}")
if lv.get("hist") != 4: bad.append(f"built hist {lv.get('hist')} != 4 (B's)")
if lv.get("hmax") != 16: bad.append(f"built hmax {lv.get('hmax')} != 16 (the arm's auto value, B's)")
if lv.get("window_start") != 3: bad.append(f"built window_start {lv.get('window_start')} != 3 (B's effective first window, hist - 1)")
if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (3, 3, 3): bad.append(f"window start (asked / first t of the built window list / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != (3, 3, 3)")
# THE LEVER ADDS NO PARAMETER: X's count equals B's in every group and in total
pg, pb = lv["params"], bl["params"]
if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
for k in sorted(set(pg) | set(pb)):
    if pg.get(k, 0) != pb.get(k, 0): bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
# THE QUEUE'S OWN PRINTED LINE: --w-acc <the rule's value> exactly once, --arm M2b, and none of the other levers' flags (X differs from B by this one flag)
toks = line.split()
if toks.count("--w-acc") != 1 or float(toks[toks.index("--w-acc") + 1]) != w: bad.append("the queue's gateoacc line does not carry --w-acc <the rule's value> exactly once")
if toks.count("--arm") != 1 or toks[toks.index("--arm") + 1] != "M2b": bad.append("the queue's gateoacc line is not --arm M2b")
for fl in ("--hist", "--hmax", "--window-start", "--p-gt-start", "--p-gt-end", "--grid", "--w-tac", "--w-cons", "--w-traj", "--code-mix", "--o-interface"):
    if fl in toks: bad.append(f"the queue's gateoacc line sets {fl}: a second lever beside --w-acc")
if not i.get("lru"): bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
print("OACC gate:", "PASS" if not bad else bad, "| clips", i["n_clips"], "steps", sc["steps_total"], "windows", i.get("n_windows_all_clips"), "| built w_acc", lv.get("w_acc"), "| probe contribution", (pr or {}).get("contribution"),
      "head grad diff", (pr or {}).get("head_o_grad_diff"), "| params", lv["n_params_trainable"], "(B", bl["n_params_trainable"], ") | lru", i.get("lru"))
sys.exit(1 if bad else 0)
EOF
# the token record: the exact commit, the sha256 of the queue and of the gate's selfcheck, the key files box 2 was checked against, and the data manifests the gate read (written by the chain, never by hand)
$PY - "$(cygpath -m $E/gate_M2b-oacc/selfcheck.json)" "$(cygpath -m $E/gate_token.json)" "$T" "$(sha256sum $QT | cut -c1-64)" "{${KEYSHA%, }}" "$WACC" <<'EOF' || stop "write the gate token"
import hashlib, json, sys
sc = json.load(open(sys.argv[1]))
tok = {"prereg": "PREREG_V8F_OACC", "gate": "PASS", "commit": sys.argv[3], "queue_sha256": sys.argv[4], "selfcheck_sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(),
       "argv_sha": sc.get("argv_sha"), "w_acc": float(sys.argv[6]), "key_files_sha256": json.loads(sys.argv[5]), "train_sha256": sc["info"]["train_sha256"], "excluded_sha256": sc["info"]["excluded_sha256"],
       "v9_md5": {"v9_labels_train.npz": "83e30bd3791f0767707eade5f4671759", "v9_labels_eval139.npz": "6b5c7f207cffc3b7eb3cd527fd433599"}}
json.dump(tok, open(sys.argv[2], "w"), indent=1)
print("gate token written for commit", tok["commit"])
EOF
# ---- 6. seed 0: trained, evaluated, HX's weights re-fitted; NOTHING is decided before the read
runone 0
# ---- 7. the seed-0 read: PRIMARY-ACC, PRIMARY-KEEP, DOMINANCE. Seed 1 runs ONLY IF seed 0 passes PRIMARY-ACC and PRIMARY-KEEP (the PREREG): the reader says so (s1_required)
oaccread 0 oacc_read_s0.json
if [ "$(jget oacc_read_s0.json s1_required)" != "True" ]; then
  finish "$(jget oacc_read_s0.json reading) | $(jget oacc_read_s0.json next) | seed 1 NOT run: seed 0 failed PRIMARY-ACC or PRIMARY-KEEP"
fi
# ---- 8. seed 1 (conditional), then the final read: the PREREG's outcome table verbatim (a refusal is rc 2 -> STOP)
runone 1
oaccread 0,1 oacc_read.json
finish "$(jget oacc_read.json reading) | $(jget oacc_read.json next)"
