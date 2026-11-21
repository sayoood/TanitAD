#!/usr/bin/env bash
# PREREG_V8F_KSCORE on box 2: does a small SCENE-READING scorer pick the best of O's 7 longitudinal hypotheses better than B's own code AND better than the best constant class?
# (B = the banked M2b checkpoints, FROZEN; the scorer is the ONLY thing that trains; seeds 0 and 1 follow B's.) ARMED ONLY ON THE PI's go for box 2 (it is the PI's personal machine; one job at a
# time under SAFE MODE). Derived from chain_v8f_mixo.sh (code/box2_ops): the predecessor wait with an ATOMIC busy marker, the code refresh from the REMOTE tip with sha-checked key files, the
# banked-checkpoint sha256 assertions, script polling (waitjob.sh), the v8f_guard.sh temperature guard, and the ZZV8FMINI-DONE / -STOP markers in ITS OWN log (chain_v8f_kscore.out).
#   chain_v8f_kscore.sh --dry-run   PRINTS the exact command every stage runs on box 2 (v8f_kscore.py's own `cmdline`: the line printed IS the line the job .cmd embeds), the checkpoint sha256
#                                   values it asserts and the order, and exits; nothing is shipped, launched or waited for.
#   chain_v8f_kscore.sh --classify-predecessor <out file> [<guard out file>]
#                                   prints what the wait loop would do with a predecessor log (PROCEED / HOLD / WAIT, with the reason) and exits 0 / 1 / 2. The wait loop calls the SAME function.
#   0. wait for the predecessor (default: FLAT's chain, chain_v8f_flat.out; box-2 order MIXO -> FLAT -> KSCORE -> OACC):
#        ZZV8FMINI-DONE                                 -> PROCEED
#        a ZZV8FMINI-STOP that is a RUN-level stop (a FLAT gate / train / eval job returned rc != 0: "<job> rc=<n>"), a VERDICT-level stop (FLAT's reader refused: "flat read") or a GATE-level
#        stop (FLAT's launch gate refused: "gate FLAT assertions") -> PROCEED, PROVIDED the log carries no box-level evidence (ZZGUARD-STOP, ZZCRASH, ZZUNREACHABLE, ZZTIMEOUT)
#        every other stop (guard, thermal, reboot, unreachable, timeout, busy marker, a failed ship / launch / pull / hash, a pre-flight check, the predecessor's own held predecessor) and ANY
#        stop this predicate does not recognise (FAIL CLOSED) -> HOLD: this chain stops with ZZV8FMINI-STOP, because the box is not known to be healthy. A predecessor that ended on its own
#        verdict must not leave box 2 idle for hours (an unconditional "any STOP stops me" idled a GPU for 8.3 h); a predecessor that left the box unhealthy must not hand it a 10-hour job
#      AFTER_OUT / AFTER_DONE / AFTER_STOP / AFTER_GUARD override the log and the markers; then the busy marker is taken ATOMICALLY (noclobber)
#   1. refresh box 2's code from the REMOTE tip; sha-check the key files; write TIP_V8FKSCORE.txt (the commit the gate token is bound to). v8f_guard.sh must know v8f_kscore (WIRING.patch)
#   2. B = M2b is fixed in the PREREG: box 2's two M2b checkpoints ARE the banked runs' (sha256); a different file stops the chain while box 2 is idle
#   3. HYPOTHESES (streamed clip by clip, resumable shards, no merged file): eval s0, train s0, eval s1, train s1. The train split is the TRAINER's own window list (4,512 clips, 306,566 windows)
#   4. THE SCORER LAUNCH GATE (`fit --gate-only`) for s0, s0-permuted, s1, s1-permuted: G-DVB on every flag, G-LIVE (CE finite, scorer gradient non-zero, B's parameters EXACTLY 0.0 with a positive
#      control, the scene-zeroed control EXACTLY 0.0), G-INPUTS (the shards are what B makes). The selfcheck and the PASS token are pulled and asserted HERE (`v8f_kscore.py assert-gate`);
#      no PASS => no training. The trainer-side `fit --require-token` refuses as well (ZZDVB-REFUSEDZZ)
#   5. FIT s0, s0-permuted (control 2), s1, s1-permuted; EVAL both seeds (the scorer, the permuted scorer, the scorer FORCED to T's argmax = control 1); READ
#      (the PREREG outcome table verbatim; rc 2 on a failed stop-control or any state the table does not map) -> READ.json + KSCORE_RESULT.md pulled back
# Stops on a reboot (waitjob 2), unreachable (3), timeout (4), any nonzero rc, or a guard stop. Script polling, not model turns.
# TIMEOUTS: hyps on the train split is ESTIMATED at 1-3 h per seed on the GPU, 3-4 h on box 2's CPU in safe mode (khyp.py MEASURED 34 ms/window on the dev box, 9,582 windows in 326 s);
# the wait is 1,320 polls of 30 s = 11 h. The fits are minutes (a 0.2 M-parameter head, 3,594 steps).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=D:/tanitad/runs/v8f_mini_full
KRUN=D:/tanitad/runs/v8f_kscore
CKPT_SHA=(601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
PYX=${KSCORE_PYX:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
KSPY=${KSCORE_PY:-$HERE/v8f_kscore.py}
ksm() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
# the stage table: <job> <polls of 30 s> <v8f_kscore.py stage arguments>. COMMIT is appended to the gate / fit stages (the token is bound to it)
stage_table() {
  cat <<'EOF'
hyps_eval_0 600 hyps --seed 0 --split eval
hyps_train_0 1320 hyps --seed 0 --split train
hyps_eval_1 600 hyps --seed 1 --split eval
hyps_train_1 1320 hyps --seed 1 --split train
gate_s0 240 fit --seed 0 --gate-only @COMMIT
gate_s0_perm 240 fit --seed 0 --perm-labels --gate-only @COMMIT
gate_s1 240 fit --seed 1 --gate-only @COMMIT
gate_s1_perm 240 fit --seed 1 --perm-labels --gate-only @COMMIT
fit_s0 240 fit --seed 0 --require-token @COMMIT
fit_s0_perm 240 fit --seed 0 --perm-labels --require-token @COMMIT
fit_s1 240 fit --seed 1 --require-token @COMMIT
fit_s1_perm 240 fit --seed 1 --perm-labels --require-token @COMMIT
eval 240 eval --seeds 0 1
read 240 read
EOF
}
stage_args() { local jn=$1 c=${2:-}; stage_table | awk -v j="$jn" '$1==j {for (i=3;i<=NF;i++) printf "%s ", $i; print ""}' | sed "s/@COMMIT/${c:+--commit $c}/" | sed 's/[[:space:]]*$//'; }
# ---- the predecessor predicate (PREREG_V8F_KSCORE "Compute and order"). ONE function: the wait loop below calls it, and so does `--classify-predecessor`, which a test runs on synthetic logs.
# It sits BEFORE every statement that touches box state (the mkdir of $E, the ssh / scp definitions), so the classify mode has no side effect.
AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
# What the predecessor's STOP may say for this chain to proceed. The predecessor is FLAT's chain (chain_v8f_flat.sh); EVERY text it can print after ZZV8FMINI-STOP is its own `stop "<msg>"`
# (43 calls, 42 distinct texts), one of three direct `echo ZZV8FMINI-STOP ...` lines, or (--dry-run only) a fourth. Classified here by what they say about the BOX (not about the arm); a test
# (test_v8f_kscore.py) lists all 42 + 3 and fails if FLAT's chain grows a text this table has not classified:
#   RUN     PROCEED  "<job> rc=<n>"        a job that ran FLAT's code on box 2 came back nonzero: the gate, a training or an eval job. The job name is FLAT's own, v8fflat<MMDDHHMM>_ followed by
#                                          gateflat | t_M2b_s0_flat | t_M2b_s1_flat | e_M2b_s0_flat | e_M2b_s1_flat (queue() in the chain). rc must be a positive integer: "rc=" (nothing parsed),
#                                          "rc=0" and a negative / signed value (a native crash) are not a run's own failure and HOLD. THE REFRESH JOB'S rc DOES NOT PROCEED: it runs no model code
#                                          (a tar into box 2's repo), so its failure is the box's (disk, tar, a half-extracted tree)
#   VERDICT PROCEED  "flat read"           FLAT's reader refused its inputs (rc 2): the read, not the box, failed
#   GATE    PROCEED  "gate FLAT assertions"  FLAT's launch gate did not pass: nothing was trained
#   PREFLIGHT PROCEED  the 21 pre-flight texts below: FLAT's chain stopped on the DEV BOX (the mirror, `git archive`, a local `cp` / `tar`, the tip's source) before box 2 was asked to run anything,
#                    so they say nothing about box 2's health; and KSCORE runs its own pre-flight on its own tip, so a cause it shares with FLAT (a stale mirror, a full dev-box disk) stops it
#                    there. Holding on them would leave box 2 idle for a night on a dev-box staging problem (MM 2026-10-25: memory 'chain refusal must not stall the pod').
#   BOX     HOLD     everything else, in the groups below; ANY text not listed is treated the same (FAIL CLOSED: the box is not known to be healthy):
#     transport / box   "<job> wait=<n> (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)", "box 2 unreachable", "ship <job>", "launch <job>", "pull <path>", "ship the code archive", "ship refresh job",
#                       "launch refresh", "<refresh job> rc=<n>", "cannot hash box 2's M2b_s<n>[_flat] checkpoint ...", "no fresh telemetry (<t1> / <t2>): the temperature guard would be blind"
#     box state         "guard stop" (thermal / time), "box 2 <file> != tip <sha>", "box 2 v9 <file> md5 != <md5>", "<phase>: box 2's M2b_s<n> checkpoint sha256 <a> != the banked run's <b>",
#                       "M2b_s<n>_flat's checkpoint changed during the evals: <a> != <b>"
#     marker / chain    "busy marker present: ...", "busy marker taken while arming: ...", "the predecessor's end does not clear the box: HOLD ..." (FLAT's own predecessor held), "write the gate token",
#                       "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
#   (pre-flight, PROCEED as above)  the tip or the local staging is not what FLAT expects (the box was never asked to run FLAT's code): "remote tip <sha> is not in the mirror", "archive", "extract the reader",
#                       "the tip's queue has no <stage> stage", "the tip's queue prints no gateflat line", "the tip has no <file>", "the tip has no per-window strata file", "the tip has no B1 lead block",
#                       "the tip's built_levers does not record o_interface | p_gt_start | p_gt_end", "the tip's v8f_mini.py has no O_INTERFACE_MODES | cut line", "the tip's trainer has no --o-interface
#                       flag | G-DVB entry (FLAG_CHECKS) for --o-interface", "the tip's trainer does not run the o_interface probe in its in-process gate", "stage the banked M2b_s<n>/<file>",
#                       "stage eval_M2b_s<n>", "stage <file>", "stage floors_perwindow.npz (the P10 reproduction guard)", "the reader cannot use the M2b side (check-inputs)"
#     dry-run only      "no queue at the tip" (printed by `chain_v8f_flat.sh --dry-run`, never by a real run)
# Independent of the message, a log that carries ZZGUARD-STOP (the temperature / time guard), ZZCRASH (a reboot), ZZUNREACHABLE or ZZTIMEOUT (waitjob.sh) holds: a run-level `rc=1` that came with a
# thermal stop or a reboot is the box's failure. All four are printed at column 0 (v8f_guard.sh lines 28 / 56, waitjob.sh lines 9 / 10 / 14). A DONE anywhere in the log wins over everything.
# Re-pointing AFTER_OUT at a different predecessor needs THAT chain's texts classified: these regexes are FLAT's, and a text they do not know HOLDS.
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
if [ "${1:-}" = "--dry-run" ]; then
  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<logs>/chain_v8f_flat.out}: ^${AFTER_DONE:-ZZV8FMINI-DONE} (stop on ^${AFTER_STOP:-ZZV8FMINI-STOP})"
  echo "== checkpoint sha256 (asserted on box 2 before anything runs)"
  for s in 0 1; do echo "M2b_s$s $R/M2b_s$s/ckpt_last.pt ${CKPT_SHA[$s]}"; done
  C=${KSCORE_COMMIT_DRY:-<TIP>}
  while read -r jn polls _; do
    echo "== $jn (polls $polls)"
    # shellcheck disable=SC2046
    "$PYX" "$(ksm "$KSPY")" cmdline $(stage_args "$jn" "$C") || exit 1
  done < <(stage_table)
  echo "== assert-gate"
  for s in 0 1; do echo "v8f_kscore.py assert-gate --selfcheck <fit>/s$s/selfcheck.json --token <fit>/s$s/gate_token.json --seed $s"; echo "v8f_kscore.py assert-gate --selfcheck <fit>/s${s}_perm/selfcheck.json --token <fit>/s${s}_perm/gate_token.json --seed $s --perm"; done
  exit 0
fi
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_kscore; mkdir -p $E/fit $E/eval
CK=$E/ckpt_sha256.json
MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803
B1="TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
TAG=$(date '+%m%d%H%M')
CUR=$L/v8fkscore_current.txt
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}; AFTER_GUARD=${AFTER_GUARD:-$L/v8fflat_guard.out}   # AFTER_DONE / AFTER_STOP are set above, with the predicate
boot() { $SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1; }
launch() { for t in 1 2 3; do o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\\$1.cmd -Log D:\tanitad\logs\\$1.log" 2>/dev/null | tr -d '\r') && { echo "$(date '+%m-%d %H:%M') $o"; return 0; }; sleep 20; done; return 1; }
clean() { tr '\r' '\n' < "$L/$1.log"; }
stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; exit 1; }
waitrc() {   # waitrc <job> <polls of 30 s>
  bash $B/waitjob.sh $1 $2 "$BOOT0"; local w=$?
  [ $w = 0 ] || stop "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
  local rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\([-0-9]*\).*/\1/p')
  echo "$(date '+%m-%d %H:%M') $1 rc=$rc"
  [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback|STOPPED|UNMAPPED' | tail -8; grep ZZGUARD-STOP $L/v8fkscore_guard.out 2>/dev/null; stop "$1 rc=$rc (the read's rc 2 = a failed stop-control or an UNMAPPED state: $KRUN/READ.json on box 2)"; }
}
kjob() {     # kjob <stage name from the table> [commit]: every job name carries this chain's TAG, so an old log's ZZJOB-END can never satisfy a new wait
  local st=$1 polls args j
  polls=$(stage_table | awk -v j="$st" '$1==j {print $2}'); args=$(stage_args "$st" "${2:-}")
  [ -n "$polls" ] && [ -n "$args" ] || stop "no such stage $st"
  j="v8fkscore${TAG}_$st"
  # shellcheck disable=SC2086
  "$PYX" "$(ksm "$KSPY")" mkjob "$j" $args > $B/$j.cmd && $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship $j"
  rm -f $L/$j.log
  launch $j || stop "launch $j"
  waitrc $j $polls
}
rsha() { $SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm $2 -LiteralPath '$1').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{32,64}$'; }
pull() {     # pull <box-2 path> <local dir>
  mkdir -p "$2"; $SCP "tanitad-box2:$1" "$2/" || stop "pull $1"
}
ckptsha_b() {   # box 2's two M2b checkpoints ARE the banked runs'
  local s g
  for s in 0 1; do
    g=$(rsha "$R/M2b_s$s/ckpt_last.pt" SHA256); [ -n "$g" ] || stop "$1: cannot hash box 2's M2b_s$s checkpoint"
    [ "$g" = "${CKPT_SHA[$s]}" ] || stop "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}"
  done
  echo "$(date '+%m-%d %H:%M') $1: both M2b checkpoints are the banked runs' (sha256 asserted)"
}
finish() {   # finish <summary>: the committed end of the chain
  $SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
  $SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8fkscore.csv 2>/dev/null
  echo "" > $CUR
  rm -f $B/BOX2_BUSY.txt
  echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)"
  exit 0
}
echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT)"
while :; do
  d=$(after_decision "$AFTER_OUT" "$AFTER_GUARD"); r=$?
  if [ $r = 0 ]; then echo "$(date '+%m-%d %H:%M') predecessor: $d"; break; fi
  if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor's end does not clear the box: $d"; exit 1; fi
  sleep 300
done
[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present: $(cat $B/BOX2_BUSY.txt)"; exit 1; }
# the marker is taken ATOMICALLY (noclobber): two chains that wait on the same predecessor cannot both pass the test above and both write it
( set -C; echo "v8F-mini PREREG_V8F_KSCORE (a scene-reading scorer over O's 7 hypotheses, B = M2b frozen; seeds 0 and 1) since $(date '+%m-%d %H:%M'), hyps ~1-3 h per seed. Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZV8FMINI-STOP busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)"; exit 1; }
BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable"
echo "$(date '+%m-%d %H:%M') boot0=$BOOT0"
# ---- 1. code refresh from the remote tip
git -C $MIRROR fetch -q origin $BR; T=$(git -C $MIRROR ls-remote origin $BR | cut -f1)
git -C $MIRROR cat-file -e "$T^{commit}" || stop "remote tip $T is not in the mirror"
for f in stack/scripts/v8f_kscore.py stack/scripts/chain_v8f_kscore.sh "$PK/code/v8f_dk_read.py" "$PK/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz" "$PK/raw/eval_grid_windows.json" "$PK/raw/eval141_kin.npz" "$PK/raw/join_eval141.npz"; do
  git -C $MIRROR cat-file -e "$T:$f" || stop "the tip has no $f"
done
git -C $MIRROR show "$T:stack/scripts/v8f_kscore.py" | grep -q '^FLAG_CHECKS = {' || stop "the tip's v8f_kscore.py has no FLAG_CHECKS (the gate would refuse every flag)"
git -C $MIRROR show "$T:$PK/code/box2_ops/v8f_guard.sh" | grep -q 'v8f_kscore' || stop "the tip's v8f_guard.sh does not know v8f_kscore (WIRING.patch not applied): the temperature guard could not stop our python"
git -C $MIRROR -c core.autocrlf=false archive --format=tar -o $B/repo_v8fkscore_tip.tar $T -- stack taniteval "$PK" "$B1" || stop "archive"
$SCP $B/repo_v8fkscore_tip.tar tanitad-box2:D:/tanitad/repo_v8fkscore_tip.tar || stop "ship the code archive"
printf '@echo off\r\necho ===== refresh %s %%DATE%% %%TIME%%\r\ntar -xf D:\\tanitad\\repo_v8fkscore_tip.tar -C D:\\tanitad\\repo\r\nset RC=%%ERRORLEVEL%%\r\necho %s> D:\\tanitad\\repo\\TIP_V8FKSCORE.txt\r\necho ===== ZZJOB-END %%DATE%% %%TIME%% rc=%%RC%%\r\n' "$T" "$T" > $B/v8fkscore${TAG}_refresh.cmd
$SCP $B/v8fkscore${TAG}_refresh.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship refresh job"
launch v8fkscore${TAG}_refresh || stop "launch refresh"
waitrc v8fkscore${TAG}_refresh 40
for f in stack/scripts/v8f_kscore.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/lead_metrics.py \
         "$PK/code/v8f_dk_read.py" "$PK/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz" "$PK/raw/eval141_kin.npz"; do
  l=$(git -C $MIRROR show "$T:$f" | sha256sum | cut -c1-64)
  [ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T"
done
echo "$(date '+%m-%d %H:%M') box 2 code = tip $T (key files sha-checked)"
# ---- 2. the base is fixed (B = M2b): its checkpoints ARE the banked runs' (box 2 is idle: a different file costs nothing)
ckptsha_b "before anything runs"
[ -n "$(rsha "$R/join_train.npz" SHA256)" ] || stop "box 2 has no $R/join_train.npz (built by the first trainer run on the full corpus)"
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }\"" 2>/dev/null
launch telemetry_long || echo "$(date '+%m-%d %H:%M') telemetry did not launch"
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" 2>/dev/null | tr -d '\r'; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry ($t1 / $t2): the temperature guard would be blind"
grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers (anchored): it would never exit"
grep -qF 'v8f_kscore' $B/v8f_guard.sh || stop "v8f_guard.sh (box2_ship) does not kill v8f_kscore: copy the tip's v8f_guard.sh over it first"
echo "" > $CUR
V8F_CHAIN_OUT=$L/chain_v8f_kscore.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh > $L/v8fkscore_guard.out 2>&1 &
# ---- 3. the hypotheses (the long stages; one job at a time)
for st in hyps_eval_0 hyps_train_0 hyps_eval_1 hyps_train_1; do
  kjob $st
  grep ZZGUARD-STOP $L/v8fkscore_guard.out 2>/dev/null && stop "guard stop"
done
# ---- 4. THE SCORER LAUNCH GATE, then the fits: no PASS => no training (asserted here AND refused by fit --require-token)
for sp in "0:" "0:_perm" "1:" "1:_perm"; do
  s=${sp%%:*}; p=${sp##*:}
  kjob "gate_s$s$p" $T
  mkdir -p $E/fit/s$s$p
  for f in selfcheck.json gate_token.json sel_const.json; do pull "$KRUN/fit/s$s$p/$f" "$E/fit/s$s$p"; done
  PYTHONIOENCODING=utf-8 "$PYX" "$(ksm "$KSPY")" assert-gate --selfcheck "$(ksm $E/fit/s$s$p/selfcheck.json)" --token "$(ksm $E/fit/s$s$p/gate_token.json)" --seed $s ${p:+--perm} || stop "gate s$s$p assertions"
done
for sp in "0:" "0:_perm" "1:" "1:_perm"; do
  s=${sp%%:*}; p=${sp##*:}
  echo "../v8f_kscore/fit/s$s$p" > $CUR        # v8f_guard.sh reads D:/tanitad/runs/v8f_mini_full/$(cat $CUR)/{train_log.jsonl,selfcheck.json}: this resolves to the fit dir
  kjob "fit_s$s$p" $T
  echo "" > $CUR
  for f in summary.json argv.json train_log.jsonl ckpt_last.pt; do pull "$KRUN/fit/s$s$p/$f" "$E/fit/s$s$p"; done
  grep ZZGUARD-STOP $L/v8fkscore_guard.out 2>/dev/null && stop "guard stop"
done
# ---- 5. EVAL and READ
kjob eval
kjob read
for f in READ.json KSCORE_RESULT.md kscore_numbers.json; do pull "$KRUN/$f" "$E"; done
jget() { "$PYX" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['decision'][sys.argv[2]])" "$(ksm $E/READ.json)" $1; }
finish "$(jget outcome): $(jget committed_next_step_quoted)"
