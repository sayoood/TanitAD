#!/usr/bin/env bash
# PREREG_V8F_KSCORE on box 2: does a small SCENE-READING scorer pick the best of O's 7 longitudinal hypotheses better than B's own code AND better than the best constant class?
# (B = the banked M2b checkpoints, FROZEN; the scorer is the ONLY thing that trains; seeds 0 and 1 follow B's.) ARMED ONLY ON THE PI's go for box 2 (it is the PI's personal machine; one job at a
# time under SAFE MODE). Derived from chain_v8f_mixo.sh (code/box2_ops): the predecessor wait with an ATOMIC busy marker, the code refresh from the REMOTE tip with sha-checked key files, the
# banked-checkpoint sha256 assertions, script polling (waitjob.sh), the v8f_guard.sh temperature guard, and the ZZV8FMINI-DONE / -STOP markers in ITS OWN log (chain_v8f_kscore.out).
#   chain_v8f_kscore.sh --dry-run   PRINTS the exact command every stage runs on box 2 (v8f_kscore.py's own `cmdline`: the line printed IS the line the job .cmd embeds), the checkpoint sha256
#                                   values it asserts and the order, and exits; nothing is shipped, launched or waited for.
#   0. wait for the predecessor (default: FLAT's chain, chain_v8f_flat.out, ZZV8FMINI-DONE (box-2 order MIXO -> FLAT -> KSCORE -> OACC); stop on its ZZV8FMINI-STOP): AFTER_OUT / AFTER_DONE / AFTER_STOP override it; then the busy marker
#      is taken ATOMICALLY (noclobber)
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
AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_flat.out}; AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}
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
until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do
  grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped: $(grep "^$AFTER_STOP" "$AFTER_OUT" | tail -1)"; exit 1; }
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
