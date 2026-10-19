#!/bin/sh
# pull_final_r8.sh -- WAIT for the refcv8 trainer's COMPLETION artifacts, then pull the FINAL checkpoint (step 48,852) READ-ONLY from the live box and prove the copy complete.   usage: sh pull_final_r8.sh
#   NEW file (2026-10-19, EvalFlyWheel).  Companion of chain_final_r8.sh (stage a).  Not a snapshot script: at the FINAL step the rolling ckpt.pt is never overwritten again, so there is no race window to catch.
#
#   WHAT THE FINAL STEP WRITES (refc_v3_train.py, md5 ecfb366e... identical in the launch tree on Thor):
#     :232        MILESTONES = (5000, 15000, 20000, 30000)  -> NO ckpt_48852.pt, NO ckpt_ema_48852.pt are ever written.
#     :10822-10832 ckpt.pt (model + opt + step + data_pos + ema + b1_guard), rewritten IN PLACE at `step % save_every == 0 or step == args.steps`; the line `[v3:hier] ckpt step N -> ckpt.pt` is printed
#                 AFTER the write, so the line for 48852 in train.log proves the file was complete when it was printed.
#     :10837-10842 `if step == args.steps:` model_ema.pt = the EMA weights, model-only (`_save_ema_model`, :5205), written INSIDE the loop only (a resume AT step == steps skips it).
#     :11039-11046 summary.json {"done": true, "step", "arm", "seed", "wallclock_s", "ema_decay", "ema_num_updates", "b1_guard"} AFTER the final eval, then "DONE at N".
#     stack/ops/sup_refcv7.sh:178-199,201-215,313-317  the SUPERVISOR then OVERWRITES summary.json with {"done": true, "final_step", "steps_requested", ...} (a different schema: `final_step`, not `step`)
#                 whenever metrics.jsonl reaches STEPS-1 -- and metrics.jsonl gets its step-48852 row at :10723, BEFORE the ckpt.pt save at :10822.  A kill between the two (the 2026-10-19 06:18 global OOM
#                 is that class) therefore yields a summary.json that says done@48852 while ckpt.pt is still 48500 and no model_ema.pt exists.  summary.json is a NECESSARY condition, never the proof:
#                 the proof is (train.log carries `ckpt step 48852 -> ckpt.pt`) + (stat stable) + (md5 on both ends) + (the step INSIDE the file, asserted by extract_ema_r8.py --expect-step).
#   THOR IS READ-ONLY here: `ssh -n` with stat / md5sum / grep / cat, and scp FROM it.  No python, no GPU, no writes, no pgrep/pkill.  Markers (opaque ZZPFN...ZZ) -> $KIT/chain/pull_final.log.
#   Flow:  poll (FIN_POLL_S, 300) a remote probe that prints ONLY an opaque token ZZ<done>_<step>_<ckpt-lines>ZZ
#            done=0 -> wait   |  done=1 and step != 48852 -> WRONGSTEP   |  done=1, step==48852, no ckpt line -> HOLE (re-probed once)   |  else settle (stat snapshots FIN_SETTLE_S apart must agree) -> pull
#          pull = remote stat -> size band -> scp to .part -> remote stat + md5sum -> local size + md5 -> ACCEPT iff stat unchanged AND local size == remote size AND md5 equal (both 32 hex) -> rename + sidecars.
#          model_ema.pt (optional, ~403 MB) and summary.json are pulled the same way / by cat.   Idempotent: an ALREADY-banked, md5-consistent ckpt_full_48852.pt is not copied again.
#   Env overrides (FIN_*) exist ONLY so the control flow can be dry-run against a fake "Thor" (test_chain_final_r8.py).
STEP="${FIN_STEP:-48852}"
HOSTS="${FIN_HOSTS:-tanitad-thor tanitad-thor-wifi}"
RDIR="${FIN_RDIR:-/home/nvidia/refcv8_run/runs/refcv8-r101-s0}"
KIT="${FIN_KIT:-/e/refcv8_eval_kit}"
CKD="${FIN_CKPT_DIR:-$KIT/ckpt}"
LOG="${FIN_LOG:-$KIT/chain/pull_final.log}"
POLL="${FIN_POLL_S:-300}"; SETTLE="${FIN_SETTLE_S:-180}"; HOLE_S="${FIN_HOLE_RECHECK_S:-60}"
MAXW="${FIN_MAX_WAIT_S:-345600}"; MAXFAIL="${FIN_MAX_SSH_FAIL:-144}"   # 144 consecutive failed probes x 300 s = 12 h: the wait is ~a day long and Thor sits on wifi/ethernet that flap (snap40k allows 24 = 2 h)
CK_MIN="${FIN_CKPT_MIN:-1500000000}"; CK_MAX="${FIN_CKPT_MAX:-1620000000}"           # the rolling ckpt.pt is 1,560,403,737 .. 1,560,4xx,xxx B at 10,500..36,000 (MEASURED); NOT a constant, so a band, not equality
EM_MIN="${FIN_EMA_MIN:-380000000}"; EM_MAX="${FIN_EMA_MAX:-430000000}"              # ckpt_ema_*.pt are 403,2xx,xxx B
MINFREE_KB="${FIN_MIN_FREE_KB:-6000000}"
SUPLOG="${FIN_SUPLOG:-sup_refcv8-r101-s0.log}"   # the supervisor's own log in the run dir (its give-up lines are the only record that the final step will never come)
FULL="$CKD/ckpt_full_$STEP.pt"; EMAT="$CKD/model_ema_${STEP}_thor.pt"; SUMT="$CKD/summary_thor_$STEP.json"
mkdir -p "$(dirname "$LOG")" "$CKD"
mark() { echo "ZZPFN$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
pick_host() {   # sets THOR (a command prefix) and SCPH (the scp host spec prefix); with FIN_SSH set (tests) the fake wins
  if [ -n "$FIN_SSH" ]; then THOR="$FIN_SSH"; SCPC="${FIN_SCP:-scp}"; SCPH="${FIN_SCP_HOST:-fakehost}"; return 0; fi
  for h in $HOSTS; do
    o=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$h" 'echo ZZ1ZZ' 2>/dev/null)
    case "$o" in *ZZ1ZZ*) THOR="ssh -n -o BatchMode=yes -o ConnectTimeout=20 $h"; SCPC="scp -o BatchMode=yes -o ConnectTimeout=20"; SCPH="$h"; return 0 ;; esac
  done
  return 1
}
probe() {   # prints "<done> <step> <ckptlines>" or nothing; counts are computed REMOTELY and emitted as one opaque token
  RP=$(cat <<EOF
RD=$RDIR
d=0; if [ -s "\$RD/summary.json" ] && grep -aq '"done": *true' "\$RD/summary.json"; then d=1; fi
fs=\$(grep -ao '"\(final_step\|step\)": *[0-9]*' "\$RD/summary.json" 2>/dev/null | head -n 1 | grep -o '[0-9]*\$')
ck=\$(grep -ac 'ckpt step $STEP -> ckpt.pt' "\$RD/train.log" 2>/dev/null)
sx=\$(grep -aEc 'exhausted [0-9]+ launches without a complete artifact|LAUNCH GATE REFUSED' "\$RD/$SUPLOG" 2>/dev/null)
echo ZZPR\${d}_\${fs:-0}_\${ck:-0}_\${sx:-0}ZZ
EOF
)
  o=$($THOR "$RP" 2>/dev/null)
  echo "$o" | sed -n 's/.*ZZPR\([01]\)_\([0-9][0-9]*\)_\([0-9][0-9]*\)_\([0-9][0-9]*\)ZZ.*/\1 \2 \3 \4/p' | tail -n 1
}
stat_snap() {   # one line per file: "<size> <mtime>" or empty; a single token ZZ<...>ZZ so an empty/failed ssh cannot look like agreement
  s=$($THOR "stat -c '%s_%Y' $RDIR/ckpt.pt $RDIR/summary.json" 2>/dev/null | tr '\n' '|')
  m=$($THOR "stat -c '%s_%Y' $RDIR/model_ema.pt" 2>/dev/null | tr '\n' '|')
  case "$s" in *_*\|*_*\|) echo "ZZ${s}M${m}ZZ" ;; *) echo "" ;; esac
}
echo "ZZPFNSTARTZZ $(date +%FT%T%z) pid $$ step $STEP dest $FULL" >> "$LOG"
# ---- idempotence: an md5-consistent bank is not copied again --------------------------------------------------------- #
if [ -s "$FULL" ] && [ -s "$FULL.md5" ] && [ -s "$FULL.provenance.json" ]; then
  LM=$(md5sum "$FULL" | cut -d' ' -f1); SM=$(cut -d' ' -f1 < "$FULL.md5" | head -n 1)
  if [ "${#LM}" -eq 32 ] && [ "$LM" = "$SM" ] && grep -q "\"md5\": \"$LM\"" "$FULL.provenance.json"; then mark ALREADY "$FULL md5 $LM"; exit 0; fi
  mark REBUILD "banked $FULL is not self-consistent (local $LM sidecar $SM): pulling again"
fi
# ---- wait for the completion artifacts ---------------------------------------------------------------------------------- #
n=0; fail=0; holes=0; t0=$(date +%s); settle_try=0
while :; do
  if pick_host; then T=$(probe); else T=""; fi        # pick_host in THIS shell: a command substitution would lose THOR / SCPC / SCPH
  if [ -z "$T" ]; then
    fail=$((fail+1)); [ $((fail % 6)) -eq 1 ] && mark SSHFAIL "probe returned no token ($fail consecutive)"
    if [ "$fail" -ge "$MAXFAIL" ]; then mark SSHDEAD "$fail consecutive probe failures: a human must look at Thor"; exit 5; fi
  else
    fail=0; set -- $T; D=$1; FS=$2; CKL=$3; SUPX=$4
    if [ "$D" = 1 ]; then
      if [ "$FS" = 0 ]; then mark TORN "summary.json says done but no step could be read (torn write?): re-probing"
      elif [ "$FS" != "$STEP" ]; then mark WRONGSTEP "summary.json says done at step $FS, expected $STEP: this is not the final checkpoint of the registered run"; exit 22
      elif [ "$CKL" -lt 1 ]; then
        holes=$((holes+1))
        if [ "$holes" -ge 2 ]; then mark HOLE "summary.json is done at $STEP but train.log has no 'ckpt step $STEP -> ckpt.pt' line: the final save never happened (killed between the metrics row and the save). The supervisor will NOT relaunch. A human must resume the trainer"; exit 23; fi
        sleep "$HOLE_S"; continue
      else
        holes=0
        s1=$(stat_snap); [ -n "$s1" ] || { mark SETTLEFAIL "stat returned nothing"; sleep "$POLL"; continue; }
        mark SEEN "done@$STEP, train.log carries the ckpt line ($CKL); settling $SETTLE s: $s1"
        sleep "$SETTLE"; s2=$(stat_snap)
        if [ -n "$s2" ] && [ "$s1" = "$s2" ]; then break; fi
        settle_try=$((settle_try+1)); mark SETTLEAGAIN "files still changing ($s1 -> $s2), try $settle_try"
        [ "$settle_try" -ge 6 ] && { mark SETTLEFAIL "files kept changing for 6 settle rounds"; exit 24; }
        continue
      fi
    fi
    if [ "$D" = 0 ] && [ "${SUPX:-0}" -ge 1 ]; then mark SUPDEAD "the supervisor log says it gave up (exhausted its relaunches or the launch gate refused) and no done marker exists: step $STEP will not be written. A human must look at Thor"; exit 25; fi
    n=$((n+1)); [ $((n % 12)) -eq 1 ] && mark WAIT "done=$D summary-step=$FS ckpt-lines-for-$STEP=$CKL"
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAXW" ]; then mark TIMEOUT "waited $MAXW s"; exit 6; fi
  sleep "$POLL"
done
# ---- pull ------------------------------------------------------------------------------------------------------------------ #
FREE=$(df -k "$CKD" | tail -n 1 | awk '{print $4}')
if [ -n "$FREE" ] && [ "$FREE" -lt "$MINFREE_KB" ]; then mark NODISK "free ${FREE} kB < $MINFREE_KB on $CKD"; exit 7; fi
pull_one() {   # $1 remote file, $2 dest, $3 min size, $4 max size  -> 0 ok; prints the md5 on success into PULL_MD5 / PULL_SIZE / PULL_MTIME / PULL_REC
  S1=$($THOR "stat -c '%s %Y' $RDIR/$1" 2>/dev/null); RS1=${S1%% *}; RM1=${S1#* }
  case "$RS1" in ''|*[!0-9]*) mark BADSIZE "remote stat of $1 not numeric: '$S1'"; return 8 ;; esac
  if [ "$RS1" -lt "$3" ] || [ "$RS1" -gt "$4" ]; then mark BADSIZE "$1 is $RS1 B, outside the sanity band [$3, $4]"; return 8; fi
  PART="$2.part"; rm -f "$PART"
  T1=$(date +%s); ( cd "$(dirname "$2")" && ${SCPC} "$SCPH:$RDIR/$1" "./$(basename "$2").part" ) >> "$LOG.scp" 2>&1; T2=$(date +%s)
  S2=$($THOR "stat -c '%s %Y' $RDIR/$1" 2>/dev/null)
  RMD=$($THOR "cd $RDIR && md5sum $1" 2>/dev/null); RMD=${RMD%% *}
  LS=$(wc -c < "$PART" 2>/dev/null | tr -d ' ')
  LMD=$(md5sum "$PART" 2>/dev/null); LMD=${LMD%% *}
  PULL_REC="file=$1 elapsed_s=$((T2-T1)) remote_before='$S1' remote_after='$S2' local_size=$LS remote_md5=$RMD local_md5=$LMD"
  if [ ${#RMD} -ne 32 ] || [ ${#LMD} -ne 32 ]; then rm -f "$PART"; mark INCONCLUSIVE "an md5 was not 32 hex chars (never a match): $PULL_REC"; return 9; fi
  if [ "$S2" != "$S1" ]; then rm -f "$PART"; mark CHANGED "the remote file changed during the copy: $PULL_REC"; return 10; fi
  if [ "$LS" != "$RS1" ]; then rm -f "$PART"; mark SIZEMISMATCH "$PULL_REC"; return 11; fi
  if [ "$LMD" != "$RMD" ]; then rm -f "$PART"; mark MD5MISMATCH "$PULL_REC"; return 12; fi
  mv -f "$PART" "$2"; echo "$LMD  $(basename "$2")" > "$2.md5"
  PULL_MD5=$LMD; PULL_SIZE=$LS; PULL_MTIME=$RM1
  return 0
}
# the optional trainer-written EMA file (independent route for the EMA weights); absent after a resume AT step == steps
EMD=NONE
if $THOR "test -s $RDIR/model_ema.pt" 2>/dev/null; then
  pull_one model_ema.pt "$EMAT" "$EM_MIN" "$EM_MAX"; RC=$?; [ $RC -eq 0 ] || exit $RC
  EMD=$PULL_MD5; mark EMAPULLED "model_ema.pt md5 $EMD ($PULL_REC)"
else
  rm -f "$EMAT" "$EMAT.md5"; mark NOMODELEMA "no model_ema.pt on Thor: ckpt.pt['ema'] is the only EMA record (not a failure)"
fi
$THOR "cat $RDIR/summary.json" > "$SUMT.part" 2>/dev/null; if [ -s "$SUMT.part" ]; then mv -f "$SUMT.part" "$SUMT"; else rm -f "$SUMT.part"; mark NOSUMMARY "could not copy summary.json"; fi
pull_one ckpt.pt "$FULL" "$CK_MIN" "$CK_MAX"; RC=$?; [ $RC -eq 0 ] || exit $RC
FM=$PULL_MD5; FS_=$PULL_SIZE
LASTPROBE=$(probe)
echo "{\"step\": $STEP, \"dest\": \"$FULL\", \"md5\": \"$FM\", \"size\": $FS_, \"remote_mtime_epoch\": $PULL_MTIME, \"model_ema_thor_md5\": \"$EMD\", \"summary_copy\": \"$SUMT\", \"probe_after\": \"$LASTPROBE\", \"snapshot_utc\": \"$(date -u +%FT%TZ)\", \"ok\": true, \"note\": \"the final ckpt.pt copied READ-ONLY from the finished training box; the step INSIDE the file is verified by extract_ema_r8.py --expect-step, never by summary.json\"}" > "$FULL.provenance.json"
mark DONE "ckpt_md5 $FM size $FS_ model_ema_md5 $EMD ($PULL_REC)"
