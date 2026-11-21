#!/bin/sh
# SNAP40K -- FIXED COPY (2026-10-19, EvalFlyWheel) of snap40k_r8.sh.  ONE change: the size gate.  The original compares the remote ckpt.pt size for EQUALITY with SNAP_EXPECT_SIZE=1560403737 (the size at step 10,500);
#   the live file measured 1560412761 B at step 36,000 (stat on Thor, 2026-10-19 08:27), so at 40,000 the original aborts with ZZS40BADSIZEZZ and the 40,000 window is lost (ckpt.pt is rewritten in place at 40,500).
#   This copy accepts a size within SNAP_SIZE_TOL_PERMILLE (default 10 = 1 %) of SNAP_EXPECT_SIZE.  Everything else (probe, markers ZZS40..., dest, sidecars, md5 on both ends, stat unchanged, local size == remote size) is byte-identical.
#   To swap the running one: kill the OLD snap40k by its explicit pid, then start this one with the same dest/log (it appends a new ZZS40STARTZZ line with its own pid, which chain_40k_r8.sh then follows).
# SNAP40K -- snapshot the training box's ROLLING ckpt.pt at step 40,000 BEFORE step 40,500 overwrites it.   usage: sh snap40k_r8.sh
#   NOT STARTED by its author (2026-10-18).  The trainer keeps milestone pairs only at (5000, 15000, 20000, 30000) (refc_v3_train.py:232); ckpt.pt is rewritten
#   IN PLACE (not tmp+rename, refc_v3_train.py:10815) every 500 steps, and the line `[v3:<arm>] ckpt step N -> ckpt.pt` is printed AFTER the write (:10832), so a
#   line for N proves the file is COMPLETE at that moment and DOOMED when N+500 starts writing (~1.7 h at ~300 steps/h).
#   THOR IS A LIVE TRAINING BOX: this script is READ-ONLY there.  Remote commands: grep / cut / sort / tail / stat / md5sum / date / df, and ONE scp FROM Thor.
#   `ssh -n` always; no pgrep/pkill; no python, no GPU, nothing written on Thor.
#   Flow (opaque markers ZZS40...ZZ in $KIT/chain/snap40k.log; assert on ARTIFACTS, never an exit code):
#     poll every SNAP_POLL_S (300) s: the remote probe prints ONLY an opaque `ZZ<last ckpt step in train.log>ZZ` (the searched text never appears in the output stream,
#     so an echoed command line cannot fake a hit)
#       last <  40000 -> wait                      last == 40000 -> SNAPSHOT                      last > 40000 -> ABORT LOUDLY (the 40,000 state is already gone)
#     SNAPSHOT: remote `stat` (mtime+size) -> scp ckpt.pt to <dest>.part -> remote `stat` + `md5sum` again -> local md5sum -> ACCEPT iff
#       local size == remote size == SNAP_EXPECT_SIZE (1560403737, the rolling file's measured size), remote mtime UNCHANGED across the copy, local md5 == remote md5,
#       and train.log STILL shows no ckpt line beyond 40000 -> only then rename .part -> ckpt_full_40000.pt and write <dest>.md5 + <dest>.provenance.json.
#   The step INSIDE the file (ck['step'] == 40000) is NOT checked here (no python on Thor, no 1.5 GB torch.load in a poll loop): `extract_ema_r8.py extract --expect-step 40000` checks it.
#   Paths / commands are overridable by SNAP_* env vars ONLY so the control flow can be dry-run against a fake "Thor" (test_snap40k_r8.py).
STEP="${SNAP_STEP:-40000}"
HOSTA="${SNAP_HOST:-tanitad-thor-wifi}"
THOR="${SNAP_SSH:-ssh -n -o BatchMode=yes -o ConnectTimeout=20 $HOSTA}"
SCPC="${SNAP_SCP:-scp -o BatchMode=yes -o ConnectTimeout=20}"
RDIR="${SNAP_RDIR:-/home/nvidia/refcv8_run/runs/refcv8-r101-s0}"
RSRC="${SNAP_SRC_SPEC:-$HOSTA:$RDIR/ckpt.pt}"
KIT="${SNAP_KIT:-/e/refcv8_eval_kit}"
DEST="${SNAP_DEST:-$KIT/ckpt/ckpt_full_$STEP.pt}"
POLL="${SNAP_POLL_S:-300}"
MAXW="${SNAP_MAX_WAIT_S:-216000}"
MAXFAIL="${SNAP_MAX_SSH_FAIL:-24}"
EXPECT_SIZE="${SNAP_EXPECT_SIZE:-1560403737}"
TOL_PERMILLE="${SNAP_SIZE_TOL_PERMILLE:-10}"   # 1 %: the rolling ckpt.pt is NOT a constant size (1,560,403,737 B at step 10,500; 1,560,412,761 B at step 36,000 -- MEASURED 2026-10-19), so equality is the wrong test; local size == remote size, the md5 and the unchanged stat still decide
MINFREE_KB="${SNAP_MIN_FREE_KB:-6000000}"
LOG="${SNAP_LOG:-$KIT/chain/snap40k.log}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$DEST")"
mark() { echo "$1 $(date +%FT%T%z) $2" >> "$LOG"; }
# the probe: counts are computed REMOTELY and printed as an opaque ZZ<n>ZZ token; the client parses only that token
PROBE="L=$RDIR/train.log; last=\$(grep -ao 'ckpt step [0-9]* -> ckpt.pt' \$L | cut -d' ' -f3 | sort -n | tail -n 1); echo ZZ\${last:-0}ZZ"
last_step() { OUT=$($THOR "$PROBE" 2>/dev/null); echo "$OUT" | sed -n 's/.*ZZ\([0-9][0-9]*\)ZZ.*/\1/p' | tail -n 1; }
echo "ZZS40STARTZZ $(date +%FT%T%z) pid $$ step $STEP dest $DEST" >> "$LOG"
if [ -s "$DEST" ] && [ -s "$DEST.md5" ]; then mark ZZS40ALREADYZZ "$DEST exists with a md5 sidecar; nothing to do"; exit 0; fi
n=0; fail=0; t0=$(date +%s)
while :; do
  L=$(last_step)
  if [ -z "$L" ]; then
    fail=$((fail+1)); [ $((fail % 6)) -eq 1 ] && mark ZZS40SSHFAILZZ "probe returned no token ($fail consecutive)"
    if [ "$fail" -ge "$MAXFAIL" ]; then mark ZZS40SSHDEADZZ "$fail consecutive probe failures: the 40,000 window may be lost; a human must look"; exit 5; fi
  else
    fail=0
    if [ "$L" -gt "$STEP" ]; then mark ZZS40ABORTLATERZZ "train.log already shows ckpt step $L > $STEP: ckpt.pt no longer holds step $STEP"; exit 4; fi
    if [ "$L" -eq "$STEP" ]; then break; fi
    n=$((n+1)); [ $((n % 12)) -eq 1 ] && mark ZZS40WAITZZ "last ckpt step in train.log: $L"
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAXW" ]; then mark ZZS40TIMEOUTZZ "waited $MAXW s"; exit 6; fi
  sleep "$POLL"
done
mark ZZS40SEENZZ "train.log shows ckpt step $STEP -> ckpt.pt (the file is complete NOW); deadline = the 40,500 write"
# ---- snapshot ---------------------------------------------------------------------------------------- #
FREE=$(df -k "$(dirname "$DEST")" | tail -n 1 | awk '{print $4}')
if [ -n "$FREE" ] && [ "$FREE" -lt "$MINFREE_KB" ]; then mark ZZS40NODISKZZ "free ${FREE} kB < $MINFREE_KB"; exit 7; fi
S1=$($THOR "stat -c '%Y %s' $RDIR/ckpt.pt" 2>/dev/null)
RM1=${S1% *}; RS1=${S1#* }
case "$RS1" in ''|*[!0-9]*) mark ZZS40BADSIZEZZ "remote stat '$S1' has no numeric size"; exit 8 ;; esac
DIFF=$(( RS1 > EXPECT_SIZE ? RS1 - EXPECT_SIZE : EXPECT_SIZE - RS1 ))
if [ -z "$RM1" ] || [ $(( DIFF * 1000 )) -gt $(( EXPECT_SIZE * TOL_PERMILLE )) ]; then mark ZZS40BADSIZEZZ "remote stat '$S1', expected size $EXPECT_SIZE +- ${TOL_PERMILLE} per mille"; exit 8; fi
PART="$DEST.part"; rm -f "$PART"
T1=$(date +%s)
$SCPC "$RSRC" "$PART" >> "$LOG.scp" 2>&1
T2=$(date +%s)
S2=$($THOR "stat -c '%Y %s' $RDIR/ckpt.pt" 2>/dev/null)
RMD=$($THOR "md5sum $RDIR/ckpt.pt" 2>/dev/null); RMD=${RMD%% *}
L2=$(last_step)
LS=$(wc -c < "$PART" 2>/dev/null | tr -d ' ')
LMD=$(md5sum "$PART" 2>/dev/null); LMD=${LMD%% *}
rec="elapsed_s=$((T2-T1)) remote_before='$S1' remote_after='$S2' local_size=$LS remote_md5=$RMD local_md5=$LMD last_step_after=$L2"
if [ ${#RMD} -ne 32 ] || [ ${#LMD} -ne 32 ]; then rm -f "$PART"; mark ZZS40INCONCLUSIVEZZ "an md5 was not 32 hex chars (never a match): $rec"; exit 9; fi
if [ "$S2" != "$S1" ]; then rm -f "$PART"; mark ZZS40CHANGEDZZ "the remote file changed during the copy: $rec"; exit 10; fi
if [ "$LS" != "$RS1" ]; then rm -f "$PART"; mark ZZS40SIZEMISMATCHZZ "$rec"; exit 11; fi
if [ "$LMD" != "$RMD" ]; then rm -f "$PART"; mark ZZS40MD5MISMATCHZZ "$rec"; exit 12; fi
if [ -n "$L2" ] && [ "$L2" -gt "$STEP" ]; then rm -f "$PART"; mark ZZS40TOOLATEZZ "a later ckpt line appeared during the copy: $rec"; exit 13; fi
mv "$PART" "$DEST"
echo "$LMD  $(basename "$DEST")" > "$DEST.md5"
echo "{\"step\": $STEP, \"dest\": \"$DEST\", \"md5\": \"$LMD\", \"size\": $LS, \"remote_mtime_epoch\": $RM1, \"remote_src\": \"$RSRC\", \"snapshot_utc\": \"$(date -u +%FT%TZ)\", \"copy_seconds\": $((T2-T1)), \"last_ckpt_step_in_train_log_after\": \"$L2\", \"note\": \"rolling ckpt.pt copied READ-ONLY from the live training box; the step inside the file is verified by extract_ema_r8.py --expect-step\"}" > "$DEST.provenance.json"
mark ZZS40DONEZZ "$DEST md5 $LMD size $LS ($rec)"
