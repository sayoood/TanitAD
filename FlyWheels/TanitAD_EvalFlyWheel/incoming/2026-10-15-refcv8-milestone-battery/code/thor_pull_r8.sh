#!/bin/sh
# thor_pull_r8.sh <step> -- bring a refcv8 milestone's checkpoint(s) + a metrics copy with the step's eval row to the dev box.
#   READ-ONLY on Thor (ssh -n: ls / stat / md5sum, scp from it). Never python or GPU there. Idempotent.  Log: chain/thor_pull_step<N>.log
#   1. `ckpt_<N>.pt` (+ `ckpt_ema_<N>.pt`) in the run dir  -> scp, md5 on BOTH sides must agree.
#   2. else (the trainer writes those only at MILESTONES=(5000,15000,20000,30000); the 10,000 pair never existed) the ROLLING
#      `ckpt.pt` iff it holds step N: stat -> md5sum -> scp -> local md5 -> re-stat (mtime unchanged) -> `slim_rolling_ckpt_r8.py`
#      derives `ckpt_<N>.pt` / `ckpt_ema_<N>.pt` + a provenance sidecar saying they are DERIVED, not trainer-written.
#   3. a metrics copy (`thor_reads/metrics_<ts>.jsonl`) with EXACTLY ONE eval row at step N (the live file's torn tail is dropped).
# Prints `ZZQPULLOKZZ`-style markers into the log; the LAST line of stdout is `pull:ok:<mode>` or `pull:fail:<reason>`.
# Assert on ARTIFACTS: `ckpt_<N>.md5` (written only after every check passed) + the metrics copy.
STEP="${1:?step}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"; KITW="${R8_KITW:-E:/refcv8_eval_kit}"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
THOR="${R8_THOR:-tanitad-thor}"; RUN="${R8_THOR_RUN:-/home/nvidia/refcv8_run/runs/refcv8-r101-s0}"
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=20"
MINFREE="${R8_PULL_MIN_FREE_GB:-2}"   # 2026-10-18: was 6; battery PASS dumps now go to C: (queue_30000), so E: holds only the two ~0.4 GB ckpts
mkdir -p "$KIT/chain" "$KIT/ckpt/rolling" "$KIT/thor_reads"
LOG="$KIT/chain/thor_pull_step$STEP.log"
log() { echo "[$(date +%FT%T)] $*" >> "$LOG"; }
fail() { log "ZZQPULLFAILZZ $*"; echo "pull:fail:$*"; exit 1; }
CK="$KIT/ckpt"
log "ZZQPULLSTARTZZ step $STEP pid $$"
free=$($PY -I "$PKGW/code/queue_helpers_r8.py" freegb --path "$KITW/")
log "free GB on $KITW: $free (need >= $MINFREE)"
ok_free=$($PY -I -c "print(1 if float('$free') >= float('$MINFREE') else 0)")
HAVE=0
[ -s "$CK/ckpt_$STEP.pt" ] && [ -s "$CK/ckpt_$STEP.md5" ] && HAVE=1
if [ "$HAVE" = 1 ]; then
  log "ZZQPULLHAVEZZ ckpt_$STEP.pt already verified locally ($(cut -c1-12 "$CK/ckpt_$STEP.md5" | head -1))"
  MODE=have
else
  [ "$ok_free" = 1 ] || fail "free space $free GB < $MINFREE GB on $KITW"
  if $SSH $THOR "test -s $RUN/ckpt_$STEP.pt" 2>/dev/null; then
    MODE=milestone
    for f in ckpt_$STEP.pt ckpt_ema_$STEP.pt; do
      $SSH $THOR "test -s $RUN/$f" 2>/dev/null || { log "no $f on Thor (raw only)"; continue; }
      t_md5=$(timeout 900 $SSH $THOR "cd $RUN && nice -n 19 ionice -c3 md5sum $f" | cut -d' ' -f1)
      [ "${#t_md5}" -eq 32 ] || fail "thor md5 of $f is not 32 chars ('$t_md5')"
      ( cd "$CK" && scp -q -o BatchMode=yes "$THOR:$RUN/$f" "./$f.part" ) || fail "scp $f"
      l_md5=$($PY -I "$PKGW/code/queue_helpers_r8.py" md5 "$KITW/ckpt/$f.part")
      [ "${#l_md5}" -eq 32 ] && [ "$l_md5" = "$t_md5" ] || { rm -f "$CK/$f.part"; fail "md5 mismatch $f thor=$t_md5 local=$l_md5"; }
      mv -f "$CK/$f.part" "$CK/$f"; echo "$t_md5  $f" >> "$CK/ckpt_$STEP.md5.tmp"
      log "pulled $f md5 $t_md5 (both sides)"
    done
    mv -f "$CK/ckpt_$STEP.md5.tmp" "$CK/ckpt_$STEP.md5"
  else
    MODE=rolling
    log "no ckpt_$STEP.pt in the Thor run dir -> rolling ckpt.pt route"
    try=0
    while [ $try -lt 3 ]; do
      try=$((try+1))
      st1=$($SSH $THOR "stat -c '%s %Y' $RUN/ckpt.pt")
      t_md5=$(timeout 900 $SSH $THOR "cd $RUN && nice -n 19 ionice -c3 md5sum ckpt.pt" | cut -d' ' -f1)
      [ "${#t_md5}" -eq 32 ] || fail "thor md5 of ckpt.pt is not 32 chars ('$t_md5')"
      dest="ckpt_rolling_$(echo $st1 | cut -d' ' -f2).pt"
      ( cd "$CK/rolling" && scp -q -o BatchMode=yes "$THOR:$RUN/ckpt.pt" "./$dest.part" ) || fail "scp ckpt.pt"
      st2=$($SSH $THOR "stat -c '%s %Y' $RUN/ckpt.pt")
      l_md5=$($PY -I "$PKGW/code/queue_helpers_r8.py" md5 "$KITW/ckpt/rolling/$dest.part")
      if [ "$st1" = "$st2" ] && [ "${#l_md5}" -eq 32 ] && [ "$l_md5" = "$t_md5" ]; then
        mv -f "$CK/rolling/$dest.part" "$CK/rolling/$dest"; break
      fi
      log "ckpt.pt changed or md5 differs during the pull (try $try: stat '$st1' -> '$st2', thor $t_md5 local $l_md5); retrying"
      rm -f "$CK/rolling/$dest.part"; dest=""; sleep 30
    done
    [ -n "$dest" ] || fail "rolling ckpt.pt kept changing / md5 mismatch for 3 tries"
    log "pulled rolling ckpt.pt as $dest md5 $t_md5 (stat $st1)"
    MSYS_NO_PATHCONV=1 $PY -I "$PKGW/code/slim_rolling_ckpt_r8.py" --src "$KITW/ckpt/rolling/$dest" --out-dir "$KITW/ckpt" --expect-step "$STEP" \
        --provenance "thor_run=$RUN" "thor_ckpt_pt_md5=$t_md5" "thor_stat_size_mtime=$(echo $st1 | tr ' ' ',')" "pulled=$(date +%FT%T%z)" >> "$LOG" 2>&1
    [ -s "$CK/ckpt_$STEP.provenance.json" ] || fail "slim failed (see log): the rolling ckpt.pt did not hold step $STEP or was unusable"
    { echo "$(md5sum "$CK/ckpt_$STEP.pt" | cut -d' ' -f1)  ckpt_$STEP.pt"; echo "$(md5sum "$CK/ckpt_ema_$STEP.pt" | cut -d' ' -f1)  ckpt_ema_$STEP.pt"
      echo "$t_md5  rolling/$dest (thor ckpt.pt, step $STEP)"; } > "$CK/ckpt_$STEP.md5.tmp"
    mv -f "$CK/ckpt_$STEP.md5.tmp" "$CK/ckpt_$STEP.md5"
    free2=$($PY -I "$PKGW/code/queue_helpers_r8.py" freegb --path "$KITW/")
    if [ "${R8_KEEP_ROLLING:-0}" != "1" ]; then rm -f "$CK/rolling/$dest"; log "rolling copy removed after the slim pair was written and md5'd (free GB $free2 -> after)"; fi
  fi
fi
# ---- the metrics copy: exactly one eval row at STEP ------------------------------------------------ #
n=0; res=""
while [ $n -lt 12 ]; do
  n=$((n+1))
  tmp="$KIT/thor_reads/.metrics_pull.jsonl"
  ( cd "$KIT/thor_reads" && scp -q -o BatchMode=yes "$THOR:$RUN/metrics.jsonl" "./.metrics_pull.jsonl" ) || { log "scp metrics failed (try $n)"; sleep 20; continue; }
  ts=$(date +%Y%m%dT%H%M%S)
  res=$($PY -I "$PKGW/code/queue_helpers_r8.py" metrics_clean --src "$KITW/thor_reads/.metrics_pull.jsonl" --step "$STEP" --out "$KITW/thor_reads/metrics_$ts.jsonl")
  rm -f "$tmp"
  case "$res" in ok:*) log "metrics copy metrics_$ts.jsonl $res"; break ;; esac
  log "metrics copy not usable yet (try $n): $res"; sleep 60
done
case "$res" in ok:*) ;; *) fail "metrics: $res" ;; esac
log "ZZQPULLOKZZ step $STEP mode $MODE"
echo "pull:ok:$MODE"
