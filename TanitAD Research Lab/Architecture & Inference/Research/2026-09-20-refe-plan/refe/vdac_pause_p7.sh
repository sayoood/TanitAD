#!/usr/bin/env bash
# Runs on the pod, detached.  Usage: vdac_pause_p7.sh <p7_pod.sh pid> <train.py pid> S1a|S1b      (ONE stage per P7 pause)
# Style of refe/div_pause_p7.sh (PREREG_VDAC_P6 + A2).  Waits for P7's next ckpt_last.pt, stops P7 by explicit PID, confirms the GPU is empty, runs vdac_chain.sh for the stage's arms, then ALWAYS resumes P7 (also when this script is killed or errors after P7 was stopped),
# unless /workspace/data/refe_vdac/HOLD_P7 exists.
#   S1a = HA-s0 (10,000 updates), then HA-SHUF (2,000 updates)         about 3.9 L40-h ESTIMATED (DESIGN_VDAC_P6 section 7: 3.20 + 0.64 + 0.07 + chain overhead)
#   S1b = HA-s1 (10,000 updates)                                        about 3.2 L40-h ESTIMATED.  PI go 2026-11-11 is CONDITIONAL (A2 section 4): S1b runs ONLY after the Master Mind read S1a and HA-s0 reached S1-GO with valid controls.  That is enforced here:
#                                                                       the file /workspace/data/refe_vdac/S1B_GO must exist (written by the Master Mind after that read), or the script ends BEFORE touching P7 (PAUSE_STOPPED_EARLY).
# If an arm's chain does not end in ZZVDAC_CHAIN_DONE, the stage's remaining arms are NOT run (S1a: a failed HA-s0 means HA-SHUF is not run; the pod goes back to P7).  Terminal lines in pause.log: PAUSE_STOPPED_EARLY (P7 not touched, or touched and then resumed by the trap), P7_RESUMED or P7_HELD.
set -u
CHAIN=${1:-}; TR=${2:-}; STAGEARG=${3:-}
R=/workspace/data/refe_runs/p7_reserve_head; OUT=/workspace/data/refe_vdac; L=$OUT/pause.log
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
early() { say "PAUSE_STOPPED_EARLY $*"; exit 1; }
[ -n "$CHAIN" ] && [ -n "$TR" ] && [ -n "$STAGEARG" ] || early "usage: vdac_pause_p7.sh <p7_pod.sh pid> <train.py pid> S1a|S1b"
case "$STAGEARG" in
  S1a) ARMS="HA-s0 HA-SHUF" ;;
  A5HB) ARMS="HB-s0 HB-SHUF" ;;          # VDAC A5 (DRAFT, not frozen): the LoRA trunk copy, then its label-shuffled control
  A5HC) ARMS="HC-s0 HC-SHUF" ;;          # VDAC A5 (DRAFT): registered only on S0-WEAK
  A6HBC) ARMS="HBC-s0 HBC-SHUF" ;;        # VDAC A6 (DRAFT, not frozen): HB's LoRA copy + HC's 1 m head and boundary weight, then its label-shuffled control
  A5HCSHUF) ARMS="HC-SHUF" ;;           # VDAC A5 A2: HC-SHUF alone on the fixed chain (HC-s0 completed by hand after the 2 m --n-params default failed its post-training gate)
  A5HBFIX) ARMS="HB-s0-H48X HB-SHUF" ;;   # VDAC A5 A1: HB-s0's forward-only h48x from its SAVED state (vdac_pod.sh MODE=h48x, no training, no chain), then HB-SHUF as in A5HB
  S1b) ARMS="HA-s1"; [ -e "$OUT/S1B_GO" ] || early "S1b needs $OUT/S1B_GO (the Master Mind writes it after the S1a read: HA-s0 reached S1-GO with valid controls, PREREG_VDAC_P6_A2 section 4); P7 was not touched" ;;
  *) early "unknown stage $STAGEARG (S1a | S1b | A5HB | A5HC | A5HBFIX | A5HCSHUF | A6HBC)" ;;
esac
TF=$(grep -o "/root/\.[A-Za-z0-9_.]*" /workspace/data/refe_scorer1p6/after_arms.sh | head -1); [ -f "$TF" ] || early "no token file"
tr '\0' ' ' < /proc/$CHAIN/cmdline | grep -q 'p7_pod.sh' || early "pid $CHAIN is not the launch script"
tr '\0' ' ' < /proc/$TR/cmdline | grep -q 'train.py' || early "pid $TR is not the trainer"
STOPPED=0; RESUMED=0
resume_p7() {
  [ "$STOPPED" = 1 ] && [ "$RESUMED" = 0 ] || return 0
  RESUMED=1
  for i in $(seq 1 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = "0" ] && break; sleep 2; done
  if [ -e $OUT/HOLD_P7 ]; then say "P7_HELD ($OUT/HOLD_P7 exists)"; return 0; fi
  HF_TOKEN=$(cat "$TF") P7_LABEL_WORKERS=13 P7_WAIT_P6=1 setsid nohup bash /workspace/refe-plan-p7/refe/p7_pod.sh >> /workspace/data/refe_p7/p7_stdout.log 2>&1 < /dev/null &
  say "P7_RESUMED pid $!"
}
trap 'resume_p7' EXIT
m0=$(stat -c %Y "$R/ckpt_last.pt"); say "waiting for a checkpoint newer than mtime $m0 (stage $STAGEARG, arms: $ARMS)"; m=$m0
for i in $(seq 1 3600); do m=$(stat -c %Y "$R/ckpt_last.pt"); [ "$m" != "$m0" ] && break; kill -0 $TR 2>/dev/null || early "the trainer ended by itself"; sleep 1; done
[ "$m" != "$m0" ] || early "no new checkpoint within 60 min"
sleep 15
desc() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
D="$(desc $TR | tr '\n' ' ')"
say "checkpoint mtime $m; stopping launch script $CHAIN, then trainer $TR; descendants: $D"
STOPPED=1
kill -TERM $CHAIN; kill -TERM $TR
for i in $(seq 1 60); do kill -0 $TR 2>/dev/null || break; sleep 1; done
kill -0 $TR 2>/dev/null && { say "trainer still alive after 60 s, SIGKILL"; kill -9 $TR; sleep 2; }
for p in $D; do kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null; done
sleep 10
for p in $D; do kill -0 "$p" 2>/dev/null && { say "descendant $p still alive, SIGKILL"; kill -9 "$p" 2>/dev/null; }; done
empty=0; for i in $(seq 1 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = "0" ] && { empty=1; break; }; sleep 2; done
if [ "$empty" != 1 ]; then say "PAUSE_STOPPED_EARLY GPU still has a compute process after P7 was stopped"; exit 1; fi      # the EXIT trap resumes P7
for i in $(seq 1 240); do kill -0 $CHAIN 2>/dev/null || break; sleep 1; done
if kill -0 $CHAIN 2>/dev/null; then say "PAUSE_STOPPED_EARLY launch script $CHAIN did not exit"; exit 1; fi
say "left over P7 helpers: $(ps -eo pid,cmd | grep -E 'p2r_sync|p7_pod' | grep -v grep | cut -c1-80 | tr '\n' ';')"
say "P7 stopped, GPU empty"
# the stage's arms, one chain invocation each, in the order given; the first arm that does not finish stops the loop
for A in $ARMS; do
  LA=$(echo "$A" | tr 'A-Z' 'a-z')
  case "$A" in HA-SHUF|HC-SHUF) TMO=7200 ;; HB-SHUF) TMO=14400 ;; HB-s0) TMO=36000 ;; HBC-s0) TMO=50400 ;; HBC-SHUF) TMO=21600 ;; HC-s0) TMO=21600 ;; *) TMO=18000 ;; esac      # A5: HB-s0 36,000 s = 10 h against the card's UPPER bound 7.6 L40-h + 0.55 chain overhead (PREREG_VDAC_P6_A5_DRAFT section 5; the preflight speed probe replaces it before freeze); HC-s0 21,600 s against 4.3 + 0.5; HB-SHUF 0.2 x HB-s0
  :      # 10,000-update arms: 18,000 s = 5 h against ~3.7 h ESTIMATED (3.20 L40-h + preflight, mutant, gate, post-gate, export, egress); HA-SHUF: 7,200 s against ~1.1 h ESTIMATED
  if [ "$A" = "HB-s0-H48X" ]; then       # A5HBFIX only: not a chain.  The trainer's own forward on the run's saved state; the arm's token is NOT involved (nothing trains).  1 h bound (the forward is minutes).
    say "h48x-only HB-s0 starting (timeout 3600 s)"
    ARM=HB-s0 MODE=h48x timeout 3600 bash /workspace/refe-plan-vdac/refe/vdac_pod.sh >> $OUT/chain_hb-s0-h48x.out 2>&1 < /dev/null
    crc=$?
    say "h48x-only HB-s0 rc $crc (last: $(tail -n 1 $OUT/hb-s0/h48x_only.log 2>/dev/null | cut -c1-160))"
    grep -q "^ZZSFT_H48X_ONLY_DONE" $OUT/hb-s0/h48x_only.log 2>/dev/null || { say "h48x-only HB-s0 did not finish: the remaining arms are NOT run"; break; }
    continue
  fi
  say "chain $A starting (timeout $TMO s)"
  VDAC_HF_TOKEN_FILE="$TF" timeout $TMO bash /workspace/refe-plan-vdac/refe/vdac_chain.sh "$A" >> $OUT/chain_$LA.out 2>&1 < /dev/null
  crc=$?
  say "chain $A rc $crc (last: $(grep -E 'ZZVDAC_|REFUSED' $OUT/chain_$LA.log 2>/dev/null | tail -1 | cut -c1-160))"
  grep -q "ZZVDAC_CHAIN_DONE $A" $OUT/chain_$LA.log 2>/dev/null || { say "chain $A did not finish: the remaining arms are NOT run"; break; }
done
resume_p7
