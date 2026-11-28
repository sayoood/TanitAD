#!/usr/bin/env bash
# A16 switch of refcv6-r101-s0 (PI 2026-09-26: "Stop now, resume with fixes"): 287d72e -> 82c2331a2f.
#   F3: the cascade keys pass through RefCModel.forward, and the trainer REFUSES instead of skipping.
#   Label clock: t_now = grid_start + (t + w - 1 + n_stack - 1) * dt, from the clip-clock sidecar.
# Runs ON THOR under setsid nohup. Every step is logged with a timestamp to $LOG; it ends with an
# opaque ZZ token. The procedure is the one in memory "live-run-switch-procedure":
#   config file -> SUPERVISOR first (explicit pid) -> trainer (INT, then TERM) -> its recorded workers
#   -> lock-holder scan -> fresh supervisor -> the trainer's OWN evidence lines.
set -u
RD=/home/nvidia/refcv6_run/runs.d
O=/home/nvidia/refcv6_run/runs/refcv6-r101-s0
NEW=/home/nvidia/refcv6_run/82c2331a2f
SIDECAR=/home/nvidia/data/refcv6_clip_clock_sidecar.jsonl
SUP=3346328
TR=3346338
WANT=34500
LOG=/home/nvidia/refcv6_run/switch_a16.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
say "switcher start (pid $$): NEW=$NEW WANT=ckpt step >= $WANT"
# ---- preconditions: the new tree, the sidecar, the new config, both old pids -------------------- #
for f in "$NEW/stack/scripts/refc_v3_train.py" "$NEW/ops/run_refcv6.sh" "$NEW/ops/sup_refcv6.sh" \
         "$NEW/ops/refcv6-r101-s0.env.v3" "$SIDECAR"; do
  [ -s "$f" ] || { say "ZZSWITCH-ABORT-MISSING $f ZZ"; exit 2; }
done
grep -q '"layer_u0_hat", "layer_logits"' "$NEW/stack/tanitad/refs/refc.py" \
  || { say "ZZSWITCH-ABORT-FIX-NOT-IN-TREEZZ"; exit 2; }
tr '\0' ' ' < /proc/$SUP/cmdline 2>/dev/null | grep -q 'sup_refcv6.sh refcv6-r101-s0' \
  || { say "ZZSWITCH-ABORT-SUP-PID-IS-NOT-THE-SUPERVISORZZ"; exit 2; }
tr '\0' ' ' < /proc/$TR/cmdline 2>/dev/null | grep -q 'refc_v3_train.py' \
  || { say "ZZSWITCH-ABORT-TR-PID-IS-NOT-THE-TRAINERZZ"; exit 2; }
# ---- 1. wait for the checkpoint, and for its size to be stable ---------------------------------- #
N=0
for i in $(seq 1 540); do
  N=$(grep -o 'ckpt step [0-9]*' "$O/train.log" | tail -1 | awk '{print $3}')
  [ "${N:-0}" -ge "$WANT" ] && break
  sleep 20
done
[ "${N:-0}" -ge "$WANT" ] || { say "ZZSWITCH-ABORT-NO-CKPTZZ"; exit 2; }
s1=$(stat -c %s "$O/ckpt.pt"); sleep 20; s2=$(stat -c %s "$O/ckpt.pt")
[ "$s1" = "$s2" ] || { sleep 60; s2=$(stat -c %s "$O/ckpt.pt"); }
say "checkpoint: step $N, ckpt.pt $(stat -c '%s B %y' "$O/ckpt.pt"), md5 $(md5sum < "$O/ckpt.pt" | cut -c1-32)"
# ---- 2. the config file (the supervisor sources it ONCE, at start) ------------------------------- #
cp -p "$RD/refcv6-r101-s0.env" "$RD/refcv6-r101-s0.env.v2-287d72e-final"
cp "$NEW/ops/refcv6-r101-s0.env.v3" "$RD/refcv6-r101-s0.env"
say "config now v3: md5 $(md5sum < "$RD/refcv6-r101-s0.env" | cut -c1-32)"
# ---- 3. the SUPERVISOR first, by explicit pid ----------------------------------------------------- #
WK=$(ps -o pid= --ppid "$TR" | tr -s ' \n' ' ')
say "trainer children recorded: $WK"
kill "$SUP"
for i in $(seq 1 15); do kill -0 "$SUP" 2>/dev/null || break; sleep 1; done
kill -0 "$SUP" 2>/dev/null && { say "supervisor still alive after TERM -> KILL"; kill -9 "$SUP"; }
# ---- 4. the trainer: INT, then TERM, then KILL; then its recorded children ------------------------ #
kill -INT "$TR"
for i in $(seq 1 45); do kill -0 "$TR" 2>/dev/null || break; sleep 2; done
kill -0 "$TR" 2>/dev/null && { say "trainer ignored INT -> TERM"; kill "$TR"; }
for i in $(seq 1 30); do kill -0 "$TR" 2>/dev/null || break; sleep 2; done
kill -0 "$TR" 2>/dev/null && { say "trainer ignored TERM -> KILL"; kill -9 "$TR"; sleep 3; }
for w in $WK; do kill -0 "$w" 2>/dev/null && { tr '\0' ' ' < /proc/$w/cmdline | grep -q python && kill -9 "$w"; }; done
sleep 3
say "old pids: sup $(kill -0 $SUP 2>/dev/null && echo ALIVE || echo gone), trainer $(kill -0 $TR 2>/dev/null && echo ALIVE || echo gone)"
# ---- 5. nothing may still hold the lock ----------------------------------------------------------- #
h=0
for p in /proc/[0-9]*/fd/*; do
  if [ "$(readlink "$p" 2>/dev/null)" = "$RD/refcv6-r101-s0.lock" ]; then
    say "LOCK HELD BY pid $(echo "$p" | cut -d/ -f3): $(tr '\0' ' ' < /proc/$(echo "$p" | cut -d/ -f3)/cmdline | cut -c1-120)"
    h=$((h + 1))
  fi
done
[ "$h" -eq 0 ] || { say "ZZSWITCH-ABORT-LOCK-HELD-$h ZZ"; exit 3; }
[ -e "$O/summary.json" ] && { say "ZZSWITCH-ABORT-SUMMARY-EXISTSZZ"; exit 3; }
# ---- 6. a fresh supervisor from the NEW tree ------------------------------------------------------ #
setsid nohup bash "$NEW/ops/sup_refcv6.sh" refcv6-r101-s0 "$RD" > /home/nvidia/refcv6_run/sup_launch3.out 2>&1 < /dev/null &
say "new supervisor started (launcher pid $!)"
# ---- 7. evidence from the NEW trainer ------------------------------------------------------------- #
NT=""
for i in $(seq 1 60); do
  sleep 10
  t=$(cat "$O/train.pid" 2>/dev/null)
  if [ -n "$t" ] && [ "$t" != "$TR" ] && kill -0 "$t" 2>/dev/null; then NT=$t; break; fi
done
[ -n "$NT" ] || { say "ZZSWITCH-FAIL-NO-NEW-TRAINERZZ"; exit 4; }
say "new trainer pid $NT: clip-clock flag in cmdline: $(tr '\0' '\n' < /proc/$NT/cmdline | grep -c -- '--clip-clock-sidecar')"
say "supervisors running: $(ps -eo args | grep -c 'sup[_]refcv6.sh refcv6-r101-s0')"
for i in $(seq 1 90); do
  sleep 20
  c=$(tail -n 400 "$O/metrics.jsonl" | grep -c '"cascade"')
  [ "$c" -ge 1 ] && break
done
say "metrics rows with cascade in the last 400: $c; last train.log lines:"
tail -n 5 "$O/train.log" >> "$LOG"
say "ZZSWITCH-DONE-$NT-cascade$c-ZZ"
