#!/usr/bin/env bash
# Self-acting abort watchdog for the Thor concurrency pilot.
#
# WHY IT ACTS ITSELF: the trainer logs one point every ~22 min, and a client that
# polls from a laptop can be asleep, rate-limited or disconnected when a criterion
# trips. Protecting the trainer outranks completing the pilot, so the abort lives
# on the same box as the thing it protects.
#
# ⛔ IT KILLS EXACTLY ONE PID — the build pid read from build.pid. Never a pattern
# match: `pkill -f <trainer>` self-matches the caller's own ssh command line and
# has killed sessions on this programme before (CLAUDE.md). The trainer pid 25477
# and snapshot pid 42229 are only ever READ (`kill -0`), never signalled.
#
# TRIP CRITERIA (all evaluated against the INSTANTANEOUS per-step time, first-
# differenced out of the cumulative `step_s` — see stepwatch.py for why the raw
# `step_s > 28.0` criterion in the brief is structurally unable to fire):
#   SLOW2  r_inst > 27.69 on two consecutive logged points   (= baseline median
#          26.3672 x 1.05; baseline max over 126 points was 27.2105, so this sits
#          above ALL observed pre-load variation)
#   SLOW1  r_inst > 30.00 on a single point  (+13.8 % — a fast tripwire, because
#          two consecutive points is 44 min of degraded training)
#   STALE  trainer log older than 1500 s (normal cadence ~1318 s)
#   GONE   trainer pid or snapshot pid disappeared
#   DISK   free space below 300 GB
set -uo pipefail
BASE=/home/nvidia/w120pilot
LOG=/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl
WLOG=$BASE/watchdog.log
SLOW2=27.69
SLOW1=30.00

hot=0
while true; do
  bpid=$(cat "$BASE/build.pid" 2>/dev/null || echo 0)
  t=0; s=0; b=0
  kill -0 25477 2>/dev/null && t=1
  kill -0 42229 2>/dev/null && s=1
  [ "$bpid" != 0 ] && kill -0 "$bpid" 2>/dev/null && b=1

  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  disk=$(df -B1G --output=avail /home/nvidia | tail -1 | tr -dc 0-9)
  built=$(ls "$BASE/out"/*.v2ep.pt 2>/dev/null | wc -l | tr -dc 0-9)

  read -r step rlast <<<"$(/usr/bin/python3 -c '
import json,re
rows=[]
for ln in open("'"$LOG"'"):
    try: d=json.loads(ln)
    except Exception: continue
    if "step_s" not in d: continue
    m=re.search(r"over the (\d+) steps", d.get("step_s_note",""))
    if m: rows.append((d["step"], d["step_s"], int(m.group(1))))
if len(rows)>=2:
    a,z=rows[-2],rows[-1]; dn=z[2]-a[2]
    print(z[0], "%.4f"%((z[1]*z[2]-a[1]*a[2])/dn) if dn>0 else -1.0)
else: print(-1,-1)
')"

  reason=""
  awk -v r="$rlast" -v x="$SLOW1" 'BEGIN{exit !(r>x)}' && reason="SLOW1"
  if awk -v r="$rlast" -v x="$SLOW2" 'BEGIN{exit !(r>x)}'; then
    hot=$((hot+1)); [ $hot -ge 2 ] && reason="SLOW2"
  else
    hot=0
  fi
  [ "$age" -gt 1500 ] && reason="STALE"
  [ "$t" = 0 ] && reason="GONE_TRAINER"
  [ "$s" = 0 ] && reason="GONE_SNAP"
  [ "$disk" -lt 300 ] && reason="DISK"

  echo "ZZW|$(date -u +%H:%M:%S)|${t}|${s}|${b}|${age}|${step}|${rlast}|${disk}|${built}|${hot}|${reason}|" >> "$WLOG"

  if [ -n "$reason" ]; then
    if [ "$b" = 1 ]; then kill -TERM "$bpid" 2>/dev/null; sleep 10; kill -KILL "$bpid" 2>/dev/null; fi
    echo "ZZABORT|${reason}|$(date -u +%FT%TZ)|" >> "$WLOG"
    exit 0
  fi
  [ "$b" = 0 ] && { echo "ZZDONE|build_exited|$(date -u +%FT%TZ)|" >> "$WLOG"; exit 0; }
  sleep 60
done
