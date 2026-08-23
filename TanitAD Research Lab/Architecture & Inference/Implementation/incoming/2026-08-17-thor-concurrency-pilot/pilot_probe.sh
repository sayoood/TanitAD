#!/usr/bin/env bash
# One-shot abort-armed probe. Emits ONE opaque line; every value is computed
# HERE (pod-side) and only digits cross the wire.
#
# ⛔ CLAUDE.md trap: "a polling monitor whose filter contains the pattern it
# searches for will match its own echoed command". So this emits `ZZP|<digits>|`
# and the client parses that — the client NEVER greps the stream for words like
# "Traceback" or "error" that its own command line also contains.
#
# Fields: ZZP|trainer|snap|build|log_age_s|last_step|r_prev|r_last|disk_gb|built|
#   trainer/snap/build : 1 alive, 0 gone      (kill -0 on the EXPLICIT pid)
#   r_prev/r_last      : INSTANTANEOUS s/step, first-differenced from the
#                        cumulative step_s (see stepwatch.py) — the cumulative
#                        step_s CANNOT trip an abort, this can.
set -uo pipefail
LOG=/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl
BASE=/home/nvidia/w120pilot

t=0; s=0; b=0
kill -0 25477 2>/dev/null && t=1
kill -0 42229 2>/dev/null && s=1
[ -f "$BASE/build.pid" ] && kill -0 "$(cat "$BASE/build.pid")" 2>/dev/null && b=1

age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
disk=$(df -B1G --output=avail /home/nvidia | tail -1 | tr -dc 0-9)
# NOT `grep -c || echo 0`: on an empty dir grep prints "0" AND exits 1, so the
# `|| echo 0` fires too and the field becomes two lines, splitting the record.
built=$(ls "$BASE/out"/*.v2ep.pt 2>/dev/null | wc -l | tr -dc 0-9)

read -r step rprev rlast <<<"$(/usr/bin/python3 -c '
import json,re,sys
rows=[]
for ln in open("'"$LOG"'"):
    try: d=json.loads(ln)
    except Exception: continue
    if "step_s" not in d: continue
    m=re.search(r"over the (\d+) steps", d.get("step_s_note",""))
    if m: rows.append((d["step"], d["step_s"], int(m.group(1))))
def r(a,b):
    dn=b[2]-a[2]
    return (b[1]*b[2]-a[1]*a[2])/dn if dn>0 else -1.0
if len(rows)>=3: print(rows[-1][0], "%.4f"%r(rows[-3],rows[-2]), "%.4f"%r(rows[-2],rows[-1]))
elif len(rows)>=2: print(rows[-1][0], -1.0, "%.4f"%r(rows[-2],rows[-1]))
else: print(-1,-1,-1)
')"

echo "ZZP|${t}|${s}|${b}|${age}|${step}|${rprev}|${rlast}|${disk}|${built}|"
