#!/usr/bin/env bash
# ⭐⭐ THE IDLE SENTINEL — turns IDLENESS into an EVENT instead of a silence.
#
# ⛔ THE STRUCTURAL HOLE THIS CLOSES, MEASURED TWICE TODAY. Every monitor armed in
# this campaign watched for a job COMPLETING. None watched for a box going IDLE.
# So when a job finished with nothing queued behind it, the result was SILENCE —
# and silence is indistinguishable from "still running". Thor sat idle ~50 minutes
# after `ro128p30k` finished, and `egodom` sat finished-and-unread for 20 minutes,
# for exactly this reason. The cron cannot cover it either: session crons only
# fire when the REPL is idle, and the REPL has been continuously busy, so the
# 27-minute drumbeat has almost certainly never fired once.
#
# ⇒ The fix is not vigilance. It is the Monitor rule applied honestly: *if this
# box went idle right now, would my filter emit anything?* Now it does.
#
# Emits:
#   ZZIDLE-THOR ...   no trainer process on Thor
#   ZZIDLE-DEV ...    no probe process on the dev box
#   ZZBOTH-BUSY ...   suppressed (printed only on transition, so a healthy fleet
#                     is quiet and a newly-idle box is LOUD)
# ⚠️ It reports, it does not auto-launch. Auto-launching an unchosen arm would
# spend the only GPU we have on a guess; the point is to make the DECISION
# arrive immediately rather than after an hour of nothing.
set -u
PS="/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
LAST=""
while true; do
  T=$(ssh -n -o ConnectTimeout=20 tanitad-thor-wifi \
        'ps -eo args | grep -c "[t]rain_v6_staged"' 2>/dev/null | tr -d '\r')
  D=$($PS -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'egodom|deltaz|actchan|physics|idm_oracle|rangeprobe|spatialenv|envpred' } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
  T=${T:-0}; D=${D:-0}
  STATE="T${T}D${D}"
  if [ "$T" = "0" ] && [ "$D" = "0" ]; then
    echo "ZZIDLE-BOTH $(date +%H:%M) — Thor AND dev box are both idle ZZ"
  elif [ "$T" = "0" ]; then
    echo "ZZIDLE-THOR $(date +%H:%M) — no trainer on Thor; dev has $D ZZ"
  elif [ "$D" = "0" ]; then
    echo "ZZIDLE-DEV $(date +%H:%M) — no probe on the dev box; Thor has $T ZZ"
  elif [ "$STATE" != "$LAST" ]; then
    echo "busy $(date +%H:%M) thor=$T dev=$D"
  fi
  LAST="$STATE"
  sleep 240
done
