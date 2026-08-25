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
# ⛔⛔ THE ALLOW-LIST DEFECT, FIXED 2026-08-25. The first version matched a
# HARDCODED LIST of probe names (egodom|deltaz|actchan|...). Two new probes —
# `confound.py` and `undershoot.py` — were therefore INVISIBLE to it, and the
# sentinel reported IDLE-DEV while `confound.py` was running with 2 GB resident.
# ⚠️ A WATCHDOG WITH AN ALLOW-LIST GOES BLIND TO EXACTLY THE WORK THAT IS NEWEST,
# which is the work most likely to need watching. It now matches ANY python
# running a .py out of the scratchpad, so a probe added tomorrow is covered
# without editing the sentinel. Same family as an absence-claim probed at ONE
# location: the filter answered "is one of THESE running?", never "is anything
# running?".
# ⚠️ AND THE FIRST FIX WAS WRONG TOO, caught by SELF-TESTING it rather than
# assuming: I matched 'scratchpad.*\.py', but the probes are launched after a
# `cd` into the scratchpad, so the word never appears in the CommandLine — the
# pattern saw ZERO while a 1.9 GB job was running. The working match is the VENV
# (`venvs	anitad`) plus a `.py`, which is what every probe actually shares.
# ⇒ A WATCHDOG MUST BE TESTED AGAINST A KNOWN-RUNNING JOB. "It looks right" is
# how the first version shipped — and the SECOND fix was wrong too
# (`venvs..tanitad` needs TWO characters where the path has one backslash), which
# only a THIRD self-test caught. Candidates were finally scored against the live
# process: 'venvs.tanitad.*\.py' -> 2, 'venvs\tanitad.*\.py' -> 0.
# THREE WRONG PATTERNS IN A ROW, EACH OF WHICH "LOOKED RIGHT". The lesson is not
# about regex: A MONITOR'S FILTER IS ITSELF A MEASUREMENT AND NEEDS A POSITIVE
# CONTROL, exactly like every probe panel in this campaign.
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
  D=$($PS -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'venvs.tanitad.*\.py' } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
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
