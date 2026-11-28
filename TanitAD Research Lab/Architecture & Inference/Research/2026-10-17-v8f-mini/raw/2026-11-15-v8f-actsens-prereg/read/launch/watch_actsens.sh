#!/usr/bin/env bash
# MM watch on the ACTSENS chain (polls by script, never by model turns). Exits (waking the MM) on: a new stage/run done marker,
# STOP.txt, DONE.txt, a new ARM_REFUSED_*.txt, or the chain driver gone for two polls with neither DONE nor STOP. Writes only its own file.
ST=/c/Users/Admin/actsens_state
OUT=/c/Users/Admin/actsens_launch/WATCH_EVENT.txt
POLL="${WATCH_POLL_S:-120}"; MAXP="${WATCH_MAX_POLLS:-1080}"
nd() { ls "$ST/done" 2>/dev/null | grep -v '^ship\.done$' | wc -l | tr -d ' '; }
nr() { ls "$ST"/ARM_REFUSED_*.txt 2>/dev/null | wc -l | tr -d ' '; }
alive() {   # count of bash processes running the chain script (pattern split so this watcher never matches itself)
  local pat; pat="actsens_ch""ain.sh"
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like ('*' + '$pat' + '*') }).Count" 2>/dev/null | tr -d '\r '
}
d0="$(nd)"; r0="$(nr)"; miss=0; i=0
while [ "$i" -lt "$MAXP" ]; do
  i=$((i+1))
  if [ -s "$ST/DONE.txt" ]; then echo "EVENT done $(date '+%F %T') :: $(head -c 300 "$ST/DONE.txt")" > "$OUT"; cat "$OUT"; exit 0; fi
  if [ -s "$ST/STOP.txt" ]; then echo "EVENT stop $(date '+%F %T') :: $(head -c 600 "$ST/STOP.txt")" > "$OUT"; cat "$OUT"; exit 0; fi
  if [ "$(nr)" != "$r0" ]; then echo "EVENT arm-refused $(date '+%F %T') :: $(ls "$ST"/ARM_REFUSED_*.txt | tr '\n' ' ')" > "$OUT"; cat "$OUT"; exit 0; fi
  if [ "$(nd)" != "$d0" ]; then echo "EVENT stage-done $(date '+%F %T') :: $(ls -t "$ST/done" | head -3 | tr '\n' ' ')" > "$OUT"; cat "$OUT"; exit 0; fi
  a="$(alive)"
  if [ "$a" = "0" ]; then miss=$((miss+1)); else miss=0; fi
  if [ "$miss" -ge 2 ]; then echo "EVENT driver-gone $(date '+%F %T') :: no chain bash for 2 polls, no DONE/STOP; tail: $(tail -c 300 /c/Users/Admin/actsens_launch/launcher.log 2>/dev/null | tr '\r\n' '  ')" > "$OUT"; cat "$OUT"; exit 0; fi
  sleep "$POLL"
done
echo "EVENT watch-timeout $(date '+%F %T') after $i polls" > "$OUT"; cat "$OUT"; exit 0
