#!/usr/bin/env bash
# GATE RAISED 2,500 -> 4,300 MiB on the PI's direct authorisation (Sayed, 2026-09-20),
# on MEASURED evidence, not to make a blocked chain move. The desktop alone holds
# 3,111-3,958 MiB (19 samples), so the old ceiling could NEVER clear (0/19 over ~9.4 h)
# while rejecting a configuration that in fact runs. The arm's own peak is 2,573 MiB
# allocated / ~2,939 MiB on the card, so worst-case 3,958 + 2,939 = 6,897 of 8,188 leaves
# 1,291 MiB. Evidence: .../2026-09-19-s1-collision-gate/raw/vram_probe/ .
# DO NOT restore the old ceiling: on this box it is not safer, it is unsatisfiable.
# Wake on the EVENT (the GPU becoming free), then LAUNCH the A7 panel -- the Master Mind's (a).
# ⛔ Never on a lowered gate: the panel's own per-arm gate still requires GPU <= 4300 MiB and
# host >= 8 GB, so a card busy with the PI's servers keeps refusing, by design.
# ⭐ It does NOT exit after launching: the panel runs as its child, so this task's lifetime is
# the panel's, and the completion notification is the panel's own end.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BOX=/c/Users/Admin/qland/boxstat.py
PANEL="/d/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-a7-imagenet-knockout/code/a7_run.sh"
OUT=/c/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919
LIMIT=$(( $(date +%s) + 24 * 3600 ))
[ -f "$PANEL" ] || { echo "ZZA7LAUNCH-NO-PANEL-SCRIPTZZ"; exit 2; }
while :; do
  line=$("$PY" "$BOX" 2>/dev/null); read -r GU HF <<< "$line"
  case "${GU:-x}${HF:-x}" in
    *[!0-9]*) echo "ZZA7LAUNCH-PROBE-INCONCLUSIVE '${line}'ZZ" ;;
    *) if [ "$GU" -le 4300 ] && [ "$HF" -ge 8 ]; then
         echo "ZZA7LAUNCH-CLEAR gpu=${GU}MiB host=${HF}GB $(date '+%H:%M:%S')ZZ"
         bash "$PANEL" >> "$OUT/panel.log" 2>&1 &
         child=$!
         sleep 25
         # ⛔ assert it is REALLY running: this family of failures reports success and
         # leaves nothing behind. Count the panel's own processes, not this script's.
         n=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*a7-imagenet-knockout*a7_run.sh*' }).Count" 2>/dev/null | tr -d '\r')
         echo "ZZA7LAUNCH-STARTED child=${child} panel_procs=${n:-?} $(date '+%H:%M:%S')ZZ"
         wait "$child"
         echo "ZZA7LAUNCH-PANEL-EXIT rc=$? $(date '+%H:%M:%S')ZZ"
         exit 0
       fi
       [ $(( $(date +%s) % 1800 )) -lt 60 ] && echo "ZZA7LAUNCH-WAIT gpu=${GU}MiB host=${HF}GBZZ" ;;
  esac
  [ "$(date +%s)" -ge "$LIMIT" ] && { echo "ZZA7LAUNCH-TIMEOUT-24HZZ"; exit 3; }
  sleep 60
done
