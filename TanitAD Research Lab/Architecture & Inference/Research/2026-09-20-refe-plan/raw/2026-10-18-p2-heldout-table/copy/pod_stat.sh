#!/bin/bash
# stdin: relpaths <log>/<CAM>/<hash>.jpg ; stdout: "<size>\t<shard>\t<rel>" or "MISSING\t\t<rel>"
declare -A SH
while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  log=${rel%%/*}
  if [ -z "${SH[$log]+x}" ]; then
    d=$(ls -d /workspace/data/navtrain_pixels/*/"$log" 2>/dev/null | head -1)
    SH[$log]=${d%/*}
  fi
  sh=${SH[$log]}
  f="$sh/$rel"
  if [ -f "$f" ]; then printf '%s\t%s\t%s\n' "$(stat -c %s "$f")" "${sh##*/}" "$rel"; else printf 'MISSING\t\t%s\n' "$rel"; fi
done
