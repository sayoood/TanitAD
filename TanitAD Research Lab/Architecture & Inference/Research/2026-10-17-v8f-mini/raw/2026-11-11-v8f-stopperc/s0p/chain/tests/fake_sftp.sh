#!/usr/bin/env bash
# S0P-STUB-TRANSPORT (marker the chain looks for: a file carrying it is never the real client).  Dry-run "sftp -b -": the same FAKE box 2 as fake_ssh.sh ($S0P_FAKE_BOX; /D:/x -> $BOX/D/x).  Reaches nothing real.
# Supported batch lines: "put <local> /<remote>" and "get -r /<remote> <local>".  Control files under $BOX/ctl: corrupt_once_<basename> (the first get of that file is corrupted, then the control is removed),
# corrupt_always_<basename> (every get is corrupted).
set -u
BOX="${S0P_FAKE_BOX:?}"; shift $(( $# > 0 ? $# : 0 ))
topath() { printf '%s' "$BOX/$(printf '%s' "${1#/}" | sed 's#\\#/#g; s#^\([A-Za-z]\):#\1#')"; }
while IFS= read -r line; do
  [ -n "$line" ] || continue
  echo "$line" >> "$BOX/sftp.log"
  if [[ "$line" =~ ^put\ ([^\ ]+)\ (/[^\ ]+)$ ]]; then dst="$(topath "${BASH_REMATCH[2]}")"; mkdir -p "$(dirname "$dst")"; cp "${BASH_REMATCH[1]}" "$dst" || exit 1
  elif [[ "$line" =~ ^get\ -r\ (/[^\ ]+)\ ([^\ ]+)$ ]]; then
    src="$(topath "${BASH_REMATCH[1]}")"; out="${BASH_REMATCH[2]}"; [ -f "$src" ] || exit 1; cp "$src" "$out" || exit 1; b="$(basename "$src")"
    if [ -e "$BOX/ctl/corrupt_always_$b" ] || [ -e "$BOX/ctl/corrupt_once_$b" ]; then printf 'X' >> "$out"; rm -f "$BOX/ctl/corrupt_once_$b"; fi
  else echo "UNKNOWN sftp: $line" >> "$BOX/unknown.log"; exit 1; fi
done
exit 0
