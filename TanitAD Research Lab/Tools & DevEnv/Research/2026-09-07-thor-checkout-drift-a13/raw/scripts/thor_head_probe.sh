#!/bin/bash
# Thor-side, strictly read-only.  No fetch, no checkout, no index write.
#   A) `git ls-tree -r -z HEAD`         -> path -> blob sha AT THOR'S OWN HEAD  (-z so paths
#                                          with spaces are raw, not quoted)
#   B) `git hash-object --stdin-paths`  -> path -> blob sha OF THE WORKING-TREE BYTES,
#                                          for the tracked paths that still exist
# A==B  => the file is exactly Thor's HEAD version (nothing was shipped over it).
# A!=B  => the working-tree file was shipped/edited after the checkout.
# Both tables are emitted; the join is done client-side so a missing file cannot misalign a
# `paste`.  Payload framed by its own md5 + line count so a short pull FAILS LOUDLY.
set -uo pipefail
cd /home/nvidia/TanitAD || { echo "WWFATALWW"; exit 3; }
echo "WWHEAD$(git rev-parse HEAD)WW"
echo "WWAUTOCRLF$(git config --get core.autocrlf || echo unset)WW"
t=$(mktemp); p=$(mktemp); e=$(mktemp); o=$(mktemp); j=$(mktemp)
git ls-tree -r -z HEAD | tr '\0' '\n' > "$t"
sed 's/^[^\t]*\t//' "$t" > "$p"
: > "$e"
while IFS= read -r f; do [ -f "$f" ] && printf '%s\n' "$f" >> "$e"; done < "$p"
git hash-object --stdin-paths < "$e" > "$o" 2>/dev/null
paste "$o" "$e" > "$j"
TL=$(wc -l < "$t"); EL=$(wc -l < "$e"); JL=$(wc -l < "$j")
echo "WWTREE${TL}WW"; echo "WWEXIST${EL}WW"; echo "WWJOIN${JL}WW"
# section 1 = tree table (sha TAB path, from ls-tree with mode/type stripped)
sed 's/^[0-9]* [a-z]* //' "$t" > "${j}.tree"
cat "${j}.tree" > "${j}.all"
echo "===SPLIT===" >> "${j}.all"
cat "$j" >> "${j}.all"
B=$(gzip -9c "${j}.all" | base64 -w0)
echo "WWLINES$(wc -l < "${j}.all")WW"
echo "WWB64LEN${#B}WW"
echo "WWB64MD5$(printf '%s' "$B" | md5sum | cut -c1-32)WW"
echo "WWPAYLOADSTARTWW"
printf '%s\n' "$B" | fold -w 200
echo "WWPAYLOADENDWW"
rm -f "$t" "$p" "$e" "$o" "$j" "${j}.tree" "${j}.all"
