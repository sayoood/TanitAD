#!/usr/bin/env bash
# pod_upload.sh -- resumable per-file upload of UPLOAD_MANIFEST.tsv to the pod (the address is read from the address file, never printed or logged).
# A file whose remote size already equals the local size is skipped.  Up to 4 passes, then a sha256 comparison of every file.  Ends with UPLOAD_DONE.txt or UPLOAD_STOP.txt.
set -u
export PATH="/usr/bin:/bin:$PATH"
D=/f/TanitAD-scratch/v8proof/pod; M="$D/UPLOAD_MANIFEST.tsv"; LOG="$D/upload.log"
F=/c/Users/Admin/v8pod/pod_direct.txt
H=$(grep -oP '^host=\K.*' "$F" | tr -d '\r'); P=$(grep -oP '^port=\K.*' "$F" | tr -d '\r')
O="-i $HOME/.ssh/tanitad_pod -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=30"
mask() { sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g'; }
rm -f "$D/UPLOAD_DONE.txt" "$D/UPLOAD_STOP.txt"
echo "[upload] start $(date '+%F %T') files=$(wc -l < "$M")" >> "$LOG"
cut -f2 "$M" | sed 's#/[^/]*$##' | sort -u > "$D/remote_dirs.txt"
ssh $O -p "$P" root@"$H" 'while IFS= read -r d; do mkdir -p "$d"; done' < "$D/remote_dirs.txt" 2>&1 | mask >> "$LOG"
for pass in 1 2 3 4; do
  ssh $O -p "$P" root@"$H" 'find /workspace/v8proof/art /workspace/v8proof/pulls /workspace/v8proof/w -type f -printf "%s\t%p\n" 2>/dev/null' > "$D/remote_sizes.tsv" 2>/dev/null
  n=0; sent=0
  while IFS=$'\t' read -r l r s; do
    if grep -qxF "$s"$'\t'"$r" "$D/remote_sizes.tsv"; then continue; fi
    n=$((n+1))
    # no process substitution here: under Git bash every 2> >(sed ...) left an orphan sed (202 of them held 2.3 GB of commit on 2026-11-13)
    if ssh $O -p "$P" root@"$H" "cat > '$r'" < "$l" 2> "$LOG.err"; then sent=$((sent+1)); fi
    mask < "$LOG.err" >> "$LOG"; : > "$LOG.err"
    [ $((n % 10)) = 0 ] && echo "[upload] pass $pass sent $n $(date '+%T')" >> "$LOG"
  done < "$M"
  echo "[upload] pass $pass missing_at_start=$n $(date '+%F %T')" >> "$LOG"
  [ "$n" = 0 ] && break
done
if [ "$n" != 0 ]; then echo "ZZ UPLOAD-STOP files still missing after 4 passes: $n" > "$D/UPLOAD_STOP.txt"; exit 1; fi
cut -f1 "$M" | while IFS= read -r l; do sha256sum "$l" | cut -c1-64; done > "$D/local_sha.txt"
cut -f2 "$M" > "$D/remote_paths.txt"
ssh $O -p "$P" root@"$H" 'while IFS= read -r f; do sha256sum "$f" | cut -c1-64; done' < "$D/remote_paths.txt" > "$D/remote_sha.txt" 2>/dev/null
if cmp -s "$D/local_sha.txt" "$D/remote_sha.txt"; then echo "ZZ UPLOAD-DONE $(wc -l < "$D/local_sha.txt") files sha256 equal $(date '+%F %T')" > "$D/UPLOAD_DONE.txt"
else echo "ZZ UPLOAD-STOP sha256 lists differ" > "$D/UPLOAD_STOP.txt"; exit 2; fi
