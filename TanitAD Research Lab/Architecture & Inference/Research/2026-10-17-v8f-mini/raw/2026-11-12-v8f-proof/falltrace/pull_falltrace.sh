#!/bin/bash
# pull_falltrace.sh -- pulls the small results of the fall-behind count from the pod (FALLTRACE.json, the logs, the markers) as one tar over base64 and compares the sha256 of the tar on both sides.
# The tar is built on the pod only from the members that exist (a missing optional file cannot fail the tar); its name carries a time stamp. Reads only; on the pod it writes one new tar file in the kit root.
# Refuses when the local folder falltrace/pod exists.
set -u
D=/f/TanitAD-scratch/v8proof/falltrace
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/falltrace
FILES="out/FALLTRACE.json export_fall.log selftest_pod.log START_FALLTRACE.txt DONE_FALLTRACE.txt STOP_FALLTRACE.txt"
[ -e "$D/pod" ] && { echo "REFUSED: $D/pod exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
REMOTE='cd ROOT_ && T=$(date -u +%Y%m%dT%H%M%SZ) && N=pull_falltrace_$T.tar && M="" && for f in FILES_; do if [ -e "$f" ]; then M="$M $f"; fi; done && [ -n "$M" ] && tar cf $N $M && echo "ZZNAME $N" && echo "ZZSHA $(sha256sum < $N | cut -c1-64)" && echo "ZZB64 $(base64 -w0 $N)"'
REMOTE=${REMOTE/ROOT_/$ROOT}
REMOTE=${REMOTE/FILES_/$FILES}
timeout 180 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_falltrace_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_falltrace_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_falltrace_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_falltrace_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_falltrace_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod" && tar xf "$D/tmp_pull_falltrace_$LT.tar" -C "$D/pod" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_falltrace_$LT.txt" | cut -d' ' -f2) $(find "$D/pod" -type f | wc -l) files"
