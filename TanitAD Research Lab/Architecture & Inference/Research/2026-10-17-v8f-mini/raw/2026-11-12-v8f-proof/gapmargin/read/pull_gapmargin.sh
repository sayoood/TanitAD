#!/bin/bash
# pull_gapmargin.sh -- pulls the small text results of card ADD117 from the pod (read folder *.json, *.log, *.out, *.txt; the queue log, the small_pa copies and the markers; the join record; the lead test outputs)
# as one tar over base64 and compares the sha256 of the tar on both sides. Text only: the tar is built from members that exist and whose name does not end in .npz or .pt. Every pod output goes through sed '/hf_/d'.
# No pod address is printed. Writes one new tar file in the card's own pod root; refuses when the local folder gapmargin/pod_add117 exists. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/gapmargin
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/gapmargin
FILES="read/*.json read/*.log read/*.out read/*.txt gapmargin_queue.log small_gm/* gapfiles/GAPMARGIN_GAPFILES_RECORD.json GAPMARGIN_AFTER_START.txt gapmargin_queue.out GAPMARGIN10K_START.txt GAPMARGIN10K_DONE.txt GAPMARGIN10K_STOP.txt lead_tests/lead_pytest_queue.out lead_tests/lead_pytest_read.out"
[ -e "$D/pod_add117" ] && { echo "REFUSED: $D/pod_add117 exists"; exit 2; }
REMOTE='cd ROOT_ && T=$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add117_$T.tar && M="" && for f in FILES_; do case "$f" in *.npz|*.pt) continue;; esac; if [ -f "$f" ]; then M="$M $f"; fi; done && [ -n "$M" ] && tar cf $N $M && echo "ZZNAME $N" && echo "ZZSHA $(sha256sum < $N | cut -c1-64)" && echo "ZZB64 $(base64 -w0 $N)"'
REMOTE=${REMOTE/ROOT_/$ROOT}
REMOTE=${REMOTE/FILES_/$FILES}
timeout 180 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add117.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add117.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add117.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add117.tar"
got=$(sha256sum < "$D/tmp_pull_add117.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add117" && tar xf "$D/tmp_pull_add117.tar" -C "$D/pod_add117" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add117.txt" | cut -d' ' -f2) $(find "$D/pod_add117" -type f | wc -l) files"
