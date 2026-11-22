#!/bin/bash
# pull_fallattr.sh -- pulls the small text results of card ADD110 from the pod (read folder *.json, *.log, *.out; the gap-file record; the queue logs and markers) as one tar over base64 and compares the sha256
# of the tar on both sides. Text only: the tar is built from members that exist and whose name does not end in .npz or .pt. Every pod output goes through sed '/hf_/d'. No pod address is printed.
# Writes one new tar file in the card's own pod root; refuses when the local folder fallattr/pod_add110 exists. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/fallattr
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/fallattr
FILES="read/*.json read/*.log read/*.out gapfiles/FALLATTR_GAPFILES_RECORD.json queue_fallattr*.log FALLATTR10K_START.txt FALLATTR10K_DONE.txt FALLATTR10K_STOP.txt TIMES_NOTE.txt kit_gap/lead_pytest.out kit_read/lead_pytest.out"
[ -e "$D/pod_add110" ] && { echo "REFUSED: $D/pod_add110 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
REMOTE='cd ROOT_ && T=$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add110_$T.tar && M="" && for f in FILES_; do case "$f" in *.npz|*.pt) continue;; esac; if [ -f "$f" ]; then M="$M $f"; fi; done && [ -n "$M" ] && tar cf $N $M && echo "ZZNAME $N" && echo "ZZSHA $(sha256sum < $N | cut -c1-64)" && echo "ZZB64 $(base64 -w0 $N)"'
REMOTE=${REMOTE/ROOT_/$ROOT}
REMOTE=${REMOTE/FILES_/$FILES}
timeout 180 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add110_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add110_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add110_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add110_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add110_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add110" && tar xf "$D/tmp_pull_add110_$LT.tar" -C "$D/pod_add110" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add110_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add110" -type f | wc -l) files"
