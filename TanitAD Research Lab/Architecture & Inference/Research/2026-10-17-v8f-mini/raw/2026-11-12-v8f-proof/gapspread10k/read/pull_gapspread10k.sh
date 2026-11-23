#!/bin/bash
# pull_gapspread10k.sh -- pulls the small text results of card ADD134 from the pod (read_out folder *.json, *.log, *.out, *.txt, *.md; the queue log, the small_gapspread10k copies, the record of the gap file
# and the markers; the lead test outputs) as one tar over base64 and compares the sha256 of the tar on both sides. Text only: the tar is built from members that exist and whose name does not end in .npz or .pt.
# Every pod output goes through sed '/hf_/d'. No pod address is printed. Writes one new tar file in the card's own pod root; refuses when the local folder gapspread10k/pod_add134 exists. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/gapspread10k
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/gapspread10k
FILES="read_out/*.json read_out/*.log read_out/*.out read_out/*.txt read_out/*.md gapspread10k_queue.log small_gapspread10k/* gapfiles/*.json GAPSPREAD10K_START.txt GAPSPREAD10K_DONE.txt GAPSPREAD10K_STOP.txt lead_tests/lead_pytest_queue.out lead_tests/lead_pytest_read.out"
[ -e "$D/pod_add134" ] && { echo "REFUSED: $D/pod_add134 exists"; exit 2; }
REMOTE='cd ROOT_ && T=$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add134_$T.tar && M="" && for f in FILES_; do case "$f" in *.npz|*.pt) continue;; esac; if [ -f "$f" ]; then M="$M $f"; fi; done && [ -n "$M" ] && tar cf $N $M && echo "ZZNAME $N" && echo "ZZSHA $(sha256sum < $N | cut -c1-64)" && echo "ZZB64 $(base64 -w0 $N)"'
REMOTE=${REMOTE/ROOT_/$ROOT}
REMOTE=${REMOTE/FILES_/$FILES}
timeout 180 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add134.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add134.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add134.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add134.tar"
got=$(sha256sum < "$D/tmp_pull_add134.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add134" && tar xf "$D/tmp_pull_add134.tar" -C "$D/pod_add134" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add134.txt" | cut -d' ' -f2) $(find "$D/pod_add134" -type f | wc -l) files"
