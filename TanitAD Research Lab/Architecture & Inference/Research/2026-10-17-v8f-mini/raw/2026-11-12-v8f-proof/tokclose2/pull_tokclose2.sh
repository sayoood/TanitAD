#!/bin/bash
# pull_tokclose2.sh -- pulls the small text results of card ADD86 from the pod into tokclose2/pod (score files, queue log, the two c0 lists, the nine fit logs) as one tar over base64,
# and compares the sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the chain's own root. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/tokclose2
PS=/c/Users/Admin/v8pod/pssh.sh
[ -e "$D/pod" ] && { echo "REFUSED: $D/pod exists"; exit 2; }
timeout 120 bash "$PS" 'cd /workspace/v8proof/tokclose2 && [ -f TOKCLOSE2_DONE.txt ] && [ ! -e pull_tokclose2.tar ] && mkdir -p fit_logs && cp fits/tokclose_*_log.json fit_logs/ && cp logs/c0_copies.sha256 logs/c0_sources.sha256 . && cp score/TOKCLOSE_CONTROLS.json score/TOKCLOSE_RESULT.json score/TOKCLOSE_RESULT.txt . && tar cf pull_tokclose2.tar TOKCLOSE_CONTROLS.json TOKCLOSE_RESULT.json TOKCLOSE_RESULT.txt c0_copies.sha256 c0_sources.sha256 fit_logs queue_tokclose.log TOKCLOSE2_DONE.txt START_TOKCLOSE.txt && echo "ZZSHA $(sha256sum < pull_tokclose2.tar | cut -c1-64)" && echo "ZZB64 $(base64 -w0 pull_tokclose2.tar)"' 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull.tar"
got=$(sha256sum < "$D/tmp_pull.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod" && tar xf "$D/tmp_pull.tar" -C "$D/pod" && echo "PULL-OK ${got:0:16} $(ls "$D/pod" | wc -l) entries, $(ls "$D/pod/fit_logs" | wc -l) fit logs"
