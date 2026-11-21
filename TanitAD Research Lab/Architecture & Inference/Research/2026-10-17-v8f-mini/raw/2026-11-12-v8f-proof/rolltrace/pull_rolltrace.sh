#!/bin/bash
# pull_rolltrace.sh -- pulls the small text results of the trace export from the pod (the export log, the pod selftest log and, when the control t0 passed, out/ROLLTRACE.json) as one tar over base64 and compares the sha256 of the tar on both sides.
# Reads only; on the pod it writes one new tar file in the root of the export. It opens no result value. Needs podrun_export_traces.sh to have run (it writes out/ and the two logs).
# Pulls into rolltrace/pod and refuses if that folder exists.
set -u
D=/f/TanitAD-scratch/v8proof/rolltrace
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/rolltrace
FILES="export_traces.log selftest_pod.log"
[ -e "$D/pod" ] && { echo "REFUSED: $D/pod exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_rolltrace.tar ] && F=\"$FILES\" && { [ -e out/ROLLTRACE.json ] && F=\"\$F out/ROLLTRACE.json\"; true; } && tar cf pull_rolltrace.tar \$F && echo \"ZZSHA \$(sha256sum < pull_rolltrace.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_rolltrace.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_rolltrace.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_rolltrace.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_rolltrace.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_rolltrace.tar"
got=$(sha256sum < "$D/tmp_pull_rolltrace.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod" && tar xf "$D/tmp_pull_rolltrace.tar" -C "$D/pod" && echo "PULL-OK ${got:0:16} $(find "$D/pod" -type f | wc -l) files"
