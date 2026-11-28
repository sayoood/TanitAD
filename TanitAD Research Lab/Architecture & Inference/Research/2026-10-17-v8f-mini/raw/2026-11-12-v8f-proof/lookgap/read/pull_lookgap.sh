#!/bin/bash
# pull_lookgap.sh -- pulls the small text results of card ADD108 from the pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: .json .log .out .txt under the read folder and the queue folder, the four fit records, RUNROOT.json, ENC_M1.json, ENC_M1_IDENTITY.json, XDUMP_IDENT.json, CHECK_DIRS.json,
# the marker and times files of the queues. Never a head file (.pt), a score or token file (.npz, .npy). Tar members are only those that exist.
# Reads only; on the pod it writes one new tar file in the card's own root. Refuses when the local folder pod_add108 exists. It opens no result value. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/lookgap
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/lookgap
FILES="run/RUNROOT.json run/fits/B_H5_s0/FIT_RECORD.json run/fits/B_H5_s1/FIT_RECORD.json run/fits/B_H5_s2/FIT_RECORD.json run/fits/B_H5_s0_perm/FIT_RECORD.json ENC_M1.json ENC_M1_IDENTITY.json xdump/XDUMP_IDENT.json records/CHECK_DIRS.json"
EXT='\( -name "*.json" -o -name "*.log" -o -name "*.out" -o -name "*.txt" \)'
[ -e "$D/pod_add108" ] && { echo "REFUSED: $D/pod_add108 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
REMOTE="cd $ROOT && T=\$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add108_\$T.tar && M=\"\" && for f in $FILES; do if [ -e \"\$f\" ]; then M=\"\$M \$f\"; fi; done && M=\"\$M \$(find read queue -type f $EXT 2>/dev/null | tr '\\n' ' ')\" && M=\"\$M \$(find . -maxdepth 3 -type f \\( -name 'START_*' -o -name 'DONE_*' -o -name 'STOP_*' -o -name '*times*' \\) $EXT 2>/dev/null | grep -v '^./tok/' | tr '\\n' ' ')\" && tar cf \$N --exclude=\"*.pt\" --exclude=\"*.npz\" --exclude=\"*.npy\" \$M && echo \"ZZNAME \$N\" && echo \"ZZSHA \$(sha256sum < \$N | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 \$N)\""
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add108_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add108_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add108_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add108_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add108_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add108" && tar xf "$D/tmp_pull_add108_$LT.tar" -C "$D/pod_add108" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add108_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add108" -type f | wc -l) files"
