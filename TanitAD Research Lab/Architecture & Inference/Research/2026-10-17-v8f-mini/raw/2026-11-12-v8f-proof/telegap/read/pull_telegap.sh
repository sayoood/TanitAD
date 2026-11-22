#!/bin/bash
# pull_telegap.sh -- pulls the small text results of card ADD122 from the pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: .json .log .out .txt under the read folder, the records and the fit folders, plus the markers of the queues. Never a head file (.pt), a score, token or window file (.npz, .npy), never a video (.mp4).
# Reads only; on the pod it writes one new tar file in the card's own root. Refuses when the local folder pod_add122 exists. It opens no result value. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/telegap
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/telegap
EXT='\( -name "*.json" -o -name "*.log" -o -name "*.out" -o -name "*.txt" \)'
[ -e "$D/pod_add122" ] && { echo "REFUSED: $D/pod_add122 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
REMOTE="cd $ROOT && T=\$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add122_\$T.tar && M=\"\" && M=\"\$M \$(find read records fit_out/records fit_out/fits -type f $EXT 2>/dev/null | tr '\\n' ' ')\" && M=\"\$M \$(find . -maxdepth 3 -type f \\( -name 'TELEGAP_*' -o -name 'START_*' -o -name 'DONE_*' -o -name 'STOP_*' -o -name '*times*' -o -name 'FIT_RECORD*' -o -name 'T2.json' \\) $EXT 2>/dev/null | tr '\\n' ' ')\" && tar cf \$N --exclude=\"*.pt\" --exclude=\"*.npz\" --exclude=\"*.npy\" --exclude=\"*.mp4\" \$M && echo \"ZZNAME \$N\" && echo \"ZZSHA \$(sha256sum < \$N | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 \$N)\""
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add122_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add122_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add122_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add122_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add122_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add122" && tar xf "$D/tmp_pull_add122_$LT.tar" -C "$D/pod_add122" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add122_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add122" -type f | wc -l) files"
