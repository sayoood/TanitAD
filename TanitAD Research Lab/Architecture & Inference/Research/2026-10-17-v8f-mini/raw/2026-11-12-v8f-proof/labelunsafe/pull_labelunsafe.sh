#!/bin/bash
# pull_labelunsafe.sh -- pulls the small text results of card ADD135 from the fork pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: the controls file, the result file, the verdict, the logs of the two stages and the pod test outputs. Never a dump file (.npz).
# Reads only; on the pod it writes one new tar file in the card's own folder. Refuses when the local folder pod_add135 exists. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/labelunsafe
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/labelunsafe
[ -e "$D/pod_add135" ] && { echo "REFUSED: $D/pod_add135 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
read -r -d '' BODY <<'EOS'
cd ROOT_PLACEHOLDER || exit 1
T=$(date -u +%Y%m%dT%H%M%SZ)
N=pull_add135_$T.tar
LIST=$(find read_out lead_tests -maxdepth 1 -type f \( -name '*.json' -o -name '*.out' -o -name '*.md' -o -name '*.log' \) 2>/dev/null | sort -u | tr '\n' ' ')
tar cf "$N" --exclude='*.npz' --exclude='*.npy' $LIST
echo "ZZNAME $N"
echo "ZZSHA $(sha256sum < "$N" | cut -c1-64)"
echo "ZZB64 $(base64 -w0 "$N")"
EOS
REMOTE=${BODY//ROOT_PLACEHOLDER/$ROOT}
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add135_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add135_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add135_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add135_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add135_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add135" && tar xf "$D/tmp_pull_add135_$LT.tar" -C "$D/pod_add135" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add135_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add135" -type f | wc -l) files"
echo "TRANSFER-FILES $D/tmp_pull_add135_$LT.tar $D/tmp_pull_add135_$LT.txt"
