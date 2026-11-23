#!/bin/bash
# pull_dscale_pres.sh -- pulls the small text results of card ADD126 from the fork pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: the files of the out folder of the card (.json .md .out). Never a state file (.npz, .npy), never a head file (.pt).
# Reads only; on the pod it writes one new tar file in the card's own folder. Refuses when the local folder pod_add126 exists. It opens no result value. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/dscale_lead/add126
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/dscale_lead/add126
[ -e "$D/pod_add126" ] && { echo "REFUSED: $D/pod_add126 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
read -r -d '' BODY <<'EOS'
cd ROOT_PLACEHOLDER || exit 1
T=$(date -u +%Y%m%dT%H%M%SZ)
N=pull_add126_$T.tar
LIST=$(find out -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' -o -name '*.out' \) 2>/dev/null | sort -u | tr '\n' ' ')
tar cf "$N" --exclude='*.pt' --exclude='*.npz' --exclude='*.npy' $LIST
echo "ZZNAME $N"
echo "ZZSHA $(sha256sum < "$N" | cut -c1-64)"
echo "ZZB64 $(base64 -w0 "$N")"
EOS
REMOTE=${BODY//ROOT_PLACEHOLDER/$ROOT}
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add126_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add126_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add126_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add126_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add126_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add126" && tar xf "$D/tmp_pull_add126_$LT.tar" -C "$D/pod_add126" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add126_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add126" -type f | wc -l) files"
echo "TRANSFER-FILES $D/tmp_pull_add126_$LT.tar $D/tmp_pull_add126_$LT.txt"
