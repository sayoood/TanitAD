#!/bin/bash
# pull_dscale2x.sh -- pulls the small text results of card ADD123 from the fork pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: the files of the read folder (.json .md .out), the done file of the run, the smoke estimate and the train-done record of runQ. Never a state file (.npz, .npy),
# never a head file (.pt), never a list of evaluation ids or batches.
# Reads only; on the pod it writes one new tar file in the card's own back folder. Refuses when the local folder pod_add123 exists. It opens no result value. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/dscale_lead/add123
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/dscale_lead/add123/back
[ -e "$D/pod_add123" ] && { echo "REFUSED: $D/pod_add123 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
read -r -d '' BODY <<'EOS'
cd ROOT_PLACEHOLDER || exit 1
T=$(date -u +%Y%m%dT%H%M%SZ)
N=pull_add123_$T.tar
LIST=$( { find out_2x/read -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' -o -name '*.out' \) 2>/dev/null
          ls out_2x/DSCALE_2X_DONE.txt out_2x/SMOKE_ESTIMATE.json out_2x/runQ/TRAIN_DONE.json 2>/dev/null
        } | sort -u | tr '\n' ' ')
tar cf "$N" --exclude='*.pt' --exclude='*.npz' --exclude='*.npy' $LIST
echo "ZZNAME $N"
echo "ZZSHA $(sha256sum < "$N" | cut -c1-64)"
echo "ZZB64 $(base64 -w0 "$N")"
EOS
REMOTE=${BODY//ROOT_PLACEHOLDER/$ROOT}
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add123_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add123_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add123_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add123_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add123_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add123" && tar xf "$D/tmp_pull_add123_$LT.tar" -C "$D/pod_add123" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add123_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add123" -type f | wc -l) files"
