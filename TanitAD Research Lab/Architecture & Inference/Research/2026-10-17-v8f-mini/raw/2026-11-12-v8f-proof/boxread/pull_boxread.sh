#!/bin/bash
# pull_boxread.sh -- pulls the small text results of cards ADD127 and ADD128 from the fork pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: the controls file and log of card ADD127's run (read_out), the files of card ADD128's read (read_out_c), the lead's test outputs, the weights record and
# the two records of the detector passes. Never a detection file (.npz), never weights (.pth), never a frame.
# Reads only; on the pod it writes one new tar file in the card's own folder. Refuses when the local folder pod_add128 exists. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/boxread
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/boxread
[ -e "$D/pod_add128" ] && { echo "REFUSED: $D/pod_add128 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
read -r -d '' BODY <<'EOS'
cd ROOT_PLACEHOLDER || exit 1
T=$(date -u +%Y%m%dT%H%M%SZ)
N=pull_add128_$T.tar
LIST=$(find read_out read_out_c lead_tests -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' -o -name '*.out' -o -name '*.log' \) 2>/dev/null | sort -u | tr '\n' ' ')
tar cf "$N" --exclude='*.pth' --exclude='*.npz' --exclude='*.npy' $LIST fetch/WEIGHTS_RECORD.json out_full/wide/BOXDET_RECORD.json out_full/tele/BOXDET_RECORD.json out_full/BOXREAD_START.txt out_full/BOXREAD_DONE.txt
echo "ZZNAME $N"
echo "ZZSHA $(sha256sum < "$N" | cut -c1-64)"
echo "ZZB64 $(base64 -w0 "$N")"
EOS
REMOTE=${BODY//ROOT_PLACEHOLDER/$ROOT}
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add128_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add128_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add128_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add128_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add128_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add128" && tar xf "$D/tmp_pull_add128_$LT.tar" -C "$D/pod_add128" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add128_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add128" -type f | wc -l) files"
echo "TRANSFER-FILES $D/tmp_pull_add128_$LT.tar $D/tmp_pull_add128_$LT.txt"
