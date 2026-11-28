#!/bin/bash
# pull_boxsize.sh -- pulls the small text results of card ADD124 from the pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Text only: .json .log .out .txt under the read folder, the BOXSIZE_FIT_RECORD files of the fits, the identity records and the markers of the queue. Never a FIT_RECORD.json of the locator tool
# (it holds clip lists), never a head file (.pt), a score, token or prediction file (.npz, .npy), never a video (.mp4).
# Reads only; on the pod it writes one new tar file in the card's own root. Refuses when the local folder pod_add124 exists. It opens no result value. It prints no pod address.
set -u
D=/f/TanitAD-scratch/v8proof/boxsize
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/boxsize
[ -e "$D/pod_add124" ] && { echo "REFUSED: $D/pod_add124 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
read -r -d '' BODY <<'EOS'
cd ROOT_PLACEHOLDER || exit 1
T=$(date -u +%Y%m%dT%H%M%SZ)
N=pull_add124_$T.tar
LIST=$( { find read -type f \( -name '*.json' -o -name '*.log' -o -name '*.out' -o -name '*.txt' \) 2>/dev/null
          find fit_out -maxdepth 3 -type f \( -name 'BOXSIZE_FIT_RECORD.json' -o -name 'BOXSIZE_IDENTITY_RECORD.json' -o -name 'IDENTITY_K5.json' \) 2>/dev/null
          find . -maxdepth 2 -type f \( -name 'BOXSIZE_*' -o -name 'START_*' -o -name 'DONE_*' -o -name 'STOP_*' -o -name '*times*' \) \( -name '*.json' -o -name '*.log' -o -name '*.out' -o -name '*.txt' \) 2>/dev/null
        } | sed 's#^\./##' | grep -v '^FIT_RECORD\.json$' | grep -v '/FIT_RECORD\.json$' | sort -u | tr '\n' ' ')
tar cf "$N" --exclude='*.pt' --exclude='*.npz' --exclude='*.npy' --exclude='*.mp4' --exclude='FIT_RECORD.json' $LIST
echo "ZZNAME $N"
echo "ZZSHA $(sha256sum < "$N" | cut -c1-64)"
echo "ZZB64 $(base64 -w0 "$N")"
EOS
REMOTE=${BODY//ROOT_PLACEHOLDER/$ROOT}
timeout 300 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add124_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add124_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add124_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add124_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add124_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add124" && tar xf "$D/tmp_pull_add124_$LT.tar" -C "$D/pod_add124" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add124_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add124" -type f | wc -l) files"
