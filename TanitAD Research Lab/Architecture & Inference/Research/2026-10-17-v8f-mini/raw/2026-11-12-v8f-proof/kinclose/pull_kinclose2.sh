#!/bin/bash
# pull_kinclose2.sh -- pulls the text results of card ADD89 from the pod into kinclose/pod (control file, result file, the three logs, markers) as one tar over base64 and compares the
# sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the chain's own root. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/kinclose
PS=/c/Users/Admin/v8pod/pssh.sh
[ -e "$D/pod2" ] && { echo "REFUSED: $D/pod2 exists"; exit 2; }
timeout 120 bash "$PS" 'cd /workspace/v8proof/kinclose2b && [ -f KINCLOSE_DONE.txt ] && [ ! -e pull_kinclose2.tar ] && tar cf pull_kinclose2.tar out/KINCLOSE_CONTROLS.json out/KINCLOSE_RESULT.json logs KINCLOSE_DONE.txt START_KINCLOSE.txt && echo "ZZSHA $(sha256sum < pull_kinclose2.tar | cut -c1-64)" && echo "ZZB64 $(base64 -w0 pull_kinclose2.tar)"' 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull2.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull2.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull2.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull2.tar"
got=$(sha256sum < "$D/tmp_pull2.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod2" && tar xf "$D/tmp_pull2.tar" -C "$D/pod2" && echo "PULL-OK ${got:0:16} $(find "$D/pod2" -type f | wc -l) files"
