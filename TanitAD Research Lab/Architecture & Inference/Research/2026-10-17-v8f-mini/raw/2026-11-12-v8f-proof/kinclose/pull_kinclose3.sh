#!/bin/bash
# pull_kinclose3.sh -- pulls the text results of card ADD90 from the pod into kinclose/pod (control file, result file, the three logs, markers) as one tar over base64 and compares the
# sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the chain's own root. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/kinclose
PS=/c/Users/Admin/v8pod/pssh.sh
[ -e "$D/pod3" ] && { echo "REFUSED: $D/pod3 exists"; exit 2; }
timeout 120 bash "$PS" 'cd /workspace/v8proof/kinclose3 && [ -f KINCLOSE3_DONE.txt ] && [ ! -e pull_kinclose3.tar ] && tar cf pull_kinclose3.tar out/KINCLOSE3_CONTROLS.json out/KINCLOSE3_RESULT.json logs KINCLOSE3_DONE.txt START_KINCLOSE3.txt && echo "ZZSHA $(sha256sum < pull_kinclose3.tar | cut -c1-64)" && echo "ZZB64 $(base64 -w0 pull_kinclose3.tar)"' 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull3.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull3.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull3.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull3.tar"
got=$(sha256sum < "$D/tmp_pull3.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod3" && tar xf "$D/tmp_pull3.tar" -C "$D/pod3" && echo "PULL-OK ${got:0:16} $(find "$D/pod3" -type f | wc -l) files"
