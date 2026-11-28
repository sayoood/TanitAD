#!/bin/bash
# pull_cropread.sh -- pulls the small text results of card ADD99 from the pod (control records, lists, fit records, dump meta files, the read folder, queue logs) as one tar
# over base64 and compares the sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the card's own root. No token, no frame, no weight file.
set -u
D=/f/TanitAD-scratch/v8proof/cropread
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/cropread
[ -e "$D/pod_add99" ] && { echo "REFUSED: $D/pod_add99 exists"; exit 2; }
timeout 240 bash "$PS" "cd $ROOT && [ ! -e pull_add99.tar ] && tar cf pull_add99.tar enc_out/ctl enc_out/lists enc_out/queue.log enc_out/START.txt enc_out/DONE.txt run1/ctl run1/lists/LISTS_RECORD.json run1/read run1/queue/START_FITQ.txt run1/queue/DONE_FITQ.txt run1/queue/jobs run1/queue/logs run1/fits/*/FIT_RECORD.json run1/dump/*/reader_s*_meta.json run1/podread_controls.out run1/podread_controls_attempt1.out run1/podread_controls2.out run1/podread_names.out kit/lead && echo \"ZZSHA \$(sha256sum < pull_add99.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add99.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add99.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add99.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add99.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add99.tar"
got=$(sha256sum < "$D/tmp_pull_add99.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add99" && tar xf "$D/tmp_pull_add99.tar" -C "$D/pod_add99" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add99" -type f | wc -l) files $(du -sh "$D/pod_add99" | cut -f1)"
