#!/bin/bash
# pull_steerread.sh -- lead's pull of the records of card PREREG_V8F_PROOF_ADD100 (STEERREAD) from the fork pod into cropread/pod_add100.
# Pulls control and result records, queue markers and logs, fit and localiser records, dump meta files, the manifest of arm S. No tokens, no steering files, no dumps, no camera pixels.
# Refuses an existing pod_add100. One tar on the pod (s_out/pull_add100.tar, refused when it exists), base64 through the ssh helper, sha256 compared.
set -u
D=/f/TanitAD-scratch/v8proof/cropread
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/cropread/s_out
[ -e "$D/pod_add100" ] && { echo "REFUSED: $D/pod_add100 exists"; exit 2; }
timeout 240 bash "$PS" "cd $ROOT && [ ! -e pull_add100.tar ] && tar cf pull_add100.tar ctl read queue.log START.txt DONE.txt queue/START_LOC.txt queue/DONE_LOC.txt queue/START_HEADS.txt queue/DONE_HEADS.txt queue/jobs queue/logs loc/*/LOC_RECORD.json fits/*/FIT_RECORD.json dump/*/reader_s*_meta.json tok/S/MANIFEST.json loc_queue.nohup enc_queue.nohup heads_queue.nohup podread_controls.nohup podread_names.nohup && echo \"ZZSHA \$(sha256sum < pull_add100.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add100.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add100.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add100.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add100.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add100.tar"
got=$(sha256sum < "$D/tmp_pull_add100.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add100" && tar xf "$D/tmp_pull_add100.tar" -C "$D/pod_add100" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add100" -type f | wc -l) files $(du -sh "$D/pod_add100" | cut -f1)"
