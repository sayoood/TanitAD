#!/bin/bash
# pull_visens.sh -- pulls the small text results of card ADD105 from the pod (the gap-file build output and record, the dry run output, the rollout queue logs and markers of both model seeds, the read folder)
# as one tar over base64 and compares the sha256 of the tar on both sides. No head file, no dump, no rollout file.
# Reads only; on the pod it writes one new tar file in the card's own root. It opens no result value. Needs the gap-file build (output to make_gapfile.out), a `DRY=1` run of the queue (output dry_all.out),
# the rollout queue once per model seed (GR_SEED=0 and GR_SEED=1: that is what writes the _s0 and _s1 markers named here) and both stages of podread_visens10k.sh.
set -u
D=/f/TanitAD-scratch/v8proof/visens
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/visens10k
FILES="make_gapfile.out gapfiles/VISENS_GAPFILES_RECORD.json dry_all.out queue_visens_s0.log queue_visens_s1.log START_VISENS10K_s0.txt START_VISENS10K_s1.txt VISENS10K_DONE_s0.txt VISENS10K_DONE_s1.txt read"
[ -e "$D/pod_add105" ] && { echo "REFUSED: $D/pod_add105 exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_add105.tar ] && tar cf pull_add105.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_add105.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add105.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add105.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add105.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add105.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add105.tar"
got=$(sha256sum < "$D/tmp_pull_add105.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add105" && tar xf "$D/tmp_pull_add105.tar" -C "$D/pod_add105" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add105" -type f | wc -l) files"
