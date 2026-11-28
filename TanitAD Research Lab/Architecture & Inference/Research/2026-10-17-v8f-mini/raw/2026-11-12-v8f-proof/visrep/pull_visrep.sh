#!/bin/bash
# pull_visrep.sh -- pulls the small text results of card ADD103 from the pod (the gap-file build output and record, the dry run output, the queue logs and markers of both model seeds, the read folder) as one tar over
# base64 and compares the sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the card's own root. It opens no result value. Needs the queue run once per model seed
# (GR_SEED=0 and GR_SEED=1: that is what writes the _s0 and _s1 markers named here) and both stages of podread_visrep10k.sh.
set -u
D=/f/TanitAD-scratch/v8proof/visrep
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/visrep10k
FILES="make_gapfiles.out gapfiles/VISREP_GAPFILES_RECORD.json dry_all.out queue_visrep_s0.log queue_visrep_s1.log START_VISREP10K_s0.txt START_VISREP10K_s1.txt VISREP10K_DONE_s0.txt VISREP10K_DONE_s1.txt read"
[ -e "$D/pod_add103" ] && { echo "REFUSED: $D/pod_add103 exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_add103.tar ] && tar cf pull_add103.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_add103.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add103.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add103.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add103.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add103.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add103.tar"
got=$(sha256sum < "$D/tmp_pull_add103.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add103" && tar xf "$D/tmp_pull_add103.tar" -C "$D/pod_add103" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add103" -type f | wc -l) files"
