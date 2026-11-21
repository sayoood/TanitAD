#!/bin/bash
# pull_visrule_lead.sh -- lead's pull of the records of card PREREG_V8F_PROOF_ADD102 (VISRULE10K) from the fork pod into visrule/pod_add102.
# Written after pull_visrule.sh stopped at its tar (it lists dry_s0.out; the lead's dry run wrote dry_all.out), which left a partial pull_add102.tar on the pod; this one writes
# pull_add102b.tar with the files that exist. Control and result records, queue logs and markers, the gap file record, the wrapper check record. No rollout dumps, no gap files, no tokens.
set -u
D=/f/TanitAD-scratch/v8proof/visrule
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/visrule10k
FILES="make_gapfiles.out gapfiles/VISRULE_GAPFILES_RECORD.json check_v.out check_v.nohup check_v/CHECK_WRAPPER_V_DONE.txt check_v/CHECK_WRAPPER_V.hashes dry_all.out queue_visrule_s0.log queue_visrule_s1.log queue_s0.nohup queue_s1.nohup START_VISRULE10K_s0.txt START_VISRULE10K_s1.txt VISRULE10K_DONE_s0.txt VISRULE10K_DONE_s1.txt podread_controls.nohup podread_names.nohup read"
[ -e "$D/pod_add102" ] && { echo "REFUSED: $D/pod_add102 exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_add102b.tar ] && tar cf pull_add102b.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_add102b.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add102b.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add102b.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add102b.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add102b.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add102b.tar"
got=$(sha256sum < "$D/tmp_pull_add102b.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add102" && tar xf "$D/tmp_pull_add102b.tar" -C "$D/pod_add102" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add102" -type f | wc -l) files $(du -sh "$D/pod_add102" | cut -f1)"
