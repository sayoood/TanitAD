#!/bin/bash
# pull_visseed.sh -- pulls the small text results of card ADD104 from the pod (the gap-file build output and record, the two dry run outputs, the fit queue markers, logs and job markers, the three fit records and the three dump
# meta files, the rollout queue logs and markers of both model seeds, the read folder) as one tar over base64 and compares the sha256 of the tar on both sides. No head file, no dump, no rollout file.
# Reads only; on the pod it writes one new tar file in the card's own root. It opens no result value. Needs the fit queue, the gap-file build (output to make_gapfiles.out), `DRY=1` runs of both queues (outputs dry_fits.out and
# dry_all.out), the rollout queue once per model seed (GR_SEED=0 and GR_SEED=1: that is what writes the _s0 and _s1 markers named here) and both stages of podread_visseed10k.sh.
set -u
D=/f/TanitAD-scratch/v8proof/visseed
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/visseed10k
FILES="make_gapfiles.out gapfiles/VISSEED_GAPFILES_RECORD.json dry_fits.out dry_all.out START_FITS.txt DONE_FITS.txt fits_q fitroot/fits/D_H5_s2/FIT_RECORD.json fitroot/fits/D_H5_s3/FIT_RECORD.json fitroot/fits/D_H5_s4/FIT_RECORD.json fitroot/dump/D_H5/reader_s2_meta.json fitroot/dump/D_H5/reader_s3_meta.json fitroot/dump/D_H5/reader_s4_meta.json queue_visseed_s0.log queue_visseed_s1.log START_VISSEED10K_s0.txt START_VISSEED10K_s1.txt VISSEED10K_DONE_s0.txt VISSEED10K_DONE_s1.txt read"
[ -e "$D/pod_add104" ] && { echo "REFUSED: $D/pod_add104 exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_add104.tar ] && tar cf pull_add104.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_add104.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add104.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add104.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add104.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add104.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add104.tar"
got=$(sha256sum < "$D/tmp_pull_add104.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add104" && tar xf "$D/tmp_pull_add104.tar" -C "$D/pod_add104" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add104" -type f | wc -l) files"
