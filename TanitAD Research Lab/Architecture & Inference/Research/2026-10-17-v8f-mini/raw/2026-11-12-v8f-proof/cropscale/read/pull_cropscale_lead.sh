#!/bin/bash
# pull_cropscale_lead.sh -- lead copy of pull_cropscale.sh with six more log files in the list; pulls the small text results of card ADD106 from the pod (encode controls and lists record, the gap-file build output and record, the dry run outputs, the fit queue markers, logs and job markers,
# the three fit records and the permuted-label fit record, the dump meta files, the rollout queue logs and markers of both model seeds, the read folder) as one tar over base64 and compares the sha256 of the tar on both sides.
# No head file, no dump, no token, no staged video, no rollout file. Reads only; on the pod it writes one new tar file in the card's own root.
# The tar is built on the pod only from the members that exist (a missing optional file cannot fail the tar) and its name carries a time stamp, so a second pull cannot collide with the first. It opens no result value.
# Refuses when the local folder cropscale/pod_add106 exists.
set -u
D=/f/TanitAD-scratch/v8proof/cropscale
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/cropscale
FILES="ENC_CONTROLS.json ENC_IDENTITY.json ENC_TIMES.txt START_ENC.txt DONE_ENC.txt STOP_ENC.txt lists/LISTS_LARGE_RECORD.json make_gapfiles.out gapfiles/CROPSCALE_GAPFILES_RECORD.json dry_fits.out dry_all.out fit_queue.out perm_queue.out \
START_FITS.txt DONE_FITS.txt STOP_FITS.txt START_PERM.txt DONE_PERM.txt STOP_PERM.txt fits_q perm_q fitroot/fits/D_H5_s2/FIT_RECORD.json fitroot/fits/D_H5_s3/FIT_RECORD.json fitroot/fits/D_H5_s4/FIT_RECORD.json \
fitroot/fits/D_H5_s2_perm/FIT_RECORD.json fitroot/dump/D_H5/reader_s2_meta.json fitroot/dump/D_H5/reader_s3_meta.json fitroot/dump/D_H5/reader_s4_meta.json fitroot/dump/D_H5_perm/reader_s2_meta.json \
queue_cropscale_s0.log queue_cropscale_s1.log START_CROPSCALE10K_s0.txt START_CROPSCALE10K_s1.txt CROPSCALE10K_DONE_s0.txt CROPSCALE10K_DONE_s1.txt CROPSCALE10K_STOP_s0.txt CROPSCALE10K_STOP_s1.txt TIMES_NOTE.txt fit_queue.log fit_queue_b.log fit_queue_b2.log rollq_s0.out rollq_s1.out podread_controls.out read"
[ -e "$D/pod_add106" ] && { echo "REFUSED: $D/pod_add106 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
REMOTE='cd ROOT_ && T=$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add106_$T.tar && M="" && for f in FILES_; do if [ -e "$f" ]; then M="$M $f"; fi; done && [ -n "$M" ] && tar cf $N $M && echo "ZZNAME $N" && echo "ZZSHA $(sha256sum < $N | cut -c1-64)" && echo "ZZB64 $(base64 -w0 $N)"'
REMOTE=${REMOTE/ROOT_/$ROOT}
REMOTE=${REMOTE/FILES_/$FILES}
timeout 180 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add106_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add106_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add106_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add106_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add106_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add106" && tar xf "$D/tmp_pull_add106_$LT.tar" -C "$D/pod_add106" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add106_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add106" -type f | wc -l) files"
