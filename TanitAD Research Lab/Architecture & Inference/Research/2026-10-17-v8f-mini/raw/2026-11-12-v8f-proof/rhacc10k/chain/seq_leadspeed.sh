#!/usr/bin/env bash
# seq_leadspeed.sh -- cards ADD151 and ADD150 (with amendment 1) on box 2, one job at a time (PI 2026-11-25: "Yes, run both").
# Runs chain_rhacc.sh for each job in order; the next job starts only when the previous chain ended with queue-end
# (a temperature stop, a reboot, a STOP file or a launch failure ends the sequence). Markers: ZZSEQ <job> <end line>, ZZSEQ-END <state>.
set -u
C=F:/TanitAD-scratch/v8proof/rhacc10k/chain
for j in leadspeed_rep149 leadspeed_all leadspeed_all149; do
  bash $C/chain_rhacc.sh $j train_full_box2 > $C/chain_$j.out 2>&1
  e=$(grep "ZZRHACC-END" $C/chain_$j.out | tail -1)
  echo "ZZSEQ $j $e"
  case "$e" in *"queue-end"*) ;; *) echo "ZZSEQ-END stopped-after-$j $(date -u +%FT%TZ)"; exit 0 ;; esac
done
echo "ZZSEQ-END all-done $(date -u +%FT%TZ)"
