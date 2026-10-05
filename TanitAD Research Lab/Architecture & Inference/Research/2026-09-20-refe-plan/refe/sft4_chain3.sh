#!/bin/bash
# SFT-4 chain v3 (PREREG_SFT4 Amendment 4, PI 2026-10-05: "I approve all your proposed levers, use the pod optimally").
# The A40 has idled since SFT-3 finished (00:50 UTC). v2 waited for the LAST PDM label (~07:30 UTC) plus a lane window
# (~1.5-2.5 h). v3 launches NOW on the PDM labels written so far. The joint fine-tune (PREREG_P2) will train on ALL of
# them, so nothing is lost; SFT-4's data event and G9 log the counts at launch.
# The PDM relabel keeps running beside it (both fit: ~30 GB anon measured with SFT-3 + PDM), and the lane relabel
# restarts at ZZPDM_TRAIN_EXIT (lane_restart.sh) for the joint fine-tune's lane labels.
# Order unchanged: gate -> the empty-PDM mutation must turn G9 RED -> run.
L=/workspace/data/refe_sft4/chain.log; mkdir -p /workspace/data/refe_sft4
log() { echo "$(date -u +%FT%T) $*" >> $L; }
log "chain v3 armed (Amendment 4: launch now): PDM train rows $(cat /workspace/data/refe_sft2/train_pdm/*.jsonl | wc -l), lane rows $(cat /workspace/data/refe_sft2/train_teacher_lane/*.jsonl | wc -l)"
MODE=preflight bash /workspace/refe-sft4/sft4_pod.sh
grep -q "^ZZSFT_PREFLIGHT_PASS" /workspace/data/refe_sft4/preflight.log || { log "REFUSED: preflight did not pass"; exit 1; }
log "preflight PASS"
MODE=mutate bash /workspace/refe-sft4/sft4_pod.sh
if grep -q "^ZZSFT_PREFLIGHT_FAIL" /workspace/data/refe_sft4/mutate.log && grep -q "\"G9_pdm_targets_in_bank\": {\"train_pdm\": 0" /workspace/data/refe_sft4/mutate.log; then
  log "MUTATION RED as required (empty PDM dir -> G9 false)"
  log "run launch: PDM train rows $(cat /workspace/data/refe_sft2/train_pdm/*.jsonl | wc -l)"
  MODE=run bash /workspace/refe-sft4/sft4_pod.sh; log "RUN EXIT $(tail -1 /workspace/data/refe_sft4/sft.log)"
else log "REFUSED: the mutation did NOT fail G9"; fi
