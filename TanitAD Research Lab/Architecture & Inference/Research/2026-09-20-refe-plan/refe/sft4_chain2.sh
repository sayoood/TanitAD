#!/bin/bash
# SFT-4 chain v2 (PREREG_SFT4 Amendment 3, 2026-10-04): v1 + a bounded teacher-lane catch-up window.
# Waits for SFT-3 to exit and the PDM training relabel to finish. lane_pause.sh resumes the lane relabel at that moment,
# so v1 would have launched with only the ~2,100 lane sets written before the pause. Gives the lane relabel the whole
# pod for up to 150 min, or until 7,000 lane sets exist. Then: gate -> the empty-PDM mutation must turn G9 RED -> run.
L=/workspace/data/refe_sft4/chain.log; mkdir -p /workspace/data/refe_sft4
log() { echo "$(date -u +%FT%T) $*" >> $L; }
lane_rows() { cat /workspace/data/refe_sft2/train_teacher_lane/*.jsonl 2>/dev/null | wc -l; }
log "chain v2 armed (replaces v1: lane catch-up window)"
until grep -q "^ZZEXIT" /workspace/data/refe_sft3/sft.log 2>/dev/null; do sleep 120; done
log "SFT-3 finished: $(grep ^ZZEXIT /workspace/data/refe_sft3/sft.log)"
until grep -q "ZZPDM_TRAIN_EXIT" /workspace/data/refe_sft2/pdm_train.log 2>/dev/null; do sleep 120; done
log "PDM relabel finished: $(grep ZZPDM_TRAIN_EXIT /workspace/data/refe_sft2/pdm_train.log); lane rows $(lane_rows)"
t0=$(date +%s)
until [ "$(lane_rows)" -ge 7000 ] || [ $(( $(date +%s) - t0 )) -ge 9000 ]; do sleep 120; done
log "lane window closed after $(( ($(date +%s) - t0) / 60 )) min; teacher-lane train rows at launch: $(lane_rows)"
MODE=preflight bash /workspace/refe-sft4/sft4_pod.sh
grep -q "^ZZSFT_PREFLIGHT_PASS" /workspace/data/refe_sft4/preflight.log || { log "REFUSED: preflight did not pass"; exit 1; }
log "preflight PASS"
MODE=mutate bash /workspace/refe-sft4/sft4_pod.sh
if grep -q "^ZZSFT_PREFLIGHT_FAIL" /workspace/data/refe_sft4/mutate.log && grep -q "\"G9_pdm_targets_in_bank\": {\"train_pdm\": 0" /workspace/data/refe_sft4/mutate.log; then
  log "MUTATION RED as required (empty PDM dir -> G9 false)"; MODE=run bash /workspace/refe-sft4/sft4_pod.sh; log "RUN EXIT $(tail -1 /workspace/data/refe_sft4/sft.log)"
else log "REFUSED: the mutation did NOT fail G9"; fi
