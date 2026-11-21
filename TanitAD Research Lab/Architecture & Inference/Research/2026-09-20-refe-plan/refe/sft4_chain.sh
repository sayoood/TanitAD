#!/bin/bash
# waits for SFT-3 to exit and the PDM training relabel to finish; gate -> the empty-PDM mutation must turn G9 RED -> run
L=/workspace/data/refe_sft4/chain.log; mkdir -p /workspace/data/refe_sft4
log() { echo "$(date -u +%FT%T) $*" >> $L; }
log "chain armed"
until grep -q "^ZZEXIT" /workspace/data/refe_sft3/sft.log 2>/dev/null; do sleep 120; done
log "SFT-3 finished: $(grep ^ZZEXIT /workspace/data/refe_sft3/sft.log)"
until grep -q "ZZPDM_TRAIN_EXIT" /workspace/data/refe_sft2/pdm_train.log 2>/dev/null; do sleep 120; done
log "PDM relabel finished: $(grep ZZPDM_TRAIN_EXIT /workspace/data/refe_sft2/pdm_train.log)"
log "teacher-lane train rows at launch: $(cat /workspace/data/refe_sft2/train_teacher_lane/*.jsonl 2>/dev/null | wc -l)"
MODE=preflight bash /workspace/refe-sft4/sft4_pod.sh
grep -q "^ZZSFT_PREFLIGHT_PASS" /workspace/data/refe_sft4/preflight.log || { log "REFUSED: preflight did not pass"; exit 1; }
log "preflight PASS"
MODE=mutate bash /workspace/refe-sft4/sft4_pod.sh
if grep -q "^ZZSFT_PREFLIGHT_FAIL" /workspace/data/refe_sft4/mutate.log && grep -q "\"G9_pdm_targets_in_bank\": {\"train_pdm\": 0" /workspace/data/refe_sft4/mutate.log; then
  log "MUTATION RED as required (empty PDM dir -> G9 false)"; MODE=run bash /workspace/refe-sft4/sft4_pod.sh; log "RUN EXIT $(tail -1 /workspace/data/refe_sft4/sft.log)"
else log "REFUSED: the mutation did NOT fail G9"; fi
