#!/bin/bash
# waits for SFT-1 to finish (any exit) and the training relabel to complete; then gate -> mutation must be RED -> run
L=/workspace/data/refe_sft2/chain.log
log() { echo "$(date -u +%FT%T) $*" >> $L; }
log "chain armed"
until grep -q "^ZZEXIT" /workspace/data/refe_sft1/sft.log 2>/dev/null; do sleep 120; done
log "SFT-1 finished: $(grep ^ZZEXIT /workspace/data/refe_sft1/sft.log)"
until grep -q "ZZTRAIN_RELABEL_EXIT" /workspace/data/refe_sft2/relabel_train.log 2>/dev/null; do sleep 120; done
log "relabel finished: $(grep ZZTRAIN_RELABEL_EXIT /workspace/data/refe_sft2/relabel_train.log)"
MODE=preflight bash /workspace/refe-sft2/sft2_pod.sh
if ! grep -q "^ZZSFT_PREFLIGHT_PASS" /workspace/data/refe_sft2/run/preflight.log; then log "REFUSED: preflight did not pass"; exit 1; fi
log "preflight PASS"
MODE=mutate bash /workspace/refe-sft2/sft2_pod.sh
if grep -q "^ZZSFT_PREFLIGHT_FAIL" /workspace/data/refe_sft2/run/mutate.log && grep -q "\"G8_lane_labels_in_bank\": {\"train_lane_sets\": 0" /workspace/data/refe_sft2/run/mutate.log; then
  log "MUTATION RED as required (empty lane dir -> G8 false)"
  MODE=run bash /workspace/refe-sft2/sft2_pod.sh
  log "RUN EXIT $(tail -1 /workspace/data/refe_sft2/run/sft.log)"
else
  log "REFUSED: the mutation did NOT fail G8"
fi
