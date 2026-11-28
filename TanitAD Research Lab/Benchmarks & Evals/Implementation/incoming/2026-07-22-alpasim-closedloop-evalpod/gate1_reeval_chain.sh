#!/bin/bash
# Wait for the baseline re-eval to finish, then run the FT re-evals sequentially.
# The renderer on :6011 persists across runs (gate1_run.sh only kills 6789/6007/6006).
set -uo pipefail
rm -f /workspace/gate1_reeval_CHAIN_DONE
echo "[chain] waiting for baseline DONE..."
for i in $(seq 1 240); do [ -f /workspace/gate1_reeval_base/DONE ] && break; sleep 15; done
echo "[chain] baseline done: $(cat /workspace/gate1_reeval_base/DONE 2>/dev/null)"
bash /workspace/gate1_reeval.sh ft800 /root/models/refc-gate1-ft/ckpt.pt base
bash /workspace/gate1_reeval.sh ft300 /root/models/refc-gate1-ft/ckpt_step300.pt base
echo "CHAIN_DONE" > /workspace/gate1_reeval_CHAIN_DONE
echo "[chain] all re-evals complete"
