#!/bin/bash
# Recover aggregation for scaled_refc + scaled_flag: both have 37 _complete rollouts but no
# results-summary.json (one scene failed a route sanity check + allow_aggregation was false).
# Flag is now true; autoresume skips the 37 done, retries the 1 failed scene, then aggregates.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
ML=/workspace/reagg_master.log; : > "$ML"
mlog(){ echo "[$(date -u +%H:%M:%S)] REAGG $*" | tee -a "$ML"; }
rm -f /workspace/reagg_DONE
gpu_lock.sh acquire scenario-scaleup | tee -a "$ML"
CFG=/workspace/scaled_suite/generated-user-config-0.yaml
SSHASH=$(grep -m1 data_dir "$CFG" | awk '{print $NF}' | sed 's#.*/scenesets/##' | tr -d '"')
GLOB="$ALPA/data/nre-artifacts/scenesets/$SSHASH/**/*.usdz"
if ! ss -ltn | grep -q ':6011 '; then
  mlog "launch renderer"; setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/renderer_suite.log 2>&1 &
  for i in $(seq 1 120); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/renderer_suite.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && mlog "renderer up" || { mlog "renderer FAIL"; echo FAIL > /workspace/reagg_DONE; exit 4; }

mlog ">>> REF-C reagg (autoresume; before rollouts=$(ls /workspace/scaled_refc/rollouts/ 2>/dev/null | wc -l))"
bash /workspace/vs_suite_run.sh /workspace/scaled_refc /workspace/refc_driver.py /root/models/refc-base-30k/ckpt.pt base
mlog "REF-C: $(cat /workspace/scaled_refc/DONE 2>/dev/null) agg=$([ -f /workspace/scaled_refc/aggregate/results-summary.json ] && echo YES || echo NO)"

mlog ">>> flagship reagg (autoresume; before rollouts=$(ls /workspace/scaled_flag/rollouts/ 2>/dev/null | wc -l))"
bash /workspace/vs_suite_run.sh /workspace/scaled_flag /workspace/flagship_v1_driver.py /root/models/flagship-30k/ckpt.pt
mlog "flag: $(cat /workspace/scaled_flag/DONE 2>/dev/null) agg=$([ -f /workspace/scaled_flag/aggregate/results-summary.json ] && echo YES || echo NO)"

for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "killed :$p"
done
sleep 3
gpu_lock.sh release scenario-scaleup 2>&1 | tee -a "$ML"
echo "REAGG_DONE refc_agg=$([ -f /workspace/scaled_refc/aggregate/results-summary.json ] && echo YES || echo NO) flag_agg=$([ -f /workspace/scaled_flag/aggregate/results-summary.json ] && echo YES || echo NO)" > /workspace/reagg_DONE
mlog "=== REAGG DONE ==="
