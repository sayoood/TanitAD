#!/bin/bash
# AUTONOMOUS: run REF-C base + flagship v1 over the balanced scenario suite (38 scenes, 480x854),
# then clean up ALL services + release the lock. Assumes /workspace/scaled_suite config+USDZs ready.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
ML=/workspace/scaled_master.log; : > "$ML"
mlog(){ echo "[$(date -u +%H:%M:%S)] SCALED $*" | tee -a "$ML"; }
rm -f /workspace/scaled_MASTER_DONE
mlog "=== SCALED START ==="
CFG=/workspace/scaled_suite/generated-user-config-0.yaml
SSHASH=$(grep -m1 data_dir "$CFG" | awk '{print $NF}' | sed 's#.*/scenesets/##' | tr -d '"')
SSDIR="$ALPA/data/nre-artifacts/scenesets/$SSHASH"
GLOB="$SSDIR/**/*.usdz"
mlog "sceneset=$SSHASH usdz=$(find "$SSDIR" -name '*.usdz' 2>/dev/null | wc -l)"

if ! ss -ltn | grep -q ':6011 '; then
  mlog "launch renderer"; setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/renderer_suite.log 2>&1 &
  for i in $(seq 1 120); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/renderer_suite.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && mlog "renderer up" || { mlog "renderer FAIL"; echo FAIL_renderer > /workspace/scaled_MASTER_DONE; exit 4; }

for d in scaled_refc scaled_flag; do
  rm -rf /workspace/$d; mkdir -p /workspace/$d/controller
  for f in controller-config.yaml driver-config.yaml eval-config.yaml generated-network-config.yaml \
           generated-user-config-0.yaml run_metadata.yaml trafficsim-config.yaml \
           wizard-config-loadable.yaml wizard-config.yaml; do cp /workspace/scaled_suite/$f /workspace/$d/$f; done
  mlog "built /workspace/$d ($(grep -c scene_id: /workspace/$d/generated-user-config-0.yaml) scenes)"
done

mlog ">>> REF-C base run START"
bash /workspace/vs_suite_run.sh /workspace/scaled_refc /workspace/refc_driver.py /root/models/refc-base-30k/ckpt.pt base
mlog "REF-C base: $(cat /workspace/scaled_refc/DONE 2>/dev/null)"

mlog ">>> flagship v1 run START"
bash /workspace/vs_suite_run.sh /workspace/scaled_flag /workspace/flagship_v1_driver.py /root/models/flagship-30k/ckpt.pt
mlog "flagship v1: $(cat /workspace/scaled_flag/DONE 2>/dev/null)"

for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "cleanup killed :$p pid=$pid"
done
sleep 3
gpu_lock.sh release scenario-scaleup 2>&1 | tee -a "$ML"
mlog "GPU procs: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
echo "SCALED_DONE refc=[$(cat /workspace/scaled_refc/DONE 2>/dev/null)] flag=[$(cat /workspace/scaled_flag/DONE 2>/dev/null)]" > /workspace/scaled_MASTER_DONE
mlog "=== SCALED DONE ==="
