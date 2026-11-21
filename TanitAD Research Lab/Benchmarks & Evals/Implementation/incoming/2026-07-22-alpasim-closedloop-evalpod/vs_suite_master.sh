#!/bin/bash
# vs_suite_master.sh -- AUTONOMOUS paired closed-loop suite on tanitad-eval.
# Runs REF-C base + flagship v1 over the SAME 12-scene suite, then cleans up ALL
# services (incl. the renderer on :6011) and RELEASES the GPU lock. Self-contained
# so a dropped controlling session leaves NO orphan (defined self-cleaning end).
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
SCENESET_DIR=/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/e11a2e57085844fa5d905fa259abb344
GLOB="$SCENESET_DIR/**/*.usdz"
ML=/workspace/vs_master.log
: > "$ML"
mlog(){ echo "[$(date -u +%H:%M:%S)] MASTER $*" | tee -a "$ML"; }
rm -f /workspace/vs_MASTER_DONE
mlog "=== MASTER START ==="

# 1. renderer up? (warm boot ~2 min, kernel cache persists at /workspace/nrehome/.cache)
if ss -ltn | grep -q ':6011 '; then
  mlog "renderer already up on :6011"
else
  mlog "launching renderer on :6011 ..."
  setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/renderer_suite.log 2>&1 &
  for i in $(seq 1 90); do
    ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/renderer_suite.log && break
    sleep 5
  done
fi
if ss -ltn | grep -q ':6011 '; then
  mlog "renderer READY: $(grep -c 'Available scenes' /workspace/renderer_suite.log) scenesets served"
else
  mlog "renderer FAILED"; echo "MASTER_FAIL renderer" > /workspace/vs_MASTER_DONE; exit 4
fi

# 2. rebuild clean logdirs from refcsuite configs (no stale rollouts -> clean aggregate)
for d in vs_refc vs_flag; do
  rm -rf /workspace/$d; mkdir -p /workspace/$d/controller
  for f in controller-config.yaml driver-config.yaml eval-config.yaml generated-network-config.yaml \
           generated-user-config-0.yaml run_metadata.yaml trafficsim-config.yaml \
           wizard-config-loadable.yaml wizard-config.yaml; do
    cp /workspace/refcsuite/$f /workspace/$d/$f
  done
  mlog "rebuilt /workspace/$d ($(grep -c scene_id: /workspace/$d/generated-user-config-0.yaml) scenes)"
done

# 3. REF-C base (known-good driver first -> validates the pipeline vs the prior suite)
mlog ">>> REF-C base run START"
bash /workspace/vs_suite_run.sh /workspace/vs_refc /workspace/refc_driver.py /root/models/refc-base-30k/ckpt.pt base
mlog "REF-C base: $(cat /workspace/vs_refc/DONE 2>/dev/null)"

# 4. flagship v1 (tactical-policy driver, same 12 scenes, same renderer)
mlog ">>> flagship v1 run START"
bash /workspace/vs_suite_run.sh /workspace/vs_flag /workspace/flagship_v1_driver.py /root/models/flagship-30k/ckpt.pt
mlog "flagship v1: $(cat /workspace/vs_flag/DONE 2>/dev/null)"

# 5. cleanup ALL services incl. renderer (:6011) -- leave the pod CLEAN
for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "cleanup killed :$p pid=$pid"
done
sleep 3

# 6. release GPU lock + report GPU clean
gpu_lock.sh release flagship-vs-refc 2>&1 | tee -a "$ML"
mlog "GPU compute procs now: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
echo "MASTER_DONE refc=[$(cat /workspace/vs_refc/DONE 2>/dev/null)] flag=[$(cat /workspace/vs_flag/DONE 2>/dev/null)]" > /workspace/vs_MASTER_DONE
mlog "=== MASTER DONE ==="
