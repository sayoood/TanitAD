#!/bin/bash
# vs_suite_master_1080.sh -- AUTONOMOUS paired closed-loop suite at NATIVE 1080x1920.
# IDENTICAL to the 480x854 run except the camera render resolution (the ONLY change:
# resolves the environment/resolution confound). Same 12 scenes, same vs_suite_run.sh,
# same drivers/ckpts. Self-cleaning end (kills renderer + releases lock -> no orphan).
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
SCENESET_DIR=/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/e11a2e57085844fa5d905fa259abb344
GLOB="$SCENESET_DIR/**/*.usdz"
ML=/workspace/vs_master_1080.log
: > "$ML"
mlog(){ echo "[$(date -u +%H:%M:%S)] MASTER1080 $*" | tee -a "$ML"; }
rm -f /workspace/vs_MASTER_1080_DONE
mlog "=== MASTER1080 START (native 1080x1920) ==="

# 1. renderer up? (resolution-independent: the runtime requests the render size per camera config)
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
ss -ltn | grep -q ':6011 ' && mlog "renderer READY" || { mlog "renderer FAILED"; echo "MASTER_FAIL renderer" > /workspace/vs_MASTER_1080_DONE; exit 4; }

# 2. rebuild clean logdirs from refcsuite configs, then FLIP camera res 854->1920, 480->1080
for d in vs_refc_1080 vs_flag_1080; do
  rm -rf /workspace/$d; mkdir -p /workspace/$d/controller
  for f in controller-config.yaml driver-config.yaml eval-config.yaml generated-network-config.yaml \
           generated-user-config-0.yaml run_metadata.yaml trafficsim-config.yaml \
           wizard-config-loadable.yaml wizard-config.yaml; do
    cp /workspace/refcsuite/$f /workspace/$d/$f
  done
  sed -i 's/height: 480/height: 1080/; s/width: 854/width: 1920/' /workspace/$d/generated-user-config-0.yaml
  H=$(grep -m1 'height:' /workspace/$d/generated-user-config-0.yaml | tr -dc '0-9')
  W=$(grep -m1 'width:'  /workspace/$d/generated-user-config-0.yaml | tr -dc '0-9')
  STRAY=$(grep -cE 'height: 480|width: 854' /workspace/$d/generated-user-config-0.yaml)
  mlog "rebuilt /workspace/$d ($(grep -c scene_id: /workspace/$d/generated-user-config-0.yaml) scenes) cam=${W}x${H} stray854=$STRAY"
done

# 3. REF-C base (native)
mlog ">>> REF-C base native run START"
bash /workspace/vs_suite_run.sh /workspace/vs_refc_1080 /workspace/refc_driver.py /root/models/refc-base-30k/ckpt.pt base
mlog "REF-C base native: $(cat /workspace/vs_refc_1080/DONE 2>/dev/null)"

# 4. flagship v1 (native)
mlog ">>> flagship v1 native run START"
bash /workspace/vs_suite_run.sh /workspace/vs_flag_1080 /workspace/flagship_v1_driver.py /root/models/flagship-30k/ckpt.pt
mlog "flagship v1 native: $(cat /workspace/vs_flag_1080/DONE 2>/dev/null)"

# 5. cleanup ALL services incl. renderer -> pod CLEAN
for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "cleanup killed :$p pid=$pid"
done
sleep 3

# 6. release GPU lock + report clean
gpu_lock.sh release vs-native1080 2>&1 | tee -a "$ML"
mlog "GPU compute procs now: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
echo "MASTER1080_DONE refc=[$(cat /workspace/vs_refc_1080/DONE 2>/dev/null)] flag=[$(cat /workspace/vs_flag_1080/DONE 2>/dev/null)]" > /workspace/vs_MASTER_1080_DONE
mlog "=== MASTER1080 DONE ==="
