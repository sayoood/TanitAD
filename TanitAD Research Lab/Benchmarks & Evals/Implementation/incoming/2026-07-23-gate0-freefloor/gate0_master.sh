#!/bin/bash
# GATE-0 autonomous master: acquire lock -> renderer -> REF-C-base floor-OFF (control)
# + floor-ON over the 38-scene balanced suite -> cleanup -> release lock.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
ML=/workspace/gate0_master.log; : > "$ML"
mlog(){ echo "[$(date -u +%H:%M:%S)] GATE0 $*" | tee -a "$ML"; }
rm -f /workspace/gate0_MASTER_DONE
mlog "=== GATE0 START ==="

gpu_lock.sh acquire gate0-run 2>&1 | tee -a "$ML" || { mlog "LOCK acquire FAIL"; echo FAIL_lock > /workspace/gate0_MASTER_DONE; exit 5; }
mlog "lock: $(gpu_lock.sh status 2>&1 | tr '\n' ' ')"

CFG=/workspace/scaled_suite/generated-user-config-0.yaml
SSHASH=$(grep -m1 data_dir "$CFG" | awk '{print $NF}' | sed 's#.*/scenesets/##' | tr -d '"')
SSDIR="$ALPA/data/nre-artifacts/scenesets/$SSHASH"
GLOB="$SSDIR/**/*.usdz"
mlog "sceneset=$SSHASH usdz=$(find "$SSDIR" -name '*.usdz' 2>/dev/null | wc -l)"

if ! ss -ltn | grep -q ':6011 '; then
  mlog "launch renderer"; setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/gate0_renderer.log 2>&1 &
  for i in $(seq 1 150); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/gate0_renderer.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && mlog "renderer up" || { mlog "renderer FAIL"; gpu_lock.sh release gate0-run; echo FAIL_renderer > /workspace/gate0_MASTER_DONE; exit 4; }

for d in gate0_off gate0_on; do
  rm -rf /workspace/$d; mkdir -p /workspace/$d/controller
  for f in controller-config.yaml driver-config.yaml eval-config.yaml generated-network-config.yaml \
           generated-user-config-0.yaml run_metadata.yaml trafficsim-config.yaml \
           wizard-config-loadable.yaml wizard-config.yaml; do cp /workspace/scaled_suite/$f /workspace/$d/$f; done
  mlog "built /workspace/$d ($(grep -c scene_id: /workspace/$d/generated-user-config-0.yaml) scenes)"
done

mlog ">>> FLOOR-ON START (novel arm first; existing scaled baseline is fallback control)"
bash /workspace/gate0_run.sh /workspace/gate0_on on /workspace/gate0_floor_ON.jsonl "$SSDIR"
mlog "floor-on: $(cat /workspace/gate0_on/DONE 2>/dev/null)"

mlog ">>> FLOOR-OFF (paired control) START"
bash /workspace/gate0_run.sh /workspace/gate0_off off /workspace/gate0_floor_OFF.jsonl "$SSDIR"
mlog "floor-off: $(cat /workspace/gate0_off/DONE 2>/dev/null)"

for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "cleanup killed :$p pid=$pid"
done
sleep 3
gpu_lock.sh release gate0-run 2>&1 | tee -a "$ML"
mlog "GPU procs after release: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
echo "GATE0_DONE off=[$(cat /workspace/gate0_off/DONE 2>/dev/null)] on=[$(cat /workspace/gate0_on/DONE 2>/dev/null)]" > /workspace/gate0_MASTER_DONE
mlog "=== GATE0 DONE ==="
