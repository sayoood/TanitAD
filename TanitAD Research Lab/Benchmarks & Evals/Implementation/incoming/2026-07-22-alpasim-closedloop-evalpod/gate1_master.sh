#!/bin/bash
# gate1_master.sh -- AUTONOMOUS Gate-1 on-policy rollout collection.
# Launch renderer on the junction sceneset, run REF-C-base closed-loop over the 15 junction
# scenes (intersection+roundabout) WITH per-step pose+plan logging (gate1_run.sh --log-preds),
# then kill all services incl. renderer. Does NOT release the gpu_lock (post-processing follows).
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
DST=/workspace/gate1_junc
ML=/workspace/gate1_master.log
mlog(){ echo "[$(date -u +%H:%M:%S)] GATE1 $*" | tee -a "$ML"; }
rm -f /workspace/gate1_MASTER_DONE
mlog "=== GATE1 START ==="
SS=$(grep -m1 data_dir "$DST/generated-user-config-0.yaml" | awk '{print $NF}' | tr -d '"')
GLOB="$SS/**/*.usdz"
mlog "sceneset=$SS usdz=$(find "$SS" -name '*.usdz' 2>/dev/null | wc -l) scenes_in_cfg=$(grep -c 'scene_id:' "$DST/generated-user-config-0.yaml")"

if ! ss -ltn | grep -q ':6011 '; then
  mlog "launch renderer"
  setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/gate1_renderer.log 2>&1 &
  for i in $(seq 1 120); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/gate1_renderer.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && mlog "renderer up" || { mlog "renderer FAIL"; echo FAIL_renderer > /workspace/gate1_MASTER_DONE; exit 4; }

mlog ">>> REF-C base junction run START"
bash /workspace/gate1_run.sh "$DST" /workspace/refc_driver.py /root/models/refc-base-30k/ckpt.pt base
mlog "REF-C base junction: $(cat "$DST/DONE" 2>/dev/null)"

# cleanup: kill driver/controller/physics AND renderer (leave GPU clean; lock kept for post-proc)
for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "cleanup killed :$p pid=$pid"
done
sleep 3
mlog "GPU procs after cleanup: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
echo "GATE1_DONE $(cat "$DST/DONE" 2>/dev/null)" > /workspace/gate1_MASTER_DONE
mlog "=== GATE1 DONE ==="
