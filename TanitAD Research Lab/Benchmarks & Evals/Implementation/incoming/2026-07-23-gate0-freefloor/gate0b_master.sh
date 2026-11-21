#!/bin/bash
# GATE-0b autonomous master: renderer -> REF-C-base GRADIENT-NUDGE floor over the 38-scene suite
# -> cleanup -> release lock. Control = existing same-session gate0_off (deterministic, reproduces baseline).
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
ML=/workspace/gate0b_master.log; : > "$ML"
mlog(){ echo "[$(date -u +%H:%M:%S)] GATE0B $*" | tee -a "$ML"; }
rm -f /workspace/gate0b_MASTER_DONE
mlog "=== GATE0B START ==="
mlog "lock: $(gpu_lock.sh status 2>&1 | tr '\n' ' ')"

CFG=/workspace/scaled_suite/generated-user-config-0.yaml
SSHASH=$(grep -m1 data_dir "$CFG" | awk '{print $NF}' | sed 's#.*/scenesets/##' | tr -d '"')
SSDIR="$ALPA/data/nre-artifacts/scenesets/$SSHASH"
GLOB="$SSDIR/**/*.usdz"
mlog "sceneset=$SSHASH usdz=$(find "$SSDIR" -name '*.usdz' 2>/dev/null | wc -l)"

if ! ss -ltn | grep -q ':6011 '; then
  mlog "launch renderer"; setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/gate0b_renderer.log 2>&1 &
  for i in $(seq 1 150); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/gate0b_renderer.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && mlog "renderer up" || { mlog "renderer FAIL"; gpu_lock.sh release gate0b-gradient; echo FAIL_renderer > /workspace/gate0b_MASTER_DONE; exit 4; }

d=gate0b_grad
rm -rf /workspace/$d; mkdir -p /workspace/$d/controller
for f in controller-config.yaml driver-config.yaml eval-config.yaml generated-network-config.yaml \
         generated-user-config-0.yaml run_metadata.yaml trafficsim-config.yaml \
         wizard-config-loadable.yaml wizard-config.yaml; do cp /workspace/scaled_suite/$f /workspace/$d/$f; done
mlog "built /workspace/$d ($(grep -c scene_id: /workspace/$d/generated-user-config-0.yaml) scenes)"

mlog ">>> GRAD arm START (per-denoise gradient nudge; eta=0.5 iters=4; lam=5 mu=1)"
bash /workspace/gate0b_run.sh /workspace/gate0b_grad /workspace/gate0b_floor_GRAD.jsonl "$SSDIR"
mlog "grad: $(cat /workspace/gate0b_grad/DONE 2>/dev/null)"

for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null && mlog "cleanup killed :$p pid=$pid"
done
sleep 3
gpu_lock.sh release gate0b-gradient 2>&1 | tee -a "$ML"
mlog "GPU procs after release: [$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
echo "GATE0B_DONE grad=[$(cat /workspace/gate0b_grad/DONE 2>/dev/null)]" > /workspace/gate0b_MASTER_DONE
mlog "=== GATE0B DONE ==="
