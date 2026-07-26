#!/bin/bash
# Generate AlpaSim configs for a FULL-RUNTIME closed-loop rollout with trafficsim=catk ENABLED.
#
# This is the experiment GATE_RESULTS.md 2.4 names as the decisive follow-up to gate 2:
# gate 2 drove the trafficsim gRPC service DIRECTLY with a hand-built session; this drives it
# through `alpasim_runtime.simulate`, i.e. the real integration, so the runtime itself builds the
# TrafficSessionRequest (logged_object_trajectories + handover_time_us) and issues the per-step
# TrafficRequest carrying the ego's ACTUAL pose (services/traffic_service.py:simulate_traffic).
#
# CONFIGURATION ONLY -- no AlpaSim source is modified. `trafficsim=catk` is a stock wizard config
# group (src/wizard/configs/trafficsim/catk.yaml). The NuRec renderer is RUN, never edited
# (NGC-DL-CONTAINER-LICENSE forbids derivatives).
#
# Usage: bash ts_wizard_gen.sh <scene_uuid> <n_sim_steps> <log_dir>   (HF token on stdin line 1)
set -uo pipefail
IFS= read -r TOK
TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf
cd /workspace/alpa-invest/alpasim || exit 1

SCENE="${1:?scene uuid}"
NSTEPS="${2:-150}"
LOGDIR="${3:?log dir}"
rm -rf "$LOGDIR"

# trafficsim=catk           -> runtime.endpoints.trafficsim.skip=false  (the ONLY change vs the
#                              program's previous closed-loop runs, which all used the DEFAULT
#                              trafficsim=disabled -> skip=true -> literal REPLAY)
# trafficsim.catk.device=cpu-> the pod's torch_cluster/torch_scatter are a CPU-only source build
#                              (no nvcc on tanitad-eval), so CATK must run on CPU. Config, not code.
timeout 600 uv run alpasim_wizard deploy=local topology=1gpu \
  driver=manual driver_source=external_static \
  trafficsim=catk \
  trafficsim.catk.device=cpu \
  wizard.run_method=NONE wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOGDIR" \
  scenes.scene_ids=["$SCENE"] \
  runtime.simulation_config.n_sim_steps="$NSTEPS" 2>&1 | tail -25
echo "WIZARD_EXIT=${PIPESTATUS[0]}"
echo "=== generated files ==="
ls -la "$LOGDIR" 2>/dev/null
echo "=== trafficsim skip flag (must be false) ==="
grep -rn -A3 "trafficsim" "$LOGDIR/generated-user-config-0.yaml" 2>/dev/null | head -20
echo "=== trafficsim endpoint ==="
grep -n -A2 "trafficsim" "$LOGDIR/generated-network-config.yaml" 2>/dev/null | head
