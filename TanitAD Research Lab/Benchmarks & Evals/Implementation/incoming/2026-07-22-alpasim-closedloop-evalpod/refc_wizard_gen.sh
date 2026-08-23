#!/bin/bash
# Generate AlpaSim configs for the REF-C closed-loop run (external REF-C driver on :6789).
# Renders camera_front_wide_120fov at NATIVE f-theta 1080x1920 so ftheta_crop_resize applies directly.
# Reads HF token from stdin line 1 (scene resolution). Usage: bash refc_wizard_gen.sh [n_sim_steps] [log_dir]
IFS= read -r TOK
TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf
cd /workspace/alpa-invest/alpasim || exit 1
NSTEPS="${1:-50}"
LOGDIR="${2:-/workspace/refcrun}"
rm -rf "$LOGDIR"
timeout 400 uv run alpasim_wizard deploy=local topology=1gpu \
  driver=manual driver_source=external_static \
  wizard.run_method=NONE wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOGDIR" \
  scenes.scene_ids=[clipgt-01d503d4-449b-46fc-8d78-9085e70d3554] \
  runtime.simulation_config.n_sim_steps="$NSTEPS" \
  runtime.simulation_config.cameras.0.height=1080 \
  runtime.simulation_config.cameras.0.width=1920 2>&1 | tail -25
echo "WIZARD_EXIT=${PIPESTATUS[0]}"
grep -n -e height -e width -e logical_id "$LOGDIR/generated-user-config-0.yaml" | head
