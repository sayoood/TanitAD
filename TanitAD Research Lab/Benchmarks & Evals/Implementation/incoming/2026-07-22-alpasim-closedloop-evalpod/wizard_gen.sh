#!/bin/bash
# Generate AlpaSim configs (run_method=NONE) for a bare localhost closed-loop run.
# Reads HF token from stdin line 1 (for the wizard's canonical scene download, ~20s).
# Usage: bash wizard_gen.sh [n_sim_steps] [log_dir]
IFS= read -r TOK
TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf
cd /workspace/alpa-invest/alpasim || exit 1
NSTEPS="${1:-50}"
LOGDIR="${2:-/workspace/m2run}"
rm -rf "$LOGDIR"
timeout 400 uv run alpasim_wizard deploy=local topology=1gpu \
  driver=manual driver_source=external_static \
  wizard.run_method=NONE wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOGDIR" \
  scenes.scene_ids=[clipgt-01d503d4-449b-46fc-8d78-9085e70d3554] \
  runtime.simulation_config.n_sim_steps="$NSTEPS" 2>&1 | tail -40
echo "WIZARD_EXIT=${PIPESTATUS[0]}"
echo "=== generated files ==="
ls -la "$LOGDIR" 2>/dev/null
