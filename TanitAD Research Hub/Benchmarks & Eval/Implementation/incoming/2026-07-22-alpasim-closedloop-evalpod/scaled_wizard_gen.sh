#!/bin/bash
# Wizard config + USDZ download for the balanced scenario suite (suite_clips.txt) at 480x854.
# Token via stdin line 1. Usage: <token> | bash scaled_wizard_gen.sh [logdir]
IFS= read -r TOK; TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf HF_HUB_DISABLE_XET=1
cd /workspace/alpa-invest/alpasim || exit 1
LOGDIR="${1:-/workspace/scaled_suite}"
rm -rf "$LOGDIR"
SCENES="[$(sed 's/^/clipgt-/' /workspace/suite_clips.txt | paste -sd,)]"
echo "N_SCENES=$(wc -l < /workspace/suite_clips.txt)"
timeout 2400 uv run alpasim_wizard deploy=local topology=1gpu \
  driver=manual driver_source=external_static \
  wizard.run_method=NONE wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOGDIR" \
  scenes.scene_ids="$SCENES" \
  runtime.simulation_config.n_sim_steps=50 \
  runtime.simulation_config.cameras.0.height=480 \
  runtime.simulation_config.cameras.0.width=854 \
  eval.video.render_video=false 2>&1 | tail -15
echo "WIZARD_EXIT=${PIPESTATUS[0]}"
echo "=== data_dir ==="; grep -n data_dir "$LOGDIR/generated-user-config-0.yaml" 2>/dev/null
echo "=== n scenes in config ==="; grep -c 'scene_id:' "$LOGDIR/generated-user-config-0.yaml" 2>/dev/null
echo "WIZARD_DONE" > "$LOGDIR/../scaled_wizard_DONE"
