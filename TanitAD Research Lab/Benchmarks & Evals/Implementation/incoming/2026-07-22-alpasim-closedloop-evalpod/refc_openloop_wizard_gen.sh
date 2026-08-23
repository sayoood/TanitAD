#!/bin/bash
# OPEN-LOOP diagnostic config: 4 scenes, FORCE-GT the whole rollout (ego follows GT),
# so REF-C predicts on in-distribution rendered frames along the GT path (no closed loop).
# 480x854 (matches the suite). Reads HF token from stdin (scenes already cached → no re-download).
IFS= read -r TOK
TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf
cd /workspace/alpa-invest/alpasim || exit 1
LOGDIR="${1:-/workspace/refcopenloop}"
rm -rf "$LOGDIR"
SCENES="[clipgt-00040136-e651-4abd-991d-0655ccda9430,clipgt-000525f6-3999-4812-9924-8adff40ca514,clipgt-000548db-e266-49e5-a832-6674ab53a615,clipgt-00064c58-7047-4a53-8a36-b033baaaa5fb]"
timeout 600 uv run alpasim_wizard deploy=local topology=1gpu \
  driver=manual driver_source=external_static \
  wizard.run_method=NONE wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOGDIR" \
  scenes.scene_ids="$SCENES" \
  runtime.simulation_config.n_sim_steps=80 \
  runtime.simulation_config.force_gt_duration_us=20000000 \
  runtime.simulation_config.skip_driver_during_force_gt=false \
  runtime.simulation_config.cameras.0.height=480 \
  runtime.simulation_config.cameras.0.width=854 \
  eval.video.render_video=false 2>&1 | tail -12
echo "WIZARD_EXIT=${PIPESTATUS[0]}"
grep -n -e force_gt -e n_sim_steps "$LOGDIR/generated-user-config-0.yaml"
