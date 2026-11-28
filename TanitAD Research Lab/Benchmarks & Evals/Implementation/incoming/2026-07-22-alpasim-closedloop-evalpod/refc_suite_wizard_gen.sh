#!/bin/bash
# Generate AlpaSim configs for a REF-C closed-loop SUITE (12 26.04-release scenes) at 480x854.
# External REF-C driver on :6789. Reads HF token from stdin (downloads the 12 scenes ~18GB).
IFS= read -r TOK
TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf
cd /workspace/alpa-invest/alpasim || exit 1
LOGDIR="${1:-/workspace/refcsuite}"
rm -rf "$LOGDIR"
SCENES="[clipgt-00040136-e651-4abd-991d-0655ccda9430,clipgt-000525f6-3999-4812-9924-8adff40ca514,clipgt-000548db-e266-49e5-a832-6674ab53a615,clipgt-00064c58-7047-4a53-8a36-b033baaaa5fb,clipgt-0009402a-a514-443b-9a4c-0e792f5ae581,clipgt-00097de1-5ded-4fba-a5ed-4b527678d1b0,clipgt-000a3a34-1031-4f90-9bc3-5b5c132fd1ed,clipgt-000a74ae-5c01-486e-ab6f-7f5160136357,clipgt-000e95f7-560d-4411-8069-b9f531ed3cd6,clipgt-000ff49d-aa30-46ee-af57-b4a0c1143f55,clipgt-0010ce77-d06e-43e6-bdaf-2cf8ab65cfe4,clipgt-001564ce-0019-4ec6-bb62-07ed2bd90f2e]"
timeout 1200 uv run alpasim_wizard deploy=local topology=1gpu \
  driver=manual driver_source=external_static \
  wizard.run_method=NONE wizard.debug_flags.use_localhost=True \
  wizard.log_dir="$LOGDIR" \
  scenes.scene_ids="$SCENES" \
  runtime.simulation_config.n_sim_steps=50 \
  runtime.simulation_config.cameras.0.height=480 \
  runtime.simulation_config.cameras.0.width=854 \
  eval.video.render_video=false 2>&1 | tail -20
echo "WIZARD_EXIT=${PIPESTATUS[0]}"
echo "=== sceneset ==="; grep -n data_dir "$LOGDIR/generated-user-config-0.yaml"
echo "=== n scenes in config ==="; grep -c "scene_id:" "$LOGDIR/generated-user-config-0.yaml"
