#!/bin/bash
# Download the 12 suite scenes' reference mp4s + extract one mid-clip keyframe each (640px).
# Reads the HF token from stdin line 1 (never in argv). Usage: <token-source> | bash kf_download.sh [clips...]
IFS= read -r TOK; TOK="${TOK%$'\r'}"
export HF_TOKEN="$TOK" HF_HOME=/workspace/.hf HF_HUB_DISABLE_XET=1
cd /workspace || exit 1
HF=/workspace/alpa-invest/alpasim/.venv/bin/hf
DEFAULT="00040136-e651-4abd-991d-0655ccda9430 000525f6-3999-4812-9924-8adff40ca514 000548db-e266-49e5-a832-6674ab53a615 00064c58-7047-4a53-8a36-b033baaaa5fb 0009402a-a514-443b-9a4c-0e792f5ae581 00097de1-5ded-4fba-a5ed-4b527678d1b0 000a3a34-1031-4f90-9bc3-5b5c132fd1ed 000a74ae-5c01-486e-ab6f-7f5160136357 000e95f7-560d-4411-8069-b9f531ed3cd6 000ff49d-aa30-46ee-af57-b4a0c1143f55 0010ce77-d06e-43e6-bdaf-2cf8ab65cfe4 001564ce-0019-4ec6-bb62-07ed2bd90f2e"
CLIPS="${*:-$DEFAULT}"
mkdir -p /workspace/keyframes
for c in $CLIPS; do
  "$HF" download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec --repo-type dataset --revision 26.04 \
    --include "sample_set/26.04_release/$c/camera_front_wide_120fov.mp4" --local-dir /workspace/kf_dl >/dev/null 2>&1
  MP4="/workspace/kf_dl/sample_set/26.04_release/$c/camera_front_wide_120fov.mp4"
  if [ -f "$MP4" ]; then
    ffmpeg -y -ss 2 -i "$MP4" -frames:v 1 -vf scale=640:-1 "/workspace/keyframes/${c:0:8}.jpg" >/dev/null 2>&1
    echo "OK ${c:0:8} $([ -f /workspace/keyframes/${c:0:8}.jpg ] && echo jpg || echo NOFRAME)"
  else
    echo "DL_FAIL ${c:0:8}"
  fi
done
echo "=== keyframes ==="; ls -la /workspace/keyframes/
