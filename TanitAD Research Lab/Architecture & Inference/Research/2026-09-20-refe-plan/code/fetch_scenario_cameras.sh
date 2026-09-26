#!/bin/bash
# Fetch CAM_F0 ONLY for the logs REFe's target bank actually references.
#
# The full mini front-camera set is 50.72 GB (MEASURED by indexing all 9 archives). The 8 Stage-0
# scenarios touch only SIX logs, so a targeted fetch is a small fraction of that and unblocks a
# REAL (non-synthetic) training rehearsal today instead of after the 193 GB database pull.
#
# Central directories are cached on disk, so re-surveying the archives is free after the first pass.
# Markers: SCEN_CAM_ZIP_DONE <n> / SCEN_CAM_FAIL <n> / SCENARIO_CAMERAS_DONE
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
VPY="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/driverl-venv/Scripts/python.exe"
B="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set"
OUT="D:/Projects/TanitAD/data/nuplan-camera/scenarios"
LOGS="2021.05.12.23.36.44_veh-35_01133_01535,2021.05.25.14.16.10_veh-35_01690_02183,2021.06.07.12.54.00_veh-35_01843_02314,2021.06.08.14.35.24_veh-26_02555_03004,2021.06.09.14.58.55_veh-35_01894_02311,2021.07.16.18.06.21_veh-38_04933_05307"
mkdir -p "$OUT"
echo "$(date +%T) targeting 6 logs across 9 archives"
for i in 0 1 2 3 4 5 6 7 8; do
  U="$B/nuplan-v1.1_mini_camera_${i}.zip"
  n=$("$VPY" "$PKG/code/fetch_front_camera.py" --list --only-logs "$LOGS" "$U" 2>/dev/null | grep -oE "CAM_F0 [0-9,]+ files" | tr -d ',' | awk '{print $2}')
  n=${n:-0}
  if [[ "$n" == 0 ]]; then echo "  camera_$i: none of our logs, skipping"; continue; fi
  echo "$(date +%T) === camera_$i holds $n of our frames -- fetching ==="
  if "$VPY" "$PKG/code/fetch_front_camera.py" --only-logs "$LOGS" "$U" "$OUT"; then
    echo "SCEN_CAM_ZIP_DONE $i"
  else
    echo "SCEN_CAM_FAIL $i"
  fi
done
echo "$(date +%T) SCENARIO_CAMERAS_DONE  total: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
