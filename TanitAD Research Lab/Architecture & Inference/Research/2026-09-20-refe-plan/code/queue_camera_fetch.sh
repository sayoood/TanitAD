#!/bin/bash
# Fetch CAM_F0 (front camera only) for the nuPlan MINI sensor set -- QUEUED behind the DB pull.
#
# Why queued: the dev-box uplink is link-capped (~5 MB/s; MEASURED 2026-09-19 that parallel streams
# do NOT add throughput). Running this beside the 193 GB DB pull would halve the pull that DZ-11
# needs tonight. So it waits for PULL_ALL_DONE.
#
# Why CAM_F0 only (MEASURED on nuplan-v1.1_mini_camera_0.zip): the zip is 48.63 GB and bundles all 8
# cameras; CAM_F0 is 5.95 GB = 12.2%, in 7 contiguous byte-runs. REFe uses ONE front camera (PI), so
# whole-zip downloads would waste ~88%. mini camera set = 419.7 GB -> ~51 GB front-only.
#
# Markers: CAMLIST_DONE, CAM_ZIP_DONE <n>, CAM_ZIP_FAIL <n>, CAMERA_FETCH_DONE
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
VPY="${REFE_DRIVERL_PY:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
B="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set"
OUT="D:/Projects/TanitAD/data/nuplan-camera/mini"          # D: ONLY (PI)
L="D:/Projects/TanitAD/data/nuplan/pull_splits.log"
WAIT="${1:-wait}"

if [[ "$WAIT" == wait ]]; then
  echo "$(date +%T) waiting for PULL_ALL_DONE before using the uplink"
  while ! grep -q "PULL_ALL_DONE" "$L" 2>/dev/null; do
    if grep -qE "DOWNLOAD_FAIL" "$L" 2>/dev/null; then echo "$(date +%T) CAMERA_FETCH_ABORT db pull failed"; exit 1; fi
    sleep 300
  done
  echo "$(date +%T) DB pull finished -- starting the camera fetch"
fi
mkdir -p "$OUT"
# preflight: exact CAM_F0 totals from the remote indexes (no payload)
echo "$(date +%T) indexing 9 mini camera archives (central directories only)"
for i in 0 1 2 3 4 5 6 7 8; do
  "$VPY" "$PKG/code/fetch_front_camera.py" --list "$B/nuplan-v1.1_mini_camera_${i}.zip" \
    || echo "  index failed for $i"
done
echo "CAMLIST_DONE"
for i in 0 1 2 3 4 5 6 7 8; do
  echo "$(date +%T) === mini_camera_${i} ==="
  # --container: exFAT 1 MiB clusters inflate loose JPEGs 5x (5.08 GB -> 25 GB MEASURED).
  # One zip per log removes it; see raw/exfat_cluster_inflation.txt.
  if "$VPY" "$PKG/code/fetch_front_camera.py" --container "$B/nuplan-v1.1_mini_camera_${i}.zip" "$OUT"; then
    echo "CAM_ZIP_DONE $i"
  else
    echo "CAM_ZIP_FAIL $i"
  fi
  df_free=$(powershell.exe -NoProfile -Command "[math]::Round((Get-Volume -DriveLetter D).SizeRemaining/1GB,0)" 2>/dev/null | tr -d '\r')
  echo "  D: free ${df_free} GB"
done
echo "$(date +%T) CAMERA_FETCH_DONE  total: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
