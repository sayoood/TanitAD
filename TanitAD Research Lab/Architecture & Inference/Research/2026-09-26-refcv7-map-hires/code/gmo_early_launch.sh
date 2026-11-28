#!/usr/bin/env bash
# EARLY, NON-BINDING G-MAP-OVERFIT on Thor, started the moment the TRAIN weights exist.
# Wakes on the EVENT (the weights process exits), then validates the weights file, writes its
# provenance, checks the GPU once (refuses if another compute process holds it), and runs the
# candidate's harness through gmo_early_runner.py under nice 10. Kills nothing.
set -u
R=/home/nvidia/gmo_early_0327
CODE=$R/code
WPID=3521650
W=$R/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_out
PY=/home/nvidia/venvs/tanitad-train/bin/python
mkdir -p "$OUT"
echo "[launch] waiting for weights pid $WPID at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 480); do [ -d /proc/$WPID ] || break; sleep 15; done
if [ -d /proc/$WPID ]; then echo "[launch] REFUSED: weights still running after 2 h"; exit 3; fi
echo "[launch] weights process gone at $(date -u +%H:%M:%SZ)"
if [ ! -s "$W" ]; then echo "[launch] REFUSED: no weights file"; tail -5 $R/weights/compute.log; exit 3; fi
# provenance next to the weights: sha256 of the output, the script's git blob, the argv
SCRIPT=$CODE/stack/scripts/compute_map_class_weights.py
$PY - "$W" "$SCRIPT" > $R/weights/PROVENANCE.json <<'EOF'
import hashlib, json, subprocess, sys
w, s = sys.argv[1], sys.argv[2]
d = json.load(open(w, encoding="utf-8"))
blob = subprocess.run(["git", "hash-object", "--no-filters", s], capture_output=True,
                      text=True).stdout.strip()
chk = {"schema": d.get("schema"), "dry_run": d.get("dry_run"),
       "definition_id": d.get("definition_id"), "pre_registered": d.get("pre_registered"),
       "extent": d.get("extent"), "split": d.get("split"), "n_clips_used": d.get("n_clips_used"),
       "weights": d.get("weights"), "classes": d.get("classes")}
ok = (d.get("dry_run") is False and d.get("definition_id") == "sqrt_mf"
      and d.get("pre_registered") is True
      and d.get("extent") == {"x_max_m": 100.0, "y_half_m": 30.0})
print(json.dumps({
    "output": w, "output_sha256": hashlib.sha256(open(w, "rb").read()).hexdigest(),
    "script": s, "script_git_blob": blob,
    "argv": ["--v2-cache", "/home/nvidia/data/refcv6-b1-416x1024-train",
             "--gt-root", "/home/nvidia/data/sam3_gt_v3", "--split", "train",
             "--out", w],
    "argv_defaults_in_effect": {"x_max_m": 100.0, "y_half_m": 30.0, "definition": "sqrt_mf",
                                "clip_max": 25.0, "frame_stride": 1, "min_coverage": 0.9,
                                "weight_floor": 0.0},
    "code_tree": "b4a59b9 + NEW-2 candidate, shipped md5-verified (2,828 files)",
    "python": sys.executable, "checks": chk, "valid_for_launch_form": ok}, indent=1))
sys.exit(0 if ok else 4)
EOF
rc=$?
cat $R/weights/PROVENANCE.json | head -12
if [ $rc -ne 0 ]; then echo "[launch] REFUSED: weights file failed its checks (rc $rc)"; exit 4; fi
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]')
if [ "$apps" != "0" ]; then echo "[launch] REFUSED: $apps GPU compute process(es) present"; exit 5; fi
echo "[launch] G-MAP-OVERFIT start $(date -u +%H:%M:%SZ)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_early_runner.py "$CODE" "$OUT" "$W" \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,lane_w0,s8_detached
echo "[launch] G-MAP-OVERFIT exit $? at $(date -u +%H:%M:%SZ)"
