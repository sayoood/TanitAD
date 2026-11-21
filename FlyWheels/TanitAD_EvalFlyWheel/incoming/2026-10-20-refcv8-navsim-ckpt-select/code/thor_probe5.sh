#!/bin/bash
T=/home/nvidia/refcv8_run/launch_tree
echo "ZZ-TREE"; ls $T | head; cat $T/TREE_COMMIT.txt 2>/dev/null; ls $T/FlyWheels/TanitAD_EvalFlyWheel/incoming 2>/dev/null | grep -E "refcv6-standard|refcv7-standard|navsim" | head
for f in stack/tanitad/eval/refcv7_loader.py stack/tanitad/train/refcv8_train.py stack/scripts/refc_v3_train.py stack/tanitad/configs/refcv8_join_label_defects_train4572.json; do echo "$(sha256sum $T/$f 2>&1 | cut -c1-64) $f"; done
echo "ZZ-FILES"; for f in /home/nvidia/refcv8_v9labels/release/v9_labels_eval139.npz /home/nvidia/data/refcv8_pose_sync_sidecar.jsonl /home/nvidia/data/refcv7/map_hires_class_weights_train_100x30.json /home/nvidia/data/sam3_gt_v3 /home/nvidia/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz; do [ -e "$f" ] && echo "OK $f" || echo "MISSING $f"; done
echo "ZZ-PKG"; ~/venvs/tanitad-edge/bin/python -c "
import importlib
for m in ('numpy','pandas','pyarrow','scipy','PIL','cv2','triton','timm','torchvision','yaml','shapely'):
    try:
        x=importlib.import_module(m); print(m, getattr(x,'__version__','?'))
    except Exception as e: print(m, 'MISSING', type(e).__name__)
"
echo "ZZ-PKGTRAIN"; ~/venvs/tanitad-train/bin/python -c "
import importlib
for m in ('numpy','pandas','pyarrow','scipy','PIL','cv2','triton','timm','torchvision','yaml','shapely'):
    try:
        x=importlib.import_module(m); print(m, getattr(x,'__version__','?'))
    except Exception as e: print(m, 'MISSING', type(e).__name__)
"
echo ZZ-END
