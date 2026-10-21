#!/bin/bash
# S3: fix the sample (registered rule + AMENDMENT 1). Marker from the tool: ZZSAMPLE status=... ; wrapper end marker QQS3END
set -u
N=/home/nvidia/r8ns/navtrain; X=/home/nvidia/r8ns/code/navtrain; DL=/home/nvidia/r8ns/navtrain_dl
YAML=/home/nvidia/navsim-thor/src/navsim-v11/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
t0=$(date +%s)
cd $X && python3 $X/fix_navtrain_sample.py --filter-yaml $YAML --metadata-dir $N/navsim_logs/trainval \
    --shard $DL/navtrain_current_1.tgz --shard $DL/navtrain_history_1.tgz \
    --verify-sha256-from $X/raw_hf_tree_navsim.json --out $N/NAVTRAIN_SAMPLE_TOKENS.json > $N/fix_sample.out 2>&1
echo "QQS3END rc=$? secs=$(( $(date +%s)-t0 ))" >> $N/fix_sample.out
