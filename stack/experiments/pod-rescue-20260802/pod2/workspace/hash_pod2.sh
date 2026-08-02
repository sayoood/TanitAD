#!/bin/bash
cd /workspace/rigfix/stack_head
sha256sum tanitad/data/parity.py tanitad/data/parity_manifest.json \
          tanitad/data/v2_dataset.py tanitad/data/calib.py tanitad/geometry.py \
          scripts/register_v2_sibling.py scripts/eval_flagship_v4.py \
          scripts/train_flagship_v4.py 2>&1
echo "--- manifest corpora ---"
python3 -c "import json;d=json.load(open('tanitad/data/parity_manifest.json'));print(list(d['corpora'].keys()));print('schema',d.get('schema'))"
