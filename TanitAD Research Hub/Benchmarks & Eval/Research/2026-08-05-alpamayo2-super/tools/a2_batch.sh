#!/bin/bash
export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
export MPLCONFIGDIR=/tmp/a2_mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ALPAMAYO2_SUPER_OUTPUT_DIR=/workspace/a2_batch_out
mkdir -p "$ALPAMAYO2_SUPER_OUTPUT_DIR"
# manifest from OUR 290 OOD-val clips at NVIDIA's own default t0 (5.1 s)
python3 - <<'PY'
import json
rows=[]
for l in open("/workspace/oodval_order.tsv"):
    for c in l.rstrip("\n").split("\t"):
        c=c.strip()
        if len(c)==36 and c.count("-")==4:
            rows.append({"clip_id":c,"t0_us":5100000,
                         "note":"TanitAD OOD-val (PhysicalAI official val split)"}); break
json.dump(rows, open("/workspace/tanitad_oodval_manifest.json","w"), indent=1)
print("manifest rows:", len(rows))
PY
cd /workspace/alpamayo2_repo
/workspace/a2venv/bin/python -u /workspace/a2_batch.py \
  --model-id /workspace/models/Alpamayo2-Super \
  --manifest /workspace/tanitad_oodval_manifest.json \
  --out /workspace/a2_batch_out/alpamayo_oodval.jsonl \
  --n 40
echo "A2_BATCH_RC=$?"
