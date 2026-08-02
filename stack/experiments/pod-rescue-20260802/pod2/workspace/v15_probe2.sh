#!/usr/bin/env bash
echo "=== locate anchors256.pt / probes8.pt ==="
find /workspace -maxdepth 3 \( -name 'anchors256.pt' -o -name 'probes8.pt' \) 2>/dev/null | head
ls -la /workspace/v15/ 2>/dev/null | head -20
echo "=== v15-abc metrics.json ==="
cat /workspace/experiments/flagship-v15-abc/metrics.json 2>/dev/null; echo
echo "=== v16-ab-ft metrics.json ==="
cat /workspace/experiments/flagship-v16-ab-ft/metrics.json 2>/dev/null; echo
echo "=== v16 eval_v16.json (first 700 chars) ==="
head -c 700 /workspace/experiments/flagship-v16-ab-ft/eval_v16.json 2>/dev/null; echo
echo "=== ckpt key structure ==="
/usr/bin/python3 -c "import torch
for p in ['/workspace/experiments/flagship-v15-abc/ckpt_best.pt','/workspace/experiments/flagship-v15-abc/ckpt.pt','/workspace/experiments/flagship-v16-ab-ft/ckpt_best.pt']:
    try:
        d=torch.load(p,map_location='cpu',weights_only=False)
        print(p,'->', (list(d.keys()) if isinstance(d,dict) else type(d)))
    except Exception as e:
        print(p,'ERR',repr(e))"
echo "PROBE2_DONE"
