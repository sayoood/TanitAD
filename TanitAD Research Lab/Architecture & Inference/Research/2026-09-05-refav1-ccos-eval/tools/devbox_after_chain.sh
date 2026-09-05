#!/bin/bash
# Queue behind the dev-box chain: when chain.DONE appears, run the gate-2b functional ablation
# (cos = deliberate regression, must read ~0 scene degradation; ccos naive) on every 12th window
# of the 282 grid (24 windows / 24 clusters, the sweep's own stratification), then the same for the
# compensated ccos weights.
B=/c/Users/Admin/ccos_eval/devbox
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
for i in $(seq 1 720); do [ -f "$B/chain.DONE" ] && break; sleep 60; done
[ -f "$B/chain.DONE" ] || { stamp "chain never finished"; exit 3; }
stamp "ablation start (shipped weights, cos + ccos)"
"$PY" /c/Users/Admin/ccos_eval/tools/refav1_source_ablation.py --ckpt "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt" --config "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json" \
  --cache "C:/Users/Admin/refav1_eval_full/fp8" --episodes "C:/Users/Admin/refav1_eval_full/eps" \
  --labels "C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz" --nav "C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz" \
  --device cuda --window-stride 40 --every 12 --metrics cos,ccos --out "$B/ablation_shipped.json" > "$B/ablation_shipped.log" 2>&1
stamp "ablation shipped exit=$?"
W=$(tr -d ' \r\n' < /c/Users/Admin/ccos_eval/arm2_weights.txt)
"$PY" /c/Users/Admin/ccos_eval/tools/refav1_source_ablation.py --ckpt "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt" --config "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json" \
  --cache "C:/Users/Admin/refav1_eval_full/fp8" --episodes "C:/Users/Admin/refav1_eval_full/eps" \
  --labels "C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz" --nav "C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz" \
  --device cuda --window-stride 40 --every 12 --metrics ccos --weights "$W" --out "$B/ablation_ccos_comp.json" > "$B/ablation_ccos_comp.log" 2>&1
stamp "ablation compensated exit=$?"
touch "$B/ablation.DONE"
