#!/bin/bash
# wait for shards 0 and 1, then score shard 2's remaining rows in two processes (--only lists), then ALLDONE
O=F:/tanitad_scratch/tf_design/tf2w/out
until [ -f $O/tf2_s0.done ] && [ -f $O/tf2_s1.done ]; do sleep 20; done
PY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\tf_design\tf2w\refe"
$PY - <<'PYEOF'
import json, numpy as np
z = np.load("F:/tanitad_scratch/tf_design/tf2w/p5s/p5s_cubic.npz")
keys = sorted(zip(z["logs"].tolist(), z["tokens"].tolist()))[2::3]
done = {(json.loads(l)["log"], json.loads(l)["token"]) for l in open("F:/tanitad_scratch/tf_design/tf2w/out/tf2_s2.jsonl") if l.strip()}
rest = [k for k in keys if k not in done]
for j in range(2):
    open(f"F:/tanitad_scratch/tf_design/tf2w/out/s2_rest{j}.txt", "w").write(",".join(f"{a}:{b}" for a, b in rest[j::2]))
print("rest", len(rest))
PYEOF
for j in 0 1; do
  $PY E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf2/tf2_score.py --w F:/tanitad_scratch/tf_design/tf2w --e2score E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/teacher_review/e2e1 --out $O/tf2_s2r$j.jsonl --only "$(cat $O/s2_rest$j.txt)" > $O/tf2_s2r$j.log 2>&1 &
done
wait
echo ALLDONE > $O/ALLDONE
