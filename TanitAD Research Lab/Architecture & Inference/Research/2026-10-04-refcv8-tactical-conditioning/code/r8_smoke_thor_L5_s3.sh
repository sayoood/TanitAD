#!/bin/bash
# refcv8 S3 RE-READ on the L5 tree (MM 2026-10-05): ONE short slot (<= 15 min) at FULL size -- 10 steps from
# refcv7-50,400, the gradient-share reading at step 10 (the L5 fp32 replay: bf16 AND TF32 off inside the replay),
# no in-run eval (S2 / X4 were read on the L4 G-SMOKE). Decides: smoke_summary.json S3 (trunk / bev025 / bev_pool
# lin_rel_err <= 1e-4). Same MemAvailable >= 40 GB guard INSIDE the lock. Everything else as r8_smoke_thor_L4.sh.
set -u
R=/home/nvidia/refcv8_run
T=${TREE:-$R/tree_L5}
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
OUT=$R/runs/refcv8-wpb-s3-L5
LG=$R/logs/s3_L5_train.log
MEM_MIN_GB=${MEM_MIN_GB:-40}
mkdir -p $R/logs $R/runs
export PYTHONPATH=$T/stack:$T/taniteval OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
grep -q "fp32_replay_tf32_off" $T/stack/tanitad/train/grad_share.py || { echo "[r8-s3-L5] tree lacks L5 -- stopping"; exit 1; }
ARGV=$($PY -c "
import json
a=json.load(open('$T/stack/ops/runs.d/refcv8-wpb-smoke.argv.json'))['argv']
def setf(a,f,v):
    if f in a:
        i=a.index(f); j=i+1
        while j<len(a) and not a[j].startswith('--'): j+=1
        a[i:j]=[f]+v
    else: a+= [f]+v
    return a
for f,v in (('--steps',['10']),('--log-every',['10']),('--grad-share-every',['10']),('--eval-every',['0']),
            ('--save-every',['10']),('--r8-alloc-emit-start',['10']),
            ('--join-defect-masks',['$T/stack/tanitad/configs/refcv8_join_label_defects.json']),
            ('--smoke-seed-ego-frames',[]),('--out',['$OUT'])):
    a=setf(a,f,v)
print(' '.join(a))")
[ -n "$ARGV" ] || { echo "[r8-s3-L5] could not build the argv -- stopping"; exit 1; }
echo "[r8-s3-L5] $(date -u +%H:%M:%S) queueing on the GPU lock (flock is not FIFO; waiting is expected)"
cd $T
export ARGV PY LG MEM_MIN_GB R
flock $LOCK bash -c 'm=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
  echo "[r8-s3-L5] $(date -u +%H:%M:%S) lock held; MemAvailable ${m} GB (min ${MEM_MIN_GB})"
  if [ -z "$m" ] || [ "$m" -lt "$MEM_MIN_GB" ]; then echo "MemAvailable ${m:-?} GB < ${MEM_MIN_GB}" > $R/logs/s3_L5_REFUSED_MEM; exit 97; fi
  exec timeout 900 $PY stack/scripts/refc_v3_train.py $ARGV >> $LG 2>&1 < /dev/null' 200>&-
rc=$?
echo "[r8-s3-L5] $(date -u +%H:%M:%S) trainer exit $rc (the artifact decides)"
$PY $R/code/r8_smoke_summary.py --run $OUT --init /home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt \
    > $R/logs/s3_L5_summary.log 2>&1 < /dev/null
$PY $R/code/p7_smoke_check.py --run $OUT --stack $T/stack > $R/logs/s3_L5_p7check.log 2>&1 < /dev/null
$PY - "$LG" "$OUT" > $R/logs/s3_L5_vis1.log 2>&1 < /dev/null <<'EOF'
import json, re, sys
from pathlib import Path
lg, out = Path(sys.argv[1]), Path(sys.argv[2])
m = [re.search(r"SMOKE: the first batch is (\d+) ego-stripped windows \(re-keyed (\d+)\)", l)
     for l in lg.read_text(errors="replace").splitlines()]
m = [x for x in m if x]
cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").is_file() else {}
dm = ((cfg.get("agent_join_stats") or {}).get("train") or {}).get("defect_masks") or {}
rec = {"seeded_windows": int(m[-1].group(1)) if m else None, "rekeyed_in_seeded_batch": int(m[-1].group(2)) if m else None,
       "n_ego_rows_removed_train": dm.get("n_ego_rows_removed"), "n_ego_frames_hit_train": dm.get("n_ego_frames_hit"),
       "reached_step_1": any(json.loads(l).get("step", 0) >= 1 for l in (out / "metrics.jsonl").read_text().splitlines()
                             if l.startswith("{")) if (out / "metrics.jsonl").is_file() else False}
rec["PASS"] = bool(rec["rekeyed_in_seeded_batch"] and rec["n_ego_rows_removed_train"] and rec["reached_step_1"])
(out / "vis1_rekey_check.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
EOF
echo "[r8-s3-L5] $(date -u +%H:%M:%S) summary $( [ -s $OUT/smoke_summary.json ] && echo WRITTEN || echo MISSING ) vis1 $(cat $R/logs/s3_L5_vis1.log | tail -1)"
rm -f $OUT/ckpt.pt
echo "[r8-s3-L5] $(date -u +%H:%M:%S) ckpt removed; free $(df -B1G /home | tail -1 | awk '{print $4}') GB"
