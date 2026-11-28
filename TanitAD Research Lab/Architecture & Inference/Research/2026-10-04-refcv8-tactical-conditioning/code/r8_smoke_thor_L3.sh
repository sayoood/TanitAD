#!/bin/bash
# refcv8 FULL-SIZE LAUNCH SMOKE, RE-RUN on the L3 tip (MM 2026-10-05). The L1 run died on the WP-C ego strip x VIS-1
# row key (RESULT_L3_LADDER_HARNESS.md sec. 2). Same job as r8_smoke_thor_L1.sh: the canonical smoke argv for 30 steps
# from refcv7-50,400, ONE job under the GPU lock (<= 40 min), --r8-alloc-emit-start 10. Plus ONE smoke-only token,
# --smoke-seed-ego-frames: the FIRST batch is built from listed ego-footprint windows, so the re-key path runs at full
# size. P(a listed window in 30 random steps) is only ~20 % (693 of 746,946 windows).
# The ARTIFACTS decide, never an exit code: smoke_summary.json, p7_smoke_check.json, and vis1_rekey_check.json
#   (n_rekeyed in the seeded batch > 0, read from the trainer's own SMOKE line; n_ego_rows_removed > 0 in config.json).
set -u
R=/home/nvidia/refcv8_run
T=${TREE:-$R/tree_L3}
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
OUT=$R/runs/refcv8-wpb-smoke-L3
LG=$R/logs/smoke_L3_train.log
mkdir -p $R/logs $R/runs
export PYTHONPATH=$T/stack:$T/taniteval OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
grep -q "def smoke_seed_batch" $T/stack/scripts/refc_v3_train.py || { echo "[r8-smoke-L3] tree lacks L3 -- stopping"; exit 1; }
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
for f,v in (('--steps',['30']),('--log-every',['10']),('--grad-share-every',['30']),('--eval-every',['30']),
            ('--eval-batches',['4']),('--save-every',['30']),('--r8-alloc-emit-start',['10']),
            ('--join-defect-masks',['$T/stack/tanitad/configs/refcv8_join_label_defects.json']),
            ('--smoke-seed-ego-frames',[]),('--out',['$OUT'])):
    a=setf(a,f,v)
print(' '.join(a))")
[ -n "$ARGV" ] || { echo "[r8-smoke-L3] could not build the argv -- stopping"; exit 1; }
echo "[r8-smoke-L3] $(date -u +%H:%M:%S) queueing on the GPU lock (flock is not FIFO; waiting is expected)"
cd $T
flock $LOCK timeout 2400 $PY stack/scripts/refc_v3_train.py $ARGV >> $LG 2>&1 < /dev/null 200>&-
rc=$?
echo "[r8-smoke-L3] $(date -u +%H:%M:%S) trainer exit $rc (the artifact decides)"
$PY $R/code/r8_smoke_summary.py --run $OUT --init /home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt \
    > $R/logs/smoke_L3_summary.log 2>&1 < /dev/null
$PY $R/code/p7_smoke_check.py --run $OUT --stack $T/stack > $R/logs/smoke_L3_p7check.log 2>&1 < /dev/null
$PY - "$LG" "$OUT" > $R/logs/smoke_L3_vis1.log 2>&1 < /dev/null <<'EOF'
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
echo "[r8-smoke-L3] $(date -u +%H:%M:%S) summary $( [ -s $OUT/smoke_summary.json ] && echo WRITTEN || echo MISSING ) vis1 $(cat $R/logs/smoke_L3_vis1.log | tail -1)"
rm -f $OUT/ckpt.pt
echo "[r8-smoke-L3] $(date -u +%H:%M:%S) ckpt removed; free $(df -B1G /home | tail -1 | awk '{print $4}') GB"
