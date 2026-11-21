#!/bin/bash
# Master Mind 2026-10-16: does PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True remove G-CKPT OOM (U/H1 rc 1 at 18:15Z: 37.72 GiB allocated + 5.79 GiB reserved-unallocated)?
# Scratch out-dir, scratch key, --only G-LIVE,G-CKPT: cannot mint a token the chain accepts.
D=/workspace/data; OUT=$D/refe_p6_probe/run_${1:-es}; rm -rf "$OUT"; mkdir -p "$OUT"
ARGS="--backbone vitl16 --targets $D/refe_navtrain/train_grow_cmd --images $D/navtrain_pixels --calib $D/refe_navtrain/train_grow/calib_table.json --metric-cache $D/refe_navtrain/metric_cache --scorer-sees-trunk --yaw-wrap off --nav-mode command --labeller online --label-workers 7 --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --tf32 --compile --workers 2 --log-every 1 --ckpt-every-min 20 --out $D/refe_runs/p6_paper_faithful --resume"
echo "ARGS sha $(printf "%s" "$ARGS" | sha256sum | cut -c1-16)" > "$OUT/probe.log"
export REFE_BACKBONE_ROOT=$D/backbones NUPLAN_DATA_ROOT=$D REFE_NUPLAN_DB_ROOT=$D/navtrain_dbs REFE_NAVTRAIN_YAML=/workspace/refe-plan-p6/splits/navtrain.yaml PYTHONUNBUFFERED=1
cd /workspace/refe-plan-p6/refe
[ "${1:-es}" = es ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-unset}" >> "$OUT/probe.log"
env -u HF_TOKEN OMP_NUM_THREADS=4 /workspace/venv-teacher/bin/python launch_gate.py run --profile refe_p6 --pkg /workspace/refe-plan-p6 --args "$ARGS" --out-dir "$OUT/lg" \
  --key-file "$OUT/probe.key" --only G-LIVE,G-CKPT --gate-bank "$OUT/gate_bank" --gate-bank-n 64 >> "$OUT/probe.log" 2>&1
echo "ZZPROBE_EXIT $?" >> "$OUT/probe.log"
