#!/usr/bin/env bash
# A3.3 read-only real-data proof ON THE POD against run 3 (/workspace/data/p5c_label_a32).  Writes ONLY under /workspace/data/p5c_a33_check.  No GPU, nice 19, one step per call.
#   bash podproof.sh fallback|control_new|control_old|main_new|main_old|tb0r_new|tb0r_old
set -u
R3=/workspace/data/p5c_label_a32
SHIP=/workspace/data/p5c_ship
CHK=/workspace/data/p5c_a33_check
BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7
PY=/workspace/venv-teacher/bin/python
NEWG="$CHK/code/p5c_table_gates.py"
NEWR="$CHK/code/p5c_rj5_path.py"
TOK=d06ff3cbd9025da7
RJSHA=926ffc6698300e477c7d38e78dd13c0277ffdfb4d59bb91c233f475cd3ff5f91
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$CHK/rekeyed"
case "$1" in
  fallback)
    nice -n 19 "$PY" "$NEWR" fallback --rekeyed "$R3/rekeyed/labels_rekeyed.jsonl" --out "$CHK/rekeyed/labels_rekeyed_fb.jsonl" --allow-row "$TOK:$RJSHA" ;;
  control_new)
    nice -n 19 "$PY" "$NEWG" --label p5r --bank "$R3/control/bank" --labels "$R3"/control/out/labels_s*of15.jsonl \
      --control "$SHIP/dev_labels/labels_s0of2.jsonl" "$SHIP/dev_labels/labels_s1of2.jsonl" --eval-control "$SHIP/p5r_eval/eval_r0.jsonl" "$SHIP/p5r_eval/eval_r1.jsonl" --eval-n 2400 \
      --lambda-star "$SHIP/lambda_star.json" --p5r-verdict "$SHIP/p5r_eval/verdict.json" --tc20s-tol 1e-9 --tb0-tol 1e-8 --tc20s-ranks all --tb1 "$SHIP/TB1_control_1018.json" \
      --expect-rows 2400 --out-dir "$CHK/gates_control_new" > "$CHK/gates_control_new.stdout" 2>&1; echo "rc=$?" ;;
  control_old)
    nice -n 19 "$PY" "$NEWG" --label p5r --bank "$R3/control/bank" --labels "$R3"/control/out/labels_s*of15.jsonl \
      --control "$SHIP/dev_labels/labels_s0of2.jsonl" "$SHIP/dev_labels/labels_s1of2.jsonl" --eval-control "$SHIP/p5r_eval/eval_r0.jsonl" "$SHIP/p5r_eval/eval_r1.jsonl" --eval-n 2400 \
      --lambda-star "$SHIP/lambda_star.json" --p5r-verdict "$SHIP/p5r_eval/verdict.json" --tc20s-tol 1e-9 --tc20s-ranks 0 --tb1 "$SHIP/TB1_control_1018.json" \
      --expect-rows 2400 --out-dir "$CHK/gates_control_old" > "$CHK/gates_control_old.stdout" 2>&1; echo "rc=$?" ;;
  main_new)
    nice -n 19 "$PY" "$NEWG" --label p5r --bank "$BANK" --labels "$CHK/rekeyed/labels_rekeyed_fb.jsonl" --repair0 "$R3/repair/repair_rank0_a3.json" --repair1 "$R3/repair/repair_rank1_a3.json" \
      --expect-rows 150603 --lambda-star "$SHIP/lambda_star.json" --tc20s-tol 1e-9 --tb5-fallback-allow "$TOK:$RJSHA" \
      --code-file "$R3/pkg/eval/p5_target_labels.py" "$SHIP/p5c_dual_labels.py" "$SHIP/p5r_labels.py" "$SHIP/p5r_run.py" "$NEWG" "$SHIP/traj_label_table.py" "$NEWR" \
      --out-dir "$CHK/gates_main_new" > "$CHK/gates_main_new.stdout" 2>&1; echo "rc=$?" ;;
  main_old)
    nice -n 19 "$PY" "$NEWG" --label p5r --bank "$BANK" --labels "$R3/rekeyed/labels_rekeyed.jsonl" --repair0 "$R3/repair/repair_rank0_a3.json" --repair1 "$R3/repair/repair_rank1_a3.json" \
      --expect-rows 150603 --lambda-star "$SHIP/lambda_star.json" --tc20s-tol 1e-9 \
      --code-file "$R3/pkg/eval/p5_target_labels.py" "$SHIP/p5c_dual_labels.py" "$SHIP/p5r_labels.py" "$SHIP/p5r_run.py" "$NEWG" "$SHIP/traj_label_table.py" "$NEWR" \
      --out-dir "$CHK/gates_main_old" > "$CHK/gates_main_old.stdout" 2>&1; echo "rc=$?" ;;
  tb0r_new)
    nice -n 19 "$PY" "$NEWR" tb0r --gates-module "$NEWG" --feed-bank "$R3/feed/bank" --pod-labels "$R3"/out/labels_s*of15.jsonl --dev-labels "$SHIP/rj5_labels_local.jsonl.gz" --expect-n 16133 \
      --tc20s-tol 1e-8 --out "$CHK/tb0r_new.json" ;;
  tb0r_old)
    nice -n 19 "$PY" "$NEWR" tb0r --gates-module "$NEWG" --feed-bank "$R3/feed/bank" --pod-labels "$R3"/out/labels_s*of15.jsonl --dev-labels "$SHIP/rj5_labels_local.jsonl.gz" --expect-n 16133 \
      --out "$CHK/tb0r_old.json" ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
