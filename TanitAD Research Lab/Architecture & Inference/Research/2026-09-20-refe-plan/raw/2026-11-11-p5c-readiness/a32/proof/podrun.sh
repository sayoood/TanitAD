S=/workspace/data/p5c_ship; C=/workspace/data/p5c_label/control; W=/workspace/data/p5c_a32_check
for R in 0 all; do
mkdir -p $W/r$R
nice -n 19 /workspace/venv-teacher/bin/python $W/p5c_table_gates.py --label p5r --bank $C/bank --labels $C/out/labels_s*of15.jsonl \
  --control $S/dev_labels/labels_s0of2.jsonl $S/dev_labels/labels_s1of2.jsonl \
  --eval-control $S/p5r_eval/eval_r0.jsonl $S/p5r_eval/eval_r1.jsonl --eval-n 2400 --lambda-star $S/lambda_star.json --p5r-verdict $S/p5r_eval/verdict.json --tc20s-tol 1e-9 --tc20s-ranks $R \
  --tb1 $S/TB1_control_1018.json --expect-rows 2400 --out-dir $W/r$R > $W/r$R/gates_stdout.txt 2>&1
echo "rc[$R]=$?"
done
