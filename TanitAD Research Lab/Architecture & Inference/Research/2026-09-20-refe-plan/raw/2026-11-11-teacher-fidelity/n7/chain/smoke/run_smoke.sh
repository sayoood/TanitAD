#!/bin/bash
# the navtrain smoke of the TF-N7 build CLI (no navtest). Needs the run tree (N4_RUN), the navsim venv, F:/tanitad_scratch/rj5nt/repair_nt (read), and about 25 min CPU x 2 procs.
# 1. python mk_smoke_inputs7.py <S>/in         (20 real navtrain tokens: the 17 residual + 3 of the 73 certified at T0)
# 2. reach --real, derive in 2 shards (detached), merge, unch, then mutate_real_rows.py
set -u
S="${1:?smoke dir}"; R="${2:?run tree}"; PY="${PY:-C:/Users/Admin/navsim-crun/venv/Scripts/python.exe}"
export CUDA_VISIBLE_DEVICES=-1 N4_RUN="$R" PYTHONIOENCODING=utf-8
cd "$R/n7"
$PY -B n7_nt_build.py reach --real --out "$S/reach_real.json" > "$S/reach_real.log" 2>&1
for i in 0 1; do
  $PY -B n7_nt_build.py derive --route navtrain --n5a-plans "$S/in/n5a_plans.npz" --n2r-plans "$S/in/n2r_plans.npz" --rolls "$S/in/rolls.jsonl" --resid-tokens "$S/in/resid.json" \
      --out "$S/derive_clean_s${i}of2.jsonl" --shard $i/2 > "$S/derive_clean_s$i.log" 2>&1 &
done
wait
$PY -B n7_nt_build.py merge --n5a-plans "$S/in/n5a_plans.npz" --n5a-seam "$S/in/n5a_seam.npz" --n2r-plans "$S/in/n2r_plans.npz" --expect-residual-json "$S/in/resid.json" \
    --replay-expect-json "$S/in/replay_expect.json" --rows "$S/derive_clean_s*of2.jsonl" --harness-csv-n5a "$S/in/n5a_harness.csv" --reach-json "$S/reach_real.json" \
    --out-dir "$S/out_clean" --expect-residual 20 > "$S/merge_clean.log" 2>&1
$PY -B n7_nt_build.py unch --out-dir "$S/out_clean" --n5a-plans "$S/in/n5a_plans.npz" --n5a-seam "$S/in/n5a_seam.npz" --expect-residual 20 --out "$S/unch_clean.json" > "$S/unch_clean.log" 2>&1
