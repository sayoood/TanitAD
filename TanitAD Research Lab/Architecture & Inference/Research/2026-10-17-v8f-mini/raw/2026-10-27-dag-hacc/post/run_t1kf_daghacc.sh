#!/bin/bash
# the t1kf_read_a1 ARM VARIANT for DAG-HACC (TACTICAL TAC-LAT / TAC-LON and DISTANCE KEEPING DK-*), exactly as dagreread/code/run_t1kf_arm.sh did for DAG:
# mk_t1kf_arm.py's substitutions only (arm label; C5's banked targets from the arm's own run JSON). Decisive / logged-limit / strict dumps of the arm (DECLARATIONS D6).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PK=E:/Projects/TanitAD-artifacts/daghacc
RS=E:/Projects/TanitAD-artifacts/dagreread/reader_src
TREE=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
G=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/gitdumps
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ARM=DAGHACC
OUT=${1:-$PK/raw/t1kf_$ARM}
mkdir -p $OUT
cd $PK
$PY -I E:/Projects/TanitAD-artifacts/dagreread/code/mk_t1kf_arm.py $RS/t1kf_read_a1.py $OUT/t1kf_read_a1_arm.py $ARM > $OUT/mk.log 2>&1
for s in 0 1; do
  export T1KF_RUNJSON_DECISIVE_S$s=$PK/t1/t1v8f_${ARM}_s${s}_m10_causal_heads.json
  export T1KF_RUNJSON_LOGGED_S$s=$PK/t1/t1v8f_${ARM}_s${s}_m10_logged_heads.json
  export T1KF_RUNJSON_STRICT_S$s=$PK/t1/t1v8f_${ARM}STRICT_s${s}_m10_frozen_causal_heads.json
done
PKD=$PK/packed
$PY -I $OUT/t1kf_read_a1_arm.py --tree $TREE --t1k-read $RS/t1k_read.py --lead-block $G/b1_eval_lead_block_k30.npz \
  --dump-decisive-s0 $PKD/dumps_${ARM}_s0_m10_causal.npz --dump-decisive-s1 $PKD/dumps_${ARM}_s1_m10_causal.npz \
  --dump-logged-s0 $PKD/dumps_${ARM}_s0_m10_logged.npz --dump-logged-s1 $PKD/dumps_${ARM}_s1_m10_logged.npz \
  --dump-strict-s0 $PKD/dumps_${ARM}STRICT_s0_m10_frozen_causal.npz --dump-strict-s1 $PKD/dumps_${ARM}STRICT_s1_m10_frozen_causal.npz \
  --out $OUT/T1KF_READ_A1_$ARM.json > $OUT/run.log 2>&1
echo "ZZDONE rc=$?" >> $OUT/run.log
