#!/bin/bash
# C1 for the t1kf reader: the UNMODIFIED t1kf_read_a1.py (bdf8d561 bytes) on HACC's banked dumps; controls_daghacc.py compares its rows to the banked T1KF_READ_A1.json (max abs diff).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PK=E:/Projects/TanitAD-artifacts/daghacc
RS=E:/Projects/TanitAD-artifacts/dagreread/reader_src
TREE=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
G=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/gitdumps
mkdir -p $PK/raw/c1_t1kf
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I $RS/t1kf_read_a1.py --tree $TREE --t1k-read $RS/t1k_read.py --lead-block $G/b1_eval_lead_block_k30.npz \
  --dump-decisive-s0 $G/dumps_HACC_s0_m10_causal.npz --dump-decisive-s1 $G/dumps_HACC_s1_m10_causal.npz \
  --dump-logged-s0 $G/dumps_HACC_s0_m10.npz --dump-logged-s1 $G/dumps_HACC_s1_m10.npz \
  --dump-strict-s0 $G/dumps_HACC_s0_m10_strict.npz --dump-strict-s1 $G/dumps_HACC_s1_m10_strict.npz \
  --out $PK/raw/c1_t1kf/T1KF_READ_A1_HACC_reproduction.json > $PK/raw/c1_t1kf/run.log 2>&1
echo "ZZDONE rc=$?" >> $PK/raw/c1_t1kf/run.log
