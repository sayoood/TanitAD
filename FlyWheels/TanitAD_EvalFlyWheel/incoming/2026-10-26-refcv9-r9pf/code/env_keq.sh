# source me: the K-EQ roll environment (dev-box GPU; the R9 LAUNCH TREE on PYTHONPATH, NOT tree_fb47a58)
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
LT=F:/refcv9_r9pf_scratch/launch_tree
export R8_LAUNCH_TREE=$LT
export PYTHONPATH="$LT/stack;$LT/taniteval;C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
