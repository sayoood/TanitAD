#!/bin/bash
# smoke_roll_cpu.sh -- toolchain smoke of the L1C roll on the PLATFORM kit, CPU only (CUDA_VISIBLE_DEVICES=-1; no GPU lock needed: no GPU is touched).
# 2 clips x 4 banked windows.  Output: F:/l1c/smoke2/.   Not part of the verdict; it proves the kit swap / remap / v9 md5 / windows / hook / bank path before the lock is won.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l1c
T=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=F:/l1c/smoke2
mkdir -p "$OUT" F:/l1c/tmp
export R8_LAUNCH_TREE=$T PYTHONPATH="$T/stack;$T/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1 TMP=F:/l1c/tmp TEMP=F:/l1c/tmp CUDA_VISIBLE_DEVICES=-1
unset REFCV6_KIT REFCV7_KIT R8_EVAL_KIT
cd "$W"
timeout 1500 "$PY" -X utf8 code/l1c_ztac_roll.py --mode eval --kit F:/l1c/kit_plat --remap code/remap_l1c_plat.json \
    --v9-md5 b13a3da5ac57c8abc0cbbf1954b07e3b --ztac-out "$OUT/ztac_smoke.npz" --require-seal -- \
    --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
    --dump-dir "$OUT/dump_s0" --windows-json F:/l1c/kit_plat/windows_smoke.json --device cpu --out-json "$OUT/roll.json" --os-only > "$OUT/roll.log" 2>&1
echo "SMOKE_RC=$?" >> "$OUT/roll.log"
