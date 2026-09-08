#!/bin/bash
# ⛔ CORRECTION, MEASURED: `refcv3_arm.py` DOES rebuild the BEV aux head. My earlier
# "it has no bev_aux handling" came from ONE probe (`grep bev_aux refcv3_arm.py` ->
# nothing) and is FALSE: the handling arrives INDIRECTLY, because `rebuild_config`
# reconstructs the model through `refc_v3_train.build_parser` + `_pin_trainer_cfg`
# from D1's OWN argv, which carries `--bev-aux col`. The rebuilt model therefore HAS
# the head, and the STRIPPED checkpoint is refused for 6 MISSING keys.
# ⇒ D1 is dumped from its RAW checkpoint. That is still a clean comparison: the
# prereg PROVES the head is called LAST, consumes no RNG and leaves the planner's
# output bit-identical, so its presence cannot move a trajectory.
# (Absence found at ONE location is not absence -- CLAUDE.md.)
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TOOL=/c/Users/Admin/wpd-run/taniteval/tools/refcv3_arm.py
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
echo "[5b] === DUMP wpdD1 (RAW ckpt) $(date -u +%FT%TZ) ==="
"$PY" -u "$TOOL" --ckpt "C:\Users\Admin\wpd-probe\ckpt\ckpt_D1.pt" \
  --config "C:\Users\Admin\wpd-probe\ckpt\config_D1.json" \
  --episodes "C:\Users\Admin\tanitad-data\refav1-eval141\eps" \
  --labels "C:\Users\Admin\wpd-probe\labels\s2_labels_v7.2_eval.jsonl.gz" \
  --lead-block "C:\Users\Admin\wpd-probe\leadblk\b1_eval_lead_block.npz" \
  --arm wpdD1 --grid 2s --device cuda --episodes-n 40 --window-stride 2 \
  --dump-dir "C:\Users\Admin\wpd-probe\dumps\wpdD1" --dump-only \
  --out "C:\Users\Admin\wpd-probe\raw\wpdD1.json"
echo "[5b] dump wpdD1 rc=$? files=$(ls /c/Users/Admin/wpd-probe/dumps/wpdD1/ep*.npz 2>/dev/null | wc -l)"
echo "[5b] === ANALYZE wpdD1 $(date -u +%FT%TZ) ==="
"$PY" -u "$TOOL" --analyze-only "C:\Users\Admin\wpd-probe\dumps\wpdD1" --arm wpdD1 \
  --lead-block "C:\Users\Admin\wpd-probe\leadblk\b1_eval_lead_block.npz" \
  --n-boot 2000 --out "C:\Users\Admin\wpd-probe\raw\wpdD1.json"
echo "[5b] analyze wpdD1 rc=$? json=$(ls /c/Users/Admin/wpd-probe/raw/wpdD1.json 2>/dev/null | wc -l)"
echo "[5b] D1 CHAIN COMPLETE $(date -u +%FT%TZ)"
