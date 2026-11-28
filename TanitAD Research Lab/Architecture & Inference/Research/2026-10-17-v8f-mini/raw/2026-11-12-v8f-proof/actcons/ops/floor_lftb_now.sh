#!/usr/bin/env bash
# floor_lftb_now.sh : dev box. Card PREREG_V8F_PROOF_ADD37, control a5 (floor): the imagination dump of the rerun control LFTb (chain leadcond_a, ended 16:02Z)
# on the pod CPU with the same script the waiter uses (idempotent: the stage-2 waiter finds the records and does not run them again), pulled with sha256,
# read with the tool of card ADD36 and paired with LFT. Starts no GPU job and no training.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
H=/f/TanitAD-scratch/v8proof
D=$H/imaghead/dump; R=$H/actcons/read
ts() { date -u +%H:%M:%SZ; }
out=$(timeout 900 bash "$P" "bash -s -- leadcond_a tree_leadsup LFTb_s0 LFTb_s1" < $H/imaghead/ops/pod_imag_dump_arm.sh 2>/dev/null | sed '/hf_/d' | tr -d '\r')
echo "$out" | cut -c1-330 >> "$R/pod_dump_floor.out"
echo "$out" | grep -q "DUMP-NOT-RUN" && { echo "ZZ FLOOR NOT RUN: pod memory low"; exit 1; }
echo "$out" | grep "c2b\|^   c1" | cut -c1-300
echo "$out" | grep "^   SHA " | while read -r _ a s1 s2; do
  for pr in "DUMP_$a.npz:$s1" "DUMP_$a.npz.json:$s2"; do f=${pr%%:*}; s=${pr##*:}
    [ -f "$D/$f" ] && [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && continue
    timeout 300 bash $C /workspace/v8proof/runs/imag_z/dump/$f "$D/$f" < /dev/null; [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
  done
done
[ -f "$D/DUMP_LFTb_s0.npz" ] && [ -f "$D/DUMP_LFTb_s1.npz" ] || { echo "ZZ FLOOR NOT READ: dumps absent"; exit 1; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
RW=F:/TanitAD-scratch/v8proof/actcons/read; DW=F:/TanitAD-scratch/v8proof/imaghead/dump
( cd $H/imaghead && "$PY" imag_read.py read --dump "$DW/DUMP_LFTb_s0.npz" "$DW/DUMP_LFTb_s1.npz" --arm LFTb --seeds 0,1 --out-dir "$RW/LFTb" > "$RW/LFTb.log" 2>&1 ); echo "[$(ts)] read LFTb rc $? : $(tail -n 1 "$R/LFTb.log" | cut -c1-260)"
( cd $H/imaghead && "$PY" imag_read.py pair --a-dir "$RW/LFTb" --b-dir "$RW/LFT" --a-arm LFTb --b-arm LFT --out-dir "$RW/PAIR_LFTb_LFT" > "$RW/PAIR_LFTb_LFT.log" 2>&1 ); echo "[$(ts)] pair LFTb minus LFT rc $? : $(tail -n 1 "$R/PAIR_LFTb_LFT.log" | cut -c1-300)"
echo "ZZ FLOOR LFTB DONE $(ts)"
